"""Training script for CPN with FP16 mixed precision."""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
from tqdm import tqdm

from model.cpn import CPN
from model.losses import CPNLoss
from utils.dataset import CPNDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Train CPN for super small object detection')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='/home/quanghien/aivn',
                        help='Root directory containing train/val folders')
    parser.add_argument('--template_size', type=int, default=384,
                        help='Template resize resolution')
    parser.add_argument('--search_width', type=int, default=1024,
                        help='Search image width')
    parser.add_argument('--search_height', type=int, default=576,
                        help='Search image height')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--max_lr', type=float, default=1e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--augment_prob', type=float, default=0.3,
                        help='Augmentation probability for search images')
    
    # Model
    parser.add_argument('--pretrained_backbone', type=str, default='imagenet',
                        choices=['imagenet', 'none'],
                        help='Pretrained backbone weights')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (only model weights)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    return parser.parse_args()


def build_optimizer(model, max_lr, weight_decay):
    """Build optimizer with different lr for backbone and detection."""
    # Backbone: 0.1x max_lr
    # Detection head: 1.0x max_lr
    backbone_params = []
    detection_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            detection_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': max_lr * 0.1},
        {'params': detection_params, 'lr': max_lr}
    ], weight_decay=weight_decay)
    
    return optimizer


def build_scheduler(optimizer, warmup_epochs, total_epochs, max_lr, min_lr):
    """Build cosine annealing scheduler with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing to min_lr
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return (min_lr / max_lr) + (1 - min_lr / max_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_focal = 0
    total_l1 = 0
    total_giou = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
    
    for templates, searches, targets in pbar:
        # Move to device
        templates = templates.to(device)
        searches = searches.to(device)
        target_bboxes = [t['bbox'].to(device) for t in targets]
        
        # Forward with autocast
        optimizer.zero_grad()
        
        with autocast('cuda'):
            pred_objectness, pred_bbox = model(templates, searches)
            losses = criterion(pred_objectness, pred_bbox, target_bboxes)
        
        # Backward with scaler
        scaler.scale(losses['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += losses['loss'].item()
        total_focal += losses['loss_focal'].item()
        total_l1 += losses['loss_l1'].item()
        total_giou += losses['loss_giou'].item()
    
    # Return average losses
    return {
        'loss': total_loss / len(dataloader),
        'focal': total_focal / len(dataloader),
        'l1': total_l1 / len(dataloader),
        'giou': total_giou / len(dataloader)
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    total_focal = 0
    total_l1 = 0
    total_giou = 0
    
    pbar = tqdm(dataloader, desc='Validating', leave=False)
    
    for templates, searches, targets in pbar:
        # Move to device
        templates = templates.to(device)
        searches = searches.to(device)
        target_bboxes = [t['bbox'].to(device) for t in targets]
        
        # Forward
        with autocast('cuda'):
            pred_objectness, pred_bbox = model(templates, searches)
            losses = criterion(pred_objectness, pred_bbox, target_bboxes)
        
        # Accumulate losses
        total_loss += losses['loss'].item()
        total_focal += losses['loss_focal'].item()
        total_l1 += losses['loss_l1'].item()
        total_giou += losses['loss_giou'].item()
    
    # Return average losses
    return {
        'loss': total_loss / len(dataloader),
        'focal': total_focal / len(dataloader),
        'l1': total_l1 / len(dataloader),
        'giou': total_giou / len(dataloader)
    }


def main():
    args = parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build datasets
    print('\n=== Building datasets ===')
    train_dataset = CPNDataset(
        data_dir=args.data_dir,
        split='train',
        template_size=args.template_size,
        search_size=(args.search_width, args.search_height),
        augment_prob=args.augment_prob
    )
    
    val_dataset = CPNDataset(
        data_dir=args.data_dir,
        split='val',
        template_size=args.template_size,
        search_size=(args.search_width, args.search_height),
        augment_prob=0.0
    )
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f'Train: {len(train_dataset)} images, {len(train_loader)} batches')
    print(f'Val: {len(val_dataset)} images, {len(val_loader)} batches')
    
    # Build model
    print('\n=== Building model ===')
    use_pretrained = (args.pretrained_backbone == 'imagenet')
    model = CPN(pretrained=use_pretrained)
    
    # Load checkpoint if provided (only model weights, always start from epoch 0)
    start_epoch = 0
    if args.checkpoint is not None:
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Check if checkpoint is FP32 or FP16
        first_param = next(iter(checkpoint.values()))
        ckpt_dtype = first_param.dtype
        print(f'Checkpoint dtype: {ckpt_dtype}')
        
        # Convert to FP32 if needed (mixed precision training requires FP32 model)
        if ckpt_dtype == torch.float16:
            print('Converting FP16 checkpoint to FP32 for mixed precision training...')
            checkpoint = {k: v.float() if v.dtype == torch.float16 else v 
                         for k, v in checkpoint.items()}
        
        model.load_state_dict(checkpoint)
        print('Checkpoint loaded (starting from epoch 0 as requested)')
    
    # Move model to device (keep FP32 for mixed precision training)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params / 1e6:.2f}M')
    print(f'Trainable params: {trainable_params / 1e6:.2f}M')
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, args.max_lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs, args.max_lr, args.min_lr)
    
    # Build loss
    criterion = CPNLoss().to(device)
    
    # GradScaler for mixed precision (autocast will handle FP16 automatically)
    scaler = GradScaler('cuda')
    
    # Training loop
    print('\n=== Training ===')
    print(f'Max LR: {args.max_lr:.2e}, Min LR: {args.min_lr:.2e}\n')
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | "
              f"Train Loss: {train_losses['loss']:.4f} (F:{train_losses['focal']:.3f}, "
              f"L1:{train_losses['l1']:.3f}, G:{train_losses['giou']:.3f}) | "
              f"Val Loss: {val_losses['loss']:.4f} (F:{val_losses['focal']:.3f}, "
              f"L1:{val_losses['l1']:.3f}, G:{val_losses['giou']:.3f})")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  → Saved: {checkpoint_path.name}')
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            best_path = save_dir / 'best_checkpoint.pth'
            torch.save(model.state_dict(), best_path)
            print(f'  → Best: {best_path.name} (val_loss={val_losses["loss"]:.4f})')
    
    # Save last checkpoint
    last_path = save_dir / 'last_checkpoint.pth'
    torch.save(model.state_dict(), last_path)
    print(f'\n=== Training completed ===')
    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Saved last checkpoint: {last_path}')


if __name__ == '__main__':
    main()
