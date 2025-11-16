"""Loss functions for CPN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred_logits, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for objectness classification.
    
    Args:
        pred_logits: (B, H, W) predicted logits (before sigmoid)
        target: (B, H, W) binary labels (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
    
    Returns:
        Scalar loss
    """
    # BCE loss with logits (safe for autocast)
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    
    # Get probabilities for focal weighting
    pred_prob = torch.sigmoid(pred_logits)
    pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weighting
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    
    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()


def l1_loss(pred, target, mask):
    """
    L1 loss for bbox regression (only at positive locations).
    
    Args:
        pred: (B, 4, H, W) predicted bbox offsets
        target: (B, 4) target bbox in YOLO format (cx, cy, w, h)
        mask: (B, H, W) binary mask indicating positive locations
    
    Returns:
        Scalar loss
    """
    B, _, H, W = pred.shape
    
    # Expand target to spatial dimensions
    target_expanded = target.view(B, 4, 1, 1).expand(B, 4, H, W)
    
    # Compute L1 loss
    diff = torch.abs(pred - target_expanded)
    
    # Apply mask (only compute loss at positive locations)
    mask_expanded = mask.unsqueeze(1).expand_as(diff)
    masked_diff = diff * mask_expanded
    
    # Average over positive locations
    num_pos = mask.sum() + 1e-6
    loss = masked_diff.sum() / num_pos
    
    return loss


def giou_loss(pred, target, mask):
    """
    GIoU loss for bbox regression.
    
    Args:
        pred: (B, 4, H, W) predicted bbox offsets in YOLO format (cx, cy, w, h)
        target: (B, 4) target bbox in YOLO format (cx, cy, w, h)
        mask: (B, H, W) binary mask indicating positive locations
    
    Returns:
        Scalar loss
    """
    B, _, H, W = pred.shape
    
    # Find positive locations
    pos_indices = torch.nonzero(mask, as_tuple=False)  # (N, 3) where N = num_pos
    
    if len(pos_indices) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Extract predicted bboxes at positive locations
    pred_bboxes = []
    target_bboxes = []
    
    for b, y, x in pos_indices:
        pred_bbox = pred[b, :, y, x]  # (4,)
        target_bbox = target[b]       # (4,)
        pred_bboxes.append(pred_bbox)
        target_bboxes.append(target_bbox)
    
    pred_bboxes = torch.stack(pred_bboxes)    # (N, 4)
    target_bboxes = torch.stack(target_bboxes)  # (N, 4)
    
    # Convert YOLO format (cx, cy, w, h) to xyxy format
    pred_xyxy = bbox_yolo_to_xyxy(pred_bboxes)
    target_xyxy = bbox_yolo_to_xyxy(target_bboxes)
    
    # Compute GIoU
    giou = compute_giou(pred_xyxy, target_xyxy)
    
    # GIoU loss = 1 - GIoU
    loss = (1.0 - giou).mean()
    
    return loss


def bbox_yolo_to_xyxy(bbox):
    """
    Convert YOLO format to xyxy format.
    
    Args:
        bbox: (N, 4) in format (cx, cy, w, h), normalized [0, 1]
    
    Returns:
        (N, 4) in format (x1, y1, x2, y2)
    """
    cx, cy, w, h = bbox.unbind(dim=-1)
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_giou(pred_xyxy, target_xyxy):
    """
    Compute GIoU between predicted and target bboxes.
    
    Args:
        pred_xyxy: (N, 4) predicted bboxes in xyxy format
        target_xyxy: (N, 4) target bboxes in xyxy format
    
    Returns:
        (N,) GIoU values in range [-1, 1]
    """
    # Intersection area
    x1_inter = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1_inter = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2_inter = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2_inter = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    
    inter_w = (x2_inter - x1_inter).clamp(min=0)
    inter_h = (y2_inter - y1_inter).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union area
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    x1_enclosing = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1_enclosing = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2_enclosing = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2_enclosing = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    
    enclosing_area = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)
    
    # GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-6)
    
    return giou


class CPNLoss(nn.Module):
    """
    Combined loss for CPN training.
    
    Loss = w_focal * Focal + w_l1 * L1 + w_giou * GIoU
    
    Hardcoded weights:
    - w_focal = 1.0
    - w_l1 = 5.0
    - w_giou = 2.0
    """
    
    def __init__(self):
        super().__init__()
        
        # Hardcoded loss weights
        self.w_focal = 1.0
        self.w_l1 = 5.0
        self.w_giou = 2.0
        
        # Focal loss hyperparameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # Positive threshold (correlation above this = positive)
        self.pos_threshold = 0.5
    
    def forward(self, pred_objectness, pred_bbox, target_bboxes):
        """
        Compute combined loss.
        
        Args:
            pred_objectness: (B, 1, H, W) or (B, H, W) predicted objectness logits (before sigmoid)
            pred_bbox: (B, 4, H, W) predicted bbox offsets
            target_bboxes: List of B tensors, each (4,) in YOLO format
        
        Returns:
            Dictionary of losses
        """
        # Handle both (B, 1, H, W) and (B, H, W) shapes
        if pred_objectness.dim() == 4:
            pred_objectness = pred_objectness.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        B, H, W = pred_objectness.shape
        device = pred_objectness.device
        
        # Stack target bboxes
        target_bboxes = torch.stack(target_bboxes)  # (B, 4)
        
        # Generate target objectness map (Gaussian around target center)
        target_objectness = self._generate_target_map(target_bboxes, H, W, device)
        
        # Compute focal loss (pred_objectness is logits now)
        loss_focal = focal_loss(
            pred_objectness,
            target_objectness,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma
        )
        
        # Positive mask (locations where target > threshold)
        pos_mask = (target_objectness > self.pos_threshold).float()
        
        # Normalize bbox predictions to [0, 1] before regression losses
        pred_bbox_norm = torch.sigmoid(pred_bbox)
        
        # Compute L1 loss
        loss_l1 = l1_loss(pred_bbox_norm, target_bboxes, pos_mask)
        
        # Compute GIoU loss
        loss_giou = giou_loss(pred_bbox_norm, target_bboxes, pos_mask)
        
        # Combined loss
        loss_total = (
            self.w_focal * loss_focal +
            self.w_l1 * loss_l1 +
            self.w_giou * loss_giou
        )
        
        return {
            'loss': loss_total,
            'loss_focal': loss_focal,
            'loss_l1': loss_l1,
            'loss_giou': loss_giou
        }
    
    def _generate_target_map(self, target_bboxes, H, W, device):
        """
        Generate Gaussian target map around bbox center.
        
        Args:
            target_bboxes: (B, 4) in YOLO format (cx, cy, w, h), normalized
            H, W: Spatial dimensions of feature map
            device: torch device
        
        Returns:
            (B, H, W) Gaussian target map
        """
        B = target_bboxes.shape[0]
        
        # Create coordinate grid (normalized [0, 1])
        y_coords = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W)
        
        # Target centers
        cx = target_bboxes[:, 0].view(B, 1, 1).expand(B, H, W)
        cy = target_bboxes[:, 1].view(B, 1, 1).expand(B, H, W)
        
        # Gaussian std based on bbox size (smaller objects = smaller Gaussian)
        w = target_bboxes[:, 2].view(B, 1, 1)
        h = target_bboxes[:, 3].view(B, 1, 1)
        sigma = torch.sqrt(w * h) / 6.0  # std = ~1/6 of bbox diagonal
        sigma = sigma.expand(B, H, W)
        
        # Compute Gaussian
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        target_map = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-6))
        
        return target_map


if __name__ == "__main__":
    # Test losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy predictions
    B, H, W = 4, 72, 128
    pred_objectness = torch.rand(B, H, W, device=device)
    pred_bbox = torch.rand(B, 4, H, W, device=device)
    
    # Dummy targets
    target_bboxes = [torch.tensor([0.5, 0.5, 0.1, 0.1], device=device) for _ in range(B)]
    
    # Compute loss
    criterion = CPNLoss()
    losses = criterion(pred_objectness, pred_bbox, target_bboxes)
    
    print("Loss test:")
    print(f"  Total: {losses['loss']:.4f}")
    print(f"  Focal: {losses['loss_focal']:.4f}")
    print(f"  L1: {losses['loss_l1']:.4f}")
    print(f"  GIoU: {losses['loss_giou']:.4f}")
