"""Dataset and data loading utilities."""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CPNDataset(Dataset):
    """
    Dataset for CPN training.
    
    Structure:
    data_dir/
    ├── train/ or val/
    │   ├── templates/
    │   │   ├── Class_0_ref_001.jpg
    │   │   ├── Class_0_ref_002.jpg
    │   │   └── Class_0_ref_003.jpg
    │   └── search/
    │       ├── images/
    │       │   └── Class_0_frame_XXXXXX.jpg
    │       └── labels/
    │           └── Class_0_frame_XXXXXX.txt (YOLO format: cls cx cy w h)
    """
    
    def __init__(self, data_dir, split='train', template_size=384, 
                 search_size=(1024, 576), augment_prob=0.3):
        """
        Args:
            data_dir: Root directory
            split: 'train' or 'val'
            template_size: Template resize resolution
            search_size: (W, H) for search images (keep original)
            augment_prob: Probability of applying augmentation (search only)
        """
        self.data_dir = Path(data_dir) / split
        self.template_size = template_size
        self.search_size = search_size
        self.augment_prob = augment_prob if split == 'train' else 0.0
        
        # Get all search images
        self.search_dir = self.data_dir / 'search' / 'images'
        self.label_dir = self.data_dir / 'search' / 'labels'
        self.template_dir = self.data_dir / 'templates'
        
        self.search_images = sorted(list(self.search_dir.glob('*.jpg')))
        
        # Build template mapping: class_name -> list of 3 template paths
        self.template_map = self._build_template_map()
        
        # Augmentation (for search only, geometric + light color)
        self.search_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
                scale=(0.93, 1.07),
                rotate=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.03,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.15),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
            clip=True  # Clip bboxes to [0, 1] range
        ))
        
        # Template transform (no augmentation, just resize + normalize)
        self.template_transform = A.Compose([
            A.Resize(template_size, template_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Search transform (resize + normalize, augmentation applied separately)
        self.search_transform = A.Compose([
            A.Resize(search_size[1], search_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"Loaded {len(self.search_images)} images from {split}")
        print(f"  Templates: {len(self.template_map)} classes")
        print(f"  Augmentation: {self.augment_prob}")
    
    def _build_template_map(self):
        """Build mapping from class name to 3 template paths."""
        template_map = {}
        
        for template_path in sorted(self.template_dir.glob('*_ref_*.jpg')):
            # Parse: Class_0_ref_001.jpg -> Class_0
            name = template_path.stem  # e.g., "Class_0_ref_001"
            class_name = '_'.join(name.split('_')[:-2])  # "Class_0"
            
            if class_name not in template_map:
                template_map[class_name] = []
            
            template_map[class_name].append(template_path)
        
        # Verify each class has exactly 3 templates
        for class_name, paths in template_map.items():
            assert len(paths) == 3, f"Class {class_name} has {len(paths)} templates (need 3)"
        
        return template_map
    
    def _get_class_name(self, search_path):
        """Extract class name from search image filename."""
        # e.g., "Class_0_frame_003585.jpg" -> "Class_0"
        name = search_path.stem
        parts = name.split('_frame_')
        return parts[0]
    
    def __len__(self):
        return len(self.search_images)
    
    def __getitem__(self, idx):
        # Load search image
        search_path = self.search_images[idx]
        search_img = cv2.imread(str(search_path))
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
        
        # Load label (YOLO format: cls cx cy w h - but we ignore cls)
        label_path = self.label_dir / (search_path.stem + '.txt')
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            cx, cy, w, h = map(float, parts[1:5])  # Skip class, only take bbox
        
        # Apply search augmentation (before resize)
        if np.random.rand() < self.augment_prob:
            augmented = self.search_aug(
                image=search_img,
                bboxes=[[cx, cy, w, h]],
                class_labels=[0]  # Dummy class for albumentations
            )
            search_img = augmented['image']
            if len(augmented['bboxes']) > 0:
                cx, cy, w, h = augmented['bboxes'][0]
            else:
                # If bbox was cropped out, use original
                pass
        
        # Resize + normalize search
        search_transformed = self.search_transform(image=search_img)
        search_tensor = search_transformed['image']
        
        # Load 3 templates
        class_name = self._get_class_name(search_path)
        template_paths = self.template_map[class_name]
        
        template_tensors = []
        for template_path in template_paths:
            template_img = cv2.imread(str(template_path))
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
            
            # Transform template (no augmentation)
            template_transformed = self.template_transform(image=template_img)
            template_tensors.append(template_transformed['image'])
        
        templates = torch.stack(template_tensors, dim=0)  # (3, 3, H, W)
        
        # Target bbox (normalized YOLO format) - no class needed
        target = {
            'bbox': torch.tensor([cx, cy, w, h], dtype=torch.float32)
        }
        
        return templates, search_tensor, target


def collate_fn(batch):
    """Custom collate function."""
    templates = torch.stack([item[0] for item in batch])  # (B, 3, 3, H, W)
    searches = torch.stack([item[1] for item in batch])   # (B, 3, H, W)
    targets = [item[2] for item in batch]  # List of dicts
    
    return templates, searches, targets


if __name__ == "__main__":
    # Test dataset
    dataset = CPNDataset(
        data_dir='/home/quanghien/aivn',
        split='val',
        template_size=384,
        search_size=(1024, 576),
        augment_prob=0.0
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test sample
    templates, search, target = dataset[0]
    print(f"\nSample 0:")
    print(f"  Templates: {templates.shape}")
    print(f"  Search: {search.shape}")
    print(f"  Target bbox: {target['bbox']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"\nDataLoader test:")
    for batch_idx, (templates, searches, targets) in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        print(f"    Templates: {templates.shape}")
        print(f"    Searches: {searches.shape}")
        print(f"    Targets: {len(targets)} items")
        if batch_idx == 0:
            break
