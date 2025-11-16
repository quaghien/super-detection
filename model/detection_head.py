"""Lightweight detection head for bbox prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Lightweight detection head: 320 -> 160 -> 160 -> outputs.
    
    Takes fused correlation + P3 features, outputs objectness + bbox offsets.
    """
    
    def __init__(self, in_channels=321, hidden_dim=160):
        """
        Args:
            in_channels: 320 (P3) + 1 (correlation) = 321
            hidden_dim: 160 (lightweight)
        """
        super().__init__()
        
        # Shared feature extraction: 321 -> 160 -> 160
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification branch: objectness (1 channel)
        self.cls_head = nn.Conv2d(hidden_dim, 1, 1)
        
        # Regression branch: bbox offsets (4 channels: dx, dy, dw, dh)
        self.reg_head = nn.Conv2d(hidden_dim, 4, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special init for cls head (to have low initial confidence)
        nn.init.constant_(self.cls_head.bias, -2.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 321, H, W) - concatenated correlation + P3 features
        
        Returns:
            objectness: (B, 1, H, W) - sigmoid scores in [0, 1]
            bbox_offsets: (B, 4, H, W) - (dx, dy, dw, dh)
        """
        # Shared feature extraction
        x = self.conv1(x)  # (B, 160, H, W)
        x = self.conv2(x)  # (B, 160, H, W)
        
        # Classification: objectness with sigmoid
        objectness = torch.sigmoid(self.cls_head(x))  # (B, 1, H, W)
        
        # Regression: bbox offsets (no activation, direct regression)
        bbox_offsets = self.reg_head(x)  # (B, 4, H, W)
        
        return objectness, bbox_offsets


class ScaleFusion(nn.Module):
    """
    Fuse multi-scale correlation maps with learnable weights.
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable scalar weights for each scale
        self.w3 = nn.Parameter(torch.tensor(0.5))
        self.w4 = nn.Parameter(torch.tensor(0.3))
        self.w5 = nn.Parameter(torch.tensor(0.2))
    
    def forward(self, corr_maps, target_size):
        """
        Args:
            corr_maps: Dict with keys ['p3', 'p4', 'p5']
            target_size: (H, W) - target resolution (usually P3 size)
        
        Returns:
            fused: (B, 1, H, W) - fused correlation heatmap
        """
        # Upsample all to target size
        corr_p3 = F.interpolate(corr_maps['p3'], size=target_size, mode='bilinear', align_corners=False)
        corr_p4 = F.interpolate(corr_maps['p4'], size=target_size, mode='bilinear', align_corners=False)
        corr_p5 = F.interpolate(corr_maps['p5'], size=target_size, mode='bilinear', align_corners=False)
        
        # Weighted sum
        fused = self.w3 * corr_p3 + self.w4 * corr_p4 + self.w5 * corr_p5
        
        return fused


if __name__ == "__main__":
    # Test detection head
    B, H, W = 2, 121, 65
    x = torch.randn(B, 321, H, W)
    
    head = DetectionHead(in_channels=321, hidden_dim=160)
    
    with torch.no_grad():
        objectness, bbox_offsets = head(x)
    
    print("Detection Head Test:")
    print(f"  Input: {x.shape}")
    print(f"  Objectness: {objectness.shape}, range: [{objectness.min():.3f}, {objectness.max():.3f}]")
    print(f"  BBox offsets: {bbox_offsets.shape}")
    
    total_params = sum(p.numel() for p in head.parameters())
    print(f"  Total params: {total_params/1e3:.1f}K")
    
    # Test scale fusion
    print("\nScale Fusion Test:")
    corr_maps = {
        'p3': torch.randn(B, 1, 121, 65),
        'p4': torch.randn(B, 1, 57, 29),
        'p5': torch.randn(B, 1, 25, 11)
    }
    
    fusion = ScaleFusion()
    fused = fusion(corr_maps, target_size=(121, 65))
    print(f"  Fused correlation: {fused.shape}")
    print(f"  Weights: w3={fusion.w3.item():.3f}, w4={fusion.w4.item():.3f}, w5={fusion.w5.item():.3f}")
