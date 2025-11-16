"""ConvNeXt-Tiny backbone with FPN for multi-scale feature extraction."""

import torch
import torch.nn as nn
import timm


class ConvNeXtFPN(nn.Module):
    """
    ConvNeXt-Tiny + FPN backbone.
    
    Output: 3-level pyramid (P3, P4, P5) all with 320 channels.
    - P3: 1/8 scale (high-res for tiny objects)
    - P4: 1/16 scale (medium-res)
    - P5: 1/32 scale (context)
    """
    
    def __init__(self, pretrained=True, fpn_channels=320):
        super().__init__()
        
        # Load pretrained ConvNeXt-Tiny from timm
        self.convnext = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # C3 (1/8), C4 (1/16), C5 (1/32)
        )
        
        # ConvNeXt-Tiny feature channels: [192, 384, 768]
        feature_channels = self.convnext.feature_info.channels()
        self.c3_channels = feature_channels[0]  # 192
        self.c4_channels = feature_channels[1]  # 384
        self.c5_channels = feature_channels[2]  # 768
        
        # FPN lateral convs (1x1 to reduce channels to fpn_channels)
        self.lateral_c3 = nn.Conv2d(self.c3_channels, fpn_channels, 1)
        self.lateral_c4 = nn.Conv2d(self.c4_channels, fpn_channels, 1)
        self.lateral_c5 = nn.Conv2d(self.c5_channels, fpn_channels, 1)
        
        # FPN smooth convs (3x3 to reduce upsampling artifacts)
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        
        self.fpn_channels = fpn_channels
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize FPN layers."""
        for m in [self.lateral_c3, self.lateral_c4, self.lateral_c5,
                  self.smooth_p3, self.smooth_p4]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        
        Returns:
            dict: {
                'p3': (B, 320, H/8, W/8),
                'p4': (B, 320, H/16, W/16),
                'p5': (B, 320, H/32, W/32)
            }
        """
        # Extract multi-scale features from ConvNeXt
        features = self.convnext(x)  # List of 3 tensors
        c3, c4, c5 = features
        
        # FPN lateral connections
        l3 = self.lateral_c3(c3)  # (B, 320, H/8, W/8)
        l4 = self.lateral_c4(c4)  # (B, 320, H/16, W/16)
        l5 = self.lateral_c5(c5)  # (B, 320, H/32, W/32)
        
        # Top-down pathway
        p5 = l5  # (B, 320, H/32, W/32)
        
        # Upsample P5 and add to L4
        p4 = l4 + nn.functional.interpolate(
            p5, size=l4.shape[2:], mode='nearest'
        )
        p4 = self.smooth_p4(p4)  # (B, 320, H/16, W/16)
        
        # Upsample P4 and add to L3
        p3 = l3 + nn.functional.interpolate(
            p4, size=l3.shape[2:], mode='nearest'
        )
        p3 = self.smooth_p3(p3)  # (B, 320, H/8, W/8)
        
        return {
            'p3': p3,
            'p4': p4,
            'p5': p5
        }


def build_backbone(pretrained=True, fpn_channels=320):
    """Build ConvNeXt-Tiny + FPN backbone."""
    return ConvNeXtFPN(pretrained=pretrained, fpn_channels=fpn_channels)


if __name__ == "__main__":
    # Test backbone
    model = build_backbone(pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    
    with torch.no_grad():
        out = model(x)
    
    print("ConvNeXt-Tiny + FPN Test:")
    for key, val in out.items():
        print(f"  {key}: {val.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.2f}M")
