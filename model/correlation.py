"""Depthwise correlation module for template matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateFusion(nn.Module):
    """
    Learnable fusion of 3 templates with attention weights.
    """
    
    def __init__(self, channels=320):
        super().__init__()
        
        # MLP to score each template (GAP -> FC -> score)
        self.score_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1)
        )
    
    def forward(self, template_features):
        """
        Args:
            template_features: List of 3 tensors, each (B, C, H, W)
        
        Returns:
            fused: (B, C, H, W) - attention-weighted fusion
        """
        B, C, H, W = template_features[0].shape
        
        # Global average pool each template: (B, C, H, W) -> (B, C)
        template_vecs = [F.adaptive_avg_pool2d(feat, 1).flatten(1) 
                        for feat in template_features]
        
        # Score each template: (B, C) -> (B, 1)
        scores = [self.score_mlp(vec) for vec in template_vecs]
        scores = torch.cat(scores, dim=1)  # (B, 3)
        
        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=1)  # (B, 3)
        
        # Weighted fusion
        fused = sum(attn_weights[:, i:i+1, None, None] * template_features[i]
                   for i in range(3))
        
        return fused


class CorrelationPyramid(nn.Module):
    """
    Multi-scale depthwise correlation between template and search features.
    """
    
    def __init__(self, channels=320, kernel_size=8):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Adaptive pooling to fixed kernel size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(kernel_size)
        
        # Template fusion for each scale
        self.fusion_p3 = TemplateFusion(channels)
        self.fusion_p4 = TemplateFusion(channels)
        self.fusion_p5 = TemplateFusion(channels)
    
    def _normalize_features(self, x):
        """L2 normalize per channel."""
        return F.normalize(x, p=2, dim=1)
    
    def _depthwise_correlation(self, template_kernel, search_features):
        """
        Depthwise correlation using grouped convolution.
        
        Args:
            template_kernel: (B, C, K, K) - template kernel
            search_features: (B, C, H, W) - search features
        
        Returns:
            corr: (B, 1, H_out, W_out) - correlation heatmap
        """
        B, C, H, W = search_features.shape
        K = template_kernel.shape[2]
        
        # Normalize
        template_norm = self._normalize_features(template_kernel)
        search_norm = self._normalize_features(search_features)
        
        # Reshape template for depthwise conv: (B, C, K, K) -> (B*C, 1, K, K)
        template_weight = template_norm.reshape(B * C, 1, K, K)
        
        # Reshape search: (B, C, H, W) -> (1, B*C, H, W) for grouped conv
        search_input = search_norm.reshape(1, B * C, H, W)
        
        # Depthwise correlation via grouped conv
        # groups=B*C means each channel correlates independently
        corr = F.conv2d(search_input, template_weight, groups=B * C, padding=0)
        
        # Reshape back: (1, B*C, H_out, W_out) -> (B, C, H_out, W_out)
        _, _, H_out, W_out = corr.shape
        corr = corr.reshape(B, C, H_out, W_out)
        
        # Sum across channels -> single heatmap
        corr = corr.sum(dim=1, keepdim=True)  # (B, 1, H_out, W_out)
        
        return corr
    
    def forward(self, template_pyramids, search_pyramid):
        """
        Args:
            template_pyramids: List of 3 dicts, each with keys ['p3', 'p4', 'p5']
                Each pyramid level: (B, C, H, W)
            search_pyramid: Dict with keys ['p3', 'p4', 'p5']
                Each level: (B, C, H, W)
        
        Returns:
            corr_maps: Dict with keys ['p3', 'p4', 'p5']
                Each: (B, 1, H_out, W_out) - correlation heatmaps
        """
        # Fuse 3 templates at each scale
        template_p3 = self.fusion_p3([t['p3'] for t in template_pyramids])
        template_p4 = self.fusion_p4([t['p4'] for t in template_pyramids])
        template_p5 = self.fusion_p5([t['p5'] for t in template_pyramids])
        
        # Pool templates to fixed kernel size
        template_p3_kernel = self.adaptive_pool(template_p3)  # (B, C, 8, 8)
        template_p4_kernel = self.adaptive_pool(template_p4)
        template_p5_kernel = self.adaptive_pool(template_p5)
        
        # Compute depthwise correlation at each scale
        corr_p3 = self._depthwise_correlation(template_p3_kernel, search_pyramid['p3'])
        corr_p4 = self._depthwise_correlation(template_p4_kernel, search_pyramid['p4'])
        corr_p5 = self._depthwise_correlation(template_p5_kernel, search_pyramid['p5'])
        
        return {
            'p3': corr_p3,
            'p4': corr_p4,
            'p5': corr_p5
        }


if __name__ == "__main__":
    # Test correlation module
    B, C, K = 2, 320, 8
    
    # Create dummy template and search features
    template_pyramids = [
        {
            'p3': torch.randn(B, C, 48, 48),
            'p4': torch.randn(B, C, 24, 24),
            'p5': torch.randn(B, C, 12, 12)
        }
        for _ in range(3)
    ]
    
    search_pyramid = {
        'p3': torch.randn(B, C, 128, 72),
        'p4': torch.randn(B, C, 64, 36),
        'p5': torch.randn(B, C, 32, 18)
    }
    
    model = CorrelationPyramid(channels=C, kernel_size=K)
    
    with torch.no_grad():
        corr_maps = model(template_pyramids, search_pyramid)
    
    print("Correlation Pyramid Test:")
    for key, val in corr_maps.items():
        print(f"  {key}: {val.shape}")
