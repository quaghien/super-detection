"""Main CPN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .correlation import CorrelationPyramid
from .detection_head import DetectionHead, ScaleFusion


class CPN(nn.Module):
    """
    Correlation Pyramid Network for Super Small Object Detection.
    
    Architecture:
    1. ConvNeXt-Tiny + FPN backbone (shared for templates & search)
    2. Multi-template fusion with learnable attention
    3. Depthwise correlation at 3 scales
    4. Scale fusion + lightweight detection head
    """
    
    def __init__(self, pretrained=True, fpn_channels=320, kernel_size=8, hidden_dim=160):
        super().__init__()
        
        self.fpn_channels = fpn_channels
        
        # Shared backbone for templates and search
        self.backbone = build_backbone(pretrained=pretrained, fpn_channels=fpn_channels)
        
        # Correlation pyramid with template fusion
        self.correlation = CorrelationPyramid(channels=fpn_channels, kernel_size=kernel_size)
        
        # Multi-scale correlation fusion
        self.scale_fusion = ScaleFusion()
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=fpn_channels + 1,  # P3 features + fused correlation
            hidden_dim=hidden_dim
        )
    
    def forward(self, templates, search):
        """
        Args:
            templates: (B, 3, 3, H_t, W_t) - 3 template images
            search: (B, 3, H_s, W_s) - search image
        
        Returns:
            objectness: (B, 1, H_out, W_out) - detection confidence
            bbox_offsets: (B, 4, H_out, W_out) - (dx, dy, dw, dh)
        """
        B = search.shape[0]
        
        # Extract template features (3 templates)
        template_pyramids = []
        for i in range(3):
            template_i = templates[:, i]  # (B, 3, H_t, W_t)
            pyramid_i = self.backbone(template_i)
            template_pyramids.append(pyramid_i)
        
        # Extract search features
        search_pyramid = self.backbone(search)
        
        # Compute multi-scale correlation
        corr_maps = self.correlation(template_pyramids, search_pyramid)
        
        # Get P3 correlation size for fusion target
        _, _, H_corr, W_corr = corr_maps['p3'].shape
        
        # Fuse multi-scale correlations
        corr_fused = self.scale_fusion(corr_maps, target_size=(H_corr, W_corr))
        
        # Crop P3 search features to match correlation size
        # (due to valid correlation padding)
        p3_search = search_pyramid['p3']
        _, _, H_p3, W_p3 = p3_search.shape
        
        # Center crop P3 to match correlation size
        h_start = (H_p3 - H_corr) // 2
        w_start = (W_p3 - W_corr) // 2
        p3_cropped = p3_search[:, :, h_start:h_start+H_corr, w_start:w_start+W_corr]
        
        # Concatenate correlation + P3 features
        combined = torch.cat([corr_fused, p3_cropped], dim=1)  # (B, 321, H, W)
        
        # Detection head
        objectness, bbox_offsets = self.detection_head(combined)
        
        return objectness, bbox_offsets
    
    def inference(self, templates, search, conf_threshold=0.5):
        """
        Inference mode: return decoded bboxes.
        
        Args:
            templates: (B, 3, 3, H_t, W_t)
            search: (B, 3, H_s, W_s)
            conf_threshold: float - minimum confidence
        
        Returns:
            List of detections per batch item:
            Each detection: dict with keys ['bbox', 'confidence']
            bbox: [cx, cy, w, h] in absolute coordinates
        """
        self.eval()
        with torch.no_grad():
            objectness, bbox_offsets = self.forward(templates, search)
        
        B, _, H_out, W_out = objectness.shape
        _, _, H_search, W_search = search.shape
        
        # Decode predictions
        detections = []
        for b in range(B):
            obj_map = objectness[b, 0]  # (H_out, W_out)
            offsets = bbox_offsets[b]  # (4, H_out, W_out)
            
            # Find peak
            max_conf = obj_map.max().item()
            if max_conf < conf_threshold:
                detections.append(None)
                continue
            
            # Peak location
            max_idx = obj_map.argmax()
            i_max = max_idx // W_out
            j_max = max_idx % W_out
            
            # Extract offsets at peak
            dx = offsets[0, i_max, j_max].item()
            dy = offsets[1, i_max, j_max].item()
            dw = offsets[2, i_max, j_max].item()
            dh = offsets[3, i_max, j_max].item()
            
            # Decode bbox (P3 is 1/8 scale)
            cell_size = 8
            cx = (j_max + dx) * cell_size
            cy = (i_max + dy) * cell_size
            w = dw * W_search
            h = dh * H_search
            
            detections.append({
                'bbox': [cx, cy, w, h],
                'confidence': max_conf
            })
        
        return detections


def build_cpn(pretrained=True, fpn_channels=320, kernel_size=8, hidden_dim=160):
    """Build CPN model."""
    return CPN(
        pretrained=pretrained,
        fpn_channels=fpn_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim
    )


if __name__ == "__main__":
    # Test full model
    B = 2
    templates = torch.randn(B, 3, 3, 384, 384)
    search = torch.randn(B, 3, 1024, 576)
    
    model = build_cpn(pretrained=False)
    
    print("CPN Model Test:")
    print(f"  Templates: {templates.shape}")
    print(f"  Search: {search.shape}")
    
    with torch.no_grad():
        objectness, bbox_offsets = model(templates, search)
    
    print(f"  Objectness: {objectness.shape}")
    print(f"  BBox offsets: {bbox_offsets.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Stats:")
    print(f"  Total params: {total_params/1e6:.2f}M")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M")
    
    # Test inference
    print("\nInference Test:")
    detections = model.inference(templates, search, conf_threshold=0.1)
    for i, det in enumerate(detections):
        if det is not None:
            print(f"  Batch {i}: bbox={det['bbox']}, conf={det['confidence']:.3f}")
        else:
            print(f"  Batch {i}: No detection")
