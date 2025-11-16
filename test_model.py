"""Test script to check model parameters."""

import torch
from model.cpn import CPN


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model Architecture: CPN (Correlation Pyramid Network)")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"{'='*60}\n")
    
    # Count by component
    backbone_params = 0
    correlation_params = 0
    detection_params = 0
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params += param.numel()
        elif 'correlation' in name or 'template_fusion' in name:
            correlation_params += param.numel()
        elif 'detection_head' in name or 'scale_fusion' in name:
            detection_params += param.numel()
    
    print("Parameters by component:")
    print(f"  Backbone (ConvNeXt-Tiny + FPN): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    print(f"  Correlation + Fusion: {correlation_params:,} ({correlation_params/1e6:.2f}M)")
    print(f"  Detection Head: {detection_params:,} ({detection_params/1e6:.2f}M)")
    print(f"{'='*60}\n")


def test_forward():
    """Test forward pass."""
    print("Testing forward pass...")
    
    # Create model
    model = CPN(pretrained=False)
    model.eval()
    
    # Dummy inputs
    templates = torch.randn(2, 3, 3, 384, 384)  # (B=2, 3 templates, C=3, H=384, W=384)
    search = torch.randn(2, 3, 576, 1024)       # (B=2, C=3, H=576, W=1024)
    
    # Forward
    with torch.no_grad():
        objectness, bbox = model(templates, search)
    
    print(f"✓ Forward pass successful")
    print(f"  Input templates: {templates.shape}")
    print(f"  Input search: {search.shape}")
    print(f"  Output objectness: {objectness.shape}")
    print(f"  Output bbox: {bbox.shape}")
    print()


def test_inference():
    """Test inference mode."""
    print("Testing inference mode...")
    
    # Create model
    model = CPN(pretrained=False)
    model.eval()
    
    # Dummy inputs
    templates = torch.randn(1, 3, 3, 384, 384)
    search = torch.randn(1, 3, 576, 1024)
    
    # Inference
    with torch.no_grad():
        results = model.inference(templates, search, conf_threshold=0.5)
    
    print(f"✓ Inference successful")
    print(f"  Detected {len(results)} object(s)")
    if len(results) > 0:
        for i, result in enumerate(results):
            print(f"  Object {i+1}: conf={result['confidence']:.3f}, "
                  f"bbox=[{result['bbox'][0]:.3f}, {result['bbox'][1]:.3f}, "
                  f"{result['bbox'][2]:.3f}, {result['bbox'][3]:.3f}]")
    print()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CPN Model Test")
    print("="*60)
    
    # Create model
    model = CPN(pretrained=False)
    
    # Count parameters
    count_parameters(model)
    
    # Test forward
    test_forward()
    
    # Test inference
    test_inference()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
