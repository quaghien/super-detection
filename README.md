# CPN (Correlation Pyramid Network) for Super Small Object Detection

## Overview

Correlation-based detection model for tiny objects (~72×65px in 1024×576 images). Uses direct feature correlation between templates and search images instead of attention mechanisms.

**Key Features:**
- ConvNeXt-Tiny + FPN backbone (**30.86M params total**)
- Depthwise correlation with 8×8 template kernels
- Learnable template fusion (attention-weighted)
- Lightweight detection head (321→160→160)
- FP16 mixed precision training

**Model Breakdown:**
- Backbone (ConvNeXt-Tiny + FPN): 30.09M params
- Correlation + Fusion: 0.08M params  
- Detection Head: 0.69M params

## Architecture

```
Templates (3×3×384×384)     Search (3×1024×576)
         ↓                            ↓
    ConvNeXt-Tiny              ConvNeXt-Tiny
         ↓                            ↓
    FPN (320ch)                 FPN (320ch)
    P3, P4, P5                 P3, P4, P5
         ↓                            ↓
   Template Fusion          [Match at each scale]
   (3→1 weighted)                    ↓
         ↓                    Depthwise Correlation
    Fused Template            (8×8 kernels, 320ch)
         ↓                            ↓
         └──────────[Correlate]───────┘
                       ↓
                 Correlation Maps
                 (P3: 128×72, P4: 64×36, P5: 32×18)
                       ↓
                 Scale Fusion
                 (learned w3, w4, w5)
                       ↓
                 Fused Correlation
                 (321ch: 320 corr + 1 scale)
                       ↓
                Detection Head
                (321→160→160)
                       ↓
        ┌──────────────┴──────────────┐
  Objectness (1ch)            Bbox (4ch)
  [sigmoid, 0-1]         [cx, cy, w, h]
```

## Model Components

### 1. Backbone (ConvNeXt-Tiny + FPN)
- **Base:** ConvNeXt-Tiny from timm (28M params)
- **FPN:** 3 levels at 1/8, 1/16, 1/32 scales
- **Output:** 320 channels at each scale
- **Features:** P3 (128×72), P4 (64×36), P5 (32×18)

### 2. Correlation Module
- **Template Fusion:** MLP(320→80→1) with softmax attention
- **Kernel Size:** 8×8 (via AdaptiveAvgPool)
- **Method:** Depthwise via grouped convolution
- **Normalization:** L2 normalization for robustness

### 3. Detection Head
- **Input:** 321 channels (320 correlation + 1 scale indicator)
- **Architecture:** Conv(321→160) → Conv(160→160) → Branches
- **Outputs:**
  - Objectness: Conv(160→1) + sigmoid
  - Bbox: Conv(160→4) for (cx, cy, w, h) offsets
- **Params:** ~80K (lightweight)

### 4. Scale Fusion
- **Learnable weights:** w3, w4, w5 (initialized 0.5, 0.3, 0.2)
- **Method:** Softmax normalization → weighted average
- **Scale indicator:** Additional channel marking dominant scale

## Training

### Quick Start
```bash
cd /home/quanghien/aivn/super-small-object
python train.py
```

### Production Command (Full Parameters)
```bash
python train.py \
  --data_dir /home/quanghien/aivn \
  --batch_size 16 \
  --max_lr 1e-4 \
  --min_lr 1e-6 \
  --epochs 100 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --augment_prob 0.3 \
  --template_size 384 \
  --search_width 1024 \
  --search_height 576 \
  --pretrained_backbone imagenet \
  --save_dir checkpoints \
  --save_freq 5 \
  --num_workers 8 \
  --device cuda
```

**Key Parameters:**
- `--max_lr`: Maximum learning rate (default: 1e-4)
- `--min_lr`: Minimum learning rate for cosine annealing (default: 1e-6)
- `--warmup_epochs`: Linear warmup epochs (default: 5)
- Logging: **Per epoch only** (no per-iteration logs)

**Expected Output:**
```
=== Training ===
Max LR: 1.00e-04, Min LR: 1.00e-06

Epoch 1/100 | LR: 2.00e-05 | Train Loss: 0.5234 (F:0.123, L1:2.456, G:-1.234) | Val Loss: 0.4567 (F:0.098, L1:2.123, G:-0.987)
  → Best: best_checkpoint.pth (val_loss=0.4567)
Epoch 2/100 | LR: 4.00e-05 | Train Loss: 0.4123 (F:0.089, L1:1.987, G:-0.765) | Val Loss: 0.3890 (F:0.076, L1:1.876, G:-0.654)
  → Best: best_checkpoint.pth (val_loss=0.3890)
...
Epoch 5/100 | LR: 1.00e-04 | Train Loss: 0.2567 (F:0.045, L1:1.234, G:-0.321) | Val Loss: 0.2345 (F:0.038, L1:1.123, G:-0.234)
  → Saved: checkpoint_epoch_5.pth
  → Best: best_checkpoint.pth (val_loss=0.2345)
```

### Continue from Checkpoint
```bash
python train.py \
  --checkpoint checkpoints/checkpoint_epoch_50.pth \
  --max_lr 1e-4 \
  --min_lr 1e-6 \
  --batch_size 16 \
  --epochs 100
```
**Note:** Always restarts from epoch 0 even when loading checkpoint (as requested).

### Key Configuration
- **Input:**
  - Templates: 384×384 (resized from 2992×2992)
  - Search: 1024×576 (original resolution)
- **Augmentation:** Search images only (not templates)
  - Horizontal flip (p=0.5)
  - Shift/scale/rotate (slight, p=0.7)
  - Color jitter (p=0.5)
  - Gaussian blur (p=0.3)
  - Gaussian noise (p=0.15)
- **Loss weights (hardcoded):**
  - Focal loss: 1.0
  - L1 loss: 5.0
  - GIoU loss: 2.0
- **Optimizer:**
  - AdamW
  - Backbone: 0.1× max_lr
  - Detection head: 1.0× max_lr
- **Scheduler:**
  - Linear warmup (5 epochs)
  - Cosine annealing from max_lr to min_lr
  - Always uses cosine scheduler
- **Precision:** 
  - Mixed precision training (autocast + GradScaler)
  - Model in FP32, operations auto-cast to FP16 when beneficial
  - Gradients computed in FP32 for numerical stability
- **Logging:** Per epoch only (no per-iteration logging)

## Dataset Structure

```
data_dir/
├── train/
│   ├── templates/
│   │   ├── Class_0_ref_001.jpg  (2992×2992)
│   │   ├── Class_0_ref_002.jpg
│   │   ├── Class_0_ref_003.jpg
│   │   └── ...
│   └── search/
│       ├── images/
│       │   ├── Class_0_frame_XXXXXX.jpg  (1024×576)
│       │   └── ...
│       └── labels/
│           ├── Class_0_frame_XXXXXX.txt  (YOLO: cls cx cy w h)
│           └── ...
└── val/
    └── [same structure]
```

## Checkpoint Format

- **Saved:** Model state_dict only (no optimizer/scheduler)
- **Precision:** FP32 (for mixed precision training compatibility)
- **Loading:** Automatic FP16→FP32 conversion if loading old FP16 checkpoints
- **Frequency:** Every 5 epochs + best model + last checkpoint
- **Note:** Checkpoints are saved in FP32 to ensure compatibility with mixed precision training

## Design Rationale

### Why Correlation over Attention?
1. **Identical appearance:** Templates and objects have exactly the same appearance
2. **Direct matching:** Correlation explicitly computes feature similarity
3. **Interpretability:** Peak correlation = object location
4. **Efficiency:** Simpler than multi-head attention

### Why Depthwise Correlation?
1. **Spatial structure:** 8×8 kernels preserve local patterns
2. **Channel-wise:** Each feature channel matches independently
3. **Robustness:** Better than global pooling (1×1) for small objects

### Why Lightweight Detection Head?
1. **Location known:** Correlation already finds object position
2. **Refinement only:** Just need to fine-tune bbox and confidence
3. **Efficiency:** ~80K params vs millions in attention-based heads

### Why Multi-Scale?
1. **Tiny objects:** Need high-resolution features (P3 at 1/8 scale)
2. **Robustness:** Multiple scales improve detection stability
3. **Learnable fusion:** Network learns optimal scale combination

## Performance Targets

- **Objects:** ~72×65px (0.25% of image area)
- **Dataset:** ~20K training images
- **Target metric:** IoU > 0.5
- **Model size:** **30.86M parameters**
  - Backbone (ConvNeXt-Tiny + FPN): 30.09M
  - Correlation + Fusion: 0.08M
  - Detection Head: 0.69M
- **Speed:** Real-time inference on GPU

## Testing Model

```bash
# Check model parameters and architecture
python test_model.py
```

Output:
```
============================================================
Model Architecture: CPN (Correlation Pyramid Network)
============================================================
Total parameters: 30,864,843 (30.86M)
Trainable parameters: 30,864,843 (30.86M)
============================================================

Parameters by component:
  Backbone (ConvNeXt-Tiny + FPN): 30,093,472 (30.09M)
  Correlation + Fusion: 77,283 (0.08M)
  Detection Head: 694,088 (0.69M)
============================================================
```

## Files

```
super-small-object/
├── model/
│   ├── __init__.py           # Module exports
│   ├── backbone.py           # ConvNeXt-Tiny + FPN
│   ├── correlation.py        # Template fusion + correlation
│   ├── detection_head.py     # Lightweight detection head
│   ├── losses.py             # Focal + L1 + GIoU losses
│   └── cpn.py                # Main CPN model
├── utils/
│   ├── __init__.py
│   └── dataset.py            # Dataset + augmentation
├── train.py                  # Training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Next Steps

1. **Training:** Run training with production config
2. **Validation:** Monitor IoU and loss convergence
3. **Inference:** Implement inference script for evaluation
4. **Optimization:** Fine-tune hyperparameters if needed

## Notes

- **Augmentation verified:** Bbox transformations match image augmentations (albumentations handles this automatically)
- **FP16 safe:** All operations compatible with mixed precision
- **Checkpoint strategy:** Save model only (not optimizer), restart from epoch 0 on load
- **No config files:** Loss weights and critical params are hardcoded in code
