# Solution Report: Neural Architecture, Data & Computational Requirements

**Project:** Wildfire Detection from Sentinel-2 Satellite Imagery  
**For:** Postgraduate Project Defense

---

## 1. Overview

We deploy two segmentation models for wildfire burned-area mapping:

| Model | Role | Fire IoU (Phase 1) | Mean IoU (Phase 2) |
|-------|------|--------------------|--------------------|
| **ResNet50 + U-Net++** | Best accuracy | 0.78 | 0.34 |
| **ResNet34 + U-Net** | Balance of speed & performance | 0.77 | 0.34 |

Both use the same pipeline: **encoder–decoder** segmentation with pretrained backbones, 8-channel multispectral input, and a two-phase training strategy (binary → severity).

---

## 2. Neural Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WILDFIRE SEGMENTATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input (256×256×8)     Encoder           Decoder         Output Heads       │
│   ┌──────────────┐      ┌──────┐          ┌──────┐       ┌──────────────┐   │
│   │ 7 bands +    │      │      │          │      │       │ Binary (2)   │   │
│   │ NDVI         │ ──►  │ CNN  │ ──────►  │ Up   │ ───►  │ Severity (5) │   │
│   │ float32      │      │ Back │   skip   │ conv │       │ (dual-head)  │   │
│   └──────────────┘      │ bone │   conn.  │      │       └──────────────┘   │
│                         └──────┘          └──────┘                           │
│                         ResNet34/50       U-Net / U-Net++                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Encoder (Backbone)

The encoder extracts multi-scale features from 8-channel patches. We use **ResNet** pretrained on ImageNet, with the first conv layer adapted for 8 input channels.

```
                    ResNet Encoder (simplified)
                    ─────────────────────────

  Input 256×256×8
       │
       ▼
  ┌─────────────┐
  │ Conv 7×7    │  stride 2  ──► 128×128×64
  │ (8→64 ch)   │
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ MaxPool     │  stride 2  ──► 64×64×64
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ Layer 1     │  ResNet blocks  ──► 64×64×64   (ResNet34) or 256 (ResNet50)
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ Layer 2     │  stride 2       ──► 32×32×128  or 512
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ Layer 3     │  stride 2       ──► 16×16×256  or 1024
  └─────────────┘
       │
       ▼
  ┌─────────────┐
  │ Layer 4     │  stride 2       ──► 8×8×512    or 2048  (bottleneck)
  └─────────────┘
       │
       └──────────────────────────────────────────► Skip connections to decoder
```

| Encoder | Blocks | Params (encoder) | Output channels (bottleneck) |
|---------|--------|------------------|------------------------------|
| ResNet34 | 3,4,6,3 | ~21M | 512 |
| ResNet50 | 3,4,6,3 (bottleneck) | ~25M | 2048 |

### 2.3 Decoder: U-Net vs U-Net++

**U-Net (ResNet34 model):** Classic encoder–decoder with skip connections.

```
  U-Net: Direct skip connections
  ─────────────────────────────

  Encoder          Decoder
  ───────          ───────
     │                │
  64×64 ──────────► concat ──► up + conv
     │                │
  32×32 ──────────► concat ──► up + conv
     │                │
  16×16 ──────────► concat ──► up + conv
     │                │
   8×8  ──────────► concat ──► up + conv
     │                │
                     ▼
              256×256 × num_classes
```

**U-Net++ (ResNet50 model):** Nested skip connections for denser feature fusion.

```
  U-Net++: Nested dense skip connections
  ──────────────────────────────────────

  Encoder (down)     Decoder (up, dense)
  ─────────────      ───────────────────

  X0,0 ──────────────────────────────────────────────► output
    │
  X1,0 ──► X1,1 ──────────────────────────────────────►
    │         │
  X2,0 ──► X2,1 ──► X2,2 ─────────────────────────────►
    │         │         │
  X3,0 ──► X3,1 ──► X3,2 ──► X3,3 ─────────────────────►
    │         │         │         │
  X4,0 ──► X4,1 ──► X4,2 ──► X4,3 ──► X4,4 (bottleneck)

  Each node: concat(upsampled from below, skip from encoder, sibling outputs)
  → Denser decoder, better boundary refinement
  → ~2× params vs U-Net decoder
```

### 2.4 Dual-Head Output

After training, the model outputs two heads:

```
  Decoder output
       │
       ├──────────────────► Binary head (2 classes)  ──► fire / no-fire
       │
       └──────────────────► Severity head (5 classes) ──► 0: no damage
                                                                1: negligible
                                                                2: moderate
                                                                3: high
                                                                4: destroyed
```

Phase 1 trains the binary head; Phase 2 adds and trains the severity head while keeping the binary head frozen.

### 2.5 Model Comparison

| Component | ResNet34 + U-Net | ResNet50 + U-Net++ |
|-----------|------------------|--------------------|
| Encoder | ResNet34 (21M) | ResNet50 (25M) |
| Decoder | U-Net (direct skips) | U-Net++ (nested dense skips) |
| **Total params** | **~24.5M** | **~49M** |
| Best for | Speed, deployment | Accuracy |

---

## 3. Data Requirements

### 3.1 Input Format

| Property | Value |
|----------|-------|
| Patch size | 256 × 256 pixels |
| Channels | 8 (7 Sentinel-2 bands + NDVI) |
| Data type | float32, normalized [0, 1] |
| Source | Sentinel-2 L2A (surface reflectance) |

### 3.2 Spectral Bands (7 + NDVI)

| Index | Band | Wavelength | Role |
|-------|------|------------|------|
| 0 | B02 Blue | 490 nm | Water, smoke |
| 1 | B03 Green | 560 nm | Vegetation |
| 2 | B04 Red | 665 nm | Chlorophyll, burn scars |
| 3 | B08 NIR | 842 nm | Healthy vegetation (drops when burned) |
| 4 | B8A NIR-narrow | 865 nm | Vegetation boundary |
| 5 | B11 SWIR1 | 1610 nm | **Fire & burn detection** |
| 6 | B12 SWIR2 | 2190 nm | **Fire & burn detection** |
| 7 | NDVI | (NIR−Red)/(NIR+Red) | Vegetation index |

**Fire signal:** Burned areas show LOW NIR and HIGH SWIR. NDVI helps separate burn scars from water/shadow.

### 3.3 Datasets

| Dataset | Patches | Mask type | Use |
|---------|---------|-----------|-----|
| **CEMS DEL** | ~560 images → thousands of patches | Binary (0/1) | Phase 1 (binary) |
| **Sen2Fire** | 2,466 patches (512→256 crop) | Binary | Phase 1 (combined) |
| **CEMS GRA** | Subset with GRA annotations | 5-class (0–4) | Phase 2 (severity) |

### 3.4 Storage (from disk: `du -h`)

| Item | Size |
|------|------|
| CEMS DEL patches | 23 GB |
| CEMS GRA patches | 14 GB |
| Sen2Fire | 6 GB |
| **Total training data** | **43 GB** |

---

## 4. Computational Requirements

**Data sources:** Parameters from model code. Epochs and training time from W&B and observed runs (early stopping; V3 scripts do not log `train_time_s`).

### 4.1 Training

| Phase | Model | Epochs (actual) | Max epochs | Batch size | GPU memory | Training time |
|-------|-------|-----------------|------------|------------|------------|---------------|
| Phase 1 (binary) | ResNet34+U-Net | 17 | 50 | 16 | ~6–8 GB | ~15 min |
| Phase 1 (binary) | ResNet50+U-Net++ | 27 | 50 | 16 | ~10–12 GB | ~30 min |
| Phase 2 (severity) | Either | ~8–19 | 30 | 16 | ~6–12 GB | ~10–20 min |

**Hardware:** Single GPU (e.g. NVIDIA T4, V100, or consumer RTX 3080+). CPU-only training is possible but much slower.

### 4.2 Inference

| Model | Params | Checkpoint size | Inference (256×256, batch 1) | Use case |
|-------|--------|-----------------|-----------------------------|----------|
| ResNet34 + U-Net | 24.5M | 94 MB | ~10–20 ms/patch (GPU) | Real-time, edge |
| ResNet50 + U-Net++ | 49M | 188 MB | ~20–40 ms/patch (GPU) | Best accuracy |

*Checkpoint sizes from `du -h` on `fire-pipeline/output/*/checkpoints/best_model.pt` (dual-head, Phase 2).*

### 4.3 Summary Table

| Requirement | ResNet34 + U-Net | ResNet50 + U-Net++ |
|-------------|------------------|--------------------|
| Parameters | 24.5M | 49M |
| Checkpoint size | 94 MB | 188 MB |
| GPU memory (train) | ~6–8 GB | ~10–12 GB |
| GPU memory (inference) | ~2 GB | ~4 GB |
| Training time (Phase 1) | ~15 min (17 epochs) | ~30 min (27 epochs) |
| Inference speed | Faster | Slower |

---

## 5. References

- **Architecture:** [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models_pytorch)
- **V3 pipeline:** `docs/V3_PIPELINE_ARCHITECTURES.md`
- **Patch format:** `fire-pipeline/PATCHES.md`
- **Experimental results:** `docs/WANDB_ANALYSIS_REPORT.md`
