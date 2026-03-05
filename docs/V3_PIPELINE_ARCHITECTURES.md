# V3 Pipeline: Model Architectures & Training Commands

This document describes the model architectures used in the combined binary + severity workflow, recommended encoder/decoder combinations, and **copy-paste commands** to train each variant.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Available Options](#available-options)
- [Recommended Model Combinations](#recommended-model-combinations)
- [Training Commands by Model](#training-commands-by-model)
- [Phase 2 (Severity) Commands](#phase-2-severity-commands)

---

## Architecture Overview

The pipeline uses **segmentation_models_pytorch (smp)** with two components:

| Component | Role | Example |
|-----------|------|---------|
| **Encoder** | Backbone for feature extraction (pretrained on ImageNet) | ResNet34, EfficientNet-B0 |
| **Decoder** | Reconstructs segmentation map from encoder features | U-Net, U-Net++, DeepLabV3+ |

```
Input (8 ch) → Encoder → Decoder → Heads (binary 2-class, severity 5-class)
```

- **Phase 1**: Single-head binary (encoder + decoder + binary head)
- **Phase 2**: Dual-head (encoder + decoder + binary head frozen + severity head trained)

---

## Available Options

### Encoders (backbones)

| Encoder | Params (approx) | Use case |
|---------|-----------------|----------|
| `resnet18` | ~11M | Fast, lightweight |
| `resnet34` | ~21M | Default, balanced |
| `resnet50` | ~25M | Higher accuracy |
| `efficientnet-b0` | ~5M | Efficient, good quality |
| `efficientnet-b1` | ~7M | Stronger EfficientNet |
| `efficientnet-b2` | ~9M | Best quality (EfficientNet family) |
| `mobilenet_v2` | ~3.5M | Edge deployment |

### Decoder architectures

| Architecture | Description | Best for |
|--------------|-------------|----------|
| `unet` | Classic U-Net with skip connections | General purpose, robust baseline |
| `unetplusplus` | Nested skip connections, denser decoder | Fine boundaries, small burn scars |
| `deeplabv3plus` | Atrous convolutions, multi-scale context | Large burned areas, global context |

---

## Recommended Model Combinations

### 1. Default / Balanced

| Encoder | Architecture | Why |
|---------|--------------|-----|
| **resnet34** | **unet** | Solid baseline, moderate size, widely used. Good starting point. |

### 2. Higher Accuracy

| Encoder | Architecture | Why |
|---------|--------------|-----|
| **resnet50** | **unetplusplus** | Deeper encoder + denser skip connections. Better for fine-grained severity. |
| **efficientnet-b1** | **unetplusplus** | EfficientNet scales well; U-Net++ improves boundary detail. |

### 3. Fast / Lightweight

| Encoder | Architecture | Why |
|---------|--------------|-----|
| **resnet18** | **unet** | Smaller, faster. Good for quick experiments or limited compute. |
| **mobilenet_v2** | **unet** | Very lightweight. Suitable for edge or real-time deployment. |

### 4. Best Quality (more compute)

| Encoder | Architecture | Why |
|---------|--------------|-----|
| **efficientnet-b2** | **unetplusplus** | Strong encoder + decoder. Best for severity and small burn scars. |
| **resnet50** | **deeplabv3plus** | Atrous convolutions for multi-scale context. Good for large fires. |

---

## Training Commands by Model

**Prerequisites:** CEMS DEL patches in `./patches`, Sen2Fire in `../data-sen2fire`, CEMS GRA patches in `./patches_gra`.

```bash
cd fire-pipeline
uv sync --extra train
```

---

### Model 1: Default (ResNet34 + U-Net)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_resnet34_unet \
    --encoder resnet34 \
    --architecture unet \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_resnet34_unet/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_resnet34_unet \
    --epochs 30
```

---

### Model 2: Higher Accuracy (ResNet50 + U-Net++)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_resnet50_unetpp \
    --encoder resnet50 \
    --architecture unetplusplus \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_resnet50_unetpp/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_resnet50_unetpp \
    --epochs 30
```

---

### Model 3: EfficientNet + U-Net++ (EfficientNet-B1)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_efficientnet-b1_unetpp \
    --encoder efficientnet-b1 \
    --architecture unetplusplus \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_efficientnet-b1_unetpp/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_efficientnet-b1_unetpp \
    --epochs 30
```

---

### Model 4: Fast / Lightweight (ResNet18 + U-Net)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_resnet18_unet \
    --encoder resnet18 \
    --architecture unet \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_resnet18_unet/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_resnet18_unet \
    --epochs 30
```

---

### Model 5: Edge Deployment (MobileNet-V2 + U-Net)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_mobilenet_unet \
    --encoder mobilenet_v2 \
    --architecture unet \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_mobilenet_unet/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_mobilenet_unet \
    --epochs 30
```

---

### Model 6: Best Quality (EfficientNet-B2 + U-Net++)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_efficientnet-b2_unetpp \
    --encoder efficientnet-b2 \
    --architecture unetplusplus \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_efficientnet-b2_unetpp/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_efficientnet-b2_unetpp \
    --epochs 30
```

---

### Model 7: DeepLabV3+ (ResNet50 + DeepLabV3+)

```bash
# Phase 1
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_resnet50_deeplabv3plus \
    --encoder resnet50 \
    --architecture deeplabv3plus \
    --epochs 50

# Phase 2
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_resnet50_deeplabv3plus/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_resnet50_deeplabv3plus \
    --epochs 30
```

---

## Phase 2 (Severity) Commands

Phase 2 **inherits** encoder and architecture from the Phase 1 checkpoint. You do **not** pass `--encoder` or `--architecture` to `train_severity_finetune.py`; it reads them from the checkpoint config.

Each Phase 2 command above uses the matching Phase 1 checkpoint. Ensure the checkpoint path matches the Phase 1 output directory.

---

## Quick Reference Table

| Model | Encoder | Architecture | Output dir (Phase 1) | Output dir (Phase 2) |
|-------|---------|--------------|---------------------|----------------------|
| Default | resnet34 | unet | `combined_resnet34_unet` | `severity_resnet34_unet` |
| Higher accuracy | resnet50 | unetplusplus | `combined_resnet50_unetpp` | `severity_resnet50_unetpp` |
| EfficientNet-B1 | efficientnet-b1 | unetplusplus | `combined_efficientnet-b1_unetpp` | `severity_efficientnet-b1_unetpp` |
| Fast | resnet18 | unet | `combined_resnet18_unet` | `severity_resnet18_unet` |
| Edge | mobilenet_v2 | unet | `combined_mobilenet_unet` | `severity_mobilenet_unet` |
| Best quality | efficientnet-b2 | unetplusplus | `combined_efficientnet-b2_unetpp` | `severity_efficientnet-b2_unetpp` |
| DeepLabV3+ | resnet50 | deeplabv3plus | `combined_resnet50_deeplabv3plus` | `severity_resnet50_deeplabv3plus` |

---

## Data Preparation (run once)

```bash
cd fire-pipeline

# CEMS DEL patches (for Phase 1)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches \
    --mask-type DEL

# CEMS GRA patches (for Phase 2)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches_gra \
    --mask-type GRA
```

Sen2Fire: download from [Zenodo 10881058](https://zenodo.org/records/10881058) and place in `../data-sen2fire` with `scene1`, `scene2`, `scene3`, `scene4` subdirs.
