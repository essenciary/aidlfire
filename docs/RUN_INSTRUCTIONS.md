# Run Instructions

Complete instructions to download data, generate patches, train models, run the app, and execute tests.

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager.

---

## 1. Install Dependencies

```bash
cd fire-pipeline

# Base (data processing, patch generation)
uv sync

# + Training (PyTorch, segmentation-models-pytorch, wandb, s2cloudless)
uv sync --extra train

# + App (Streamlit, folium, planetary-computer)
uv sync --extra app

# + Download (huggingface_hub for CEMS)
uv sync --extra download

# + Tests (pytest)
uv sync --extra test

# Everything at once
uv sync --extra all --extra download --extra test
```

**Recommended for full workflow:** `uv sync --extra train --extra download --extra app --extra test`

---

## 2. Data Pipeline

### 2.1 Download CEMS Dataset

Download the CEMS Wildfire Dataset from HuggingFace. Supports resume if interrupted.

```bash
cd fire-pipeline

# Download to project root (creates ../wildfires-cems/)
uv run python download_dataset.py --output-dir ../wildfires-cems

# Download only train split (faster for testing)
uv run python download_dataset.py --output-dir ../wildfires-cems --splits train

# Skip extraction (download archives only)
uv run python download_dataset.py --output-dir ../wildfires-cems --no-extract

# Download + extract + generate binary patches in one go
uv run python download_dataset.py --output-dir ../wildfires-cems --generate-patches --patches-dir ../patches --mask-type DEL
```

**Output structure:**
```
../wildfires-cems/
├── data/
│   ├── train/   # EMSR* folders with GeoTIFFs
│   ├── val/
│   └── test/
└── csv_files/
```

### 2.2 Download Sen2Fire Dataset

Sen2Fire is a separate dataset. Download from [Zenodo 10881058](https://zenodo.org/records/10881058) and place it so you have:

```
../data-sen2fire/
├── scene1/   # .npz files (train)
├── scene2/   # .npz files (train)
├── scene3/   # .npz files (val)
└── scene4/   # .npz files (test)
```

No patch generation needed—the loader reads `.npz` directly and center-crops 512→256.

### 2.3 Generate Binary Patches (CEMS DEL)

Binary fire/no-fire masks for Phase 1 training.

```bash
cd fire-pipeline

uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ../patches \
    --mask-type DEL
```

**Options:**
- `--skip-extraction` — Skip tar extraction if already done
- `--no-ndvi` — Use 7 channels instead of 8 (default: 8 with NDVI)
- `--force` — Regenerate all patches (default: skip existing)

**Output:** `../patches/train/`, `../patches/val/`, `../patches/test/` with `*_image.npy` and `*_mask.npy` (binary 0/1).

### 2.4 Generate Severity Patches (CEMS GRA)

5-class severity masks for Phase 2 training. Only CEMS images with GRA annotations are included.

```bash
cd fire-pipeline

uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ../patches_gra \
    --mask-type GRA
```

**Output:** `../patches_gra/train/`, `../patches_gra/val/`, `../patches_gra/test/` with masks 0–4 (no damage, negligible, moderate, high, destroyed).

### 2.5 Optional: Analyze Class Distribution

```bash
uv run python analyze_patches.py ../patches
uv run python analyze_patches.py ../patches --plot   # with histogram
```

---

## 3. Training

### 3.1 Model 1: ResNet34 + U-Net (Balance)

**Phase 1 — Binary (CEMS DEL + Sen2Fire):**
```bash
cd fire-pipeline

uv run python train_combined_binary.py \
    --patches-dir ../patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/v3_combined_binary_resnet34_unet \
    --encoder resnet34 \
    --architecture unet \
    --epochs 50
```

**Phase 2 — Severity (CEMS GRA):**
```bash
uv run python train_severity_finetune.py \
    --checkpoint ./output/v3_combined_binary_resnet34_unet/checkpoints/best_model.pt \
    --patches-dir ../patches_gra \
    --output-dir ./output/v3_finetune_severity_resnet34_unet \
    --epochs 30
```

**Checkpoints:** `./output/v3_finetune_severity_resnet34_unet/checkpoints/best_model.pt`

### 3.2 Model 2: ResNet50 + U-Net++ (Best Accuracy)

**Phase 1 — Binary:**
```bash
cd fire-pipeline

uv run python train_combined_binary.py \
    --patches-dir ../patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/v3_combined_binary_resnet50_unetpp \
    --encoder resnet50 \
    --architecture unetplusplus \
    --epochs 50
```

**Phase 2 — Severity:**
```bash
uv run python train_severity_finetune.py \
    --checkpoint ./output/v3_combined_binary_resnet50_unetpp/checkpoints/best_model.pt \
    --patches-dir ../patches_gra \
    --output-dir ./output/v3_finetune_severity_resnet50_unetpp \
    --epochs 30
```

**Checkpoints:** `./output/v3_finetune_severity_resnet50_unetpp/checkpoints/best_model.pt`

### 3.3 Optional: W&B Logging

Add `--wandb` and `--project fire-detection` to any training command:

```bash
uv run python train_combined_binary.py \
    --patches-dir ../patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/v3_combined_binary_resnet50_unetpp \
    --encoder resnet50 \
    --architecture unetplusplus \
    --epochs 50 \
    --wandb \
    --project fire-detection
```

---

## 4. Run the App

### 4.1 With a Single Model

```bash
cd fire-pipeline

# Copy checkpoint to app location
mkdir -p checkpoints
cp ./output/v3_finetune_severity_resnet50_unetpp/checkpoints/best_model.pt ./checkpoints/

# Run Streamlit app
env FIRE_MODELS_DIR=./output FIRE_MODEL_NAME=best_model FIRE_USE_MOCK=false uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8555
```

### 4.2 With Multiple Models (Dropdown)

```bash
cd fire-pipeline

# Point to output dir; app lists subdirs as model options
env FIRE_MODELS_DIR=./output FIRE_MODEL_NAME=best_model FIRE_USE_MOCK=false uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8555
```

### 4.3 Demo Mode (No Model)

```bash
uv run streamlit run app.py
```

Uses mock data for demonstration without a trained model.

### 4.4 Inference (CLI)

```bash
uv run python inference.py checkpoints/best_model.pt path/to/satellite_image.tif
```

---

## 5. Run Tests

```bash
cd fire-pipeline

# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Specific test file
uv run pytest tests/test_inference.py

# With coverage report
uv run pytest --cov=. --cov-report=html
```

---

## 6. Quick Reference: Full Workflow

```bash
cd fire-pipeline

# 1. Install
uv sync --extra train --extra download --extra app --extra test

# 2. Download CEMS
uv run python download_dataset.py --output-dir ../wildfires-cems

# 3. Download Sen2Fire (manual: Zenodo 10881058) → ../data-sen2fire/

# 4. Generate patches
uv run python run_pipeline.py --dataset-dir ../wildfires-cems --output-dir ../patches --mask-type DEL
uv run python run_pipeline.py --dataset-dir ../wildfires-cems --output-dir ../patches_gra --mask-type GRA

# 5. Train ResNet50+U-Net++ (best accuracy)
uv run python train_combined_binary.py --patches-dir ../patches --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/v3_combined_binary_resnet50_unetpp --encoder resnet50 --architecture unetplusplus --epochs 50
uv run python train_severity_finetune.py \
    --checkpoint ./output/v3_combined_binary_resnet50_unetpp/checkpoints/best_model.pt \
    --patches-dir ../patches_gra --output-dir ./output/v3_finetune_severity_resnet50_unetpp --epochs 30

# 6. Run app
env FIRE_MODELS_DIR=./output FIRE_MODEL_NAME=best_model FIRE_USE_MOCK=false uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8555

# 7. Run tests
uv run pytest
```

---

## 7. Path Summary

| Path | Description |
|------|--------------|
| `../wildfires-cems` | CEMS dataset (download output) |
| `../data-sen2fire` | Sen2Fire dataset (manual download) |
| `../patches` | Binary patches (DEL) |
| `../patches_gra` | Severity patches (GRA) |
| `./output/v3_*` | Training outputs, checkpoints |

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` for patches | Run patch generation first; check `--dataset-dir` and `--output-dir` |
| Sen2Fire not found | Ensure `../data-sen2fire` has `scene1/`, `scene2/`, `scene3/`, `scene4/` |
| Out of GPU memory | Reduce `--batch-size` (default 16) in training |
| Download interrupted | Re-run `download_dataset.py`; it resumes automatically |
| Patch generation interrupted | Re-run `run_pipeline.py`; it skips existing patches |
