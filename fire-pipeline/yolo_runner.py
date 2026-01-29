"""

This file is responsible ONLY for:
  - Training and validating Ultralytics YOLOv8 DETECTION on a prepared dataset.

It expects a standard Ultralytics dataset layout like:

  <dataset_root>/
    images/train/*.tif
    images/val/*.tif
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

Where:
  - Each image is a 7-channel multi-band TIFF (CHW: 7,H,W)
  - Each label is YOLO-detection format:
      <class> <x_center> <y_center> <width> <height>
    with all coordinates normalized to [0..1].

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class YoloDetTrainCfg:
    """
    Training configuration.

    imgsz:  Ultralytics training resolution (square)
    batch:  Ultralytics batch size
    epochs: Training epochs
    device: Ultralytics device string ("0", "cpu", "mps", etc.)
    model_weights: which YOLO weights to start from (detection weights)
    """
    imgsz: int = 512
    batch: int = 16
    epochs: int = 50
    device: str = "0"
    model_weights: str = "yolov8n.pt"


def train_and_validate_yolo_det7(
    data_yaml: Path | str,
    output_dir: Path | str,
    cfg: YoloDetTrainCfg = YoloDetTrainCfg(),
) -> dict[str, Any]:
    """
    Train + validate YOLOv8 detection using Ultralytics.

    Args:
        data_yaml: Path to the exported dataset YAML (contains channels: 7).
        output_dir: Directory where Ultralytics will write runs/ artifacts.
        cfg: training configuration.

    Returns:
        Dict with train/val metrics (when available).
    """
    from ultralytics import YOLO

    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)

    # Load a detection model (NOT segmentation).
    model = YOLO(cfg.model_weights)

    # Train
    train_res = model.train(
        data=str(data_yaml),
        task="detect",
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        epochs=cfg.epochs,
        device=cfg.device,
        project=str(output_dir),
        name="runs",
        exist_ok=True,

        # IMPORTANT for multispectral:
        # HSV augmentations assume RGB semantics -> disable.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    # Validate
    val_res = model.val(
        data=str(data_yaml),
        task="detect",
        device=cfg.device,
    )

    return {
        "data_yaml": str(data_yaml),
        "train_results": getattr(train_res, "results_dict", None),
        "val_results": getattr(val_res, "results_dict", None),
    }
