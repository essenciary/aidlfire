"""
Evaluation Metrics for Fire Detection and Segmentation.

Provides metrics for both tasks:
1. Segmentation: IoU, Dice, Precision, Recall per class
2. Binary Detection: Accuracy, Precision, Recall, F1

Usage:
    from metrics import SegmentationMetrics, DetectionMetrics

    seg_metrics = SegmentationMetrics(num_classes=5)
    det_metrics = DetectionMetrics()

    for images, masks in dataloader:
        logits = model(images)
        seg_metrics.update(logits, masks)
        det_metrics.update(logits, masks)

    print(seg_metrics.compute())
    print(det_metrics.compute())
"""


import torch
import torch.nn as nn
import numpy as np


class SegmentationMetrics:
    """
    Metrics for semantic segmentation evaluation.

    Computes per-class and mean metrics:
    - IoU (Intersection over Union / Jaccard Index)
    - Dice coefficient (F1 for segmentation)
    - Precision and Recall

    Args:
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore (e.g., background)
        class_names: Optional list of class names for reporting
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        class_names: list[str] | None = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices
        """
        preds = logits.argmax(dim=1)  # (B, H, W)

        # Flatten
        preds = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()

        # Filter ignored pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # Update confusion matrix using vectorized bincount
        # This is O(n) instead of O(n) per-element loop overhead
        indices = targets.long() * self.num_classes + preds.long()
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """
        Compute all metrics from accumulated confusion matrix.

        Returns:
            Dictionary containing:
            - per_class_iou: IoU for each class
            - per_class_dice: Dice for each class
            - per_class_precision: Precision for each class
            - per_class_recall: Recall for each class
            - mean_iou: Mean IoU across classes
            - mean_dice: Mean Dice across classes
            - fire_iou: IoU for fire classes (class > 0)
            - fire_dice: Dice for fire classes
        """
        cm = self.confusion_matrix.float()

        # Per-class metrics
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-6)

        # Dice = 2*TP / (2*TP + FP + FN)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-6)

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-6)

        # Mean metrics (excluding ignored class)
        valid_classes = torch.ones(self.num_classes, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_classes[self.ignore_index] = False

        mean_iou = iou[valid_classes].mean().item()
        mean_dice = dice[valid_classes].mean().item()

        # Fire-specific metrics (all classes except background)
        fire_classes = torch.arange(1, self.num_classes)
        fire_tp = tp[fire_classes].sum()
        fire_fp = fp[fire_classes].sum()
        fire_fn = fn[fire_classes].sum()

        fire_iou = (fire_tp / (fire_tp + fire_fp + fire_fn + 1e-6)).item()
        fire_dice = (2 * fire_tp / (2 * fire_tp + fire_fp + fire_fn + 1e-6)).item()
        fire_precision = (fire_tp / (fire_tp + fire_fp + 1e-6)).item()
        fire_recall = (fire_tp / (fire_tp + fire_fn + 1e-6)).item()

        results = {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "fire_iou": fire_iou,
            "fire_dice": fire_dice,
            "fire_precision": fire_precision,
            "fire_recall": fire_recall,
        }

        # Add per-class metrics
        for i, name in enumerate(self.class_names):
            results[f"iou_{name}"] = iou[i].item()
            results[f"dice_{name}"] = dice[i].item()
            results[f"precision_{name}"] = precision[i].item()
            results[f"recall_{name}"] = recall[i].item()

        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix as numpy array."""
        return self.confusion_matrix.numpy()


class DetectionMetrics:
    """
    Metrics for binary fire detection (patch-level classification).

    Fire detection is derived from segmentation: a patch "has fire" if
    any pixel is predicted as fire (class > 0).

    Computes:
    - Accuracy
    - Precision (of fire detection)
    - Recall (of fire detection)
    - F1 Score
    """

    def __init__(self, fire_threshold: float = 0.0):
        """
        Args:
            fire_threshold: Minimum fraction of fire pixels to consider
                           a patch as "has fire" in ground truth
        """
        self.fire_threshold = fire_threshold
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.tp = 0  # True positives (correctly detected fire)
        self.fp = 0  # False positives (predicted fire, no fire)
        self.tn = 0  # True negatives (correctly predicted no fire)
        self.fn = 0  # False negatives (missed fire)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices
        """
        preds = logits.argmax(dim=1)  # (B, H, W)

        # Derive binary detection
        # Prediction: any pixel predicted as fire?
        pred_has_fire = (preds > 0).any(dim=(1, 2))  # (B,)

        # Ground truth: any pixel is fire (above threshold)?
        if self.fire_threshold > 0:
            fire_fraction = (targets > 0).float().mean(dim=(1, 2))
            target_has_fire = fire_fraction > self.fire_threshold
        else:
            target_has_fire = (targets > 0).any(dim=(1, 2))  # (B,)

        # Update counts
        pred_has_fire = pred_has_fire.cpu()
        target_has_fire = target_has_fire.cpu()

        self.tp += ((pred_has_fire == 1) & (target_has_fire == 1)).sum().item()
        self.fp += ((pred_has_fire == 1) & (target_has_fire == 0)).sum().item()
        self.tn += ((pred_has_fire == 0) & (target_has_fire == 0)).sum().item()
        self.fn += ((pred_has_fire == 0) & (target_has_fire == 1)).sum().item()

    def compute(self) -> dict:
        """
        Compute detection metrics.

        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        total = self.tp + self.fp + self.tn + self.fn

        accuracy = (self.tp + self.tn) / (total + 1e-6)
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "detection_accuracy": accuracy,
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1,
            "detection_tp": self.tp,
            "detection_fp": self.fp,
            "detection_tn": self.tn,
            "detection_fn": self.fn,
        }


class CombinedMetrics:
    """
    Combined metrics for both segmentation and detection tasks.

    Convenience class that wraps both SegmentationMetrics and DetectionMetrics.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
        ignore_index: int | None = None,
    ):
        self.seg_metrics = SegmentationMetrics(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index,
        )
        self.det_metrics = DetectionMetrics()

    def reset(self):
        """Reset all metrics."""
        self.seg_metrics.reset()
        self.det_metrics.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update both segmentation and detection metrics."""
        self.seg_metrics.update(logits, targets)
        self.det_metrics.update(logits, targets)

    def compute(self) -> dict:
        """Compute all metrics."""
        results = {}
        results.update(self.seg_metrics.compute())
        results.update(self.det_metrics.compute())
        return results


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary as a readable string."""
    lines = []

    # Group metrics
    segmentation = {}
    detection = {}
    per_class = {}

    for key, value in metrics.items():
        if key.startswith("detection_"):
            detection[key] = value
        elif key.startswith(("iou_", "dice_", "precision_", "recall_")) and "fire" not in key:
            per_class[key] = value
        else:
            segmentation[key] = value

    # Format segmentation metrics
    if segmentation:
        lines.append(f"{prefix}Segmentation Metrics:")
        for key, value in segmentation.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

    # Format detection metrics
    if detection:
        lines.append(f"{prefix}Detection Metrics:")
        for key, value in detection.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...\n")

    num_classes = 5
    batch_size = 4
    height, width = 256, 256

    # Create fake predictions and targets
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test segmentation metrics
    seg_metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=["background", "negligible", "moderate", "high", "destroyed"],
    )
    seg_metrics.update(logits, targets)
    seg_results = seg_metrics.compute()

    print("Segmentation Metrics:")
    print(f"  Mean IoU: {seg_results['mean_iou']:.4f}")
    print(f"  Mean Dice: {seg_results['mean_dice']:.4f}")
    print(f"  Fire IoU: {seg_results['fire_iou']:.4f}")
    print(f"  Fire Recall: {seg_results['fire_recall']:.4f}")

    # Test detection metrics
    det_metrics = DetectionMetrics()
    det_metrics.update(logits, targets)
    det_results = det_metrics.compute()

    print("\nDetection Metrics:")
    print(f"  Accuracy: {det_results['detection_accuracy']:.4f}")
    print(f"  Precision: {det_results['detection_precision']:.4f}")
    print(f"  Recall: {det_results['detection_recall']:.4f}")
    print(f"  F1: {det_results['detection_f1']:.4f}")

    # Test combined metrics
    combined = CombinedMetrics(num_classes=num_classes)
    combined.update(logits, targets)
    all_results = combined.compute()

    print("\nCombined Metrics:")
    print(format_metrics(all_results))

    print("\n✅ All tests passed!")
