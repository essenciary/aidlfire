"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
import torch

from metrics import (
    SegmentationMetrics,
    DetectionMetrics,
    CombinedMetrics,
    format_metrics,
)


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        metrics = SegmentationMetrics(num_classes=2)

        # Perfect predictions with both classes present
        logits = torch.zeros(4, 2, 8, 8)
        targets = torch.zeros(4, 8, 8, dtype=torch.long)

        # Half class 0, half class 1
        logits[:, 0, :4, :] = 10.0   # Predict class 0 for top half
        logits[:, 1, :4, :] = -10.0
        logits[:, 0, 4:, :] = -10.0  # Predict class 1 for bottom half
        logits[:, 1, 4:, :] = 10.0
        targets[:, 4:, :] = 1        # Ground truth: class 1 in bottom half

        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["mean_iou"] == pytest.approx(1.0, abs=0.01)
        assert results["iou_class_0"] == pytest.approx(1.0, abs=0.01)
        assert results["iou_class_1"] == pytest.approx(1.0, abs=0.01)

    def test_fire_metrics(self):
        """Test fire-specific metrics."""
        metrics = SegmentationMetrics(
            num_classes=2,
            class_names=["background", "fire"],
        )

        # Create predictions with some fire
        logits = torch.zeros(2, 2, 4, 4)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)

        # Set up: half pixels are fire in both pred and target
        logits[0, 1, :2, :] = 10.0  # Predict fire in top half
        logits[0, 0, 2:, :] = 10.0  # Predict background in bottom half
        targets[0, :2, :] = 1  # Fire in top half

        logits[1, 0, :, :] = 10.0  # All background
        targets[1, :, :] = 0  # All background

        metrics.update(logits, targets)
        results = metrics.compute()

        assert "fire_iou" in results
        assert "fire_dice" in results
        assert "fire_precision" in results
        assert "fire_recall" in results

    def test_multiclass(self):
        """Test with 5 severity classes."""
        metrics = SegmentationMetrics(
            num_classes=5,
            class_names=["none", "low", "moderate", "high", "destroyed"],
        )

        logits = torch.randn(2, 5, 8, 8)
        targets = torch.randint(0, 5, (2, 8, 8))

        metrics.update(logits, targets)
        results = metrics.compute()

        assert "mean_iou" in results
        assert "iou_none" in results
        assert "iou_destroyed" in results

    def test_reset(self):
        """Test reset clears accumulated data."""
        metrics = SegmentationMetrics(num_classes=2)

        logits = torch.randn(2, 2, 8, 8)
        targets = torch.randint(0, 2, (2, 8, 8))
        metrics.update(logits, targets)

        metrics.reset()

        # Confusion matrix should be zeros
        cm = metrics.get_confusion_matrix()
        assert cm.sum() == 0

    def test_get_confusion_matrix(self):
        """Test confusion matrix extraction."""
        metrics = SegmentationMetrics(num_classes=2)

        # All predictions are class 0, all targets are class 1
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :] = 10.0
        targets = torch.ones(1, 4, 4, dtype=torch.long)

        metrics.update(logits, targets)
        cm = metrics.get_confusion_matrix()

        assert cm.shape == (2, 2)
        assert cm[1, 0] == 16  # 16 pixels: target=1, pred=0


class TestDetectionMetrics:
    """Tests for DetectionMetrics."""

    def test_perfect_detection(self):
        """Test with perfect fire detection."""
        metrics = DetectionMetrics()

        # Batch 1: has fire, predict fire
        logits1 = torch.zeros(1, 2, 8, 8)
        logits1[:, 1, 4, 4] = 10.0  # One fire pixel
        targets1 = torch.zeros(1, 8, 8, dtype=torch.long)
        targets1[:, 4, 4] = 1

        # Batch 2: no fire, predict no fire
        logits2 = torch.zeros(1, 2, 8, 8)
        logits2[:, 0, :, :] = 10.0
        targets2 = torch.zeros(1, 8, 8, dtype=torch.long)

        metrics.update(logits1, targets1)
        metrics.update(logits2, targets2)
        results = metrics.compute()

        assert results["detection_accuracy"] == pytest.approx(1.0, abs=1e-4)
        assert results["detection_precision"] == pytest.approx(1.0, abs=1e-4)
        assert results["detection_recall"] == pytest.approx(1.0, abs=1e-4)
        assert results["detection_f1"] == pytest.approx(1.0, abs=1e-4)

    def test_false_positive(self):
        """Test false positive detection."""
        metrics = DetectionMetrics()

        # Predict fire when there is none
        logits = torch.zeros(1, 2, 8, 8)
        logits[:, 1, 4, 4] = 10.0  # Predict fire
        targets = torch.zeros(1, 8, 8, dtype=torch.long)  # No actual fire

        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["detection_fp"] == 1
        assert results["detection_tp"] == 0
        assert results["detection_precision"] < 1.0

    def test_false_negative(self):
        """Test false negative (missed fire)."""
        metrics = DetectionMetrics()

        # Predict no fire when there is fire
        logits = torch.zeros(1, 2, 8, 8)
        logits[:, 0, :, :] = 10.0  # Predict all background
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[:, 4, 4] = 1  # Actual fire

        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["detection_fn"] == 1
        assert results["detection_recall"] < 1.0

    def test_fire_threshold(self):
        """Test fire threshold for detection."""
        metrics = DetectionMetrics(fire_threshold=0.1)  # 10% threshold

        # Small fire that doesn't meet threshold
        # Prediction predicts fire, but ground truth fire is below threshold
        logits = torch.zeros(1, 2, 10, 10)
        logits[:, 1, 0, 0] = 10.0  # 1 pixel fire prediction
        targets = torch.zeros(1, 10, 10, dtype=torch.long)
        targets[:, 0, 0] = 1  # 1 pixel = 1% < 10% threshold

        metrics.update(logits, targets)
        results = metrics.compute()

        # Prediction has fire (any pixel), but target doesn't meet threshold
        # So this is a false positive
        assert results["detection_fp"] == 1

        # Test case where no fire is predicted and target is below threshold
        metrics2 = DetectionMetrics(fire_threshold=0.1)
        logits2 = torch.zeros(1, 2, 10, 10)
        logits2[:, 0, :, :] = 10.0  # All background prediction
        targets2 = torch.zeros(1, 10, 10, dtype=torch.long)
        targets2[:, 0, 0] = 1  # 1 pixel = 1% < 10% threshold

        metrics2.update(logits2, targets2)
        results2 = metrics2.compute()

        # No fire predicted, target below threshold = true negative
        assert results2["detection_tn"] == 1


class TestCombinedMetrics:
    """Tests for CombinedMetrics."""

    def test_combines_both(self):
        """Test that combined metrics includes both types."""
        metrics = CombinedMetrics(num_classes=2)

        logits = torch.randn(2, 2, 8, 8)
        targets = torch.randint(0, 2, (2, 8, 8))

        metrics.update(logits, targets)
        results = metrics.compute()

        # Should have segmentation metrics
        assert "mean_iou" in results
        assert "fire_iou" in results

        # Should have detection metrics
        assert "detection_accuracy" in results
        assert "detection_f1" in results

    def test_reset_both(self):
        """Test that reset clears both metric types."""
        metrics = CombinedMetrics(num_classes=2)

        logits = torch.randn(2, 2, 8, 8)
        targets = torch.randint(0, 2, (2, 8, 8))
        metrics.update(logits, targets)

        metrics.reset()

        # Should be able to compute with zeros
        results = metrics.compute()
        assert results["detection_tp"] == 0


class TestFormatMetrics:
    """Tests for format_metrics function."""

    def test_formats_segmentation(self):
        """Test formatting segmentation metrics."""
        metrics = {
            "mean_iou": 0.75,
            "fire_iou": 0.65,
        }

        output = format_metrics(metrics)

        assert "Segmentation Metrics" in output
        assert "0.75" in output

    def test_formats_detection(self):
        """Test formatting detection metrics."""
        metrics = {
            "detection_accuracy": 0.95,
            "detection_f1": 0.88,
        }

        output = format_metrics(metrics)

        assert "Detection Metrics" in output
        assert "0.95" in output
