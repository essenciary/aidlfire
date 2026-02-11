"""Integration and end-to-end tests for fire detection pipeline."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from satellite_fetcher import MockSatelliteFetcher, SatelliteImage
from storage import StorageManager
from inference import InferenceResult, create_visualization


class TestFullPipeline:
    """End-to-end tests for the complete pipeline."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create temporary storage."""
        return StorageManager(tmp_path)

    @pytest.fixture
    def fetcher(self):
        """Create mock satellite fetcher."""
        return MockSatelliteFetcher()

    def test_fetch_to_storage_pipeline(self, fetcher, storage):
        """Test complete pipeline: fetch → analyze → store → retrieve."""
        # 1. Fetch satellite imagery
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-06-01", "2024-06-30")

        image = fetcher.fetch_region(bbox, date_range, max_cloud_cover=30)

        assert image is not None
        assert image.data.shape[2] == 7

        # 2. Create mock inference result (simulating model output)
        h, w = image.data.shape[:2]
        segmentation = np.random.randint(0, 2, (h, w), dtype=np.uint8)
        probabilities = np.random.rand(h, w, 2).astype(np.float32)

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=probabilities,
            has_fire=bool(segmentation.any()),
            fire_confidence=float(probabilities[:, :, 1].max()),
            fire_fraction=float(segmentation.mean()),
            severity_counts={
                "background": int((segmentation == 0).sum()),
                "fire": int((segmentation == 1).sum()),
            },
            num_classes=2,
            image_shape=(h, w),
        )

        # 3. Create visualization
        visualization = create_visualization(image.data, result, alpha=0.5)

        assert visualization.shape == (h, w, 3)
        assert visualization.dtype == np.uint8

        # 4. Save to storage
        analysis_id = storage.save_analysis(
            satellite_image=image,
            inference_result=result,
            visualization=visualization,
            metadata={"bbox": list(bbox), "date_range": list(date_range)},
        )

        assert analysis_id is not None

        # 5. Retrieve and verify
        loaded = storage.load_analysis(analysis_id)

        assert loaded is not None
        assert loaded["record"].scene_id == image.scene_id
        assert loaded["record"].has_fire == result.has_fire
        np.testing.assert_array_equal(loaded["image"], image.data)
        np.testing.assert_array_equal(loaded["segmentation"], segmentation)

    def test_multiple_analyses_history(self, fetcher, storage):
        """Test storing and retrieving multiple analyses."""
        bbox = (-122.5, 37.5, -122.0, 38.0)

        analysis_ids = []
        for month in range(1, 4):  # 3 months
            date_range = (f"2024-{month:02d}-01", f"2024-{month:02d}-28")
            image = fetcher.fetch_region(bbox, date_range)

            result = InferenceResult(
                segmentation=np.zeros((256, 256), dtype=np.uint8),
                probabilities=np.zeros((256, 256, 2), dtype=np.float32),
                has_fire=month % 2 == 0,  # Alternate fire/no-fire
                fire_confidence=0.5 if month % 2 == 0 else 0.1,
                fire_fraction=0.1 if month % 2 == 0 else 0.0,
                severity_counts={},
                num_classes=2,
                image_shape=(256, 256),
            )

            aid = storage.save_analysis(
                satellite_image=image,
                inference_result=result,
            )
            analysis_ids.append(aid)

        # Test history retrieval
        all_history = storage.get_history(limit=10)
        assert len(all_history) == 3

        fire_history = storage.get_history(fire_only=True)
        assert len(fire_history) == 1  # Only month 2 has fire

        # Test statistics
        stats = storage.get_statistics()
        assert stats["total_analyses"] == 3
        assert stats["fire_detections"] == 1


class TestModelIntegration:
    """Integration tests for model components."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="No GPU available"
    )
    def test_model_forward_pass_gpu(self):
        """Test model forward pass on GPU."""
        from model import FireSegmentationModel

        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model = FireSegmentationModel(
            encoder_name="resnet18",  # Smaller for testing
            num_classes=2,
            in_channels=7,
        ).to(device)

        # Random input
        x = torch.randn(2, 7, 256, 256).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2, 256, 256)

    def test_model_forward_pass_cpu(self):
        """Test model forward pass on CPU."""
        from model import FireSegmentationModel

        model = FireSegmentationModel(
            encoder_name="resnet18",
            num_classes=2,
            in_channels=7,
        )

        x = torch.randn(1, 7, 128, 128)  # Smaller for CPU

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2, 128, 128)

    def test_model_prediction_methods(self):
        """Test model prediction helper methods."""
        from model import FireSegmentationModel

        model = FireSegmentationModel(
            encoder_name="resnet18",
            num_classes=5,
            in_channels=7,
        )
        model.eval()

        x = torch.randn(2, 7, 64, 64)

        with torch.no_grad():
            # Test segmentation prediction
            seg = model.predict_segmentation(x)
            assert seg.shape == (2, 64, 64)
            assert seg.dtype == torch.int64

            # Test fire detection
            has_fire = model.predict_fire_detection(x)
            assert has_fire.shape == (2,)
            assert has_fire.dtype == torch.bool

            # Test fire confidence
            confidence = model.predict_fire_confidence(x)
            assert confidence.shape == (2,)
            assert (confidence >= 0).all() and (confidence <= 1).all()


class TestLossFunctions:
    """Integration tests for loss functions."""

    def test_combined_loss(self):
        """Test combined loss function."""
        from model import CombinedLoss

        loss_fn = CombinedLoss(dice_weight=0.5, ce_weight=0.5)

        logits = torch.randn(2, 2, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64))

        loss, components = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0
        assert "dice_loss" in components
        assert "ce_loss" in components

    def test_focal_loss(self):
        """Test focal loss function."""
        from model import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)

        logits = torch.randn(2, 2, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_focal_loss_focuses_hard_examples(self):
        """Test that focal loss reduces weight on easy examples."""
        from model import FocalLoss

        focal_loss = FocalLoss(gamma=2.0)

        # Easy example: high confidence correct prediction
        # logits shape: (N, C, H, W) = (1, 2, 2, 2)
        logits_easy = torch.zeros(1, 2, 2, 2)
        logits_easy[:, 0, :, :] = 5.0   # Confident class 0
        logits_easy[:, 1, :, :] = -5.0
        targets_easy = torch.zeros(1, 2, 2, dtype=torch.long)  # Class 0 is correct

        # Hard example: low confidence (logits close to 0)
        logits_hard = torch.zeros(1, 2, 2, 2)
        logits_hard[:, 0, :, :] = 0.2
        logits_hard[:, 1, :, :] = -0.2
        targets_hard = torch.zeros(1, 2, 2, dtype=torch.long)

        fl_easy = focal_loss(logits_easy, targets_easy)
        fl_hard = focal_loss(logits_hard, targets_hard)

        # Focal loss on easy examples should be much smaller than on hard examples
        # because (1-p)^gamma down-weights high confidence predictions
        assert fl_easy < fl_hard
        # The ratio should be very small (easy is down-weighted significantly)
        assert fl_easy / fl_hard < 0.1


class TestDatasetIntegration:
    """Integration tests for dataset components."""

    def test_dataset_with_augmentation(self, tmp_path):
        """Test dataset with augmentation pipeline."""
        from dataset import WildfirePatchDataset, get_training_augmentation

        # Create mock patches
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()

        for i in range(5):
            image = np.random.rand(256, 256, 7).astype(np.float32)
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
            np.save(patches_dir / f"patch_{i}_image.npy", image)
            np.save(patches_dir / f"patch_{i}_mask.npy", mask)

        # Create dataset with augmentation
        dataset = WildfirePatchDataset(
            patches_dir=patches_dir,
            augment=get_training_augmentation(),
        )

        assert len(dataset) == 5

        # Get item and verify shape
        image, mask = dataset[0]

        assert image.shape == (7, 256, 256)
        assert mask.shape == (256, 256)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_datamodule_splits(self, tmp_path):
        """Test data module with train/val/test splits."""
        from dataset import WildfireDataModule

        # Create mock patch directories
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / split
            split_dir.mkdir()
            for i in range(3):
                np.save(split_dir / f"patch_{i}_image.npy", np.random.rand(256, 256, 7).astype(np.float32))
                np.save(split_dir / f"patch_{i}_mask.npy", np.random.randint(0, 2, (256, 256), dtype=np.uint8))

        # Create data module
        dm = WildfireDataModule(
            patches_root=tmp_path,
            batch_size=2,
            num_workers=0,
        )

        # Get dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Verify batch
        images, masks = next(iter(train_loader))
        assert images.shape[0] <= 2  # Batch size
        assert images.shape[1] == 7  # Channels
