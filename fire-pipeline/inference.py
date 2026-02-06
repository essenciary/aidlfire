"""
Inference Pipeline for Fire Detection.

Runs trained model on new satellite imagery and generates predictions.

Usage:
    from inference import FireInferencePipeline

    pipeline = FireInferencePipeline("checkpoints/best_model.pt")
    result = pipeline.predict_from_array(image_array)  # (H, W, 7)
    result = pipeline.predict_from_file("satellite_image.tif")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from model import FireSegmentationModel
from constants import (
    BAND_INDICES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE_INFERENCE,
    get_device,
    get_class_names,
)


@dataclass
class InferenceResult:
    """Result of fire detection inference."""

    # Raw outputs
    segmentation: np.ndarray  # (H, W) class indices
    probabilities: np.ndarray  # (H, W, num_classes) probabilities

    # Derived metrics
    has_fire: bool  # Binary detection
    fire_confidence: float  # Max fire probability
    fire_fraction: float  # Fraction of pixels with fire
    severity_counts: dict  # Count per severity level

    # Metadata
    num_classes: int
    image_shape: tuple

    def __post_init__(self):
        """Validate the inference result."""
        if self.segmentation.ndim != 2:
            raise ValueError(f"segmentation must be 2D, got shape {self.segmentation.shape}")
        if self.probabilities.ndim != 3:
            raise ValueError(f"probabilities must be 3D, got shape {self.probabilities.shape}")
        if self.probabilities.shape[2] != self.num_classes:
            raise ValueError(
                f"probabilities last dim ({self.probabilities.shape[2]}) must match "
                f"num_classes ({self.num_classes})"
            )
        if not 0 <= self.fire_confidence <= 1:
            raise ValueError(f"fire_confidence must be in [0, 1], got {self.fire_confidence}")
        if not 0 <= self.fire_fraction <= 1:
            raise ValueError(f"fire_fraction must be in [0, 1], got {self.fire_fraction}")

    def get_fire_mask(self) -> np.ndarray:
        """Get binary fire mask (1 = fire, 0 = no fire)."""
        return (self.segmentation > 0).astype(np.uint8)

    def get_severity_map(self) -> np.ndarray:
        """Get severity map (same as segmentation for GRA)."""
        return self.segmentation

    def get_fire_probability_map(self) -> np.ndarray:
        """Get probability map for fire (sum of all fire classes)."""
        if self.num_classes == 2:
            return self.probabilities[:, :, 1]
        else:
            return self.probabilities[:, :, 1:].sum(axis=2)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "has_fire": self.has_fire,
            "fire_confidence": self.fire_confidence,
            "fire_fraction": self.fire_fraction,
            "severity_counts": self.severity_counts,
            "num_classes": self.num_classes,
            "image_shape": self.image_shape,
        }


class FireInferencePipeline:
    """
    Pipeline for running fire detection inference on new images.

    Handles:
    - Loading trained model
    - Preprocessing (band selection, normalization, patching)
    - Batch inference
    - Stitching predictions back together
    - Post-processing and result generation

    Args:
        model_path: Path to trained model checkpoint
        device: Device to run inference on (auto, cuda, mps, cpu)
        patch_size: Size of patches for inference (default: 256)
        stride: Stride for patch extraction (default: 256, no overlap)
        batch_size: Batch size for inference
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        patch_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE_INFERENCE,
        batch_size: int = 8,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

        # Setup device using shared utility
        self.device = get_device(device)

        # Load model
        self.model, self.config = self._load_model(model_path)
        self.num_classes = self.config.get("num_classes", 2)

    def _load_model(self, model_path: Union[str, Path]) -> tuple[nn.Module, dict]:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")

        config = checkpoint.get("config", {})
        num_classes = config.get("num_classes", 2)
        encoder_name = config.get("encoder_name", "resnet34")
        architecture = config.get("architecture", "unet")

        model = FireSegmentationModel(
            encoder_name=encoder_name,
            num_classes=num_classes,
            in_channels=7,
            encoder_weights=None,  # Don't load pretrained, we have our weights
            architecture=architecture,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        return model, config

    def preprocess_image(
        self,
        image: np.ndarray,
        select_bands: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image array
                   - If (H, W, 12): Select 7 bands
                   - If (H, W, 7): Use as-is
            select_bands: Whether to select bands from 12-band input

        Returns:
            Preprocessed image (H, W, 7) float32, normalized to 0-1
        """
        if image.size == 0:
            raise ValueError(
                "Cannot process empty image. The satellite fetcher may have returned "
                "empty data—check that the region intersects the scene."
            )

        # Handle band selection
        if select_bands and image.shape[-1] == 12:
            image = image[:, :, list(BAND_INDICES)]
        elif image.shape[-1] != 7:
            raise ValueError(
                f"Expected 7 or 12 channels, got {image.shape[-1]}. "
                "Set select_bands=False if image already has 7 channels."
            )

        # Normalize to 0-1 if needed
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 10000.0  # Sentinel-2 scaling
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / image.max()
        else:
            image = image.astype(np.float32)

        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)

        return image

    def extract_patches(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Extract patches from image using sliding window.

        Args:
            image: Input image (H, W, C)

        Returns:
            patches: Array of patches (N, H, W, C)
            positions: List of (row, col) positions for each patch

        Raises:
            ValueError: If image is smaller than patch_size
        """
        h, w, c = image.shape

        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than patch_size ({self.patch_size}). "
                f"Minimum image dimensions required: {self.patch_size}x{self.patch_size}"
            )

        patches = []
        positions = []
        positions_set = set()  # Track unique positions to avoid duplicates

        def add_patch(row: int, col: int):
            """Add patch if position hasn't been added yet."""
            if (row, col) not in positions_set:
                patch = image[row:row + self.patch_size, col:col + self.patch_size]
                patches.append(patch)
                positions.append((row, col))
                positions_set.add((row, col))

        # Main grid (row-major order)
        for row in range(0, h - self.patch_size + 1, self.stride):
            for col in range(0, w - self.patch_size + 1, self.stride):
                add_patch(row, col)

        # Handle right edge - add patches aligned to right boundary
        right_col = w - self.patch_size
        if right_col > 0 and (w - self.patch_size) % self.stride != 0:
            for row in range(0, h - self.patch_size + 1, self.stride):
                add_patch(row, right_col)

        # Handle bottom edge - add patches aligned to bottom boundary
        bottom_row = h - self.patch_size
        if bottom_row > 0 and (h - self.patch_size) % self.stride != 0:
            for col in range(0, w - self.patch_size + 1, self.stride):
                add_patch(bottom_row, col)

        # Handle bottom-right corner
        if bottom_row > 0 and right_col > 0:
            add_patch(bottom_row, right_col)

        return np.array(patches), positions

    def stitch_predictions(
        self,
        predictions: np.ndarray,
        positions: list[tuple[int, int]],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Stitch patch predictions back into full image.

        Uses averaging for overlapping regions.

        Args:
            predictions: Patch predictions (N, num_classes, H, W)
            positions: (row, col) position of each patch
            output_shape: (H, W) of output image

        Returns:
            Stitched predictions (num_classes, H, W)
        """
        h, w = output_shape
        num_classes = predictions.shape[1]

        # Accumulator and count for averaging
        output = np.zeros((num_classes, h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        for pred, (row, col) in zip(predictions, positions):
            output[:, row:row + self.patch_size, col:col + self.patch_size] += pred
            counts[row:row + self.patch_size, col:col + self.patch_size] += 1

        # Average overlapping regions
        counts = np.maximum(counts, 1)  # Avoid division by zero
        output = output / counts[np.newaxis, :, :]

        return output

    @torch.no_grad()
    def predict_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Run inference on patches.

        Args:
            patches: (N, H, W, C) numpy array

        Returns:
            predictions: (N, num_classes, H, W) probabilities
        """
        all_predictions = []

        for i in range(0, len(patches), self.batch_size):
            batch = patches[i:i + self.batch_size]

            # Convert to tensor: (N, H, W, C) -> (N, C, H, W)
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
            batch_tensor = batch_tensor.to(self.device)

            # Run inference
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

            all_predictions.append(probs.cpu().numpy())

        return np.concatenate(all_predictions, axis=0)

    def predict_from_array(
        self,
        image: np.ndarray,
        select_bands: bool = True,
    ) -> InferenceResult:
        """
        Run inference on a numpy array.

        Args:
            image: Input image (H, W, 7) or (H, W, 12)
            select_bands: Whether to select bands from 12-band input

        Returns:
            InferenceResult with segmentation and metrics
        """
        original_shape = image.shape[:2]

        # Preprocess
        image = self.preprocess_image(image, select_bands=select_bands)

        # Pad image to be divisible by patch_size
        h, w = image.shape[:2]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )

        # Extract patches
        patches, positions = self.extract_patches(image)

        # Run inference
        predictions = self.predict_patches(patches)

        # Stitch predictions
        stitched = self.stitch_predictions(
            predictions, positions, image.shape[:2]
        )

        # Remove padding
        stitched = stitched[:, :original_shape[0], :original_shape[1]]

        # Convert to numpy results
        probabilities = stitched.transpose(1, 2, 0)  # (H, W, C)
        segmentation = stitched.argmax(axis=0).astype(np.uint8)  # (H, W)

        # Compute metrics
        has_fire = (segmentation > 0).any()
        fire_probs = probabilities[:, :, 1:].sum(axis=2)
        fire_confidence = float(fire_probs.max())
        fire_fraction = float((segmentation > 0).mean())

        # Severity counts
        severity_counts = {}
        try:
            class_names = get_class_names(self.num_classes)
        except ValueError:
            class_names = tuple(f"class_{i}" for i in range(self.num_classes))
        for i, name in enumerate(class_names):
            severity_counts[name] = int((segmentation == i).sum())

        return InferenceResult(
            segmentation=segmentation,
            probabilities=probabilities,
            has_fire=has_fire,
            fire_confidence=fire_confidence,
            fire_fraction=fire_fraction,
            severity_counts=severity_counts,
            num_classes=self.num_classes,
            image_shape=original_shape,
        )

    def predict_from_file(
        self,
        file_path: Union[str, Path],
        select_bands: bool = True,
    ) -> InferenceResult:
        """
        Run inference on a GeoTIFF file.

        Args:
            file_path: Path to GeoTIFF file
            select_bands: Whether to select bands from 12-band input

        Returns:
            InferenceResult with segmentation and metrics
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for reading GeoTIFF files")

        with rasterio.open(file_path) as src:
            # Read all bands: (C, H, W) -> (H, W, C)
            image = src.read().transpose(1, 2, 0)

        return self.predict_from_array(image, select_bands=select_bands)


def create_visualization(
    image: np.ndarray,
    result: InferenceResult,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create visualization overlay of predictions on image.

    Args:
        image: Original image (H, W, 7) - uses bands 2,1,0 for RGB
        result: Inference result
        alpha: Overlay transparency

    Returns:
        RGB visualization (H, W, 3) uint8
    """
    # Create RGB from bands (Red=band2, Green=band1, Blue=band0)
    if image.shape[-1] >= 3:
        rgb = image[:, :, [2, 1, 0]]
    else:
        rgb = np.stack([image[:, :, 0]] * 3, axis=-1)

    # Normalize for display
    rgb = np.clip(rgb * 3.0, 0, 1)  # Boost brightness
    rgb = (rgb * 255).astype(np.uint8)

    # Create severity colormap
    colors = {
        0: [0, 0, 0, 0],        # No damage - transparent
        1: [181, 254, 142, 200],  # Negligible - light green
        2: [254, 217, 142, 200],  # Moderate - yellow
        3: [254, 153, 41, 200],   # High - orange
        4: [204, 76, 2, 200],     # Destroyed - dark red
    }

    # For binary (2 classes), use red for fire
    if result.num_classes == 2:
        colors = {
            0: [0, 0, 0, 0],
            1: [255, 0, 0, 200],  # Fire - red
        }

    # Create overlay
    overlay = np.zeros((result.segmentation.shape[0], result.segmentation.shape[1], 4), dtype=np.uint8)
    for class_idx, color in colors.items():
        mask = result.segmentation == class_idx
        overlay[mask] = color

    # Blend
    rgb_float = rgb.astype(np.float32) / 255.0
    overlay_rgb = overlay[:, :, :3].astype(np.float32) / 255.0
    overlay_alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0 * alpha

    blended = rgb_float * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
    blended = (blended * 255).astype(np.uint8)

    return blended


def create_fire_mask_visualization(
    segmentation: np.ndarray,
    num_classes: int = 2,
    dilate_pixels: int = 3,
) -> np.ndarray:
    """
    Create a high-contrast fire mask for easy visibility of sparse detections.

    Fire pixels: bright red. Background: dark grey. Optionally dilates fire
    pixels so single-pixel detections are visible.

    Returns:
        RGB image (H, W, 3) uint8
    """
    fire_mask = segmentation > 0
    if dilate_pixels > 0 and fire_mask.any():
        from scipy.ndimage import binary_dilation
        fire_mask = binary_dilation(fire_mask, iterations=dilate_pixels)

    out = np.full((*segmentation.shape, 3), 40, dtype=np.uint8)  # Dark grey bg
    out[fire_mask] = [255, 50, 50]  # Bright red for fire
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print(f"Loading model from: {model_path}")
    pipeline = FireInferencePipeline(model_path)

    print(f"Running inference on: {image_path}")
    result = pipeline.predict_from_file(image_path)

    print("\nResults:")
    print(f"  Has fire: {result.has_fire}")
    print(f"  Fire confidence: {result.fire_confidence:.2%}")
    print(f"  Fire fraction: {result.fire_fraction:.2%}")
    print(f"  Severity counts: {result.severity_counts}")
