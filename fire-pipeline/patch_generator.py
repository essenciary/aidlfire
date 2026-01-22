"""
Patch Generation Pipeline for CEMS Wildfire Dataset

Extracts 256x256 patches from Sentinel-2 images with corresponding masks
for training fire detection/severity models.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import rasterio

from constants import (
    BAND_INDICES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE_TRAIN,
    DEFAULT_STRIDE_INFERENCE,
    DEFAULT_MAX_CLOUD_COVER,
)


@dataclass
class PatchConfig:
    """Configuration for patch extraction."""
    patch_size: int = DEFAULT_PATCH_SIZE
    stride_train: int = DEFAULT_STRIDE_TRAIN  # 50% overlap for training
    stride_inference: int = DEFAULT_STRIDE_INFERENCE  # No overlap for inference
    max_cloud_cover: float = DEFAULT_MAX_CLOUD_COVER  # Reject patches with >50% cloud

    # Band indices for 7-band selection: B02, B03, B04, B08, B8A, B11, B12
    band_indices: tuple = BAND_INDICES

    # Value range for clipping (data is already ~normalized)
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class PatchMetadata:
    """Metadata for a single patch."""
    patch_id: str
    source_image: str
    row: int
    col: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    cloud_cover: float
    burn_fraction: float  # Fraction of burned pixels in mask

    def __post_init__(self):
        """Validate metadata values."""
        if self.row < 0:
            raise ValueError(f"row must be non-negative, got {self.row}")
        if self.col < 0:
            raise ValueError(f"col must be non-negative, got {self.col}")
        if not 0 <= self.cloud_cover <= 1:
            raise ValueError(f"cloud_cover must be in [0, 1], got {self.cloud_cover}")
        if not 0 <= self.burn_fraction <= 1:
            raise ValueError(f"burn_fraction must be in [0, 1], got {self.burn_fraction}")


class PatchGenerator:
    """Generates patches from CEMS wildfire dataset."""

    def __init__(self, config: PatchConfig = None):
        self.config = config or PatchConfig()

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess satellite image.

        Returns:
            Array of shape (H, W, 7) with selected bands, clipped to [0, 1]
        """
        with rasterio.open(image_path) as src:
            # Read all bands and select the ones we need
            data = src.read()  # (12, H, W)

            # Select bands and transpose to (H, W, C)
            selected = data[list(self.config.band_indices), :, :]  # (7, H, W)
            selected = np.transpose(selected, (1, 2, 0))  # (H, W, 7)

            # Clip to valid range
            selected = np.clip(selected, self.config.clip_min, self.config.clip_max)

            return selected.astype(np.float32)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask file (DEL, GRA, or CM).

        Returns:
            Array of shape (H, W) with uint8 values
        """
        with rasterio.open(mask_path) as src:
            data = src.read(1)  # Read first band
            return data.astype(np.uint8)

    def _get_transform(self, image_path: Path):
        """Get the geotransform from an image."""
        with rasterio.open(image_path) as src:
            return src.transform, src.crs

    def _calculate_cloud_cover(self, cloud_patch: np.ndarray) -> float:
        """Calculate fraction of cloudy pixels.

        Cloud mask values: 0=clear, 1=cloud, 2=light cloud, 3=shadow
        We consider anything >= 1 as "not clear"
        """
        not_clear = np.sum(cloud_patch >= 1)
        total = cloud_patch.size
        return not_clear / total if total > 0 else 0.0

    def _calculate_burn_fraction(self, mask_patch: np.ndarray) -> float:
        """Calculate fraction of burned pixels in DEL mask."""
        burned = np.sum(mask_patch > 0)
        total = mask_patch.size
        return burned / total if total > 0 else 0.0

    def _extract_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cloud_mask: np.ndarray,
        stride: int,
        source_name: str,
        transform,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, PatchMetadata]]:
        """Extract patches using sliding window.

        Yields:
            Tuples of (image_patch, mask_patch, metadata)
        """
        h, w = image.shape[:2]
        patch_size = self.config.patch_size
        patch_idx = 0

        for row in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                # Extract patches
                img_patch = image[row:row+patch_size, col:col+patch_size, :]
                mask_patch = mask[row:row+patch_size, col:col+patch_size]
                cloud_patch = cloud_mask[row:row+patch_size, col:col+patch_size]

                # Calculate cloud cover
                cloud_cover = self._calculate_cloud_cover(cloud_patch)

                # Skip if too cloudy
                if cloud_cover > self.config.max_cloud_cover:
                    continue

                # Calculate geographic bounds
                x_min, y_max = transform * (col, row)
                x_max, y_min = transform * (col + patch_size, row + patch_size)

                # Create metadata
                metadata = PatchMetadata(
                    patch_id=f"{source_name}_r{row}_c{col}",
                    source_image=source_name,
                    row=row,
                    col=col,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    cloud_cover=cloud_cover,
                    burn_fraction=self._calculate_burn_fraction(mask_patch),
                )

                patch_idx += 1
                yield img_patch, mask_patch, metadata

    def process_image(
        self,
        image_dir: Path,
        output_dir: Path,
        mask_type: Literal["DEL", "GRA"] = "DEL",
        mode: Literal["train", "inference"] = "train",
        skip_existing: bool = True,
    ) -> list[PatchMetadata]:
        """Process a single image directory and extract patches.

        Args:
            image_dir: Directory containing S2L2A.tif, DEL.tif, CM.tif, etc.
            output_dir: Directory to save patches
            mask_type: "DEL" for binary or "GRA" for severity
            mode: "train" (50% overlap) or "inference" (no overlap)
            skip_existing: Skip patches that already exist (for resume support)

        Returns:
            List of metadata for extracted patches
        """
        # Find files
        base_name = image_dir.name
        image_file = image_dir / f"{base_name}_S2L2A.tif"

        # Handle both .tif and .tiff extensions
        if not image_file.exists():
            image_file = image_dir / f"{base_name}_S2L2A.tiff"

        mask_file = image_dir / f"{base_name}_{mask_type}.tif"
        cloud_file = image_dir / f"{base_name}_CM.tif"

        # Validate files exist
        for f, name in [(image_file, "image"), (mask_file, "mask"), (cloud_file, "cloud")]:
            if not f.exists():
                print(f"Warning: Missing {name} file: {f}")
                return []

        # Load data
        image = self._load_image(image_file)
        mask = self._load_mask(mask_file)
        cloud_mask = self._load_mask(cloud_file)
        transform, crs = self._get_transform(image_file)

        # Determine stride
        stride = self.config.stride_train if mode == "train" else self.config.stride_inference

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract and save patches
        metadata_list = []
        skipped_count = 0
        for img_patch, mask_patch, metadata in self._extract_patches(
            image, mask, cloud_mask, stride, base_name, transform
        ):
            image_path = output_dir / f"{metadata.patch_id}_image.npy"
            mask_path = output_dir / f"{metadata.patch_id}_mask.npy"

            # Skip if both files already exist (resume support)
            if skip_existing and image_path.exists() and mask_path.exists():
                skipped_count += 1
                metadata_list.append(metadata)
                continue

            # Save patches
            np.save(image_path, img_patch)
            np.save(mask_path, mask_patch)
            metadata_list.append(metadata)

        return metadata_list, skipped_count

    def process_dataset(
        self,
        dataset_root: Path,
        output_root: Path,
        mask_type: Literal["DEL", "GRA"] = "DEL",
        splits: list[str] = None,
        skip_existing: bool = True,
    ) -> dict[str, list[PatchMetadata]]:
        """Process entire dataset (train/val/test splits).

        Args:
            dataset_root: Root directory containing train/val/test folders
            output_root: Root directory for output patches
            mask_type: "DEL" for binary or "GRA" for severity
            splits: List of splits to process (default: ["train", "val", "test"])
            skip_existing: Skip patches that already exist (for resume support)

        Returns:
            Dictionary mapping split names to lists of metadata
        """
        splits = splits or ["train", "val", "test"]
        all_metadata = {}

        for split in splits:
            split_dir = dataset_root / split
            if not split_dir.exists():
                print(f"Warning: Split directory not found: {split_dir}")
                continue

            print(f"\nProcessing {split} split...")
            mode = "inference" if split == "test" else "train"
            output_split_dir = output_root / split
            split_metadata = []
            total_skipped = 0

            # Find all image directories (EMSR*/AOI*/EMSR*_AOI*_*)
            image_dirs = list(split_dir.glob("EMSR*/AOI*/EMSR*_AOI*_*"))

            for i, image_dir in enumerate(image_dirs):
                if not image_dir.is_dir():
                    continue

                print(f"  [{i+1}/{len(image_dirs)}] {image_dir.name}", end="")

                metadata, skipped = self.process_image(
                    image_dir, output_split_dir, mask_type, mode, skip_existing
                )
                split_metadata.extend(metadata)
                total_skipped += skipped

                if skipped > 0:
                    print(f" -> {len(metadata)} patches ({skipped} skipped)")
                else:
                    print(f" -> {len(metadata)} patches")

            all_metadata[split] = split_metadata

            # Save metadata CSV
            self._save_metadata_csv(split_metadata, output_split_dir / "metadata.csv")
            if total_skipped > 0:
                print(f"  Total {split} patches: {len(split_metadata)} ({total_skipped} already existed)")
            else:
                print(f"  Total {split} patches: {len(split_metadata)}")

        return all_metadata

    def _save_metadata_csv(self, metadata_list: list[PatchMetadata], output_path: Path):
        """Save metadata to CSV file."""
        if not metadata_list:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Header
            f.write("patch_id,source_image,row,col,x_min,y_min,x_max,y_max,cloud_cover,burn_fraction\n")

            # Data rows
            for m in metadata_list:
                f.write(f"{m.patch_id},{m.source_image},{m.row},{m.col},"
                       f"{m.x_min:.6f},{m.y_min:.6f},{m.x_max:.6f},{m.y_max:.6f},"
                       f"{m.cloud_cover:.4f},{m.burn_fraction:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate patches from CEMS Wildfire Dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the dataset (containing train/val/test)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for patches",
    )
    parser.add_argument(
        "--mask-type",
        choices=["DEL", "GRA"],
        default="DEL",
        help="Mask type: DEL (binary) or GRA (severity)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels",
    )
    parser.add_argument(
        "--stride-train",
        type=int,
        default=128,
        help="Stride for training patches (50%% overlap)",
    )
    parser.add_argument(
        "--stride-inference",
        type=int,
        default=256,
        help="Stride for inference patches (no overlap)",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=0.5,
        help="Maximum cloud cover fraction (0-1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all patches, even if they already exist",
    )

    args = parser.parse_args()

    # Create config
    config = PatchConfig(
        patch_size=args.patch_size,
        stride_train=args.stride_train,
        stride_inference=args.stride_inference,
        max_cloud_cover=args.max_cloud_cover,
    )

    # Process dataset
    generator = PatchGenerator(config)
    skip_existing = not args.force
    if skip_existing:
        print("Resume mode: Skipping patches that already exist")
    else:
        print("Force mode: Regenerating all patches")

    metadata = generator.process_dataset(
        args.dataset_root,
        args.output_root,
        args.mask_type,
        args.splits,
        skip_existing=skip_existing,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for split, meta_list in metadata.items():
        print(f"  {split}: {len(meta_list)} patches")
        total += len(meta_list)
    print(f"  Total: {total} patches")


if __name__ == "__main__":
    main()
