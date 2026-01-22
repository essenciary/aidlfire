#!/usr/bin/env python3
"""
Download the CEMS Wildfire Dataset from HuggingFace.

This script automates downloading and extracting the dataset for cloud deployment.
Uses huggingface_hub for reliable downloads with progress tracking and resume support.

Usage:
    # Download and extract everything
    uv run python download_dataset.py

    # Download to specific location
    uv run python download_dataset.py --output-dir /data/wildfires-cems

    # Download only specific splits
    uv run python download_dataset.py --splits train val

    # Skip extraction (download archives only)
    uv run python download_dataset.py --no-extract

    # Full pipeline: download, extract, and generate patches
    uv run python download_dataset.py --generate-patches --patches-dir ./patches
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        missing.append("huggingface_hub")

    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        print(f"Or: uv add {' '.join(missing)}")
        sys.exit(1)


def download_dataset(
    output_dir: Path,
    splits: list[str] | None = None,
) -> Path:
    """
    Download the CEMS Wildfire Dataset from HuggingFace.

    Args:
        output_dir: Directory to download to
        splits: Which splits to download (None = all)

    Returns:
        Path to the downloaded dataset
    """
    from huggingface_hub import snapshot_download

    repo_id = "links-ads/wildfires-cems"

    print(f"\n📥 Downloading CEMS Wildfire Dataset")
    print(f"   Repository: {repo_id}")
    print(f"   Output: {output_dir}")

    # Build allow_patterns based on splits
    allow_patterns = None
    if splits:
        # Always include csv_files and root files
        allow_patterns = [
            "csv_files/*",
            "*.md",
            "*.txt",
        ]
        for split in splits:
            allow_patterns.append(f"data/{split}/*")
        print(f"   Splits: {', '.join(splits)}")
    else:
        print(f"   Splits: all (train, val, test)")

    # Download using huggingface_hub
    # This handles:
    # - Progress bars
    # - Resume interrupted downloads
    # - Parallel downloads
    # - Git LFS files
    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
    )

    print(f"\n✅ Download complete: {dataset_path}")
    return Path(dataset_path)


def extract_archives(dataset_dir: Path, splits: list[str] | None = None):
    """
    Extract the tar archives for each split.

    The dataset comes as split tar archives that need to be concatenated and extracted.

    Args:
        dataset_dir: Path to downloaded dataset
        splits: Which splits to extract (None = all)
    """
    if splits is None:
        splits = ["train", "val", "test"]

    data_dir = dataset_dir / "data"

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}")
            continue

        # Check for tar.gz.part files
        part_files = sorted(split_dir.glob(f"{split}.tar.*.gz.part"))
        if not part_files:
            # Check if already extracted
            extracted_dirs = list(split_dir.glob("EMSR*"))
            if extracted_dirs:
                print(f"✓ {split}: Already extracted ({len(extracted_dirs)} EMSR folders)")
                continue
            else:
                print(f"⚠️  {split}: No archives or extracted data found")
                continue

        print(f"\n📦 Extracting {split} ({len(part_files)} parts)...")

        # Concatenate parts and extract
        # Command: cat train.tar.*.gz.part | tar -xzvf - -i -C train/
        cmd = f"cat {split}.tar.*.gz.part | tar -xzf - -i"

        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=split_dir,
                check=True,
            )
            print(f"✅ {split}: Extraction complete")

            # Count extracted folders
            extracted_dirs = list(split_dir.glob("EMSR*"))
            if extracted_dirs:
                # Move from nested structure if needed
                # Some archives extract to split/split/EMSR...
                nested_split = split_dir / split
                if nested_split.exists():
                    print(f"   Moving from nested {split}/{split}/ structure...")
                    for emsr_dir in nested_split.iterdir():
                        if emsr_dir.is_dir() and emsr_dir.name.startswith("EMSR"):
                            target = split_dir / emsr_dir.name
                            if not target.exists():
                                emsr_dir.rename(target)

                extracted_dirs = list(split_dir.glob("EMSR*"))
                print(f"   Found {len(extracted_dirs)} EMSR folders")

        except subprocess.CalledProcessError as e:
            print(f"❌ {split}: Extraction failed: {e}")
            continue


def generate_patches(
    dataset_dir: Path,
    patches_dir: Path,
    mask_type: str = "DEL",
    splits: list[str] | None = None,
):
    """
    Generate patches from the extracted dataset.

    Args:
        dataset_dir: Path to extracted dataset
        patches_dir: Output directory for patches
        mask_type: DEL or GRA
        splits: Which splits to process
    """
    print(f"\n🔧 Generating patches...")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Output: {patches_dir}")
    print(f"   Mask type: {mask_type}")

    # Import here to avoid dependency issues
    try:
        from patch_generator import PatchGenerator, PatchConfig
    except ImportError:
        print("❌ Could not import patch_generator. Make sure you're in the fire-pipeline directory.")
        return

    config = PatchConfig()
    generator = PatchGenerator(config)

    data_dir = dataset_dir / "data"

    generator.process_dataset(
        dataset_root=data_dir,
        output_root=patches_dir,
        mask_type=mask_type,
        splits=splits or ["train", "val", "test"],
    )

    print(f"\n✅ Patches generated at: {patches_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CEMS Wildfire Dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract everything
  uv run python download_dataset.py

  # Download to specific location
  uv run python download_dataset.py --output-dir /data/wildfires-cems

  # Download, extract, and generate patches in one go
  uv run python download_dataset.py --generate-patches --patches-dir ./patches

  # Download only training data
  uv run python download_dataset.py --splits train
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "wildfires-cems",
        help="Directory to download dataset to (default: ../wildfires-cems)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Which splits to download (default: all)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction of tar archives",
    )
    parser.add_argument(
        "--generate-patches",
        action="store_true",
        help="Generate patches after downloading and extracting",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=Path("./patches"),
        help="Output directory for patches (default: ./patches)",
    )
    parser.add_argument(
        "--mask-type",
        choices=["DEL", "GRA"],
        default="DEL",
        help="Mask type for patch generation (default: DEL)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing data)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  CEMS WILDFIRE DATASET DOWNLOADER")
    print("=" * 60)

    # Check dependencies
    if not args.skip_download:
        check_dependencies()

    # Step 1: Download
    if not args.skip_download:
        dataset_path = download_dataset(
            output_dir=args.output_dir,
            splits=args.splits,
        )
    else:
        dataset_path = args.output_dir
        print(f"\n⏭️  Skipping download, using: {dataset_path}")

    # Step 2: Extract
    if not args.no_extract:
        extract_archives(dataset_path, splits=args.splits)
    else:
        print(f"\n⏭️  Skipping extraction")

    # Step 3: Generate patches (optional)
    if args.generate_patches:
        generate_patches(
            dataset_dir=dataset_path,
            patches_dir=args.patches_dir,
            mask_type=args.mask_type,
            splits=args.splits,
        )

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)

    if args.generate_patches:
        print(f"\n  Patches ready at: {args.patches_dir}")
        print(f"\n  Next steps:")
        print(f"    1. Analyze class distribution:")
        print(f"       uv run python analyze_patches.py {args.patches_dir}")
        print(f"    2. Start training with your model")
    else:
        print(f"\n  Dataset ready at: {dataset_path}")
        print(f"\n  Next steps:")
        print(f"    1. Generate patches:")
        print(f"       uv run python run_pipeline.py --skip-extraction")
        print(f"    2. Or run this script with --generate-patches")


if __name__ == "__main__":
    main()
