#!/usr/bin/env python3
"""
Remove Catalan fire data from the CEMS Wildfire Dataset.

Identifies fires located in Catalonia based on geographic coordinates
and moves them from train/val/test splits to a separate 'cat' folder.
This allows using Catalan data for testing while training on non-Catalan data.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


# Catalonia approximate bounding box
CATALONIA_BOUNDS = {
    "lon_min": 0.15,
    "lon_max": 3.35,
    "lat_min": 40.5,
    "lat_max": 42.9,
}


def find_catalan_emsrs(csv_path: Path) -> set[str]:
    """Identify EMSR codes for fires located in Catalonia.

    Args:
        csv_path: Path to satelliteData.csv

    Returns:
        Set of EMSR codes (e.g., {'EMSR123', 'EMSR456'})
    """
    df = pd.read_csv(csv_path)

    # Filter by Spain AND coordinates within Catalonia
    catalonia = df[
        (df["country"] == "Spain")
        & (df["centerBoxLong"] >= CATALONIA_BOUNDS["lon_min"])
        & (df["centerBoxLong"] <= CATALONIA_BOUNDS["lon_max"])
        & (df["centerBoxLat"] >= CATALONIA_BOUNDS["lat_min"])
        & (df["centerBoxLat"] <= CATALONIA_BOUNDS["lat_max"])
    ]

    return set(catalonia["EMSR"].unique())


def move_emsr_folders(
    data_root: Path,
    emsr_codes: set[str],
    dry_run: bool = True,
) -> dict[str, list[str]]:
    """Move EMSR folders from train/val/test splits to cat/ folder.

    Args:
        data_root: Path to wildfires-cems/data/
        emsr_codes: Set of EMSR codes to move
        dry_run: If True, only print what would be moved

    Returns:
        Dictionary mapping split names to list of moved folders
    """
    moved = {}

    # Create cat folder structure (mirroring train/val/test)
    cat_root = data_root / "cat"

    # Handle nested structure: train/train/, val/val/, test/test/
    for split in ["train", "val", "test"]:
        split_dir = data_root / split / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            continue

        # Destination: cat/{split}/
        cat_split_dir = cat_root / split
        moved[split] = []

        for emsr_dir in split_dir.iterdir():
            if not emsr_dir.is_dir():
                continue

            emsr_code = emsr_dir.name
            if emsr_code in emsr_codes:
                dest_dir = cat_split_dir / emsr_code
                moved[split].append(emsr_code)

                if dry_run:
                    print(f"[DRY RUN] Would move: {emsr_dir}")
                    print(f"                  to: {dest_dir}")
                else:
                    print(f"Moving: {emsr_dir}")
                    print(f"    to: {dest_dir}")
                    cat_split_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(emsr_dir), str(dest_dir))

    return moved


def main():
    parser = argparse.ArgumentParser(
        description="Move Catalan fire data to separate folder for region-specific testing"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).parent.parent / "wildfires-cems" / "data",
        help="Path to wildfires-cems/data/ directory",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(__file__).parent.parent / "csv_files" / "satelliteData.csv",
        help="Path to satelliteData.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only print what would be moved (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the folders (overrides --dry-run)",
    )

    args = parser.parse_args()

    dry_run = not args.execute

    print("=" * 60)
    print("MOVE CATALAN DATA TO SEPARATE FOLDER")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"CSV path: {args.csv_path}")
    print(f"Destination: {args.dataset_root / 'cat'}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()

    # Find Catalan EMSR codes
    print("Finding Catalan fire events...")
    catalan_emsrs = find_catalan_emsrs(args.csv_path)
    print(f"Found {len(catalan_emsrs)} Catalan EMSR codes:")
    for emsr in sorted(catalan_emsrs):
        print(f"  - {emsr}")
    print()

    # Move folders
    print("Scanning dataset directories...")
    moved = move_emsr_folders(args.dataset_root, catalan_emsrs, dry_run=dry_run)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for split, folders in moved.items():
        action = "would be moved" if dry_run else "moved"
        print(f"  {split}: {len(folders)} EMSR folders {action}")
        total += len(folders)
    print(f"  Total: {total} EMSR folders")

    if dry_run:
        print()
        print("This was a dry run. To actually move the folders, run with --execute")


if __name__ == "__main__":
    main()
