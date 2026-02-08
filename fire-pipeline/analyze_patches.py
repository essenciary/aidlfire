#!/usr/bin/env python3
"""
Analyze patch statistics to understand class imbalance.

Run this after generating patches to determine the right class imbalance strategy.

Usage:
    uv run python analyze_patches.py ./patches
    uv run python analyze_patches.py ./patches --split train
    uv run python analyze_patches.py ./patches --num-classes 5  # For GRA masks
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def analyze_split(split_dir: Path, num_classes: int = 2) -> dict:
    """Analyze a single split directory."""

    mask_files = sorted(split_dir.glob("*_mask.npy"))

    if not mask_files:
        return None

    # Collect statistics
    fire_fractions = []
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for mask_file in mask_files:
        mask = np.load(mask_file)
        total_pixels += mask.size

        # Fire fraction for this patch (any class > 0 is "fire")
        fire_fraction = (mask > 0).sum() / mask.size
        fire_fractions.append(fire_fraction)

        # Per-class counts
        for c in range(num_classes):
            class_pixel_counts[c] += (mask == c).sum()

    fire_fractions = np.array(fire_fractions)

    return {
        "n_patches": len(mask_files),
        "total_pixels": total_pixels,
        "fire_fractions": fire_fractions,
        "class_pixel_counts": class_pixel_counts,
    }


def print_analysis(stats: dict, split_name: str, num_classes: int = 2):
    """Print detailed analysis for a split."""

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SPLIT ANALYSIS")
    print(f"{'='*60}")

    n_patches = stats["n_patches"]
    fire_fractions = stats["fire_fractions"]
    class_counts = stats["class_pixel_counts"]
    total_pixels = stats["total_pixels"]

    # Patch-level statistics
    print(f"\n📦 PATCH-LEVEL STATISTICS ({n_patches:,} patches)")
    print("-" * 40)

    thresholds = [0, 0.01, 0.05, 0.10, 0.25, 0.50]
    for i, thresh in enumerate(thresholds):
        if i < len(thresholds) - 1:
            next_thresh = thresholds[i + 1]
            count = ((fire_fractions > thresh) & (fire_fractions <= next_thresh)).sum()
            pct = count / n_patches * 100
            print(f"  Fire {thresh*100:5.1f}% - {next_thresh*100:5.1f}%: {count:6,} patches ({pct:5.1f}%)")

    count = (fire_fractions > 0.50).sum()
    pct = count / n_patches * 100
    print(f"  Fire  > 50.0%: {count:6,} patches ({pct:5.1f}%)")

    print()
    no_fire = (fire_fractions == 0).sum()
    has_fire = (fire_fractions > 0).sum()
    has_fire_1pct = (fire_fractions > 0.01).sum()
    has_fire_5pct = (fire_fractions > 0.05).sum()

    print(f"  🔥 Patches with ANY fire:  {has_fire:6,} ({has_fire/n_patches*100:5.1f}%)")
    print(f"  🔥 Patches with >1% fire:  {has_fire_1pct:6,} ({has_fire_1pct/n_patches*100:5.1f}%)")
    print(f"  🔥 Patches with >5% fire:  {has_fire_5pct:6,} ({has_fire_5pct/n_patches*100:5.1f}%)")
    print(f"  ⬜ Patches with NO fire:   {no_fire:6,} ({no_fire/n_patches*100:5.1f}%)")

    print()
    print(f"  Mean fire fraction:   {fire_fractions.mean()*100:6.2f}%")
    print(f"  Median fire fraction: {np.median(fire_fractions)*100:6.2f}%")
    print(f"  Max fire fraction:    {fire_fractions.max()*100:6.2f}%")

    # Pixel-level statistics
    print(f"\n🔢 PIXEL-LEVEL STATISTICS ({total_pixels:,} pixels)")
    print("-" * 40)

    class_names = ["Background", "Fire"] if num_classes == 2 else [f"Class {i}" for i in range(num_classes)]
    if num_classes == 5:
        class_names = ["No damage", "Negligible", "Moderate", "High", "Destroyed"]

    for c in range(num_classes):
        count = class_counts[c]
        pct = count / total_pixels * 100
        print(f"  {class_names[c]:12}: {count:15,} pixels ({pct:6.2f}%)")

    # Class imbalance ratio
    if num_classes == 2:
        bg_count = class_counts[0]
        fire_count = class_counts[1]
        if fire_count > 0:
            ratio = bg_count / fire_count
            print(f"\n  ⚖️  Imbalance ratio: {ratio:.1f}:1 (background:fire)")
        else:
            print(f"\n  ⚠️  No fire pixels found!")

    # Recommended class weights
    print(f"\n⚖️  RECOMMENDED CLASS WEIGHTS")
    print("-" * 40)

    # Inverse frequency weighting
    weights = total_pixels / (num_classes * class_counts + 1e-6)
    weights = weights / weights.min()  # Normalize so min weight is 1.0

    print("  Inverse frequency weights:")
    for c in range(num_classes):
        print(f"    {class_names[c]:12}: {weights[c]:.2f}")

    print(f"\n  Use in PyTorch:")
    weights_str = ", ".join([f"{w:.2f}" for w in weights])
    print(f"    weights = torch.tensor([{weights_str}])")
    print(f"    criterion = nn.CrossEntropyLoss(weight=weights)")

    return {
        "no_fire_patches": no_fire,
        "has_fire_patches": has_fire,
        "mean_fire_fraction": fire_fractions.mean(),
        "imbalance_ratio": class_counts[0] / (class_counts[1] + 1e-6) if num_classes == 2 else None,
        "class_weights": weights,
    }


def print_recommendations(analysis_results: dict):
    """Print recommendations based on analysis."""

    print(f"\n{'='*60}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*60}")

    # Get train stats if available
    train = analysis_results.get("train", {})

    if not train:
        print("\n  ⚠️  No training data analyzed. Run with --split train")
        return

    patch_fire_rate = train.get("has_fire_patches", 0) / max(train.get("has_fire_patches", 0) + train.get("no_fire_patches", 1), 1)
    mean_fire_frac = train.get("mean_fire_fraction", 0)
    imbalance = train.get("imbalance_ratio", 1)

    print(f"\n  Based on your data:")
    print(f"  • {patch_fire_rate*100:.1f}% of patches contain fire")
    print(f"  • Mean fire coverage per patch: {mean_fire_frac*100:.2f}%")
    print(f"  • Pixel imbalance ratio: {imbalance:.1f}:1")

    print(f"\n  📋 SUGGESTED STRATEGY:")
    print("-" * 40)

    # Patch-level recommendations
    if patch_fire_rate < 0.3:
        print(f"  ✓ Use WEIGHTED SAMPLING (fire patches are minority)")
        print(f"    dm = WildfireDataModule(")
        print(f"        './patches',")
        print(f"        use_weighted_sampling=True,")
        print(f"        fire_sample_weight={min(10.0, 1/patch_fire_rate):.1f},")
        print(f"    )")
    elif patch_fire_rate < 0.5:
        print(f"  ✓ MODERATE weighted sampling recommended")
        print(f"    fire_sample_weight=3.0 to 5.0")
    else:
        print(f"  ✓ Patch distribution is balanced - weighted sampling optional")

    # Pixel-level recommendations
    print()
    if imbalance > 20:
        print(f"  ✓ Use HEAVY LOSS WEIGHTING (severe pixel imbalance)")
        print(f"    Consider Focal Loss or Dice Loss")
    elif imbalance > 10:
        print(f"  ✓ Use MODERATE LOSS WEIGHTING")
        weights = train.get("class_weights", [1.0, 10.0])
        print(f"    weights = torch.tensor([{weights[0]:.1f}, {weights[1]:.1f}])")
    elif imbalance > 5:
        print(f"  ✓ Use LIGHT LOSS WEIGHTING")
        print(f"    weights = torch.tensor([1.0, {imbalance:.1f}])")
    else:
        print(f"  ✓ Pixel distribution is relatively balanced")
        print(f"    Standard CrossEntropyLoss should work")

    # Augmentation recommendations
    print()
    if patch_fire_rate < 0.4:
        print(f"  ✓ Use FIRE-AWARE AUGMENTATION (stronger for fire patches)")
        print(f"    fire_augment=get_strong_augmentation()")
    else:
        print(f"  ✓ Standard augmentation should be sufficient")

    print()


def plot_distribution(all_stats: dict, output_path: Path = None):
    """Create visualization of class distribution."""

    if not MATPLOTLIB_AVAILABLE:
        print("\n  ⚠️  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    splits = [s for s in ["train", "val", "test"] if s in all_stats]
    colors = {"train": "steelblue", "val": "orange", "test": "green"}

    # Plot 1: Fire fraction distribution
    ax1 = axes[0]
    for split in splits:
        fracs = all_stats[split]["fire_fractions"]
        ax1.hist(fracs, bins=50, alpha=0.6, label=split, color=colors[split])
    ax1.set_xlabel("Fire Fraction per Patch")
    ax1.set_ylabel("Number of Patches")
    ax1.set_title("Fire Fraction Distribution")
    ax1.legend()
    ax1.set_xlim(0, 1)

    # Plot 2: Patches with/without fire
    ax2 = axes[1]
    x = np.arange(len(splits))
    width = 0.35

    has_fire = [all_stats[s]["has_fire_patches"] for s in splits]
    no_fire = [all_stats[s]["no_fire_patches"] for s in splits]

    ax2.bar(x - width/2, has_fire, width, label="Has Fire", color="orangered")
    ax2.bar(x + width/2, no_fire, width, label="No Fire", color="steelblue")
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.set_ylabel("Number of Patches")
    ax2.set_title("Patches With/Without Fire")
    ax2.legend()

    # Plot 3: Pixel distribution (train only)
    ax3 = axes[2]
    if "train" in all_stats:
        counts = all_stats["train"]["class_pixel_counts"]
        labels = ["Background", "Fire"] if len(counts) == 2 else [f"Class {i}" for i in range(len(counts))]
        colors_pie = ["steelblue", "orangered"] if len(counts) == 2 else plt.cm.Set3(np.linspace(0, 1, len(counts)))

        # Only show non-zero classes in pie
        nonzero = counts > 0
        ax3.pie(counts[nonzero], labels=np.array(labels)[nonzero], autopct='%1.1f%%', colors=np.array(colors_pie)[nonzero])
        ax3.set_title("Pixel Distribution (Train)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n  📊 Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze patch statistics for class imbalance"
    )
    parser.add_argument(
        "patches_dir",
        type=Path,
        help="Root patches directory (containing train/val/test subdirs)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Analyze only this split (train, val, test)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (2 for DEL, 5 for GRA)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Save plot to file instead of displaying",
    )

    args = parser.parse_args()

    patches_dir = args.patches_dir

    print(f"\n🔍 PATCH ANALYSIS")
    print(f"   Directory: {patches_dir}")
    print(f"   Classes: {args.num_classes}")

    # Determine splits to analyze
    if args.split:
        splits = [args.split]
    else:
        splits = ["train", "val", "test"]

    all_stats = {}
    analysis_results = {}

    for split in splits:
        split_dir = patches_dir / split
        if not split_dir.exists():
            print(f"\n  ⚠️  Split '{split}' not found at {split_dir}")
            continue

        stats = analyze_split(split_dir, args.num_classes)
        if stats:
            all_stats[split] = stats
            result = print_analysis(stats, split, args.num_classes)
            analysis_results[split] = result

    if not all_stats:
        print("\n  ❌ No patches found!")
        return

    # Print recommendations
    print_recommendations(analysis_results)

    # Generate plots if requested
    if args.plot or args.output_plot:
        # Add analysis results to stats for plotting
        for split in all_stats:
            all_stats[split].update(analysis_results.get(split, {}))
        plot_distribution(all_stats, args.output_plot)


if __name__ == "__main__":
    main()
