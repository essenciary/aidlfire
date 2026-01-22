"""
Visualization tools for CEMS Wildfire Dataset patches.

Provides functions for visualizing satellite imagery, masks, and patch statistics.
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from constants import (
    BAND_DESCRIPTIONS as BAND_NAMES,
    DEL_COLORS,
    DEL_LABELS,
    GRA_COLORS,
    GRA_LABELS,
    CLOUD_COLORS,
    CLOUD_LABELS,
    BAND_INDICES,
    get_mask_colors,
)


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: uv add matplotlib")


def normalize_for_display(image: np.ndarray, percentile: tuple = (2, 98)) -> np.ndarray:
    """Normalize image to 0-1 range using percentile clipping.

    Args:
        image: Input array
        percentile: (low, high) percentiles for clipping

    Returns:
        Normalized array in [0, 1] range
    """
    low, high = np.percentile(image, percentile)
    if high - low < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def create_rgb_composite(
    patch: np.ndarray,
    mode: Literal["true_color", "false_color", "swir"] = "true_color",
    brightness: float = 1.0,
) -> np.ndarray:
    """Create RGB composite from 7-band patch.

    Args:
        patch: Array of shape (256, 256, 7) or (7, 256, 256)
        mode: Composite type
            - "true_color": B04, B03, B02 (Red, Green, Blue)
            - "false_color": B08, B04, B03 (NIR, Red, Green) - vegetation appears red
            - "swir": B12, B08, B04 (SWIR2, NIR, Red) - fire scars appear cyan
        brightness: Brightness multiplier

    Returns:
        RGB array of shape (256, 256, 3) in [0, 1] range
    """
    # Handle channel-first format
    if patch.shape[0] == 7:
        patch = np.transpose(patch, (1, 2, 0))

    # Band mapping: 0=B02, 1=B03, 2=B04, 3=B08, 4=B8A, 5=B11, 6=B12
    if mode == "true_color":
        # RGB = Red, Green, Blue = B04, B03, B02
        rgb = patch[:, :, [2, 1, 0]]
    elif mode == "false_color":
        # RGB = NIR, Red, Green = B08, B04, B03
        rgb = patch[:, :, [3, 2, 1]]
    elif mode == "swir":
        # RGB = SWIR2, NIR, Red = B12, B08, B04
        rgb = patch[:, :, [6, 3, 2]]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize each channel
    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        rgb_norm[:, :, i] = normalize_for_display(rgb[:, :, i])

    # Apply brightness
    rgb_norm = np.clip(rgb_norm * brightness, 0, 1)

    return rgb_norm


def plot_patch(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    mask_type: Literal["DEL", "GRA", "CM"] = "DEL",
    title: str = "",
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot a single patch with optional mask overlay.

    Args:
        image: Patch array (256, 256, 7) or (7, 256, 256)
        mask: Optional mask array (256, 256)
        mask_type: Type of mask for colormap
        title: Plot title
        save_path: If provided, save figure to this path
        show: Whether to display the plot
    """
    _check_matplotlib()

    # Handle channel-first format
    if image.shape[0] == 7:
        image = np.transpose(image, (1, 2, 0))

    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # True color
    rgb_true = create_rgb_composite(image, "true_color")
    axes[0].imshow(rgb_true)
    axes[0].set_title("True Color (RGB)")
    axes[0].axis("off")

    # False color (vegetation)
    rgb_false = create_rgb_composite(image, "false_color")
    axes[1].imshow(rgb_false)
    axes[1].set_title("False Color (NIR-R-G)")
    axes[1].axis("off")

    # SWIR composite (fire scars)
    rgb_swir = create_rgb_composite(image, "swir")
    axes[2].imshow(rgb_swir)
    axes[2].set_title("SWIR Composite")
    axes[2].axis("off")

    # Mask
    if mask is not None:
        if mask_type == "DEL":
            colors = DEL_COLORS
            labels = DEL_LABELS
            n_classes = 2
        elif mask_type == "GRA":
            colors = GRA_COLORS
            labels = GRA_LABELS
            n_classes = 5
        else:  # CM
            colors = CLOUD_COLORS
            labels = CLOUD_LABELS
            n_classes = 4

        cmap = ListedColormap(colors[:n_classes])

        # Overlay mask on true color
        axes[3].imshow(rgb_true)
        mask_display = np.ma.masked_where(mask == 0, mask) if mask_type != "CM" else mask
        axes[3].imshow(mask_display, cmap=cmap, alpha=0.6, vmin=0, vmax=n_classes-1)
        axes[3].set_title(f"{mask_type} Mask Overlay")
        axes[3].axis("off")

        # Add legend
        patches = [mpatches.Patch(color=colors[i], label=labels[i])
                   for i in range(min(len(labels), n_classes))]
        axes[3].legend(handles=patches, loc="lower right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_bands(
    image: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot all 7 bands individually.

    Args:
        image: Patch array (256, 256, 7) or (7, 256, 256)
        title: Plot title
        save_path: If provided, save figure to this path
        show: Whether to display the plot
    """
    _check_matplotlib()

    # Handle channel-first format
    if image.shape[0] == 7:
        image = np.transpose(image, (1, 2, 0))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(7):
        band = image[:, :, i]
        im = axes[i].imshow(band, cmap="viridis")
        axes[i].set_title(f"{BAND_NAMES[i]}\n[{band.min():.3f}, {band.max():.3f}]")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide the 8th subplot
    axes[7].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_grid(
    patches_dir: Path,
    n_samples: int = 16,
    mode: Literal["true_color", "false_color", "swir"] = "true_color",
    with_mask: bool = True,
    mask_type: Literal["DEL", "GRA"] = "DEL",
    save_path: Path | None = None,
    show: bool = True,
    seed: int = 42,
):
    """Plot a grid of sample patches.

    Args:
        patches_dir: Directory containing patch .npy files
        n_samples: Number of samples to display
        mode: RGB composite mode
        with_mask: Whether to overlay masks
        mask_type: Type of mask
        save_path: If provided, save figure to this path
        show: Whether to display the plot
        seed: Random seed for reproducibility
    """
    _check_matplotlib()

    patches_dir = Path(patches_dir)
    image_files = sorted(patches_dir.glob("*_image.npy"))

    if not image_files:
        raise ValueError(f"No patches found in {patches_dir}")

    # Sample patches
    np.random.seed(seed)
    n_samples = min(n_samples, len(image_files))
    indices = np.random.choice(len(image_files), n_samples, replace=False)

    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color setup for mask
    if mask_type == "DEL":
        colors = DEL_COLORS
        n_classes = 2
    else:
        colors = GRA_COLORS
        n_classes = 5
    cmap = ListedColormap(colors[:n_classes])

    for idx, ax in enumerate(axes.flatten()):
        if idx >= n_samples:
            ax.axis("off")
            continue

        img_file = image_files[indices[idx]]
        patch_id = img_file.stem.replace("_image", "")

        # Load image
        image = np.load(img_file)
        rgb = create_rgb_composite(image, mode)

        ax.imshow(rgb)

        # Overlay mask
        if with_mask:
            mask_file = patches_dir / f"{patch_id}_mask.npy"
            if mask_file.exists():
                mask = np.load(mask_file)
                mask_display = np.ma.masked_where(mask == 0, mask)
                ax.imshow(mask_display, cmap=cmap, alpha=0.5, vmin=0, vmax=n_classes-1)

                # Show burn percentage
                burn_pct = 100 * np.sum(mask > 0) / mask.size
                ax.set_title(f"{burn_pct:.1f}% burned", fontsize=9)

        ax.axis("off")

    plt.suptitle(f"Sample Patches ({mode.replace('_', ' ').title()})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_statistics(
    patches_dir: Path,
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot statistics about the patch dataset.

    Args:
        patches_dir: Directory containing patch .npy files and metadata.csv
        save_path: If provided, save figure to this path
        show: Whether to display the plot
    """
    _check_matplotlib()

    patches_dir = Path(patches_dir)

    # Load metadata if available
    metadata_file = patches_dir / "metadata.csv"

    if metadata_file.exists():
        # Parse CSV manually (no pandas dependency)
        with open(metadata_file) as f:
            lines = f.readlines()

        headers = lines[0].strip().split(",")
        cloud_idx = headers.index("cloud_cover")
        burn_idx = headers.index("burn_fraction")

        cloud_cover = []
        burn_fraction = []

        for line in lines[1:]:
            parts = line.strip().split(",")
            cloud_cover.append(float(parts[cloud_idx]))
            burn_fraction.append(float(parts[burn_idx]))

        cloud_cover = np.array(cloud_cover)
        burn_fraction = np.array(burn_fraction)
    else:
        # Compute from files
        print("No metadata.csv found, computing statistics from files...")
        image_files = list(patches_dir.glob("*_image.npy"))

        burn_fraction = []
        for img_file in image_files:
            patch_id = img_file.stem.replace("_image", "")
            mask_file = patches_dir / f"{patch_id}_mask.npy"
            if mask_file.exists():
                mask = np.load(mask_file)
                burn_fraction.append(np.sum(mask > 0) / mask.size)

        burn_fraction = np.array(burn_fraction)
        cloud_cover = None

    # Create figure
    n_plots = 3 if cloud_cover is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Burn fraction histogram
    axes[0].hist(burn_fraction * 100, bins=50, color="orangered", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Burn Fraction (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Burn Distribution\nMean: {burn_fraction.mean()*100:.1f}%")
    axes[0].axvline(burn_fraction.mean() * 100, color="red", linestyle="--", label="Mean")

    # Burn fraction by category
    burn_categories = ["0%", "0-10%", "10-30%", "30-50%", "50-100%"]
    burn_counts = [
        np.sum(burn_fraction == 0),
        np.sum((burn_fraction > 0) & (burn_fraction <= 0.1)),
        np.sum((burn_fraction > 0.1) & (burn_fraction <= 0.3)),
        np.sum((burn_fraction > 0.3) & (burn_fraction <= 0.5)),
        np.sum(burn_fraction > 0.5),
    ]

    axes[1].bar(burn_categories, burn_counts, color="coral", edgecolor="black")
    axes[1].set_xlabel("Burn Fraction")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Patches by Burn Category")

    # Add count labels
    for i, count in enumerate(burn_counts):
        axes[1].text(i, count + 0.5, str(count), ha="center", fontsize=9)

    # Cloud cover histogram (if available)
    if cloud_cover is not None:
        axes[2].hist(cloud_cover * 100, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[2].set_xlabel("Cloud Cover (%)")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Cloud Distribution\nMean: {cloud_cover.mean()*100:.1f}%")
        axes[2].axvline(cloud_cover.mean() * 100, color="blue", linestyle="--", label="Mean")

    plt.suptitle(f"Dataset Statistics ({len(burn_fraction)} patches)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_source_image(
    image_path: Path,
    mask_path: Path | None = None,
    mask_type: Literal["DEL", "GRA", "CM"] = "DEL",
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot a full source GeoTIFF image (before patch extraction).

    Args:
        image_path: Path to S2L2A.tif file
        mask_path: Optional path to mask file
        mask_type: Type of mask
        save_path: If provided, save figure to this path
        show: Whether to display the plot
    """
    _check_matplotlib()

    import rasterio

    # Load image
    with rasterio.open(image_path) as src:
        data = src.read()  # (12, H, W)

    # Select 7 bands and transpose
    image = data[list(BAND_INDICES), :, :].transpose(1, 2, 0)  # (H, W, 7)

    # Load mask if provided
    mask = None
    if mask_path and mask_path.exists():
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

    # Create figure
    n_cols = 3 if mask is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # True color
    rgb_true = create_rgb_composite(image, "true_color", brightness=1.2)
    axes[0].imshow(rgb_true)
    axes[0].set_title("True Color")
    axes[0].axis("off")

    # False color
    rgb_false = create_rgb_composite(image, "false_color")
    axes[1].imshow(rgb_false)
    axes[1].set_title("False Color (Vegetation)")
    axes[1].axis("off")

    # SWIR
    rgb_swir = create_rgb_composite(image, "swir")
    axes[2].imshow(rgb_swir)
    axes[2].set_title("SWIR (Fire Scars)")
    axes[2].axis("off")

    # Mask overlay
    if mask is not None:
        if mask_type == "DEL":
            colors = DEL_COLORS
            labels = DEL_LABELS
            n_classes = 2
        elif mask_type == "GRA":
            colors = GRA_COLORS
            labels = GRA_LABELS
            n_classes = 5
        else:
            colors = CLOUD_COLORS
            labels = CLOUD_LABELS
            n_classes = 4

        cmap = ListedColormap(colors[:n_classes])

        axes[3].imshow(rgb_true)
        mask_display = np.ma.masked_where(mask == 0, mask) if mask_type != "CM" else mask
        axes[3].imshow(mask_display, cmap=cmap, alpha=0.6, vmin=0, vmax=n_classes-1)
        axes[3].set_title(f"{mask_type} Mask")
        axes[3].axis("off")

        patches = [mpatches.Patch(color=colors[i], label=labels[i])
                   for i in range(min(len(labels), n_classes))]
        axes[3].legend(handles=patches, loc="lower right", fontsize=8)

    plt.suptitle(f"Source Image: {image_path.name}\nShape: {image.shape[0]}x{image.shape[1]}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize CEMS Wildfire patches")
    subparsers = parser.add_subparsers(dest="command", help="Visualization command")

    # Patch visualization
    patch_parser = subparsers.add_parser("patch", help="Visualize a single patch")
    patch_parser.add_argument("patch_id", help="Patch ID or path to image .npy file")
    patch_parser.add_argument("--patches-dir", type=Path, default=Path("."), help="Patches directory")
    patch_parser.add_argument("--mask-type", choices=["DEL", "GRA", "CM"], default="DEL")
    patch_parser.add_argument("--save", type=Path, help="Save path")

    # Grid visualization
    grid_parser = subparsers.add_parser("grid", help="Plot sample grid")
    grid_parser.add_argument("patches_dir", type=Path, help="Patches directory")
    grid_parser.add_argument("--n-samples", type=int, default=16, help="Number of samples")
    grid_parser.add_argument("--mode", choices=["true_color", "false_color", "swir"], default="true_color")
    grid_parser.add_argument("--mask-type", choices=["DEL", "GRA"], default="DEL")
    grid_parser.add_argument("--no-mask", action="store_true", help="Don't show masks")
    grid_parser.add_argument("--save", type=Path, help="Save path")

    # Statistics
    stats_parser = subparsers.add_parser("stats", help="Plot dataset statistics")
    stats_parser.add_argument("patches_dir", type=Path, help="Patches directory")
    stats_parser.add_argument("--save", type=Path, help="Save path")

    # Source image
    source_parser = subparsers.add_parser("source", help="Visualize source GeoTIFF")
    source_parser.add_argument("image_path", type=Path, help="Path to S2L2A.tif")
    source_parser.add_argument("--mask", type=Path, help="Path to mask file")
    source_parser.add_argument("--mask-type", choices=["DEL", "GRA", "CM"], default="DEL")
    source_parser.add_argument("--save", type=Path, help="Save path")

    args = parser.parse_args()

    if args.command == "patch":
        # Load patch
        if args.patch_id.endswith(".npy"):
            img_path = Path(args.patch_id)
            patch_id = img_path.stem.replace("_image", "")
            patches_dir = img_path.parent
        else:
            patch_id = args.patch_id
            patches_dir = args.patches_dir
            img_path = patches_dir / f"{patch_id}_image.npy"

        image = np.load(img_path)

        mask_path = patches_dir / f"{patch_id}_mask.npy"
        mask = np.load(mask_path) if mask_path.exists() else None

        plot_patch(image, mask, args.mask_type, title=patch_id, save_path=args.save)

    elif args.command == "grid":
        plot_sample_grid(
            args.patches_dir,
            n_samples=args.n_samples,
            mode=args.mode,
            with_mask=not args.no_mask,
            mask_type=args.mask_type,
            save_path=args.save,
        )

    elif args.command == "stats":
        plot_dataset_statistics(args.patches_dir, save_path=args.save)

    elif args.command == "source":
        plot_source_image(
            args.image_path,
            mask_path=args.mask,
            mask_type=args.mask_type,
            save_path=args.save,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
