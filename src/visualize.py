"""TerraTorch Building Segmentation - Result visualization."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_comparison(
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    output_path: Path,
):
    """Plot side-by-side comparison of input, ground truth, and prediction."""
    with rasterio.open(image_path) as src:
        # Use RGB bands (B04, B03, B02) for visualization
        rgb = src.read([3, 2, 1])  # Assuming bands ordered as in config
        rgb = np.clip(rgb / 3000.0, 0, 1).transpose(1, 2, 0)

    with rasterio.open(gt_path) as src:
        gt = src.read(1)

    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[2].set_title("Prediction (Prithvi + UperNet)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison plot to {output_path}")


def plot_metrics_comparison(metrics_files: dict[str, Path], output_path: Path):
    """Plot bar chart comparing metrics across models."""
    import json

    models = {}
    for name, path in metrics_files.items():
        with open(path) as f:
            models[name] = json.load(f)

    metric_names = ["accuracy", "precision", "recall", "f1_score", "iou"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, metrics) in enumerate(models.items()):
        values = [metrics[m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Building Segmentation (Algiers)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "IoU"])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metrics comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--results", type=Path, required=True, help="Results directory")
    parser.add_argument("--output", type=Path, required=True, help="Output figures directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating visualizations in {args.output}")


if __name__ == "__main__":
    main()
