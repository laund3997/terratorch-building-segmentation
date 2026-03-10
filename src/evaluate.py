"""TerraTorch Building Segmentation - Evaluation metrics and reporting."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(predictions_dir: Path, ground_truth_dir: Path) -> dict:
    """Compute segmentation metrics between predictions and ground truth masks."""
    all_preds = []
    all_labels = []

    pred_files = sorted(predictions_dir.glob("*.tif"))
    logger.info(f"Evaluating {len(pred_files)} prediction files")

    for pred_file in pred_files:
        gt_file = ground_truth_dir / pred_file.name
        if not gt_file.exists():
            logger.warning(f"Ground truth not found for {pred_file.name}, skipping")
            continue

        with rasterio.open(pred_file) as pred_src, rasterio.open(gt_file) as gt_src:
            pred = pred_src.read(1).flatten()
            gt = gt_src.read(1).flatten()

            # Exclude nodata pixels
            valid = gt != 255
            all_preds.append(pred[valid])
            all_labels.append(gt[valid])

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary")),
        "recall": float(recall_score(labels, preds, average="binary")),
        "f1_score": float(f1_score(labels, preds, average="binary")),
        "iou": float(jaccard_score(labels, preds, average="binary")),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, target_names=["background", "building"]),
        "num_samples": int(len(preds)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = compute_metrics(args.predictions, args.ground_truth)

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "metrics.json"
    with open(output_file, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    report_file = args.output / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(metrics["classification_report"])

    logger.info(f"F1: {metrics['f1_score']:.4f} | IoU: {metrics['iou']:.4f} | Acc: {metrics['accuracy']:.4f}")
    logger.info(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
