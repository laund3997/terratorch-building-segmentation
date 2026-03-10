"""TerraTorch Building Segmentation - Inference pipeline."""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from terratorch.tasks import SemanticSegmentationTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def predict_tiles(model: SemanticSegmentationTask, input_dir: Path, output_dir: Path, device: str = "cuda"):
    """Run inference on a directory of tiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)

    tile_files = sorted(input_dir.glob("*.tif"))
    logger.info(f"Running inference on {len(tile_files)} tiles")

    for tile_file in tile_files:
        with rasterio.open(tile_file) as src:
            image = src.read().astype(np.float32)
            profile = src.profile.copy()

        tensor = torch.from_numpy(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        profile.update(count=1, dtype="uint8")
        output_file = output_dir / tile_file.name
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(pred[np.newaxis, :, :])

    logger.info(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=Path, required=True, help="Input tiles directory")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.checkpoint}")
    model = SemanticSegmentationTask.load_from_checkpoint(str(args.checkpoint))
    predict_tiles(model, args.input, args.output, args.device)


if __name__ == "__main__":
    main()
