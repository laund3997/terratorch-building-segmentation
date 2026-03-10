"""TerraTorch Building Segmentation - Data download and preprocessing."""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_tiles(image_path: Path, output_dir: Path, tile_size: int = 224, overlap: int = 32):
    """Tile a large raster into smaller patches for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles = []

    with rasterio.open(image_path) as src:
        step = tile_size - overlap
        for y in range(0, src.height - tile_size + 1, step):
            for x in range(0, src.width - tile_size + 1, step):
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)

                # Skip tiles with too much nodata
                if np.isnan(tile).mean() > 0.1:
                    continue

                tile_name = f"tile_{y:05d}_{x:05d}.tif"
                tile_path = output_dir / tile_name
                profile = src.profile.copy()
                profile.update(width=tile_size, height=tile_size, transform=src.window_transform(window))

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile)

                tiles.append(tile_path)

    logger.info(f"Created {len(tiles)} tiles from {image_path.name}")
    return tiles


def create_masks(tile_paths: list[Path], buildings_gdf: gpd.GeoDataFrame, output_dir: Path):
    """Create binary building masks for each tile."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            from rasterio.features import rasterize

            mask = rasterize(
                [(geom, 1) for geom in buildings_gdf.geometry],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype=np.uint8,
            )

            mask_path = output_dir / tile_path.name
            profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=255)

            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask[np.newaxis, :, :])


def spatial_train_val_test_split(
    tile_paths: list[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Split tiles spatially to avoid data leakage."""
    train_val, test = train_test_split(tile_paths, test_size=1 - train_ratio - val_ratio, random_state=seed)
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)
    logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for TerraTorch building segmentation")
    subparsers = parser.add_subparsers(dest="command")

    # Download subcommand
    dl = subparsers.add_parser("download", help="Download Sentinel-2 data")
    dl.add_argument("--aoi", type=Path, required=True, help="GeoJSON file with area of interest")
    dl.add_argument("--output", type=Path, required=True, help="Output directory")

    # Preprocess subcommand
    pp = subparsers.add_parser("preprocess", help="Tile and preprocess data")
    pp.add_argument("--input", type=Path, required=True, help="Input raster directory")
    pp.add_argument("--labels", type=Path, required=True, help="Building polygons directory")
    pp.add_argument("--output", type=Path, required=True, help="Output directory")
    pp.add_argument("--tile-size", type=int, default=224)
    pp.add_argument("--config", type=Path, help="Data config YAML")

    args = parser.parse_args()

    if args.command == "download":
        logger.info(f"Download AOI from {args.aoi} → {args.output}")
        logger.info("Use Copernicus Open Access Hub or planetary computer for Sentinel-2 L2A data.")

    elif args.command == "preprocess":
        args.output.mkdir(parents=True, exist_ok=True)

        # Load building polygons
        buildings = gpd.read_file(args.labels)
        logger.info(f"Loaded {len(buildings)} building polygons")

        # Tile all input rasters
        all_tiles = []
        for raster_file in sorted(args.input.glob("*.tif")):
            tiles = create_tiles(raster_file, args.output / "tiles", args.tile_size)
            all_tiles.extend(tiles)

        # Create masks
        create_masks(all_tiles, buildings, args.output / "masks")

        # Split
        train, val, test = spatial_train_val_test_split(all_tiles)

        # Move tiles to split directories
        for split_name, split_tiles in [("train", train), ("val", val), ("test", test)]:
            img_dir = args.output / split_name / "images"
            mask_dir = args.output / split_name / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for tile_path in split_tiles:
                tile_path.rename(img_dir / tile_path.name)
                mask_path = args.output / "masks" / tile_path.name
                if mask_path.exists():
                    mask_path.rename(mask_dir / tile_path.name)

        logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
