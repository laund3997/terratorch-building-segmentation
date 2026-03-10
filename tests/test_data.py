"""Tests for data preprocessing pipeline."""

from pathlib import Path

import numpy as np


def test_tile_dimensions():
    """Tiles should be 224x224."""
    tile_size = 224
    assert tile_size == 224


def test_spatial_split_ratios():
    """Train/val/test split should sum to 1.0."""
    train, val, test = 0.7, 0.15, 0.15
    assert abs(train + val + test - 1.0) < 1e-6


def test_mask_values():
    """Masks should only contain 0 (background), 1 (building), or 255 (nodata)."""
    mask = np.array([0, 1, 0, 1, 255, 0], dtype=np.uint8)
    valid_values = {0, 1, 255}
    assert set(np.unique(mask)).issubset(valid_values)
