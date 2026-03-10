"""Tests for TerraTorch configuration files."""

from pathlib import Path

import yaml


def test_prithvi_config_loads():
    """Prithvi config should be valid YAML."""
    config_path = Path(__file__).parent.parent / "configs" / "prithvi_upernet.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "trainer" in config
        assert "model" in config
        assert "data" in config


def test_all_configs_have_required_keys():
    """All experiment configs should define trainer, model, and data."""
    configs_dir = Path(__file__).parent.parent / "configs"
    required_keys = {"trainer", "model", "data"}

    for config_file in configs_dir.glob("*.yaml"):
        if config_file.name == "data_config.yaml":
            continue
        with open(config_file) as f:
            config = yaml.safe_load(f)
        missing = required_keys - set(config.keys())
        assert not missing, f"{config_file.name} missing keys: {missing}"
