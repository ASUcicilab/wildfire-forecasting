"""Configuration utilities for CMPF experiments."""

from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)

    # Handle inheritance
    if "inherit" in config:
        parent_path = Path(config_path).parent / config.inherit
        if parent_path.exists():
            parent_config = load_config(str(parent_path))
            config = OmegaConf.merge(parent_config, config)

    return config


def save_config(config: DictConfig, save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def merge_configs(base: DictConfig, override: Dict[str, Any]) -> DictConfig:
    """Merge base config with overrides from command line."""
    return OmegaConf.merge(base, OmegaConf.create(override))


def get_shared_bands_config(config_path: str = "configs/shared_bands.yaml") -> Dict:
    """Load shared bands configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_experiment_config(
    experiment_name: str,
    backbone: str = "videomaev2",
    T: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    epochs: int = 100,
    **kwargs,
) -> DictConfig:
    """Create a configuration for an experiment."""
    config = {
        "experiment": experiment_name,
        "backbone": backbone,
        "T": T,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        **kwargs,
    }
    return OmegaConf.create(config)
