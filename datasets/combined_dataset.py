"""Combined Dataset for CMPF S3 strategy (combined intermediate training)."""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .ndws_dataset import NDWSDataset
from .wts_dataset import WTSDataset


class CombinedDataset(Dataset):
    """Combined WTS + NDWS dataset for CMPF S3 strategy.

    Aligns both datasets to use shared bands and pads to same channel count.
    """

    # Mapping from shared band names to indices in each dataset
    SHARED_BAND_MAPPING = {
        "NDVI": {"wts": 3, "ndws": 3},
        "precipitation": {"wts": 5, "ndws": 4},
        "wind_speed": {"wts": 6, "ndws": 5},
        "wind_direction": {"wts": 7, "ndws": 6},
        "min_temp": {"wts": 8, "ndws": 7},
        "max_temp": {"wts": 9, "ndws": 8},
        "ERC": {"wts": 10, "ndws": 9},
        "specific_humidity": {"wts": 11, "ndws": 10},
        "slope": {"wts": 12, "ndws": 0},
        "elevation": {"wts": 14, "ndws": 1},
        "active_fire": {"wts": 22, "ndws": 11},
    }

    def __init__(
        self,
        wts_root: str,
        ndws_root: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
        use_shared_bands_only: bool = True,  # For S1/S3
        wts_fraction: float = 1.0,
        ndws_fraction: float = 1.0,
    ):
        super().__init__()
        self.use_shared_bands_only = use_shared_bands_only

        # Get shared band indices for each dataset
        wts_bands = [
            self.SHARED_BAND_MAPPING[k]["wts"] for k in self.SHARED_BAND_MAPPING
        ]
        ndws_bands = [
            self.SHARED_BAND_MAPPING[k]["ndws"] for k in self.SHARED_BAND_MAPPING
        ]

        if use_shared_bands_only:
            wts_band_indices = wts_bands
            ndws_band_indices = ndws_bands
        else:
            wts_band_indices = None
            ndws_band_indices = None

        # Create wrapped datasets
        self.wts_dataset = WTSDataset(
            wts_root,
            split,
            T=1,
            img_size=img_size,
            band_indices=wts_band_indices,
            augment=augment,
        )

        self.ndws_dataset = NDWSDataset(
            ndws_root,
            split,
            img_size=img_size,
            band_indices=ndws_band_indices,
            augment=augment,
            fraction=ndws_fraction,
        )

        # Combined length
        self.wts_len = len(self.wts_dataset)
        self.ndws_len = len(self.ndws_dataset)
        self.total_len = self.wts_len + self.ndws_len

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        if idx < self.wts_len:
            x, y, meta = self.wts_dataset[idx]
            meta["source"] = "wts"
        else:
            x, y, meta = self.ndws_dataset[idx - self.wts_len]
            meta["source"] = "ndws"

        return x, y, meta


def get_shared_band_indices(dataset: str) -> List[int]:
    """Get indices of shared bands for a specific dataset."""
    key = "wts" if dataset == "wts" else "ndws"
    return [
        CombinedDataset.SHARED_BAND_MAPPING[k][key]
        for k in CombinedDataset.SHARED_BAND_MAPPING
    ]
