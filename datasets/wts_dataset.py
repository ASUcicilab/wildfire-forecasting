"""WildfireSpreadTS Dataset for CMPF experiments."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class WTSDataset(Dataset):
    """WildfireSpreadTS dataset for wildfire spread prediction.

    Each sample consists of T days of multi-modal inputs (H×W×23 per day)
    and a binary fire mask target for day T+1.
    """

    # Channel information for WTS (23 base channels)
    CHANNEL_NAMES = [
        "VIIRS_M11",
        "VIIRS_I2",
        "VIIRS_I1",
        "NDVI",
        "EVI2",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "min_temp",
        "max_temp",
        "ERC",
        "specific_humidity",
        "slope",
        "aspect",
        "elevation",
        "PDSI",
        "landcover",
        "fcst_precipitation",
        "fcst_wind_speed",
        "fcst_wind_direction",
        "fcst_temperature",
        "fcst_humidity",
        "active_fire",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",  # 'train', 'val', 'test'
        T: int = 5,  # temporal context
        img_size: int = 224,
        band_indices: Optional[List[int]] = None,  # None = all bands
        augment: bool = True,
        normalize: bool = True,
        metadata_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.T = T
        self.img_size = img_size
        self.band_indices = band_indices
        self.augment = augment and (split == "train")
        self.normalize = normalize

        # Determine split directory
        if split == "train":
            self.split_dir = self.data_root / "train"
        else:
            self.split_dir = self.data_root / "val_2021"

        # Build sample list
        self.samples = self._build_sample_list()

        # Load normalization stats
        self.mean, self.std = self._load_stats()

    def _build_sample_list(self) -> List[Tuple[str, int]]:
        """Build list of (event_id, day_idx) tuples."""
        samples = []

        if not self.split_dir.exists():
            return samples

        for event_dir in sorted(self.split_dir.iterdir()):
            if not event_dir.is_dir():
                continue

            # Count available days
            x_files = sorted(event_dir.glob("day_*_X.npy"))
            n_days = len(x_files)

            # Each sample uses T days as input, predicting day T+1
            for day_idx in range(n_days - self.T):
                samples.append((event_dir.name, day_idx))

        return samples

    def _load_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load channel-wise mean and std for normalization."""
        # Default stats (can be computed from training data)
        n_channels = len(self.band_indices) if self.band_indices else 23
        mean = np.zeros(n_channels, dtype=np.float32)
        std = np.ones(n_channels, dtype=np.float32)

        stats_path = self.data_root / "stats.npz"
        if stats_path.exists():
            stats = np.load(stats_path)
            mean = stats["mean"]
            std = stats["std"]
            if self.band_indices:
                mean = mean[self.band_indices]
                std = std[self.band_indices]

        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        event_id, day_idx = self.samples[idx]
        event_dir = self.split_dir / event_id

        # Load T days of input
        x_list = []
        for t in range(self.T):
            x_path = event_dir / f"day_{day_idx + t:03d}_X.npy"
            x_t = np.load(x_path)  # H×W×C
            x_list.append(x_t)

        # Stack temporal dimension: T×H×W×C
        x = np.stack(x_list, axis=0)

        # Load target (next day's fire mask)
        y_path = event_dir / f"day_{day_idx + self.T:03d}_y.npy"
        y = np.load(y_path)  # H×W

        # Select bands if specified
        if self.band_indices:
            x = x[..., self.band_indices]

        # Convert to torch and rearrange to T×C×H×W
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        y = torch.from_numpy(y).long()

        # Normalize
        if self.normalize:
            mean = torch.tensor(self.mean).view(1, -1, 1, 1)
            std = torch.tensor(self.std).view(1, -1, 1, 1)
            x = (x - mean) / (std + 1e-8)

        # Augmentation and resize
        x, y = self._transform(x, y)

        meta = {
            "event_id": event_id,
            "day_idx": day_idx,
        }

        return x, y, meta

    def _transform(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms (resize, augmentation)."""
        T, C, H, W = x.shape

        if self.augment:
            # Random crop
            crop_params = self._get_crop_params(H, W, self.img_size)
            x = TF.crop(x.view(T * C, H, W), *crop_params).view(
                T, C, self.img_size, self.img_size
            )
            y = TF.crop(y.unsqueeze(0), *crop_params).squeeze(0)

            # Random flip
            if torch.rand(1) > 0.5:
                x = TF.hflip(x.view(T * C, self.img_size, self.img_size)).view(
                    T, C, self.img_size, self.img_size
                )
                y = TF.hflip(y.unsqueeze(0)).squeeze(0)
            if torch.rand(1) > 0.5:
                x = TF.vflip(x.view(T * C, self.img_size, self.img_size)).view(
                    T, C, self.img_size, self.img_size
                )
                y = TF.vflip(y.unsqueeze(0)).squeeze(0)

            # Random rotation (0, 90, 180, 270)
            angle = int(torch.randint(0, 4, (1,))) * 90
            if angle > 0:
                x = TF.rotate(x.view(T * C, self.img_size, self.img_size), angle).view(
                    T, C, self.img_size, self.img_size
                )
                y = TF.rotate(y.unsqueeze(0), angle).squeeze(0)
        else:
            # Center crop and resize for validation/test
            x = TF.resize(
                x.view(T * C, H, W), [self.img_size, self.img_size], antialias=True
            )
            x = x.view(T, C, self.img_size, self.img_size)
            y = (
                TF.resize(
                    y.unsqueeze(0).float(),
                    [self.img_size, self.img_size],
                    antialias=True,
                )
                .squeeze(0)
                .long()
            )

        return x, y

    def _get_crop_params(self, H: int, W: int, size: int) -> Tuple[int, int, int, int]:
        """Get random crop parameters."""
        if H < size or W < size:
            # Pad if needed
            return 0, 0, min(H, size), min(W, size)
        top = torch.randint(0, H - size + 1, (1,)).item()
        left = torch.randint(0, W - size + 1, (1,)).item()
        return top, left, size, size


class WTSDataModule:
    """Data module for WildfireSpreadTS dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        T: int = 5,
        img_size: int = 224,
        band_indices: Optional[List[int]] = None,
        num_workers: int = 4,
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.T = T
        self.img_size = img_size
        self.band_indices = band_indices
        self.num_workers = num_workers

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = WTSDataset(
            self.data_root,
            "train",
            self.T,
            self.img_size,
            self.band_indices,
            augment=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = WTSDataset(
            self.data_root,
            "val",
            self.T,
            self.img_size,
            self.band_indices,
            augment=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = WTSDataset(
            self.data_root,
            "test",
            self.T,
            self.img_size,
            self.band_indices,
            augment=False,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
