"""NextDayWildfireSpread Dataset for CMPF intermediate fine-tuning."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class NDWSDataset(Dataset):
    """NextDayWildfireSpread dataset for intermediate fine-tuning.

    Each sample consists of a single-day input (H×W×12)
    and a binary fire mask target for the next day.
    """

    # Channel information for NDWS (12 channels)
    CHANNEL_NAMES = [
        "slope",
        "elevation",
        "aspect",
        "NDVI",
        "precipitation",
        "wind_speed",
        "wind_direction",
        "min_temp",
        "max_temp",
        "ERC",
        "specific_humidity",
        "active_fire",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
        band_indices: Optional[List[int]] = None,
        augment: bool = True,
        normalize: bool = True,
        fraction: float = 1.0,  # Fraction of data to use (for Exp 4)
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.band_indices = band_indices
        self.augment = augment and (split == "train")
        self.normalize = normalize
        self.fraction = fraction
        self.seed = seed

        # Build sample list
        self.samples = self._build_sample_list()

        # Subsample if fraction < 1.0
        if fraction < 1.0:
            rng = np.random.RandomState(seed)
            n_samples = int(len(self.samples) * fraction)
            indices = rng.choice(len(self.samples), n_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]

        # Load normalization stats
        self.mean, self.std = self._load_stats()

    def _build_sample_list(self) -> List[str]:
        """Build list of event_id strings."""
        samples = []
        events_dir = self.data_root / "events"

        if not events_dir.exists():
            return samples

        for event_dir in sorted(events_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            # Check if both X and y files exist
            x_path = event_dir / "X_t.npy"
            y_path = event_dir / "y_t+1.npy"
            if x_path.exists() and y_path.exists():
                samples.append(event_dir.name)

        return samples

    def _load_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load channel-wise mean and std for normalization."""
        n_channels = len(self.band_indices) if self.band_indices else 12
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
        event_id = self.samples[idx]
        event_dir = self.data_root / "events" / event_id

        # Load input and target
        x = np.load(event_dir / "X_t.npy")  # H×W×C
        y = np.load(event_dir / "y_t+1.npy")  # H×W

        # Select bands if specified
        if self.band_indices:
            x = x[..., self.band_indices]

        # Add temporal dimension to match WTS format: 1×H×W×C -> 1×C×H×W
        x = torch.from_numpy(x).float().unsqueeze(0).permute(0, 3, 1, 2)
        y = torch.from_numpy(y).long()

        # Normalize
        if self.normalize:
            mean = torch.tensor(self.mean).view(1, -1, 1, 1)
            std = torch.tensor(self.std).view(1, -1, 1, 1)
            x = (x - mean) / (std + 1e-8)

        # Augmentation and resize
        x, y = self._transform(x, y)

        meta = {"event_id": event_id}

        return x, y, meta

    def _transform(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms (resize, augmentation)."""
        T, C, H, W = x.shape

        if self.augment:
            # Random crop
            if H >= self.img_size and W >= self.img_size:
                top = torch.randint(0, H - self.img_size + 1, (1,)).item()
                left = torch.randint(0, W - self.img_size + 1, (1,)).item()
                x = x[:, :, top : top + self.img_size, left : left + self.img_size]
                y = y[top : top + self.img_size, left : left + self.img_size]
            else:
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

            # Random flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-1])
                y = torch.flip(y, [-1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-2])
                y = torch.flip(y, [-2])
        else:
            # Resize for validation/test
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


class NDWSDataModule:
    """Data module for NextDayWildfireSpread dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        img_size: int = 224,
        band_indices: Optional[List[int]] = None,
        num_workers: int = 4,
        fraction: float = 1.0,
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.img_size = img_size
        self.band_indices = band_indices
        self.num_workers = num_workers
        self.fraction = fraction

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        dataset = NDWSDataset(
            self.data_root,
            "train",
            self.img_size,
            self.band_indices,
            augment=True,
            fraction=self.fraction,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
