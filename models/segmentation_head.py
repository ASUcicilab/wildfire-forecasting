"""Segmentation heads for dense prediction from transformer features."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class UPerHead(nn.Module):
    """UPerNet-style decoder head for semantic segmentation.

    Uses Pyramid Pooling Module and Feature Pyramid Network.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        img_size: int = 224,
        patch_size: int = 16,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.feature_size = img_size // patch_size

        # PPM (Pyramid Pooling Module)
        self.ppm = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )

        # Bottleneck after PPM
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_dim + hidden_dim * len(pool_scales),
                hidden_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B×N×D patch features
        Returns:
            logits: B×C×H×W
        """
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        # Reshape to spatial: B×D×H×W
        x = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

        # PPM
        ppm_outs = [x]
        for ppm in self.ppm:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(
                ppm_out, size=(H, W), mode="bilinear", align_corners=False
            )
            ppm_outs.append(ppm_out)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)

        # Predict and upsample
        x = self.head(x)
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

        return x


class SimpleHead(nn.Module):
    """Simple convolutional head for segmentation."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        # Reshape and upsample
        x = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)
        x = self.head(x)
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

        return x


class ProgressiveUpsampling(nn.Module):
    """Progressive upsampling head with skip-like refinement."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_upsample = int(math.log2(patch_size))

        # Progressive upsampling layers
        self.upsample_layers = nn.ModuleList()
        dim = in_dim
        for i in range(self.n_upsample):
            out_dim = hidden_dim if i < self.n_upsample - 1 else hidden_dim
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, out_dim, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            )
            dim = out_dim

        # Final prediction
        self.final = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        x = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

        for layer in self.upsample_layers:
            x = layer(x)

        x = self.final(x)

        # Ensure exact output size
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        return x


def create_seg_head(
    name: str,
    in_dim: int,
    num_classes: int = 1,
    img_size: int = 224,
    patch_size: int = 16,
    **kwargs,
) -> nn.Module:
    """Factory function to create segmentation head by name."""

    heads = {
        "uper": UPerHead,
        "simple": SimpleHead,
        "progressive": ProgressiveUpsampling,
    }

    if name not in heads:
        raise ValueError(f"Unknown head: {name}. Choose from {list(heads.keys())}")

    return heads[name](
        in_dim=in_dim,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        **kwargs,
    )
