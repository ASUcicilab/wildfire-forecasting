"""CMPF Model - Video Foundation Model with Segmentation Head."""

from typing import Optional

import torch
import torch.nn as nn

from .backbones import VideoMAEv2Backbone, ViTBackbone, create_backbone
from .base import BaseModel
from .segmentation_head import create_seg_head


class CMPFModel(BaseModel):
    """Cross-Modal Progressive Fine-tuning Model.

    Combines a video foundation model backbone (ViT, MViTv2, VideoMAEv2)
    with a segmentation head for wildfire spread prediction.
    """

    def __init__(
        self,
        backbone_name: str = "videomaev2",
        in_channels: int = 23,
        num_classes: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        T: int = 5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        head_type: str = "uper",
        pretrained_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            T=T,
            **kwargs,
        )

        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone

        # Create backbone
        self.backbone = create_backbone(
            name=backbone_name,
            in_channels=in_channels,
            T=T,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            pretrained_ckpt=pretrained_ckpt,
        )

        # Get backbone output dimension
        if hasattr(self.backbone, "final_dim"):
            backbone_dim = self.backbone.final_dim
        else:
            backbone_dim = embed_dim

        # Create segmentation head
        self.head = create_seg_head(
            name=head_type,
            in_dim=backbone_dim,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B×T×C×H×W input tensor
        Returns:
            logits: B×1×H×W prediction logits
        """
        # Handle single-frame input for NDWS
        if x.dim() == 4:
            x = x.unsqueeze(1)  # B×C×H×W -> B×1×C×H×W

        # Pad temporal dimension if needed
        B, T, C, H, W = x.shape
        if T < self.T:
            # Repeat last frame
            pad = x[:, -1:].repeat(1, self.T - T, 1, 1, 1)
            x = torch.cat([x, pad], dim=1)

        # Extract features
        features = self.backbone(x)  # B×N×D

        # Decode to dense prediction
        logits = self.head(features)  # B×1×H×W

        return logits

    def adapt_input_channels(self, new_in_channels: int):
        """Adapt model for different number of input channels.

        Used when switching between shared-band and full-band training.
        """
        if new_in_channels == self.in_channels:
            return

        # Get old projection weights
        old_weight = None
        if isinstance(self.backbone, (ViTBackbone,)):
            old_weight = self.backbone.patch_embed.weight.data
            old_bias = (
                self.backbone.patch_embed.bias.data
                if self.backbone.patch_embed.bias is not None
                else None
            )

            # Create new projection
            self.backbone.patch_embed = nn.Conv2d(
                new_in_channels * self.T,
                self.backbone.embed_dim,
                kernel_size=self.backbone.patch_size,
                stride=self.backbone.patch_size,
            )

        elif isinstance(self.backbone, (VideoMAEv2Backbone,)):
            old_weight = self.backbone.patch_embed.weight.data
            old_bias = (
                self.backbone.patch_embed.bias.data
                if self.backbone.patch_embed.bias is not None
                else None
            )

            # Create new 3D projection
            self.backbone.patch_embed = nn.Conv3d(
                new_in_channels,
                self.backbone.embed_dim,
                kernel_size=(
                    self.backbone.tubelet_size,
                    self.backbone.patch_size,
                    self.backbone.patch_size,
                ),
                stride=(
                    self.backbone.tubelet_size,
                    self.backbone.patch_size,
                    self.backbone.patch_size,
                ),
            )

        # Initialize new weights from old if possible
        if old_weight is not None:
            with torch.no_grad():
                new_weight = self.backbone.patch_embed.weight.data
                # Copy what we can
                min_ch = min(old_weight.shape[1], new_weight.shape[1])
                new_weight[:, :min_ch] = old_weight[:, :min_ch]

                # Initialize remaining channels with mean of existing
                if new_weight.shape[1] > min_ch:
                    new_weight[:, min_ch:] = old_weight.mean(
                        dim=1, keepdim=True
                    ).expand_as(new_weight[:, min_ch:])

                if old_bias is not None:
                    self.backbone.patch_embed.bias.data = old_bias

        self.in_channels = new_in_channels
        self.backbone.total_channels = (
            new_in_channels * self.T
            if hasattr(self.backbone, "total_channels")
            else new_in_channels
        )


class CMPFModelWithChannelAdapter(CMPFModel):
    """CMPF Model with learnable channel adapter for S2/S3 strategies.

    Uses a small network to project different input channels to a
    common representation before the backbone.
    """

    def __init__(
        self,
        source_channels: int,  # Original pretrained channels
        target_channels: int,  # New input channels
        adapter_hidden: int = 64,
        **kwargs,
    ):
        # Initialize with target channels
        kwargs["in_channels"] = target_channels
        super().__init__(**kwargs)

        self.source_channels = source_channels
        self.target_channels = target_channels

        # Channel adapter: projects target channels to source channels
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(target_channels, adapter_hidden, 1),
            nn.BatchNorm2d(adapter_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_hidden, source_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape

        # Apply channel adapter per frame
        x = x.view(B * T, C, H, W)
        x = self.channel_adapter(x)
        x = x.view(B, T, -1, H, W)

        # Pad temporal if needed
        if T < self.T:
            pad = x[:, -1:].repeat(1, self.T - T, 1, 1, 1)
            x = torch.cat([x, pad], dim=1)

        features = self.backbone(x)
        logits = self.head(features)

        return logits
