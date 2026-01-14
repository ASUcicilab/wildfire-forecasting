"""Video foundation model backbones for CMPF."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video/temporal data."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        temporal_size: int = 5,
        temporal_patch_size: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        self.temporal_patch_size = temporal_patch_size

        self.num_patches_spatial = (img_size // patch_size) ** 2
        self.num_patches_temporal = temporal_size // temporal_patch_size
        self.num_patches = self.num_patches_spatial * self.num_patches_temporal

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B×T×C×H×W -> B×C×T×H×W
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # B×D×T'×H'×W'
        x = x.flatten(2).transpose(1, 2)  # B×N×D
        return x


class ViTBackbone(nn.Module):
    """Vision Transformer backbone adapted for multi-channel temporal input.

    Handles T×C×H×W input by treating temporal frames as additional channels
    or using temporal attention.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 23,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        T: int = 5,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Flatten temporal dimension into channels for ViT
        self.total_channels = in_channels * T

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.total_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        self._init_weights()

        # Load pretrained if provided
        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def load_pretrained(self, ckpt_path: str):
        """Load pretrained weights, adapting input projection if needed."""
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Adapt patch_embed if channel count differs
        if "patch_embed.proj.weight" in state_dict:
            pretrained_weight = state_dict["patch_embed.proj.weight"]
            if pretrained_weight.shape[1] != self.total_channels:
                # Repeat or average to match channel count
                pretrained_weight = self._adapt_input_channels(
                    pretrained_weight, self.total_channels
                )
                state_dict["patch_embed.proj.weight"] = pretrained_weight

        # Load with strict=False for flexibility
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded pretrained ViT. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    def _adapt_input_channels(
        self, weight: torch.Tensor, target_channels: int
    ) -> torch.Tensor:
        """Adapt pretrained weights to different input channel count."""
        src_channels = weight.shape[1]
        if target_channels > src_channels:
            # Repeat weights
            repeats = math.ceil(target_channels / src_channels)
            weight = weight.repeat(1, repeats, 1, 1)[:, :target_channels]
        else:
            # Average weights
            weight = weight[:, :target_channels]
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B×T×C×H×W
        Returns:
            features: B×N×D (N = num_patches)
        """
        B, T, C, H, W = x.shape

        # Flatten temporal into channels: B×(T*C)×H×W
        x = x.view(B, T * C, H, W)

        # Patch embedding
        x = self.patch_embed(x)  # B×D×H'×W'
        x = x.flatten(2).transpose(1, 2)  # B×N×D

        # Add cls token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 1:]  # Remove cls token, return patch features


class MViTv2Backbone(nn.Module):
    """Multiscale Vision Transformer v2 backbone for video understanding.

    Uses pooling attention for efficient multi-scale feature extraction.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 23,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 3, 16, 3),
        num_heads: Tuple[int, ...] = (1, 2, 4, 8),
        T: int = 5,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.depths = depths

        # 3D patch embedding for video
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            temporal_size=T,
            temporal_patch_size=1 if T <= 2 else 2,
        )

        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Build stages with increasing dimensions
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList(
                [TransformerBlock(dim, heads, mlp_ratio=4.0) for _ in range(depth)]
            )
            self.stages.append(stage)

            # Downsample between stages (except last)
            if i < len(depths) - 1:
                self.stages.append(nn.Linear(dim, dim * 2))
                dim = dim * 2

        self.final_dim = dim
        self.norm = nn.LayerNorm(dim)

        self._init_weights()

        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def load_pretrained(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded pretrained MViTv2. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B×T×C×H×W
        Returns:
            features: B×N×D
        """
        x = self.patch_embed(x)  # B×N×D
        x = x + self.pos_embed

        for module in self.stages:
            if isinstance(module, nn.ModuleList):
                for blk in module:
                    x = blk(x)
            else:
                x = module(x)

        x = self.norm(x)
        return x


class VideoMAEv2Backbone(nn.Module):
    """VideoMAE v2 backbone - self-supervised video transformer.

    Pretrained with masked autoencoding on video data, particularly effective
    for video understanding tasks.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 23,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        T: int = 5,
        tubelet_size: int = 2,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_size = min(tubelet_size, T)

        # Calculate patch dimensions
        self.num_patches_spatial = (img_size // patch_size) ** 2
        self.num_patches_temporal = T // self.tubelet_size
        self.num_patches = self.num_patches_spatial * self.num_patches_temporal

        # Tubelet embedding (3D patches)
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(self.tubelet_size, patch_size, patch_size),
            stride=(self.tubelet_size, patch_size, patch_size),
        )

        # Positional embeddings (separate spatial and temporal)
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_patches_spatial, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_patches_temporal, embed_dim)
        )

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

    def load_pretrained(self, ckpt_path: str):
        """Load pretrained VideoMAE weights with channel adaptation."""
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Adapt patch embedding for different input channels
        key = "patch_embed.proj.weight"
        if key not in state_dict:
            key = "patch_embed.weight"
        if key in state_dict:
            pretrained_weight = state_dict[key]
            # VideoMAE uses 3D conv: out×in×T×H×W
            if pretrained_weight.dim() == 5:
                src_in = pretrained_weight.shape[1]
                if src_in != self.patch_embed.in_channels:
                    # Adapt input channels
                    new_weight = self._adapt_3d_weight(
                        pretrained_weight, self.patch_embed.in_channels
                    )
                    state_dict[key] = new_weight

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded pretrained VideoMAEv2. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    def _adapt_3d_weight(
        self, weight: torch.Tensor, target_channels: int
    ) -> torch.Tensor:
        """Adapt 3D conv weights to different channel count."""
        src_channels = weight.shape[1]
        if target_channels > src_channels:
            repeats = math.ceil(target_channels / src_channels)
            weight = weight.repeat(1, repeats, 1, 1, 1)[:, :target_channels]
        else:
            weight = weight[:, :target_channels]
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: B×T×C×H×W
        Returns:
            features: B×N×D (N = num_patches_spatial × num_patches_temporal)
        """
        B, T, C, H, W = x.shape

        # B×T×C×H×W -> B×C×T×H×W
        x = x.permute(0, 2, 1, 3, 4)

        # Tubelet embedding
        x = self.patch_embed(x)  # B×D×T'×H'×W'

        # Reshape to sequence
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Add factorized positional embeddings
        T_patches = self.num_patches_temporal
        S_patches = self.num_patches_spatial

        pos_embed = self.pos_embed_spatial.repeat(
            1, T_patches, 1
        ) + self.pos_embed_temporal.repeat_interleave(S_patches, dim=1)
        x = x + pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def create_backbone(
    name: str,
    in_channels: int,
    T: int,
    img_size: int = 224,
    pretrained_ckpt: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create backbone by name."""

    backbones = {
        "vit": ViTBackbone,
        "mvitv2": MViTv2Backbone,
        "videomaev2": VideoMAEv2Backbone,
    }

    if name not in backbones:
        raise ValueError(
            f"Unknown backbone: {name}. Choose from {list(backbones.keys())}"
        )

    return backbones[name](
        img_size=img_size,
        in_channels=in_channels,
        T=T,
        pretrained_ckpt=pretrained_ckpt,
        **kwargs,
    )
