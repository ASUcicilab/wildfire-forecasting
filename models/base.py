"""Base model class for CMPF experiments."""

from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, LovaszLoss
from torchvision.ops import sigmoid_focal_loss


class BaseModel(pl.LightningModule, ABC):
    """Base class for all segmentation models in CMPF experiments."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        img_size: int = 224,
        T: int = 5,
        loss_function: str = "dice",
        pos_class_weight: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler: str = "cosine",
        max_epochs: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.T = T
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.max_epochs = max_epochs

        # Loss function
        self.loss_fn = self._build_loss(loss_function, pos_class_weight)

        # Metrics
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_ap = torchmetrics.AveragePrecision(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_ap = torchmetrics.AveragePrecision(task="binary")
        self.test_iou = torchmetrics.JaccardIndex(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")

    def _build_loss(self, loss_function: str, pos_weight: float):
        """Build loss function."""
        if loss_function == "bce":
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        elif loss_function == "focal":
            return lambda pred, target: sigmoid_focal_loss(
                pred, target.float(), alpha=0.25, gamma=2.0, reduction="mean"
            )
        elif loss_function == "dice":
            return DiceLoss(mode="binary")
        elif loss_function == "jaccard":
            return JaccardLoss(mode="binary")
        elif loss_function == "lovasz":
            return LovaszLoss(mode="binary")
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: B×T×C×H×W, Output: B×1×H×W"""
        pass

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        return self.loss_fn(y_hat, y.float())

    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x).squeeze(1)  # B×H×W

        loss = self.compute_loss(y_hat, y)
        self.train_f1(y_hat.sigmoid(), y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x).squeeze(1)

        loss = self.compute_loss(y_hat, y)
        self.val_f1(y_hat.sigmoid(), y)
        self.val_ap(y_hat.sigmoid(), y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        y_hat = self(x).squeeze(1)

        loss = self.compute_loss(y_hat, y)
        pred_prob = y_hat.sigmoid()

        self.test_f1(pred_prob, y)
        self.test_ap(pred_prob, y)
        self.test_iou(pred_prob, y)
        self.test_precision(pred_prob, y)
        self.test_recall(pred_prob, y)

        self.log("test_loss", loss)
        self.log("test_f1", self.test_f1)
        self.log("test_ap", self.test_ap)
        self.log("test_iou", self.test_iou)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)

        return {"loss": loss, "pred": pred_prob, "target": y, "meta": meta}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
