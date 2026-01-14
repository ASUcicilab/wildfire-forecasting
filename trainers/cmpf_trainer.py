"""CMPF Trainer - Progressive Fine-tuning with Intermediate Stage."""

import os
from typing import Any, Dict, Literal, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .base_trainer import BaseTrainer


class CMPFTrainer(BaseTrainer):
    """Trainer for Cross-Modal Progressive Fine-tuning.

    Implements the three CMPF strategies:
    - S1: Shared bands only during intermediate training
    - S2: Full bands with zero-padding
    - S3: Combined dataset training
    """

    def __init__(
        self,
        output_dir: str,
        strategy: Literal["S1", "S2", "S3"] = "S1",
        epochs_intermediate: int = 60,
        epochs_final: int = 60,
        lr_intermediate: float = 1e-4,
        lr_final: float = 1e-5,
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, **kwargs)

        self.strategy = strategy
        self.epochs_intermediate = epochs_intermediate
        self.epochs_final = epochs_final
        self.lr_intermediate = lr_intermediate
        self.lr_final = lr_final

    def train_intermediate(
        self,
        model: pl.LightningModule,
        aux_dataloader,
        val_dataloader=None,
    ) -> Dict[str, Any]:
        """Run intermediate fine-tuning on auxiliary dataset (NDWS).

        Args:
            model: Model to train
            aux_dataloader: DataLoader for auxiliary dataset (NDWS)
            val_dataloader: Optional validation DataLoader

        Returns:
            Dict with training results including best checkpoint path
        """
        print(f"\n{'=' * 60}")
        print(f"CMPF Intermediate Stage - Strategy {self.strategy}")
        print(f"Training for {self.epochs_intermediate} epochs")
        print(f"{'=' * 60}\n")

        # Update learning rate
        model.lr = self.lr_intermediate

        # Create callbacks for intermediate stage
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints" / "intermediate",
                filename="intermediate-{epoch:02d}-{val_ap:.4f}"
                if val_dataloader
                else "intermediate-{epoch:02d}",
                monitor="val_ap" if val_dataloader else "train_loss",
                mode="max" if val_dataloader else "min",
                save_top_k=1,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Create logger
        logger = self._create_logger(run_name=f"{self.wandb_run_name}_intermediate")

        # Create trainer
        trainer = self._create_trainer(
            max_epochs=self.epochs_intermediate,
            callbacks=callbacks,
            logger=logger,
        )

        # Train
        if val_dataloader:
            trainer.fit(
                model, train_dataloaders=aux_dataloader, val_dataloaders=val_dataloader
            )
        else:
            trainer.fit(model, train_dataloaders=aux_dataloader)

        return {
            "best_ckpt": trainer.checkpoint_callback.best_model_path,
            "last_ckpt": trainer.checkpoint_callback.last_model_path,
        }

    def train_final(
        self,
        model: pl.LightningModule,
        train_dataloader,
        val_dataloader,
        intermediate_ckpt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run final fine-tuning on target dataset (WTS).

        Args:
            model: Model to train
            train_dataloader: DataLoader for target dataset training split
            val_dataloader: DataLoader for target dataset validation split
            intermediate_ckpt: Path to intermediate checkpoint to load

        Returns:
            Dict with training results
        """
        print(f"\n{'=' * 60}")
        print("CMPF Final Stage")
        print(f"Training for {self.epochs_final} epochs")
        print(f"{'=' * 60}\n")

        # Load intermediate checkpoint if provided
        if intermediate_ckpt and os.path.exists(intermediate_ckpt):
            print(f"Loading intermediate checkpoint: {intermediate_ckpt}")
            checkpoint = torch.load(intermediate_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        # Update learning rate
        model.lr = self.lr_final

        # Create callbacks for final stage
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints" / "final",
                filename="final-{epoch:02d}-{val_ap:.4f}",
                monitor="val_ap",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Create logger
        logger = self._create_logger(run_name=f"{self.wandb_run_name}_final")

        # Create trainer
        trainer = self._create_trainer(
            max_epochs=self.epochs_final,
            callbacks=callbacks,
            logger=logger,
        )

        # Train
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        return {
            "best_ckpt": trainer.checkpoint_callback.best_model_path,
            "last_ckpt": trainer.checkpoint_callback.last_model_path,
            "best_score": trainer.checkpoint_callback.best_model_score,
        }

    def train_cmpf(
        self,
        model: pl.LightningModule,
        aux_dataloader,
        target_train_dataloader,
        target_val_dataloader,
        adapt_channels: bool = True,
    ) -> Dict[str, Any]:
        """Run complete CMPF training pipeline.

        Args:
            model: Model to train
            aux_dataloader: DataLoader for auxiliary dataset
            target_train_dataloader: DataLoader for target dataset training
            target_val_dataloader: DataLoader for target dataset validation
            adapt_channels: Whether to adapt input channels between stages

        Returns:
            Dict with all training results
        """
        results = {}

        # Stage 1: Intermediate fine-tuning
        intermediate_results = self.train_intermediate(
            model=model,
            aux_dataloader=aux_dataloader,
        )
        results["intermediate"] = intermediate_results

        # Adapt model channels if switching from shared to full bands
        if adapt_channels and self.strategy == "S1":
            # Model was trained on shared bands, now needs full bands
            # This should be handled by the experiment script
            pass

        # Stage 2: Final fine-tuning
        final_results = self.train_final(
            model=model,
            train_dataloader=target_train_dataloader,
            val_dataloader=target_val_dataloader,
            intermediate_ckpt=intermediate_results["best_ckpt"],
        )
        results["final"] = final_results

        return results


class DirectFinetuneTrainer(BaseTrainer):
    """Trainer for direct fine-tuning baseline (no intermediate stage)."""

    def __init__(self, output_dir: str, epochs: int = 120, **kwargs):
        super().__init__(output_dir=output_dir, max_epochs=epochs, **kwargs)

    def train_direct(
        self,
        model: pl.LightningModule,
        train_dataloader,
        val_dataloader,
        pretrained_ckpt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run direct fine-tuning on target dataset.

        Args:
            model: Model to train
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            pretrained_ckpt: Path to pretrained weights

        Returns:
            Dict with training results
        """
        print(f"\n{'=' * 60}")
        print("Direct Fine-tuning Baseline")
        print(f"Training for {self.max_epochs} epochs")
        print(f"{'=' * 60}\n")

        # Load pretrained weights if provided
        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            print(f"Loading pretrained weights: {pretrained_ckpt}")
            model.backbone.load_pretrained(pretrained_ckpt)

        return self.train(model, train_dataloader, val_dataloader)
