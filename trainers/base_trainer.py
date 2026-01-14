"""Base trainer for CMPF experiments."""

from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class BaseTrainer:
    """Base trainer class for all experiments."""

    def __init__(
        self,
        output_dir: str,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "16-mixed",
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        val_check_interval: float = 1.0,
        log_every_n_steps: int = 50,
        use_wandb: bool = True,
        wandb_project: str = "cmpf-wildfire",
        wandb_run_name: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.seed = seed

        # Set seed
        pl.seed_everything(seed)

    def _create_callbacks(self, monitor: str = "val_ap", mode: str = "max") -> list:
        """Create training callbacks."""
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename="best-{epoch:02d}-{val_ap:.4f}",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=True,
            ),
            EarlyStopping(
                monitor=monitor,
                patience=20,
                mode=mode,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        return callbacks

    def _create_logger(self, run_name: Optional[str] = None):
        """Create experiment logger."""
        if self.use_wandb:
            return WandbLogger(
                project=self.wandb_project,
                name=run_name or self.wandb_run_name,
                save_dir=str(self.output_dir),
            )
        else:
            return TensorBoardLogger(
                save_dir=str(self.output_dir),
                name="logs",
            )

    def _create_trainer(
        self,
        max_epochs: Optional[int] = None,
        callbacks: Optional[list] = None,
        logger=None,
        **kwargs,
    ) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        return pl.Trainer(
            default_root_dir=str(self.output_dir),
            max_epochs=max_epochs or self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_grad_batches,
            val_check_interval=self.val_check_interval,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=callbacks or self._create_callbacks(),
            logger=logger or self._create_logger(),
            **kwargs,
        )

    def train(
        self,
        model: pl.LightningModule,
        train_dataloader,
        val_dataloader,
        ckpt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training loop."""
        trainer = self._create_trainer()

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
        )

        # Return best checkpoint path and metrics
        return {
            "best_ckpt": trainer.checkpoint_callback.best_model_path,
            "best_score": trainer.checkpoint_callback.best_model_score,
        }

    def test(
        self,
        model: pl.LightningModule,
        test_dataloader,
        ckpt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run test evaluation."""
        trainer = self._create_trainer(max_epochs=0)

        results = trainer.test(
            model,
            dataloaders=test_dataloader,
            ckpt_path=ckpt_path,
        )

        return results[0] if results else {}

    def finish(self):
        """Clean up after training."""
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
