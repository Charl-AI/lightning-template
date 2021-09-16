import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any
import torch
import wandb


class WandbImageClassificationCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices.

    Adapted from the example code at:
    https://docs.wandb.ai/guides/integrations/lightning
    """

    def __init__(self, num_samples=32):
        super().__init__()

        self.num_samples = num_samples

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called at the end of the first training batch of an epoch."""

        if batch_idx == 0:
            imgs = outputs["imgs"][: self.num_samples]
            targets = outputs["targets"][: self.num_samples]
            preds = torch.argmax(outputs["preds"][: self.num_samples], 1)

            trainer.logger.experiment.log(
                {
                    "val/examples": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(imgs, preds, targets)
                    ],
                    "global_step": trainer.global_step,
                }
            )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called at the end of the first validation batch of an epoch."""

        if batch_idx == 0:
            imgs = outputs["imgs"][: self.num_samples]
            targets = outputs["targets"][: self.num_samples]
            preds = torch.argmax(outputs["preds"][: self.num_samples], 1)

            trainer.logger.experiment.log(
                {
                    "train/examples": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(imgs, preds, targets)
                    ],
                    "global_step": trainer.global_step,
                }
            )
