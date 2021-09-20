import pytorch_lightning as pl
import torch
import wandb


class WandbImageClassificationCallback(pl.Callback):
    """Logs the input images and output predictions of a module, as well
    as a histogram of the logits.
    Predictions and labels are logged as class indices.

    Adapted from the example code at:
    https://docs.wandb.ai/guides/integrations/lightning
    """

    def __init__(self, num_samples=32):
        """
        Args:
            data (pl.LightningDataModule): Data to use for plots.
            num_samples (int, optional): Maximum number of samples to display. Defaults to 32.
        """
        super().__init__()

        self.num_samples = num_samples

    def _log_examples(self, trainer, imgs, logits, labels, mode="train"):
        batch_size = len(imgs)
        if self.num_samples > batch_size:
            self.num_samples = batch_size

        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                f"{mode}/examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        imgs[: self.num_samples],
                        preds[: self.num_samples],
                        labels[: self.num_samples],
                    )
                ],
                "global_step": trainer.global_step,
            }
        )

    def _log_logits(self, trainer, logits, mode="train"):
        flattened_logits = torch.flatten(logits)
        trainer.logger.experiment.log(
            {
                f"{mode}/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": trainer.global_step,
            }
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._log_examples(
            trainer,
            outputs["imgs"],
            outputs["logits"],
            outputs["targets"],
            mode="train",
        )
        self._log_logits(trainer, outputs["logits"], mode="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._log_examples(
            trainer,
            outputs["imgs"],
            outputs["logits"],
            outputs["targets"],
            mode="validation",
        )
        self._log_logits(trainer, outputs["logits"], mode="validation")
