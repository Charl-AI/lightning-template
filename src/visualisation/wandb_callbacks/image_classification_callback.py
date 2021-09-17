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

    def __init__(self, data: pl.LightningDataModule, num_samples=32):
        """
        Args:
            data (pl.LightningDataModule): Data to use for plots.
            num_samples (int, optional): Maximum number of samples to display. Defaults to 32.
        """
        super().__init__()

        data.prepare_data()
        data.setup()
        train_samples = next(iter(data.train_dataloader()))
        val_samples = next(iter(data.val_dataloader()))

        self.num_samples = num_samples
        if self.num_samples > data.batch_size:
            self.num_samples = data.batch_size

        self.train_imgs, self.train_labels = train_samples
        self.val_imgs, self.val_labels = val_samples

    def _log_examples(self, trainer, imgs, preds, labels, mode="train"):

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

    def on_train_epoch_end(self, trainer, pl_module):
        imgs = self.train_imgs.to(device=pl_module.device)
        logits = pl_module(imgs).detach()
        preds = torch.argmax(logits, 1)
        labels = self.train_labels

        self._log_examples(trainer, imgs, preds, labels, mode="train")
        self._log_logits(trainer, logits, mode="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(imgs).detach()
        preds = torch.argmax(logits, 1)
        labels = self.val_labels

        self._log_examples(trainer, imgs, preds, labels, mode="validation")
        self._log_logits(trainer, logits, mode="validation")
