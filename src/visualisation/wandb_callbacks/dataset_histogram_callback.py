import pytorch_lightning as pl
import torch
import wandb


class WandbDatasetHistogramCallback(pl.Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch

            trainer.logger.experiment.log(
                {
                    f"Dataset/Inputs:{str(x.shape)}": wandb.Histogram(x),
                    "global_step": trainer.global_step,
                }
            )

            trainer.logger.experiment.log(
                {
                    f"Dataset/Targets:{str(y.shape)}": wandb.Histogram(y),
                    "global_step": trainer.global_step,
                }
            )
