from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import os
from pathlib import Path
from visualisation.wandb_callbacks.image_classification_callback import (
    WandbImageClassificationCallback,
)
from visualisation.wandb_callbacks.dataset_histogram_callback import (
    WandbDatasetHistogramCallback,
)
from models.mnist_example_model import MNISTResNetModule
from datasets.mnist_data import MNISTDataModule

# dataloader workers get different seeds to prevent augmentations being repeated
seed_everything(1, workers=True)


def main(args):

    model = MNISTResNetModule(lr=args.learning_rate)
    data = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    callbacks = []

    if args.logger == True:
        # Get name of project root. Assumes structure of root/src/train.py
        root_name = os.path.basename(Path(__file__).resolve().parent.parent)

        wandb_logger = WandbLogger(log_model=False, project=f"{root_name}-logs")
        wandb_logger.watch(model)

        logger = wandb_logger
        callbacks.extend(
            [
                WandbImageClassificationCallback(num_samples=32),
                WandbDatasetHistogramCallback(),
            ]
        )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = MNISTResNetModule.add_model_specific_args(parser)
    parser = MNISTDataModule.add_dataset_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
