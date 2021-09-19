from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import os
from visualisation.wandb_callbacks.image_classification_callback import (
    WandbImageClassificationCallback,
)
from models.MNIST_example_model import MNISTResNetModule
from datasets.MNIST_example_data import MNISTDataModule

seed_everything(1)


def main(args):

    dict_args = vars(args)

    model = MNISTResNetModule(lr=dict_args["learning_rate"])
    data = MNISTDataModule(batch_size=dict_args["batch_size"])

    wandb_logger = WandbLogger(name=f"{input('Title of run:' )}", log_model="all")
    wandb_logger.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[WandbImageClassificationCallback(data, num_samples=32)],
    )
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = MNISTResNetModule.add_model_specific_args(parser)
    parser = MNISTDataModule.add_dataset_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
