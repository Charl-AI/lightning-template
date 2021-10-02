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
from datasets.mnist_example_data import MNISTDataModule
import wandb

# dataloader workers get different seeds to prevent augmentations being repeated
seed_everything(1, workers=True)


def main(args):

    dict_args = vars(args)

    model = MNISTResNetModule(lr=dict_args["learning_rate"])
    data = MNISTDataModule(batch_size=dict_args["batch_size"])

    if args.title:
        title = f"{input('Title of run:' )}"
    else:
        title = ""

    # Get name of project root. Assumes structure of root/src/train.py
    root_name = os.path.basename(Path(__file__).resolve().parent.parent)

    wandb_logger = WandbLogger(
        name=f"{title}", log_model="all", project=f"{root_name}-logs"
    )
    wandb_logger.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[
            WandbImageClassificationCallback(num_samples=32),
            WandbDatasetHistogramCallback(),
        ],
    )
    trainer.fit(model, data)

    if args.alert:
        wandb.alert(title="Run Finished", text="Login to W&B to see results :)")


def add_program_specific_args(parent_parser):

    parser = parent_parser.add_argument_group("Program Arguments")
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Send E-mail alert when finished. Relys on Weights and Biases Alerts.",
    )

    parser.add_argument(
        "--title",
        action="store_true",
        help="Enable title prompt",
    )
    return parent_parser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = add_program_specific_args(parser)
    parser = MNISTResNetModule.add_model_specific_args(parser)
    parser = MNISTDataModule.add_dataset_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
