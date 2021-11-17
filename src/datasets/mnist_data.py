import os
import torch
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    """LightningDataModule implementation of MNIST dataset.

    Uses default train/test splits from the torchvision data class
    and also splits the train data into a train/val split with an
    80:20 ratio."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):

        # only download data if it hasn't been already
        if not os.path.isdir(self.data_dir + "MNIST"):
            MNIST(root=self.data_dir, train=True, download=True)
            MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            data = MNIST(
                root=self.data_dir,
                train=True,
                transform=self.transforms,
            )

            n_val = int(len(data) * 0.2)
            n_train = len(data) - n_val
            self.train, self.val = random_split(
                data, [n_train, n_val], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            self.test = data = MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--data_dir",
            help="directory containing 'MNIST/' data directory",
            type=str,
            default="data/",
        )
        parser.add_argument(
            "--batch_size",
            help="batch size",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--num_workers",
            help="number of dataloader workers. 4*num_gpus is usually fine",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--drop_last",
            help="whether to drop last batch from dataloader to keep batch sizes constant",
            type=bool,
            default=False,
        )
        return parent_parser
