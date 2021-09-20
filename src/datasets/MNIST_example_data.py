import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    """LightningDataModule implementation of MNIST dataset"""

    def __init__(self, batch_size: int = 50):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        )

    def prepare_data(self):
        MNIST(root="data/", train=True, download=True)
        MNIST(root="data/", train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            data = MNIST(
                root="data/",
                train=True,
                transform=self.transforms,
            )

            n_val = int(len(data) * 0.2)
            n_train = len(data) - n_val
            self.train, self.val = random_split(data, [n_train, n_val])

        if stage == "test" or stage is None:
            self.test = data = MNIST(
                root="data/",
                train=False,
                transform=self.transforms,
            )

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNISTDataModule")
        parser.add_argument(
            "--batch_size",
            help="batch size",
            type=int,
            default=50,
        )
        return parent_parser
