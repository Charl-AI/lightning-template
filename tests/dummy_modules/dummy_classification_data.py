import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional


class _DummyClassificationData(Dataset):
    """Dummy image classification data. The data consists of random images
    (with a Gaussian distribution) and the labels are generated by summing the
    pixel values and thresholding at 0"""

    def __init__(self, img_shape: tuple = (1, 28, 28), length: int = 10):
        """
        Args:
            img_shape (tuple, optional): Image shape (CxHxW). Defaults to (1, 100, 100).
            length (int, optional): Size of dataset to generate. Defaults to 10.
        """
        self.img_shape = img_shape
        self.length = length

        self.imgs = torch.randn((length, img_shape[0], img_shape[1], img_shape[2]))
        self.summed_imgs = torch.sum(self.imgs, dim=[1, 2, 3])

        self.targets = torch.where(
            self.summed_imgs > 0,
            torch.ones_like(self.summed_imgs),
            torch.zeros_like(self.summed_imgs),
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.imgs[index], self.targets[index]


class DummyClassificationDataModule(pl.LightningDataModule):
    """Dummy data for testing classification models"""

    def __init__(self, img_shape: tuple = (1, 28, 28), batch_size: int = 1):
        super().__init__()
        self.img_shape = img_shape
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            data = _DummyClassificationData(
                img_shape=self.img_shape, length=(2 * self.batch_size)
            )

            self.train, self.val = random_split(
                data, [self.batch_size, self.batch_size]
            )

        if stage == "test" or stage is None:
            self.test = _DummyClassificationData(
                img_shape=self.img_shape, length=(self.batch_size)
            )

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)