import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

seed_everything(1)


def check_data_range(
    DataModule: pl.LightningDataModule,
    data_range: tuple = (0, 1),
):
    """Raises an error if the data is not within a specified range. Usually [0,1]
    This is most useful to check that images have been properly scaled.

    Args:
        DataModule (pl.LightningDataModule): DataModule to test.
        data_range (tuple, optional): Desired range of data values. Defaults to (0, 1).
    """
    raise NotImplementedError("Sorry! Not yet implemented.")


def check_data_normality(DataModule: pl.LightningDataModule):
    """Check that the mean and std of the data are equal to [0,1].
    This is most useful to check imgs have been properly normalised.

    Args:
        DataModule (pl.LightningDataModule): DataModule to test.
    """
    raise NotImplementedError("Sorry! Not yet implemented.")


def check_for_invalid_values(DataModule: pl.LightningDataModule):
    """Check for NaN and inf values in the dataset. Raises an
    Exception if any are found.

    Args:
        DataModule (pl.LightningDataModule): DataModule to test.
    """
    raise NotImplementedError("Sorry! Not yet implemented.")
