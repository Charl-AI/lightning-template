"""Model testing utils

These are useful classes and functions for testing ml models with Pytorch Lightining.
Insipred by:
https://thenerdstation.medium.com/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything


seed_everything(1)


def check_logits_range(
    LitModule: pl.LightningModule,
    DataModule: pl.LightningDataModule,
    logits_range: tuple = (-1, 1),
    enforce_less_than_zero: bool = True,
):
    """Feeds a batch into the model and checks the output range of the logits.
    Raises an error if there are values outside the range or NaN or inf values.

    Args:
        LitModule (pl.LightningModule): Model to test.
        DataModule (pl.LightningDataModule): DataModule to test with.
        logits_range (tuple, optional): Desired range of output logits. Defaults to (-1, 1).
        enforce_less_than_zero (bool, optional): If true, raises an error if there are no
            values less than zero in the logits. This could be caused by accidentally using ReLu.
            Defaults to True.
    """
    DataModule.setup()
    batch = next(iter(DataModule.train_dataloader()))
    logits = LitModule(batch[0])

    min_val = torch.min(logits)
    max_val = torch.max(logits)

    try:
        assert min_val > logits_range[0]
    except AssertionError:
        raise Exception(
            f"Minimum value, {min_val}, in the logits is less than {logits_range[0]}"
        )

    try:
        assert max_val < logits_range[1]
    except AssertionError:
        raise Exception(
            f"Maximum value, f{max_val}, in the logits is greater than {logits_range[1]}"
        )

    if enforce_less_than_zero:
        try:
            assert min_val < 0
        except AssertionError:
            raise Exception(
                f"Minimum value, f{min_val}, in the logits is greater than 0, did you accidentally use ReLu?"
            )
    try:
        assert not torch.isnan(logits).byte().any()
    except AssertionError:
        raise Exception("There was a NaN value in logits")

    try:
        assert torch.isfinite(logits).byte().any()
    except AssertionError:
        raise Exception("There was an Inf value in logits")


def check_training_params(
    LitModule: pl.LightningModule,
    DataModule: pl.LightningDataModule,
    training_params: list = None,
):
    """Runs a training step and checks whether the training parameters have changed.

    Args:
        LitModule (pl.LightningModule): Model to test.
        DataModule (pl.LightningDataModule): DataModule to test with.
        training_params (list, optional): List of parameters of form (name, variable)
        to check are being optimised. If None, all parameters with requires_grad will
        be checked. Defaults to None.
    """

    if training_params is None:
        # get a list of params that are allowed to change
        training_params = [
            np for np in LitModule.named_parameters() if np[1].requires_grad
        ]
    initial_params = [(name, p.clone()) for (name, p) in training_params]

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(LitModule, DataModule)

    for (_, p0), (name, p1) in zip(initial_params, training_params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            raise Exception(f"{name} was not changed during the training step")


def overfit_batch(
    LitModule: pl.LightningModule,
    DataModule: pl.LightningDataModule,
    max_epochs: int = 10,
):

    trainer = pl.Trainer(
        overfit_batches=1, max_epochs=max_epochs, weights_summary="full"
    )
    trainer.fit(LitModule, DataModule)
