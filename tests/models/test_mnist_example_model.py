import pytest

from tests.test_suite_utils.dummy_classification_data import (
    DummyClassificationDataModule,
)
from src.models.mnist_example_model import MNISTResNetModule
from src.datasets.mnist_example_data import MNISTDataModule
from tests.test_suite_utils.model_test_utils import (
    check_logits_range,
    check_training_params,
    overfit_batch,
    check_for_batch_mixing,
)


def test_MNIST_example_model_logits():
    model = MNISTResNetModule()
    data = DummyClassificationDataModule(
        img_shape=(1, 28, 28), batch_size=2, classes=10
    )
    check_logits_range(model, data, logits_range=(-10, 10), enforce_less_than_zero=True)


def test_MNIST_example_training_params():
    model = MNISTResNetModule()
    data = DummyClassificationDataModule(
        img_shape=(1, 28, 28), batch_size=2, classes=10
    )
    check_training_params(model, data)


def test_MNIST_example_batch_mixing():
    model = MNISTResNetModule().net
    data = DummyClassificationDataModule(
        img_shape=(1, 28, 28), batch_size=3, classes=10
    )
    check_for_batch_mixing(model, data)


@pytest.mark.local
def test_MNIST_example_by_overfitting():
    model = MNISTResNetModule()
    data = MNISTDataModule(batch_size=2)
    overfit_batch(
        model, data, param_to_monitor="train/accuracy", target_value=0.9, max_epochs=15
    )
