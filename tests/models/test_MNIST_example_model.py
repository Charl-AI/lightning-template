from tests.local_test import local_test
from tests.dummy_modules.dummy_segmentation_data import DummySegmentationDataModule
from tests.dummy_modules.dummy_classification_data import DummyClassificationDataModule
from src.models.MNIST_example_model import MNISTResNetModule
from tests.models.model_test_utils import (
    check_logits_range,
    check_training_params,
    overfit_batch,
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


# @local_test
# def test_MNIST_example_with_overfitting():
#     model = MNISTResNetModule()
#     data = # use actual data, not dummy data
#     overfit_batch()
