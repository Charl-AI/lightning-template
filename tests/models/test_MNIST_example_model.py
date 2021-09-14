from tests.test_suite_utils.local_test import local_test
from tests.test_suite_utils.dummy_classification_data import (
    DummyClassificationDataModule,
)
from src.models.MNIST_example_model import MNISTResNetModule
from src.datasets.MNIST_example_data import MNISTDataModule
from tests.test_suite_utils.model_test_utils import (
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


@local_test
def test_MNIST_example_by_overfitting():
    model = MNISTResNetModule()
    data = MNISTDataModule(batch_size=2, download=True)
    overfit_batch(
        model, data, param_to_monitor="train_acc", target_value=0.9, max_epochs=15
    )
