# Testing files go here

Pytest is used for testing. Each file must be named ```test_*```, with all functions following the same convention. Tests are run on Ubuntu using GitHub actions by default, but should also be run locally (vscode has a nice GUI for it).

Oftentimes tests require datasets or excessive compute, which is not practical with GitHub actions. In these cases use the ``` @local_test ``` decorator to signal to GitHub actions to skip the test.

Tests look like this:

```python
from tests.local_test import local_test


def test_example():
    # this test isn't very useful...
    assert True

# This test could be used for ensuring a model trains ok, it will be skipped when running remotely
@local_test
def test_lit_classifier():
    seed_everything(1234)

    model = LitClassifier()
    train, val, test = mnist()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train, val)

    results = trainer.test(test_dataloaders=test)
    assert results[0]['test_acc'] > 0.7

```
