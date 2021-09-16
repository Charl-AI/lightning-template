import pytest


def test_example():
    # this test isn't very useful...
    assert True


# This test wont run on Actions CI
@pytest.mark.local
def test_example_local():
    assert True
