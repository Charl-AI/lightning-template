from tests.local_test import local_test


def test_example():
    # this test isn't very useful...
    assert True


# This test wont run on Actions CI
@local_test
def test_example_local():
    assert True
