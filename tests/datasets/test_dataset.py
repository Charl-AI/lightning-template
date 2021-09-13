from tests.test_suite_utils.local_test import local_test

# this file contains an example test for the
# rename this file to match its custom dataset equivalent... e.g. test_tomography.py


def test_example():
    # this test isn't very useful...
    assert True


# This test wont run on Actions CI
@local_test
def test_example_local():
    assert False
