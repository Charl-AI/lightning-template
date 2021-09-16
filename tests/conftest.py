import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--remote",
        action="store_true",
        default=False,
        help="ignore tests marked with local decorator (usually because they rely on excessive data or compute)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "local: mark test to run locally only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--remote"):

        skip_local = pytest.mark.skip(reason="skipped test with local decorator")
        for item in items:
            if "local" in item.keywords:
                item.add_marker(skip_local)
