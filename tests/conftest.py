"""
Configures pytest to ignore unit tests marked as expensive unless we use the --expensive flag.
For example, training an RL algorithm for 100,000s of timesteps would be too expensive to include
in the usual CI run, but is useful to run manually periodically or after making high-risk changes.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--expensive", action="store_true",
                     help="run expensive tests (which are otherwise skipped).")


def pytest_runtest_setup(item):
    if 'expensive' in item.keywords and not item.config.getoption("--expensive"):
        pytest.skip("Skipping test unless --expensive is flagged")
