"""Shared pytest configuration and fixtures."""

import pytest

pytest_plugins = ["tests.support.slow_tier"]


@pytest.fixture
def copper_torch_kernel(monkeypatch):
    """Keep historical copper pins on their original arithmetic (the Torch copper step).

    In-process runs use the oracle directly; fresh-interpreter BITID probes apply it
    themselves (``copper_torch.apply()``) on both the baseline and the working tree.
    """
    from tests.oracles import copper_torch
    copper_torch.install(monkeypatch)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA device")
    config.addinivalue_line('markers',
        'host_validation: validates metadata or rejects invalid input without CUDA allocation')
    # Tests use experimental objects on purpose; the warning itself is tested in
    # tests/api/public/test_public_api_stability.py. A configuration filter also holds
    # under `python -W error`; `pytest -W error` takes precedence over it.
    config.addinivalue_line("filterwarnings", "ignore::witwin.maxwell.errors.ExperimentalWarning")


def pytest_addoption(parser):
    try:
        parser.addoption("--gpu", action="store_true", default=False,
                         help="Run GPU-only tests (FDFD/FDTD cross-validation)")
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --gpu flag is passed."""
    import torch
    if not config.getoption("--gpu") or not torch.cuda.is_available():
        skip = __import__('pytest').mark.skip(reason="needs --gpu flag and CUDA")
        for item in items:
            # Cross-validation tests require GPU
            if "test_fdfd_vs_fdtd" in str(item.fspath):
                item.add_marker(skip)
