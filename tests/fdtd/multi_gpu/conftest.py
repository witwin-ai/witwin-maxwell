from __future__ import annotations

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nccl: multi-process NCCL transport conformance launched via torchrun",
    )


@pytest.fixture(scope="session")
def cuda_p2p_devices() -> tuple[torch.device, torch.device]:
    """Return two mutually peer-accessible CUDA devices or skip the test session."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("cuda:0 cannot directly access cuda:1")
    if not torch.cuda.can_device_access_peer(1, 0):
        pytest.skip("cuda:1 cannot directly access cuda:0")

    properties = tuple(torch.cuda.get_device_properties(device) for device in devices)
    signatures = {(item.name, item.major, item.minor) for item in properties}
    if len(signatures) != 1:
        pytest.skip("multi-GPU FDTD acceptance requires homogeneous CUDA devices")
    return devices


@pytest.fixture
def cuda_memory_cleanup(cuda_p2p_devices):
    """Synchronize and release allocator caches around an isolated GPU test."""

    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    yield
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
