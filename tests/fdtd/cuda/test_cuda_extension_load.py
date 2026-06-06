from __future__ import annotations

import os

import pytest
import torch


def test_native_cuda_backend_reports_availability_without_build_side_effects(monkeypatch):
    from witwin.maxwell.fdtd.cuda import backend

    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", None)
    info = backend.build_info()

    assert isinstance(backend.is_available(), bool)
    assert info["backend"] == "torch-cuda"
    assert info["compiled_extension_loaded"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to build the native CUDA extension.")
@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_native_cuda_extension_builds_and_runs_debug_kernel():
    from witwin.maxwell.fdtd.cuda import backend

    extension = backend.get_compiled_extension()
    extension.synchronize_noop()
    linear, i_index, j_index, k_index = backend.debug_linear_indices((2, 3, 4), use_extension=True)
    expected = torch.arange(2 * 3 * 4, device="cuda", dtype=torch.int64).reshape(2, 3, 4)
    torch.cuda.synchronize()

    assert extension.is_available() is True
    torch.testing.assert_close(linear, expected)
    torch.testing.assert_close(i_index, expected // (3 * 4))
    torch.testing.assert_close(j_index, (expected // 4) % 3)
    torch.testing.assert_close(k_index, expected % 4)


def test_backend_selector_defaults_to_slang(monkeypatch):
    from witwin.maxwell.fdtd.runtime.module_cache import resolve_fdtd_backend_name

    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_BACKEND", raising=False)

    assert resolve_fdtd_backend_name() == "slang"


def test_backend_selector_accepts_cuda(monkeypatch):
    from witwin.maxwell.fdtd.runtime.module_cache import resolve_fdtd_backend_name

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")

    if not torch.cuda.is_available():
        assert resolve_fdtd_backend_name() == "cuda"
        return

    assert resolve_fdtd_backend_name() == "cuda"
