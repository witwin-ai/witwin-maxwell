from __future__ import annotations


def test_native_cuda_backend_reports_availability_without_build_side_effects(monkeypatch):
    from witwin.maxwell.fdtd.cuda import backend

    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", None)
    info = backend.build_info()

    assert isinstance(backend.is_available(), bool)
    assert info["backend"] == "torch-cuda"
    assert info["compiled_extension_loaded"] is False
