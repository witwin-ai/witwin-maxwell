from pathlib import Path

import pytest
from torch.utils import cpp_extension

from witwin.maxwell.fdtd.cuda import build


@pytest.mark.parametrize("source", ["CUDA_HOME", "CUDA_PATH", "nvcc"])
def test_late_toolkit_discovery_updates_extension_cache(monkeypatch, tmp_path, source):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    toolkit = tmp_path / "toolkit"
    monkeypatch.setattr(build.shutil, "which", lambda name: str(toolkit / "bin" / "nvcc") if source == "nvcc" else None)
    if source != "nvcc":
        monkeypatch.setenv(source, str(toolkit))
    build._ensure_cuda_home_from_nvcc()
    assert Path(cpp_extension.CUDA_HOME) == toolkit


def test_missing_toolkit_reports_compilation_requirement(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="compilation needs a CUDA toolkit"):
        build._ensure_cuda_home_from_nvcc()
