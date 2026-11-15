"""benchmark/datagen scheduler: configure_environment keeps the caller's CUDA_HOME and build dir."""
import os
import sysconfig

from benchmark.datagen import scheduler

KEYS = ("CUDA_HOME", "WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR")


def test_configure_environment_keeps_caller_values(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", str(tmp_path / "build"))
    scheduler.configure_environment()
    assert os.environ["CUDA_HOME"] == str(tmp_path / "cuda")
    assert os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] == str(tmp_path / "build")


def test_configure_environment_fills_unset_values(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    scheduler.configure_environment()
    assert os.environ["CUDA_HOME"] == os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
    assert os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] == os.path.expanduser(
        "~/.cache/witwin-bc-fdtd-build"
    )


def test_configure_environment_fills_only_the_unset_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", raising=False)
    scheduler.configure_environment()
    assert os.environ["CUDA_HOME"] == str(tmp_path / "cuda")
    assert os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] == os.path.expanduser(
        "~/.cache/witwin-bc-fdtd-build"
    )
