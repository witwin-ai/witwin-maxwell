from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

import torch


def _ensure_current_device_arch() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    current_arch = f"{major}.{minor}"
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    entries = [entry.strip() for entry in arch_list.split(";") if entry.strip()]
    normalized = {entry.removesuffix("+PTX") for entry in entries}
    if current_arch not in normalized:
        entries.append(current_arch)
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(entries)


def _load_cuda_build_module():
    repo_root = Path(__file__).resolve().parents[1]
    build_path = repo_root / "witwin" / "maxwell" / "fdtd" / "cuda" / "build.py"
    spec = importlib.util.spec_from_file_location("witwin_maxwell_fdtd_cuda_build", build_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load FDTD CUDA build module from {build_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the packaged FDTD CUDA extension.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    build_dir = Path(
        os.environ.get(
            "WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR",
            Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda_wheel",
        )
    )
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(build_dir)
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT"] = "1"
    _ensure_current_device_arch()

    cuda_build = _load_cuda_build_module()
    module = cuda_build.build_extension(verbose=args.verbose)
    module_file = Path(module.__file__).resolve()

    target_dir = cuda_build.prebuilt_root()
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pyd", ".so"):
        existing = target_dir / f"witwin_maxwell_fdtd_cuda{suffix}"
        if existing.exists():
            existing.unlink()
    target = cuda_build.prebuilt_extension_path()
    shutil.copy2(module_file, target)
    print(f"Built prebuilt FDTD CUDA extension: {target}")


if __name__ == "__main__":
    main()
