from __future__ import annotations

import os
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load


def _ensure_windows_build_tools_on_path() -> None:
    if os.name != "nt":
        return

    current_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    prefixes: list[str] = []
    vc_tools = os.environ.get("VCToolsInstallDir")
    if vc_tools:
        prefixes.append(str(Path(vc_tools) / "bin" / "Hostx64" / "x64"))
    vs_install = os.environ.get("VSINSTALLDIR")
    if vs_install:
        prefixes.append(
            str(Path(vs_install) / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja")
        )
    if not prefixes:
        return
    merged_path = os.pathsep.join([*prefixes, current_path])
    os.environ["PATH"] = merged_path
    os.environ["Path"] = merged_path


def source_root() -> Path:
    return Path(__file__).resolve().parent


def extension_sources() -> list[Path]:
    root = source_root()
    return [
        root / "extension.cpp",
        root / "kernels" / "common.cu",
        root / "kernels" / "magnetic.cu",
        root / "kernels" / "electric.cu",
        root / "kernels" / "boundary.cu",
        root / "kernels" / "projection.cu",
        root / "kernels" / "dispersive.cu",
        root / "kernels" / "sources.cu",
        root / "kernels" / "spectral.cu",
        root / "kernels" / "observers.cu",
        root / "kernels" / "adjoint.cu",
    ]


def build_extension(*, verbose: bool = False):
    _ensure_windows_build_tools_on_path()
    root = source_root()
    default_build_directory = Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda"
    build_directory = Path(os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", default_build_directory))
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="witwin_maxwell_fdtd_cuda",
        sources=[str(path) for path in extension_sources()],
        build_directory=str(build_directory),
        extra_include_paths=[str(root), str(root / "kernels")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
