from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import tempfile

from torch.utils.cpp_extension import load

from ..build import (
    _conda_torch_ldflags,
    _cuda_gencode_flags,
    _ensure_cuda_home_from_nvcc,
    _ensure_windows_build_tools_on_path,
)


CORE_ABI_VERSION = "core_abi_v11"


@lru_cache(maxsize=1)
def build_extension():
    root = Path(__file__).parent
    base = os.environ.get(
        "WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR",
        str(Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda" / CORE_ABI_VERSION),
    )
    directory = Path(base) / "core"
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_windows_build_tools_on_path()
    _ensure_cuda_home_from_nvcc()
    module = load(
        name="witwin_maxwell_fdtd_core_abi_v11",
        sources=[str(root / name) for name in ("binding.cpp", "yee.cu", "step.cu", "observers.cu")],
        build_directory=str(directory),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", *_cuda_gencode_flags()],
        extra_ldflags=_conda_torch_ldflags(),
    )
    (directory / "abi.txt").write_text(CORE_ABI_VERSION + "\n")
    return module
