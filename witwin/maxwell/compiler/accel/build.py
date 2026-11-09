from functools import lru_cache
import os
from pathlib import Path
import tempfile

from torch.utils.cpp_extension import load

from ...fdtd.cuda.build import _conda_torch_ldflags, _cuda_gencode_flags


SETUP_ABI_VERSION = "setup_abi_v6"


@lru_cache(maxsize=1)
def build_extension():
    root = Path(__file__).parent / "cuda"
    base = os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR",
                          str(Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda"))
    directory = Path(base) / "setup"
    directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name="witwin_maxwell_setup_abi_v6",
        sources=[str(root / name) for name in ("binding.cpp", "setup.cu", "poles.cu", "subpixel.cu", "coefficients.cu", "tables.cu", "mesh.cu", "bvh.cu")],
        build_directory=str(directory),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", *_cuda_gencode_flags()],
        extra_ldflags=_conda_torch_ldflags(),
    )
    (directory / "abi.txt").write_text(SETUP_ABI_VERSION + "\n")
    return module
