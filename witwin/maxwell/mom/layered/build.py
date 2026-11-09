from functools import lru_cache
import os
from pathlib import Path

from torch.utils.cpp_extension import load

from ...fdtd.cuda.build import _conda_torch_ldflags, _cuda_gencode_flags


LAYERED_ABI_VERSION = "mom_layered_abi_v1"


@lru_cache(maxsize=1)
def build_extension():
    root = Path(__file__).parent / "cuda"
    directory = Path(os.environ.get("WITWIN_MAXWELL_MOM_LAYERED_BUILD_DIR",
                                    "/tmp/witwin-build-rfic-mom"))
    directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name="witwin_maxwell_mom_layered",
        sources=[str(root / name) for name in
                 ("binding.cpp", "spectral.cu", "bessel.cu", "interp.cu", "assembly.cu", "vertical.cu",
                  "lowfreq.cu")],
        build_directory=str(directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", *_cuda_gencode_flags()],
        extra_ldflags=_conda_torch_ldflags(),
    )
    (directory / "abi.txt").write_text(LAYERED_ABI_VERSION + "\n")
    return module
