from functools import lru_cache

from ...fdtd.cuda import build as cuda_build


SETUP_ABI_VERSION = cuda_build.SETUP.abi


@lru_cache(maxsize=1)
def build_extension():
    return cuda_build.build_extension(library=cuda_build.SETUP)
