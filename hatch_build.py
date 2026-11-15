from __future__ import annotations

import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


# The FDTD stepping (MAIN), core path (CORE) and accelerated setup (SETUP) libraries
# (witwin/maxwell/fdtd/cuda/build.py LIBRARIES). FDTD needs all three; there is no
# other execution path to fall back on, so a wheel ships all of them or none.
PREBUILT_LIBRARIES = ("witwin_maxwell_fdtd_cuda", "witwin_maxwell_fdtd_core", "witwin_maxwell_setup")


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        del version
        if self.target_name != "wheel":
            return
        prebuilt_dir = Path(self.root) / "witwin" / "maxwell" / "fdtd" / "cuda" / "prebuilt"
        present = [name for name in PREBUILT_LIBRARIES
                   if any(path.suffix in (".so", ".pyd") for path in prebuilt_dir.glob(f"{name}.*"))]
        if present and len(present) != len(PREBUILT_LIBRARIES):
            missing = sorted(set(PREBUILT_LIBRARIES) - set(present))
            raise RuntimeError(f"Incomplete prebuilt FDTD CUDA libraries in {prebuilt_dir}: missing {missing}; "
                               "run scripts/build_fdtd_cuda_prebuilt.py to build all three.")
        if present:
            platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
            build_data["tag"] = f"py3-none-{platform_tag}"
            build_data["pure_python"] = False
