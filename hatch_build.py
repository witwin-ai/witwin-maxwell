from __future__ import annotations

import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        del version
        if self.target_name != "wheel":
            return
        prebuilt_dir = Path(self.root) / "witwin" / "maxwell" / "fdtd" / "cuda" / "prebuilt"
        has_prebuilt_extension = any(prebuilt_dir.glob("witwin_maxwell_fdtd_cuda.*"))
        if has_prebuilt_extension:
            platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
            build_data["tag"] = f"py3-none-{platform_tag}"
            build_data["pure_python"] = False
