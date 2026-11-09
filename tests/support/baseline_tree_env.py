"""Child-process environments for fresh-interpreter bit-identity probes.

Bit-identity tests run one probe on a ``git archive`` baseline tree and one on the
current tree. The baseline tree carries its own CUDA sources, so it must JIT-build
its own native module: the caller's native-build selection (a prebuilt image, a
skipped packaged prebuilt, or a build directory) describes the current tree and
must not leak into the baseline child. The current-tree child keeps the caller's
environment unchanged, so it loads exactly the native image under test.
"""
from __future__ import annotations

import os
from pathlib import Path

# Native-build selection read by ``witwin/maxwell/fdtd/cuda/build.py``.
NATIVE_BUILD_VARIABLES = (
    "WITWIN_MAXWELL_FDTD_CUDA_PREBUILT",
    "WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT",
    "WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR",
)


def baseline_build_directory(tree: Path) -> Path:
    """Private JIT build directory of one extracted baseline tree (beside it, not inside it)."""
    tree = Path(tree)
    return tree.parent / f"{tree.name}_native_build"


def probe_environment(tree: Path, *, baseline: bool) -> dict[str, str]:
    """Environment for a probe importing ``witwin`` from ``tree``.

    ``baseline=True`` drops the caller's native-build variables and points the
    build at ``baseline_build_directory(tree)``; ``baseline=False`` keeps them.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tree)
    if baseline:
        for name in NATIVE_BUILD_VARIABLES:
            environment.pop(name, None)
        environment["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(baseline_build_directory(tree))
    return environment
