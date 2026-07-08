"""Backend dispatcher used by the CUDA parity tests.

The production backend (``witwin.maxwell.fdtd.cuda.backend``) only ever runs the
compiled CUDA kernels. To keep comparing those kernels against an independent
Torch reference, the parity tests route their calls through this dispatcher: it
forwards to the frozen Torch reference when
``WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=0`` and to the compiled backend
otherwise, preserving the historical ``monkeypatch.setenv(...)`` toggle used by
the tests.
"""

from __future__ import annotations

import os

from witwin.maxwell.fdtd.cuda import backend as _compiled
from tests.fdtd.cuda import torch_reference as _torch


def _use_torch_reference() -> bool:
    return os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1") == "0"


class _Dispatch:
    def __getattr__(self, name: str):
        source = _torch if _use_torch_reference() else _compiled
        return getattr(source, name)


backend = _Dispatch()
