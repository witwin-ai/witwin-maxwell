from __future__ import annotations

from . import core as _core
from .bridge import (
    _FDTDGradientBridge as _FDTDGradientBridge,
    run_fdtd_with_gradient_bridge as run_fdtd_with_gradient_bridge,
)
from .core import *  # noqa: F401,F403
from .dispatch import reverse_step as reverse_step
from .native import register_native_reverse_backends

# Populate the only production reverse implementation.
register_native_reverse_backends()


def __getattr__(name: str):
    return getattr(_core, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_core)))
