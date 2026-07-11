from __future__ import annotations

from . import core as _core
from .bridge import _FDTDGradientBridge
from .core import *  # noqa: F401,F403
from .dispatch import reverse_step
from .native import register_native_reverse_backends
from .reference import (
    reverse_step_bloch_python_reference,
    reverse_step_cpml_python_reference,
    reverse_step_dispersive_python_reference,
    reverse_step_standard_python_reference,
    reverse_step_tfsf,
    reverse_step_torch_vjp,
)

# Populate the native CUDA reverse-backend registry so ``auto`` mode prefers the
# fused native standard reverse step on a qualifying CUDA scene.
register_native_reverse_backends()


def __getattr__(name: str):
    return getattr(_core, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_core)))
