"""Central physical constants and torch dtype policy for ``witwin.maxwell``.

Vacuum constants follow CODATA 2018. ``EPSILON_0`` is shared with
``witwin.core`` so material compilation and solver runtimes agree bit-exactly.

Every constant here is the CODATA-recommended decimal literal, including
``ETA_0``. Do not "derive" ``ETA_0`` as ``MU_0 * C_0`` or ``1 / (EPSILON_0 *
C_0)``: ``MU_0`` and ``EPSILON_0`` are themselves only quoted to twelve
significant digits, so either product reproduces the recommended impedance only
to ``376.730313667`` -- a 3e-12 relative shift away from the recommended
``376.730313668``. That shift silently moves every source normalization,
modal impedance and far-field constant in the package.

Note: the ``*_reference.py`` oracle modules under ``fdtd/`` intentionally keep
independently written constant literals and must not import from this module.
"""

from __future__ import annotations

import torch
from witwin.core.material import VACUUM_PERMITTIVITY

# Vacuum permittivity [F/m] (CODATA 2018), shared with witwin.core.
EPSILON_0: float = VACUUM_PERMITTIVITY

# Vacuum permeability [H/m] (CODATA 2018).
MU_0: float = 1.25663706212e-6

# Speed of light in vacuum [m/s] (exact by SI definition).
C_0: float = 299792458.0

# Vacuum wave impedance [ohm] (CODATA 2018 recommended value, not a product of
# the truncated MU_0/EPSILON_0 literals -- see the module docstring).
ETA_0: float = 376.730313668


_REAL_FOR_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_COMPLEX_FOR_REAL = {
    torch.float16: torch.complex64,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Canonical real dtype paired with ``dtype``.

    Complex dtypes map to their real component dtype; real floating dtypes map
    to themselves.
    """
    real = _REAL_FOR_COMPLEX.get(dtype)
    if real is not None:
        return real
    if dtype.is_floating_point:
        return dtype
    raise TypeError(f"No real floating dtype policy for {dtype}.")


def complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Canonical complex dtype paired with ``dtype``.

    Real floating dtypes map to ``complex64``/``complex128``; complex dtypes
    map to themselves.
    """
    if dtype.is_complex:
        return dtype
    complex_dtype = _COMPLEX_FOR_REAL.get(dtype)
    if complex_dtype is not None:
        return complex_dtype
    raise TypeError(f"No complex dtype policy for {dtype}.")


def resolve_real_dtype(*values) -> torch.dtype:
    """Real dtype of the first tensor argument; ``float64`` when none is given."""
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.real.dtype
    return torch.float64


def resolve_complex_dtype(*values) -> torch.dtype:
    """Complex dtype of the first tensor argument; ``complex128`` when none is given."""
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.dtype if torch.is_complex(value) else complex_dtype_for(value.dtype)
    return torch.complex128
