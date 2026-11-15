"""Typed CPML absorber configuration of ``FDTDConfig``."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from ._enums import StrEnum, parse_enum
from .errors import SetupError


class CPMLMemoryMode(StrEnum):
    """Storage of the CPML auxiliary ``psi`` fields.

    ``AUTO`` picks ``DENSE`` (full-grid tensors) when that storage is at most
    ``CPMLConfig.dense_memory_limit_mib`` and ``SLAB`` (the PML layers only)
    above it, from the scene alone. Dense and slab storage round differently.
    """

    AUTO = "auto"
    DENSE = "dense"
    SLAB = "slab"


def _finite(name, value, *, minimum, inclusive=True):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"CPMLConfig.{name} must be a real number, got {value!r}.") from exc
    if not math.isfinite(number) or number < minimum or (not inclusive and number == minimum):
        bound = f">= {minimum}" if inclusive else f"> {minimum}"
        raise SetupError(f"CPMLConfig.{name} must be finite and {bound}, got {value!r}.")
    return number


@dataclass(frozen=True)
class CPMLConfig:
    """Grading and storage of the FDTD CPML absorber (``absorber="cpml"``/``"stablepml"``).

    The conductivity ramp is ``sigma_max * u**grading_order`` over the normalized
    layer depth ``u``, with ``sigma_max`` set by the nominal normal-incidence
    ``reflection``; ``kappa_max`` stretches the layer and ``alpha_max`` is the
    complex-frequency shift (S/m).

    ``None`` takes the solver default. A real-field run uses ``grading_order=3``,
    ``alpha_max=0`` and ``reflection=1e-6``. A complex fixed-k Bloch run with an
    unset ``alpha_max`` shifts the CPML at the lowest requested frequency and
    then defaults ``grading_order`` to 2 and ``reflection`` to an angle-dependent
    target; an explicit value of any field is kept. ``absorber="stablepml"``
    replaces the four grading fields with its own stable profile and keeps the
    memory fields.
    """

    grading_order: float | None = None
    kappa_max: float = 1.0
    alpha_max: float | None = None
    reflection: float | None = None
    memory_mode: CPMLMemoryMode = CPMLMemoryMode.AUTO
    dense_memory_limit_mib: float = 64.0

    def __post_init__(self):
        from ._fixed_inputs import require_fixed_input

        for item in fields(self):
            require_fixed_input(getattr(self, item.name), name="CPMLConfig." + item.name)
        set_field = object.__setattr__
        if self.grading_order is not None:
            set_field(self, "grading_order", _finite("grading_order", self.grading_order, minimum=0.0, inclusive=False))
        set_field(self, "kappa_max", _finite("kappa_max", self.kappa_max, minimum=1.0))
        if self.alpha_max is not None:
            set_field(self, "alpha_max", _finite("alpha_max", self.alpha_max, minimum=0.0))
        if self.reflection is not None:
            reflection = _finite("reflection", self.reflection, minimum=0.0, inclusive=False)
            if reflection >= 1.0:
                raise SetupError(f"CPMLConfig.reflection must be below 1, got {self.reflection!r}.")
            set_field(self, "reflection", reflection)
        set_field(self, "memory_mode", parse_enum(CPMLMemoryMode, self.memory_mode, name="CPMLConfig.memory_mode"))
        set_field(self, "dense_memory_limit_mib",
                  _finite("dense_memory_limit_mib", self.dense_memory_limit_mib, minimum=0.0))


def require_cpml_config(value) -> CPMLConfig:
    """``value`` as a ``CPMLConfig``; ``None`` is the default configuration."""
    if value is None:
        return CPMLConfig()
    if not isinstance(value, CPMLConfig):
        raise SetupError(
            f"cpml must be a CPMLConfig instance, not {type(value).__name__}; "
            "the cpml_config dict was removed in 0.7.0, pass CPMLConfig(...) instead."
        )
    return value


__all__ = ["CPMLConfig", "CPMLMemoryMode", "require_cpml_config"]
