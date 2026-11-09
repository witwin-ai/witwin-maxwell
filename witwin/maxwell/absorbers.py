"""Declarative one-way absorbing boundary conditions.

``MurABC`` and ``ModeABC`` parameterize absorbing faces declared through
``BoundarySpec`` (face kinds ``"mur"`` and ``"mode_abc"``) and the one-way
update used by an ``InternalAbsorber`` plane placed inside the domain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ._fixed_inputs import require_fixed_input

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class MurABC:
    """Mur one-way wave-equation absorbing boundary.

    ``order=1`` is the first-order (normal-incidence) Mur update. ``order=2`` is
    the second-order Higdon product of two normal-incidence Mur operators: its
    reflection is the square of the first-order reflection (second-order
    accurate in the incidence angle, like the Engquist-Majda condition) without a
    transverse stencil, so it remains passive next to transverse PML, at edges
    and corners, and in media slower than the one-way speed.

    ``permittivity`` sets the relative permittivity that defines the one-way
    propagation speed ``c / sqrt(permittivity)``. ``None`` uses vacuum speed.
    """

    order: int = 1
    permittivity: float | None = None

    def __post_init__(self):
        require_fixed_input((self.order, self.permittivity), name="MurABC controls")
        if isinstance(self.order, bool) or int(self.order) != self.order or int(self.order) not in (1, 2):
            raise ValueError("MurABC.order must be 1 or 2.")
        object.__setattr__(self, "order", int(self.order))
        if self.permittivity is not None:
            permittivity = float(self.permittivity)
            if not math.isfinite(permittivity) or permittivity < 1.0:
                raise ValueError("MurABC.permittivity must be a finite relative permittivity >= 1.")
            object.__setattr__(self, "permittivity", permittivity)


@dataclass(frozen=True)
class ModeABC:
    """One-way absorbing boundary matched to a guided mode.

    The mode is solved on the boundary cross-section with the same eigen-solver
    as ``ModeSource``. ``frequency=None`` uses the central frequency of the
    primary source. The one-step update coefficient is chosen so the discrete
    outgoing Yee mode at that frequency is annihilated exactly.

    ``frequency_range=(f_min, f_max)`` requests a broadband termination: a
    Higdon product of ``order`` such mode-matched one-step operators, matched at
    Chebyshev nodes of the band. Its reflection is the product of the factor
    reflections, each of modulus at most one, so the boundary stays passive for
    every propagating wave while the band-edge reflection drops geometrically
    with ``order`` (``order`` is 1 or 2).
    """

    mode_index: int = 0
    polarization: str = "auto"
    frequency: float | None = None
    frequency_range: tuple[float, float] | None = None
    order: int = 2

    def __post_init__(self):
        require_fixed_input(
            (self.mode_index, self.frequency, self.frequency_range, self.order),
            name="ModeABC controls",
        )
        if isinstance(self.mode_index, bool) or int(self.mode_index) != self.mode_index or int(self.mode_index) < 0:
            raise ValueError("ModeABC.mode_index must be a non-negative integer.")
        object.__setattr__(self, "mode_index", int(self.mode_index))
        object.__setattr__(self, "polarization", str(self.polarization))
        if self.frequency is not None:
            frequency = float(self.frequency)
            if not math.isfinite(frequency) or frequency <= 0.0:
                raise ValueError("ModeABC.frequency must be positive.")
            object.__setattr__(self, "frequency", frequency)
        if self.frequency_range is not None:
            if self.frequency is not None:
                raise ValueError("ModeABC accepts either frequency or frequency_range, not both.")
            if len(self.frequency_range) != 2:
                raise ValueError("ModeABC.frequency_range must be (f_min, f_max).")
            low, high = (float(value) for value in self.frequency_range)
            if not (math.isfinite(low) and math.isfinite(high)) or low <= 0.0 or high <= low:
                raise ValueError("ModeABC.frequency_range must satisfy 0 < f_min < f_max.")
            object.__setattr__(self, "frequency_range", (low, high))
        # Products of three or more undamped mode factors were measured to grow at
        # late times between two opposite faces; order two is the stable maximum.
        if isinstance(self.order, bool) or int(self.order) != self.order or not 1 <= int(self.order) <= 2:
            raise ValueError("ModeABC.order must be 1 or 2.")
        object.__setattr__(self, "order", int(self.order))

    @property
    def is_broadband(self) -> bool:
        return self.frequency_range is not None


@dataclass(frozen=True)
class InternalAbsorber:
    """A one-way absorbing plane placed inside the simulation domain.

    Waves travelling along ``direction`` (``"+"`` or ``"-"`` along the plane
    normal) are absorbed at the plane: its tangential electric field is driven by
    the one-way update from the upstream side only, so the region behind the
    plane cannot feed back into the upstream region. The plane must span the
    whole domain cross-section (use ``inf`` for the transverse sizes).

    ``grid_shift`` moves the plane by whole cells along the normal (positive
    shifts move toward ``+``), which places an absorber directly behind a
    source sharing the same ``position``/``size``.
    """

    position: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: str = "+"
    boundary: MurABC | ModeABC = MurABC()
    grid_shift: int = 0
    name: str | None = None

    def __init__(self, position, size, direction="+", boundary=None, grid_shift=0, name=None):
        require_fixed_input((position, size, grid_shift), name="InternalAbsorber geometry")
        resolved_position = tuple(float(value) for value in position)
        resolved_size = tuple(float(value) for value in size)
        if len(resolved_position) != 3 or len(resolved_size) != 3:
            raise ValueError("InternalAbsorber position and size must have three components.")
        if any(not math.isfinite(value) for value in resolved_position):
            raise ValueError("InternalAbsorber position must be finite.")
        if any(value < 0.0 or math.isnan(value) for value in resolved_size):
            raise ValueError("InternalAbsorber size components must be >= 0.")
        zero_axes = [axis for axis, value in zip(_AXES, resolved_size) if value == 0.0]
        if len(zero_axes) != 1:
            raise ValueError("InternalAbsorber size must contain exactly one zero component (the plane normal).")
        resolved_direction = str(direction)
        if resolved_direction not in ("+", "-"):
            raise ValueError("InternalAbsorber.direction must be '+' or '-'.")
        resolved_boundary = MurABC() if boundary is None else boundary
        if not isinstance(resolved_boundary, (MurABC, ModeABC)):
            raise TypeError("InternalAbsorber.boundary must be a MurABC or ModeABC.")
        if isinstance(grid_shift, bool) or int(grid_shift) != grid_shift:
            raise ValueError("InternalAbsorber.grid_shift must be an integer.")
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "boundary", resolved_boundary)
        object.__setattr__(self, "grid_shift", int(grid_shift))
        object.__setattr__(self, "name", None if name is None else str(name))

    @property
    def normal_axis(self) -> str:
        return _AXES[self.size.index(0.0)]

    @property
    def direction_sign(self) -> int:
        return 1 if self.direction == "+" else -1


__all__ = ["InternalAbsorber", "ModeABC", "MurABC"]
