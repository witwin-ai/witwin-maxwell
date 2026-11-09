"""Public contracts for native CUDA frequency-domain local responses."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class LocalDesignRegion:
    """Fixed physical envelope of all material edits, including interpolation halo.

    Bounds are in metres. Selection includes an edge whenever either endpoint
    is in the envelope. Explicit global electric indices can instead specify a
    sparse design support. Indices follow Ex, Ey, Ez, each in C order.
    """
    bounds: tuple | None = None
    indices: torch.Tensor | None = None
    components: tuple[str, ...] = ("Ex", "Ey", "Ez")

    def __post_init__(self):
        if (self.bounds is None) == (self.indices is None):
            raise ValueError("Specify exactly one of design bounds or indices.")
        if self.bounds is not None:
            if len(self.bounds) != 3 or any(len(pair) != 2 or not all(math.isfinite(v) for v in pair)
                                             or pair[0] > pair[1] for pair in self.bounds):
                raise ValueError("Design bounds require three finite ordered pairs.")
        if not self.components or any(c not in ("Ex", "Ey", "Ez") for c in self.components):
            raise ValueError("Design components must be Ex, Ey or Ez.")


@dataclass(frozen=True)
class FieldProjection:
    """One linear complex field observable, without implicit conjugation.

    A point interpolates the native Yee locations. Weighted indices refer to
    the concatenated electric field. Weights may represent a voltage integral,
    a modal overlap, or another explicitly normalized linear measurement.
    """
    name: str
    component: str = "Ez"
    position: tuple[float, float, float] | None = None
    indices: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    path: tuple | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Projection name must be nonempty.")
        if sum(v is not None for v in (self.position, self.indices, self.path)) != 1:
            raise ValueError("Specify a point, weighted indices, or a voltage path.")
        if self.position is not None:
            if self.component not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                raise ValueError("Projection component must be an electric or magnetic field.")
            if len(self.position) != 3 or not all(math.isfinite(v) for v in self.position):
                raise ValueError("Projection position must be a finite 3-vector.")
        elif self.path is not None:
            _validate_path(*self.path)
        elif self.weights is None:
            raise ValueError("Indexed projections require weights.")

    @classmethod
    def voltage(cls, name, start, end):
        """Voltage rise from start to end, -integral(E dot dl), in volts."""
        return cls(name=name, path=(tuple(start), tuple(end)))


def _validate_path(start, end):
    if len(start) != 3 or len(end) != 3 or not all(math.isfinite(v) for v in (*start, *end)):
        raise ValueError("Path endpoints must be finite 3-vectors.")
    if sum(a != b for a, b in zip(start, end)) != 1:
        raise ValueError("Current and voltage paths must be nonzero and axis-aligned.")


@dataclass(frozen=True)
class LumpedCurrentSource:
    """Prescribed phasor current along an axis-aligned Yee gap, in amperes.

    Endpoints snap to grid nodes. Transverse support is one dual control area;
    current density is normalized by that area on nonuniform grids. Use a
    matching FieldProjection.voltage to compute input impedance and reflection.
    Phasors multiply exp(-i omega t).
    """
    start: tuple
    end: tuple
    current: complex = 1.0
    name: str = "feed"

    def __post_init__(self):
        _validate_path(self.start, self.end)
        if not math.isfinite(complex(self.current).real) or not math.isfinite(complex(self.current).imag):
            raise ValueError("Current amplitude must be finite.")


@dataclass(frozen=True)
class FarFieldProjection:
    """Transverse far-field amplitude r*exp(-ikr)*E in a vacuum exterior.

    The enclosing box must lie inside the physical domain, enclose all sources
    and scatterers, and have vacuum on its faces. Bounds snap to grid nodes.
    direction and polarization are unit vectors and must be perpendicular.
    """
    name: str
    bounds: tuple
    direction: tuple
    polarization: tuple

    def __post_init__(self):
        if not self.name or len(self.bounds) != 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) or p[0] >= p[1] for p in self.bounds):
            raise ValueError("Far-field bounds require three finite increasing pairs and a name.")
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v) or abs(sum(x*x for x in v)-1) > 1e-6
               for v in (self.direction, self.polarization)):
            raise ValueError("Far-field direction and polarization must be unit 3-vectors.")
        if abs(sum(a*b for a,b in zip(self.direction,self.polarization))) > 1e-6:
            raise ValueError("Far-field polarization must be transverse to direction.")


@dataclass(frozen=True)
class LocalResponseConfig:
    """GPU precomputation settings; stored with the FDFD Simulation.

    The memory budget covers persistent dense response arrays and blocked RHS
    storage, excluding the background sparse factorization and solver workspace.
    No host factor storage is enabled by this workflow.
    """
    design: LocalDesignRegion
    observations: tuple[FieldProjection | FarFieldProjection, ...]
    rhs_batch_size: int = 32
    max_response_bytes: int = 2**30
    tolerance: float = 1e-6
    rebuild_interval: int = 128

    def __post_init__(self):
        if not isinstance(self.design, LocalDesignRegion):
            raise TypeError("design must be a LocalDesignRegion.")
        if not self.observations or any(not isinstance(p, (FieldProjection, FarFieldProjection)) for p in self.observations):
            raise ValueError("At least one FieldProjection is required.")
        if len({p.name for p in self.observations}) != len(self.observations):
            raise ValueError("Projection names must be unique.")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.rhs_batch_size, self.max_response_bytes, self.rebuild_interval)):
            raise ValueError("Batch size, memory budget and rebuild interval must be positive integers.")
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("Local response tolerance must lie between zero and one.")
