"""Discretization and port controls for the planar layered-medium MPIE solver."""

from dataclasses import asdict, dataclass
import math


def _length(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite length in metres")


@dataclass(frozen=True)
class AdaptiveSweep:
    """Greedy rational sweep in Hz; tolerance bounds absolute S model changes.

    Start with nine linear samples, bisect the interval with largest nested-fit
    disagreement, and require independent midpoint validation before accepting.
    max_points includes validation solves. A 2000-point passivity check rejects
    a violating fit rather than changing the physical response.
    """

    f_min: float
    f_max: float
    tolerance: float = 1e-3
    max_points: int = 25

    def __post_init__(self):
        if not (math.isfinite(self.f_min) and math.isfinite(self.f_max) and
                1 <= self.f_min < self.f_max):
            raise ValueError("AdaptiveSweep needs finite 1 <= f_min < f_max in Hz")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("AdaptiveSweep tolerance must be finite and positive")
        if type(self.max_points) is not int or self.max_points < 11:
            raise ValueError("AdaptiveSweep max_points must be an integer >= 11")

    def seed_frequencies(self):
        return tuple(self.f_min + (self.f_max - self.f_min) * i / 8 for i in range(9))


@dataclass(frozen=True)
class LayeredSurfaceConfig:
    """Mesh and port controls for sheets, Box vias and vertical bridge ports in a LayerStack.

    ``bridge_width_m`` is the width of the metal strip that joins each
    LumpedPort's two terminals; the delta-gap source sits on its midpoint cut.
    Bridges are ideal/lossless, including when the physical sheets are lossy.
    It has no default because it is part of the port model. ``max_edge_length_m``
    caps the cell size; ``edge_cell_m`` grades cells down to that size at every
    sheet edge, port cut and breakpoint with growth ratio ``growth_ratio``.
    ``breakpoint_tolerance_m`` optionally clusters all x/y layout coordinates
    (including via rims and ports) on that SI lattice; z is always exact.
    Use at most one tenth the minimum feature; 5 nm repairs +/-2 nm RFIC jitter.
    ``sweep=AdaptiveSweep(...)`` selects adaptive sampling and owns the band;
    omit Simulation.mom frequency/frequencies in that case. Result.network
    contains solved samples; Result.surface_currents.sweep contains the model,
    dense NetworkData and passivity report.
    """

    bridge_width_m: float
    max_edge_length_m: float
    edge_cell_m: float | None = None
    growth_ratio: float = 1.5
    breakpoint_tolerance_m: float = 0.0
    sweep: AdaptiveSweep | None = None

    def __post_init__(self):
        if not math.isfinite(self.breakpoint_tolerance_m) or self.breakpoint_tolerance_m < 0:
            raise ValueError("breakpoint_tolerance_m must be finite and nonnegative")
        if self.sweep is not None and not isinstance(self.sweep, AdaptiveSweep):
            raise TypeError("sweep must be an AdaptiveSweep")
        _length("bridge_width_m", self.bridge_width_m)
        _length("max_edge_length_m", self.max_edge_length_m)
        if self.edge_cell_m is not None:
            _length("edge_cell_m", self.edge_cell_m)
            if self.edge_cell_m > self.max_edge_length_m:
                raise ValueError("edge_cell_m must not exceed max_edge_length_m")
        if type(self.growth_ratio) not in (int, float) or not 1 <= self.growth_ratio <= 1.5:
            raise ValueError("growth_ratio must lie in [1, 1.5]")

    def payload(self):
        return asdict(self)
