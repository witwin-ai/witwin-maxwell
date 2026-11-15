"""Discretization and port controls for the planar layered-medium MPIE solver."""

from dataclasses import dataclass
import math

from ..._enums import parse_enum
from ..._stability import experimental
from ..options import MoMOption, option_asdict


@experimental("mom_layered")
class LayeredOperatorBackend(MoMOption):
    """Layered MPIE operator backend selected by ``LayeredSurfaceConfig.operator_backend``."""
    DENSE = "dense"
    MATRIX_FREE = "matrix_free"
    HMATRIX = "hmatrix"


@experimental("mom_layered")
class LayeredConductorModel(MoMOption):
    """Conductor model selected by ``LayeredSurfaceConfig.conductor_model``."""
    AUTO = "auto"
    SHEET = "sheet"
    CLOSED = "closed"
    SURFACE_ADMITTANCE = "surface_admittance"


@experimental("mom_layered")
class PlanarMesher(MoMOption):
    """Planar mesher selected by ``LayeredSurfaceConfig.planar_mesher``."""
    AUTO = "auto"
    TRIANGLE = "triangle"


def _length(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite length in metres")


@experimental("mom_layered")
@dataclass(frozen=True)
class AdaptiveSweep:
    """Greedy rational sweep in Hz; tolerance bounds absolute S model changes.

    Start with nine linear samples, bisect the interval with largest nested-fit
    disagreement, and require independent midpoint validation before accepting.
    max_points includes validation solves. A fit that fails the 2000-point
    passivity check is enforced (Hamiltonian residue perturbation) if that moves
    it by <= tolerance/4, else rejected; the solved samples never change.
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


@experimental("mom_layered")
@dataclass(frozen=True)
class LayeredSurfaceConfig:
    """Mesh and port controls for sheets, Box vias and vertical bridge ports in a LayerStack.

    ``bridge_width_m`` is the width of the metal strip that joins each
    LumpedPort's two terminals; the delta-gap source sits on its midpoint cut.
    Bridges are ideal/lossless, including when the physical sheets are lossy.
    It has no default because it is part of the port model. A LumpedPort with a
    ``voltage_aperture`` (edge port, port_system.py) uses the aperture width instead.
    ``max_edge_length_m``
    caps the cell size; ``edge_cell_m`` grades cells down to that size at every
    sheet edge, port cut and breakpoint with growth ratio ``growth_ratio``.
    ``breakpoint_tolerance_m`` optionally clusters all x/y layout coordinates
    (including via rims and ports) on that SI lattice; z is always exact.
    Use at most one tenth the minimum feature; 5 nm repairs +/-2 nm RFIC jitter.
    ``sweep=AdaptiveSweep(...)`` selects adaptive sampling and owns the band;
    omit Simulation.mom frequency/frequencies in that case. Result.network
    contains solved samples; mw.mom.surface_currents(result).sweep contains the model,
    dense NetworkData and passivity report.
    conductor_model="closed" extrudes positive physical_thickness symmetrically
    about each PlanarPEC mid-plane. Every exterior face uses single-face SIBC;
    the interior field is zero. Validity requires skin depth << thickness
    and width; this model has no finite-conductivity DC limit.
    "sheet" retains two-face finite-thickness impedance and DC resistance.
    The default conductor_model="auto" selects "surface_admittance" when a
    lossy PlanarPEC reaches t/delta >= 0.3 at the highest requested frequency
    (the weak-skin onset) and "sheet" otherwise. When it selects
    surface_admittance for a layout with a feature that model refuses (polygon
    layout, zero-thickness sheet, roughness, vertical port, gradients) it
    raises; declare "sheet" explicitly there. metadata["conductor_model"] is the
    selected model and metadata["conductor_model_selection"] the rule record
    (runtime.select_conductor_model).
    conductor_model="surface_admittance" extrudes like "closed" but replaces
    each lossy body by its differential surface admittance (interior lossy
    Helmholtz problem per body): exact DC resistance, weak and strong skin
    effect. PEC contacts are explicit contact faces, and each horizontal port
    bridge becomes an ideal PEC block of the terminal metal thickness whose
    end faces contact both terminals; the delta gap is the block's
    mid cross-section ring. An edge port's block spans the aperture width
    (flush with the terminal sidewalls, full end-face contacts). No roughness
    model. The matrix_free and hmatrix backends fold each body's dense
    admittance block into their near-field operator and preconditioner, so
    their memory grows with the square of the largest body's exterior basis
    count.
    Thickness semantics: "sheet" does not model geometric thickness; for
    LossyMetalMedium physical_thickness enters only the surface impedance,
    and a Material.pec() sheet is infinitely thin whatever its declared
    physical_thickness. "closed" models the conductor volume. Result.metadata
    and Result.network.metadata record conductor_model and
    thickness_modelled_geometrically (False for "sheet", True for "closed").
    Horizontal ideal delta-gap bridges attach to the top face of both terminals;
    a closed ``voltage_aperture`` edge port is a thick-terminal PEC block over
    the aperture width and the terminal thickness (network.py).
    ``operator_backend="dense"`` (default) assembles each frequency
    as one complex128 equilibrated loop/tree system plus a complex64 LU
    workspace, at most 24*N^2 bytes per frequency for F <= N faces.
    ``operator_backend="matrix_free"`` solves the same system by block GMRES
    over all ports with a complex128 sparse LU of the near-field interactions
    as right preconditioner; far interactions are re-integrated on the fly in
    every action, so memory grows with N, not N^2. It stops when the
    admittance error bound |X|_F |R|_F / |B^T X|_F is below 1e-5.
    ``operator_backend="hmatrix"`` is matrix_free with the far interactions
    compressed once per frequency (cluster trees per z plane, lateral
    admissibility, adaptive cross approximation); the compression adds an
    a-posteriori admittance bound of at most 1e-5 (hmatrix.py).
    ``hmatrix_exact_check=True`` additionally verifies every hmatrix solve
    with one exact far action (O(N^2) work per frequency): the first-order
    compression error of the admittances must stay below 1e-5.
    ``planar_mesher="auto"`` (default) meshes Manhattan layouts with the
    rectangle mesher and switches to the conforming planar triangle mesher for
    non-Manhattan PlanarPEC polygons, z-axis Prism or ExtrudedPolygon vias and
    vias that pass a metal plane; ``"triangle"`` forces the triangle mesher for every layout.
    Polygon layouts take every operator backend and delta-gap or
    ``voltage_aperture`` edge ports (horizontal).
    """

    bridge_width_m: float
    max_edge_length_m: float
    edge_cell_m: float | None = None
    growth_ratio: float = 1.5
    breakpoint_tolerance_m: float = 0.0
    sweep: AdaptiveSweep | None = None
    conductor_model: LayeredConductorModel = LayeredConductorModel.AUTO
    operator_backend: LayeredOperatorBackend = LayeredOperatorBackend.DENSE
    hmatrix_exact_check: bool = False
    planar_mesher: PlanarMesher = PlanarMesher.AUTO

    def __post_init__(self):
        object.__setattr__(self, "planar_mesher",
                           parse_enum(PlanarMesher, self.planar_mesher, name="LayeredSurfaceConfig.planar_mesher"))
        object.__setattr__(self, "operator_backend", parse_enum(
            LayeredOperatorBackend, self.operator_backend, name="LayeredSurfaceConfig.operator_backend"))
        object.__setattr__(self, "conductor_model", parse_enum(
            LayeredConductorModel, self.conductor_model, name="LayeredSurfaceConfig.conductor_model"))
        if self.hmatrix_exact_check and self.operator_backend != "hmatrix":
            raise ValueError("hmatrix_exact_check requires operator_backend='hmatrix'")
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
        payload = option_asdict(self)
        # Keep the established sheet descriptor stable for frozen evidence.
        # Network/result metadata always records the selected model.
        if self.conductor_model == "sheet":
            del payload["conductor_model"]
        if self.operator_backend == "dense":
            del payload["operator_backend"]
        if not self.hmatrix_exact_check:
            del payload["hmatrix_exact_check"]
        if self.planar_mesher == "auto":
            del payload["planar_mesher"]
        return payload
