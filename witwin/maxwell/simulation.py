from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable

import torch

from ._enums import StrEnum, parse_enum
from .adjoint_inputs import geometry_tensors, named_scene_tensors, require_material_gradient_channels
from .capability import CapabilityReport, check_scene
from .compiler.accel.dispatch import requires_grad, requires_grad_scope
from .electrostatic.api import ElectrostaticConfig
from .electrostatic.capacitance import CapacitanceConfig
from .errors import MaxwellWarning, SetupError, UnsupportedFeatureError
from .fdtd_cpml import CPMLConfig, CPMLMemoryMode
from .fdtd_parallel import FDTDParallelConfig
from .fdfd_response import FieldProjection, FarFieldProjection, LocalResponseConfig
from .lumped import CoaxialLumpedResistor, PortExcitation, PortSweep
from .material_cache import FDTDMaterialCache
from .modal_sweep import ModalPortSweep
from .monitors import MediumMonitor, PermittivityMonitor
from .ports import CoaxialLumpedPort, LumpedPort, TerminalPort, TerminalWavePort, WavePort
from .rational import RationalModel, StateSpaceNetwork
from .result import Result
from ._stability import experimental
from .mom.runtime import MoMConfig
from .scene import Scene, SceneModule, SubpixelAveraging, prepare_scene, scene_declarations
from .compiler.circuits import compile_circuits

FDFD = None
FDTD = None
calculate_required_steps = None


class _AcceleratedSetupFailure(RuntimeError):
    """The accelerated CUDA setup of an FDTD preparation failed; the owner retries with the torch setup."""


class SimulationMethod(StrEnum):
    FDFD = "fdfd"
    FDTD = "fdtd"
    MOM = "mom"
    ELECTROSTATIC = "electrostatic"
    CAPACITANCE = "capacitance"


# Static (zero-frequency) methods and the configuration each one takes.
_STATIC_CONFIGS = {SimulationMethod.ELECTROSTATIC: ElectrostaticConfig, SimulationMethod.CAPACITANCE: CapacitanceConfig}


class SpectralWindowKind(StrEnum):
    NONE = "none"
    HANNING = "hanning"
    RAMP = "ramp"


class AbsorberKind(StrEnum):
    AUTO = "auto"
    CPML = "cpml"
    PML = "pml"
    ABSORBER = "absorber"
    STABLE_PML = "stablepml"


class FDFDSolverType(StrEnum):
    """The linear solve of ``FDFDSolverConfig.solver_type``."""

    GMRES = "gmres"
    CG = "cg"
    BICGSTAB = "bicgstab"
    TFQMR = "tfqmr"
    IDR = "idr"
    SQMR = "sqmr"
    DIRECT = "direct"


class FDFDPreconditioner(StrEnum):
    """The preconditioner of the iterative FDFD solver types (``ILU`` is ILU(0))."""

    JACOBI = "jacobi"
    SSOR = "ssor"
    ILU = "ilu"
    AMS = "ams"
    NONE = "none"


class Precision(StrEnum):
    """The working precision of the iterative FDFD solve."""

    SINGLE = "single"
    DOUBLE = "double"


def _normalize_frequencies(*, frequencies) -> tuple[float, ...]:
    from ._fixed_inputs import require_fixed_input
    if frequencies is None:
        raise ValueError("A target frequency is required.")
    if getattr(frequencies, "ndim", None) != 0 and isinstance(frequencies, Iterable) \
            and not isinstance(frequencies, (str, bytes)):
        raw_values = tuple(frequencies)
    else:
        raw_values = (frequencies,)
    require_fixed_input(raw_values, name="Simulation frequencies")
    values = tuple(float(value) for value in raw_values)
    if not values:
        raise ValueError("frequencies must not be empty.")
    return values


def _require_fixed_time_inputs(value):
    from ._fixed_inputs import require_fixed_input
    for name in ("time_steps", "steady_cycles", "transient_cycles", "dt", "max_electric_field"):
        require_fixed_input(getattr(value, name, None), name="TimeConfig." + name)


def _validate_fixed_fdtd_controls(config):
    """Check fixed physical/execution inputs before coercion or reuse.

    Material, source amplitude, and circuit parameters are intentionally absent:
    those have separately registered differentiable paths.
    """
    from ._fixed_inputs import require_fixed_input
    for name in ("enable_plot", "full_field_dft", "volume_dft_components", "shutoff", "shutoff_check_interval",
                 "cuda_graph", "adjoint_checkpoint_stride", "absorber", "field_dtype", "reference_frequency",
                 "verbose"):
        require_fixed_input(getattr(config, name, None), name="FDTDConfig." + name)
    for item in fields(config.cpml):
        require_fixed_input(getattr(config.cpml, item.name), name="CPMLConfig." + item.name)
    _require_fixed_time_inputs(config.run_time)
    for name in ("window", "normalize_source", "observer_dtype"):
        require_fixed_input(getattr(config.spectral_sampler, name, None), name="SpectralSampler." + name)


_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
_MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")


def _normalize_volume_dft_components(value) -> tuple[str, ...]:
    """``FDTDConfig.volume_dft_components`` in canonical ``Ex ... Hz`` order (E always recorded)."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError("volume_dft_components must be a tuple of Yee component names.")
    names = tuple(str(name) for name in value)
    allowed = _ELECTRIC_COMPONENTS + _MAGNETIC_COMPONENTS
    if len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError("volume_dft_components must be unique Yee component names among Ex, Ey, Ez, Hx, Hy, Hz.")
    if not set(_ELECTRIC_COMPONENTS) <= set(names):
        raise ValueError("volume_dft_components must contain Ex, Ey and Ez; Hx, Hy and Hz are optional additions.")
    return tuple(name for name in allowed if name in names)


def _resolve_full_field_dft(requested: bool | None, frequencies, ports) -> bool:
    if requested is not None:
        return requested
    return len(frequencies) > 1 and not any(
        isinstance(port, (LumpedPort, TerminalPort, CoaxialLumpedPort)) for port in ports
    )


def _normalize_port_excitations(excitations) -> tuple[PortExcitation, ...]:
    if excitations is None:
        return ()
    if isinstance(excitations, PortExcitation):
        resolved = (excitations,)
    elif isinstance(excitations, Iterable) and not isinstance(excitations, (str, bytes)):
        resolved = tuple(excitations)
    else:
        raise TypeError("excitations must be a PortExcitation or an iterable of PortExcitation entries.")
    if any(not isinstance(excitation, PortExcitation) for excitation in resolved):
        raise TypeError("excitations must contain only PortExcitation entries.")
    names = tuple(excitation.port_name for excitation in resolved)
    if len(set(names)) != len(names):
        raise ValueError("Each port may appear in excitations at most once.")
    return resolved


@dataclass(frozen=True)
class FDFDSolverConfig:
    """Linear-solver settings of ``FDFDConfig.solver``.

    ``solver_type`` selects the solve (``FDFDSolverType``, default ``"gmres"``); the
    iterative types use ``preconditioner`` (``FDFDPreconditioner``) and work in
    ``precision`` (``Precision``).
    """

    max_iter: int = 5000
    tol: float = 1e-6
    restart: int = 200
    solver_type: FDFDSolverType = FDFDSolverType.GMRES
    preconditioner: FDFDPreconditioner = FDFDPreconditioner.JACOBI
    precision: Precision = Precision.SINGLE
    ssor_omega: float = 0.8  # relaxation factor for the 'ssor' preconditioner
    device_memory_limit_bytes: int | None = None

    def __post_init__(self):
        set_field = object.__setattr__
        set_field(self, "solver_type", parse_enum(FDFDSolverType, self.solver_type, name="FDFDSolverConfig.solver_type"))
        set_field(self, "preconditioner",
                  parse_enum(FDFDPreconditioner, self.preconditioner, name="FDFDSolverConfig.preconditioner"))
        set_field(self, "precision", parse_enum(Precision, self.precision, name="FDFDSolverConfig.precision"))
        limit = self.device_memory_limit_bytes
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("device_memory_limit_bytes must be a positive integer.")
            if self.solver_type is not FDFDSolverType.DIRECT:
                raise ValueError("device_memory_limit_bytes applies to solver_type='direct'")


@dataclass(frozen=True)
class TimeConfig:
    time_steps: int | None = None
    steady_cycles: int = 20
    transient_cycles: int = 15
    dt: float | None = None
    max_electric_field: float | None = None

    def __post_init__(self):
        _require_fixed_time_inputs(self)
        if self.time_steps is not None and (isinstance(self.time_steps, bool) or not isinstance(self.time_steps, int) or self.time_steps < 1):
            raise ValueError("TimeConfig.time_steps must be a positive integer.")
        for name in ("transient_cycles", "steady_cycles"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"TimeConfig.{name} must be nonnegative and finite.")
        if self.dt is not None and (not math.isfinite(float(self.dt)) or float(self.dt) <= 0):
            raise ValueError("TimeConfig.dt must be positive and finite.")
        if self.max_electric_field is not None and (not math.isfinite(float(self.max_electric_field)) or float(self.max_electric_field) <= 0):
            raise ValueError("TimeConfig.max_electric_field must be positive and finite.")

    @classmethod
    def auto(cls, steady_cycles: int = 20, transient_cycles: int = 15) -> "TimeConfig":
        return cls(
            time_steps=None,
            steady_cycles=steady_cycles,
            transient_cycles=transient_cycles,
        )


@dataclass(frozen=True)
class SpectralSampler:
    window: SpectralWindowKind = SpectralWindowKind.HANNING
    normalize_source: bool = False
    observer_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        from ._fixed_inputs import require_fixed_input
        require_fixed_input((self.window, self.normalize_source), name="SpectralSampler controls")
        object.__setattr__(self, "window", parse_enum(SpectralWindowKind, self.window, name="SpectralSampler.window"))
        object.__setattr__(self, "normalize_source", bool(self.normalize_source))
        if self.observer_dtype not in (torch.float32, torch.float64):
            raise ValueError("observer_dtype must be torch.float32 or torch.float64.")
        if self.observer_dtype == torch.float64 and self.normalize_source:
            raise ValueError("Float64 spectral observers currently require normalize_source=False.")


@dataclass(frozen=True)
class FDFDConfig:
    solver: FDFDSolverConfig = field(default_factory=FDFDSolverConfig)
    enable_plot: bool = False
    verbose: bool = False
    local_response: LocalResponseConfig | None = None
    observations: tuple = ()
    device: str = "cuda"

    def __post_init__(self):
        object.__setattr__(self, "device", str(torch.device(self.device)))
        if not isinstance(self.solver, FDFDSolverConfig):
            raise TypeError("solver must be an FDFDSolverConfig instance.")
        if self.local_response is not None and not isinstance(self.local_response, LocalResponseConfig):
            raise TypeError("local_response must be a LocalResponseConfig.")
        object.__setattr__(self, "observations", tuple(self.observations))
        if any(not isinstance(value, (FieldProjection, FarFieldProjection)) for value in self.observations):
            raise TypeError("observations must contain field or far-field projections.")
        if len({value.name for value in self.observations}) != len(self.observations):
            raise ValueError("Observation names must be unique.")
        if self.local_response is not None and self.observations:
            raise ValueError("Specify observations inside local_response when precomputation is enabled.")
        object.__setattr__(self, "enable_plot", bool(self.enable_plot))
        object.__setattr__(self, "verbose", bool(self.verbose))


@dataclass(frozen=True)
class FDTDConfig:
    run_time: TimeConfig = field(default_factory=TimeConfig.auto)
    spectral_sampler: SpectralSampler = field(default_factory=SpectralSampler)
    reference_frequency: float | None = None
    enable_plot: bool = False
    full_field_dft: bool | None = None
    field_dtype: torch.dtype = torch.float32
    absorber: AbsorberKind = AbsorberKind.AUTO
    cpml: CPMLConfig = field(default_factory=CPMLConfig)
    adjoint_checkpoint_stride: int | None = None
    shutoff: float = 0.0  # relative E-energy threshold for auto-shutoff; 0 disables (opt-in)
    shutoff_check_interval: int = 100
    cuda_graph: bool = True  # capture the field-update core into a CUDA graph (default; eager fallback)
    parallel: FDTDParallelConfig | None = None
    initial_condition: Any = None  # ElectrostaticInitialCondition pre-bias (opt-in)
    material_cache: FDTDMaterialCache | None = None  # static material compilation reuse (opt-in)
    device: str = "cuda"
    verbose: bool = False
    # Yee components of the full-field (volume) DFT; adding Hx/Hy/Hz records H phasors (opt-in).
    volume_dft_components: tuple[str, ...] = _ELECTRIC_COMPONENTS

    def __post_init__(self):
        if not isinstance(self.run_time, TimeConfig):
            raise TypeError("run_time must be a TimeConfig instance.")
        if not isinstance(self.spectral_sampler, SpectralSampler):
            raise TypeError("spectral_sampler must be a SpectralSampler instance.")
        if not isinstance(self.cpml, CPMLConfig):
            raise SetupError(
                f"FDTDConfig.cpml must be a CPMLConfig instance, not {type(self.cpml).__name__}; "
                "the cpml_config dict was removed in 0.7.0."
            )
        _validate_fixed_fdtd_controls(self)
        set_field = object.__setattr__
        if self.reference_frequency is not None:
            set_field(self, "reference_frequency", float(self.reference_frequency))
            if not math.isfinite(self.reference_frequency) or self.reference_frequency <= 0:
                raise ValueError("reference_frequency must be finite and positive.")
        set_field(self, "enable_plot", bool(self.enable_plot))
        set_field(self, "verbose", bool(self.verbose))
        set_field(self, "device", str(torch.device(self.device)))
        if self.field_dtype not in (torch.float32, torch.float64):
            raise ValueError("FDTD field_dtype must be torch.float32 or torch.float64.")
        set_field(self, "absorber", parse_enum(AbsorberKind, self.absorber, name="FDTDConfig.absorber"))
        set_field(self, "shutoff", float(self.shutoff))
        if self.shutoff < 0:
            raise ValueError("shutoff must be >= 0.")
        set_field(self, "shutoff_check_interval", int(self.shutoff_check_interval))
        if self.shutoff_check_interval <= 0:
            raise ValueError("shutoff_check_interval must be > 0.")
        if self.parallel is not None and not isinstance(self.parallel, FDTDParallelConfig):
            raise TypeError("parallel must be an FDTDParallelConfig instance or None.")
        if self.material_cache is not None and not isinstance(self.material_cache, FDTDMaterialCache):
            raise TypeError("material_cache must be an FDTDMaterialCache instance or None.")
        set_field(self, "cuda_graph", bool(self.cuda_graph))
        if self.parallel is not None:
            # Multi-GPU FDTD does not capture peer communication in CUDA graphs;
            # keep the distributed path on eager stepping regardless of the
            # single-GPU default.
            set_field(self, "cuda_graph", False)
        if self.parallel is not None and self.enable_plot:
            raise ValueError(
                "Multi-GPU FDTD plotting requires running first and requesting gathered fields."
            )
        if self.adjoint_checkpoint_stride is not None:
            set_field(self, "adjoint_checkpoint_stride", int(self.adjoint_checkpoint_stride))
            if self.adjoint_checkpoint_stride <= 0:
                raise ValueError("adjoint_checkpoint_stride must be > 0 when provided.")
        set_field(self, "volume_dft_components", _normalize_volume_dft_components(self.volume_dft_components))
        if self.records_magnetic_volume_dft:
            if self.full_field_dft is not True:
                raise ValueError("Magnetic volume_dft_components (Hx, Hy, Hz) require full_field_dft=True.")
            if self.parallel is not None:
                raise UnsupportedFeatureError(
                    "Multi-GPU FDTD records no magnetic volume DFT; remove Hx/Hy/Hz from volume_dft_components "
                    "or run on one device.", feature="guard:ae727bf9", method="fdtd")

    @property
    def records_magnetic_volume_dft(self) -> bool:
        """Whether the volume DFT also records H (``volume_dft_components`` includes Hx, Hy or Hz)."""
        return any(name in _MAGNETIC_COMPONENTS for name in self.volume_dft_components)


def _to_tensor_fields(field_mapping: dict[str, Any], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        name.upper(): torch.as_tensor(value, device=device)
        for name, value in field_mapping.items()
    }


def _resolve_fdfd_backend():
    global FDFD
    if FDFD is None:
        from .fdfd.solver import FDFD as backend

        FDFD = backend
    return FDFD


def _resolve_fdtd_backend():
    global FDTD, calculate_required_steps
    if FDTD is None or calculate_required_steps is None:
        from .fdtd.solver import FDTD as fdtd_backend
        from .fdtd.solver import calculate_required_steps as calculate_steps_backend

        FDTD = fdtd_backend
        calculate_required_steps = calculate_steps_backend
    return FDTD, calculate_required_steps


def _resolve_scene_input(scene_like):
    if isinstance(scene_like, Scene):
        return scene_like, None
    if isinstance(scene_like, SceneModule):
        return scene_like.to_scene(), scene_like
    raise TypeError("scene must be a maxwell.Scene or maxwell.SceneModule.")


def _scene_trainable_density_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    trainable = []
    for region in scene.material_regions:
        density = getattr(region, "density", None)
        if isinstance(density, torch.Tensor) and density.requires_grad:
            trainable.append(density)
    return tuple(trainable)


def _scene_trainable_geometry_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    return tuple(
        value
        for owner in (*scene.structures, *scene.material_regions)
        for value in geometry_tensors(getattr(owner, "geometry", None))
    )


def _scene_trainable_material_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    from .adjoint_inputs import material_pole_tensors

    trainable = []
    materials = [getattr(structure, 'material', None) for structure in scene.structures]
    materials.append(scene.background)
    for material in materials:
        trainable.extend(value for value in material_pole_tensors(material) if value.requires_grad)
        trainable.extend(value for name in ("eps_r", "sigma_e", "thickness", "conductivity")
                         if isinstance((value := getattr(material, name, None)), torch.Tensor)
                         and value.requires_grad)
        perturbation = getattr(material, "perturbation", None)
        if isinstance(perturbation, torch.Tensor) and perturbation.requires_grad:
            trainable.append(perturbation)
    return tuple(trainable)


def _scene_trainable_circuit_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for circuit in scene.circuits:
        candidates.extend(circuit.parameters.values())
        candidates.extend(value for value, _constraint in circuit.initial_conditions.values())
        for device in circuit.devices:
            for value in device.parameters.values():
                if isinstance(value, torch.Tensor):
                    candidates.append(value)
                elif value is not None and hasattr(value, "__dict__"):
                    candidates.extend(vars(value).values())
    unique = []
    seen = set()
    for value in candidates:
        if isinstance(value, torch.Tensor) and value.requires_grad and id(value) not in seen:
            unique.append(value)
            seen.add(id(value))
    return tuple(unique)


def _scene_trainable_wire_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    trainable = []
    for wire in scene.thin_wires:
        for value in (getattr(wire, "points", None), getattr(wire, "radius", None)):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                trainable.append(value)
    return tuple(trainable)


def _trainable_rf_parameters(
    scene: Scene,
    *,
    excitations=(),
    port_sweep: PortSweep | None = None,
) -> tuple[torch.Tensor, ...]:
    candidates = []
    for port in scene.ports:
        candidates.append(getattr(port, "reference_impedance", None))
        termination = getattr(port, "termination", None)
        if termination is not None:
            candidates.extend(
                getattr(termination, name, None)
                for name in ("r", "l", "c")
            )
    for element in scene.lumped_elements:
        candidates.append(getattr(element, "value", None))
    for block in scene.networks:
        model = getattr(block, "model", None)
        candidates.extend(
            getattr(model, name, None)
            for name in (
                "poles",
                "residues",
                "direct",
                "proportional",
                "A",
                "B",
                "C",
                "D",
            )
        )
    for excitation in excitations:
        candidates.extend((excitation.amplitude, excitation.source_impedance))
    if port_sweep is not None:
        candidates.append(port_sweep.amplitude)
    return tuple(
        value
        for value in candidates
        if isinstance(value, torch.Tensor) and value.requires_grad
    )


def _scene_trainable_census(scene: Scene) -> tuple[torch.Tensor, ...]:
    """Every grad-carrying scene tensor that a gradient channel of the FDTD bridge declares."""
    return (
        _scene_trainable_density_parameters(scene)
        + _scene_trainable_geometry_parameters(scene)
        + _scene_trainable_material_parameters(scene)
        + _scene_trainable_circuit_parameters(scene)
        + _scene_trainable_wire_parameters(scene)
        + _trainable_rf_parameters(scene)
    )


def _reject_uncensused_trainable_tensors(scene: Scene) -> None:
    """Refuse grad-carrying scene tensors that no FDTD gradient channel reads.

    The trainable census decides whether a run goes through the gradient bridge.
    A grad-carrying tensor outside it, such as ``Material(mu_r=tensor)`` or a
    tensor ``ModulationSpec.amplitude``, would leave the run forward-only with
    detached outputs, so it is refused by attribute path. A ``mass_density`` only
    feeds SAR post-processing, which differentiates it directly.
    """
    if not requires_grad(scene):
        return
    claimed = {id(tensor) for tensor in _scene_trainable_census(scene)}
    claimed.update(id(getattr(structure.material, "mass_density", None)) for structure in scene.structures)
    unclaimed = tuple(path for path, tensor in named_scene_tensors(scene) if id(tensor) not in claimed)
    if unclaimed:
        raise UnsupportedFeatureError(
            f"Simulation.fdtd(...) has no gradient channel for {len(unclaimed)} grad-carrying scene tensor(s); "
            "the run would return outputs detached from them:\n"
            + "\n".join(f"- {path}" for path in unclaimed)
            + "\nKeep these values fixed (requires_grad=False or .detach()), or run under torch.no_grad().",
            feature="gradient.unclaimed_tensor",
            method="fdtd",
            alternative="keep the listed values fixed",
            items=unclaimed,
        )


def _nonstandard_medium_reason(scene: Scene) -> str | None:
    """Describe the first medium that leaves the pure real standard reverse class.

    The distributed joint-solve adjoint has a verified reverse core only for the
    real standard (open-boundary, non-dispersive, non-conductive, isotropic, linear,
    static) update, so a scene carrying any richer medium is rejected before it
    reaches the distributed bridge. Returns ``None`` when every material is standard.
    """

    for structure in scene.structures:
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if getattr(material, "is_medium2d", False):
            return f"structure {structure.name!r} uses a 2D sheet (Medium2D) material"
        if getattr(material, "is_lossy_metal", False):
            return f"structure {structure.name!r} uses a lossy-metal (SIBC) material"
        if getattr(material, "is_electric_dispersive", False):
            return f"structure {structure.name!r} uses electric dispersive (ADE) media"
        if getattr(material, "is_magnetic_dispersive", False):
            return f"structure {structure.name!r} uses magnetic dispersive (ADE) media"
        if getattr(material, "is_nonlinear", False):
            return f"structure {structure.name!r} uses nonlinear media"
        if getattr(material, "is_anisotropic", False):
            return f"structure {structure.name!r} uses anisotropic media"
        if getattr(material, "is_modulated", False):
            return f"structure {structure.name!r} uses time-modulated media"
        sigma_e = getattr(material, "sigma_e", 0.0)
        if sigma_e is not None and float(sigma_e) != 0.0:
            return f"structure {structure.name!r} uses static-conductive media"
        sigma_tensor = getattr(material, "sigma_e_tensor", None)
        if sigma_tensor is not None and any(
            float(component) != 0.0 for component in sigma_tensor.as_tuple()
        ):
            return f"structure {structure.name!r} uses static-conductive media"
    return None


def _tiled_monitor_objective_reason(scene: Scene) -> str | None:
    """Describe the first monitor whose adjoint seed would need tiled scatter.

    Baseline distributed seed routing supports point-monitor spectra (single owner)
    and full-field DFT (owned x-slices). A plane / flux / mode / closed-surface /
    diffraction monitor is stitched across shards in the forward, so its cotangent
    would need the same owned-interval scatter -- the follow-up slice. Returns
    ``None`` when every monitor is point/field-time/material (analytic).
    """

    from .monitors import (
        ClosedSurfaceMonitor,
        DiffractionMonitor,
        FieldTimeMonitor,
        FinitePlaneMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ModeMonitor,
        PlaneMonitor,
        PointMonitor,
    )

    tiled = (
        PlaneMonitor,
        FinitePlaneMonitor,
        ModeMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ClosedSurfaceMonitor,
        DiffractionMonitor,
    )
    for monitor in scene.resolved_monitors():
        if isinstance(monitor, tiled):
            return f"monitor {getattr(monitor, 'name', '?')!r} is a {type(monitor).__name__}"
        if isinstance(monitor, FieldTimeMonitor) and monitor.region_kind != "point":
            return f"FieldTimeMonitor {monitor.name!r} is a {monitor.region_kind} region"
    return None


# Mode-source gauges chosen by a preparation, per Scene or SceneModule. A later preparation of
# the same input reuses them, so replays keep one gauge. Kept beside the immutable Scene.
MODE_SOURCE_GAUGES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _require_cuda_device(device, *, method: str) -> None:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"Simulation.{method}(...) requires config.device to be CUDA, got {device}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Simulation.{method}(...) requires CUDA, but torch.cuda.is_available() is False."
        )


class Simulation:
    @staticmethod
    @experimental("batch")
    def batch(simulations, *, execution, parameters=None, variant_ids=None,
              variation=None, outputs=None):
        """Plan and execute supported same-scene variants through native CUDA cohorts."""
        from .execution.batch import BatchSimulation

        return BatchSimulation(
            simulations, execution=execution, parameters=parameters,
            variant_ids=variant_ids, variation=variation, outputs=outputs,
        )

    def __init__(
        self,
        *,
        scene,
        method: SimulationMethod,
        frequencies: tuple[float, ...],
        config: FDFDConfig | FDTDConfig | MoMConfig | ElectrostaticConfig | CapacitanceConfig | None = None,
        excitations=None,
        metadata: dict[str, Any] | None = None,
    ):
        from ._fixed_inputs import require_fixed_input
        frequencies = tuple(frequencies)
        require_fixed_input(frequencies, name="Simulation.frequencies")
        self.method = SimulationMethod(method)
        if self.method in _STATIC_CONFIGS:
            if frequencies:
                raise ValueError("Electrostatic and capacitance simulations are static and take no frequencies.")
        elif not frequencies:
            raise ValueError("At least one frequency is required.")
        resolved_scene, scene_module = _resolve_scene_input(scene)
        self.scene_input = scene
        self.scene = resolved_scene
        self.scene_module = scene_module
        self.frequencies = tuple(float(freq) for freq in frequencies)
        self.frequency = self.frequencies[0] if self.frequencies else None
        self._validate_network_solver()
        self._validate_method_scene_features(config)
        self.modal_port_sweep = excitations if isinstance(excitations, ModalPortSweep) else None
        if self.modal_port_sweep is not None:
            excitations = None
        self.port_sweep = excitations if isinstance(excitations, PortSweep) else None
        self.excitations = (
            () if self.port_sweep is not None else _normalize_port_excitations(excitations)
        )
        trainable_rf_parameters = _trainable_rf_parameters(
            resolved_scene,
            excitations=self.excitations,
            port_sweep=self.port_sweep,
        )
        self._refresh_trainable_parameters(config=config, frequencies=frequencies)
        if self.method != SimulationMethod.FDTD and (
            self.excitations or self.port_sweep or self.modal_port_sweep is not None
        ):
            raise ValueError("RF port excitation is supported by Simulation.fdtd(...) only.")
        if self.method != SimulationMethod.FDTD and trainable_rf_parameters:
            raise UnsupportedFeatureError(
                "Trainable RF port, source, and R/L/C parameters require Simulation.fdtd(...).",
                feature="port", method=self.method.value
            )
        defer_lumped_sweep_validation = self.port_sweep is not None and all(
            not isinstance(port, WavePort) for port in resolved_scene.ports
        )
        if not defer_lumped_sweep_validation:
            self._validate_trainable_rf_support()
        self._validate_pcb_material_frequency()
        if self.method == SimulationMethod.FDFD:
            if any(not math.isfinite(f) or f <= 0 for f in self.frequencies) or len(set(self.frequencies)) != len(self.frequencies):
                raise ValueError("FDFD frequencies must be finite, positive and distinct.")
            if config is None:
                config = FDFDConfig()
            if not isinstance(config, FDFDConfig):
                raise TypeError("FDFD simulations require an FDFDConfig.")
        elif self.method == SimulationMethod.MOM:
            from .layers import LayerStack
            layered = isinstance(self.scene.background, LayerStack)
            if len(self.frequencies) != 1 and not layered:
                raise ValueError("MoM simulations require one frequency.")
            config = MoMConfig() if config is None else config
            if not isinstance(config, MoMConfig):
                raise TypeError("MoM simulations require an MoMConfig.")
            if layered != (config.layered is not None):
                raise ValueError("A LayerStack background requires MoMConfig(layered=LayeredSurfaceConfig(...)), "
                                 "and MoMConfig.layered requires a LayerStack background.")
            if layered and (any(not math.isfinite(f) or f <= 0 for f in self.frequencies)
                            or len(set(self.frequencies)) != len(self.frequencies)):
                raise ValueError("Layered MoM frequencies must be finite, positive and distinct.")
        elif self.method in _STATIC_CONFIGS:
            config_type = _STATIC_CONFIGS[self.method]
            config = config_type() if config is None else config
            if not isinstance(config, config_type):
                raise TypeError(f"{self.method.value} simulations take config={config_type.__name__}(...), "
                                f"not {type(config).__name__}.")
        else:
            if config is None:
                config = FDTDConfig()
            if not isinstance(config, FDTDConfig):
                raise TypeError("FDTD simulations require an FDTDConfig.")
        self.config = config
        self.metadata = dict(metadata or {})

    @classmethod
    @experimental("mom")
    def mom(cls, scene, *, frequencies=None, config: MoMConfig | None = None) -> "Simulation":
        """Prepare a surface-integral solve in SI units.

        ``frequencies`` is one frequency or a sequence. Homogeneous-body scenes
        solve one frequency. Planar PEC sheets in a LayerStack background accept a
        sweep and return an N-port ``Result.network`` from their LumpedPorts; an
        adaptive layered sweep (``LayeredSurfaceConfig.sweep``) defines its own
        frequencies, so ``frequencies`` is omitted there.
        """
        if isinstance(config, MoMConfig) and config.layered is not None and config.layered.sweep is not None:
            if frequencies is not None:
                raise ValueError("Adaptive layered sweep defines its own frequencies; omit frequencies")
            frequencies = config.layered.sweep.seed_frequencies()
        return cls(scene=scene, method=SimulationMethod.MOM,
                   frequencies=_normalize_frequencies(frequencies=frequencies), config=config)

    @classmethod
    @experimental("fdfd")
    def fdfd(cls, scene, *, frequencies, config: FDFDConfig | None = None) -> "Simulation":
        """Prepare a frequency-domain solve; ``frequencies`` is one frequency or a sequence (Hz)."""
        return cls(scene=scene, method=SimulationMethod.FDFD,
                   frequencies=_normalize_frequencies(frequencies=frequencies), config=config)

    @classmethod
    def fdtd(cls, scene, *, frequencies, excitations=None, config: FDTDConfig | None = None) -> "Simulation":
        """Prepare a time-domain solve; ``frequencies`` is one frequency or a sequence (Hz).

        Every run option lives in ``config`` (an ``FDTDConfig``); ``excitations`` selects
        the driven ports (``PortExcitation`` entries, a ``PortSweep`` or a ``ModalPortSweep``).
        """
        return cls(scene=scene, method=SimulationMethod.FDTD,
                   frequencies=_normalize_frequencies(frequencies=frequencies), config=config,
                   excitations=excitations)

    @classmethod
    @experimental("electrostatics")
    def electrostatic(cls, scene, *, config: ElectrostaticConfig | None = None) -> "Simulation":
        """Prepare a static (Laplace/Poisson) solve; ``run()`` returns ``Result(method="electrostatic")``.

        Every run option lives in ``config`` (an ``ElectrostaticConfig``): the
        ``ElectrostaticBoundarySpec`` of the six outer faces (independent of the
        full-wave ``Scene.boundary``) and the ``ElectrostaticSolverConfig``.
        """
        return cls(scene=scene, method=SimulationMethod.ELECTROSTATIC, frequencies=(), config=config)

    @classmethod
    @experimental("electrostatics")
    def capacitance(cls, scene, *, config: CapacitanceConfig | None = None) -> "Simulation":
        """Prepare an N-terminal Maxwell capacitance-matrix extraction.

        ``run()`` returns ``Result(method="capacitance")`` with the typed
        ``mw.electrostatic.capacitance(result)`` accessor (``CapacitanceData``). Every run option lives
        in ``config`` (a ``CapacitanceConfig``): the terminal selection and order,
        the grounded ``reference`` conductor, the boundary, the solver settings and
        the opt-in ``truncation_estimate`` (one additional enlarged grounded-box solve).
        """
        return cls(scene=scene, method=SimulationMethod.CAPACITANCE, frequencies=(), config=config)

    @requires_grad_scope()
    def prepare(self, *, response_cache=None):
        if response_cache is not None and (self.method != SimulationMethod.FDFD or self.config.local_response is None):
            raise ValueError("response_cache requires a local-response FDFD simulation.")
        self._refresh_scene()
        if self.method in _STATIC_CONFIGS:
            # The method table is the whole admission of a static solve.
            self._validate_method_scene_features(self.config)
            from .compiler.electrostatic import compile_electrostatics
            return PreparedSimulation(self, compile_electrostatics(
                self.scene, self.config.boundary, dtype=self.config.solver.dtype, device=self.config.device))
        self._validate_circuit_execution()
        self._validate_network_solver()
        self._validate_method_scene_features(self.config)
        self._validate_port_excitations()
        self._validate_trainable_rf_support()
        self._validate_breakdown_support()
        self._validate_initial_condition_support()
        waveport_excitation = self._waveport_excitation()
        if waveport_excitation is not None:
            prepared_scene = prepare_scene(self.scene, device=self.config.device)
            _require_cuda_device(prepared_scene.device, method="fdtd")
            return PreparedWavePortExcitation(
                self,
                prepared_scene,
                self._resolve_waveport_excitation_manifest(
                    prepared_scene,
                    waveport_excitation,
                ),
            )
        if self.port_sweep is not None:
            prepared_scene = prepare_scene(self.scene, device=self.config.device)
            _require_cuda_device(prepared_scene.device, method="fdtd")
            return PreparedNetworkSweep(
                self,
                prepared_scene,
                self._resolve_port_sweep_manifest(prepared_scene),
            )
        if self.method == SimulationMethod.MOM:
            if self.config.layered is not None:
                from .mom.layered.runtime import LayeredNetworkRuntime
                return PreparedSimulation(self, LayeredNetworkRuntime(self))
            if self.config.linear_coupled is not None:
                from .mom.linear_runtime import LinearCoupledRuntime
                return PreparedSimulation(self, LinearCoupledRuntime(self))
            if self.config.harmonic_balance is not None:
                from .mom.harmonic_runtime import HarmonicRuntime
                if self.config.dyadic_integration is not None:
                    from .mom.tensor_harmonic_runtime import MixedTensorHarmonicRuntime
                    return PreparedSimulation(self, MixedTensorHarmonicRuntime(self))
                return PreparedSimulation(self, HarmonicRuntime(self))
            from .mom.runtime import select_runtime
            return PreparedSimulation(self, select_runtime(self))
        if self.method == SimulationMethod.FDFD:
            if len(self.frequencies) > 1:
                from .fdfd.sweep import PreparedFDFDSweep
                return PreparedFDFDSweep(self, response_cache=response_cache)
            if self.config.local_response is not None:
                from .fdfd.response import PreparedLocalResponse
                return PreparedLocalResponse(self, self._build_fdfd_solver(), response_cache=response_cache)
            # Admit trainable inputs before any CUDA work; run() reuses this bridge.
            bridge = self._fdfd_gradient_bridge()
            return PreparedSimulation(self, self._build_fdfd_solver(), fdfd_gradient_bridge=bridge)
        elif self.method == SimulationMethod.FDTD:
            self._validate_trainable_parallel_fdtd()
            self._validate_magnetic_volume_dft()
            if self.config.parallel is not None and self.has_trainable_parameters:
                # Trainable multi-GPU FDTD runs through the distributed joint-solve
                # adjoint bridge; build and init the distributed solver, then run the
                # same medium/boundary/objective validators the bridge runs at run()
                # so prepare() genuinely surfaces every fail-closed guard before the
                # caller ever reaches run() (magnetic surface source terms, embedded
                # circuit coupling, and non-point/tiled-monitor objectives included).
                solver = self._build_fdtd_solver(initialize=True)
                from .fdtd.distributed.adjoint import (
                    require_distributed_adjoint_objective_support,
                    require_distributed_adjoint_support,
                )

                require_distributed_adjoint_support(solver)
                require_distributed_adjoint_objective_support(solver)
                return PreparedSimulation(self, solver)
            solver = self._build_fdtd_solver(initialize=True)
            if self.has_trainable_parameters:
                from .fdtd.adjoint.dispatch import validate_native_adjoint_preparation

                validate_native_adjoint_preparation(solver)
        else:
            raise ValueError(f"Unsupported simulation method {self.method!r}.")
        return PreparedSimulation(self, solver)

    def run(self, *, initial_harmonic_electric=None) -> Result:
        if initial_harmonic_electric is not None:
            if self.method != SimulationMethod.MOM or self.config.harmonic_balance is None:
                raise ValueError("Initial harmonic fields require an MoM harmonic-balance simulation")
            from .mom.harmonic_runtime import validate_initial_harmonic_electric
            initial_harmonic_electric = validate_initial_harmonic_electric(initial_harmonic_electric,self.config)
        if self.method in _STATIC_CONFIGS:
            return self.prepare().run()
        with requires_grad_scope():
            self._refresh_scene()
            self._validate_circuit_execution()
            self._validate_network_solver()
            self._validate_method_scene_features(self.config)
            self._validate_port_excitations()
            self._validate_trainable_rf_support()
            self._validate_breakdown_support()
            self._validate_initial_condition_support()
            if self.method == SimulationMethod.MOM:
                return self.prepare().run(initial_harmonic_electric=initial_harmonic_electric)
            if self.method == SimulationMethod.FDFD:
                if self.config.local_response is not None or len(self.frequencies) > 1:
                    with self.prepare() as prepared:
                        return prepared.run()
                return self._run_fdfd()
            if self.method != SimulationMethod.FDTD:
                raise ValueError(f"Unsupported simulation method {self.method!r}.")
            execute = self._prepare_fdtd_run()
        return execute()

    def _validate_initial_condition_support(self) -> None:
        """Fail closed for unsupported electrostatic pre-bias combinations.

        The pre-bias seeds the staggered E buffers of the single-device, real-field
        FDTD run before step 0. Distributed/multi-GPU seeding (shard-local E buffers
        with halo/ownership), differentiable/adjoint runs (the DC field enters the
        taped state), and complex (Bloch) field runs (the seed is real-valued and
        does not carry the Bloch phase) are genuine capability gaps, rejected here so
        they never silently drop or mis-seed the pre-bias.
        """
        from .electrostatic.initial_condition import ElectrostaticInitialCondition

        condition = getattr(self.config, "initial_condition", None)
        if condition is None:
            return
        if not isinstance(condition, ElectrostaticInitialCondition):
            raise TypeError(
                "FDTDConfig(initial_condition=...) must be an "
                "ElectrostaticInitialCondition (build it with "
                "ElectrostaticInitialCondition.from_result(dc_result))."
            )
        if self.method != SimulationMethod.FDTD:
            raise UnsupportedFeatureError(
                "Electrostatic pre-bias initial conditions seed the time-domain FDTD "
                "field buffers; the frequency-domain solver has no time-stepped state.",
                feature="guard:b45d40b5", method=self.method.value
            )
        if self.config.parallel is not None:
            raise UnsupportedFeatureError(
                "Distributed/multi-GPU FDTD does not support an electrostatic pre-bias "
                "initial condition yet: seeding shard-local staggered E buffers with the "
                "correct halo/ownership is a later phase. Run the pre-bias on a single GPU.",
                feature="guard:beee2d24", method="fdtd"
            )
        if self.has_trainable_parameters:
            raise UnsupportedFeatureError(
                "Trainable/adjoint FDTD does not support an electrostatic pre-bias initial "
                "condition yet: the seeded DC field would enter the taped forward state "
                "without a differentiated map back to the electrostatic solve. Detach the "
                "scene parameters to run a forward pre-bias.",
                feature="guard:8db918c9", method="fdtd"
            )
        if self.scene.boundary.uses_kind("bloch"):
            raise UnsupportedFeatureError(
                "Bloch-periodic FDTD carries complex phase-shifted fields; the real-valued "
                "electrostatic pre-bias seed does not track the Bloch phase. Use a real-field "
                "boundary (PEC/PMC/periodic) for a pre-charged run.",
                feature="boundary.bloch", method="fdtd"
            )

    def _validate_breakdown_support(self) -> None:
        """Fail closed for unsupported deterministic-breakdown combinations.

        Deterministic dynamic breakdown is a time-domain FDTD feature. The hard
        (non-smooth) feedback is not differentiable, so a trainable scene is
        rejected rather than silently running a forward-only solve. Multi-GPU is
        rejected by the distributed solver's static-capability guard.
        """
        from .compiler.breakdown import scene_has_breakdown

        if not scene_has_breakdown(self.scene):
            return
        if self.method != SimulationMethod.FDTD:
            raise UnsupportedFeatureError(
                "DielectricBreakdown is a time-domain (FDTD) feedback model; the "
                "frequency-domain solver has no dynamic conductivity update. Run the "
                "breakdown scene with the FDTD method.",
                feature="material.breakdown", method=self.method.value
            )
        if self.has_trainable_parameters:
            raise UnsupportedFeatureError(
                "Trainable scenes cannot enable hard dielectric breakdown: the "
                "field-duration/latching switch is non-differentiable at the trigger "
                "time, so prepare() rejects backward rather than fabricating a gradient. "
                "For optimization use the separate non-feedback SmoothBreakdownRisk "
                "surrogate (mw.SmoothBreakdownRisk), which never drives the field solve.",
                feature="material.breakdown", method="fdtd"
            )

    def _validate_network_solver(self) -> None:
        # Static methods refuse Scene.networks through their method tables.
        if self.scene.networks and self.method in (SimulationMethod.FDFD, SimulationMethod.MOM):
            raise UnsupportedFeatureError(
                "Embedded network feedback is defined only for the time-domain FDTD "
                "update; frequency-domain solvers cannot ignore Scene.networks.",
                feature="network", method=self.method.value
            )

    def _reject_embedded_network_port_conflicts(self) -> None:
        """Reject excitation/termination on network-connected ports on every path.

        The single-device runtime enforces this during port-runtime preparation
        (see ``witwin/maxwell/fdtd/ports.py``). The multi-GPU path builds its own
        network port runtimes and hardcodes ``resistance=0.0`` with no excitation,
        so without this guard a PortExcitation, PortSweep, or ``port.termination``
        aimed at a network-connected port would be silently dropped rather than
        rejected. Running this here keeps the rejection identical across both the
        single-device and distributed paths.
        """

        network_by_port = {
            port_name: network.name
            for network in self.scene.networks
            for port_name in network.connected_port_names
        }
        if not network_by_port:
            return

        def _reject(port_name: str) -> None:
            embedded_network_name = network_by_port.get(port_name)
            if embedded_network_name is not None:
                raise ValueError(
                    f"Port {port_name!r} connected to embedded network "
                    f"{embedded_network_name!r} cannot also declare an excitation "
                    "or termination."
                )

        if self.port_sweep is not None:
            swept_ports = self.port_sweep.ports
            if swept_ports is None:
                swept_ports = tuple(port.name for port in self.scene.ports)
            for port_name in swept_ports:
                _reject(port_name)
        for excitation in self.excitations:
            _reject(excitation.port_name)
        for port in self.scene.ports:
            if getattr(port, "termination", None) is not None:
                _reject(port.name)

    def _validate_port_excitations(self) -> None:
        self._reject_embedded_network_port_conflicts()
        bound_port_names = {
            binding.port_name
            for circuit in self.scene.circuits
            for binding in circuit.bindings
        }
        if self.port_sweep is not None:
            if bound_port_names:
                raise UnsupportedFeatureError(
                    "Circuit-bound ports do not support PortSweep execution.",
                    feature="circuit", method="fdtd"
                )
            if any(isinstance(port, TerminalWavePort) for port in self.scene.ports):
                return
            if any(isinstance(port, WavePort) for port in self.scene.ports):
                if self.port_sweep.amplitude.requires_grad:
                    raise UnsupportedFeatureError(
                        "PortSweep amplitude is trainable, but WavePort sweeps do not "
                        "support RF-parameter gradients.",
                        feature="port.WavePort", method="fdtd"
                    )
                return
            from .network_sweep import resolve_network_run_manifest

            resolve_network_run_manifest(self.scene, self.port_sweep, self.frequencies)
            return
        if not self.excitations:
            return
        ports_by_name = {port.name: port for port in self.scene.ports}
        if len(self.excitations) > 1 and any(
            isinstance(ports_by_name.get(excitation.port_name), WavePort)
            for excitation in self.excitations
        ):
            raise UnsupportedFeatureError(
                "A direct multi-source FDTD run supports LumpedPort and TerminalPort only; "
                "WavePort channels require calibrated per-frequency execution.",
                feature="port.WavePort", method="fdtd"
            )
        for excitation in self.excitations:
            if isinstance(ports_by_name.get(excitation.port_name), TerminalWavePort):
                raise UnsupportedFeatureError(
                    "A TerminalWavePort is driven through PortSweep columns only; a direct "
                    "PortExcitation of one terminal or mode is not yet available.",
                    feature="port.TerminalWavePort", method="fdtd"
                )
            if excitation.port_name in bound_port_names:
                raise ValueError(
                    f"Circuit-bound port {excitation.port_name!r} cannot also declare "
                    "a direct PortExcitation."
                )
            port = ports_by_name.get(excitation.port_name)
            if port is None:
                raise ValueError(
                    f"PortExcitation references missing port {excitation.port_name!r}."
                )
            if not isinstance(port, (LumpedPort, TerminalPort, WavePort, CoaxialLumpedPort)):
                raise TypeError(
                    f"PortExcitation {excitation.port_name!r} requires an RF port, "
                    f"got {type(port).__name__}."
                )
            if isinstance(port, WavePort):
                if excitation.source_impedance != "matched":
                    raise ValueError(
                        "WavePort PortExcitation requires source_impedance='matched'."
                    )
                if excitation.source_time is not None:
                    raise ValueError(
                        "WavePort PortExcitation uses calibrated per-frequency CW runs; "
                        "source_time must be None."
                    )
                if excitation.amplitude.requires_grad:
                    raise UnsupportedFeatureError(
                        "WavePort PortExcitation does not yet support trainable amplitude.",
                        feature="port.WavePort", method="fdtd"
                    )
                mode_names = tuple(mode.name for mode in port.modes)
                qualified_names = tuple(port.mode_name(mode) for mode in port.modes)
                if (
                    excitation.mode_name is not None
                    and excitation.mode_name not in mode_names
                    and excitation.mode_name not in qualified_names
                ):
                    raise ValueError(
                        f"WavePort {port.name!r} has no mode {excitation.mode_name!r}; "
                        f"choices are {mode_names}."
                    )
            elif excitation.mode_name is not None:
                raise ValueError("mode_name is supported by WavePort excitations only.")

    def _validate_trainable_rf_support(self) -> None:
        if self.method != SimulationMethod.FDTD or not self.has_trainable_parameters:
            return
        if any(isinstance(port, TerminalWavePort) for port in self.scene.ports):
            raise UnsupportedFeatureError(
                "TerminalWavePort sweeps are forward-only: the multi-conductor modal basis, "
                "terminal integrals and calibrated CW columns have no FDTD adjoint.",
                feature="port.TerminalWavePort", method="fdtd"
            )
        if any(isinstance(item, (CoaxialLumpedPort, CoaxialLumpedResistor))
               for item in (*self.scene.ports, *self.scene.lumped_elements)):
            raise UnsupportedFeatureError(
                "CoaxialLumpedPort and CoaxialLumpedResistor are forward-only: the FDTD adjoint "
                "does not replay their multi-component annular branch state.",
                feature="port.CoaxialLumpedPort", method="fdtd"
            )
        for block in self.scene.networks:
            model = block.model
            if isinstance(model, RationalModel):
                unsupported = tuple(
                    name
                    for name in ("poles", "proportional")
                    if getattr(model, name).requires_grad
                )
                if unsupported:
                    raise UnsupportedFeatureError(
                        "Differentiable embedded RationalModel supports residues and direct "
                        f"terms only; trainable {unsupported!r} are not supported.",
                        feature="network", method="fdtd"
                    )
            elif isinstance(model, StateSpaceNetwork) and any(
                getattr(model, name).requires_grad for name in ("A", "B", "C", "D")
            ):
                raise UnsupportedFeatureError(
                    "Differentiable embedded networks accept trainable residues/direct on a "
                    "pre-fitted RationalModel; direct trainable state-space matrices are not supported.",
                    feature="network", method="fdtd"
                )
        rf_ports = tuple(
            port
            for port in self.scene.ports
            if isinstance(port, (LumpedPort, TerminalPort, WavePort))
        )
        wave_ports = tuple(port for port in rf_ports if isinstance(port, WavePort))
        lumped_ports = tuple(
            port for port in rf_ports if isinstance(port, (LumpedPort, TerminalPort))
        )
        uses_waveport_workflow = bool(wave_ports) and bool(
            self.port_sweep is not None or self._waveport_excitation() is not None
        )
        if uses_waveport_workflow:
            if len(wave_ports) != len(rf_ports) or self.scene.lumped_elements:
                raise UnsupportedFeatureError(
                    "Differentiable WavePort runs cannot include lumped/terminal ports "
                    "or standalone R/L/C elements because their auxiliary state is not "
                    "replayed by the FDTD adjoint.",
                    feature="port.WavePort", method="fdtd"
                )
            multimode = tuple(port.name for port in wave_ports if len(port.modes) != 1)
            if multimode:
                raise UnsupportedFeatureError(
                    "Differentiable WavePort runs currently require exactly one fixed mode "
                    f"per aperture; multimode ports are {multimode}.",
                    feature="port.WavePort", method="fdtd"
                )
            return
        if wave_ports:
            raise UnsupportedFeatureError(
                "Differentiable WavePort declarations require a fixed single-mode direct "
                "excitation or WavePort sweep and cannot be mixed into a lumped workflow.",
                feature="port.WavePort", method="fdtd"
            )
        if self.port_sweep is not None and lumped_ports:
            raise UnsupportedFeatureError(
                "Differentiable LumpedPort/TerminalPort PortSweep is not yet supported; "
                "use direct port excitations.",
                feature="port", method="fdtd"
            )
        # Exact circuit topology, field-medium, and RF-parameter guards are applied
        # after the single-device runtimes are compiled. Supported series lumped
        # ports and standalone R/C/L elements proceed through the adjoint bridge.

    def _refresh_scene(self):
        if self.method == SimulationMethod.FDTD:
            from ._fixed_inputs import require_fixed_input
            require_fixed_input((self.frequency, self.frequencies), name="Simulation frequencies")
            _validate_fixed_fdtd_controls(self.config)
        if self.scene_module is not None:
            self.scene = self.scene_module.to_scene()
        elif isinstance(self.scene_input, Scene):
            self.scene = self.scene_input
        self._refresh_trainable_parameters()
        self._validate_pcb_material_frequency()

    def _refresh_trainable_parameters(self, *, config=None, frequencies=None) -> None:
        if (self.method == SimulationMethod.FDTD and torch.is_grad_enabled()
                and self.scene.coordinates is None):
            # Mapped scenes run their own path-level admission below.
            _reject_uncensused_trainable_tensors(self.scene)
        self.has_trainable_parameters = bool(
            any(
                parameter.requires_grad
                for parameter in (
                    self.scene_module.parameters() if self.scene_module is not None else ()
                )
            )
            or _scene_trainable_census(self.scene)
            or _trainable_rf_parameters(
                self.scene,
                excitations=self.excitations,
                port_sweep=self.port_sweep,
            )
        )
        if self.scene.coordinates is not None:
            from .fdtd.adjoint.mapped_admission import assess_mapped_materials
            dependency_scene = self.scene
            if self.scene_module is not None and not torch.is_grad_enabled():
                # Custom autograd forwards deliberately compile/run without a
                # graph. Recover only the declarative dependency census, leaving
                # that numerical Scene and forward execution unchanged.
                from .fdtd.adjoint.mapped_admission import capture_grad_mode_scene, validate_grad_mode_scene
                numerical_snapshot = capture_grad_mode_scene(self.scene)
                with torch.enable_grad():
                    dependency_scene = self.scene_module.to_scene()
                validate_grad_mode_scene(self.scene, dependency_scene, snapshot=numerical_snapshot)
            admission = assess_mapped_materials(
                dependency_scene, module=self.scene_module,
                config=config if config is not None else getattr(self, "config", None),
                excitations=self.excitations, port_sweep=self.port_sweep,
                frequencies=frequencies if frequencies is not None else getattr(self, "frequencies", None),
            )
            self._mapped_material_admission = admission
            self.has_trainable_parameters = self.has_trainable_parameters or admission is not None

    def _validate_circuit_execution(self) -> None:
        if not self.scene.circuits:
            return
        compile_circuits(prepare_scene(self.scene, device=self.config.device))
        if self.method != SimulationMethod.FDTD:
            raise ValueError("Circuit-coupled scenes are supported by Simulation.fdtd(...) only.")
        if len(self.scene.circuits) != 1 or not self.scene.circuits[0].bindings:
            raise UnsupportedFeatureError(
                "Circuit-coupled FDTD requires one circuit with at least one bound port; "
                "multi-circuit execution is not yet supported.",
                feature="circuit", method="fdtd"
            )

    def check(self) -> CapabilityReport:
        """Classify every scene feature against this simulation's method table.

        Never raises for a refusal; construction, ``prepare()`` and ``run()``
        raise the refused items through ``CapabilityReport.raise_if_unsupported()``.
        """
        return self._capability_report(self.config)

    def _capability_report(self, config) -> CapabilityReport:
        from .layers import LayerStack
        method = self.method.value
        if self.method == SimulationMethod.MOM and isinstance(self.scene.background, LayerStack):
            method = "mom.layered"
        return check_scene(self.scene, method, frequencies=self.frequencies, config=config)

    def _validate_method_scene_features(self, config) -> None:
        self._capability_report(config).raise_if_unsupported()
        if self.scene.coordinates is not None:
            from .compiler.coordinates import validate_coordinate_scene
            validate_coordinate_scene(self.scene,method=self.method.value)
            if getattr(config, "material_cache", None) is not None:
                raise UnsupportedFeatureError("Mapped material cache reuse needs coordinate-aware coefficient identity.",
                                              feature="coordinates", method="fdtd")

    def _validate_pcb_material_frequency(self):
        record = self.scene.metadata.get("pcb_import", {})
        if not isinstance(record, dict) or record.get("schema") != "pcb-import/1":
            return
        reference = record.get("loss_reference_frequency_hz")
        if reference is not None and any(not math.isclose(f, reference, rel_tol=1e-12, abs_tol=0)
                                         for f in self.frequencies):
            raise ValueError("Imported tangent loss is conductivity at one reference frequency; "
                             "reimport for each frequency or declare a broadband dispersive material.")

    def _run_fdfd(self) -> Result:
        bridge = self._fdfd_gradient_bridge()
        if bridge is not None:
            from .fdfd.adjoint import run_fdfd_with_gradient_bridge

            return run_fdfd_with_gradient_bridge(self, bridge=bridge)
        solver = self._build_fdfd_solver()
        solver_cfg = self.config.solver
        return self._build_fdfd_result(solver, solver_cfg)

    def _fdfd_requires_gradient(self) -> bool:
        """Whether this single-frequency FDFD solve must carry autograd history.

        FDFD solves in CuPy, so fields never inherit a graph: any grad-carrying
        scene tensor, including ones no trainable census channel lists, must go
        through the adjoint bridge (which differentiates or refuses it by name).
        """
        return torch.is_grad_enabled() and bool(self.has_trainable_parameters or requires_grad(self.scene))

    def _fdfd_gradient_bridge(self):
        """The admitted FDFD adjoint bridge, or None for a detached solve."""
        if not self._fdfd_requires_gradient():
            return None
        from .fdfd.adjoint.bridge import _FDFDGradientBridge

        return _FDFDGradientBridge(self)

    def _build_fdfd_solver(self):
        if self.scene.subpixel.averaging == SubpixelAveraging.POLARIZED_TENSOR:
            raise ValueError("polarized_tensor subpixel averaging is FDTD-only; use averaging='polarized' for FDFD")
        prepared_scene = prepare_scene(self.scene, device=self.config.device)
        _require_cuda_device(prepared_scene.device, method="fdfd")
        fdfd_backend = _resolve_fdfd_backend()
        solver_cfg = self.config.solver
        return fdfd_backend(
            prepared_scene,
            frequency=self.frequency,
            solver_type=solver_cfg.solver_type,
            preconditioner=solver_cfg.preconditioner,
            precision=solver_cfg.precision,
            ssor_omega=solver_cfg.ssor_omega,
            device_memory_limit_bytes=solver_cfg.device_memory_limit_bytes,
            enable_plot=self.config.enable_plot,
            verbose=self.config.verbose,
        )

    def _build_fdfd_result(self, solver, solver_cfg: FDFDSolverConfig) -> Result:
        solver.solve(
            max_iter=solver_cfg.max_iter,
            tol=solver_cfg.tol,
            restart=solver_cfg.restart,
        )
        if solver.E_field is None:
            raise RuntimeError("FDFD solve did not produce any field data.")

        fields = {
            "EX": solver.E_field[0],
            "EY": solver.E_field[1],
            "EZ": solver.E_field[2],
        }
        return Result(
            method="fdfd",
            scene=self.scene,
            prepared_scene=solver.scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=solver,
            fields=fields,
            monitors=self._fdfd_projection_monitors(solver, tuple(fields.values())),
            metadata=self.metadata,
            solver_stats=self._fdfd_solver_stats(solver, solver_cfg),
        )

    @staticmethod
    def _fdfd_solver_stats(solver, solver_cfg: FDFDSolverConfig) -> dict:
        return {
            "pml_extrusions": solver.scene.pml_extrusions,
            "solver": {
                "type": solver_cfg.solver_type,
                "max_iter": solver_cfg.max_iter,
                "tol": solver_cfg.tol,
                "restart": solver_cfg.restart,
                "preconditioner": solver_cfg.preconditioner,
                "precision": solver_cfg.precision,
            },
            "converged": getattr(solver, "converged", None),
            "solver_info": getattr(solver, "solver_info", None),
            "final_residual": getattr(solver, "final_residual", None),
        }

    def _fdfd_projection_monitors(self, solver, fields):
        if not self.config.observations:
            return {}
        from .fdfd.observations import compile_projection, validate_farfield_support
        from .fdfd.results import observation_data
        validate_farfield_support(self.config.observations, solver)
        electric = torch.cat([field.flatten() for field in fields])
        projection = compile_projection(self.config.observations, solver.scene, electric.dtype, self.frequency)
        values = torch.sparse.mm(projection, electric[:, None])[:, 0]
        return observation_data(self.config.observations, values, self.frequency)

    def _run_fdtd(self) -> Result:
        with requires_grad_scope():
            execute = self._prepare_fdtd_run()
        return execute()

    def _prepare_fdtd_run(self):
        """Resolve the run path and initialize its solver before time stepping."""
        self._validate_trainable_parallel_fdtd()
        self._validate_magnetic_volume_dft()
        if self.config.parallel is not None and self.has_trainable_parameters:
            return self._run_distributed_fdtd_with_gradient_bridge
        if self.port_sweep is not None:
            return self._run_fdtd_network_sweep
        if self.modal_port_sweep is not None:
            from .modal_sweep import run_modal_port_sweep

            return lambda: run_modal_port_sweep(self, self.modal_port_sweep)
        waveport_excitation = self._waveport_excitation()
        if waveport_excitation is not None:
            return lambda: self._run_fdtd_waveport_excitation(waveport_excitation)
        if self.has_trainable_parameters:
            return self._run_fdtd_with_gradient_bridge
        solver = self._build_fdtd_solver(initialize=True)
        return lambda: self._run_fdtd_from_solver(solver)

    def _waveport_excitation(self):
        if len(self.excitations) != 1:
            return None
        excitation = self.excitations[0]
        port = next(
            (port for port in self.scene.ports if port.name == excitation.port_name),
            None,
        )
        return excitation if isinstance(port, WavePort) else None

    def _resolve_waveport_excitation_manifest(self, scene, excitation):
        from .waveport_sweep import resolve_waveport_run_manifest

        return resolve_waveport_run_manifest(
            scene,
            PortSweep(ports=(excitation.port_name,), amplitude=excitation.amplitude),
            self.frequencies,
            requested_dt=self.config.run_time.dt,
            base_frequency=self.frequency,
            max_electric_field=self.config.run_time.max_electric_field,
            device=self.config.device,
        )

    def _run_fdtd_waveport_excitation(self, excitation, *, scene=None, manifest=None):
        from .waveport_sweep import run_waveport_excitation

        run_scene = self.scene if scene is None else scene
        if manifest is None:
            manifest = self._resolve_waveport_excitation_manifest(
                run_scene,
                excitation,
            )
        return run_waveport_excitation(
            self,
            run_scene,
            excitation,
            manifest,
        )

    def _run_fdtd_network_sweep(self, *, scene=None, manifest=None) -> Result:
        from .network_sweep import (
            aggregate_network_columns,
            build_network_column_run,
        )

        sweep = self.port_sweep
        run_scene = self.scene if scene is None else scene
        if manifest is None:
            manifest = self._resolve_port_sweep_manifest(run_scene)
        from .waveport_sweep import WavePortRunManifest, run_waveport_sweep

        if isinstance(manifest, WavePortRunManifest):
            return run_waveport_sweep(self, run_scene, manifest)
        from .terminal_waveport import TerminalWavePortRunManifest, run_terminal_waveport_sweep

        if isinstance(manifest, TerminalWavePortRunManifest):
            return run_terminal_waveport_sweep(self, run_scene, manifest)
        columns = []
        column_results = []
        column_stats = []
        last_solver = None
        shared_prepared_scene = None
        from .array_execution import compact_array_column_result

        for active_name in manifest.port_names:
            excitations, overrides = build_network_column_run(
                run_scene,
                sweep,
                manifest,
                active_name,
            )
            solver = self._build_fdtd_solver_for_scene(
                run_scene,
                initialize=True,
                port_excitations=excitations,
                termination_overrides=overrides,
            )
            column_result = self._run_fdtd_from_solver(solver)
            columns.append(column_result._ports)
            if shared_prepared_scene is None:
                shared_prepared_scene = column_result.prepared_scene
            column_results.append(
                compact_array_column_result(
                    column_result,
                    prepared_scene=shared_prepared_scene,
                )
            )
            column_stats.append(column_result.stats())
            last_solver = solver

        stacked_ports, network = aggregate_network_columns(manifest, tuple(columns))
        incident = torch.stack(
            [
                column[port_name].a
                for column, port_name in zip(columns, manifest.port_names)
            ],
            dim=-1,
        )
        from .array_execution import ArrayRunData

        metadata = dict(self.metadata)
        metadata["network_run_manifest"] = manifest.metadata()
        metadata["conductor_loss_columns"] = {
            name: result._metadata["conductor_losses"] for name, result in zip(manifest.port_names, column_results)
            if result._metadata.get("conductor_losses")}
        return Result(
            method="fdtd",
            scene=scene_declarations(run_scene),
            prepared_scene=last_solver.scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=last_solver,
            fields={},
            monitors={},
            ports=stacked_ports,
            network=network,
            array_run_data=ArrayRunData(
                manifest=manifest,
                column_results=tuple((result,) for result in column_results),
                incident=incident,
            ),
            metadata=metadata,
            solver_stats={
                "network_sweep": manifest.metadata(),
                "columns": tuple(column_stats),
            },
            raw_output={"network_run_manifest": manifest.metadata(), "conductor_losses": {}},
        )

    def _run_fdtd_network_sweep_ensemble(self, execution, *, scene=None) -> Result:
        """Distribute independent excitation columns of an RF sweep over GPUs.

        Each column is an independent single-active-port FDTD run, so they map
        cleanly onto the shared ensemble executor. Plan 01's column builder and
        matrix assembler are reused verbatim; the executor only guarantees
        ordered execution, device leasing and a submission-ordered sequence, and
        this method assembles the identical ``NetworkData`` matrix as the serial
        path from the per-column ``PortData`` gathered onto one result device.
        """

        from .execution.capacity import estimate_scene_footprint_bytes
        from .execution.ensemble import MultiGPUExecution
        from .execution.executor import execute_plan
        from .execution.plan import ExecutionPlan, ExecutionTask
        from .network_sweep import aggregate_network_columns, build_network_column_run
        from .waveport_sweep import WavePortRunManifest

        if not isinstance(execution, MultiGPUExecution):
            raise TypeError("execution must be a maxwell.execution.MultiGPUExecution.")
        if self.config.parallel is not None:
            raise ValueError(
                "A network sweep cannot combine ensemble execution with an "
                "FDTDParallelConfig joint solve on the same Simulation."
            )
        if self.has_trainable_parameters:
            raise ValueError(
                "Ensemble execution does not run the FDTD adjoint through run_many; "
                "run the trainable sweep without an execution descriptor."
            )

        run_scene = self.scene if scene is None else scene
        manifest = self._resolve_port_sweep_manifest(run_scene)
        if isinstance(manifest, WavePortRunManifest):
            raise ValueError(
                "Ensemble execution of a WavePort sweep is not yet available; run the "
                "WavePort sweep without an execution descriptor."
            )
        from .terminal_waveport import TerminalWavePortRunManifest

        if isinstance(manifest, TerminalWavePortRunManifest):
            raise ValueError(
                "Ensemble execution of a TerminalWavePort sweep is not yet available; run the "
                "sweep without an execution descriptor."
            )

        sweep = self.port_sweep
        result_device = torch.device(execution.devices[0])
        estimated = estimate_scene_footprint_bytes(
            run_scene,
            frequencies=self.frequencies,
            full_field_dft=_resolve_full_field_dft(
                self.config.full_field_dft, self.frequencies, run_scene.ports
            ),
            magnetic_volume_dft=self.config.records_magnetic_volume_dft,
        )

        def _make_run(active_name):
            def _run(device):
                excitations, overrides = build_network_column_run(
                    run_scene,
                    sweep,
                    manifest,
                    active_name,
                )
                solver = self._build_fdtd_solver_for_scene(
                    run_scene,
                    device=str(device),
                    initialize=True,
                    port_excitations=excitations,
                    termination_overrides=overrides,
                )
                return self._run_fdtd_from_solver(solver)

            return _run

        tasks = tuple(
            ExecutionTask(
                index=index,
                run=_make_run(active_name),
                estimated_bytes=estimated,
                label=f"column[{active_name}]",
            )
            for index, active_name in enumerate(manifest.port_names)
        )
        plan = ExecutionPlan(
            tasks=tasks,
            placement=execution.placement,
            max_concurrency=execution.max_concurrency,
            fail_fast=execution.fail_fast,
        )
        pool = execution.build_pool(require_cuda=True)
        sequence = execute_plan(plan, pool)
        column_results = sequence.results()

        columns = tuple(
            _move_port_mapping_to_device(result._ports, result_device)
            for result in column_results
        )
        column_stats = tuple(result.stats() for result in column_results)
        stacked_ports, network = aggregate_network_columns(manifest, columns)
        # Present a device-consistent Result: the assembled network/ports live on
        # result_device, so pick the column that actually ran on result_device for
        # the representative solver/prepared_scene rather than the last column
        # (which may have leased a different GPU). At least one column always leases
        # result_device (devices[0]) since it is the first device in pool order.
        result_device_str = str(result_device)
        representative = next(
            (
                result
                for record, result in zip(sequence.records, column_results)
                if record.device == result_device_str
            ),
            column_results[-1],
        )
        metadata = dict(self.metadata)
        metadata["network_run_manifest"] = manifest.metadata()
        metadata["ensemble_execution"] = {
            "devices": tuple(execution.devices),
            "placement": execution.placement,
            "column_devices": tuple(record.device for record in sequence.records),
        }
        metadata["conductor_loss_columns"] = {
            name: result._metadata["conductor_losses"] for name, result in zip(manifest.port_names, column_results)
            if result._metadata.get("conductor_losses")}
        return Result(
            method="fdtd",
            scene=run_scene,
            prepared_scene=representative.prepared_scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=representative.solver,
            fields={},
            monitors={},
            ports=stacked_ports,
            network=network,
            metadata=metadata,
            solver_stats={
                "network_sweep": manifest.metadata(),
                "columns": column_stats,
                "ensemble": metadata["ensemble_execution"],
            },
            raw_output={"network_run_manifest": manifest.metadata(), "conductor_losses": {}},
        )

    def _resolve_port_sweep_manifest(self, scene):
        declared = scene_declarations(scene)
        if any(isinstance(port, TerminalWavePort) for port in declared.ports):
            from .terminal_waveport import resolve_terminal_waveport_run_manifest

            return resolve_terminal_waveport_run_manifest(
                scene,
                self.port_sweep,
                self.frequencies,
                requested_dt=self.config.run_time.dt,
                base_frequency=self.frequency,
                max_electric_field=self.config.run_time.max_electric_field,
                device=self.config.device,
            )
        if any(isinstance(port, WavePort) for port in declared.ports):
            from .waveport_sweep import resolve_waveport_run_manifest

            return resolve_waveport_run_manifest(
                scene,
                self.port_sweep,
                self.frequencies,
                requested_dt=self.config.run_time.dt,
                base_frequency=self.frequency,
                max_electric_field=self.config.run_time.max_electric_field,
                device=self.config.device,
            )
        from .network_sweep import resolve_network_run_manifest

        return resolve_network_run_manifest(
            declared,
            self.port_sweep,
            self.frequencies,
        )

    def _apply_initial_condition(self, solver) -> None:
        """Seed the solver's E buffers with the electrostatic pre-bias, if any.

        Runs after ``init_field()`` (the compiled edge permittivity and the zeroed
        field/CPML buffers exist) and records the mapping provenance on the solver so
        the run result can surface it. Fail-closed guards ran already in
        ``_validate_initial_condition_support``.
        """
        condition = getattr(self.config, "initial_condition", None)
        if condition is None:
            return
        condition.apply_to_solver(solver)
        solver._electrostatic_prebias_provenance = dict(condition.provenance)

    def _build_fdtd_solver(self, *, initialize: bool):
        return self._build_fdtd_solver_for_scene(self.scene, initialize=initialize)

    def _validate_magnetic_volume_dft(self) -> None:
        """Refuse an H volume DFT on a differentiable run: the adjoint seeds only the E phasors."""
        if self.config.records_magnetic_volume_dft and self.has_trainable_parameters:
            raise UnsupportedFeatureError(
                "The FDTD adjoint does not differentiate the magnetic volume DFT; record Hx/Hy/Hz "
                "(volume_dft_components) in a forward-only run.", feature="guard:f6b6310e", method="fdtd")

    def _validate_trainable_parallel_fdtd(self) -> None:
        """Capability-scoped validation of a trainable multi-GPU FDTD run.

        The distributed joint-solve adjoint bridge differentiates Box
        material-region densities on the pure real standard path. Every other
        trainable channel -- structure geometry, material perturbation tensors,
        circuit parameters, and RF/port/excitation parameters -- has no verified
        distributed reverse core, so a trainable+parallel scene carrying one is
        rejected here at prepare/run before any distributed allocation. The
        remaining medium/boundary/objective guards are enforced by the distributed
        solver and the distributed adjoint bridge once the solver is built.
        """

        if self.config.parallel is None or not self.has_trainable_parameters:
            return
        if self.scene.thin_wires:
            raise ValueError(
                "Multi-GPU FDTD adjoint does not cover thin-wire state; a trainable scene "
                "carrying a ThinWire has no distributed wire reverse (the Phase 7 adjoint "
                "bridge does not checkpoint or replay wire I/q). Run forward-only without "
                "trainable parameters, or use the single-GPU adjoint path by omitting parallel."
            )
        unsupported = (
            _scene_trainable_geometry_parameters(self.scene)
            + _scene_trainable_material_parameters(self.scene)
            + _scene_trainable_circuit_parameters(self.scene)
            + _trainable_rf_parameters(
                self.scene,
                excitations=self.excitations,
                port_sweep=self.port_sweep,
            )
        )
        if self.excitations or self.port_sweep is not None:
            raise ValueError(
                "Multi-GPU FDTD adjoint does not support RF port excitation or port "
                "sweeps; the distributed reverse has no port/excitation channel yet."
            )
        if unsupported:
            raise ValueError(
                "Multi-GPU FDTD adjoint supports trainable Box material-region densities "
                "only; trainable geometry, material perturbation, circuit, embedded-network, "
                "and RF/port parameters have no distributed reverse core yet. Use the "
                "single-GPU adjoint path by omitting parallel."
            )
        require_material_gradient_channels(self.scene)
        # Scene/config-static guards for the trainable distributed path, all raised
        # here before the distributed solver allocates any shard. The pure real
        # standard reverse is the only verified distributed adjoint core, so any
        # absorbing boundary, a dispersive/conductive/nonlinear/anisotropic/modulated
        # medium, field shutoff, multi-source normalization, or a tiled-monitor
        # objective is rejected up front rather than after a full forward.
        #
        # The absorber only activates when the boundary declares a PML kind
        # (fdtd/boundary/runtime.py sets active_absorber_type from absorber_type only
        # then). The verified distributed adjoint envelope covers two boundary
        # regimes: the open-boundary standard update AND the CPML absorbing update
        # (absorber "cpml"/"stablepml"), whose parity/FD gates run on an x-CPML
        # dielectric scene. The legacy graded-sigma absorbers ("pml"/"absorber")
        # have no verified distributed reverse core and stay rejected here so no
        # unverified absorber slips through. The distributed solver additionally
        # asserts the x-CPML pinning invariant before trusting the per-shard reverse.
        if self.scene.boundary.uses_kind("pml"):
            from .fdtd.boundary.runtime import resolve_absorber
            absorber_name, _ = resolve_absorber(
                self.scene, self.config.absorber,
                complex_fields=self.scene.boundary.uses_kind("bloch"), distributed=True,
            )
            if absorber_name not in ("cpml", "stablepml"):
                raise ValueError(
                    "Multi-GPU FDTD adjoint supports open/PEC or CPML absorbing "
                    f"boundaries only; the graded-sigma absorber={absorber_name!r} has "
                    "no verified distributed reverse core. Use absorber='cpml' (or "
                    "'stablepml') for an absorbing trainable distributed run."
                )
        medium_reason = _nonstandard_medium_reason(self.scene)
        if medium_reason is not None:
            raise ValueError(
                f"Multi-GPU FDTD adjoint supports the pure real standard medium only; {medium_reason}."
            )
        if float(getattr(self.config, "shutoff", 0.0)) > 0.0:
            raise ValueError(
                "Multi-GPU FDTD adjoint does not support field shutoff (shutoff>0) on "
                "trainable runs; the reverse pass replays a fixed step count."
            )
        if self.config.spectral_sampler.normalize_source and len(self.scene.sources) != 1:
            raise ValueError(
                "Multi-GPU FDTD adjoint source normalization requires exactly one logical source."
            )
        tiled_reason = _tiled_monitor_objective_reason(self.scene)
        if tiled_reason is not None:
            raise ValueError(
                "Multi-GPU FDTD adjoint objectives support point-monitor spectra and "
                f"full-field DFT only; {tiled_reason}."
            )

    def _configure_fdtd_solver_time(self, solver) -> None:
        """Bind explicit or optimization-stable timing before material initialization."""
        solver._spectral_reference_frequency = self.config.reference_frequency
        solver._requested_dt = (
            self.config.run_time.dt
            if self.config.run_time.dt is not None
            else getattr(self, "_fdtd_optimization_dt", None)
        )
        solver._max_electric_field = self.config.run_time.max_electric_field
        solver._requested_source_frequency = self._fdtd_time_step_frequency(
            getattr(solver, "logical_scene", solver.scene),
            excitations=getattr(solver, "_port_excitations", self.excitations),
        )

    def _remember_fdtd_solver_time(self, solver) -> None:
        if self.has_trainable_parameters and self.config.run_time.dt is None:
            self._fdtd_optimization_dt = float(solver.dt)

    def _fdtd_time_step_frequency(self, scene=None, *, excitations=None) -> float:
        from .fdtd.runtime.initialization import maximum_declared_frequency

        return maximum_declared_frequency(
            scene_declarations(self.scene if scene is None else scene),
            self.frequency,
            self.excitations if excitations is None else excitations,
        )

    @requires_grad_scope(reuse=True)
    def _build_fdtd_solver_for_scene(
        self,
        scene,
        *,
        initialize: bool,
        device=None,
        port_excitations=None,
        termination_overrides=None,
        batch_lane: bool = False,
    ):
        options = dict(scene=scene, device=self.config.device if device is None else device,
                       initialize=initialize, port_excitations=port_excitations,
                       termination_overrides=termination_overrides)
        # Every FDTD scene runs on the core path (W7 row 16): CPU scenes are refused by the
        # CUDA requirement, every public source and monitor is admitted, and features no path
        # supports refuse in their own compilers. A batch lane's NativeFieldSlot binds the core
        # Yee launches to its lane storage and owns the lane's sources, observers and adjoint
        # checkpoints, so its solver is prepared without core activation.
        options["batch_lane"] = batch_lane
        if self.config.field_dtype == torch.float64:
            from .fdtd.runtime.precision import validate_fp64_forward
            validate_fp64_forward(self, scene_declarations(scene))
        if self.config.material_cache is not None and (self.config.parallel is not None or self.has_trainable_parameters):
            raise UnsupportedFeatureError("FDTDMaterialCache requires single-device detached forward execution",
                                          feature="guard:5839b12d", method="fdtd")
        owner = self.scene_module if self.scene_module is not None else scene
        # These are the only preparation outputs written back to public inputs.
        # Restore them before rebuilding so failed mode compilation cannot seed the retry.
        state = [(MODE_SOURCE_GAUGES, owner, owner in MODE_SOURCE_GAUGES, MODE_SOURCE_GAUGES.get(owner)),
                 (vars(self), "_fdtd_optimization_dt", "_fdtd_optimization_dt" in vars(self),
                  vars(self).get("_fdtd_optimization_dt"))]
        try:
            try:
                solver = self._initialize_fdtd_solver_for_scene(**options)
            except _AcceleratedSetupFailure as failure:
                # A failed accelerated setup retries once with the torch setup on the same path
                # (the material-cache numerics); a second failure, and every activation or
                # core-load failure, raises.
                self._restore_fdtd_preparation_inputs(state)
                solver = self._initialize_fdtd_solver_for_scene(**options, setup_auto_allowed=False)
                cause = failure.__cause__
                warnings.warn(f"Accelerated CUDA setup failed ({type(cause).__name__}: {cause}); "
                              "the FDTD solver was prepared with the torch setup.",
                              category=MaxwellWarning, stacklevel=3)
        except Exception:
            # A failed preparation never seeds the next one.
            self._restore_fdtd_preparation_inputs(state)
            raise
        return solver

    @staticmethod
    def _restore_fdtd_preparation_inputs(state):
        for mapping, key, present, value in state:
            if present:
                mapping[key] = value
            else:
                mapping.pop(key, None)

    def _initialize_fdtd_solver_for_scene(
        self, scene, *, device, initialize, port_excitations, termination_overrides,
        batch_lane=False, setup_auto_allowed=True,
    ):
        resolved_excitations = (
            self.excitations if port_excitations is None else tuple(port_excitations)
        )
        if self.config.parallel is not None:
            if self.config.spectral_sampler.observer_dtype == torch.float64:
                raise ValueError("Float64 spectral observers require single-device execution; distributed observation ownership is not qualified.")
            _require_cuda_device(device, method="fdtd")
            from .fdtd.distributed import DistributedFDTD

            solver = DistributedFDTD(
                scene,
                frequency=self.frequency,
                parallel=self.config.parallel,
                absorber_type=self.config.absorber,
                cpml=self.config.cpml,
                device=device,
            )
            # Mirror the single-device path so the owner shard enforces the
            # embedded-network fitted-band 'reject' contract on the full set of
            # requested output frequencies, not just the time-stepping frequency.
            solver._requested_port_frequencies = self.frequencies
            solver._port_excitations = resolved_excitations
            self._configure_fdtd_solver_time(solver)
            if initialize:
                solver.init_field()
                self._remember_fdtd_solver_time(solver)
                self._resolve_fdtd_time_steps(solver, scene)
            return solver

        prepared_scene = prepare_scene(scene, device=device, material_dtype=self.config.field_dtype)
        _require_cuda_device(prepared_scene.device, method="fdtd")
        material_cache_info = None
        if self.config.material_cache is not None:
            material_cache_info = self.config.material_cache._bind(prepared_scene, frequency=self.frequency)
        fdtd_backend, _ = _resolve_fdtd_backend()
        solver = fdtd_backend(
            prepared_scene,
            frequency=self.frequency,
            absorber_type=self.config.absorber,
            cpml=self.config.cpml,
        )
        solver.verbose = self.config.verbose
        gauge_owner = self.scene_module if self.scene_module is not None else scene
        solver._mode_source_vector_state = {
            name: {"gauge": gauge}
            for name, gauge in MODE_SOURCE_GAUGES.get(gauge_owner, {}).items()
        }
        if self.config.field_dtype == torch.float64:
            solver.cpml = replace(solver.cpml, memory_mode=CPMLMemoryMode.DENSE)
            solver._cpml_memory_mode_requested = CPMLMemoryMode.DENSE.value
        solver._observer_dtype = self.config.spectral_sampler.observer_dtype
        solver._requested_port_frequencies = self.frequencies
        solver._material_cache_info = material_cache_info
        solver._port_excitations = resolved_excitations
        self._configure_fdtd_solver_time(solver)
        solver._port_termination_overrides = dict(termination_overrides or {})
        if initialize:
            from .compiler.accel.dispatch import setup_enabled
            # A bound material cache supplies the PyTorch-compiled material model, so its
            # coefficients keep PyTorch setup; either stepping path reads the same tables.
            accelerated = (setup_auto_allowed and self.config.material_cache is None
                           and (not batch_lane or self.config.field_dtype == torch.float32))
            setup_token = setup_enabled.set(accelerated)
            try:
                solver.init_field()
            except Exception as error:
                if accelerated:
                    raise _AcceleratedSetupFailure(f"{type(error).__name__}: {error}") from error
                raise
            finally:
                setup_enabled.reset(setup_token)
            MODE_SOURCE_GAUGES[gauge_owner] = {
                name: state["gauge"]
                for name, state in solver._mode_source_vector_state.items()
                if "gauge" in state
            }
            self._remember_fdtd_solver_time(solver)
            self._apply_initial_condition(solver)
            self._resolve_fdtd_time_steps(solver, scene)
            if not batch_lane:
                from .fdtd.runtime.core_path import activate_core_path
                activate_core_path(solver)
        if getattr(self, "_fixed_waveport_mode_sources", False):
            compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
            frozen_sources = []
            for source in compiled_sources:
                if source.get("kind") != "mode_source":
                    frozen_sources.append(source)
                    continue
                frozen = dict(source)
                frozen["kind"] = "fixed_waveport_mode_source"
                frozen_sources.append(frozen)
            solver._compiled_sources = tuple(frozen_sources)
            if frozen_sources:
                solver._compiled_source = frozen_sources[0]
        return solver

    def _collect_fdtd_requested_frequencies(self, scene=None) -> tuple[float, ...]:
        resolved_scene = scene_declarations(self.scene if scene is None else scene)
        ordered = list(self.frequencies)
        seen = set(ordered)
        for monitor in resolved_scene.resolved_monitors():
            if isinstance(monitor, (PermittivityMonitor, MediumMonitor)):
                # Material monitors are evaluated analytically from compiled tensors and
                # do not require the simulation to accumulate a DFT at their frequencies.
                continue
            monitor_frequencies = getattr(monitor, "frequencies", None)
            if monitor_frequencies is None:
                continue
            for freq in monitor_frequencies:
                freq_value = float(freq)
                if freq_value in seen:
                    continue
                ordered.append(freq_value)
                seen.add(freq_value)
        return tuple(ordered)

    def _resolve_fdtd_time_steps(self, solver, scene=None) -> int:
        from .fdtd.runtime.timeplan import compile_fdtd_time_plan

        resolved_scene = scene_declarations(self.scene if scene is None else scene)
        plan = compile_fdtd_time_plan(solver, resolved_scene, self.config.run_time,
            self._collect_fdtd_requested_frequencies(resolved_scene),
            default_source_frequency=self.frequency,
            excitations=getattr(solver, "_port_excitations", self.excitations),
            normalize_source=self.config.spectral_sampler.normalize_source,
            window=str(self.config.spectral_sampler.window))
        from .fdtd.excitation.incident_line_spectrum import check_prepared_source_spectra

        if not hasattr(self, "_incident_spectrum_checks"):
            self._incident_spectrum_checks = set()
        check_prepared_source_spectra(solver, record_steps=plan["total_steps"],
                                     checked=self._incident_spectrum_checks)
        return plan["total_steps"]

    def _execute_fdtd_solve(
        self,
        solver,
        scene=None,
        *,
        resume_from=None,
        stop_step: int | None = None,
    ):
        resolved_scene = self.scene if scene is None else scene
        time_steps = self._resolve_fdtd_time_steps(solver, resolved_scene)
        dft_cfg = self.config.spectral_sampler
        use_full_field_dft = _resolve_full_field_dft(
            self.config.full_field_dft, self.frequencies, resolved_scene.ports)
        dft_request = self.frequency if len(self.frequencies) == 1 else self.frequencies
        # The default (E-only) request keeps the existing call; distributed solvers refuse H in FDTDConfig.
        components = ({"dft_components": self.config.volume_dft_components}
                      if self.config.records_magnetic_volume_dft else {})
        raw_output = solver.solve(
            time_steps=time_steps,
            dft_frequency=dft_request,
            enable_plot=self.config.enable_plot,
            dft_window=dft_cfg.window,
            full_field_dft=use_full_field_dft,
            normalize_source=dft_cfg.normalize_source,
            shutoff=self.config.shutoff,
            shutoff_check_interval=self.config.shutoff_check_interval,
            use_cuda_graph=self.config.cuda_graph,
            resume_from=resume_from,
            stop_step=stop_step,
            **components,
        )
        return raw_output, time_steps, use_full_field_dft, dft_cfg

    def _fdtd_last_step_field_payload(self, solver) -> dict[str, torch.Tensor]:
        def _field(name: str):
            value = getattr(solver, name, None)
            if value is not None:
                return value
            return getattr(solver, name.lower())

        if getattr(solver, "complex_fields_enabled", False) and hasattr(solver, "Ex_imag"):
            return {
                "Ex": torch.complex(_field("Ex"), _field("Ex_imag")),
                "Ey": torch.complex(_field("Ey"), _field("Ey_imag")),
                "Ez": torch.complex(_field("Ez"), _field("Ez_imag")),
            }
        return {
            "Ex": _field("Ex"),
            "Ey": _field("Ey"),
            "Ez": _field("Ez"),
        }

    def _run_fdtd_from_solver(self, solver, *, resume_from=None) -> Result:
        output = self._execute_fdtd_solve(solver, self.scene, resume_from=resume_from)
        raw_output, time_steps, use_full_field_dft, dft_cfg = output
        if raw_output is None:
            if self.config.parallel is None or self.config.parallel.gather_fields:
                raise RuntimeError("FDTD solve did not return any output.")
            raw_output = {}

        monitors = raw_output.get("observers", {}) if isinstance(raw_output, dict) else {}
        ports = raw_output.get("ports", {}) if isinstance(raw_output, dict) else {}
        circuits = raw_output.get("circuits", {}) if isinstance(raw_output, dict) else {}
        embedded_networks = (
            raw_output.get("embedded_networks", {})
            if isinstance(raw_output, dict)
            else {}
        )
        field_payload = {
            key: value
            for key, value in (raw_output.items() if isinstance(raw_output, dict) else [])
            if key in {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        }
        if self.config.parallel is not None and self.config.parallel.gather_fields and not field_payload:
            raise RuntimeError(
                "Multi-GPU FDTD was configured with gather_fields=True but returned no fields."
            )
        if not field_payload and len(self.frequencies) == 1 and self.config.parallel is None:
            field_payload = self._fdtd_last_step_field_payload(solver)

        result_device = (
            self.config.device
            if self.config.parallel is None
            else self.config.parallel.result_device
        )
        fields = _to_tensor_fields(field_payload, result_device)
        solver_stats = self._build_fdtd_solver_stats(
            solver,
            time_steps=time_steps,
            use_full_field_dft=use_full_field_dft,
            dft_cfg=dft_cfg,
        )
        breakdown = None
        if getattr(solver, "breakdown_enabled", False):
            from .fdtd.runtime.breakdown import finalize_breakdown_data

            breakdown = finalize_breakdown_data(solver)
        metadata = dict(self.metadata, reference_frequency=(
            "per-frequency" if solver._spectral_reference_frequency is None
            else solver._spectral_reference_frequency),
            absorber=solver_stats["absorber"],
            absorber_resolution_reason=solver_stats["absorber_resolution_reason"],
            passivity_guard=solver_stats["passivity_guard"],
            passivity_guard_warning=solver_stats["passivity_guard_warning"])
        tfsf_state = getattr(solver, "_tfsf_state", None)
        if tfsf_state is not None and tfsf_state["provider"] == "plane_wave_fixed_k":
            metadata["tfsf_incident_reference"] = tfsf_state["auxiliary_grid"].incident_spectrum(solver)
        prebias_provenance = getattr(solver, "_electrostatic_prebias_provenance", None)
        if prebias_provenance is not None:
            metadata = dict(metadata)
            metadata["electrostatic_prebias"] = prebias_provenance
        return Result(
            method="fdtd",
            scene=self.scene,
            prepared_scene=solver.scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=solver,
            fields=fields,
            monitors=monitors,
            ports=ports,
            circuits=circuits,
            embedded_networks=embedded_networks,
            metadata=metadata,
            solver_stats=solver_stats,
            raw_output=raw_output,
            breakdown=breakdown,
        )

    def _build_fdtd_solver_stats(
        self,
        solver,
        *,
        time_steps: int,
        use_full_field_dft: bool,
        dft_cfg: SpectralSampler,
    ) -> dict[str, Any]:
        from .fdtd.postprocess import last_step_field_ratio, tfsf_run_record

        # A pulse (non-CW) time plan records every spectrum without a window, whatever was requested.
        pulse_plan = (getattr(solver, "_time_plan", None) or {}).get("response", "cw_steady_state") != "cw_steady_state"
        elapsed_s = getattr(solver, "last_solve_elapsed_s", None)
        step_loop_elapsed_s = getattr(solver, "last_step_loop_elapsed_s", None)
        step_loop_steps = getattr(solver, "last_step_loop_steps", None)
        dft_sample_counts = tuple(getattr(solver, "dft_sample_counts", (getattr(solver, "dft_sample_count", 0),)))
        observer_sample_counts = tuple(
            getattr(
                solver,
                "observer_sample_counts",
                (getattr(solver, "observer_sample_count", 0),),
            )
        )
        shutoff_triggered = bool(getattr(solver, "_shutoff_triggered", False))
        shutoff_step = getattr(solver, "_shutoff_step", None)
        wire_runtime = getattr(solver, "_wire_runtime", None)
        # Partitioned runtimes own shard-local fields and have no global Ex.
        # Their validated execution configuration supplies the scalar dtype;
        # reading metadata must not gather or allocate a global field.
        field_dtype = getattr(getattr(solver, "Ex", None), "dtype", self.config.field_dtype)
        observer_dtype = getattr(solver, "_observer_dtype", field_dtype)
        stats = {
            "pml_extrusions": solver.scene.pml_extrusions,
            **({"field_dtype": str(field_dtype), "observer_dtype": str(observer_dtype)}
               if field_dtype == torch.float64 or observer_dtype == torch.float64 else {}),
            "time_steps": time_steps,
            "shutoff": self.config.shutoff,
            "shutoff_check_interval": self.config.shutoff_check_interval,
            "shutoff_triggered": shutoff_triggered,
            "shutoff_step": shutoff_step,
            "steps_run": (shutoff_step + 1) if shutoff_triggered else time_steps,
            "dt": solver.dt,
            "time_step": dict(getattr(solver, "_time_step_metadata", {})),
            "time_plan": dict(getattr(solver, "_time_plan", {})),
            "cuda_graph_active": bool(getattr(solver, "_cuda_graph_active", False)),
            "observer_cuda_graph_active": bool(getattr(solver, "_observer_cuda_graph_active", False)),
            "material_cache": getattr(solver, "_material_cache_info", None),
            "circuit_cuda_graph_active": bool(
                getattr(solver, "_circuit_graph_active", False)
            ),
            "tail_graph_active": bool(getattr(solver, "_tail_graph_active", False)),
            "port_cuda_graph_active": bool(getattr(solver, "_port_cuda_graph_active", False)),
            "network_cuda_graph_active": bool(
                getattr(solver, "_network_cuda_graph_active", False)
            ),
            "boundary": getattr(solver, "boundary_kind", self.scene.boundary.kind),
            "absorber": getattr(solver, "active_absorber_type", self.config.absorber),
            "absorber_resolution_reason": getattr(solver, "absorber_resolution_reason", ""),
            "passivity_guard": getattr(solver, "passivity_guard_status", "inactive"),
            "passivity_guard_warning": getattr(solver, "passivity_guard_warning", None),
            "cpml_memory_mode": getattr(solver, "_cpml_memory_mode", None),
            "cpml_requested_memory_mode": getattr(solver, "_cpml_memory_mode_requested", None),
            "cpml_allocated_memory_bytes": getattr(solver, "_cpml_allocated_memory_bytes", None),
            "cpml_dense_memory_bytes": getattr(solver, "_cpml_dense_memory_bytes", None),
            "cpml_slab_memory_bytes": getattr(solver, "_cpml_slab_memory_bytes", None),
            "thin_wire_enabled": wire_runtime is not None,
            "thin_wire_segments": (
                0 if wire_runtime is None else int(wire_runtime.current.numel())
            ),
            "thin_wire_nodes": (
                0 if wire_runtime is None else int(wire_runtime.charge.numel())
            ),
            "thin_wire_state_bytes": (
                0 if wire_runtime is None else int(wire_runtime.state_bytes)
            ),
            "thin_wire_cfl_limit": (
                None if wire_runtime is None else float(wire_runtime.cfl_limit)
            ),
            "thin_wire_uncoupled_cfl_limit": (
                None if wire_runtime is None else float(wire_runtime.wire_cfl_limit)
            ),
            "thin_wire_maxwell_cfl_limit": (
                None if wire_runtime is None else float(wire_runtime.maxwell_cfl_limit)
            ),
            "thin_wire_time_step_adjusted": (
                False if wire_runtime is None else bool(wire_runtime.dt_adjusted)
            ),
            "dft_window": "none" if pulse_plan else dft_cfg.window,
            "dft_window_requested": dft_cfg.window,
            "last_step_field_ratio": last_step_field_ratio(solver),
            "bloch_wavevector": (None if getattr(solver, "resolved_bloch_wavevector", None) is None
                                 else tuple(float(k) for k in solver.resolved_bloch_wavevector)),
            "bloch_phase": (None if getattr(solver, "boundary_phase_cos", None) is None
                            else tuple(complex(c, s) for c, s in zip(solver.boundary_phase_cos,
                                                                     solver.boundary_phase_sin))),
            "tfsf": tfsf_run_record(solver),
            "normalize_source": dft_cfg.normalize_source,
            "reference_frequency": ("per-frequency" if solver._spectral_reference_frequency is None
                                    else solver._spectral_reference_frequency),
            "frequency": self.frequency,
            "frequencies": self.frequencies,
            "num_frequencies": len(self.frequencies),
            "dft_samples": dft_sample_counts[0] if dft_sample_counts else 0,
            "dft_sample_counts": dft_sample_counts,
            "observer_samples": observer_sample_counts[0] if observer_sample_counts else 0,
            "observer_sample_counts": observer_sample_counts,
            "full_field_dft": use_full_field_dft,
            "full_field_dft_source": "auto" if self.config.full_field_dft is None else "explicit",
            "elapsed_s": elapsed_s,
            "steady_step_elapsed_s": step_loop_elapsed_s,
            "steady_steps": step_loop_steps,
            "steady_ms_per_step": (
                None
                if step_loop_elapsed_s is None or not step_loop_steps
                else step_loop_elapsed_s * 1e3 / step_loop_steps
            ),
            "ms_per_step": (
                elapsed_s * 1e3 / time_steps
                if elapsed_s is not None and time_steps > 0
                else None
            ),
            "steps_per_second": (
                time_steps / elapsed_s
                if elapsed_s is not None and elapsed_s > 0.0
                else None
            ),
        }
        if self.config.parallel is not None:
            stats["parallel_stats"] = dict(getattr(solver, "parallel_stats", {}))
        return stats

    def _run_fdtd_with_gradient_bridge(self) -> Result:
        from .fdtd.adjoint import run_fdtd_with_gradient_bridge

        return run_fdtd_with_gradient_bridge(self)

    def _run_distributed_fdtd_with_gradient_bridge(self) -> Result:
        from .fdtd.distributed.adjoint import run_distributed_fdtd_with_gradient_bridge

        return run_distributed_fdtd_with_gradient_bridge(self)


class PreparedNetworkSweep:
    def __init__(self, simulation: Simulation, scene, manifest):
        self.simulation = simulation
        self.scene = scene
        self.manifest = manifest

    def run(self, *, execution=None) -> Result:
        if execution is not None:
            # Distribute the excitation columns over the ensemble executor. The
            # coordinator pre-prepared scene is intentionally not reused: each
            # column prepares on its own leased device instead of materializing a
            # full PreparedScene on one GPU.
            return self.simulation._run_fdtd_network_sweep_ensemble(
                execution,
                scene=self.simulation.scene,
            )
        return self.simulation._run_fdtd_network_sweep(
            scene=self.scene,
            manifest=self.manifest,
        )


class PreparedWavePortExcitation:
    def __init__(self, simulation: Simulation, scene, manifest):
        self.simulation = simulation
        self.scene = scene
        self.manifest = manifest

    def run(self) -> Result:
        excitation = self.simulation._waveport_excitation()
        return self.simulation._run_fdtd_waveport_excitation(
            excitation,
            scene=self.scene,
            manifest=self.manifest,
        )


class PreparedSimulation:
    def __init__(self, simulation: Simulation, solver, *, fdfd_gradient_bridge=None):
        self.simulation = simulation
        self.solver = solver
        self._fdfd_gradient_bridge = fdfd_gradient_bridge
        self._consumed = False

    @property
    def diagnostics(self):
        """Inspect actual compiled single-device FDTD geometry, ports and time cost."""
        if self.simulation.method != SimulationMethod.FDTD:
            raise ValueError("Preparation diagnostics currently require FDTD.")
        from .compiler.fdtd_audit import compile_fdtd_audit

        return compile_fdtd_audit(self.solver)

    def solve_modes(self, **kwargs) -> Result:
        """Solve native GPU transverse FDFD modes inside a rectangular PEC aperture."""
        if self.simulation.method != SimulationMethod.FDFD:
            raise ValueError("solve_modes requires a single-frequency FDFD preparation.")
        self._claim()
        from .fdfd.modes import solve_modes
        return solve_modes(self, **kwargs)

    def _claim(self) -> None:
        if self._consumed:
            raise RuntimeError(
                "A PreparedSimulation can execute only once; call Simulation.prepare() "
                "again for a fresh solver."
            )
        self._consumed = True

    def _validate_resume_support(self) -> None:
        if self.simulation.has_trainable_parameters:
            raise UnsupportedFeatureError("FDTD resume requires a detached forward run.",
                                          feature="guard:0041be83", method="fdtd")
        if self.simulation.config.parallel is not None and getattr(self.solver,"_nccl",False):
            raise UnsupportedFeatureError("Distributed FDTD resume currently requires cuda_p2p transport.",
                                          feature="guard:477902bc", method="fdtd")

    def run(self, *, resume_from=None, initial_harmonic_electric=None) -> Result:
        if initial_harmonic_electric is not None:
            if self.simulation.method != SimulationMethod.MOM or self.simulation.config.harmonic_balance is None:
                raise ValueError("Initial harmonic fields require an MoM harmonic-balance preparation")
            from .mom.harmonic_runtime import validate_initial_harmonic_electric
            initial_harmonic_electric = validate_initial_harmonic_electric(initial_harmonic_electric,self.simulation.config)
        if self.simulation.method in _STATIC_CONFIGS:
            if resume_from is not None:
                raise TypeError("resume_from is available only for FDTD simulations.")
            self._claim()
            if self.simulation.method == SimulationMethod.ELECTROSTATIC:
                from .electrostatic.runtime import electrostatic_result
                return electrostatic_result(self.simulation.scene, self.solver, self.simulation.config)
            from .electrostatic.capacitance import capacitance_result
            return capacitance_result(self.simulation.scene, self.solver, self.simulation.config)
        if self.simulation.method == SimulationMethod.MOM:
            if resume_from is not None:
                raise TypeError("resume_from is available only for FDTD simulations.")
            self._claim()
            if self.simulation.config.harmonic_balance is not None:
                return self.solver.run(initial_harmonic_electric=initial_harmonic_electric)
            return self.solver.run()
        if self.simulation.method == SimulationMethod.FDFD:
            if resume_from is not None:
                raise TypeError("resume_from is available only for FDTD simulations.")
            self._claim()
            bridge = self._fdfd_gradient_bridge or self.simulation._fdfd_gradient_bridge()
            if bridge is not None:
                from .fdfd.adjoint import run_fdfd_with_gradient_bridge

                return run_fdfd_with_gradient_bridge(self.simulation, bridge=bridge, solver=self.solver)
            solver_cfg = self.simulation.config.solver
            return self.simulation._build_fdfd_result(self.solver, solver_cfg)
        if resume_from is not None:
            self._validate_resume_support()
            total_steps = self.simulation._resolve_fdtd_time_steps(
                self.solver,
                self.simulation.scene,
            )
            from .fdtd.resume import preflight_resume_checkpoint

            preflight_resume_checkpoint(
                self.solver,
                resume_from,
                total_steps=total_steps,
            )
            self._claim()
            result = self.simulation._run_fdtd_from_solver(
                self.solver,
                resume_from=resume_from,
            )
            self.solver = result.solver
            return result
        self._claim()
        if self.simulation.has_trainable_parameters:
            return self.simulation._run_fdtd()
        result = self.simulation._run_fdtd_from_solver(self.solver)
        self.solver = result.solver
        return result

    def run_until(self, step: int):
        """Advance exactly ``step`` FDTD steps and return a detached resume state."""

        if self.simulation.method != SimulationMethod.FDTD:
            raise TypeError("run_until() is available only for FDTD simulations.")
        self._validate_resume_support()
        total_steps = self.simulation._resolve_fdtd_time_steps(
            self.solver,
            self.simulation.scene,
        )
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer.")
        if not 0 <= step < total_steps:
            raise ValueError(
                f"step must satisfy 0 <= step < {total_steps} for this simulation."
            )
        self._claim()
        output = self.simulation._execute_fdtd_solve(self.solver, self.simulation.scene, stop_step=step)
        raw_output, resolved_steps, _use_full_field_dft, _dft_cfg = output
        if raw_output is not None or resolved_steps != total_steps:
            raise RuntimeError("Partial FDTD execution did not stop at its checkpoint boundary.")
        from .fdtd.resume import capture_resume_checkpoint

        return capture_resume_checkpoint(
            self.solver,
            step=step,
            total_steps=total_steps,
        )



def _move_port_mapping_to_device(ports, device: torch.device) -> dict:
    """Gather a column's PortData mapping onto a single result device.

    Ensemble columns run on different GPUs, so their PortData tensors live on
    different devices. Matrix assembly requires one common device; this rebuilds
    each PortData with every tensor field moved to ``device`` while preserving
    the port contract (convention, direction, reference plane, metadata).
    """

    import dataclasses

    from .fdtd.distributed.output import move_tensors_to_device

    def _move_value(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        return value

    moved = {}
    for name, entry in ports.items():
        tensor_fields = {}
        for data_field in dataclasses.fields(entry):
            current = getattr(entry, data_field.name)
            if data_field.name == "metadata":
                tensor_fields["metadata"] = move_tensors_to_device(dict(current), device)
            elif isinstance(current, torch.Tensor):
                tensor_fields[data_field.name] = current.to(device=device)
            elif data_field.name == "z0":
                tensor_fields["z0"] = _move_value(current)
        moved[name] = dataclasses.replace(entry, **tensor_fields)
    return moved
