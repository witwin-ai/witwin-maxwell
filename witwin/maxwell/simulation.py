from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import torch

from .lumped import PortExcitation, PortSweep
from .monitors import MediumMonitor, PermittivityMonitor
from .ports import LumpedPort, TerminalPort, WavePort
from .result import Result
from .scene import Scene, SceneModule, prepare_scene

FDFD = None
FDTD = None
calculate_required_steps = None


class SimulationMethod(str, Enum):
    FDFD = "fdfd"
    FDTD = "fdtd"

    def __str__(self) -> str:
        return self.value


class SpectralWindowKind(str, Enum):
    NONE = "none"
    HANNING = "hanning"
    RAMP = "ramp"

    def __str__(self) -> str:
        return self.value


class AbsorberKind(str, Enum):
    CPML = "cpml"
    PML = "pml"
    ABSORBER = "absorber"
    STABLE_PML = "stablepml"

    def __str__(self) -> str:
        return self.value


def _normalize_frequencies(*, frequencies=None, frequency=None) -> tuple[float, ...]:
    if frequency is not None and frequencies is not None:
        raise ValueError("Pass either frequency or frequencies, not both.")
    if frequencies is None:
        if frequency is None:
            raise ValueError("A target frequency is required.")
        return (float(frequency),)

    if isinstance(frequencies, Iterable) and not isinstance(frequencies, (str, bytes)):
        values = tuple(float(value) for value in frequencies)
    else:
        values = (float(frequencies),)
    if not values:
        raise ValueError("frequencies must not be empty.")
    return values


def _normalize_single_frequency(*, frequency=None) -> float:
    values = _normalize_frequencies(frequency=frequency)
    if len(values) != 1:
        raise ValueError("This solver path only supports a single frequency.")
    return values[0]


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
    if len(resolved) > 1:
        raise NotImplementedError(
            "A direct FDTD run supports one active RF port; use PortSweep for independent N-port columns."
        )
    return resolved


def _normalize_spectral_window_kind(value) -> SpectralWindowKind:
    if isinstance(value, SpectralWindowKind):
        return value
    try:
        return SpectralWindowKind(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in SpectralWindowKind)
        raise ValueError(f"window must be one of: {choices}.") from exc


def _normalize_absorber_kind(value) -> AbsorberKind:
    if isinstance(value, AbsorberKind):
        return value
    try:
        return AbsorberKind(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in AbsorberKind)
        raise ValueError(f"absorber must be one of: {choices}.") from exc


@dataclass(frozen=True)
class GMRES:
    max_iter: int = 5000
    tol: float = 1e-6
    restart: int = 200
    solver_type: str = "gmres"
    preconditioner: str = "jacobi"
    precision: str = "single"  # 'single' | 'double': working precision of the iterative solve
    ssor_omega: float = 0.8  # relaxation factor for the 'ssor' preconditioner


@dataclass(frozen=True)
class TimeConfig:
    time_steps: int | None = None
    steady_cycles: int = 20
    transient_cycles: int = 15

    @classmethod
    def auto(cls, steady_cycles: int = 20, transient_cycles: int = 15) -> "TimeConfig":
        return cls(
            time_steps=None,
            steady_cycles=steady_cycles,
            transient_cycles=transient_cycles,
        )


@dataclass
class SpectralSampler:
    window: SpectralWindowKind = SpectralWindowKind.HANNING
    normalize_source: bool = False

    def __post_init__(self):
        self.window = _normalize_spectral_window_kind(self.window)
        self.normalize_source = bool(self.normalize_source)


@dataclass
class FDFDConfig:
    solver: GMRES = field(default_factory=GMRES)
    enable_plot: bool = False
    verbose: bool = False

    def __post_init__(self):
        if not isinstance(self.solver, GMRES):
            raise TypeError("solver must be a GMRES instance.")
        self.enable_plot = bool(self.enable_plot)
        self.verbose = bool(self.verbose)


@dataclass
class FDTDConfig:
    run_time: TimeConfig = field(default_factory=TimeConfig.auto)
    spectral_sampler: SpectralSampler = field(default_factory=SpectralSampler)
    enable_plot: bool = False
    full_field_dft: bool = False
    absorber: AbsorberKind = AbsorberKind.CPML
    cpml_config: dict[str, Any] = field(default_factory=dict)
    adjoint_checkpoint_stride: int | None = None
    shutoff: float = 0.0  # relative E-energy threshold for auto-shutoff; 0 disables (opt-in)
    shutoff_check_interval: int = 100
    cuda_graph: bool = False  # capture the field-update core into a CUDA graph (opt-in)

    def __post_init__(self):
        if not isinstance(self.run_time, TimeConfig):
            raise TypeError("run_time must be a TimeConfig instance.")
        if not isinstance(self.spectral_sampler, SpectralSampler):
            raise TypeError("spectral_sampler must be a SpectralSampler instance.")
        self.enable_plot = bool(self.enable_plot)
        self.full_field_dft = bool(self.full_field_dft)
        self.absorber = _normalize_absorber_kind(self.absorber)
        self.cpml_config = dict(self.cpml_config or {})
        self.shutoff = float(self.shutoff)
        if self.shutoff < 0:
            raise ValueError("shutoff must be >= 0.")
        self.shutoff_check_interval = int(self.shutoff_check_interval)
        if self.shutoff_check_interval <= 0:
            raise ValueError("shutoff_check_interval must be > 0.")
        self.cuda_graph = bool(self.cuda_graph)
        if self.adjoint_checkpoint_stride is not None:
            self.adjoint_checkpoint_stride = int(self.adjoint_checkpoint_stride)
            if self.adjoint_checkpoint_stride <= 0:
                raise ValueError("adjoint_checkpoint_stride must be > 0 when provided.")


def _to_tensor_fields(field_mapping: dict[str, Any], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        name.upper(): torch.as_tensor(value, device=device)
        for name, value in field_mapping.items()
    }


def _resolve_fdfd_backend():
    global FDFD
    if FDFD is None:
        from .fdfd import FDFD as backend

        FDFD = backend
    return FDFD


def _resolve_fdtd_backend():
    global FDTD, calculate_required_steps
    if FDTD is None or calculate_required_steps is None:
        from .fdtd import FDTD as fdtd_backend
        from .fdtd import calculate_required_steps as calculate_steps_backend

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
    for region in getattr(scene, "material_regions", ()):
        density = getattr(region, "density", None)
        if isinstance(density, torch.Tensor) and density.requires_grad:
            trainable.append(density)
    return tuple(trainable)


def _scene_trainable_geometry_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    trainable = []
    for structure in scene.structures:
        geometry = getattr(structure, "geometry", None)
        if geometry is None:
            continue
        for value in vars(geometry).values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                trainable.append(value)
    return tuple(trainable)


def _scene_trainable_material_parameters(scene: Scene) -> tuple[torch.Tensor, ...]:
    trainable = []
    for structure in scene.structures:
        material = getattr(structure, "material", None)
        perturbation = getattr(material, "perturbation", None)
        if isinstance(perturbation, torch.Tensor) and perturbation.requires_grad:
            trainable.append(perturbation)
    return tuple(trainable)


def _require_cuda_scene(scene: Scene, *, method: str) -> None:
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(
            f"Simulation.{method}(...) requires scene.device to be CUDA, got {device}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Simulation.{method}(...) requires CUDA, but torch.cuda.is_available() is False."
        )


class Simulation:
    def __init__(
        self,
        *,
        scene,
        method: SimulationMethod,
        frequencies: tuple[float, ...],
        config: FDFDConfig | FDTDConfig | None = None,
        excitations=None,
        metadata: dict[str, Any] | None = None,
    ):
        if not frequencies:
            raise ValueError("At least one frequency is required.")
        resolved_scene, scene_module = _resolve_scene_input(scene)
        self.scene_input = scene
        self.scene = resolved_scene
        self.scene_module = scene_module
        self.has_trainable_parameters = bool(
            any(parameter.requires_grad for parameter in (scene_module.parameters() if scene_module is not None else ()))
            or _scene_trainable_density_parameters(resolved_scene)
            or _scene_trainable_geometry_parameters(resolved_scene)
            or _scene_trainable_material_parameters(resolved_scene)
        )
        self.method = SimulationMethod(method)
        self.port_sweep = excitations if isinstance(excitations, PortSweep) else None
        self.excitations = (
            () if self.port_sweep is not None else _normalize_port_excitations(excitations)
        )
        if self.method != SimulationMethod.FDTD and (self.excitations or self.port_sweep):
            raise ValueError("RF port excitation is supported by Simulation.fdtd(...) only.")
        self._validate_trainable_rf_support()
        self.frequencies = tuple(float(freq) for freq in frequencies)
        self.frequency = self.frequencies[0]
        if self.method == SimulationMethod.FDFD:
            if config is None:
                config = FDFDConfig()
            if not isinstance(config, FDFDConfig):
                raise TypeError("FDFD simulations require an FDFDConfig.")
        else:
            if config is None:
                config = FDTDConfig()
            if not isinstance(config, FDTDConfig):
                raise TypeError("FDTD simulations require an FDTDConfig.")
        self.config = config
        self.metadata = dict(metadata or {})

    @classmethod
    def fdfd(
        cls,
        scene,
        *,
        frequency=None,
        solver: GMRES | None = None,
        enable_plot: bool = False,
        verbose: bool = False,
    ) -> "Simulation":
        return cls(
            scene=scene,
            method=SimulationMethod.FDFD,
            frequencies=(_normalize_single_frequency(frequency=frequency),),
            config=FDFDConfig(
                solver=solver or GMRES(),
                enable_plot=enable_plot,
                verbose=verbose,
            ),
        )

    @classmethod
    def fdtd(
        cls,
        scene,
        *,
        frequencies=None,
        frequency=None,
        run_time: TimeConfig | None = None,
        absorber: AbsorberKind | str = AbsorberKind.CPML,
        spectral_sampler: SpectralSampler | None = None,
        cpml_config: dict[str, Any] | None = None,
        enable_plot: bool = False,
        full_field_dft: bool = False,
        shutoff: float = 0.0,
        shutoff_check_interval: int = 100,
        cuda_graph: bool = False,
        excitations=None,
    ) -> "Simulation":
        return cls(
            scene=scene,
            method=SimulationMethod.FDTD,
            frequencies=_normalize_frequencies(frequencies=frequencies, frequency=frequency),
            config=FDTDConfig(
                run_time=run_time or TimeConfig.auto(),
                absorber=absorber,
                spectral_sampler=spectral_sampler or SpectralSampler(),
                cpml_config={} if cpml_config is None else cpml_config,
                enable_plot=enable_plot,
                full_field_dft=full_field_dft,
                shutoff=shutoff,
                shutoff_check_interval=shutoff_check_interval,
                cuda_graph=cuda_graph,
            ),
            excitations=excitations,
        )

    def prepare(self):
        self._refresh_scene()
        self._validate_trainable_rf_support()
        self._validate_port_excitations()
        waveport_excitation = self._waveport_excitation()
        if waveport_excitation is not None:
            prepared_scene = prepare_scene(self.scene)
            _require_cuda_scene(prepared_scene, method="fdtd")
            return PreparedWavePortExcitation(
                self,
                prepared_scene,
                self._resolve_waveport_excitation_manifest(
                    prepared_scene,
                    waveport_excitation,
                ),
            )
        if self.port_sweep is not None:
            prepared_scene = prepare_scene(self.scene)
            _require_cuda_scene(prepared_scene, method="fdtd")
            return PreparedNetworkSweep(
                self,
                prepared_scene,
                self._resolve_port_sweep_manifest(prepared_scene),
            )
        if self.method == SimulationMethod.FDFD:
            solver = self._build_fdfd_solver()
        elif self.method == SimulationMethod.FDTD:
            solver = self._build_fdtd_solver(initialize=True)
            if self.has_trainable_parameters:
                from .fdtd.adjoint.dispatch import validate_native_adjoint_preparation

                validate_native_adjoint_preparation(solver)
        else:
            raise ValueError(f"Unsupported simulation method {self.method!r}.")
        return PreparedSimulation(self, solver)

    def run(self) -> Result:
        self._refresh_scene()
        self._validate_trainable_rf_support()
        self._validate_port_excitations()
        if self.method == SimulationMethod.FDFD:
            return self._run_fdfd()
        if self.method == SimulationMethod.FDTD:
            return self._run_fdtd()
        raise ValueError(f"Unsupported simulation method {self.method!r}.")

    def _validate_port_excitations(self) -> None:
        if self.port_sweep is not None:
            if any(isinstance(port, WavePort) for port in self.scene.ports):
                if self.port_sweep.amplitude.requires_grad:
                    raise NotImplementedError(
                        "PortSweep amplitude is trainable, but WavePort sweeps do not "
                        "support RF-parameter gradients."
                    )
                return
            from .network_sweep import resolve_network_run_manifest

            resolve_network_run_manifest(self.scene, self.port_sweep, self.frequencies)
            return
        if not self.excitations:
            return
        ports_by_name = {port.name: port for port in self.scene.ports}
        for excitation in self.excitations:
            port = ports_by_name.get(excitation.port_name)
            if port is None:
                raise ValueError(
                    f"PortExcitation references missing port {excitation.port_name!r}."
                )
            if not isinstance(port, (LumpedPort, TerminalPort, WavePort)):
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
                    raise NotImplementedError(
                        "WavePort PortExcitation does not yet support trainable amplitude."
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
        has_rf_ports = any(
            isinstance(port, (LumpedPort, TerminalPort, WavePort))
            for port in self.scene.ports
        )
        if has_rf_ports or self.scene.lumped_elements or self.excitations or self.port_sweep:
            raise NotImplementedError(
                "Trainable FDTD scenes cannot yet include RF ports, port excitations, "
                "or standalone R/L/C elements because the adjoint does not replay "
                "their auxiliary state."
            )

    def _refresh_scene(self):
        if self.scene_module is not None:
            self.scene = self.scene_module.to_scene()
            return
        if isinstance(self.scene_input, Scene):
            self.scene = self.scene_input

    def _run_fdfd(self) -> Result:
        if self.has_trainable_parameters:
            return self._run_fdfd_with_gradient_bridge()
        solver = self._build_fdfd_solver()
        solver_cfg = self.config.solver
        return self._build_fdfd_result(solver, solver_cfg)

    def _run_fdfd_with_gradient_bridge(self) -> Result:
        from .fdfd.adjoint import run_fdfd_with_gradient_bridge

        return run_fdfd_with_gradient_bridge(self)

    def _build_fdfd_solver(self):
        prepared_scene = prepare_scene(self.scene)
        if prepared_scene.grid.is_custom:
            raise NotImplementedError(
                "FDFD does not support nonuniform (GridSpec.custom / GridSpec.auto) grids yet; "
                "use Simulation.fdtd(...) or a uniform GridSpec."
            )
        _require_cuda_scene(prepared_scene, method="fdfd")
        fdfd_backend = _resolve_fdfd_backend()
        solver_cfg = self.config.solver
        return fdfd_backend(
            prepared_scene,
            frequency=self.frequency,
            solver_type=solver_cfg.solver_type,
            preconditioner=solver_cfg.preconditioner,
            precision=solver_cfg.precision,
            ssor_omega=solver_cfg.ssor_omega,
            enable_plot=self.config.enable_plot,
            verbose=self.config.verbose,
        )

    def _build_fdfd_result(self, solver, solver_cfg: GMRES) -> Result:
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
        solver_stats = {
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
        return Result(
            method="fdfd",
            scene=self.scene,
            prepared_scene=solver.scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=solver,
            fields=fields,
            metadata=self.metadata,
            solver_stats=solver_stats,
        )

    def _run_fdtd(self) -> Result:
        if self.has_trainable_parameters:
            return self._run_fdtd_with_gradient_bridge()
        if self.port_sweep is not None:
            return self._run_fdtd_network_sweep()
        waveport_excitation = self._waveport_excitation()
        if waveport_excitation is not None:
            return self._run_fdtd_waveport_excitation(waveport_excitation)
        solver = self._build_fdtd_solver(initialize=True)
        return self._run_fdtd_from_solver(solver)

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
        columns = []
        column_stats = []
        last_solver = None
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
            columns.append(column_result.ports)
            column_stats.append(column_result.stats())
            last_solver = solver

        stacked_ports, network = aggregate_network_columns(manifest, tuple(columns))
        metadata = dict(self.metadata)
        metadata["network_run_manifest"] = manifest.metadata()
        return Result(
            method="fdtd",
            scene=run_scene,
            prepared_scene=last_solver.scene,
            frequency=self.frequency,
            frequencies=self.frequencies,
            solver=last_solver,
            fields={},
            monitors={},
            ports=stacked_ports,
            network=network,
            metadata=metadata,
            solver_stats={
                "network_sweep": manifest.metadata(),
                "columns": tuple(column_stats),
            },
            raw_output={"network_run_manifest": manifest.metadata()},
        )

    def _resolve_port_sweep_manifest(self, scene):
        if any(isinstance(port, WavePort) for port in scene.ports):
            from .waveport_sweep import resolve_waveport_run_manifest

            return resolve_waveport_run_manifest(
                scene,
                self.port_sweep,
                self.frequencies,
            )
        from .network_sweep import resolve_network_run_manifest

        return resolve_network_run_manifest(
            scene,
            self.port_sweep,
            self.frequencies,
        )

    def _build_fdtd_solver(self, *, initialize: bool):
        return self._build_fdtd_solver_for_scene(self.scene, initialize=initialize)

    def _build_fdtd_solver_for_scene(
        self,
        scene,
        *,
        initialize: bool,
        port_excitations=None,
        termination_overrides=None,
    ):
        prepared_scene = prepare_scene(scene)
        _require_cuda_scene(prepared_scene, method="fdtd")
        fdtd_backend, _ = _resolve_fdtd_backend()
        resolved_excitations = (
            self.excitations if port_excitations is None else tuple(port_excitations)
        )
        time_step_frequency = self.frequency
        for excitation in resolved_excitations:
            source_time = excitation.source_time
            if source_time is not None:
                time_step_frequency = max(
                    time_step_frequency,
                    float(source_time.characteristic_frequency),
                )
        solver = fdtd_backend(
            prepared_scene,
            frequency=time_step_frequency,
            absorber_type=self.config.absorber,
            cpml_config=self.config.cpml_config,
        )
        solver._requested_port_frequencies = self.frequencies
        solver._port_excitations = resolved_excitations
        solver._port_termination_overrides = dict(termination_overrides or {})
        if initialize:
            solver.init_field()
        return solver

    def _collect_fdtd_requested_frequencies(self, scene=None) -> tuple[float, ...]:
        resolved_scene = self.scene if scene is None else scene
        ordered = list(self.frequencies)
        seen = set(ordered)
        monitors = resolved_scene.resolved_monitors() if hasattr(resolved_scene, "resolved_monitors") else resolved_scene.monitors
        for monitor in monitors:
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
        _, steps_helper = _resolve_fdtd_backend()
        resolved_scene = self.scene if scene is None else scene
        runtime_cfg = self.config.run_time
        if runtime_cfg.time_steps is not None:
            return runtime_cfg.time_steps

        domain_size = max(
            float(resolved_scene.domain.bounds[0][1] - resolved_scene.domain.bounds[0][0]),
            float(resolved_scene.domain.bounds[1][1] - resolved_scene.domain.bounds[1][0]),
            float(resolved_scene.domain.bounds[2][1] - resolved_scene.domain.bounds[2][0]),
        )
        requested_frequencies = self._collect_fdtd_requested_frequencies(resolved_scene)
        source_time = getattr(solver, "_source_time", None)
        port_source_times = tuple(
            excitation.source_time
            for excitation in getattr(solver, "_port_excitations", self.excitations)
            if excitation.source_time is not None
        )
        if port_source_times:
            candidates = ((source_time,) if source_time is not None else ()) + port_source_times
            source_time = max(
                candidates,
                key=lambda value: float(
                    value.get("settling_time", 0.0)
                    if isinstance(value, dict)
                    else getattr(value, "settling_time", 0.0)
                ),
            )
        return steps_helper(
            frequency=min(requested_frequencies),
            dt=solver.dt,
            c=solver.c,
            num_cycles=runtime_cfg.steady_cycles,
            transient_cycles=runtime_cfg.transient_cycles,
            domain_size=domain_size,
            source_time=source_time,
        )

    def _execute_fdtd_solve(self, solver, scene=None):
        resolved_scene = self.scene if scene is None else scene
        time_steps = self._resolve_fdtd_time_steps(solver, resolved_scene)
        dft_cfg = self.config.spectral_sampler
        requested_full_field_dft = self.config.full_field_dft
        has_lumped_ports = any(isinstance(port, LumpedPort) for port in resolved_scene.ports)
        use_full_field_dft = requested_full_field_dft or (
            len(self.frequencies) > 1 and not has_lumped_ports
        )
        dft_request = self.frequency if len(self.frequencies) == 1 else self.frequencies
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

    def _run_fdtd_from_solver(self, solver) -> Result:
        raw_output, time_steps, use_full_field_dft, dft_cfg = self._execute_fdtd_solve(solver, self.scene)
        if raw_output is None:
            raise RuntimeError("FDTD solve did not return any output.")

        monitors = raw_output.get("observers", {}) if isinstance(raw_output, dict) else {}
        ports = raw_output.get("ports", {}) if isinstance(raw_output, dict) else {}
        field_payload = {
            key: value
            for key, value in (raw_output.items() if isinstance(raw_output, dict) else [])
            if key in {"Ex", "Ey", "Ez"}
        }
        if not field_payload and len(self.frequencies) == 1:
            field_payload = self._fdtd_last_step_field_payload(solver)

        fields = _to_tensor_fields(field_payload, self.scene.device)
        solver_stats = self._build_fdtd_solver_stats(
            solver,
            time_steps=time_steps,
            use_full_field_dft=use_full_field_dft,
            dft_cfg=dft_cfg,
        )
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
            metadata=self.metadata,
            solver_stats=solver_stats,
            raw_output=raw_output,
        )

    def _build_fdtd_solver_stats(
        self,
        solver,
        *,
        time_steps: int,
        use_full_field_dft: bool,
        dft_cfg: SpectralSampler,
    ) -> dict[str, Any]:
        elapsed_s = getattr(solver, "last_solve_elapsed_s", None)
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
        return {
            "time_steps": time_steps,
            "shutoff": self.config.shutoff,
            "shutoff_check_interval": self.config.shutoff_check_interval,
            "shutoff_triggered": shutoff_triggered,
            "shutoff_step": shutoff_step,
            "steps_run": (shutoff_step + 1) if shutoff_triggered else time_steps,
            "dt": solver.dt,
            "cuda_graph_active": bool(getattr(solver, "_cuda_graph_active", False)),
            "tail_graph_active": bool(getattr(solver, "_tail_graph_active", False)),
            "boundary": getattr(solver, "boundary_kind", self.scene.boundary.kind),
            "absorber": getattr(solver, "active_absorber_type", self.config.absorber),
            "cpml_memory_mode": getattr(solver, "_cpml_memory_mode", None),
            "cpml_requested_memory_mode": getattr(solver, "_cpml_memory_mode_requested", None),
            "cpml_allocated_memory_bytes": getattr(solver, "_cpml_allocated_memory_bytes", None),
            "cpml_dense_memory_bytes": getattr(solver, "_cpml_dense_memory_bytes", None),
            "cpml_slab_memory_bytes": getattr(solver, "_cpml_slab_memory_bytes", None),
            "dft_window": dft_cfg.window,
            "frequency": self.frequency,
            "frequencies": self.frequencies,
            "num_frequencies": len(self.frequencies),
            "dft_samples": dft_sample_counts[0] if dft_sample_counts else 0,
            "dft_sample_counts": dft_sample_counts,
            "observer_samples": observer_sample_counts[0] if observer_sample_counts else 0,
            "observer_sample_counts": observer_sample_counts,
            "full_field_dft": use_full_field_dft,
            "elapsed_s": elapsed_s,
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

    def _run_fdtd_with_gradient_bridge(self) -> Result:
        from .fdtd.adjoint import run_fdtd_with_gradient_bridge

        return run_fdtd_with_gradient_bridge(self)


class PreparedNetworkSweep:
    def __init__(self, simulation: Simulation, scene, manifest):
        self.simulation = simulation
        self.scene = scene
        self.manifest = manifest

    def run(self) -> Result:
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
    def __init__(self, simulation: Simulation, solver):
        self.simulation = simulation
        self.solver = solver

    def run(self) -> Result:
        if self.simulation.method == SimulationMethod.FDFD:
            solver_cfg = self.simulation.config.solver
            return self.simulation._build_fdfd_result(self.solver, solver_cfg)
        if self.simulation.has_trainable_parameters:
            return self.simulation._run_fdtd()
        return self.simulation._run_fdtd_from_solver(self.solver)



def run(simulation: Simulation) -> Result:
    if not isinstance(simulation, Simulation):
        raise TypeError("run() expects a maxwell.Simulation instance.")
    return simulation.run()
