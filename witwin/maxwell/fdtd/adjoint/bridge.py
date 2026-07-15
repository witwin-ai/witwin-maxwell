from __future__ import annotations

import time

import torch

from ...adjoint_inputs import (
    material_dependent_inputs as _material_dependent_inputs,
    rf_dependent_inputs as _rf_dependent_inputs,
    scene_trainable_material_tensors as _scene_trainable_material_tensors,
    scene_trainable_rf_tensors as _scene_trainable_rf_tensors,
    unique_trainable_tensors as _unique_trainable_tensors,
)
from ...media import Tensor3x3
from ...result import Result
from ..boundary import BOUNDARY_BLOCH, BOUNDARY_NONE, BOUNDARY_PEC, BOUNDARY_PML, has_complex_fields
from .profiler import _BackwardProfiler, _clone_backward_profile, _empty_backward_profile
from ..excitation import (
    advance_tfsf_auxiliary_electric,
    advance_tfsf_auxiliary_magnetic,
    apply_tfsf_e_correction,
    apply_tfsf_h_correction,
)
from ..runtime.stepping import apply_full_aniso_corrections, update_electric_fields_bloch_cpml
from ..ports import (
    accumulate_port_observers,
    apply_port_runtimes,
    finalize_port_data,
    prepare_port_spectral_accumulators,
    pullback_port_runtimes,
)
from .seeds import (
    _apply_seed_runtime,
    _build_output_seeds,
    _port_sample_adjoints,
    _schedule_to_tensor_pack,
)
from ..checkpoint import capture_checkpoint_state
from ..material_pullback import pullback_material_input_gradients
from .dispatch import reverse_step


def _runtime():
    from . import core as _adjoint

    return _adjoint


def _resolve_grad_accumulator_backend(device):
    """Require native in-place gradient accumulation for CUDA-only adjoints."""
    if torch.device(device).type != "cuda":
        raise RuntimeError("Differentiable FDTD requires CUDA gradient tensors.")
    from ..cuda import backend as _cuda_backend

    if not _cuda_backend.is_available():
        raise RuntimeError("Differentiable FDTD requires the native CUDA extension.")
    return _cuda_backend


def _accumulate_grad(cuda_backend, accumulator, increment):
    """Fold one gradient contribution into its native CUDA accumulator."""
    if not accumulator.is_contiguous() or not increment.is_contiguous():
        raise RuntimeError("Native FDTD gradient accumulation requires contiguous tensors.")
    cuda_backend._accumulate_in_place(dst=accumulator, src=increment)
    return accumulator


def _material_has_conductivity(material) -> bool:
    sigma_e = getattr(material, "sigma_e", 0.0)
    if sigma_e is not None and float(sigma_e) != 0.0:
        return True
    sigma_e_tensor = getattr(material, "sigma_e_tensor", None)
    if sigma_e_tensor is not None:
        return any(float(component) != 0.0 for component in sigma_e_tensor.as_tuple())
    return False


def _structure_has_trainable_geometry(structure) -> bool:
    geometry = getattr(structure, "geometry", None)
    if geometry is None:
        return False
    return any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in vars(geometry).values()
    )


def _source_spectrum_key(source_time):
    if source_time is None:
        return None
    if source_time.get("times") is not None:
        # Sampled custom waveforms are only spectrum-equivalent when they share
        # the same table object; identity keeps the comparison conservative.
        return ("custom", id(source_time))
    fields = ("kind_code", "frequency", "amplitude", "phase", "delay", "fwidth")
    return tuple(
        None if source_time.get(field) is None else float(source_time.get(field))
        for field in fields
    )


def _sources_share_spectrum(compiled_sources) -> bool:
    """Whether every compiled source carries the same source-time spectrum.

    Source normalization divides monitor spectra by one incident spectrum, which
    is only well-defined when all sources share it (a single source, or several
    driven by an identical waveform).
    """
    keys = {_source_spectrum_key(source.get("source_time")) for source in (compiled_sources or ())}
    return len(keys) <= 1


def _unsupported_adjoint_medium(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if getattr(material, "is_medium2d", False):
            return "FDTD adjoint does not support 2D sheet (Medium2D) media yet."
        if getattr(material, "is_lossy_metal", False):
            # The surface-impedance (Leontovich) boundary evolves per-face recursive
            # H state and overrides the tangential surface E outside the differentiable
            # material-tensor replay, so it carries no reverse gradient channel; a design
            # differentiated through a lossy-metal SIBC surface would drop its sensitivity.
            return (
                "FDTD adjoint does not support LossyMetalMedium (surface-impedance boundary) media: "
                "the Leontovich surface update runs outside the differentiable material replay and "
                "carries no reverse gradient channel."
            )
        if _material_has_conductivity(material) and getattr(material, "is_nonlinear", False):
            # Static conduction (sigma_e) is differentiable on its own through the
            # semi-implicit lossy-coefficient replay, but the instantaneous
            # nonlinear dynamic-curl kernel folds the loss into a field-dependent
            # coefficient the linear conduction replica does not reproduce.
            return (
                "FDTD adjoint does not support electric conductivity combined with nonlinear "
                "(chi3/chi2/TPA) media in the same material; the nonlinear dynamic-curl replay "
                "does not carry the static semi-implicit conduction-loss coefficient."
            )
        if getattr(material, "is_nonlinear", False) and getattr(material, "is_electric_dispersive", False):
            # The forward run subtracts the ADE polarization current against the
            # field-dependent effective permittivity that the instantaneous
            # nonlinearity produces (the dynamic curl coefficient), while the
            # reverse replay divides that current by the static linear
            # permittivity; the two disagree by the nonlinear index shift, so the
            # gradient through a same-material chi2/chi3 + dispersive medium would
            # not match its forward. Keep the forward composition (SHG phase
            # matching) usable while its adjoint pullback is unresolved.
            return (
                "FDTD adjoint does not support instantaneous nonlinear (chi2/chi3/TPA) media "
                "combined with electric dispersion in the same material; the reverse ADE "
                "current is divided by the static permittivity, not the field-dependent "
                "effective permittivity the forward nonlinear update uses."
            )
        if getattr(material, "is_anisotropic", False):
            # Diagonal epsilon anisotropy maps onto the per-axis eps_Ex/Ey/Ez
            # coefficient layout, and full 3x3 epsilon is now differentiable too:
            # the reverse step replicates the off-diagonal curl(H) coupling and
            # routes to the torch-VJP backend, so a design differentiated through
            # a full-anisotropic scene gets correct gradients. A magnetic tensor
            # still has no reverse gradient channel, and a trainable geometry on
            # a full-anisotropic structure would drop the off-diagonal coefficient
            # sensitivity, so both stay guarded with a physical reason.
            if getattr(material, "mu_tensor", None) is not None:
                return (
                    "FDTD adjoint does not support anisotropic magnetic (mu_tensor) media: "
                    "the magnetic reverse update carries no tensor gradient channel."
                )
            if isinstance(getattr(material, "epsilon_tensor", None), Tensor3x3) and _material_has_conductivity(material):
                # The forward coupled-tensor conduction folds the loss through the
                # off-diagonal inverse of (eps_inf + dt/2 diag(sigma)) applied to both
                # curl(H) and the conduction current sigma . E, but the reverse
                # conduction replica recomputes the semi-implicit decay/curl from the
                # per-axis effective permittivity and carries no off-diagonal (coupled
                # tensor) conduction channel, so the gradient through a full-anisotropic
                # lossy medium would not match its forward. Keep the forward composition
                # (lossy anisotropic crystal) usable while its adjoint pullback is
                # unresolved.
                return (
                    "FDTD adjoint does not support full (off-diagonal) anisotropic permittivity "
                    "combined with electric conductivity in the same material; the reverse conduction "
                    "replica folds the loss through the per-axis effective permittivity and lacks the "
                    "off-diagonal coupled-tensor conduction channel the forward semi-implicit update applies."
                )
            if isinstance(getattr(material, "epsilon_tensor", None), Tensor3x3) and getattr(
                material, "is_electric_dispersive", False
            ):
                # The forward coupled-tensor ADE folds the polarization current
                # through the full off-diagonal inverse permittivity tensor, but the
                # reverse dispersive correction only divides the current by the
                # per-axis effective permittivity and carries no off-diagonal
                # coupling, so the gradient through a full-anisotropic dispersive
                # medium would not match its forward. Keep the forward composition
                # (rotated birefringent dispersive crystal) usable while its adjoint
                # pullback is unresolved.
                return (
                    "FDTD adjoint does not support full (off-diagonal) anisotropic permittivity "
                    "combined with electric dispersion in the same material; the reverse ADE "
                    "correction lacks the off-diagonal inverse-permittivity coupling the forward "
                    "coupled-tensor update applies."
                )
            if isinstance(getattr(material, "epsilon_tensor", None), Tensor3x3) and _structure_has_trainable_geometry(structure):
                return (
                    "FDTD adjoint does not support trainable geometry on full (off-diagonal) "
                    "anisotropic structures: the off-diagonal coupling coefficients carry no "
                    "material gradient channel, so the geometry sensitivity would be dropped."
                )
        if getattr(material, "is_magnetic_dispersive", False) and _structure_has_trainable_geometry(structure):
            # The reverse step models static magnetic ADE poles, but there is no
            # mu-side material gradient channel, so a trainable geometry on a
            # magnetic-dispersive structure would silently drop the pole-weight
            # sensitivity.
            return (
                "FDTD adjoint does not support trainable geometry on magnetic dispersive "
                "structures (no mu gradient channel) yet."
            )
        if getattr(material, "is_modulated", False):
            return "FDTD adjoint does not support time-modulated media yet."
    return None


class _FDTDGradientBridge:
    def __init__(self, simulation):
        self.simulation = simulation
        self.base_scene = simulation.scene
        self.material_inputs = self._resolve_material_inputs()
        self.rf_scene_inputs = self._resolve_rf_scene_inputs()
        self.excitation_inputs = _unique_trainable_tensors(
            value
            for excitation in getattr(simulation, "excitations", ())
            for value in (excitation.amplitude, excitation.source_impedance)
        )
        self.trainable_inputs = _unique_trainable_tensors(
            self.material_inputs + self.rf_scene_inputs + self.excitation_inputs
        )
        if not self.trainable_inputs:
            raise NotImplementedError(
                "FDTD backward requires an input that contributes to prepared-scene material tensors, "
                "series R/L/C values, or a port amplitude."
            )
        if any(not tensor.is_cuda for tensor in self.trainable_inputs):
            raise RuntimeError("Differentiable FDTD RF and material inputs must be CUDA tensors.")
        self._last_solver = None
        self._last_pack = None
        self._last_solver_stats = None
        self._last_checkpoints = ()
        self._checkpoint_schema = None
        self._time_steps = 0
        self._dft_schedule = ()
        self._observer_schedule = ()
        self._dft_schedule_pack = None
        self._observer_schedule_pack = None
        self._last_backward_profile = None

    def _material_graph_scene(self):
        if self.simulation.scene_module is not None:
            return self.simulation.scene_module.to_scene()
        self.simulation._refresh_scene()
        return self.simulation.scene

    def _candidate_material_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.simulation.scene_module is not None:
            return tuple(parameter for parameter in self.simulation.scene_module.parameters() if parameter.requires_grad)
        return _scene_trainable_material_tensors(self.base_scene)

    def _resolve_material_inputs(self) -> tuple[torch.Tensor, ...]:
        scene = self._material_graph_scene()
        return _material_dependent_inputs(scene, self._candidate_material_inputs())

    def _resolve_rf_scene_inputs(self) -> tuple[torch.Tensor, ...]:
        scene = self._material_graph_scene()
        if self.simulation.scene_module is not None:
            candidates = tuple(
                parameter
                for parameter in self.simulation.scene_module.parameters()
                if parameter.requires_grad
            )
        else:
            candidates = _scene_trainable_rf_tensors(self.base_scene)
        return _rf_dependent_inputs(scene, candidates)

    @staticmethod
    def _input_key(tensor: torch.Tensor):
        return tensor.data_ptr() if tensor.data_ptr() != 0 else id(tensor)

    def _pullback_rf_inputs(self, base_inputs, semantic_grads):
        if not semantic_grads:
            return tuple(torch.zeros_like(tensor) for tensor in base_inputs)
        scene = self._material_graph_scene()
        values = {}
        for port in getattr(scene, "ports", ()):
            termination = getattr(port, "termination", None)
            if termination is None:
                continue
            for component in ("r", "l", "c"):
                value = getattr(termination, component, None)
                if isinstance(value, torch.Tensor):
                    values[("port", port.name, component)] = value
        for element in getattr(scene, "lumped_elements", ()):
            values[("element", element.name, "value")] = element.value
        for excitation in getattr(self.simulation, "excitations", ()):
            values[("excitation", excitation.port_name, "amplitude")] = excitation.amplitude

        outputs = []
        output_grads = []
        for key, gradient in semantic_grads.items():
            value = values.get(key)
            if value is None or not value.requires_grad:
                continue
            outputs.append(value)
            output_grads.append(
                gradient.to(device=value.device, dtype=value.dtype)
            )
        if not outputs:
            return tuple(torch.zeros_like(tensor) for tensor in base_inputs)
        gradients = torch.autograd.grad(
            outputs,
            base_inputs,
            grad_outputs=output_grads,
            allow_unused=True,
        )
        return tuple(
            torch.zeros_like(tensor) if gradient is None else gradient
            for tensor, gradient in zip(base_inputs, gradients)
        )

    def _validate_supported_configuration(self, solver):
        unsupported_medium_message = _unsupported_adjoint_medium(solver.scene)
        if unsupported_medium_message is not None:
            raise NotImplementedError(unsupported_medium_message)
        if getattr(solver, "_full_aniso_cpml_overlap", False):
            # The forward off-diagonal correction coordinate-stretches the tensor
            # coupling inside the CPML with per-direction psi memory, but the
            # reverse replay reconstructs the off-diagonal coupling from the raw
            # (un-stretched) collocated curl, so a design differentiated through an
            # anisotropic structure overlapping the absorber would receive a
            # gradient inconsistent with its forward. Keep the anisotropic
            # structure clear of the absorber for adjoint runs.
            raise NotImplementedError(
                "FDTD adjoint does not support full (off-diagonal) anisotropic media overlapping "
                "the CPML absorber: the reverse off-diagonal correction uses the un-stretched "
                "collocated curl and does not replay the CPML coordinate stretch the forward applies."
            )
        face_codes = (
            int(solver.boundary_x_low_code),
            int(solver.boundary_x_high_code),
            int(solver.boundary_y_low_code),
            int(solver.boundary_y_high_code),
            int(solver.boundary_z_low_code),
            int(solver.boundary_z_high_code),
        )
        if any(code == BOUNDARY_BLOCH for code in face_codes):
            if not has_complex_fields(solver):
                raise NotImplementedError("FDTD adjoint requires complex field state for Bloch faces.")
            if getattr(solver, "conductive_enabled", False):
                raise NotImplementedError(
                    "FDTD adjoint does not support electric conductivity on Bloch boundaries; the "
                    "complex-field reverse replay applies the semi-implicit conduction-loss "
                    "coefficient on the real electric update only."
                )
        elif any(code not in {BOUNDARY_NONE, BOUNDARY_PML, BOUNDARY_PEC} for code in face_codes):
            raise NotImplementedError("FDTD adjoint currently supports none, pml, pec, and Bloch faces only.")
        # Every compiled source kind (point/plane/beam/mode plus the uniform and
        # custom current / field sources) injects an additive current whose patch
        # divides by eps at the injection cell, so the explicit reverse's analytic
        # 1/eps source pullback (or the torch-VJP fallback) differentiates the
        # design through it; there is no source-kind that has to be rejected here.
        for runtime in getattr(solver, "_port_runtimes", ()):
            if runtime.lumped is None:
                raise NotImplementedError(
                    f"Differentiable RF port {runtime.port.name!r} requires an active source "
                    "or a series R/L/C termination; observer-only contour ports are not yet supported."
                )
            if runtime.lumped.topology != "series":
                raise NotImplementedError(
                    f"Differentiable RF port {runtime.port.name!r} requires series R/L/C topology."
                )
            if bool(runtime.lumped.resistance_is_open):
                raise NotImplementedError(
                    f"Differentiable RF port {runtime.port.name!r} does not support an open "
                    "internal series resistance."
                )
        for runtime, _field_name in getattr(solver, "_lumped_element_runtimes", ()):
            if runtime.topology != "series":
                raise NotImplementedError(
                    f"Differentiable lumped element {runtime.port_name!r} requires series topology."
                )
            if bool(runtime.resistance_is_open):
                raise NotImplementedError(
                    f"Differentiable lumped element {runtime.port_name!r} does not support an "
                    "open internal series resistance."
                )

    def _run_forward_with_checkpoints(self, solver, *, time_steps, dft_frequency, dft_window, full_field_dft, normalize_source):
        runtime = _runtime()
        if normalize_source and not _sources_share_spectrum(getattr(solver, "_compiled_sources", ())):
            raise NotImplementedError(
                "normalize_source divides monitor spectra by a single incident source "
                "spectrum, which is ill-defined when the scene drives sources with "
                "different source-time spectra; give the sources a common source_time "
                "or disable normalize_source."
            )
        solver._normalize_source = normalize_source
        solver._synchronize_device()
        solve_start = time.perf_counter()

        if dft_frequency is not None and full_field_dft:
            solver.enable_dft(dft_frequency, window_type=dft_window, end_step=time_steps)
        else:
            solver.dft_enabled = False
            solver._dft_entries = []
            solver._sync_dft_legacy_state()

        observer_frequency = dft_frequency if dft_frequency is not None else solver.source_frequency
        if solver.observers:
            solver._prepare_observers(observer_frequency, dft_window, time_steps)
        prepare_port_spectral_accumulators(solver, time_steps, dft_window)

        checkpoint_stride = runtime._checkpoint_stride(self.simulation, time_steps)
        checkpoints = [capture_checkpoint_state(solver, step=0)]
        for step_index in range(time_steps):
            if step_index > 0 and step_index % checkpoint_stride == 0:
                checkpoints.append(capture_checkpoint_state(solver, step=step_index))

            time_value = step_index * solver.dt
            solver._advance_magnetic_dispersive_state()
            solver._update_magnetic_fields(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
            if has_complex_fields(solver):
                solver._update_magnetic_fields(
                    solver.Hx_imag,
                    solver.Hy_imag,
                    solver.Hz_imag,
                    solver.Ex_imag,
                    solver.Ey_imag,
                    solver.Ez_imag,
                )
            if getattr(solver, "tfsf_enabled", False):
                apply_tfsf_h_correction(solver, time_value)
                advance_tfsf_auxiliary_magnetic(solver)
            if solver._magnetic_source_terms:
                from ..excitation import inject_magnetic_surface_source_terms

                inject_magnetic_surface_source_terms(solver, time_value=time_value)
            solver._apply_magnetic_dispersive_corrections()

            solver._advance_dispersive_state()
            if has_complex_fields(solver):
                if getattr(solver, "uses_cpml", False):
                    update_electric_fields_bloch_cpml(solver)
                else:
                    solver._update_electric_fields_bloch()
            else:
                if getattr(solver, "nonlinear_enabled", False):
                    solver._update_nonlinear_electric_coefficients()
                solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
                if getattr(solver, "full_aniso_enabled", False):
                    apply_full_aniso_corrections(solver)

            if getattr(solver, "tfsf_enabled", False):
                apply_tfsf_e_correction(solver, time_value)
                advance_tfsf_auxiliary_electric(solver)
            if solver._electric_source_terms:
                from ..excitation import inject_electric_surface_source_terms

                inject_electric_surface_source_terms(solver, time_value=time_value)

            if solver._source_terms:
                solver.add_source(time_value=time_value)
            apply_port_runtimes(solver)
            solver._apply_dispersive_corrections()
            if not getattr(solver, "tfsf_enabled", False):
                solver._enforce_pec_boundaries()
            solver.accumulate_dft(step_index)
            accumulate_port_observers(solver)
            solver.accumulate_observers(step_index)

        solver._synchronize_device()
        solver.last_solve_elapsed_s = time.perf_counter() - solve_start

        if solver.dft_enabled:
            solver._sync_dft_legacy_state()
        if solver.observers_enabled:
            solver._sync_observer_legacy_state()

        raw_output = {}
        if solver.dft_enabled:
            raw_output.update(solver.get_frequency_solution(all_frequencies=True))
        if solver.observers_enabled:
            raw_output["observers"] = solver.get_observer_results()
        ports = finalize_port_data(solver)
        if ports:
            raw_output["ports"] = ports
        return raw_output or None, tuple(checkpoints)

    def forward(self, material_inputs):
        runtime = _runtime()
        del material_inputs
        with torch.no_grad():
            self.simulation._refresh_scene()
            scene = self.simulation.scene
            solver = self.simulation._build_fdtd_solver_for_scene(scene, initialize=True)
            self._validate_supported_configuration(solver)
            from .dispatch import validate_native_adjoint_preparation

            validate_native_adjoint_preparation(solver)

            time_steps = self.simulation._resolve_fdtd_time_steps(solver, scene)
            dft_cfg = self.simulation.config.spectral_sampler
            requested_full_field_dft = self.simulation.config.full_field_dft
            use_full_field_dft = requested_full_field_dft or len(self.simulation.frequencies) > 1
            dft_request = self.simulation.frequency if len(self.simulation.frequencies) == 1 else self.simulation.frequencies
            raw_output, checkpoints = self._run_forward_with_checkpoints(
                solver,
                time_steps=time_steps,
                dft_frequency=dft_request,
                dft_window=dft_cfg.window,
                full_field_dft=use_full_field_dft,
                normalize_source=dft_cfg.normalize_source,
            )
        if raw_output is None:
            raise RuntimeError("FDTD solve did not return any output.")

        pack = runtime._prepare_forward_pack(raw_output)
        checkpoint_schema = runtime._validate_checkpoint_sequence(checkpoints)
        self._last_solver = solver
        self._last_pack = pack
        self._last_checkpoints = checkpoints
        self._checkpoint_schema = checkpoint_schema
        self._time_steps = time_steps
        self._dft_schedule = runtime._build_spectral_weight_schedule(
            getattr(solver, "_dft_entries", ()),
            time_steps=time_steps,
            window_type=getattr(solver, "dft_window_type", "none"),
        )
        self._observer_schedule = runtime._build_spectral_weight_schedule(
            getattr(solver, "_observer_spectral_entries", ()),
            time_steps=time_steps,
            window_type=getattr(solver, "observer_window_type", "none"),
        )
        self._dft_schedule_pack = _schedule_to_tensor_pack(
            self._dft_schedule,
            device=solver.device,
            dtype=solver.eps_Ex.dtype,
        )
        self._observer_schedule_pack = _schedule_to_tensor_pack(
            self._observer_schedule,
            device=solver.device,
            dtype=solver.eps_Ex.dtype,
        )
        self._last_solver_stats = self.simulation._build_fdtd_solver_stats(
            solver,
            time_steps=time_steps,
            use_full_field_dft=use_full_field_dft,
            dft_cfg=dft_cfg,
        )
        self._last_backward_profile = None
        return pack.output_tensors

    def rebuild_forward_outputs(self, output_tensors):
        runtime = _runtime()
        pack = self._last_pack
        fields = {
            field_name.upper(): output_tensors[index]
            for index, field_name in enumerate(pack.field_names)
        }
        monitors = runtime._rebuild_monitors(pack.monitor_templates, output_tensors, len(pack.field_names))
        ports = runtime._rebuild_ports(
            pack.port_templates,
            output_tensors,
            pack.port_offset,
        )
        raw_output = {
            field_name: fields[field_name.upper()]
            for field_name in pack.field_names
        }
        if monitors:
            raw_output["observers"] = monitors
        if ports:
            raw_output["ports"] = ports
        return fields, monitors, ports, raw_output

    def _backward_impl(self, base_inputs, grad_outputs, *, profile_enabled: bool):
        runtime = _runtime()
        checkpoint_stride = runtime._checkpoint_stride(self.simulation, self._time_steps)
        if all(grad_output is None for grad_output in grad_outputs):
            if profile_enabled:
                solver = self._last_solver
                timer = (
                    "cuda_event"
                    if solver is not None
                    and torch.cuda.is_available()
                    and torch.device(solver.device).type == "cuda"
                    else "perf_counter"
                )
                self._last_backward_profile = _empty_backward_profile(
                    timer=timer,
                    steps=self._time_steps,
                    checkpoint_stride=checkpoint_stride,
                )
            return tuple(torch.zeros_like(tensor) for tensor in base_inputs)

        solver = self._last_solver
        if solver is None or self._last_pack is None or not self._last_checkpoints:
            raise RuntimeError("FDTD backward called before a forward pass initialized the adjoint bridge.")

        checkpoint_schema = self._checkpoint_schema or runtime._validate_checkpoint_sequence(self._last_checkpoints)
        profiler = _BackwardProfiler(enabled=profile_enabled, device=torch.device(solver.device))
        profiler.start_total()

        with profiler.section("seed_build"):
            seed_runtime = _build_output_seeds(
                solver,
                self._last_pack,
                grad_outputs,
                dft_schedule=(
                    self._dft_schedule_pack
                    if self._dft_schedule_pack is not None
                    else _schedule_to_tensor_pack(self._dft_schedule, device=solver.device, dtype=solver.eps_Ex.dtype)
                ),
                observer_schedule=(
                    self._observer_schedule_pack
                    if self._observer_schedule_pack is not None
                    else _schedule_to_tensor_pack(self._observer_schedule, device=solver.device, dtype=solver.eps_Ex.dtype)
                ),
            )
        profiler.record_seed_runtime(
            seed_runtime.backend,
            dense=len(seed_runtime.dense_batches),
            point=len(seed_runtime.point_batches),
            plane=len(seed_runtime.plane_batches),
        )

        checkpoint_lookup = {checkpoint.step: checkpoint for checkpoint in self._last_checkpoints}
        checkpoint_steps = [checkpoint.step for checkpoint in self._last_checkpoints]
        segment_bounds = list(zip(checkpoint_steps, checkpoint_steps[1:] + [self._time_steps]))

        state_names = checkpoint_schema.state_names
        adjoint_state = {
            name: torch.zeros_like(self._last_checkpoints[0].tensors[name])
            for name in state_names
        }

        grad_accumulator_backend = _resolve_grad_accumulator_backend(solver.device)
        eps_ex = solver.eps_Ex.detach().clone().requires_grad_(True)
        eps_ey = solver.eps_Ey.detach().clone().requires_grad_(True)
        eps_ez = solver.eps_Ez.detach().clone().requires_grad_(True)
        grad_eps_ex = torch.zeros_like(solver.eps_Ex)
        grad_eps_ey = torch.zeros_like(solver.eps_Ey)
        grad_eps_ez = torch.zeros_like(solver.eps_Ez)
        semantic_rf_grads = {}
        nonlinear_enabled = bool(getattr(solver, "nonlinear_enabled", False))
        general_enabled = bool(getattr(solver, "nonlinear_general_enabled", False))
        chi3_ex = chi3_ey = chi3_ez = None
        grad_chi3_ex = grad_chi3_ey = grad_chi3_ez = None
        chi2_ex = chi2_ey = chi2_ez = None
        grad_chi2_ex = grad_chi2_ey = grad_chi2_ez = None
        tpa_ex = tpa_ey = tpa_ez = None
        grad_tpa_ex = grad_tpa_ey = grad_tpa_ez = None
        if nonlinear_enabled:
            # chi3 leaves feed both the curl-only Kerr kernel and the general
            # nonlinear kernel (which also reads chi3), so allocate them whenever
            # any instantaneous nonlinear channel is active.
            chi3_ex = solver.kerr_chi3_Ex.detach().clone().requires_grad_(True)
            chi3_ey = solver.kerr_chi3_Ey.detach().clone().requires_grad_(True)
            chi3_ez = solver.kerr_chi3_Ez.detach().clone().requires_grad_(True)
            grad_chi3_ex = torch.zeros_like(solver.kerr_chi3_Ex)
            grad_chi3_ey = torch.zeros_like(solver.kerr_chi3_Ey)
            grad_chi3_ez = torch.zeros_like(solver.kerr_chi3_Ez)
        if general_enabled:
            # chi2 second-order susceptibility and the TPA conductivity scale
            # drive the general nonlinear coefficient kernel; carry their own
            # gradient channels so a trainable chi2 / TPA design is differentiable.
            chi2_ex = solver.nonlinear_chi2_Ex.detach().clone().requires_grad_(True)
            chi2_ey = solver.nonlinear_chi2_Ey.detach().clone().requires_grad_(True)
            chi2_ez = solver.nonlinear_chi2_Ez.detach().clone().requires_grad_(True)
            tpa_ex = solver.tpa_sigma_Ex.detach().clone().requires_grad_(True)
            tpa_ey = solver.tpa_sigma_Ey.detach().clone().requires_grad_(True)
            tpa_ez = solver.tpa_sigma_Ez.detach().clone().requires_grad_(True)
            grad_chi2_ex = torch.zeros_like(solver.nonlinear_chi2_Ex)
            grad_chi2_ey = torch.zeros_like(solver.nonlinear_chi2_Ey)
            grad_chi2_ez = torch.zeros_like(solver.nonlinear_chi2_Ez)
            grad_tpa_ex = torch.zeros_like(solver.tpa_sigma_Ex)
            grad_tpa_ey = torch.zeros_like(solver.tpa_sigma_Ey)
            grad_tpa_ez = torch.zeros_like(solver.tpa_sigma_Ez)
        # The compiled source-term set (curl-coefficient patches and, for mode
        # sources, the eigensolve profile) depends only on the solver and the frozen
        # eps leaves, both constant across the backward sweep, so resolve it once and
        # hand the same lists to every reverse step instead of rebuilding the replay
        # solver (and re-running the mode eigensolve) per step.
        resolved_source_terms = runtime._resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez)
        if hasattr(solver, "_mode_source_cotangent_accum"):
            delattr(solver, "_mode_source_cotangent_accum")

        # The checkpoint replay reconstructs the mid-step (post-magnetic) H on its
        # way to each step's electric update; for the pure real standard / CPML
        # path it is bit-identical to the reverse reference backend's own magnetic
        # half-step recompute, so capture it once and hand it to the reverse step
        # instead of recomputing it there.
        capture_mid_magnetic = runtime._replay_can_capture_mid_magnetic(solver)
        try:
            for start_step, end_step in reversed(segment_bounds):
                mid_magnetic = [] if capture_mid_magnetic else None
                lumped_traces = [] if checkpoint_schema.lumped_state_names else None
                with profiler.section("segment_replay"):
                    states = runtime._replay_segment_states(
                        solver,
                        checkpoint_lookup[start_step],
                        start_step,
                        end_step,
                        mid_magnetic_out=mid_magnetic,
                        lumped_traces_out=lumped_traces,
                    )
                for offset in range(end_step - start_step - 1, -1, -1):
                    step_index = start_step + offset
                    post_step_adjoint = {
                        name: value.clone()
                        for name, value in adjoint_state.items()
                    }
                    with profiler.section("seed_injection"):
                        _apply_seed_runtime(post_step_adjoint, seed_runtime, step_index)

                    local_grad_eps = {
                        "Ex": torch.zeros_like(eps_ex),
                        "Ey": torch.zeros_like(eps_ey),
                        "Ez": torch.zeros_like(eps_ez),
                    }
                    if lumped_traces is not None:
                        post_step_adjoint, local_grad_eps, step_rf_grads = pullback_port_runtimes(
                            solver,
                            lumped_traces[offset],
                            post_step_adjoint,
                            port_sample_adjoints=_port_sample_adjoints(
                                seed_runtime,
                                step_index,
                            ),
                            eps_by_field={"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez},
                            time_value=step_index * solver.dt,
                        )
                        for key, value in step_rf_grads.items():
                            semantic_rf_grads[key] = semantic_rf_grads.get(
                                key,
                                torch.zeros_like(value),
                            ) + value

                    step_result = reverse_step(
                        solver,
                        states[offset],
                        post_step_adjoint,
                        time_value=step_index * solver.dt,
                        eps_ex=eps_ex,
                        eps_ey=eps_ey,
                        eps_ez=eps_ez,
                        resolved_source_terms=resolved_source_terms,
                        forward_magnetic_fields=(mid_magnetic[offset] if mid_magnetic else None),
                        chi3_ex=chi3_ex,
                        chi3_ey=chi3_ey,
                        chi3_ez=chi3_ez,
                        chi2_ex=chi2_ex,
                        chi2_ey=chi2_ey,
                        chi2_ez=chi2_ez,
                        tpa_ex=tpa_ex,
                        tpa_ey=tpa_ey,
                        tpa_ez=tpa_ez,
                        profiler=profiler,
                    )
                    profiler.record_reverse_backend(step_result.backend)
                    grad_eps_ex = _accumulate_grad(grad_accumulator_backend, grad_eps_ex, step_result.grad_eps_ex)
                    grad_eps_ey = _accumulate_grad(grad_accumulator_backend, grad_eps_ey, step_result.grad_eps_ey)
                    grad_eps_ez = _accumulate_grad(grad_accumulator_backend, grad_eps_ez, step_result.grad_eps_ez)
                    grad_eps_ex = _accumulate_grad(
                        grad_accumulator_backend,
                        grad_eps_ex,
                        local_grad_eps["Ex"].contiguous(),
                    )
                    grad_eps_ey = _accumulate_grad(
                        grad_accumulator_backend,
                        grad_eps_ey,
                        local_grad_eps["Ey"].contiguous(),
                    )
                    grad_eps_ez = _accumulate_grad(
                        grad_accumulator_backend,
                        grad_eps_ez,
                        local_grad_eps["Ez"].contiguous(),
                    )
                    if nonlinear_enabled:
                        if step_result.grad_chi3_ex is None:
                            raise RuntimeError(
                                "Nonlinear reverse step did not produce chi3 gradients; "
                                f"backend {step_result.backend!r} lacks the chi3 channel."
                            )
                        grad_chi3_ex = _accumulate_grad(grad_accumulator_backend, grad_chi3_ex, step_result.grad_chi3_ex)
                        grad_chi3_ey = _accumulate_grad(grad_accumulator_backend, grad_chi3_ey, step_result.grad_chi3_ey)
                        grad_chi3_ez = _accumulate_grad(grad_accumulator_backend, grad_chi3_ez, step_result.grad_chi3_ez)
                    if general_enabled:
                        if step_result.grad_chi2_ex is None:
                            raise RuntimeError(
                                "General-nonlinear reverse step did not produce chi2 / TPA "
                                f"gradients; backend {step_result.backend!r} lacks the channels."
                            )
                        grad_chi2_ex = _accumulate_grad(grad_accumulator_backend, grad_chi2_ex, step_result.grad_chi2_ex)
                        grad_chi2_ey = _accumulate_grad(grad_accumulator_backend, grad_chi2_ey, step_result.grad_chi2_ey)
                        grad_chi2_ez = _accumulate_grad(grad_accumulator_backend, grad_chi2_ez, step_result.grad_chi2_ez)
                        grad_tpa_ex = _accumulate_grad(grad_accumulator_backend, grad_tpa_ex, step_result.grad_tpa_ex)
                        grad_tpa_ey = _accumulate_grad(grad_accumulator_backend, grad_tpa_ey, step_result.grad_tpa_ey)
                        grad_tpa_ez = _accumulate_grad(grad_accumulator_backend, grad_tpa_ez, step_result.grad_tpa_ez)
                    adjoint_state = dict(step_result.pre_step_adjoint)
                    for name in checkpoint_schema.lumped_state_names:
                        adjoint_state[name] = post_step_adjoint[name]

            with profiler.section("material_pullback"):
                # Mode-source profile terms deferred their eigensolve VJP to keep
                # torch.autograd.grad off the per-step hot path; apply the single
                # accumulated pullback here (over the retained source-term graph)
                # before the material pullback consumes grad_eps.
                profile_grads = runtime._mode_source_profile_pullback(
                    solver, resolved_source_terms, eps_ex, eps_ey, eps_ez
                )
                if profile_grads is not None:
                    grad_profile_ex, grad_profile_ey, grad_profile_ez = profile_grads
                    if grad_profile_ex is not None:
                        grad_eps_ex = _accumulate_grad(grad_accumulator_backend, grad_eps_ex, grad_profile_ex.detach())
                    if grad_profile_ey is not None:
                        grad_eps_ey = _accumulate_grad(grad_accumulator_backend, grad_eps_ey, grad_profile_ey.detach())
                    if grad_profile_ez is not None:
                        grad_eps_ez = _accumulate_grad(grad_accumulator_backend, grad_eps_ez, grad_profile_ez.detach())
                with torch.enable_grad():
                    material_outputs = ()
                    if self.material_inputs:
                        scene = self._material_graph_scene()
                        material_outputs = pullback_material_input_gradients(
                            scene,
                            inputs=self.material_inputs,
                            grad_eps_ex=grad_eps_ex,
                            grad_eps_ey=grad_eps_ey,
                            grad_eps_ez=grad_eps_ez,
                            eps0=solver.eps0,
                            grad_chi3_ex=grad_chi3_ex,
                            grad_chi3_ey=grad_chi3_ey,
                            grad_chi3_ez=grad_chi3_ez,
                            grad_chi2_ex=grad_chi2_ex,
                            grad_chi2_ey=grad_chi2_ey,
                            grad_chi2_ez=grad_chi2_ez,
                            grad_tpa_ex=grad_tpa_ex,
                            grad_tpa_ey=grad_tpa_ey,
                            grad_tpa_ez=grad_tpa_ez,
                        )
                    rf_outputs = self._pullback_rf_inputs(
                        base_inputs,
                        semantic_rf_grads,
                    )
                    material_by_key = {
                        self._input_key(tensor): gradient
                        for tensor, gradient in zip(
                            self.material_inputs,
                            material_outputs,
                        )
                    }
                    outputs = tuple(
                        rf_gradient
                        + material_by_key.get(
                            self._input_key(tensor),
                            torch.zeros_like(tensor),
                        )
                        for tensor, rf_gradient in zip(base_inputs, rf_outputs)
                    )
        finally:
            if hasattr(solver, "_mode_source_cotangent_accum"):
                delattr(solver, "_mode_source_cotangent_accum")
        profiler.record_material_pullback_backend(
            (
                "autograd_material_and_rf_graph"
                if self.rf_scene_inputs or self.excitation_inputs
                else "autograd_material_graph"
            )
            if base_inputs
            else "none"
        )

        profiler.stop_total()
        if profile_enabled:
            self._last_backward_profile = profiler.summary(
                steps=self._time_steps,
                segments=len(segment_bounds),
                checkpoint_stride=checkpoint_stride,
            )
        return outputs

    def backward(self, base_inputs, grad_outputs):
        return self._backward_impl(base_inputs, grad_outputs, profile_enabled=False)

    def backward_profile(self, grad_outputs=None, *, base_inputs=None):
        if self._last_pack is None or self._last_solver is None:
            raise RuntimeError("Run forward() before requesting a backward profile.")
        resolved_grad_outputs = (
            tuple(torch.ones_like(output) for output in self._last_pack.output_tensors)
            if grad_outputs is None
            else tuple(grad_outputs)
        )
        resolved_base_inputs = tuple(self.trainable_inputs if base_inputs is None else base_inputs)
        self._backward_impl(
            resolved_base_inputs,
            resolved_grad_outputs,
            profile_enabled=True,
        )
        return _clone_backward_profile(self._last_backward_profile)


class _FDTDMaterialGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, *material_inputs):
        ctx.bridge = bridge
        ctx.material_inputs = tuple(material_inputs)
        outputs = bridge.forward(tuple(material_inputs))
        return outputs if len(outputs) != 1 else outputs[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        gradients = ctx.bridge.backward(ctx.material_inputs, grad_outputs)
        return (None, *gradients)


def run_fdtd_with_gradient_bridge(simulation) -> Result:
    bridge = _FDTDGradientBridge(simulation)
    raw_outputs = _FDTDMaterialGradientFunction.apply(bridge, *bridge.trainable_inputs)
    output_tensors = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)
    fields, monitors, ports, raw_output = bridge.rebuild_forward_outputs(output_tensors)
    return Result(
        method="fdtd",
        scene=simulation.scene,
        prepared_scene=bridge._last_solver.scene if bridge._last_solver is not None else None,
        frequency=simulation.frequency,
        frequencies=simulation.frequencies,
        solver=bridge._last_solver,
        fields=fields,
        monitors=monitors,
        ports=ports,
        metadata=simulation.metadata,
        solver_stats=bridge._last_solver_stats,
        raw_output=raw_output,
    )
