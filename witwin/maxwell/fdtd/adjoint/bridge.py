from __future__ import annotations

import time

import torch

from ...adjoint_inputs import (
    material_dependent_inputs as _material_dependent_inputs,
    scene_trainable_material_tensors as _scene_trainable_material_tensors,
)
from ...result import Result
from ..boundary import BOUNDARY_BLOCH, BOUNDARY_NONE, BOUNDARY_PEC, BOUNDARY_PML, has_complex_fields
from .profiler import _BackwardProfiler, _clone_backward_profile, _empty_backward_profile
from ..excitation import (
    advance_tfsf_auxiliary_electric,
    advance_tfsf_auxiliary_magnetic,
    apply_tfsf_e_correction,
    apply_tfsf_h_correction,
)
from ..runtime.stepping import update_electric_fields_bloch_cpml
from .seeds import _build_output_seeds, _schedule_to_tensor_pack, _apply_seed_runtime
from ..checkpoint import capture_checkpoint_state
from ..material_pullback import pullback_material_input_gradients
from .dispatch import reverse_step


def _runtime():
    from . import core as _adjoint

    return _adjoint


def _material_has_conductivity(material) -> bool:
    sigma_e = getattr(material, "sigma_e", 0.0)
    if sigma_e is not None and float(sigma_e) != 0.0:
        return True
    sigma_e_tensor = getattr(material, "sigma_e_tensor", None)
    if sigma_e_tensor is not None:
        return any(float(component) != 0.0 for component in sigma_e_tensor.as_tuple())
    return False


def _unsupported_adjoint_medium(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if _material_has_conductivity(material):
            return "FDTD adjoint does not support static conductive (sigma_e) media yet."
        if getattr(material, "is_anisotropic", False):
            return "FDTD adjoint does not support anisotropic media yet."
        if getattr(material, "is_magnetic_dispersive", False):
            return "FDTD adjoint does not support magnetic dispersive media yet."
        if getattr(material, "is_nonlinear", False):
            return "FDTD adjoint does not support Kerr nonlinear media yet."
    return None


class _FDTDGradientBridge:
    def __init__(self, simulation):
        self.simulation = simulation
        self.base_scene = simulation.scene
        self.material_inputs = self._resolve_material_inputs()
        if not self.material_inputs:
            raise NotImplementedError(
                "FDTD backward currently supports trainable scene inputs that contribute to "
                "prepared-scene material tensors."
            )
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

    def _validate_supported_configuration(self, solver):
        unsupported_medium_message = _unsupported_adjoint_medium(solver.scene)
        if unsupported_medium_message is not None:
            raise NotImplementedError(unsupported_medium_message)
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
            if getattr(solver, "dispersive_enabled", False):
                raise NotImplementedError("FDTD adjoint does not support Bloch boundaries with dispersive media.")
        elif any(code not in {BOUNDARY_NONE, BOUNDARY_PML, BOUNDARY_PEC} for code in face_codes):
            raise NotImplementedError("FDTD adjoint currently supports none, pml, pec, and Bloch faces only.")
        compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
        if len(compiled_sources) > 1:
            raise NotImplementedError("FDTD adjoint currently supports at most one source per scene.")
        compiled_source = getattr(solver, "_compiled_source", None)
        if compiled_source is not None and compiled_source["kind"] not in {
            "point_dipole",
            "plane_wave",
            "gaussian_beam",
            "mode_source",
        }:
            raise NotImplementedError(
                "FDTD adjoint currently supports PointDipole, PlaneWave, GaussianBeam, and ModeSource source pullback only."
            )

    def _run_forward_with_checkpoints(self, solver, *, time_steps, dft_frequency, dft_window, full_field_dft, normalize_source):
        runtime = _runtime()
        if normalize_source and len(getattr(solver, "_compiled_sources", ())) != 1:
            raise NotImplementedError("normalize_source currently requires exactly one compiled source.")
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

        checkpoint_stride = runtime._checkpoint_stride(self.simulation, time_steps)
        checkpoints = [capture_checkpoint_state(solver, step=0)]
        for step_index in range(time_steps):
            if step_index > 0 and step_index % checkpoint_stride == 0:
                checkpoints.append(capture_checkpoint_state(solver, step=step_index))

            time_value = step_index * solver.dt
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

            solver._advance_dispersive_state()
            if has_complex_fields(solver):
                if getattr(solver, "uses_cpml", False):
                    update_electric_fields_bloch_cpml(solver)
                else:
                    solver._update_electric_fields_bloch()
            else:
                solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)

            if getattr(solver, "tfsf_enabled", False):
                apply_tfsf_e_correction(solver, time_value)
                advance_tfsf_auxiliary_electric(solver)
            if solver._electric_source_terms:
                from ..excitation import inject_electric_surface_source_terms

                inject_electric_surface_source_terms(solver, time_value=time_value)

            if solver._source_terms:
                solver.add_source(time_value=time_value)
            solver._apply_dispersive_corrections()
            if not getattr(solver, "tfsf_enabled", False):
                solver._enforce_pec_boundaries()
            solver.accumulate_dft(step_index)
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
        return raw_output or None, tuple(checkpoints)

    def forward(self, material_inputs):
        runtime = _runtime()
        del material_inputs
        with torch.no_grad():
            self.simulation._refresh_scene()
            scene = self.simulation.scene
            solver = self.simulation._build_fdtd_solver_for_scene(scene, initialize=True)
            self._validate_supported_configuration(solver)

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
        raw_output = {
            field_name: fields[field_name.upper()]
            for field_name in pack.field_names
        }
        if monitors:
            raw_output["observers"] = monitors
        return fields, monitors, raw_output

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

        eps_ex = solver.eps_Ex.detach().clone().requires_grad_(True)
        eps_ey = solver.eps_Ey.detach().clone().requires_grad_(True)
        eps_ez = solver.eps_Ez.detach().clone().requires_grad_(True)
        grad_eps_ex = torch.zeros_like(solver.eps_Ex)
        grad_eps_ey = torch.zeros_like(solver.eps_Ey)
        grad_eps_ez = torch.zeros_like(solver.eps_Ez)
        compiled_source = getattr(solver, "_compiled_source", None)
        cache_mode_source_terms = compiled_source is not None and compiled_source.get("kind") == "mode_source"
        if cache_mode_source_terms:
            solver._source_replay_term_cache = {}
            solver._mode_source_explicit_vjp_remaining = self._time_steps

        try:
            for start_step, end_step in reversed(segment_bounds):
                with profiler.section("segment_replay"):
                    states = runtime._replay_segment_states(
                        solver,
                        checkpoint_lookup[start_step],
                        start_step,
                        end_step,
                    )
                for offset in range(end_step - start_step - 1, -1, -1):
                    step_index = start_step + offset
                    post_step_adjoint = {
                        name: value.clone()
                        for name, value in adjoint_state.items()
                    }
                    with profiler.section("seed_injection"):
                        _apply_seed_runtime(post_step_adjoint, seed_runtime, step_index)

                    step_result = reverse_step(
                        solver,
                        states[offset],
                        post_step_adjoint,
                        time_value=step_index * solver.dt,
                        eps_ex=eps_ex,
                        eps_ey=eps_ey,
                        eps_ez=eps_ez,
                        profiler=profiler,
                    )
                    profiler.record_reverse_backend(step_result.backend)
                    grad_eps_ex = grad_eps_ex + step_result.grad_eps_ex
                    grad_eps_ey = grad_eps_ey + step_result.grad_eps_ey
                    grad_eps_ez = grad_eps_ez + step_result.grad_eps_ez
                    adjoint_state = step_result.pre_step_adjoint

            with profiler.section("material_pullback"):
                with torch.enable_grad():
                    scene = self._material_graph_scene()
                    outputs = pullback_material_input_gradients(
                        scene,
                        inputs=base_inputs,
                        grad_eps_ex=grad_eps_ex,
                        grad_eps_ey=grad_eps_ey,
                        grad_eps_ez=grad_eps_ez,
                        eps0=solver.eps0,
                    )
        finally:
            if hasattr(solver, "_mode_source_explicit_vjp_remaining"):
                delattr(solver, "_mode_source_explicit_vjp_remaining")
            if hasattr(solver, "_source_replay_term_cache"):
                cache = getattr(solver, "_source_replay_term_cache")
                if isinstance(cache, dict):
                    cache.clear()
                delattr(solver, "_source_replay_term_cache")
        profiler.record_material_pullback_backend("autograd_material_graph" if base_inputs else "none")

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
        resolved_base_inputs = tuple(self.material_inputs if base_inputs is None else base_inputs)
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
    raw_outputs = _FDTDMaterialGradientFunction.apply(bridge, *bridge.material_inputs)
    output_tensors = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)
    fields, monitors, raw_output = bridge.rebuild_forward_outputs(output_tensors)
    return Result(
        method="fdtd",
        scene=simulation.scene,
        prepared_scene=bridge._last_solver.scene if bridge._last_solver is not None else None,
        frequency=simulation.frequency,
        frequencies=simulation.frequencies,
        solver=bridge._last_solver,
        fields=fields,
        monitors=monitors,
        metadata=simulation.metadata,
        solver_stats=bridge._last_solver_stats,
        raw_output=raw_output,
    )
