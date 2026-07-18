from __future__ import annotations

import numpy as np
import torch

from ..scene import prepare_scene
from .excitation import inject_source_terms
from .io import save_frequency_solution as save_frequency_solution_impl
from .observers import (
    accumulate_observers as accumulate_observers_impl,
    accumulate_time_observers as accumulate_time_observers_impl,
    add_plane_observer as add_plane_observer_impl,
    add_point_observer as add_point_observer_impl,
    clear_observers as clear_observers_impl,
    clear_time_observers as clear_time_observers_impl,
    get_component_coords as get_component_coords_impl,
    get_observer_results as get_observer_results_impl,
    get_time_observer_results as get_time_observer_results_impl,
    prepare_observers as prepare_observers_impl,
    prepare_time_observers as prepare_time_observers_impl,
    resolve_plane_observer as resolve_plane_observer_impl,
    resolve_point_observer as resolve_point_observer_impl,
)
from .plotting import (
    plot_cross_section as plot_cross_section_impl,
    plot_isotropic_3views as plot_isotropic_3views_impl,
)
from .postprocess import (
    get_centered_permittivity as get_centered_permittivity_impl,
    get_frequency_solution as get_frequency_solution_impl,
    get_material_permittivity as get_material_permittivity_impl,
    interpolate_yee_to_center as interpolate_yee_to_center_impl,
)
from .runtime import (
    initialization as runtime_initialization,
    accumulate_dft as accumulate_dft_impl,
    accumulate_dft_gpu as accumulate_dft_gpu_impl,
    build_dft_step_tables as build_dft_step_tables_impl,
    advance_magnetic_component_dispersive_state as advance_magnetic_component_dispersive_state_impl,
    advance_magnetic_dispersive_state as advance_magnetic_dispersive_state_impl,
    advance_component_dispersive_state as advance_component_dispersive_state_impl,
    advance_dispersive_state as advance_dispersive_state_impl,
    advance_phase as advance_phase_impl,
    apply_magnetic_component_dispersive_currents as apply_magnetic_component_dispersive_currents_impl,
    apply_magnetic_dispersive_corrections as apply_magnetic_dispersive_corrections_impl,
    apply_component_dispersive_currents as apply_component_dispersive_currents_impl,
    apply_dispersive_corrections as apply_dispersive_corrections_impl,
    advance_gyromagnetic_state as advance_gyromagnetic_state_impl,
    apply_gyromagnetic_correction as apply_gyromagnetic_correction_impl,
    initialize_gyromagnetic_state as initialize_gyromagnetic_state_impl,
    average_node_to_component as average_node_to_component_impl,
    average_node_to_magnetic_component as average_node_to_magnetic_component_impl,
    build_dispersive_templates as build_dispersive_templates_impl,
    build_magnetic_dispersive_templates as build_magnetic_dispersive_templates_impl,
    build_materials as build_materials_impl,
    build_update_coefficients as build_update_coefficients_impl,
    clamp_field_face as clamp_field_face_impl,
    clamp_pec_boundaries as clamp_pec_boundaries_impl,
    compute_spectral_start_step as compute_spectral_start_step_impl,
    compute_window_weight as compute_window_weight_impl,
    cpml_layout_params as cpml_layout_params_impl,
    enable_dft as enable_dft_impl,
    enforce_pec_boundaries as enforce_pec_boundaries_impl,
    get_fdtd_module,
    init_field as init_field_impl,
    initialize_dispersive_state as initialize_dispersive_state_impl,
    initialize_magnetic_dispersive_state as initialize_magnetic_dispersive_state_impl,
    initialize_solver,
    iter_cpml_memory_regions as iter_cpml_memory_regions_impl,
    normalize_target_frequencies as normalize_target_frequencies_impl,
    reset_dft_runtime_state as reset_dft_runtime_state_impl,
    require_cuda_scene,
    resolve_spectral_window_type as resolve_spectral_window_type_impl,
    solve as solve_impl,
    source_time_kind as source_time_kind_impl,
    sync_dft_entries_from_runtime_state as sync_dft_entries_from_runtime_state_impl,
    sync_dft_legacy_state as sync_dft_legacy_state_impl,
    sync_dft_primary_runtime_state as sync_dft_primary_runtime_state_impl,
    sync_observer_legacy_state as sync_observer_legacy_state_impl,
    sync_observer_primary_state as sync_observer_primary_state_impl,
    synchronize_device as synchronize_device_impl,
    update_kerr_electric_curls as update_kerr_electric_curls_impl,
    update_nonlinear_electric_coefficients as update_nonlinear_electric_coefficients_impl,
    update_electric_fields as update_electric_fields_impl,
    update_electric_fields_bloch as update_electric_fields_bloch_impl,
    update_electric_fields_cpml as update_electric_fields_cpml_impl,
    update_electric_fields_cpml_compressed as update_electric_fields_cpml_compressed_impl,
    update_electric_fields_cpml_dense as update_electric_fields_cpml_dense_impl,
    update_electric_fields_standard as update_electric_fields_standard_impl,
    update_magnetic_fields as update_magnetic_fields_impl,
    update_magnetic_fields_cpml as update_magnetic_fields_cpml_impl,
    update_magnetic_fields_cpml_compressed as update_magnetic_fields_cpml_compressed_impl,
    update_magnetic_fields_cpml_dense as update_magnetic_fields_cpml_dense_impl,
    update_magnetic_fields_standard as update_magnetic_fields_standard_impl,
)

_get_fdtd_module = get_fdtd_module
_require_cuda_scene = require_cuda_scene


def calculate_required_steps(
    frequency,
    dt,
    c=299792458.0,
    num_cycles=20,
    transient_cycles=15,
    domain_size=None,
    source_time=None,
):
    period = 1.0 / frequency
    transient_time = max(transient_cycles, 15) * period
    if domain_size is not None:
        propagation_time = 2 * domain_size / c
        transient_time = max(transient_time, propagation_time * 5)
    if source_time is not None:
        settling_time = (
            float(source_time.get("settling_time", 0.0))
            if isinstance(source_time, dict)
            else float(getattr(source_time, "settling_time", 0.0))
        )
        transient_time = max(transient_time, settling_time)
    total_time = transient_time + num_cycles * period
    return int(total_time / dt)


class FDTD:
    def __init__(self, scene, frequency=1e9, absorber_type="cpml", cpml_config=None):
        runtime_initialization.get_fdtd_module = _get_fdtd_module
        runtime_initialization.require_cuda_scene = _require_cuda_scene
        initialize_solver(
            self,
            prepare_scene(scene),
            frequency=frequency,
            absorber_type=absorber_type,
            cpml_config=cpml_config,
        )

    def auto_dt(self, dx, dy, dz, frequency, source_time=None, c=299792458.0, steps_per_cycle=30):
        # dx/dy/dz are the per-axis MINIMUM primal spacings (identical to the
        # uniform steps on a uniform grid), so the Courant bound holds on
        # nonuniform grids as well.
        if source_time is not None:
            characteristic_frequency = (
                float(source_time.get("characteristic_frequency", frequency))
                if isinstance(source_time, dict)
                else float(getattr(source_time, "characteristic_frequency", frequency))
            )
            frequency = max(float(frequency), characteristic_frequency)
        period = 1 / frequency
        dt1 = period / steps_per_cycle
        dt2 = 1 / (c * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))
        return min(dt1, dt2)

    def _normalize_target_frequencies(self, frequencies) -> tuple[float, ...]:
        return normalize_target_frequencies_impl(self, frequencies)

    def _reset_dft_runtime_state(self):
        reset_dft_runtime_state_impl(self)

    def _sync_dft_entries_from_runtime_state(self):
        sync_dft_entries_from_runtime_state_impl(self)

    def _sync_dft_primary_runtime_state(self):
        sync_dft_primary_runtime_state_impl(self)

    def _sync_dft_legacy_state(self):
        sync_dft_legacy_state_impl(self)

    def _sync_dft_primary_state(self):
        sync_dft_primary_runtime_state_impl(self)

    def _sync_observer_legacy_state(self):
        sync_observer_legacy_state_impl(self)

    def _sync_observer_primary_state(self):
        sync_observer_primary_state_impl(self)

    def enable_dft(self, frequencies, window_type="hanning", end_step=None):
        enable_dft_impl(self, frequencies, window_type=window_type, end_step=end_step)

    def _source_time_kind(self):
        return source_time_kind_impl(self)

    def _resolve_spectral_window_type(self, window_type):
        return resolve_spectral_window_type_impl(self, window_type)

    def _compute_spectral_start_step(self, frequency, *, window_type=None):
        return compute_spectral_start_step_impl(self, frequency, window_type=window_type)

    def _compute_window_weight(self, n, start_step=None, end_step=None, window_type=None):
        return compute_window_weight_impl(self, n, start_step=start_step, end_step=end_step, window_type=window_type)

    def _advance_phase(self, cos_phase, sin_phase, step_cos, step_sin):
        return advance_phase_impl(self, cos_phase, sin_phase, step_cos, step_sin)

    def _synchronize_device(self):
        synchronize_device_impl(self)

    def accumulate_dft(self, n, phase_cos=None, phase_sin=None):
        accumulate_dft_impl(self, n, phase_cos=phase_cos, phase_sin=phase_sin)

    def build_dft_step_tables(self, time_steps):
        return build_dft_step_tables_impl(self, time_steps)

    def accumulate_dft_gpu(self):
        accumulate_dft_gpu_impl(self)

    def clear_observers(self):
        clear_observers_impl(self)

    def add_point_observer(self, name, position, component="Ez"):
        add_point_observer_impl(self, name, position, component=component)

    def add_plane_observer(self, name, axis="z", position=0.0, component="Ez"):
        add_plane_observer_impl(self, name, axis=axis, position=position, component=component)

    def _get_component_coords(self, component):
        return get_component_coords_impl(self, component)

    def _resolve_point_observer(self, observer):
        return resolve_point_observer_impl(self, observer)

    def _resolve_plane_observer(self, observer):
        return resolve_plane_observer_impl(self, observer)

    def _prepare_observers(self, frequencies, window_type, time_steps):
        prepare_observers_impl(self, frequencies, window_type, time_steps)

    def accumulate_observers(self, n, phase_cos=None, phase_sin=None):
        accumulate_observers_impl(self, n, phase_cos=phase_cos, phase_sin=phase_sin)

    def get_observer_results(self):
        return get_observer_results_impl(self)

    def clear_time_observers(self):
        clear_time_observers_impl(self)

    def _prepare_time_observers(self, time_steps):
        prepare_time_observers_impl(self, time_steps)

    def accumulate_time_observers(self, n):
        accumulate_time_observers_impl(self, n)

    def get_time_observer_results(self):
        return get_time_observer_results_impl(self)

    def get_frequency_solution(self, *, frequency=None, freq_index=None, all_frequencies=False):
        return get_frequency_solution_impl(
            self,
            frequency=frequency,
            freq_index=freq_index,
            all_frequencies=all_frequencies,
        )

    def save_frequency_solution(self, output_path: str):
        save_frequency_solution_impl(self, output_path)

    def build_materials(self, scene):
        build_materials_impl(self, scene)

    def _average_node_to_component(self, node_tensor, component_name):
        return average_node_to_component_impl(self, node_tensor, component_name)

    def _average_node_to_magnetic_component(self, node_tensor, component_name):
        return average_node_to_magnetic_component_impl(self, node_tensor, component_name)

    def _build_dispersive_templates(self, material_model):
        build_dispersive_templates_impl(self, material_model)

    def _build_magnetic_dispersive_templates(self, material_model):
        build_magnetic_dispersive_templates_impl(self, material_model)

    def _initialize_dispersive_state(self):
        initialize_dispersive_state_impl(self)

    def _initialize_magnetic_dispersive_state(self):
        initialize_magnetic_dispersive_state_impl(self)

    def _advance_component_dispersive_state(self, component_name, field, *, imag=False):
        advance_component_dispersive_state_impl(self, component_name, field, imag=imag)

    def _advance_dispersive_state(self):
        advance_dispersive_state_impl(self)

    def _advance_magnetic_component_dispersive_state(self, component_name, field, *, imag=False):
        advance_magnetic_component_dispersive_state_impl(self, component_name, field, imag=imag)

    def _advance_magnetic_dispersive_state(self):
        advance_magnetic_dispersive_state_impl(self)

    def _apply_component_dispersive_currents(self, component_name, field, *, imag=False):
        apply_component_dispersive_currents_impl(self, component_name, field, imag=imag)

    def _apply_dispersive_corrections(self):
        apply_dispersive_corrections_impl(self)

    def _apply_magnetic_component_dispersive_currents(self, component_name, field, *, imag=False):
        apply_magnetic_component_dispersive_currents_impl(self, component_name, field, imag=imag)

    def _apply_magnetic_dispersive_corrections(self):
        apply_magnetic_dispersive_corrections_impl(self)

    def _initialize_gyromagnetic_state(self):
        initialize_gyromagnetic_state_impl(self)

    def _advance_gyromagnetic_state(self):
        advance_gyromagnetic_state_impl(self)

    def _apply_gyromagnetic_correction(self):
        apply_gyromagnetic_correction_impl(self)

    def _build_update_coefficients(self):
        build_update_coefficients_impl(self)

    def _update_kerr_electric_curls(self):
        update_kerr_electric_curls_impl(self)

    def _update_nonlinear_electric_coefficients(self):
        update_nonlinear_electric_coefficients_impl(self)

    def _get_material_permittivity(self):
        return get_material_permittivity_impl(self)

    def _iter_source_images(self, source_position, cutoff):
        src_x, src_y, src_z = source_position
        boundary = self.scene.boundary
        if not (boundary.uses_kind("periodic") or boundary.uses_kind("bloch")):
            yield (src_x, src_y, src_z), 1.0, 0.0
            return

        lengths = (
            self.scene.domain_range[1] - self.scene.domain_range[0],
            self.scene.domain_range[3] - self.scene.domain_range[2],
            self.scene.domain_range[5] - self.scene.domain_range[4],
        )
        axis_kinds = tuple(boundary.axis_kind(axis) for axis in ("x", "y", "z"))
        spans = [
            max(0, int(np.ceil(cutoff / length))) if axis_kind in {"periodic", "bloch"} else 0
            for length, axis_kind in zip(lengths, axis_kinds)
        ]
        wavevector = self.scene.bloch_wavevector

        for sx in range(-spans[0], spans[0] + 1):
            for sy in range(-spans[1], spans[1] + 1):
                for sz in range(-spans[2], spans[2] + 1):
                    phase_real = 1.0
                    phase_imag = 0.0
                    if boundary.uses_kind("bloch"):
                        phase_argument = 0.0
                        for axis_kind, shift, wave_number, length in zip(axis_kinds, (sx, sy, sz), wavevector, lengths):
                            if axis_kind == "bloch":
                                phase_argument += wave_number * shift * length
                        phase = np.exp(1j * phase_argument)
                        phase_real = float(phase.real)
                        phase_imag = float(phase.imag)
                    yield (
                        src_x + sx * lengths[0],
                        src_y + sy * lengths[1],
                        src_z + sz * lengths[2],
                    ), phase_real, phase_imag

    def _component_plane_spec(self, field_name, axis, plane_index):
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        field = getattr(self, field_name)
        sizes = tuple(int(dim) for dim in field.shape)
        offsets = [0, 0, 0]
        offsets[axis_index] = min(max(int(plane_index), 0), sizes[axis_index] - 1)
        shape = list(sizes)
        shape[axis_index] = 1
        return tuple(offsets), tuple(shape)

    _COMPONENT_HALF_OFFSET_AXES = {
        "Ex": (True, False, False),
        "Ey": (False, True, False),
        "Ez": (False, False, True),
        "Hx": (False, True, True),
        "Hy": (True, False, True),
        "Hz": (True, True, False),
    }

    def _component_positions(self, field_name, offsets, shape, dtype):
        half_axes = self._COMPONENT_HALF_OFFSET_AXES.get(field_name)
        if half_axes is None:
            raise ValueError(f"Unsupported field component {field_name!r}.")
        scene = self.scene
        axis_coords = (
            (scene.x, scene.x_half),
            (scene.y, scene.y_half),
            (scene.z, scene.z_half),
        )
        px, py, pz = (
            coords[half][offset : offset + size].to(device=self.device, dtype=dtype)
            for coords, half, offset, size in zip(axis_coords, half_axes, offsets, shape)
        )

        px = px[:, None, None].expand(shape[0], shape[1], shape[2])
        py = py[None, :, None].expand(shape[0], shape[1], shape[2])
        pz = pz[None, None, :].expand(shape[0], shape[1], shape[2])
        return torch.stack((px, py, pz), dim=-1)

    def _plane_coordinate(self, axis, plane_index):
        axis_values = {"x": self.scene.x, "y": self.scene.y, "z": self.scene.z}[axis]
        plane_index = min(max(int(plane_index), 0), len(axis_values) - 1)
        return float(axis_values[plane_index].item())

    def _iter_cpml_memory_regions(self, attr_name):
        yield from iter_cpml_memory_regions_impl(self, attr_name)

    def _cpml_layout_params(self, attr_name):
        return cpml_layout_params_impl(self, attr_name)

    def _update_magnetic_fields_cpml_dense(self, hx, hy, hz, ex, ey, ez):
        update_magnetic_fields_cpml_dense_impl(self, hx, hy, hz, ex, ey, ez)

    def _update_magnetic_fields_cpml_compressed(self, hx, hy, hz, ex, ey, ez):
        update_magnetic_fields_cpml_compressed_impl(self, hx, hy, hz, ex, ey, ez)

    def _update_magnetic_fields_cpml(self, hx, hy, hz, ex, ey, ez):
        update_magnetic_fields_cpml_impl(self, hx, hy, hz, ex, ey, ez)

    def _update_magnetic_fields_standard(self, hx, hy, hz, ex, ey, ez):
        update_magnetic_fields_standard_impl(self, hx, hy, hz, ex, ey, ez)

    def _update_magnetic_fields(self, hx, hy, hz, ex, ey, ez):
        update_magnetic_fields_impl(self, hx, hy, hz, ex, ey, ez)

    def _update_electric_fields_cpml_dense(self, ex, ey, ez, hx, hy, hz):
        update_electric_fields_cpml_dense_impl(self, ex, ey, ez, hx, hy, hz)

    def _update_electric_fields_cpml_compressed(self, ex, ey, ez, hx, hy, hz):
        update_electric_fields_cpml_compressed_impl(self, ex, ey, ez, hx, hy, hz)

    def _update_electric_fields_cpml(self, ex, ey, ez, hx, hy, hz):
        update_electric_fields_cpml_impl(self, ex, ey, ez, hx, hy, hz)

    def _update_electric_fields_standard(self, ex, ey, ez, hx, hy, hz):
        update_electric_fields_standard_impl(self, ex, ey, ez, hx, hy, hz)

    def _update_electric_fields(self, ex, ey, ez, hx, hy, hz, *, time_value=None):
        update_electric_fields_impl(self, ex, ey, ez, hx, hy, hz, time_value=time_value)

    def _update_electric_fields_bloch(self):
        update_electric_fields_bloch_impl(self)

    def _clamp_field_face(self, field, axis, side):
        clamp_field_face_impl(self, field, axis, side)

    def _enforce_pec_boundaries(self):
        enforce_pec_boundaries_impl(self)

    def _clamp_pec_boundaries(self):
        clamp_pec_boundaries_impl(self)

    def init_field(self):
        init_field_impl(self)

    def add_source(self, n=None, signal=None, time_value=None):
        inject_source_terms(self, n=n, signal=signal, time_value=time_value)

    def solve(
        self,
        time_steps: int,
        dft_frequency: float = None,
        enable_plot: bool = False,
        dft_window: str = "hanning",
        full_field_dft: bool = True,
        normalize_source: bool = False,
        shutoff: float = 0.0,
        shutoff_check_interval: int = 100,
        use_cuda_graph: bool = False,
        resume_from=None,
        stop_step: int | None = None,
    ):
        return solve_impl(
            self,
            time_steps,
            dft_frequency=dft_frequency,
            enable_plot=enable_plot,
            dft_window=dft_window,
            full_field_dft=full_field_dft,
            normalize_source=normalize_source,
            shutoff=shutoff,
            shutoff_check_interval=shutoff_check_interval,
            use_cuda_graph=use_cuda_graph,
            resume_from=resume_from,
            stop_step=stop_step,
        )

    def _interpolate_yee_to_center(self, freq_solution):
        return interpolate_yee_to_center_impl(self, freq_solution)

    def _get_centered_permittivity(self):
        return get_centered_permittivity_impl(self)

    def plot_cross_section(self, axis="z", position=0.0, component="abs", field_log_scale=False,
                           figsize=(12, 5), save_path=None, verbose=True):
        plot_cross_section_impl(
            self,
            axis=axis,
            position=position,
            component=component,
            field_log_scale=field_log_scale,
            figsize=figsize,
            save_path=save_path,
            verbose=verbose,
        )

    def plot_isotropic_3views(self, position=0.0, field_log_scale=True, figsize=(18, 5),
                              save_path=None, vmin_db=-60, verbose=True):
        plot_isotropic_3views_impl(
            self,
            position=position,
            field_log_scale=field_log_scale,
            figsize=figsize,
            save_path=save_path,
            vmin_db=vmin_db,
            verbose=verbose,
        )
