from __future__ import annotations

import os

import numpy as np
import torch

from ...compiler.monitors import compile_fdtd_observers, compile_fdtd_time_observers
from ...compiler.sources import compile_fdtd_sources
from ..boundary import DEFAULT_CPML_CONFIG
from .module_cache import get_fdtd_module, require_cuda_scene


def _pole_parameter_bound(value) -> float:
    """Scalar upper bound of a pole parameter (per-cell custom poles carry tensors)."""
    if torch.is_tensor(value):
        return float(value.detach().max())
    return float(value)


def _material_characteristic_frequency(material) -> float:
    if material is None:
        return 0.0

    characteristic = 0.0
    for pole in getattr(material, "debye_poles", ()):
        characteristic = max(characteristic, 1.0 / (2.0 * np.pi * float(pole.tau)))
    for pole in getattr(material, "drude_poles", ()):
        characteristic = max(
            characteristic,
            _pole_parameter_bound(pole.plasma_frequency),
            float(pole.gamma),
        )
    for pole in getattr(material, "lorentz_poles", ()):
        characteristic = max(
            characteristic,
            float(pole.resonance_frequency),
            float(pole.gamma),
        )
    for pole in getattr(material, "mu_debye_poles", ()):
        characteristic = max(characteristic, 1.0 / (2.0 * np.pi * float(pole.tau)))
    for pole in getattr(material, "mu_drude_poles", ()):
        characteristic = max(
            characteristic,
            _pole_parameter_bound(pole.plasma_frequency),
            float(pole.gamma),
        )
    for pole in getattr(material, "mu_lorentz_poles", ()):
        characteristic = max(
            characteristic,
            float(pole.resonance_frequency),
            float(pole.gamma),
        )
    # Sheet media (e.g. Graphene) expose their relaxation rate directly.
    characteristic = max(
        characteristic, float(getattr(material, "characteristic_frequency", 0.0) or 0.0)
    )
    return characteristic


def _scene_material_characteristic_frequency(scene) -> float:
    characteristic = 0.0
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        characteristic = max(characteristic, _material_characteristic_frequency(material))
    return characteristic


def initialize_solver(solver, scene, frequency=1e9, absorber_type="cpml", cpml_config=None):
    require_cuda_scene(scene)
    solver.scene = scene
    solver.Nx = scene.Nx
    solver.Ny = scene.Ny
    solver.Nz = scene.Nz
    solver.min_dx = float(scene.dx_primal64.min())
    solver.min_dy = float(scene.dy_primal64.min())
    solver.min_dz = float(scene.dz_primal64.min())
    solver.device = torch.device(scene.device)
    solver.verbose = bool(scene.verbose)
    solver.c = 299792458.0
    solver.mu0 = 4 * np.pi * 1e-7
    solver.eps0 = 1 / (solver.mu0 * solver.c**2)
    solver.plot_interval = 50
    solver.progress_update_interval = 25
    compiled_sources = compile_fdtd_sources(scene, default_frequency=float(frequency))
    solver._compiled_sources = tuple(compiled_sources)
    primary_tfsf_source = next(
        (source for source in solver._compiled_sources if source.get("injection", {}).get("kind") == "tfsf"),
        None,
    )
    solver._compiled_source = primary_tfsf_source or (
        solver._compiled_sources[0] if solver._compiled_sources else None
    )
    solver._source_time = None if solver._compiled_source is None else solver._compiled_source["source_time"]
    time_step_frequency = (
        max(
            float(source["source_time"].get("characteristic_frequency", source["source_time"]["frequency"]))
            for source in solver._compiled_sources
        )
        if solver._compiled_sources
        else float(frequency)
    )
    time_step_frequency = max(time_step_frequency, _scene_material_characteristic_frequency(scene))
    solver.source_frequency = (
        float(solver._source_time["frequency"])
        if solver._source_time is not None
        else float(frequency)
    )
    solver.source_omega = 2 * np.pi * solver.source_frequency
    solver.absorber_type = absorber_type.lower()
    solver.boundary_kind = scene.boundary.kind
    solver.active_absorber_type = "none"
    solver.complex_fields_enabled = False
    solver.cpml_config = dict(DEFAULT_CPML_CONFIG)
    if cpml_config is not None:
        solver.cpml_config.update(cpml_config)
    solver._cpml_memory_mode_requested = str(solver.cpml_config.get("memory_mode", "auto")).strip().lower()
    solver._cpml_memory_mode = "none"
    solver._cpml_memory_layouts = {}
    solver._cpml_layout_cache = {}
    solver._cpml_dense_memory_bytes = 0
    solver._cpml_slab_memory_bytes = 0
    solver._cpml_allocated_memory_bytes = 0
    solver._cpml_dense_memory_limit_bytes = None
    solver._cpml_auto_free_bytes = None

    solver.dt = solver.auto_dt(solver.min_dx, solver.min_dy, solver.min_dz, time_step_frequency)
    # Per-axis spacing arrays: `_e` scales backward diffs in E updates (dual
    # spacing, length N); `_h` scales forward diffs in H updates (primal
    # spacing, length N-1). Reciprocals are computed in float64 then cast.
    solver.inv_dx_e = torch.tensor(1.0 / scene.dx_dual64, dtype=torch.float32, device=solver.device)
    solver.inv_dy_e = torch.tensor(1.0 / scene.dy_dual64, dtype=torch.float32, device=solver.device)
    solver.inv_dz_e = torch.tensor(1.0 / scene.dz_dual64, dtype=torch.float32, device=solver.device)
    solver.inv_dx_h = torch.tensor(1.0 / scene.dx_primal64, dtype=torch.float32, device=solver.device)
    solver.inv_dy_h = torch.tensor(1.0 / scene.dy_primal64, dtype=torch.float32, device=solver.device)
    solver.inv_dz_h = torch.tensor(1.0 / scene.dz_primal64, dtype=torch.float32, device=solver.device)
    solver.fdtd_module = get_fdtd_module()

    solver.dft_enabled = False
    solver.dft_frequency = None
    solver.dft_frequencies = ()
    solver.dft_Ex_real = None
    solver.dft_Ex_imag = None
    solver.dft_Ey_real = None
    solver.dft_Ey_imag = None
    solver.dft_Ez_real = None
    solver.dft_Ez_imag = None
    solver.dft_Ex_aux_real = None
    solver.dft_Ex_aux_imag = None
    solver.dft_Ey_aux_real = None
    solver.dft_Ey_aux_imag = None
    solver.dft_Ez_aux_real = None
    solver.dft_Ez_aux_imag = None
    solver.dft_sample_count = 0
    solver.dft_sample_counts = ()
    solver.dft_start_step = None
    solver.dft_start_steps = ()
    solver.dft_end_step = None
    solver.dft_end_steps = ()
    solver.dft_window_normalization = 0.0
    solver.dft_window_normalizations = ()
    solver.dft_phase_step_cos = None
    solver.dft_phase_step_sin = None
    solver._dft_entries = []
    solver._dft_batched_fields = {}
    solver._dft_phase_cos = None
    solver._dft_phase_sin = None
    solver._dft_phase_step_cos_values = None
    solver._dft_phase_step_sin_values = None
    solver._dft_start_steps = None
    solver._dft_end_steps = None
    solver._dft_window_normalization_values = None
    solver._dft_sample_count_values = None
    solver._dft_source_dft_real_values = None
    solver._dft_source_dft_imag_values = None

    solver._source_terms = None
    solver._magnetic_source_terms = None
    solver._electric_source_terms = None
    solver.material_eps_r = None
    solver.material_mu_r = None
    solver._compiled_material_model = None
    solver.dispersive_enabled = False
    solver.electric_dispersive_enabled = False
    solver.magnetic_dispersive_enabled = False
    solver._dispersive_templates = {}
    solver._magnetic_dispersive_templates = {}
    solver.kerr_enabled = False
    solver.chi2_enabled = False
    solver.tpa_enabled = False
    solver.nonlinear_enabled = False
    solver.nonlinear_general_enabled = False
    solver.kerr_chi3 = None
    solver.kerr_chi3_Ex = None
    solver.kerr_chi3_Ey = None
    solver.kerr_chi3_Ez = None
    solver.nonlinear_chi2_Ex = None
    solver.nonlinear_chi2_Ey = None
    solver.nonlinear_chi2_Ez = None
    solver.tpa_sigma_Ex = None
    solver.tpa_sigma_Ey = None
    solver.tpa_sigma_Ez = None
    solver.cex_curl_dynamic = None
    solver.cey_curl_dynamic = None
    solver.cez_curl_dynamic = None
    solver.cex_decay_dynamic = None
    solver.cey_decay_dynamic = None
    solver.cez_decay_dynamic = None
    solver.cex_decay_external = None
    solver.cey_decay_external = None
    solver.cez_decay_external = None
    solver._point_observer_groups = {}
    solver._plane_observer_groups = {}
    solver.last_solve_elapsed_s = None
    solver.tfsf_enabled = False
    solver._tfsf_state = None

    solver.observers = compile_fdtd_observers(scene)
    solver.observers_enabled = False
    solver.time_observers = compile_fdtd_time_observers(scene)
    solver.time_observers_enabled = False
    solver.observer_frequency = None
    solver.observer_frequencies = ()
    solver.observer_window_type = "hanning"
    solver.observer_start_step = None
    solver.observer_start_steps = ()
    solver.observer_end_step = None
    solver.observer_end_steps = ()
    solver.observer_window_normalization = 0.0
    solver.observer_sample_count = 0
    solver.observer_sample_counts = ()
    solver.observer_phase_step_cos = None
    solver.observer_phase_step_sin = None
    solver._observer_spectral_entries = []
    solver._observer_point_groups_by_frequency = []
    solver._observer_plane_groups_by_frequency = []
