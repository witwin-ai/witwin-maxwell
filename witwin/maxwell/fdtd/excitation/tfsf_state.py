from __future__ import annotations

import numpy as np
import torch

from ..boundary.bloch import validate_grating_tfsf_slab_topology
from .spatial import AuxiliaryGrid1D
from .tfsf_common import (
    box_center_tensor,
    build_batched_aux_terms,
    build_batched_reference_terms,
    build_term_from_profile,
    build_terms_from_specs,
    entry_projection,
    incident_profile,
    make_aux_term,
    nearest_index,
    project_positions,
    projection_extrema,
    resolve_bounds_indices,
    solve_auxiliary_step,
    solve_numerical_wavenumber,
    validate_background_is_vacuum,
    validate_bounds,
)
from .tfsf_specs import (
    DELTA_ATTR,
    E_CURL_ATTR,
    H_CURL_ATTR,
    axis_aligned_direction,
    axis_aligned_sample_indices,
    axis_aligned_sample_view,
    build_discrete_tfsf_specs,
    constant_line_index_tensor,
    discrete_plane_wave_vectors,
    is_reference_plane_wave_x_ez,
    line_index_tensor,
    magnetic_physical_vector,
    magnetic_unit_vector,
    make_reference_term,
)


ABSORBER_CELLS = 20


def _domain_bounds(solver):
    return (
        (solver.scene.domain_range[0], solver.scene.domain_range[1]),
        (solver.scene.domain_range[2], solver.scene.domain_range[3]),
        (solver.scene.domain_range[4], solver.scene.domain_range[5]),
    )


def _make_auxiliary_grid(solver, *, s_min, s_max, ds):
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    return AuxiliaryGrid1D(
        s_min=s_min,
        s_max=s_max + ABSORBER_CELLS * ds,
        ds=ds,
        dt=solver.dt,
        wave_speed=solver.c,
        impedance=eta0,
        source_time=solver._source_time,
        device=solver.device,
        dtype=solver.Ex.dtype,
        absorber_cells=ABSORBER_CELLS,
        fdtd_module=solver.fdtd_module,
        kernel_block_size=solver.kernel_block_size,
    )


def _make_directional_auxiliary_grid(solver, direction, ds):
    s_min_domain, s_max_domain = projection_extrema(_domain_bounds(solver), direction)
    return _make_auxiliary_grid(solver, s_min=s_min_domain, s_max=s_max_domain, ds=ds)


def _set_tfsf_state(
    solver,
    *,
    provider,
    source,
    lower,
    upper,
    electric_terms,
    magnetic_terms,
    auxiliary_grid=None,
    phase_speed=None,
):
    state = {
        "provider": provider,
        "lower": lower,
        "upper": upper,
        "bounds": source["injection"]["bounds"],
        "electric_terms": electric_terms,
        "magnetic_terms": magnetic_terms,
    }
    if provider in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}:
        state["electric_batch"] = build_batched_reference_terms(solver, electric_terms)
        state["magnetic_batch"] = build_batched_reference_terms(solver, magnetic_terms)
    elif provider == "plane_wave_aux":
        state["electric_batch"] = build_batched_aux_terms(solver, electric_terms)
        state["magnetic_batch"] = build_batched_aux_terms(solver, magnetic_terms)
    if auxiliary_grid is not None:
        state["auxiliary_grid"] = auxiliary_grid
    if phase_speed is not None:
        state["phase_speed"] = phase_speed
    solver._tfsf_state = state
    solver.tfsf_enabled = True


def _set_pending_tfsf_slab_state(solver, source):
    injection = source["injection"]
    axis = injection["axis"]
    axis_index = "xyz".index(axis)
    shape = (solver.Nx, solver.Ny, solver.Nz)
    lower = []
    upper = []
    for current_axis, axis_coords, size in zip("xyz", (solver.scene.x, solver.scene.y, solver.scene.z), shape):
        if current_axis == axis:
            axis_bounds = injection["axis_bounds"]
            lower.append(nearest_index(axis_coords, axis_bounds[0]))
            upper.append(nearest_index(axis_coords, axis_bounds[1]))
        else:
            lower.append(int(solver.scene.pml_thickness_for_face(current_axis, "low")))
            upper.append(int(size - solver.scene.pml_thickness_for_face(current_axis, "high") - 1))

    if upper[axis_index] <= lower[axis_index] + 1:
        raise ValueError(f"TFSF slab bounds must span at least two cells along {axis}.")

    solver._tfsf_state = {
        "provider": "pending_grating_slab",
        "runtime_pending": True,
        "mode": "slab",
        "axis": axis,
        "bounds": None,
        "lower": tuple(lower),
        "upper": tuple(upper),
        "electric_terms": [],
        "magnetic_terms": [],
        "source": source,
    }
    solver.tfsf_enabled = False


def _spec_positions(solver, spec):
    return solver._component_positions(
        spec["incident_name"],
        spec["sample_offsets"],
        spec["sample_shape"],
        dtype=solver.Ex.dtype,
    )


def _initialize_axis_aligned_plane_wave_auxiliary_state(solver, source, lower, upper, axis: str, direction_sign: int):
    electric_vector = {
        "Ex": source["polarization"][0],
        "Ey": source["polarization"][1],
        "Ez": source["polarization"][2],
    }
    magnetic_vector = magnetic_unit_vector(source["direction"], source["polarization"])
    aux = _make_directional_auxiliary_grid(solver, source["direction"], float(getattr(solver, DELTA_ATTR[axis])))

    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)

    def build_axis_term(spec, coeff_patch, component_scale):
        sample_indices = axis_aligned_sample_indices(
            solver,
            axis,
            direction_sign,
            spec["sample_kind"],
            spec["sample_offsets"],
            spec["sample_shape"],
        )
        term = make_reference_term(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            coeff_patch=spec["sign"] * coeff_patch,
            sample_kind=spec["sample_kind"],
            sample_indices=sample_indices,
            component_scale=component_scale,
        )
        term["sample_view"] = axis_aligned_sample_view(axis, sample_indices.numel())
        return term

    electric_terms = build_terms_from_specs(
        solver,
        electric_specs,
        magnetic_vector,
        E_CURL_ATTR,
        build_axis_term,
    )
    magnetic_terms = build_terms_from_specs(
        solver,
        magnetic_specs,
        electric_vector,
        H_CURL_ATTR,
        build_axis_term,
    )

    _set_tfsf_state(
        solver,
        provider="plane_wave_axis_aligned",
        source=source,
        lower=lower,
        upper=upper,
        auxiliary_grid=aux,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def _initialize_plane_wave_discrete_cw_state(solver, source, lower, upper):
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    k_numeric = solve_numerical_wavenumber(solver, source["direction"], DELTA_ATTR)
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    electric_vector, magnetic_unit_vector_map = discrete_plane_wave_vectors(
        solver,
        source["direction"],
        source["polarization"],
        k_numeric,
    )
    magnetic_vector = {name: value / eta0 for name, value in magnetic_unit_vector_map.items()}
    box_center = box_center_tensor(solver, source["injection"]["bounds"])

    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)

    def build_discrete_cw_term(spec, coeff_patch, component_scale):
        _, delay_patch = incident_profile(
            solver,
            {"kind": "plane_wave", "direction": source["direction"]},
            _spec_positions(solver, spec),
            box_center,
            phase_speed=phase_speed,
            source_frequency=source_frequency,
        )
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=spec["sign"] * component_scale * coeff_patch,
            delay_patch=delay_patch,
            activation_delay_patch=None,
            source_time=source_time,
            omega=source_omega,
        )

    electric_terms = build_terms_from_specs(
        solver,
        electric_specs,
        magnetic_vector,
        E_CURL_ATTR,
        build_discrete_cw_term,
    )
    magnetic_terms = build_terms_from_specs(
        solver,
        magnetic_specs,
        electric_vector,
        H_CURL_ATTR,
        build_discrete_cw_term,
    )

    _set_tfsf_state(
        solver,
        provider="plane_wave_discrete_cw",
        source=source,
        lower=lower,
        upper=upper,
        phase_speed=phase_speed,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def _initialize_reference_plane_wave_auxiliary_state(solver, source, lower, upper):
    ix0, iy0, iz0 = lower
    ix1, iy1, iz1 = upper

    electric_scale = float(source["polarization"][2])
    aux = _make_auxiliary_grid(
        solver,
        s_min=float(solver.scene.domain_range[0]),
        s_max=float(solver.scene.domain_range[1]),
        ds=float(solver.dx),
    )

    magnetic_terms = [
        make_reference_term(
            solver,
            field_name="Hy",
            offsets=(ix0 - 1, iy0, iz0),
            coeff_patch=solver.chy_curl[ix0 - 1 : ix0, iy0 : iy1 + 1, iz0:iz1] / float(solver.dx),
            sample_kind="electric",
            sample_indices=constant_line_index_tensor(solver, ix0),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hy",
            offsets=(ix1, iy0, iz0),
            coeff_patch=-solver.chy_curl[ix1 : ix1 + 1, iy0 : iy1 + 1, iz0:iz1] / float(solver.dx),
            sample_kind="electric",
            sample_indices=constant_line_index_tensor(solver, ix1),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hx",
            offsets=(ix0, iy0 - 1, iz0),
            coeff_patch=-solver.chx_curl[ix0 : ix1 + 1, iy0 - 1 : iy0, iz0:iz1] / float(solver.dy),
            sample_kind="electric",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0 + 1),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hx",
            offsets=(ix0, iy1, iz0),
            coeff_patch=solver.chx_curl[ix0 : ix1 + 1, iy1 : iy1 + 1, iz0:iz1] / float(solver.dy),
            sample_kind="electric",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0 + 1),
            component_scale=+electric_scale,
        ),
    ]

    electric_terms = [
        make_reference_term(
            solver,
            field_name="Ez",
            offsets=(ix0, iy0, iz0),
            coeff_patch=-solver.cez_curl[ix0 : ix0 + 1, iy0 : iy1 + 1, iz0:iz1] / float(solver.dx),
            sample_kind="magnetic",
            sample_indices=constant_line_index_tensor(solver, ix0 - 1),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ez",
            offsets=(ix1, iy0, iz0),
            coeff_patch=solver.cez_curl[ix1 : ix1 + 1, iy0 : iy1 + 1, iz0:iz1] / float(solver.dx),
            sample_kind="magnetic",
            sample_indices=constant_line_index_tensor(solver, ix1),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ex",
            offsets=(ix0, iy0, iz0),
            coeff_patch=solver.cex_curl[ix0:ix1, iy0 : iy1 + 1, iz0 : iz0 + 1] / float(solver.dz),
            sample_kind="magnetic",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ex",
            offsets=(ix0, iy0, iz1),
            coeff_patch=-solver.cex_curl[ix0:ix1, iy0 : iy1 + 1, iz1 : iz1 + 1] / float(solver.dz),
            sample_kind="magnetic",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0),
            component_scale=1.0,
        ),
    ]

    _set_tfsf_state(
        solver,
        provider="plane_wave_ref_x_ez",
        source=source,
        lower=lower,
        upper=upper,
        auxiliary_grid=aux,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def _initialize_plane_wave_auxiliary_state(solver, source, lower, upper):
    if is_reference_plane_wave_x_ez(source):
        _initialize_reference_plane_wave_auxiliary_state(solver, source, lower, upper)
        return

    axis_direction = axis_aligned_direction(source["direction"])
    if axis_direction is not None:
        _initialize_axis_aligned_plane_wave_auxiliary_state(
            solver,
            source,
            lower,
            upper,
            axis_direction[0],
            axis_direction[1],
        )
        return

    if source["source_time"]["kind"] == "cw":
        _initialize_plane_wave_discrete_cw_state(solver, source, lower, upper)
        return

    k_numeric = solve_numerical_wavenumber(solver, source["direction"], DELTA_ATTR)
    source_omega = 2.0 * np.pi * float(source["source_time"]["frequency"])
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    electric_vector, magnetic_vector = discrete_plane_wave_vectors(
        solver,
        source["direction"],
        source["polarization"],
        k_numeric,
    )
    ds = solve_auxiliary_step(solver, k_numeric)
    if ds <= 0.0:
        raise ValueError("Failed to construct a stable auxiliary TFSF line: ds must be > 0.")

    aux = _make_directional_auxiliary_grid(solver, source["direction"], ds)
    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)

    def build_aux_term(spec, coeff_patch, component_scale):
        sample_positions = project_positions(_spec_positions(solver, spec), source["direction"])
        return make_aux_term(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            coeff_patch=spec["sign"] * coeff_patch,
            sample_positions=sample_positions,
            component_scale=component_scale,
        )

    electric_terms = build_terms_from_specs(
        solver,
        electric_specs,
        magnetic_vector,
        E_CURL_ATTR,
        build_aux_term,
    )
    magnetic_terms = build_terms_from_specs(
        solver,
        magnetic_specs,
        electric_vector,
        H_CURL_ATTR,
        build_aux_term,
    )

    _set_tfsf_state(
        solver,
        provider="plane_wave_aux",
        source=source,
        lower=lower,
        upper=upper,
        auxiliary_grid=aux,
        phase_speed=phase_speed,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def _initialize_analytic_tfsf_state(solver, source, lower, upper):
    box_center = box_center_tensor(solver, source["injection"]["bounds"])
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    k_numeric = solve_numerical_wavenumber(solver, source["direction"], DELTA_ATTR)
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    entry_s = entry_projection(source["injection"]["bounds"], source["direction"])

    if source["kind"] == "gaussian_beam":
        electric_vector, magnetic_unit_vector_map = discrete_plane_wave_vectors(
            solver,
            source["direction"],
            source["polarization"],
            k_numeric,
        )
        profile_polarization = (
            electric_vector["Ex"],
            electric_vector["Ey"],
            electric_vector["Ez"],
        )
        magnetic_vector = {name: value / eta0 for name, value in magnetic_unit_vector_map.items()}
    else:
        electric_vector = {
            "Ex": source["polarization"][0],
            "Ey": source["polarization"][1],
            "Ez": source["polarization"][2],
        }
        profile_polarization = None
        magnetic_vector = magnetic_physical_vector(source["direction"], source["polarization"], eta0)

    electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)

    def build_profile_term(spec, coeff_patch, component_scale):
        positions = _spec_positions(solver, spec)
        spatial_amplitude, delay_patch = incident_profile(
            solver,
            source,
            positions,
            box_center,
            phase_speed=phase_speed,
            source_frequency=source_frequency,
            polarization_override=profile_polarization,
        )
        activation_delay_patch = torch.clamp(
            project_positions(positions, source["direction"]) - entry_s,
            min=0.0,
        ) / float(phase_speed)
        if source["kind"] == "plane_wave":
            delay_patch = activation_delay_patch
        scale = spec["sign"] * component_scale * spatial_amplitude * coeff_patch
        return build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=scale,
            delay_patch=delay_patch,
            activation_delay_patch=activation_delay_patch,
            source_time=source_time,
            omega=source_omega,
        )

    electric_terms = build_terms_from_specs(
        solver,
        electric_specs,
        magnetic_vector,
        E_CURL_ATTR,
        build_profile_term,
    )
    magnetic_terms = build_terms_from_specs(
        solver,
        magnetic_specs,
        electric_vector,
        H_CURL_ATTR,
        build_profile_term,
    )

    _set_tfsf_state(
        solver,
        provider="analytic_profile",
        source=source,
        lower=lower,
        upper=upper,
        phase_speed=phase_speed,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def initialize_tfsf_state(solver):
    compiled_sources = tuple(getattr(solver, "_compiled_sources", ()) or ())
    if not compiled_sources and getattr(solver, "_compiled_source", None) is not None:
        compiled_sources = (solver._compiled_source,)
    tfsf_sources = [source for source in compiled_sources if source.get("injection", {}).get("kind") == "tfsf"]
    if not tfsf_sources:
        solver.tfsf_enabled = False
        solver._tfsf_state = None
        return

    if len(tfsf_sources) > 1:
        raise ValueError("FDTD currently supports at most one TFSF source per scene.")

    source = tfsf_sources[0]
    if source["kind"] not in {"plane_wave", "gaussian_beam"}:
        raise ValueError("TFSF injection currently supports PlaneWave and GaussianBeam only.")
    if source["injection"].get("mode", "box") == "slab":
        if solver.scene.boundary.uses_kind("bloch"):
            validate_grating_tfsf_slab_topology(solver)
            _set_pending_tfsf_slab_state(solver, source)
            return
        raise NotImplementedError("TFSF slab runtime support is not implemented yet.")
    if solver.scene.boundary.uses_kind("periodic") or solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError("TFSF injection currently supports only none, pml, pec, or pmc boundaries.")

    lower, upper = resolve_bounds_indices(solver.scene, source["injection"]["bounds"])
    validate_bounds(solver, lower, upper)
    validate_background_is_vacuum(solver, lower, upper)
    if source["kind"] == "plane_wave":
        _initialize_plane_wave_auxiliary_state(solver, source, lower, upper)
        return
    _initialize_analytic_tfsf_state(solver, source, lower, upper)
