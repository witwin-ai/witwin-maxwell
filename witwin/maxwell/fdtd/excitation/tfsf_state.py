from __future__ import annotations

import numpy as np
import torch
import math

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
    require_locally_uniform_axis,
    resolve_bounds_indices,
    solve_auxiliary_step,
    solve_numerical_wavenumber,
    validate_background_is_vacuum,
    validate_bounds,
)
from .tfsf_specs import (
    AXIS_INDEX,
    E_CURL_ATTR,
    H_CURL_ATTR,
    axis_aligned_direction,
    axis_aligned_sample_indices,
    axis_aligned_sample_view,
    build_discrete_tfsf_specs,
    build_slab_tfsf_specs,
    constant_line_index_tensor,
    discrete_plane_wave_vectors,
    is_reference_plane_wave_x_ez,
    line_index_tensor,
    magnetic_physical_vector,
    magnetic_unit_vector,
    make_reference_term,
)


ABSORBER_CELLS = 320
_DOWNSTREAM_MARGIN_CELLS = 20


def _domain_bounds(solver):
    return (
        (solver.scene.domain_range[0], solver.scene.domain_range[1]),
        (solver.scene.domain_range[2], solver.scene.domain_range[3]),
        (solver.scene.domain_range[4], solver.scene.domain_range[5]),
    )


def _make_auxiliary_grid(solver, *, s_min, s_max, ds):
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_time = solver._source_time
    incident_reference = None
    extent64 = None
    if source_time["kind"] == "cw":
        source = solver._compiled_source
        injection = source["injection"]
        bounds = injection.get("bounds")
        if bounds is None:
            bounds = list(_domain_bounds(solver))
            bounds[AXIS_INDEX[injection["axis"]]] = injection["axis_bounds"]
        center = tuple((low + high) / 2 for low, high in bounds)
        reference = sum(value * direction for value, direction in zip(center, source["direction"]))
        omega = 2 * math.pi * float(source_time["frequency"])
        argument = math.sin(omega * solver.dt / 2) * ds / (solver.c * solver.dt)
        if argument >= 1:
            raise ValueError("CW incident frequency exceeds the auxiliary grid's propagating band; "
                             "refine the spatial grid before defining an incident phase reference.")
        wavenumber = 2 * math.asin(argument) / ds
        # Transport the physically timed hard seed's CW phase to the declared
        # injection-box centre, retaining the auxiliary discrete wavenumber.
        source_time = dict(source_time, phase=float(source_time["phase"]) +
                           wavenumber * (reference - s_min))
    else:
        source = solver._compiled_source
        lower, upper, _ = resolve_tfsf_region_indices(solver, source["injection"])
        bounds = tuple((float(coords[lo]), float(coords[hi])) for coords, lo, hi in
                       zip((solver.scene.x, solver.scene.y, solver.scene.z), lower, upper))
        incident_reference, exit_plane = projection_extrema(bounds, source["direction"])
        # Start the incident line at the physical entry plane, with upstream
        # ghost samples for staggered face corrections. Its entire extent is
        # independent of PML padding. The entry E is the declared waveform.
        projected_step = max(abs(direction) * float(np.diff(getattr(solver.scene, f"{axis}_nodes64"))[
            max(lo - 1, 0):hi + 1].max()) for axis, direction, lo, hi in
            zip("xyz", source["direction"], lower, upper))
        upstream_cells = max(2, math.ceil(0.5 * projected_step / ds + 0.5))
        s_min = incident_reference - upstream_cells * ds
        s_max = exit_plane + 6 * ds
        # Count from the original double-precision geometry. FP32 face
        # coordinates can exceed the integer snap window; their sampled origin
        # and positions still retain the exact existing arithmetic above.
        bounds64 = tuple((getattr(solver.scene, axis + "_nodes64")[lo],
                          getattr(solver.scene, axis + "_nodes64")[hi])
                         for axis, lo, hi in zip("xyz", lower, upper))
        entry64, exit64 = projection_extrema(bounds64, source["direction"])
        extent64 = exit64 - entry64 + (upstream_cells + 6 + _DOWNSTREAM_MARGIN_CELLS) * ds
    return AuxiliaryGrid1D(
        s_min=s_min,
        s_max=s_max + _DOWNSTREAM_MARGIN_CELLS * ds,
        ds=ds,
        dt=solver.dt,
        wave_speed=solver.c,
        impedance=eta0,
        source_time=source_time,
        device=solver.device,
        dtype=solver.Ex.dtype,
        absorber_cells=ABSORBER_CELLS,
        extent64=extent64,
        fdtd_module=solver.fdtd_module,
        incident_reference=incident_reference,
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
    bounds=None,
):
    injection = source["injection"]
    if auxiliary_grid is not None and auxiliary_grid.incident_reference is not None:
        # Reference terms were built with indices measured from the padded
        # domain minimum; translate them to the physical incident-line origin.
        padded_min, _ = projection_extrema(_domain_bounds(solver), source["direction"])
        shift = round((padded_min - auxiliary_grid.s_min) / auxiliary_grid.ds)
        for term in (*electric_terms, *magnetic_terms):
            if "sample_indices" in term:
                term["sample_indices"] = term["sample_indices"] + shift
                if term["scalar_sample_index"] is not None:
                    term["scalar_sample_index"] += shift
    if auxiliary_grid is not None:
        auxiliary_grid.assert_lossless_samples(electric_terms, magnetic_terms)
    for term in (*electric_terms, *magnetic_terms):
        # The shared face specs use the surface-source orientation. A TFSF
        # enclosure requires its opposite so the interior contains +E_inc and
        # the exterior contains only the scattered field. Keep modal/open-plane
        # source orientation independent by applying this at the enclosure.
        for key in ("coeff_patch", "patch", "cw_cos_patch", "cw_sin_patch"):
            if term.get(key) is not None:
                term[key] = -term[key]
    source["reference_provider_kind"] = provider
    if auxiliary_grid is not None:
        solver._source_time_offset = 0.0 if provider == "plane_wave_fixed_k" else 1.0
    source["reference_waveform_kind"] = source["source_time"]["kind"]
    source["reference_incident_direction"] = tuple(float(v) for v in source["direction"])
    source["reference_incident_bounds"] = tuple(
        (float(coords[lo]), float(coords[hi])) for coords, lo, hi in
        zip((solver.scene.x, solver.scene.y, solver.scene.z), lower, upper))
    if auxiliary_grid is not None and auxiliary_grid.incident_reference is not None:
        source["reference_incident_kind"] = "physical_upstream_yee_entry_plane"
        source["reference_incident_projection"] = float(auxiliary_grid.incident_reference)
    else:
        source["reference_incident_kind"] = ("box_center" if source["source_time"]["kind"] == "cw"
                                             else "analytic_profile")
        if source["source_time"]["kind"] == "cw":
            reference_bounds = injection.get("bounds") if bounds is None else bounds
            if reference_bounds is None:
                reference_bounds = list(_domain_bounds(solver))
                reference_bounds[AXIS_INDEX[injection["axis"]]] = injection["axis_bounds"]
            source["reference_incident_projection"] = sum(
                (float(lo) + float(hi)) * .5 * float(direction)
                for (lo, hi), direction in zip(reference_bounds, source["direction"]))
    state = {
        "provider": provider,
        "lower": lower,
        "upper": upper,
        "bounds": injection.get("bounds") if bounds is None else bounds,
        "mode": injection.get("mode", "box"),
        "electric_terms": electric_terms,
        "magnetic_terms": magnetic_terms,
    }
    if injection.get("axis") is not None:
        state["axis"] = injection["axis"]
    if provider in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}:
        state["electric_batch"] = build_batched_reference_terms(solver, electric_terms)
        state["magnetic_batch"] = build_batched_reference_terms(solver, magnetic_terms)
    elif provider == "plane_wave_aux":
        state["electric_batch"] = build_batched_aux_terms(solver, electric_terms)
        state["magnetic_batch"] = build_batched_aux_terms(solver, magnetic_terms)
    if auxiliary_grid is not None:
        state["auxiliary_grid"] = auxiliary_grid
        if auxiliary_grid.incident_reference is not None:
            state["incident_reference"] = auxiliary_grid.incident_reference
    if phase_speed is not None:
        state["phase_speed"] = phase_speed
    solver._tfsf_state = state
    solver.tfsf_enabled = True


def _validate_locally_uniform_region(solver, lower, upper):
    """Validate all three axes of a TFSF region and return the local spacings.

    The specs touch the ``lower - 1`` / ``upper`` faces, so the primal-spacing
    window is the region expanded by one cell on each side.
    """
    return {
        axis: require_locally_uniform_axis(
            solver,
            axis,
            int(lo) - 1,
            int(hi) + 1,
            context="TFSF/PlaneWave injection",
        )
        for axis, lo, hi in zip("xyz", lower, upper)
    }


def resolve_tfsf_region_indices(solver, injection):
    mode = injection.get("mode", "box")
    if mode == "box":
        lower, upper = resolve_bounds_indices(solver.scene, injection["bounds"])
        return lower, upper, injection["bounds"]
    if mode != "slab":
        raise ValueError(f"Unsupported TFSF injection mode: {mode!r}.")

    axis = injection["axis"]
    axis_index = AXIS_INDEX[axis]
    shape = (solver.Nx, solver.Ny, solver.Nz)
    lower = []
    upper = []
    bounds = []
    for current_axis, axis_coords, size in zip("xyz", (solver.scene.x, solver.scene.y, solver.scene.z), shape):
        if current_axis == axis:
            axis_bounds = injection["axis_bounds"]
            lower.append(nearest_index(axis_coords, axis_bounds[0]))
            upper.append(nearest_index(axis_coords, axis_bounds[1]))
            bounds.append((float(axis_bounds[0]), float(axis_bounds[1])))
        else:
            lo = int(solver.scene.pml_thickness_for_face(current_axis, "low"))
            hi = int(size - solver.scene.pml_thickness_for_face(current_axis, "high") - 1)
            lower.append(lo)
            upper.append(hi)
            bounds.append((float(axis_coords[lo].item()), float(axis_coords[hi].item())))

    if upper[axis_index] <= lower[axis_index] + 1:
        raise ValueError(f"TFSF slab bounds must span at least two cells along {axis}.")
    for transverse_axis, lo, hi in zip("xyz", lower, upper):
        if hi <= lo:
            raise ValueError(f"TFSF slab transverse span is empty along {transverse_axis}.")
    return tuple(lower), tuple(upper), tuple(bounds)


def _validate_slab_interfaces_are_vacuum(solver, lower, upper, axis: str, tol=1e-5):
    axis_index = AXIS_INDEX[axis]
    for face_index in (lower[axis_index], upper[axis_index]):
        face_slice = [
            slice(lower[0], upper[0] + 1),
            slice(lower[1], upper[1] + 1),
            slice(lower[2], upper[2] + 1),
        ]
        face_slice[axis_index] = slice(face_index, face_index + 1)
        face_slice = tuple(face_slice)
        eps_face = solver.epsilon_r[face_slice]
        mu_face = solver.mu_r[face_slice]
        if not torch.allclose(eps_face, torch.ones_like(eps_face), atol=tol, rtol=0.0):
            raise ValueError("TFSF slab interfaces must remain in vacuum.")
        if not torch.allclose(mu_face, torch.ones_like(mu_face), atol=tol, rtol=0.0):
            raise ValueError("TFSF slab interfaces must remain in vacuum.")


def _validate_slab_normal_bounds(solver, lower, upper, axis: str):
    axis_index = AXIS_INDEX[axis]
    size = (solver.Nx, solver.Ny, solver.Nz)[axis_index]
    lo = lower[axis_index]
    hi = upper[axis_index]
    low_margin = solver.scene.pml_thickness_for_face(axis, "low")
    high_margin = solver.scene.pml_thickness_for_face(axis, "high")
    if hi <= lo + 1:
        raise ValueError(f"TFSF slab bounds must span at least two cells along {axis}.")
    if lo < low_margin or hi > size - high_margin - 1:
        raise ValueError("TFSF slab bounds must lie strictly inside the non-PML simulation region.")


def _initialize_grating_slab_cw_state(solver, source, lower, upper, bounds, deltas):
    if source["kind"] != "plane_wave":
        raise ValueError("Grating TFSF slab injection requires a PlaneWave source.")
    axis = source["injection"]["axis"]
    # Any of the three normal axes is supported: ``build_slab_tfsf_specs`` selects
    # the two faces normal to ``axis``, ``discrete_plane_wave_vectors`` resolves the
    # oblique polarization from the direction, and the mixed Bloch/CPML runtime and
    # adjoint replay each dispatch on the single PML axis rather than assuming z.
    # The transverse-axis topology (two Bloch axes, PML on the normal axis) is
    # validated upstream by ``validate_grating_tfsf_slab_topology``.
    #
    # Nonzero fixed-k pulses use the complex line before reaching this path.
    # Keep the analytic CW and zero-k pulse construction unchanged.
    source_time = source["source_time"]

    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    k_numeric = solve_numerical_wavenumber(solver, source["direction"], deltas)
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    electric_vector, magnetic_unit_vector_map = discrete_plane_wave_vectors(
        source["direction"],
        source["polarization"],
        k_numeric,
        deltas,
    )
    magnetic_vector = {name: value / eta0 for name, value in magnetic_unit_vector_map.items()}
    reference_point = box_center_tensor(solver, bounds)
    electric_specs, magnetic_specs = build_slab_tfsf_specs(lower, upper, axis=axis)

    def build_discrete_cw_term(spec, coeff_patch, component_scale):
        _, delay_patch = incident_profile(
            solver,
            {"kind": "plane_wave", "direction": source["direction"]},
            _spec_positions(solver, spec),
            reference_point,
            phase_speed=phase_speed,
            source_frequency=source_frequency,
        )
        term = build_term_from_profile(
            solver,
            field_name=spec["field_name"],
            offsets=spec["offsets"],
            scale=spec["sign"] * component_scale * coeff_patch,
            delay_patch=delay_patch,
            activation_delay_patch=None,
            source_time=source_time,
            omega=source_omega,
        )
        if term is not None:
            # A complete slab face already contains both native Bloch endpoint
            # representatives. Extra wrap copies double-deposit those sources.
            term["periodic_wrap_axes"] = (0, 0)
        return term

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
        provider="plane_wave_grating_slab_cw",
        source=source,
        lower=lower,
        upper=upper,
        bounds=bounds,
        phase_speed=phase_speed,
        electric_terms=electric_terms,
        magnetic_terms=magnetic_terms,
    )


def _spec_positions(solver, spec):
    return solver._component_positions(
        spec["incident_name"],
        spec["sample_offsets"],
        spec["sample_shape"],
        dtype=solver.Ex.dtype,
    )


def _initialize_axis_aligned_plane_wave_auxiliary_state(
    solver, source, lower, upper, axis: str, direction_sign: int, deltas, *, slab_axis=None
):
    electric_vector = {
        "Ex": source["polarization"][0],
        "Ey": source["polarization"][1],
        "Ez": source["polarization"][2],
    }
    magnetic_vector = magnetic_unit_vector(source["direction"], source["polarization"])
    aux = _make_directional_auxiliary_grid(solver, source["direction"], float(deltas[axis]))

    # A slab injection keeps only the two faces normal to the propagation axis; the
    # total-field region fills the whole transverse span, so no transverse-face
    # corrections exist (the transverse boundary absorbs/wraps the field directly).
    if slab_axis is None:
        electric_specs, magnetic_specs = build_discrete_tfsf_specs(lower, upper)
    else:
        electric_specs, magnetic_specs = build_slab_tfsf_specs(lower, upper, axis=slab_axis)

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


def _initialize_plane_wave_discrete_cw_state(solver, source, lower, upper, deltas):
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    k_numeric = solve_numerical_wavenumber(solver, source["direction"], deltas)
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    electric_vector, magnetic_unit_vector_map = discrete_plane_wave_vectors(
        source["direction"],
        source["polarization"],
        k_numeric,
        deltas,
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


def _initialize_reference_plane_wave_auxiliary_state(solver, source, lower, upper, deltas):
    ix0, iy0, iz0 = lower
    ix1, iy1, iz1 = upper

    dx_local = float(deltas["x"])
    dy_local = float(deltas["y"])
    dz_local = float(deltas["z"])
    electric_scale = float(source["polarization"][2])
    aux = _make_auxiliary_grid(
        solver,
        s_min=float(solver.scene.domain_range[0]),
        s_max=float(solver.scene.domain_range[1]),
        ds=dx_local,
    )

    magnetic_terms = [
        make_reference_term(
            solver,
            field_name="Hy",
            offsets=(ix0 - 1, iy0, iz0),
            coeff_patch=solver.chy_curl[ix0 - 1 : ix0, iy0 : iy1 + 1, iz0:iz1] / dx_local,
            sample_kind="electric",
            sample_indices=constant_line_index_tensor(solver, ix0),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hy",
            offsets=(ix1, iy0, iz0),
            coeff_patch=-solver.chy_curl[ix1 : ix1 + 1, iy0 : iy1 + 1, iz0:iz1] / dx_local,
            sample_kind="electric",
            sample_indices=constant_line_index_tensor(solver, ix1),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hx",
            offsets=(ix0, iy0 - 1, iz0),
            coeff_patch=-solver.chx_curl[ix0 : ix1 + 1, iy0 - 1 : iy0, iz0:iz1] / dy_local,
            sample_kind="electric",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0 + 1),
            component_scale=+electric_scale,
        ),
        make_reference_term(
            solver,
            field_name="Hx",
            offsets=(ix0, iy1, iz0),
            coeff_patch=solver.chx_curl[ix0 : ix1 + 1, iy1 : iy1 + 1, iz0:iz1] / dy_local,
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
            coeff_patch=-solver.cez_curl[ix0 : ix0 + 1, iy0 : iy1 + 1, iz0:iz1] / dx_local,
            sample_kind="magnetic",
            sample_indices=constant_line_index_tensor(solver, ix0 - 1),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ez",
            offsets=(ix1, iy0, iz0),
            coeff_patch=solver.cez_curl[ix1 : ix1 + 1, iy0 : iy1 + 1, iz0:iz1] / dx_local,
            sample_kind="magnetic",
            sample_indices=constant_line_index_tensor(solver, ix1),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ex",
            offsets=(ix0, iy0, iz0),
            coeff_patch=solver.cex_curl[ix0:ix1, iy0 : iy1 + 1, iz0 : iz0 + 1] / dz_local,
            sample_kind="magnetic",
            sample_indices=line_index_tensor(solver, ix0, ix1 - ix0),
            component_scale=1.0,
        ),
        make_reference_term(
            solver,
            field_name="Ex",
            offsets=(ix0, iy0, iz1),
            coeff_patch=-solver.cex_curl[ix0:ix1, iy0 : iy1 + 1, iz1 : iz1 + 1] / dz_local,
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


def _initialize_plane_wave_auxiliary_state(solver, source, lower, upper, deltas):
    if is_reference_plane_wave_x_ez(source):
        _initialize_reference_plane_wave_auxiliary_state(solver, source, lower, upper, deltas)
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
            deltas,
        )
        return

    if source["source_time"]["kind"] == "cw":
        _initialize_plane_wave_discrete_cw_state(solver, source, lower, upper, deltas)
        return

    k_numeric = solve_numerical_wavenumber(solver, source["direction"], deltas)
    source_omega = 2.0 * np.pi * float(source["source_time"]["frequency"])
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    electric_vector, magnetic_vector = discrete_plane_wave_vectors(
        source["direction"],
        source["polarization"],
        k_numeric,
        deltas,
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


def _initialize_analytic_tfsf_state(solver, source, lower, upper, deltas):
    box_center = box_center_tensor(solver, source["injection"]["bounds"])
    eta0 = (solver.mu0 / solver.eps0) ** 0.5
    source_time = source["source_time"]
    source_frequency = float(source_time["frequency"])
    source_omega = 2.0 * np.pi * source_frequency
    k_numeric = solve_numerical_wavenumber(solver, source["direction"], deltas)
    phase_speed = solver.c if k_numeric <= 1e-12 else source_omega / k_numeric
    entry_s = entry_projection(source["injection"]["bounds"], source["direction"])

    if source["kind"] in {"gaussian_beam", "astigmatic_gaussian_beam"}:
        electric_vector, magnetic_unit_vector_map = discrete_plane_wave_vectors(
            source["direction"],
            source["polarization"],
            k_numeric,
            deltas,
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


def _initialize_nonperiodic_slab_state(solver, source, lower, upper, deltas):
    """Slab TFSF with non-periodic (PML/absorbing) transverse boundaries.

    A two-face slab injects a plane wave that fills the whole transverse cross
    section. With absorbing transverse boundaries this is only well posed for a
    wave that is uniform across that cross section, i.e. normal incidence along
    the slab axis; an oblique wave would carry a transverse phase gradient that
    the omitted transverse faces cannot terminate (that case needs Bloch
    transverse boundaries and is handled by the grating slab path). The state is
    built on the same axis-aligned auxiliary-line provider the box TFSF uses, so
    both CW and pulsed source_times are supported and the forward/adjoint runtime
    needs no slab-specific branch.
    """
    axis = source["injection"]["axis"]
    if source["kind"] != "plane_wave":
        raise ValueError("Non-periodic TFSF slab injection requires a PlaneWave source.")
    axis_direction = axis_aligned_direction(source["direction"])
    if axis_direction is None or axis_direction[0] != axis:
        raise ValueError(
            "A TFSF slab with non-periodic transverse boundaries must be a plane wave "
            f"normally incident along the slab axis {axis!r}; use Bloch transverse "
            "boundaries for oblique-incidence (grating) slab injection."
        )
    _initialize_axis_aligned_plane_wave_auxiliary_state(
        solver,
        source,
        lower,
        upper,
        axis_direction[0],
        axis_direction[1],
        deltas,
        slab_axis=axis,
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
    if source["kind"] not in {"plane_wave", "gaussian_beam", "astigmatic_gaussian_beam"}:
        raise ValueError(
            "TFSF injection currently supports PlaneWave, GaussianBeam, and AstigmaticGaussianBeam only."
        )
    if source["injection"].get("mode", "box") == "slab":
        if solver.scene.boundary.uses_kind("bloch"):
            validate_grating_tfsf_slab_topology(solver)
            lower, upper, bounds = resolve_tfsf_region_indices(solver, source["injection"])
            deltas = _validate_locally_uniform_region(solver, lower, upper)
            _validate_slab_normal_bounds(solver, lower, upper, source["injection"]["axis"])
            _validate_slab_interfaces_are_vacuum(solver, lower, upper, source["injection"]["axis"])
            if source["source_time"]["kind"] != "cw" and any(
                k != 0.0 and solver.scene.boundary.axis_kind(axis) == "bloch"
                for axis, k in zip("xyz", solver.resolved_bloch_wavevector)
            ):
                from .fixed_k import initialize_fixed_k_slab

                initialize_fixed_k_slab(solver, source, lower, upper, bounds, deltas)
                return
            _initialize_grating_slab_cw_state(solver, source, lower, upper, bounds, deltas)
            return
        axis = source["injection"]["axis"]
        lower, upper, _bounds = resolve_tfsf_region_indices(solver, source["injection"])
        deltas = _validate_locally_uniform_region(solver, lower, upper)
        _validate_slab_normal_bounds(solver, lower, upper, axis)
        _validate_slab_interfaces_are_vacuum(solver, lower, upper, axis)
        _initialize_nonperiodic_slab_state(solver, source, lower, upper, deltas)
        return
    if solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError("TFSF slab mode is required for Bloch-boundary TFSF injection.")
    if solver.scene.boundary.uses_kind("periodic"):
        raise NotImplementedError("TFSF injection currently supports only none, pml, pec, or pmc boundaries.")

    lower, upper, _ = resolve_tfsf_region_indices(solver, source["injection"])
    deltas = _validate_locally_uniform_region(solver, lower, upper)
    validate_bounds(solver, lower, upper)
    validate_background_is_vacuum(solver, lower, upper)
    if source["kind"] == "plane_wave":
        _initialize_plane_wave_auxiliary_state(solver, source, lower, upper, deltas)
        return
    _initialize_analytic_tfsf_state(solver, source, lower, upper, deltas)
