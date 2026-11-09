"""Mode-source solver package for FDTD excitation.

Four mode families share this package:

* ``eigensolvers``: scalar-operator builders and dense/sparse eigenpair solvers.
* ``vector``: the nodal full-vector transverse operator, mode selection, and
  physics diagnostics.
* ``yee``: the Yee-staggered transverse vector operator, including the PEC
  staircase (interior conductor) path.
* ``tem``: TEM / quasistatic line modes from electrostatic potentials.
* ``common``: shared constants and mode-plane helpers.

The canonical import path stays ``witwin.maxwell.fdtd.excitation.modes``: this
``__init__`` re-exports every name the former single module defined. The
top-level dispatchers (``solve_mode_source_profile``, the vector-mode
assembler, and the sparse-eigensolver autograd wrapper) are defined directly in
this module so that attributes overridden on the canonical namespace (dense
solver size limits, eigensolver hooks) are still read here at call time,
exactly as in the previous single-module layout.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as scipy_sparse_linalg

from ....constants import ETA_0
from ..tfsf_common import _component_roundoff_floor, nearest_index
from .common import (
    _AXIS_TO_INDEX,
    _DENSE_EIGEN_LIMIT,
    _FIELD_COMPONENTS,
    _FULL_VECTOR_DENSE_LIMIT,
    _VECTOR_GRAD_MATRIX_LIMIT,
    _mode_source_trainable_plane,
    _require_lossless_vector_planes,
    _IMPLICIT_EIGEN_CG_MAX_ITER,
    _IMPLICIT_EIGEN_CG_TOL,
    _LOBPCG_MAX_ITER,
    _LOBPCG_REQUEST_PADDING,
    _LOBPCG_TOL,
    _MODE_PLANE_SPACING_SPREAD_BOUND,
    _PEC_OCCUPANCY_THRESHOLD,
    _PEC_VECTOR_MATRIX_LIMIT,
    _RIGHT_HANDED_TANGENTIAL_AXES,
    _SPURIOUS_NEAR_K0_BETA_LIMIT,
    _SPURIOUS_UNIFORMITY_LIMIT,
    _UNIFORM_SPACING_RTOL,
    _VECTOR_CHECKERBOARD_FRACTION_LIMIT,
    _VECTOR_DEGENERATE_RTOL,
    _VECTOR_DUPLICATE_BETA_RTOL,
    _VECTOR_DUPLICATE_OVERLAP_LIMIT,
    _VECTOR_EIGEN_REQUEST_PADDING,
    _VECTOR_EIGS_MAX_ITER,
    _VECTOR_EIGS_TOL,
    _apply_bend_factor,
    _average_node_tensor_to_component,
    _bend_conformal_factor,
    _bilinear_sample,
    _boundary_connected_conductor,
    _cross,
    _discrete_mode_profile_power,
    _field_component_axis_coords,
    _label_connected_components,
    _local_uniform_plane_spacing,
    _max_component_imag,
    _min_component_real,
    _mode_slice,
    _mode_source_axes,
    _mode_source_component_permittivity,
    _mode_source_field_name,
    _mode_source_node_axis_coords,
    _mode_source_pec_slice,
    _mode_source_relative_material_slices,
    _normalize_mode_profiles_to_unit_power,
    _normalize_profile_torch,
    _real_plane_numpy,
    _regular_grid_sample,
    _resolve_tangential_bounds,
    _validate_mode_source_position,
    _vector_mode_supported,
)
from .eigensolvers import (
    _apply_shifted_square_operator,
    _build_scalar_operator,
    _build_scalar_operator_torch_dense,
    _build_scalar_operator_torch_sparse,
    _conjugate_gradient_solve,
    _lobpcg_initial_guess,
    _lobpcg_request_count,
    _orient_unit_eigenvector,
    _project_orthogonal,
    _real_profile_from_complex_eigenvector,
    _solve_mode_eigenpair,
    _solve_mode_eigenpair_torch,
    _solve_mode_eigenpair_torch_sparse,
    _sparse_matvec,
)
from .tem import (
    _quasistatic_laplace_energy,
    _solve_pec_tem_mode_torch,
    _solve_quasistatic_line_modes,
    _solve_terminal_line_basis,
    _solve_terminal_line_mode,
    _tem_signal_potentials,
    _terminal_conductor_labels,
)
from .vector import (
    _build_staggered_first_differences_sparse,
    _build_vector_operator_sparse,
    _build_vector_operator_torch,
    _IsolatedVectorEigenpair,
    _is_uniform_isotropic_vector_plane,
    _normalize_vector_mode_profiles_numpy,
    _pec_first_difference,
    _pec_vector_operator_torch,
    _relative_vector_residual,
    _select_and_normalize_vector_mode_numpy,
    _select_vector_mode_numpy,
    _select_vector_mode_torch,
    _solve_pec_vector_mode_eigenpair_torch,
    _solve_vector_mode_eigenpair_dense,
    _solve_vector_mode_eigenpair_sparse,
    _split_vector_mode_components,
    _vector_mode_checkerboard_fraction_numpy,
    _vector_mode_divergence_residuals_numpy,
    _vector_mode_eigenpair_residual_numpy,
    _vector_mode_envelope_variation_numpy,
    _vector_mode_polarization_fraction_numpy,
    _vector_mode_power_inner_product_numpy,
    _vector_mode_power_sign_numpy,
    _vector_mode_request_count,
    _vector_mode_transverse_uniformity_numpy,
    _vector_mode_wall_peak_fraction_numpy,
    _vector_preferred_component_name,
)
from .yee import (
    _build_yee_transverse_operator_sparse,
    _build_yee_transverse_operator_torch,
    _yee_reconstruct_node_profiles_torch,
    _select_yee_transverse_mode_numpy,
    _solve_yee_transverse_pec_mode,
    _solve_yee_transverse_vector_mode,
    _split_yee_transverse_eigenvector,
    _yee_half_to_node_first_difference,
    _yee_half_to_node_neumann,
    _yee_interior_to_node_dirichlet,
    _yee_pec_connectivity_check,
    _yee_reconstruct_node_profiles,
    _yee_stagger_eps_from_nodes,
    _yee_stagger_pec_from_nodes,
    _yee_transverse_discrete_transverse_wavenumber,
    _yee_transverse_grids,
)


def _solve_mode_eigenpair_complex(operator, *, mode_index: int):
    """Fundamental-first eigenpair of the complex (lossy) scalar mode operator.

    When the plane permittivity carries loss, ``d2 + k0**2 eps`` is complex
    symmetric rather than Hermitian, so the guided spectrum ``beta**2`` is
    complex. Guided modes have the largest real part of ``beta**2`` (the
    fundamental has the largest of all), so the modes are ranked by descending
    ``Re(beta**2)`` and the ``mode_index``-th positive-real-part eigenvalue is
    returned. A dense general eigensolve is used up to ``_DENSE_EIGEN_LIMIT``
    unknowns and a largest-real-part Arnoldi solve above it.
    """
    size = int(operator.shape[0])
    if size <= _DENSE_EIGEN_LIMIT:
        eigenvalues, eigenvectors = scipy_linalg.eig(operator.toarray())
    else:
        requested = min(max(int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING, 4), max(1, size - 2))
        eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
            operator,
            k=requested,
            which="LR",
            tol=_VECTOR_EIGS_TOL,
            maxiter=_VECTOR_EIGS_MAX_ITER,
        )
    order = np.argsort(np.real(eigenvalues))[::-1]
    positive = [int(index) for index in order if np.isfinite(eigenvalues[index]) and np.real(eigenvalues[index]) > 0.0]
    if len(positive) <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {len(positive)} positive-beta modes were found."
        )
    mode_slot = positive[int(mode_index)]
    return complex(eigenvalues[mode_slot]), np.asarray(eigenvectors[:, mode_slot])


class _SparseModeEigenpairFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index_sq: torch.Tensor, du: float, dv: float, mode_index: int):
        operator, interior_u, interior_v = _build_scalar_operator_torch_sparse(index_sq, float(du), float(dv))
        beta_sq, eigenvector = _solve_mode_eigenpair_torch_sparse(
            operator,
            interior_u=interior_u,
            interior_v=interior_v,
            mode_index=int(mode_index),
        )
        eigenvector = _orient_unit_eigenvector(eigenvector.to(dtype=index_sq.dtype))
        beta_sq = beta_sq.to(device=index_sq.device, dtype=index_sq.dtype)
        ctx.du = float(du)
        ctx.dv = float(dv)
        ctx.save_for_backward(index_sq.detach(), beta_sq.detach(), eigenvector.detach())
        return beta_sq, eigenvector

    @staticmethod
    def backward(ctx, grad_beta_sq, grad_eigenvector):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None

        index_sq, beta_sq, eigenvector = ctx.saved_tensors
        operator, interior_u, interior_v = _build_scalar_operator_torch_sparse(index_sq, ctx.du, ctx.dv)

        gradient_vector = torch.zeros_like(eigenvector)
        if grad_eigenvector is not None:
            gradient_vector = grad_eigenvector.to(device=eigenvector.device, dtype=eigenvector.dtype)
            gradient_vector = _project_orthogonal(gradient_vector, eigenvector)

        if float(torch.linalg.vector_norm(gradient_vector).item()) > 0.0:
            solve_rhs = gradient_vector
            solve_y = _conjugate_gradient_solve(
                lambda vector: _apply_shifted_square_operator(
                    operator,
                    vector,
                    eigenvalue=beta_sq,
                    eigenvector=eigenvector,
                ),
                solve_rhs,
                tol=_IMPLICIT_EIGEN_CG_TOL,
                max_iter=_IMPLICIT_EIGEN_CG_MAX_ITER,
            )
            implicit_vector = beta_sq * solve_y - _sparse_matvec(operator, solve_y)
        else:
            implicit_vector = torch.zeros_like(eigenvector)

        grad_lambda = torch.zeros((), device=eigenvector.device, dtype=eigenvector.dtype)
        if grad_beta_sq is not None:
            grad_lambda = grad_beta_sq.to(device=eigenvector.device, dtype=eigenvector.dtype)

        interior_grad = grad_lambda * (eigenvector * eigenvector) + eigenvector * implicit_vector
        grad_index_sq = torch.zeros_like(index_sq)
        grad_index_sq[1:-1, 1:-1] = interior_grad.reshape((interior_u, interior_v))
        return grad_index_sq, None, None, None


def _solve_mode_eigenpair_torch_sparse_implicit(index_sq: torch.Tensor, *, du: float, dv: float, mode_index: int):
    return _SparseModeEigenpairFunction.apply(index_sq, float(du), float(dv), int(mode_index))


def _assemble_vector_mode_data(
    solver,
    source,
    *,
    normal_axis: str,
    tangential_axes,
    tangential_bounds,
    tangential_coord_map,
    plane_index: int,
    eps_by_axis: dict,
    mu_by_axis: dict,
    frequency: float,
    pec_occupancy: torch.Tensor | None = None,
    tem_time_step=None,
):
    axis_u, axis_v = tangential_axes
    field_names = (
        f"E{axis_u.lower()}",
        f"E{axis_v.lower()}",
        f"H{axis_u.lower()}",
        f"H{axis_v.lower()}",
    )
    preferred_field_name = _vector_preferred_component_name(source)
    if preferred_field_name not in field_names:
        raise ValueError("ModeSource polarization must be tangential to the source plane.")

    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    bend_factor = _bend_conformal_factor(
        source,
        tangential_axes=tangential_axes,
        tangential_bounds=tangential_bounds,
        coord_map=tangential_coord_map,
    )
    if bend_factor is not None:
        eps_by_axis = {axis: _apply_bend_factor(component, bend_factor) for axis, component in eps_by_axis.items()}
    has_interior_pec = pec_occupancy is not None and bool(
        torch.any(pec_occupancy[1:-1, 1:-1] >= _PEC_OCCUPANCY_THRESHOLD)
    )
    native_tem = has_interior_pec and source.get("wave_family") == "tem"
    tem_coordinates = {}
    if native_tem:
        tem_coordinates = {
            "coords_u":getattr(solver.scene,axis_u+"_nodes64")[u_lo:u_hi+1],
            "coords_v":getattr(solver.scene,axis_v+"_nodes64")[v_lo:v_hi+1],
        }
    try:
        du = _local_uniform_plane_spacing(solver.scene, axis_u, u_lo, u_hi)
        dv = _local_uniform_plane_spacing(solver.scene, axis_v, v_lo, v_hi)
    except ValueError:
        if not native_tem:
            raise
        # Native TEM uses full primal/dual metric arrays; no scalar spacing
        # approximation is supplied to its potential or gradient operators.
        du = dv = None
    k0 = 2.0 * math.pi * float(frequency) / float(solver.c)
    # Order the diagonal components as (in-plane u, in-plane v, plane-normal w) so
    # the vector operator threads eps_ww/mu_ww through the eliminated longitudinal
    # fields and eps_uu/eps_vv (mu_uu/mu_vv) through the transverse self-terms.
    tensor_eps_planes = (
        eps_by_axis[axis_u],
        eps_by_axis[axis_v],
        eps_by_axis[normal_axis],
    )
    tensor_mu_planes = (
        mu_by_axis[axis_u],
        mu_by_axis[axis_v],
        mu_by_axis[normal_axis],
    )
    candidate_diagnostics = {
        "candidates": None,
        "overlap_matrix": None,
        "selected_candidate_index": None,
    }
    if source.get("terminal_mode") is not None:
        beta_value, component_arrays, pec_node_count, terminal_record = _assemble_terminal_line_mode(
            source,
            source["terminal_mode"],
            pec_occupancy,
            tensor_eps_planes=tensor_eps_planes,
            tensor_mu_planes=tensor_mu_planes,
            k0=k0,
            du=du,
            dv=dv,
            field_names=field_names,
            tem_coordinates=tem_coordinates,
        )
        payload = _finalize_vector_mode_data(solver, source, beta_value=beta_value, component_arrays=component_arrays, pec_node_count=pec_node_count, solver_kind="quasistatic_line_torch", candidate_diagnostics=candidate_diagnostics, k0=k0, eps_by_axis=eps_by_axis, mu_by_axis=mu_by_axis, tensor_eps_planes=tensor_eps_planes, field_names=field_names, preferred_field_name=preferred_field_name, normal_axis=normal_axis, tangential_axes=tangential_axes, tangential_bounds=tangential_bounds, tangential_coord_map=tangential_coord_map, plane_index=plane_index, native_tem=native_tem, du=du, tem_coordinates=tem_coordinates)
        payload["terminal_mode"] = terminal_record
        return payload
    pec_node_count = 0
    vector_state = getattr(solver, "_mode_source_vector_state", {}).get(source["name"], {})
    trainable_vector = vector_state.get("trainable", False)
    if not has_interior_pec and trainable_vector:
        beta_value, component_arrays, candidate_diagnostics = _solve_vector_mode_torch(
            tensor_eps_planes, tensor_mu_planes, source=source, k0=k0, du=du, dv=dv,
            field_names=field_names, preferred_field_name=preferred_field_name,
            gauge_state=vector_state)
        solver_kind = "vector_dense_torch"
    elif has_interior_pec and source.get("wave_family") == "tem":
        normal_spacing = None
        if tem_time_step is not None:
            nodes = getattr(solver.scene, normal_axis + "_nodes64")
            low, high = nodes[plane_index]-nodes[plane_index-1], nodes[plane_index+1]-nodes[plane_index]
            if not math.isclose(low, high, rel_tol=1e-7, abs_tol=0.):
                raise NotImplementedError("Lossy TEM native propagation requires locally uniform axial spacing at the mode plane.")
            normal_spacing = float(low)
        try:
            beta_value, component_arrays, pec_node_count = _solve_pec_tem_mode_torch(
                tensor_eps_planes,
                tensor_mu_planes,
                pec_occupancy,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
                field_names=field_names,
                time_step=tem_time_step,
                normal_spacing=normal_spacing,
                **tem_coordinates,
            )
            solver_kind = "tem_electrostatic_torch"
        except NotImplementedError:
            if _max_component_imag(eps_by_axis) > 0 or _max_component_imag(mu_by_axis) > 0:
                raise NotImplementedError("Spatially varying lossy quasi-TEM modes require a complex material potential solve.")
            # Inhomogeneous cross-section (substrate + air microstrip / coupled pair):
            # the uniform-fill electrostatic normalization does not apply, so the legacy
            # closed-form TEM solve fails closed. Route to the quasi-static variable-eps
            # Laplace engine, which returns eps_eff = C / C0 for the true dielectric
            # profile. That engine is non-magnetic; a magnetic inhomogeneous line is out
            # of scope and re-raises the uniform-fill guard.
            mu_uniform_unit = all(
                bool(np.allclose(_real_plane_numpy(component), 1.0))
                for component in tensor_mu_planes
            )
            if not mu_uniform_unit:
                raise
            signal_potentials = _tem_signal_potentials(
                pec_occupancy, int(source["mode_index"])
            )
            line = _solve_quasistatic_line_modes(
                tensor_eps_planes,
                pec_occupancy,
                k0=k0,
                du=du,
                dv=dv,
                field_names=field_names,
                signal_potentials=signal_potentials,
                **tem_coordinates,
            )
            beta_value = line["beta"]
            component_arrays = line["component_profiles"]
            pec_node_count = int(line["conductor_count"])
            solver_kind = "quasistatic_line_torch"
    elif has_interior_pec:
        operator, active, interior_u, interior_v, pec_node_count = _pec_vector_operator_torch(
            tensor_eps_planes,
            tensor_mu_planes,
            pec_occupancy,
            k0=k0,
            du=du,
            dv=dv,
        )
        beta_value, _eigenvector, component_arrays = _solve_pec_vector_mode_eigenpair_torch(
            operator,
            active,
            interior_u=interior_u,
            interior_v=interior_v,
            mode_index=int(source["mode_index"]),
            field_names=field_names,
            preferred_field_name=preferred_field_name,
        )
        solver_kind = "vector_pec_dense_torch"
    else:
        eps_planes = tuple(_real_plane_numpy(component) for component in tensor_eps_planes)
        mu_planes = tuple(_real_plane_numpy(component) for component in tensor_mu_planes)
        unknowns = max(
            (int(eps_planes[0].shape[0]) - 2) * (int(eps_planes[0].shape[1]) - 2),
            0,
        )
        use_dense = unknowns <= _FULL_VECTOR_DENSE_LIMIT
        uniform_isotropic = _is_uniform_isotropic_vector_plane(eps_planes, mu_planes)
        # The Yee-staggered operator eliminates Hz assuming mu = 1, so it is only
        # valid for a NON-MAGNETIC cross-section. A uniformly magnetic aperture
        # (uniform mu_r != 1) also classifies as uniform-isotropic, but must stay on
        # the legacy diagonal-anisotropic operator, which threads mu through the
        # eliminated longitudinal fields. Uniform *dielectric* filling (any uniform
        # eps_r with mu = 1) is handled correctly by the Yee operator: it carries the
        # true per-component eps, not vacuum.
        nonmagnetic = bool(np.allclose(np.asarray(mu_planes[0]).reshape(-1)[0], 1.0)) and all(
            bool(np.allclose(component, 1.0)) for component in mu_planes
        )
        if uniform_isotropic and nonmagnetic:
            # A homogeneous non-magnetic aperture (the hollow metallic guide, a
            # uniformly dielectric-filled guide, and free-space WavePorts) is solved
            # on the Yee-staggered transverse full-vector operator: each transverse E
            # component stays on its own Yee location, so the symmetric PEC walls
            # yield a clean full-grid guided mode (E1b). Inhomogeneous / magnetic
            # cross-sections keep the legacy diagonal-anisotropic operator (see the
            # E1 acceptance doc for scope).
            beta_value, component_arrays, diagnostics = _solve_yee_transverse_vector_mode(
                eps_planes,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
                field_names=field_names,
                preferred_field_name=preferred_field_name,
                wave_family=source.get("wave_family"),
                uniform=True,
                use_dense=use_dense,
            )
            candidate_diagnostics = {
                "candidates": diagnostics["candidates"],
                "overlap_matrix": diagnostics["overlap_matrix"],
                "selected_candidate_index": diagnostics["selected_candidate_index"],
            }
        else:
            solve_fn = _solve_vector_mode_eigenpair_dense if use_dense else _solve_vector_mode_eigenpair_sparse
            beta_value, _eigenvector, component_arrays, candidate_diagnostics = solve_fn(
                eps_planes,
                mu_planes,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
                field_names=field_names,
                preferred_field_name=preferred_field_name,
                wave_family=source.get("wave_family"),
            )
        solver_kind = "vector_dense" if use_dense else "vector_sparse"

    return _finalize_vector_mode_data(solver, source, beta_value=beta_value, component_arrays=component_arrays, pec_node_count=pec_node_count, solver_kind=solver_kind, candidate_diagnostics=candidate_diagnostics, k0=k0, eps_by_axis=eps_by_axis, mu_by_axis=mu_by_axis, tensor_eps_planes=tensor_eps_planes, field_names=field_names, preferred_field_name=preferred_field_name, normal_axis=normal_axis, tangential_axes=tangential_axes, tangential_bounds=tangential_bounds, tangential_coord_map=tangential_coord_map, plane_index=plane_index, native_tem=native_tem, du=du, tem_coordinates=tem_coordinates)


def _finalize_vector_mode_data(
    solver, source, *, beta_value, component_arrays, pec_node_count, solver_kind, candidate_diagnostics,
    k0, eps_by_axis, mu_by_axis, tensor_eps_planes, field_names, preferred_field_name, normal_axis,
    tangential_axes, tangential_bounds, tangential_coord_map, plane_index, native_tem, du, tem_coordinates,
):
    """Normalize, place and package one solved transverse mode (shared by every solver kind)."""
    axis_u, axis_v = tangential_axes
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    beta_real = (
        float(torch.real(beta_value).item())
        if isinstance(beta_value, torch.Tensor)
        else float(np.real(beta_value))
    )
    effective_index_value = beta_real / k0
    target_device = torch.device(getattr(solver, "device", solver.scene.device))
    target_dtype = solver.Ex.dtype
    complex_tem = solver_kind == "tem_electrostatic_torch" and (
        _max_component_imag(eps_by_axis) > 0 or _max_component_imag(mu_by_axis) > 0)
    profile_dtype = (torch.complex64 if target_dtype == torch.float32 else torch.complex128) if complex_tem else target_dtype
    # The profile set is peak-normalized to 1 on the preferred component, so
    # compare the residual imaginary content against the joint profile scale.
    # An iterative (ARPACK) eigensolve leaves tolerance-level imaginary noise
    # (~1e-5 relative); a genuinely complex (lossy) mode has imag of order one.
    profile_scale = max(
        (
            float(torch.max(torch.abs(array)).item())
            if isinstance(array, torch.Tensor)
            else float(np.max(np.abs(np.asarray(array))))
            for array in component_arrays.values()
        ),
        default=0.0,
    )
    imag_bound = 1e-3 * profile_scale
    plane_shape = tuple(int(dim) for dim in tensor_eps_planes[0].shape)
    interior_shape = (plane_shape[0] - 2, plane_shape[1] - 2)
    staggered = solver_kind in {"tem_electrostatic_torch", "quasistatic_line_torch"}
    component_profiles = {}
    for name, array in component_arrays.items():
        if isinstance(array, torch.Tensor):
            resolved_array = array
            if resolved_array.is_complex() and not complex_tem:
                imag_max = float(torch.max(torch.abs(torch.imag(resolved_array))).item())
                if imag_max > imag_bound:
                    raise RuntimeError(
                        "Full-vector ModeSource forward solve returned a materially complex field profile."
                    )
                resolved_array = torch.real(resolved_array)
            resolved_profile = resolved_array.to(
                device=target_device,
                dtype=profile_dtype,
            ).contiguous()
        else:
            resolved_array = np.asarray(array)
            if np.iscomplexobj(resolved_array) and not complex_tem:
                imag_max = float(np.max(np.abs(np.imag(resolved_array))))
                if imag_max > imag_bound:
                    raise RuntimeError(
                        "Full-vector ModeSource forward solve returned a materially complex field profile."
                    )
                resolved_array = np.real(resolved_array)
            resolved_profile = torch.as_tensor(
                resolved_array,
                device=target_device,
                dtype=profile_dtype,
            ).contiguous()

        resolved_shape = tuple(int(dim) for dim in resolved_profile.shape)
        if staggered:
            expected = list(plane_shape)
            if name in (field_names[0], field_names[3]):
                expected[0] -= 1
            elif name in (field_names[1], field_names[2]):
                expected[1] -= 1
            if resolved_shape != tuple(expected):
                raise RuntimeError(f"Staggered {name} mode profile has shape {resolved_shape}, expected {expected}.")
        elif resolved_shape == interior_shape:
            component_profile = torch.zeros(
                plane_shape,
                device=target_device,
                dtype=profile_dtype,
            )
            component_profile[1:-1, 1:-1] = resolved_profile
            resolved_profile = component_profile
        elif resolved_shape != plane_shape:
            raise RuntimeError(
                f"Full-vector ModeSource {name} profile shape {resolved_shape} does not match "
                f"the mode plane {plane_shape} or operator interior {interior_shape}."
            )
        component_profiles[name] = resolved_profile.contiguous()
    reference_profile = next(iter(component_profiles.values()))
    for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        component_profiles.setdefault(field_name, torch.zeros(plane_shape, device=target_device, dtype=profile_dtype)
                                      if staggered else torch.zeros_like(reference_profile))
    if int(source["direction_sign"]) < 0:
        for field_name in field_names[2:]:
            component_profiles[field_name] = -component_profiles[field_name]
    component_profiles = _normalize_mode_profiles_to_unit_power(
        component_profiles,
        coords_u=tangential_coord_map[axis_u][u_lo : u_hi + 1].to(
            device=target_device, dtype=target_dtype
        ),
        coords_v=tangential_coord_map[axis_v][v_lo : v_hi + 1].to(
            device=target_device, dtype=target_dtype
        ),
        normal_axis=normal_axis,
        staggered=staggered,
    )

    lower = [0, 0, 0]
    upper = [0, 0, 0]
    normal_axis_index = _AXIS_TO_INDEX[normal_axis]
    lower[_AXIS_TO_INDEX[axis_u]] = int(u_lo)
    upper[_AXIS_TO_INDEX[axis_u]] = int(u_hi)
    lower[_AXIS_TO_INDEX[axis_v]] = int(v_lo)
    upper[_AXIS_TO_INDEX[axis_v]] = int(v_hi)
    if int(source["direction_sign"]) > 0:
        lower[normal_axis_index] = int(plane_index)
        upper[normal_axis_index] = int(plane_index) + 1
    else:
        lower[normal_axis_index] = int(plane_index) - 1
        upper[normal_axis_index] = int(plane_index)

    return {
        "normal_axis": normal_axis,
        "tangential_axes": tangential_axes,
        "coords_u": tangential_coord_map[axis_u][u_lo : u_hi + 1].to(device=target_device, dtype=target_dtype).contiguous(),
        "coords_v": tangential_coord_map[axis_v][v_lo : v_hi + 1].to(device=target_device, dtype=target_dtype).contiguous(),
        "profile": component_profiles[preferred_field_name],
        "profile_component": preferred_field_name,
        "component_profiles": component_profiles,
        "mode_solver_kind": solver_kind,
        "staggered_profiles": staggered,
        "native_transverse_metrics": bool(native_tem and (du is None or any(
            not np.allclose(np.diff(coords),np.diff(coords)[0],rtol=1e-10,atol=0.)
            for coords in tem_coordinates.values()))),
        "effective_index": float(effective_index_value),
        "beta": float(beta_real),
        "beta_complex": complex(beta_value.detach()) if complex_tem else None,
        "effective_index_complex": complex(beta_value.detach()) / k0 if complex_tem else None,
        "effective_index_tensor": beta_value / k0 if solver_kind == "vector_dense_torch" else None,
        "beta_tensor": beta_value if complex_tem or solver_kind == "vector_dense_torch" else None,
        "pec_node_count": int(pec_node_count),
        "candidate_diagnostics": candidate_diagnostics["candidates"],
        "candidate_overlap_matrix": candidate_diagnostics["overlap_matrix"],
        "selected_candidate_index": candidate_diagnostics["selected_candidate_index"],
        "plane_index": int(plane_index),
        "box_lower": tuple(int(value) for value in lower),
        "box_upper": tuple(int(value) for value in upper),
        "bend_radius": source.get("bend_radius"),
        "bend_axis": source.get("bend_axis"),
    }


def _assemble_terminal_line_mode(source, terminal_mode, pec_occupancy, *, tensor_eps_planes, tensor_mu_planes,
                                 k0, du, dv, field_names, tem_coordinates):
    """Quasi-TEM mode of a multi-conductor TerminalWavePort aperture.

    ``terminal_mode`` carries the plane-local node masks of every terminal and of
    the reference, plus either an explicit modal potential vector or ``None`` to
    request the full terminal basis (capacitance matrices and modal potentials),
    from which column ``mode_index`` is then solved.
    """
    if source.get("wave_family") != "tem":
        raise ValueError("A TerminalWavePort mode must use the quasi-TEM family.")
    if any(component.is_complex() and bool(torch.any(torch.imag(component) != 0))
           for component in (*tensor_eps_planes, *tensor_mu_planes)):
        raise NotImplementedError(
            "TerminalWavePort quasi-TEM modes require a lossless cross-section; lossy or "
            "complex media need a complex multi-conductor potential solve."
        )
    if not all(bool(np.allclose(_real_plane_numpy(component), 1.0)) for component in tensor_mu_planes):
        raise NotImplementedError(
            "TerminalWavePort quasi-TEM modes require a nonmagnetic cross-section; the "
            "vacuum-capacitance inductance model does not include magnetic media."
        )
    if pec_occupancy is None:
        raise ValueError("A TerminalWavePort aperture contains no conductor.")
    label_map = _terminal_conductor_labels(
        pec_occupancy,
        terminal_masks=terminal_mode["terminal_masks"],
        reference_mask=terminal_mode["reference_mask"],
        plane_shape=terminal_mode["plane_shape"],
    )
    eps_uu, eps_vv, _ = tensor_eps_planes
    eps_node = 0.5 * (torch.real(eps_uu) + torch.real(eps_vv))
    basis = None
    potentials = terminal_mode.get("potentials")
    if potentials is None:
        basis = _solve_terminal_line_basis(eps_node.detach(), label_map, du=du, dv=dv, **tem_coordinates)
        potentials = tuple(float(value) for value in basis["modal_potentials"][:, int(source["mode_index"])])
    if len(potentials) != len(terminal_mode["terminal_masks"]):
        raise ValueError("TerminalWavePort modal potentials must contain one value per terminal.")
    beta, profiles, conductor_count, eps_eff, wall_fraction = _solve_terminal_line_mode(
        tensor_eps_planes, label_map, potentials, k0=k0, du=du, dv=dv,
        field_names=field_names, **tem_coordinates)
    record = {
        "potentials": tuple(float(value) for value in potentials),
        "eps_eff": float(eps_eff),
        "wall_charge_fraction": float(wall_fraction),
        "label_map": torch.as_tensor(label_map, dtype=torch.int64),
    }
    if basis is not None:
        record["basis"] = basis
    return beta, profiles, conductor_count, record


def solve_mode_source_profile(solver, source) -> dict[str, object]:
    from .conductors import (copper_intersects_mode, prepare_good_conductor_context,
                             apply_good_conductor_perturbation)
    if not getattr(solver,"_good_conductor_context",False):
        if source.get("conductor_model") == "good_conductor":
            context=prepare_good_conductor_context(solver,source)
            data=solve_mode_source_profile(context,source)
            return apply_good_conductor_perturbation(context,source,data)
        if copper_intersects_mode(solver.scene,source):
            raise NotImplementedError("A modal aperture intersecting CopperFoil requires explicit WaveModeSpec(conductor_model='good_conductor') within its applicability contract, or a finite-Et modal solve.")
    if source.get("injection", {}).get("kind") != "soft":
        raise ValueError("ModeSource currently supports soft injection only.")
    source_time = source["source_time"]
    if solver.scene.boundary.uses_kind("periodic") or solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError(
            "ModeSource mode solving is undefined under a periodic/Bloch transverse boundary: "
            "the aperture wraps with the Bloch phase, so the guided profile is an eigenfunction "
            "of a complex-Hermitian Bloch-phase transverse operator, whereas this solver builds "
            "the real, zero-boundary transverse operator. Use none/pml/pec/pmc on the mode plane."
        )

    normal_axis, tangential_axes = _mode_source_axes(source)
    normal_axis_index = _AXIS_TO_INDEX[normal_axis]
    plane_index = nearest_index(getattr(solver.scene, normal_axis), float(source["position"][normal_axis_index]))
    _validate_mode_source_position(solver.scene, source, plane_index=plane_index)

    field_name = _mode_source_field_name(source)
    frequency = float(source_time["frequency"])

    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    rebuild_from_fields = bool(getattr(solver, "_mode_source_rebuild_from_fields", False))
    vector_state = getattr(solver, "_mode_source_vector_state", {}).get(source["name"], {})
    if compiled_material_model is not None and rebuild_from_fields:
        pec_coord_map = {
            axis: _mode_source_node_axis_coords(solver.scene, axis)
            for axis in tangential_axes
        }
        pec_bounds, _ = _resolve_tangential_bounds(
            solver.scene,
            source,
            axis_coords_by_axis=pec_coord_map,
        )
        pec_occupancy = _mode_source_pec_slice(
            solver,
            normal_axis=normal_axis,
            plane_index=plane_index,
            tangential_bounds=pec_bounds,
        )
        if pec_occupancy is not None and bool(
            torch.any(pec_occupancy[1:-1, 1:-1] >= _PEC_OCCUPANCY_THRESHOLD)
        ):
            raise NotImplementedError(
                "Differentiable ModeSource rebuilding with an internal PEC conductor "
                "requires a PEC mode-shape eigen-adjoint."
            )

    if (
        compiled_material_model is not None
        and (not rebuild_from_fields or vector_state.get("trainable", False))
    ):
        node_coord_map = {
            axis: _mode_source_node_axis_coords(solver.scene, axis)
            for axis in tangential_axes
        }
        node_tangential_bounds, tangential_axes = _resolve_tangential_bounds(
            solver.scene,
            source,
            axis_coords_by_axis=node_coord_map,
        )
        if rebuild_from_fields and vector_state.get("trainable", False):
            eps_node_by_axis, mu_node_by_axis = solver._source_vector_materials[source["name"]]
        else:
            # Retain the exact forward slices before detaching the native launch.
            # Preparation also rebuilds source terms before the backward guard runs.
            with torch.enable_grad():
                eps_node_by_axis, mu_node_by_axis = _mode_source_relative_material_slices(
                    solver, frequency=frequency, normal_axis=normal_axis,
                    plane_index=plane_index, tangential_bounds=node_tangential_bounds)
                trainable = _mode_source_trainable_plane(
                    solver, source, materials=(eps_node_by_axis, mu_node_by_axis))
            if not hasattr(solver, "_mode_source_vector_state"):
                solver._mode_source_vector_state = {}
            vector_state = solver._mode_source_vector_state.setdefault(source["name"], {})
            vector_state["trainable"] = trainable
            if trainable:
                if not hasattr(solver, "_source_vector_materials"):
                    solver._source_vector_materials = {}
                solver._source_vector_materials[source["name"]] = (eps_node_by_axis, mu_node_by_axis)
        if not torch.is_grad_enabled():
            # Retaining the compiler graph must not turn native launch slices
            # into grad-bearing views (notably on fixed PEC apertures).
            eps_node_by_axis = {axis: value.detach() for axis, value in eps_node_by_axis.items()}
            mu_node_by_axis = {axis: value.detach() for axis, value in mu_node_by_axis.items()}
        eps_node_imag = _max_component_imag(eps_node_by_axis)
        mu_node_imag = _max_component_imag(mu_node_by_axis)
        if vector_state.get("trainable", False):
            _require_lossless_vector_planes(tuple(eps_node_by_axis.values()),
                                            tuple(mu_node_by_axis.values()), source_name=source["name"])
        pec_occupancy = _mode_source_pec_slice(
            solver,
            normal_axis=normal_axis,
            plane_index=plane_index,
            tangential_bounds=node_tangential_bounds,
        )
        has_interior_pec = pec_occupancy is not None and bool(
            torch.any(pec_occupancy[1:-1, 1:-1] >= _PEC_OCCUPANCY_THRESHOLD)
        )
        if has_interior_pec and (eps_node_imag > 1.0e-7 or mu_node_imag > 1.0e-7):
            if any(value.requires_grad for value in (*eps_node_by_axis.values(), *mu_node_by_axis.values())):
                raise NotImplementedError("Differentiable lossy TEM material profiles require a complex mode-shape adjoint.")
            if source.get("wave_family") != "tem":
                raise NotImplementedError(
                    "Lossy PEC-aware guided modes require a device-resident complex vector eigensolve.")
            if any(compiled_material_model.get(prefix + kind + "_poles", ())
                   for prefix in ("", "mu_") for kind in ("debye", "drude", "lorentz")):
                raise NotImplementedError("Lossy TEM currently requires static conductivity; native pole projection is not yet implemented.")
            dt = getattr(solver, "dt", None)
            if dt is None:
                dt = solver.resolve_mode_time_step(frequency)
            angle = math.pi * frequency * dt
            correction = angle / math.tan(angle)
            # Frequency material evaluation uses exp(+iwt); native FDTD spectra
            # use exp(-iwt), with conductivity sampled at the electric midpoint.
            native = lambda values: {axis: torch.complex(value.real, -(value.imag if value.is_complex() else torch.zeros_like(value)) * correction)
                                     for axis, value in values.items()}
            return _assemble_vector_mode_data(solver, source, normal_axis=normal_axis,
                tangential_axes=tangential_axes, tangential_bounds=node_tangential_bounds,
                tangential_coord_map=node_coord_map, plane_index=plane_index,
                eps_by_axis=native(eps_node_by_axis), mu_by_axis=native(mu_node_by_axis),
                frequency=frequency, pec_occupancy=pec_occupancy, tem_time_step=dt)
        if has_interior_pec and any(
            component.requires_grad
            for component in (*eps_node_by_axis.values(), *mu_node_by_axis.values())
        ):
            raise NotImplementedError(
                "Differentiable material parameters with an internal PEC conductor require "
                "a PEC mode-shape eigen-adjoint."
            )
        # A lossy (complex) permittivity makes the full-vector operator non-Hermitian and
        # its forward path solves only the real part, which would silently drop the loss.
        # Route those planes to the scalar complex-mode solve below, which resolves the
        # complex propagation constant directly.
        if eps_node_imag <= 1e-7 and mu_node_imag <= 1e-7:
            if _min_component_real(eps_node_by_axis) <= 0.0 or _min_component_real(mu_node_by_axis) <= 0.0:
                raise ValueError("ModeSource requires positive epsilon and mu on the source plane.")
            reference_slice = eps_node_by_axis[normal_axis]
            node_unknowns = max((int(reference_slice.shape[0]) - 2) * (int(reference_slice.shape[1]) - 2), 0)
            if has_interior_pec or _vector_mode_supported(
                solver,
                eps_by_axis=eps_node_by_axis,
                mu_by_axis=mu_node_by_axis,
                unknowns=node_unknowns,
            ):
                return _assemble_vector_mode_data(
                    solver,
                    source,
                    normal_axis=normal_axis,
                    tangential_axes=tangential_axes,
                    tangential_bounds=node_tangential_bounds,
                    tangential_coord_map=node_coord_map,
                    plane_index=plane_index,
                    eps_by_axis=eps_node_by_axis,
                    mu_by_axis=mu_node_by_axis,
                    frequency=frequency,
                    pec_occupancy=pec_occupancy,
                )

    tangential_coord_map = {
        axis: _field_component_axis_coords(solver.scene, field_name, axis)
        for axis in tangential_axes
    }
    tangential_bounds, tangential_axes = _resolve_tangential_bounds(
        solver.scene,
        source,
        axis_coords_by_axis=tangential_coord_map,
    )

    eps_component = _mode_source_component_permittivity(solver, source, frequency=frequency)
    eps_slice = _mode_slice(
        eps_component,
        axis=normal_axis,
        plane_index=plane_index,
        tangential_bounds=tangential_bounds,
    )
    bend_factor = _bend_conformal_factor(
        source,
        tangential_axes=tangential_axes,
        tangential_bounds=tangential_bounds,
        coord_map=tangential_coord_map,
    )
    eps_slice = _apply_bend_factor(eps_slice, bend_factor)
    if (
        getattr(solver, "_compiled_material_model", None) is not None
        and not getattr(solver, "_mode_source_rebuild_from_fields", False)
    ):
        mu_slice = _mode_slice(
            solver._compiled_material_model["mu_r"],
            axis=normal_axis,
            plane_index=plane_index,
            tangential_bounds=tangential_bounds,
        )
    else:
        mu_slice = torch.ones_like(eps_slice)

    eps_max_imag = float(torch.max(torch.abs(torch.imag(eps_slice))).item()) if torch.is_complex(eps_slice) else 0.0
    mu_max_imag = float(torch.max(torch.abs(torch.imag(mu_slice))).item()) if torch.is_complex(mu_slice) else 0.0
    eps_is_lossy = eps_max_imag > 1e-7 or mu_max_imag > 1e-7

    eps_real = torch.real(eps_slice) if torch.is_complex(eps_slice) else eps_slice
    mu_real = torch.real(mu_slice) if torch.is_complex(mu_slice) else mu_slice
    if torch.min(eps_real).item() <= 0.0 or torch.min(mu_real).item() <= 0.0:
        raise ValueError("ModeSource requires positive epsilon and mu on the source plane.")
    if not torch.allclose(mu_real, torch.ones_like(mu_real), atol=1e-6, rtol=0.0):
        raise NotImplementedError(
            "ModeSource scalar eigenmode solve is undefined for a magnetic (mu_r != 1) plane: "
            "the scalar Helmholtz reduction folds mu into a single index and scales the magnetic "
            "profile by n_eff/eta0, both of which assume a non-magnetic medium. A magnetic plane "
            "must be solved with the full-vector operator, which carries mu per component."
        )
    if eps_is_lossy and eps_real.requires_grad:
        raise NotImplementedError(
            "Differentiable ModeSource gradients through a lossy (complex) permittivity require a "
            "non-Hermitian eigen-adjoint; the differentiable path solves the Hermitian "
            "real-permittivity problem."
        )

    axis_u, axis_v = tangential_axes
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    coords_u = tangential_coord_map[axis_u][u_lo : u_hi + 1]
    coords_v = tangential_coord_map[axis_v][v_lo : v_hi + 1]
    du = _local_uniform_plane_spacing(solver.scene, axis_u, u_lo, u_hi)
    dv = _local_uniform_plane_spacing(solver.scene, axis_v, v_lo, v_hi)
    k0 = 2.0 * math.pi * frequency / float(solver.c)
    unknowns = max((int(eps_real.shape[0]) - 2) * (int(eps_real.shape[1]) - 2), 0)

    effective_index_complex = None
    beta_complex = None
    if eps_is_lossy:
        eps_complex_np = eps_slice.detach().cpu().numpy().astype(np.complex128, copy=False)
        mu_complex_np = (
            mu_slice.detach().cpu().numpy().astype(np.complex128, copy=False)
            if torch.is_complex(mu_slice)
            else mu_real.detach().cpu().numpy().astype(np.complex128, copy=False)
        )
        index_sq_complex = (k0 * k0) * (eps_complex_np * mu_complex_np)
        operator, interior_u, interior_v = _build_scalar_operator(index_sq_complex, du, dv)
        beta_sq_complex, eigenvector = _solve_mode_eigenpair_complex(operator, mode_index=int(source["mode_index"]))
        profile_np = _real_profile_from_complex_eigenvector(
            eigenvector,
            interior_u=interior_u,
            interior_v=interior_v,
            shape=tuple(int(dim) for dim in eps_real.shape),
        )
        profile = torch.as_tensor(profile_np, device=eps_real.device, dtype=torch.float64)
        beta_complex = complex(np.sqrt(beta_sq_complex))
        if beta_complex.real < 0.0:
            beta_complex = -beta_complex
        effective_index_complex = beta_complex / max(k0, 1e-30)
        beta_value = float(beta_complex.real)
        effective_index_value = float(effective_index_complex.real)
        beta_tensor = None
        effective_index_tensor = None
        scalar_solver_kind = "scalar_complex_dense" if unknowns <= _DENSE_EIGEN_LIMIT else "scalar_complex_sparse"
    elif eps_real.requires_grad:
        index_sq = ((k0 * k0) * (eps_real * mu_real)).to(dtype=torch.float64)
        if unknowns > _DENSE_EIGEN_LIMIT:
            beta_sq, eigenvector = _solve_mode_eigenpair_torch_sparse_implicit(
                index_sq,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
            )
            interior_u = int(index_sq.shape[0]) - 2
            interior_v = int(index_sq.shape[1]) - 2
        else:
            operator, interior_u, interior_v = _build_scalar_operator_torch_dense(index_sq, du, dv)
            beta_sq, eigenvector = _solve_mode_eigenpair_torch(operator, mode_index=int(source["mode_index"]))
        profile = torch.zeros_like(index_sq)
        profile[1:-1, 1:-1] = eigenvector.reshape((interior_u, interior_v))
        profile = _normalize_profile_torch(profile)
        beta_tensor = torch.sqrt(torch.clamp(beta_sq, min=0.0))
        effective_index_tensor = beta_tensor / max(k0, 1e-30)
        beta_value = float(beta_tensor.detach().item())
        effective_index_value = float(effective_index_tensor.detach().item())
    else:
        index_sq_torch = ((k0 * k0) * (eps_real * mu_real)).detach().to(dtype=torch.float64)
        if unknowns > _DENSE_EIGEN_LIMIT:
            operator, interior_u, interior_v = _build_scalar_operator_torch_sparse(index_sq_torch, du, dv)
            beta_sq_tensor, eigenvector = _solve_mode_eigenpair_torch_sparse(
                operator,
                interior_u=interior_u,
                interior_v=interior_v,
                mode_index=int(source["mode_index"]),
            )
            profile = torch.zeros_like(index_sq_torch)
            profile[1:-1, 1:-1] = eigenvector.reshape((interior_u, interior_v))
            profile = _normalize_profile_torch(profile)
            beta_value = math.sqrt(max(float(beta_sq_tensor.detach().item()), 0.0))
            scalar_solver_kind = "scalar_sparse"
        else:
            index_sq = index_sq_torch.cpu().numpy().astype(np.float64, copy=False)
            operator, interior_u, interior_v = _build_scalar_operator(index_sq, du, dv)
            beta_sq_value, eigenvector = _solve_mode_eigenpair(operator, mode_index=int(source["mode_index"]))
            profile_np = np.zeros_like(index_sq, dtype=np.float64)
            profile_np[1:-1, 1:-1] = eigenvector.reshape((interior_u, interior_v))
            peak = float(np.max(np.abs(profile_np)))
            if peak <= 0.0:
                raise RuntimeError("ModeSource eigenmode solve returned a zero profile.")
            profile_np /= peak
            peak_index = np.unravel_index(np.argmax(np.abs(profile_np)), profile_np.shape)
            if profile_np[peak_index] < 0.0:
                profile_np *= -1.0
            profile = torch.as_tensor(profile_np, device=eps_real.device, dtype=torch.float64)
            beta_value = math.sqrt(max(beta_sq_value, 0.0))
            scalar_solver_kind = "scalar_dense"
        effective_index_value = beta_value / max(k0, 1e-30)
        beta_tensor = None
        effective_index_tensor = None
    if not eps_is_lossy and eps_real.requires_grad:
        scalar_solver_kind = "scalar_sparse_implicit" if unknowns > _DENSE_EIGEN_LIMIT else "scalar_dense_torch"

    direction_vector = tuple(float(component) for component in source["direction_vector"])
    electric_vector = {
        name: float(component)
        for name, component in zip(_FIELD_COMPONENTS, source["polarization"])
    }
    magnetic_tuple = tuple(-value for value in _cross(direction_vector, source["polarization"]))
    magnetic_scale = (
        effective_index_tensor / ETA_0
        if effective_index_tensor is not None
        else effective_index_value / ETA_0
    )

    lower = [0, 0, 0]
    upper = [0, 0, 0]
    lower[_AXIS_TO_INDEX[axis_u]] = int(u_lo)
    upper[_AXIS_TO_INDEX[axis_u]] = int(u_hi)
    lower[_AXIS_TO_INDEX[axis_v]] = int(v_lo)
    upper[_AXIS_TO_INDEX[axis_v]] = int(v_hi)
    if int(source["direction_sign"]) > 0:
        lower[normal_axis_index] = int(plane_index)
        upper[normal_axis_index] = int(plane_index) + 1
    else:
        lower[normal_axis_index] = int(plane_index) - 1
        upper[normal_axis_index] = int(plane_index)

    target_device = torch.device(getattr(solver, "device", solver.scene.device))
    target_dtype = solver.Ex.dtype
    resolved_magnetic_vector = {}
    for name, component in zip(("Hx", "Hy", "Hz"), magnetic_tuple):
        value = magnetic_scale * float(component)
        if isinstance(value, torch.Tensor):
            resolved_magnetic_vector[name] = value.to(device=target_device, dtype=target_dtype)
        else:
            resolved_magnetic_vector[name] = float(value)
    component_profiles = {
        name: torch.zeros_like(profile.to(device=target_device, dtype=target_dtype))
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    scalar_profile = profile.to(device=target_device, dtype=target_dtype).contiguous()
    electric_floor = _component_roundoff_floor(electric_vector.values(), dtype=torch.float64)
    for name, component in electric_vector.items():
        if abs(float(component)) > electric_floor:
            component_profiles[name] = float(component) * scalar_profile
    magnetic_floor = _component_roundoff_floor(
        resolved_magnetic_vector.values(),
        dtype=torch.float64,
    )
    for name, component in resolved_magnetic_vector.items():
        if isinstance(component, torch.Tensor):
            if float(component.abs().max().item()) > magnetic_floor:
                component_profiles[name] = component * scalar_profile
        elif abs(float(component)) > magnetic_floor:
            component_profiles[name] = float(component) * scalar_profile

    component_profiles = _normalize_mode_profiles_to_unit_power(
        component_profiles,
        coords_u=coords_u.to(device=target_device, dtype=target_dtype),
        coords_v=coords_v.to(device=target_device, dtype=target_dtype),
        normal_axis=normal_axis,
    )
    scalar_profile = component_profiles[field_name]

    return {
        "normal_axis": normal_axis,
        "tangential_axes": tangential_axes,
        "coords_u": coords_u.to(device=target_device, dtype=target_dtype).contiguous(),
        "coords_v": coords_v.to(device=target_device, dtype=target_dtype).contiguous(),
        "profile": scalar_profile,
        "profile_component": field_name,
        "component_profiles": component_profiles,
        "electric_vector": electric_vector,
        "magnetic_vector": resolved_magnetic_vector,
        "mode_solver_kind": scalar_solver_kind,
        "effective_index": float(effective_index_value),
        "beta": float(beta_value),
        "effective_index_complex": None if effective_index_complex is None else complex(effective_index_complex),
        "beta_complex": None if beta_complex is None else complex(beta_complex),
        "effective_index_tensor": (
            None
            if effective_index_tensor is None
            else effective_index_tensor.to(device=target_device, dtype=target_dtype)
        ),
        "beta_tensor": (
            None
            if beta_tensor is None
            else beta_tensor.to(device=target_device, dtype=target_dtype)
        ),
        "plane_index": int(plane_index),
        "box_lower": tuple(int(value) for value in lower),
        "box_upper": tuple(int(value) for value in upper),
        "bend_radius": source.get("bend_radius"),
        "bend_axis": source.get("bend_axis"),
    }


def sample_mode_source_component(mode_data, positions: torch.Tensor, field_name: str) -> torch.Tensor:
    axis_u, axis_v = mode_data["tangential_axes"]
    sample_positions = torch.stack(
        (
            positions[..., _AXIS_TO_INDEX[axis_u]],
            positions[..., _AXIS_TO_INDEX[axis_v]],
        ),
        dim=-1,
    )
    component_profiles = mode_data.get("component_profiles", {})
    profile = component_profiles.get(field_name)
    if profile is None:
        reference = mode_data["profile"]
        return torch.zeros_like(
            _bilinear_sample(
                reference,
                mode_data["coords_u"],
                mode_data["coords_v"],
                sample_positions,
            )
        )
    coords_u, coords_v = mode_data["coords_u"], mode_data["coords_v"]
    if mode_data.get("staggered_profiles", False):
        if field_name in (f"E{axis_u}", f"H{axis_v}"):
            coords_u = .5 * (coords_u[:-1] + coords_u[1:])
        elif field_name in (f"E{axis_v}", f"H{axis_u}"):
            coords_v = .5 * (coords_v[:-1] + coords_v[1:])
    return _bilinear_sample(profile, coords_u, coords_v, sample_positions)


def sample_mode_source_profile(mode_data, positions: torch.Tensor) -> torch.Tensor:
    field_name = mode_data.get("profile_component")
    if field_name is None:
        axis_u, axis_v = mode_data["tangential_axes"]
        sample_positions = torch.stack(
            (
                positions[..., _AXIS_TO_INDEX[axis_u]],
                positions[..., _AXIS_TO_INDEX[axis_v]],
            ),
            dim=-1,
        )
        return _bilinear_sample(
            mode_data["profile"],
            mode_data["coords_u"],
            mode_data["coords_v"],
            sample_positions,
        )
    return sample_mode_source_component(mode_data, positions, field_name)


def _gauge_vector_profiles(profiles, preferred_field_name, *, state=None):
    preferred = profiles[preferred_field_name].reshape(-1)
    peak_index = int(preferred.abs().argmax())
    peak_magnitude = preferred[peak_index].abs()
    gauge_index = peak_index
    if state is not None:
        previous = state.get("gauge")
        signature = (preferred_field_name, tuple(profiles[preferred_field_name].shape))
        # Keep the phase reference through competing peaks until it falls below
        # half the current peak; the Scene/SceneModule owns it across forwards.
        if previous is not None and previous[:2] == signature:
            if bool(preferred[previous[2]].abs() >= .5 * peak_magnitude):
                gauge_index = previous[2]
        state["gauge"] = (*signature, gauge_index)
    pivot = preferred[gauge_index]
    rotation = pivot.conj() / pivot.abs()
    return {name: value * rotation / peak_magnitude for name, value in profiles.items()}


def _solve_vector_mode_torch(eps_planes, mu_planes, *, source, k0, du, dv,
                             field_names, preferred_field_name, gauge_state=None):
    """One dense vector eigenproblem defines both the launch and its derivative."""
    _require_lossless_vector_planes(eps_planes, mu_planes, source_name=source['name'])
    eps_np = tuple(_real_plane_numpy(value) for value in eps_planes)
    mu_np = tuple(_real_plane_numpy(value) for value in mu_planes)
    uniform = _is_uniform_isotropic_vector_plane(eps_np, mu_np)
    yee = uniform and all(np.allclose(value, 1.) for value in mu_np)
    nu, nv = eps_planes[0].shape[0] - 2, eps_planes[0].shape[1] - 2
    size = (nu + 1) * nv + nu * (nv + 1) if yee else 4 * nu * nv
    if size > _VECTOR_GRAD_MATRIX_LIMIT:
        forward_kind = "vector_dense" if nu * nv <= _FULL_VECTOR_DENSE_LIMIT else "vector_sparse"
        raise NotImplementedError(
            f"ModeSource/ModePort {source['name']!r} plane material is trainable "
            "(material on the source plane requires grad): "
            f"forward mode solver {forward_kind!r} differs from replay 'scalar_real'. "
            "A matching mode-shape eigen-adjoint is required. "
            f"Dense vector cap n <= {_VECTOR_GRAD_MATRIX_LIMIT}; actual n={size}.")
    if yee:
        operator, meta, eps_stag = _build_yee_transverse_operator_torch(
            eps_planes, k0=k0, du=du, dv=dv, source_name=source['name'])
    else:
        operator, _, _ = _build_vector_operator_torch(
            eps_planes, mu_planes, k0=k0, du=du, dv=dv, source_name=source['name'])
    with torch.no_grad():
        values, vectors = torch.linalg.eig(operator)
    values_np, vectors_np = values.detach().cpu().numpy(), vectors.detach().cpu().numpy()
    if yee:
        detached_meta = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                         for key, value in meta.items()}
        beta_selected, _, diagnostics = _select_yee_transverse_mode_numpy(
            values_np, vectors_np, meta=detached_meta,
            eps_stag=tuple(value.detach().cpu().numpy() for value in eps_stag),
            mode_index=int(source['mode_index']), field_names=field_names,
            preferred_field_name=preferred_field_name, reject_spurious=False,
            wave_family=source.get('wave_family'))
        selected = int(np.argmin(abs(values_np - beta_selected**2)))
    else:
        beta_selected, _, _, diagnostics = _select_and_normalize_vector_mode_numpy(
            values_np, vectors_np, interior_u=nu, interior_v=nv,
            mode_index=int(source['mode_index']), field_names=field_names,
            preferred_field_name=preferred_field_name, operator=operator.detach().cpu().numpy(),
            eps_planes=eps_np, mu_planes=mu_np, k0=k0, du=du, dv=dv,
            reject_spurious=not uniform, wave_family=source.get('wave_family'))
        selected = int(np.argmin(abs(values_np - beta_selected)))
    distances = abs(values.detach() - values[selected].detach())
    distances[selected] = torch.inf
    relative_gap = float(distances.min() / values[selected].detach().abs())
    # Backward error is O(eps64 * ||A||_inf); eigenvector backward amplifies
    # this by 1/gap. Require its scale estimate below sqrt(eps64).
    delta = float(torch.finfo(torch.float64).eps**0.5 *
                  torch.linalg.matrix_norm(operator.detach(), ord=float('inf')) /
                  values[selected].detach().abs())
    if relative_gap <= delta:
        raise NotImplementedError(
            f"ModeSource/ModePort {source['name']!r} degenerate vector mode: "
            f"relative gap={relative_gap:.6g} <= delta={delta:.6g}; "
            "the eigenvector derivative is not defined.")
    value, vector = _IsolatedVectorEigenpair.apply(operator, values[selected], vectors[:, selected], source['name'])
    beta = value.sqrt() if yee else value
    if yee:
        profiles = _yee_reconstruct_node_profiles_torch(vector, beta=beta, meta=meta,
                                                       eps_stag=eps_stag, field_names=field_names)
    else:
        hu, hv, eu, ev = _split_vector_mode_components(vector, interior_u=nu, interior_v=nv, backend='torch')
        profiles = dict(zip(field_names, (eu, ev, hu, hv)))
    profiles = _gauge_vector_profiles(profiles, preferred_field_name, state=gauge_state)
    return beta, profiles, diagnostics
