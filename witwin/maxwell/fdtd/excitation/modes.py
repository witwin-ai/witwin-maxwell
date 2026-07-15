from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as scipy_sparse_linalg
from witwin.core.material import VACUUM_PERMITTIVITY

from ...compiler.materials import evaluate_material_components
from .spatial import physical_interior_indices
from .tfsf_common import nearest_index


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_FIELD_COMPONENTS = ("Ex", "Ey", "Ez")
_ETA0 = 376.730313668
_DENSE_EIGEN_LIMIT = 4096
_FULL_VECTOR_DENSE_LIMIT = 384
_LOBPCG_REQUEST_PADDING = 4
_LOBPCG_MAX_ITER = 200
_LOBPCG_TOL = 1.0e-8
_IMPLICIT_EIGEN_CG_MAX_ITER = 256
_IMPLICIT_EIGEN_CG_TOL = 1.0e-8
_VECTOR_EIGEN_REQUEST_PADDING = 4
_VECTOR_EIGS_MAX_ITER = 600
_VECTOR_EIGS_TOL = 1.0e-8
_PEC_OCCUPANCY_THRESHOLD = 0.5
_PEC_VECTOR_MATRIX_LIMIT = 4096
_VECTOR_DEGENERATE_RTOL = 5.0e-5
_VECTOR_DUPLICATE_BETA_RTOL = 1.0e-5
_VECTOR_DUPLICATE_OVERLAP_LIMIT = 0.99
_VECTOR_CHECKERBOARD_FRACTION_LIMIT = 0.35
_RIGHT_HANDED_TANGENTIAL_AXES = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def _cross(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _mode_source_axes(source) -> tuple[str, tuple[str, str]]:
    normal_axis = str(source["normal_axis"])
    tangential_axes = _RIGHT_HANDED_TANGENTIAL_AXES[normal_axis]
    return normal_axis, tangential_axes


def _mode_source_field_name(source) -> str:
    return f"E{str(source['polarization_axis']).lower()}"


def _validate_mode_source_position(scene, source, *, plane_index: int) -> None:
    normal_axis = str(source["normal_axis"])
    direction_sign = int(source["direction_sign"])
    physical_lo, physical_hi = physical_interior_indices(scene, normal_axis)
    if direction_sign > 0:
        if plane_index <= physical_lo or plane_index >= physical_hi:
            raise ValueError(
                "ModeSource position must lie at least one cell inside the non-PML region "
                "along the propagation axis."
            )
        return
    if plane_index <= physical_lo or plane_index >= physical_hi:
        raise ValueError(
            "ModeSource position must lie at least one cell inside the non-PML region "
            "along the propagation axis."
        )


def _resolve_tangential_bounds(scene, source, axis_coords_by_axis=None) -> tuple[tuple[int, int], tuple[str, str]]:
    _, tangential_axes = _mode_source_axes(source)
    bounds = []
    for axis in tangential_axes:
        axis_index = _AXIS_TO_INDEX[axis]
        axis_coords = getattr(scene, axis) if axis_coords_by_axis is None else axis_coords_by_axis[axis]
        half_span = 0.5 * float(source["size"][axis_index])
        lower_coord = float(source["position"][axis_index]) - half_span
        upper_coord = float(source["position"][axis_index]) + half_span
        lower_index = nearest_index(axis_coords, lower_coord)
        upper_index = nearest_index(axis_coords, upper_coord)
        if upper_index < lower_index:
            lower_index, upper_index = upper_index, lower_index
        physical_lo, physical_hi = physical_interior_indices(scene, axis)
        lower_limit = float(getattr(scene, axis)[physical_lo].item())
        upper_limit = float(getattr(scene, axis)[physical_hi].item())
        if lower_coord < lower_limit - 1e-12 or upper_coord > upper_limit + 1e-12:
            raise ValueError("ModeSource aperture must remain inside the non-PML region.")
        if upper_index - lower_index < 2:
            raise ValueError(
                "ModeSource aperture must span at least three grid nodes along each tangential axis."
            )
        bounds.append((int(lower_index), int(upper_index)))
    return (bounds[0], bounds[1]), tangential_axes


# Relative transverse-spacing spread at or below which the mode plane is treated
# as exactly uniform and the legacy scalar spacing is returned bit-for-bit.
_UNIFORM_SPACING_RTOL = 1e-6
# Maximum fractional transverse-spacing variation tolerated across a mode-plane
# aperture. The 2D finite-difference mode operator is assembled from a single
# du/dv; on a graded transverse grid a centered stencil evaluated with the mean
# spacing loses one order of accuracy and its leading relative error scales with
# the fractional spacing spread (d_max - d_min)/d_mean. 1e-2 keeps the induced
# effective-index / operator error at roughly the one-percent level.
_MODE_PLANE_SPACING_SPREAD_BOUND = 1e-2


def _local_uniform_plane_spacing(scene, axis: str, lower_index: int, upper_index: int) -> float:
    """Effective transverse spacing across a mode-plane aperture window.

    The 2D mode solver assembles its first/second-difference operators from a
    single transverse spacing. A perfectly uniform window (spread <=
    ``_UNIFORM_SPACING_RTOL`` relative) returns its exact spacing, bit-for-bit as
    before. A mildly graded window is accepted when its fractional spacing
    variation stays below ``_MODE_PLANE_SPACING_SPREAD_BOUND`` and the region-mean
    spacing is used; beyond that the finite-difference operator error is no longer
    controlled and the window is rejected with the predicted variation.
    """
    primal = {"x": scene.dx_primal64, "y": scene.dy_primal64, "z": scene.dz_primal64}[axis]
    window = primal[max(int(lower_index) - 1, 0) : min(int(upper_index) + 1, len(primal))]
    d_min = float(window.min())
    d_max = float(window.max())
    if (d_max - d_min) <= _UNIFORM_SPACING_RTOL * d_max:
        return d_min

    d_eff = float(window.mean())
    fractional_spread = (d_max - d_min) / d_eff
    if fractional_spread > _MODE_PLANE_SPACING_SPREAD_BOUND:
        raise ValueError(
            "ModeSource/ModeMonitor mode solving is too graded across the mode "
            f"plane along axis '{axis}': the fractional transverse-spacing "
            f"variation is {fractional_spread:.3e} (min={d_min:g}, max={d_max:g}), "
            f"above the bound {_MODE_PLANE_SPACING_SPREAD_BOUND:.0e}. Refine "
            f"GridSpec.custom for locally uniform grid spacing along axis '{axis}' "
            "across the mode plane, or move the plane."
        )
    return d_eff


def _field_component_axis_coords(scene, field_name: str, axis: str) -> torch.Tensor:
    if field_name == "Ex":
        mapping = {
            "x": scene.x_half,
            "y": scene.y,
            "z": scene.z,
        }
    elif field_name == "Ey":
        mapping = {
            "x": scene.x,
            "y": scene.y_half,
            "z": scene.z,
        }
    elif field_name == "Ez":
        mapping = {
            "x": scene.x,
            "y": scene.y,
            "z": scene.z_half,
        }
    else:
        raise ValueError(f"Unsupported ModeSource field component {field_name!r}.")
    return mapping[axis]


def _average_node_tensor_to_component(node_tensor: torch.Tensor, field_name: str) -> torch.Tensor:
    if field_name == "Ex":
        return (0.5 * (node_tensor[:-1, :, :] + node_tensor[1:, :, :])).contiguous()
    if field_name == "Ey":
        return (0.5 * (node_tensor[:, :-1, :] + node_tensor[:, 1:, :])).contiguous()
    if field_name == "Ez":
        return (0.5 * (node_tensor[:, :, :-1] + node_tensor[:, :, 1:])).contiguous()
    raise ValueError(f"Unsupported ModeSource field component {field_name!r}.")


def _mode_source_node_axis_coords(scene, axis: str) -> torch.Tensor:
    return getattr(scene, axis)


def _mode_source_relative_material_slices(
    solver,
    *,
    frequency: float,
    normal_axis: str,
    plane_index: int,
    tangential_bounds,
):
    """Per-axis diagonal eps/mu aperture slices for the full-vector mode solve.

    Returns ``(eps_by_axis, mu_by_axis)`` where each maps ``"x"/"y"/"z"`` to the
    diagonal permittivity/permeability component sliced on the source plane. The
    full-vector operator consumes the three components separately so a
    diagonal-anisotropic aperture is resolved with its true per-axis tensor
    instead of an isotropic average.
    """
    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is None:
        raise RuntimeError("Full-vector ModeSource currently requires solver._compiled_material_model.")

    eps_components, mu_components = evaluate_material_components(compiled_material_model, float(frequency))
    eps_by_axis = {
        axis: _mode_slice(eps_components[axis], axis=normal_axis, plane_index=plane_index, tangential_bounds=tangential_bounds)
        for axis in ("x", "y", "z")
    }
    mu_by_axis = {
        axis: _mode_slice(mu_components[axis], axis=normal_axis, plane_index=plane_index, tangential_bounds=tangential_bounds)
        for axis in ("x", "y", "z")
    }
    return eps_by_axis, mu_by_axis


def _mode_source_pec_slice(
    solver,
    *,
    normal_axis: str,
    plane_index: int,
    tangential_bounds,
) -> torch.Tensor | None:
    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is None:
        return None
    occupancy = compiled_material_model.get("pec_occupancy")
    if occupancy is None:
        return None
    return _mode_slice(
        occupancy,
        axis=normal_axis,
        plane_index=plane_index,
        tangential_bounds=tangential_bounds,
    )


def _max_component_imag(components: dict) -> float:
    """Largest imaginary magnitude across the per-axis diagonal aperture slices."""
    worst = 0.0
    for component in components.values():
        if torch.is_complex(component):
            worst = max(worst, float(torch.max(torch.abs(torch.imag(component))).item()))
    return worst


def _min_component_real(components: dict) -> float:
    """Smallest real part across the per-axis diagonal aperture slices."""
    return min(
        float(torch.min(torch.real(component) if torch.is_complex(component) else component).item())
        for component in components.values()
    )


def _build_staggered_first_differences_sparse(count: int, spacing: float):
    if count <= 0:
        raise ValueError("count must be > 0 for first-difference assembly.")
    diagonal = np.full((count,), -1.0 / float(spacing), dtype=np.float64)
    upper = np.full((max(count - 1, 0),), 1.0 / float(spacing), dtype=np.float64)
    forward = sparse.diags((diagonal, upper), offsets=(0, 1), shape=(count, count), format="csr")
    backward = -forward.transpose().tocsr()
    return forward, backward


def _is_uniform_isotropic_vector_plane(eps_planes, mu_planes) -> bool:
    eps_reference = np.asarray(eps_planes[0]).reshape(-1)[0]
    mu_reference = np.asarray(mu_planes[0]).reshape(-1)[0]
    return all(np.allclose(component, eps_reference) for component in eps_planes) and all(
        np.allclose(component, mu_reference) for component in mu_planes
    )


def _build_vector_operator_sparse(eps_planes, mu_planes, *, k0: float, du: float, dv: float):
    """Full-vector transverse mode operator for a diagonal-anisotropic plane.

    ``eps_planes`` / ``mu_planes`` are ``(uu, vv, ww)`` triples of 2D aperture
    slices, with ``uu``/``vv`` the in-plane (tangential) diagonal components and
    ``ww`` the plane-normal (propagation-axis) component. The placement follows
    the first-order transverse Maxwell system eliminated for the longitudinal
    fields: the normal permittivity ``eps_ww`` enters every H-block coupling
    through the eliminated ``E_w = (i / (omega eps0 eps_ww)) (d_u H_v - d_v H_u)``,
    the normal permeability ``mu_ww`` enters every E-block coupling through the
    eliminated ``H_w``, and each transverse self-term carries the matching in-plane
    component (``E_u`` sees ``eps_uu``, ``E_v`` sees ``eps_vv``, ``H_u`` sees
    ``mu_uu``, ``H_v`` sees ``mu_vv``). For an isotropic plane the three components
    coincide and this reduces to the scalar-``eps`` form bit-for-bit.
    """
    eps_uu, eps_vv, eps_ww = eps_planes
    mu_uu, mu_vv, mu_ww = mu_planes
    nu = int(eps_uu.shape[0])
    nv = int(eps_uu.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    uniform_isotropic = _is_uniform_isotropic_vector_plane(eps_planes, mu_planes)
    if uniform_isotropic:
        # In a homogeneous metallic guide the transverse vector formulation
        # carries complementary TE/TM boundary parities. The centered operator
        # preserves those parities; duplicate/checkerboard candidates are removed
        # later by the power-overlap and variation diagnostics.
        off_u = np.full((max(interior_u - 1, 0),), 0.5 / float(du), dtype=np.float64)
        off_v = np.full((max(interior_v - 1, 0),), 0.5 / float(dv), dtype=np.float64)
        d1_u_forward = sparse.diags((-off_u, off_u), offsets=(-1, 1), shape=(interior_u, interior_u), format="csr")
        d1_v_forward = sparse.diags((-off_v, off_v), offsets=(-1, 1), shape=(interior_v, interior_v), format="csr")
        d1_u_backward = d1_u_forward
        d1_v_backward = d1_v_forward
    else:
        d1_u_forward, d1_u_backward = _build_staggered_first_differences_sparse(interior_u, du)
        d1_v_forward, d1_v_backward = _build_staggered_first_differences_sparse(interior_v, dv)
    identity_u = sparse.eye(interior_u, format="csr", dtype=np.float64)
    identity_v = sparse.eye(interior_v, format="csr", dtype=np.float64)
    derivative_u_forward = sparse.kron(d1_u_forward, identity_v, format="csr")
    derivative_u_backward = sparse.kron(d1_u_backward, identity_v, format="csr")
    derivative_v_forward = sparse.kron(identity_u, d1_v_forward, format="csr")
    derivative_v_backward = sparse.kron(identity_u, d1_v_backward, format="csr")

    def _interior_diag(values: np.ndarray):
        return sparse.diags(
            np.asarray(values, dtype=np.float64)[1:-1, 1:-1].reshape(-1), offsets=0, format="csr"
        )

    eps_w_inv = _interior_diag(1.0 / np.asarray(eps_ww, dtype=np.float64))
    mu_w_inv = _interior_diag(1.0 / np.asarray(mu_ww, dtype=np.float64))
    eps_u_diag = _interior_diag(eps_uu)
    eps_v_diag = _interior_diag(eps_vv)
    mu_u_diag = _interior_diag(mu_uu)
    mu_v_diag = _interior_diag(mu_vv)
    k0_sq = float(k0) * float(k0)

    # Each eliminated curl uses the derivative on its own Yee half-grid and the
    # outer curl uses the negative-adjoint derivative. Reusing one centered
    # derivative for both curls composes a stride-two stencil, decoupling the
    # odd/even transverse sublattices and creating checkerboard mode copies.
    a_hu_hu = derivative_v_backward @ eps_w_inv @ derivative_v_forward + k0_sq * mu_u_diag
    a_hu_hv = -derivative_v_backward @ eps_w_inv @ derivative_u_forward
    a_hv_hu = -derivative_u_backward @ eps_w_inv @ derivative_v_forward
    a_hv_hv = derivative_u_backward @ eps_w_inv @ derivative_u_forward + k0_sq * mu_v_diag

    a_eu_eu = -derivative_v_forward @ mu_w_inv @ derivative_v_backward - k0_sq * eps_u_diag
    a_eu_ev = derivative_v_forward @ mu_w_inv @ derivative_u_backward
    a_ev_eu = derivative_u_forward @ mu_w_inv @ derivative_v_backward
    a_ev_ev = -derivative_u_forward @ mu_w_inv @ derivative_u_backward - k0_sq * eps_v_diag

    electric_scale = float(k0) / _ETA0
    magnetic_scale = float(k0) * _ETA0
    operator = sparse.bmat(
        [
            [None, None, a_ev_eu / magnetic_scale, a_ev_ev / magnetic_scale],
            [None, None, -a_eu_eu / magnetic_scale, -a_eu_ev / magnetic_scale],
            [a_hv_hu / electric_scale, a_hv_hv / electric_scale, None, None],
            [-a_hu_hu / electric_scale, -a_hu_hv / electric_scale, None, None],
        ],
        format="csr",
    )
    return operator, interior_u, interior_v


def _pec_first_difference(count: int, spacing: float, *, device: torch.device) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be > 0 for PEC mode first-difference assembly.")
    result = torch.zeros((count, count), device=device, dtype=torch.float64)
    if count > 1:
        indices = torch.arange(count - 1, device=device)
        coefficient = 0.5 / float(spacing)
        result[indices, indices + 1] = coefficient
        result[indices + 1, indices] = -coefficient
    return result


def _pec_vector_operator_torch(
    eps_planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mu_planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pec_occupancy: torch.Tensor,
    *,
    k0: float,
    du: float,
    dv: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Build a device-resident transverse operator with PEC nodes eliminated."""

    eps_uu, eps_vv, eps_ww = eps_planes
    mu_uu, mu_vv, mu_ww = mu_planes
    device = eps_uu.device
    tensors = (*eps_planes, *mu_planes, pec_occupancy)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("PEC mode material and occupancy tensors must share one device.")
    if pec_occupancy.requires_grad:
        raise NotImplementedError(
            "Differentiable PEC geometry requires a mode-shape eigen-adjoint; "
            "the PEC-aware forward mode solve uses a hard conductor boundary."
        )

    nu, nv = (int(value) for value in eps_uu.shape)
    if any(tuple(tensor.shape) != (nu, nv) for tensor in tensors):
        raise ValueError("PEC mode material and occupancy slices must have identical shapes.")
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    conductor = pec_occupancy >= _PEC_OCCUPANCY_THRESHOLD
    conductor_count = int(torch.count_nonzero(conductor[1:-1, 1:-1]).item())
    if conductor_count == 0:
        peak = float(torch.max(pec_occupancy).item())
        raise ValueError(
            "PEC structure is under-resolved on the mode plane: its maximum node "
            f"occupancy is {peak:.3f}, below the {_PEC_OCCUPANCY_THRESHOLD:g} boundary threshold."
        )
    active = (~conductor[1:-1, 1:-1]).reshape(-1)
    active_count = int(torch.count_nonzero(active).item())
    if active_count < 2:
        raise ValueError("PEC conductors leave fewer than two active mode-plane nodes.")

    d1_u = _pec_first_difference(interior_u, du, device=device)
    d1_v = _pec_first_difference(interior_v, dv, device=device)
    identity_u = torch.eye(interior_u, device=device, dtype=torch.float64)
    identity_v = torch.eye(interior_v, device=device, dtype=torch.float64)
    derivative_u = torch.kron(d1_u, identity_v)
    derivative_v = torch.kron(identity_u, d1_v)
    active_diagonal = torch.diag(active.to(dtype=torch.float64))
    derivative_u = active_diagonal @ derivative_u @ active_diagonal
    derivative_v = active_diagonal @ derivative_v @ active_diagonal

    def interior_diag(values: torch.Tensor) -> torch.Tensor:
        real = torch.real(values) if values.is_complex() else values
        return torch.diag(real[1:-1, 1:-1].reshape(-1).to(device=device, dtype=torch.float64))

    eps_w_inv = interior_diag(1.0 / eps_ww)
    mu_w_inv = interior_diag(1.0 / mu_ww)
    eps_u_diag = interior_diag(eps_uu)
    eps_v_diag = interior_diag(eps_vv)
    mu_u_diag = interior_diag(mu_uu)
    mu_v_diag = interior_diag(mu_vv)
    k0_sq = float(k0) * float(k0)

    a_hu_hu = derivative_v @ eps_w_inv @ derivative_v + k0_sq * mu_u_diag
    a_hu_hv = -derivative_v @ eps_w_inv @ derivative_u
    a_hv_hu = -derivative_u @ eps_w_inv @ derivative_v
    a_hv_hv = derivative_u @ eps_w_inv @ derivative_u + k0_sq * mu_v_diag
    a_eu_eu = -derivative_v @ mu_w_inv @ derivative_v - k0_sq * eps_u_diag
    a_eu_ev = derivative_v @ mu_w_inv @ derivative_u
    a_ev_eu = derivative_u @ mu_w_inv @ derivative_v
    a_ev_ev = -derivative_u @ mu_w_inv @ derivative_u - k0_sq * eps_v_diag

    zero = torch.zeros_like(a_hu_hu)
    electric_scale = float(k0) / _ETA0
    magnetic_scale = float(k0) * _ETA0
    operator = torch.cat(
        (
            torch.cat((zero, zero, a_ev_eu / magnetic_scale, a_ev_ev / magnetic_scale), dim=1),
            torch.cat((zero, zero, -a_eu_eu / magnetic_scale, -a_eu_ev / magnetic_scale), dim=1),
            torch.cat((a_hv_hu / electric_scale, a_hv_hv / electric_scale, zero, zero), dim=1),
            torch.cat((-a_hu_hu / electric_scale, -a_hu_hv / electric_scale, zero, zero), dim=1),
        ),
        dim=0,
    )
    component_active = active.repeat(4)
    reduced = operator[component_active][:, component_active].contiguous()
    if int(reduced.shape[0]) > _PEC_VECTOR_MATRIX_LIMIT:
        raise RuntimeError(
            "PEC-aware mode plane is too large for the device-resident dense eigensolve: "
            f"matrix size {int(reduced.shape[0])} exceeds {_PEC_VECTOR_MATRIX_LIMIT}. "
            "Use a coarser locally uniform mode-plane grid."
        )
    return reduced, active, interior_u, interior_v, conductor_count


def _vector_mode_request_count(matrix_size: int, *, mode_index: int) -> int:
    requested = max(2 * (int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING), 4)
    return min(requested, max(1, int(matrix_size) - 2))


def _select_vector_mode_numpy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    interior_u: int,
    interior_v: int,
    mode_index: int,
):
    order = np.lexsort((np.abs(np.imag(eigenvalues)), -np.real(eigenvalues)))
    positive = [index for index in order if np.isfinite(eigenvalues[index]) and np.real(eigenvalues[index]) > 0.0]
    if len(positive) <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {len(positive)} positive-beta modes were found."
        )
    selected = int(positive[int(mode_index)])
    beta = eigenvalues[selected]
    vector = eigenvectors[:, selected]
    if np.max(np.abs(np.imag(beta))) <= 1e-7:
        beta = float(np.real(beta))
    return beta, vector


def _select_vector_mode_torch(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    mode_index: int,
):
    real_values = torch.real(eigenvalues)
    imag_values = torch.imag(eigenvalues) if torch.is_complex(eigenvalues) else torch.zeros_like(real_values)
    order = torch.argsort(torch.abs(imag_values), stable=True)
    order = order[torch.argsort(real_values[order], descending=True, stable=True)]
    positive = [
        int(index)
        for index in order.tolist()
        if bool(torch.isfinite(real_values[index]).item())
        and bool(torch.isfinite(imag_values[index]).item())
        and float(real_values[index].item()) > 0.0
    ]
    if len(positive) <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {len(positive)} positive-beta modes were found."
        )
    selected = positive[int(mode_index)]
    return eigenvalues[selected], eigenvectors[:, selected]


def _split_vector_mode_components(
    eigenvector,
    *,
    interior_u: int,
    interior_v: int,
    backend: str,
):
    block = interior_u * interior_v
    if backend == "torch":
        hu = eigenvector[0:block].reshape((interior_u, interior_v))
        hv = eigenvector[block : 2 * block].reshape((interior_u, interior_v))
        eu = eigenvector[2 * block : 3 * block].reshape((interior_u, interior_v))
        ev = eigenvector[3 * block : 4 * block].reshape((interior_u, interior_v))
    else:
        hu = np.asarray(eigenvector[0:block]).reshape((interior_u, interior_v))
        hv = np.asarray(eigenvector[block : 2 * block]).reshape((interior_u, interior_v))
        eu = np.asarray(eigenvector[2 * block : 3 * block]).reshape((interior_u, interior_v))
        ev = np.asarray(eigenvector[3 * block : 4 * block]).reshape((interior_u, interior_v))
    return hu, hv, eu, ev


def _vector_preferred_component_name(source) -> str:
    return f"E{str(source['polarization_axis']).lower()}"


def _normalize_vector_mode_profiles_numpy(component_profiles: dict[str, np.ndarray], *, preferred_field_name: str):
    preferred = component_profiles.get(preferred_field_name)
    if preferred is None:
        raise RuntimeError(f"ModeSource full-vector solve did not produce preferred field {preferred_field_name}.")
    peak = float(np.max(np.abs(preferred)))
    if peak <= 0.0:
        electric_norm = np.sqrt(sum(np.abs(component_profiles[name]) ** 2 for name in component_profiles if name.startswith("E")))
        peak = float(np.max(electric_norm))
        if peak <= 0.0:
            raise RuntimeError("ModeSource full-vector eigenmode solve returned a zero electric field profile.")
        peak_index = np.unravel_index(np.argmax(electric_norm), electric_norm.shape)
        preferred_value = preferred[peak_index]
    else:
        peak_index = np.unravel_index(np.argmax(np.abs(preferred)), preferred.shape)
        preferred_value = preferred[peak_index]
    # Eigenvectors of the real mode operator carry an arbitrary global phase when
    # they come from a complex eigensolver (ARPACK); rotating by the conjugate
    # phase at the preferred-component peak recovers the real profile. For a real
    # eigenvector this reduces exactly to the previous +/-1 sign orientation.
    preferred_magnitude = abs(complex(preferred_value))
    if preferred_magnitude > 0.0:
        scale = np.conj(preferred_value) / (preferred_magnitude * peak)
    else:
        scale = 1.0 / peak
    for name in tuple(component_profiles):
        component_profiles[name] = scale * component_profiles[name]
    return component_profiles


def _vector_mode_power_sign_numpy(component_profiles: dict[str, np.ndarray]) -> float:
    tangential_e = [component_profiles[name] for name in component_profiles if name.startswith("E")]
    tangential_h = [component_profiles[name] for name in component_profiles if name.startswith("H")]
    if len(tangential_e) != 2 or len(tangential_h) != 2:
        return 0.0
    eu, ev = tangential_e
    hu, hv = tangential_h
    return float(np.sum(np.real(eu * np.conj(hv) - ev * np.conj(hu))))


def _vector_mode_polarization_fraction_numpy(
    component_profiles: dict[str, np.ndarray],
    *,
    preferred_field_name: str,
) -> float:
    """Return the requested tangential electric-field energy fraction."""
    electric_energy = sum(
        float(np.vdot(profile, profile).real)
        for name, profile in component_profiles.items()
        if name.startswith("E")
    )
    if electric_energy <= 0.0:
        return 0.0
    preferred = component_profiles[preferred_field_name]
    return float(np.vdot(preferred, preferred).real) / electric_energy


def _vector_mode_power_inner_product_numpy(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> complex:
    """Hermitian reciprocal-power product on one transverse Yee plane."""
    electric_names = [name for name in left if name.startswith("E")]
    magnetic_names = [name for name in left if name.startswith("H")]
    if len(electric_names) != 2 or len(magnetic_names) != 2:
        return 0.0j
    eu_l, ev_l = (left[name] for name in electric_names)
    hu_l, hv_l = (left[name] for name in magnetic_names)
    eu_r, ev_r = (right[name] for name in electric_names)
    hu_r, hv_r = (right[name] for name in magnetic_names)
    forward = eu_l * np.conj(hv_r) - ev_l * np.conj(hu_r)
    reciprocal = np.conj(eu_r) * hv_l - np.conj(ev_r) * hu_l
    return complex(0.25 * np.sum(forward + reciprocal))


def _vector_mode_checkerboard_fraction_numpy(component_profiles: dict[str, np.ndarray]) -> float:
    """Normalized one-cell electric-field variation, with a Nyquist mode near one."""
    electric_profiles = [profile for name, profile in component_profiles.items() if name.startswith("E")]
    energy = sum(float(np.vdot(profile, profile).real) for profile in electric_profiles)
    if energy <= 0.0:
        return 1.0
    variation = 0.0
    for profile in electric_profiles:
        variation += float(np.sum(np.abs(np.diff(profile, axis=0)) ** 2))
        variation += float(np.sum(np.abs(np.diff(profile, axis=1)) ** 2))
    return variation / (8.0 * energy)


def _relative_vector_residual(numerator, *terms) -> float:
    denominator = sum(float(np.linalg.norm(term)) for term in terms)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.linalg.norm(numerator)) / denominator


def _vector_mode_eigenpair_residual_numpy(operator, beta, eigenvector) -> float | None:
    if operator is None:
        return None
    applied = operator @ eigenvector
    return _relative_vector_residual(applied - beta * eigenvector, applied, beta * eigenvector)


def _vector_mode_divergence_residuals_numpy(
    eigenvector,
    *,
    beta,
    interior_u: int,
    interior_v: int,
    eps_planes,
    mu_planes,
    k0: float,
    du: float,
    dv: float,
) -> tuple[float | None, float | None]:
    if eps_planes is None or mu_planes is None or k0 is None or du is None or dv is None:
        return None, None

    hu, hv, eu, ev = _split_vector_mode_components(
        eigenvector,
        interior_u=interior_u,
        interior_v=interior_v,
        backend="numpy",
    )
    hu = hu.reshape(-1)
    hv = hv.reshape(-1)
    eu = eu.reshape(-1)
    ev = ev.reshape(-1)
    eps_u, eps_v, _ = (
        np.asarray(component, dtype=np.float64)[1:-1, 1:-1].reshape(-1)
        for component in eps_planes
    )
    mu_u, mu_v, _ = (
        np.asarray(component, dtype=np.float64)[1:-1, 1:-1].reshape(-1)
        for component in mu_planes
    )
    d1_u_forward, d1_u_backward = _build_staggered_first_differences_sparse(interior_u, du)
    d1_v_forward, d1_v_backward = _build_staggered_first_differences_sparse(interior_v, dv)
    identity_u = sparse.eye(interior_u, format="csr", dtype=np.float64)
    identity_v = sparse.eye(interior_v, format="csr", dtype=np.float64)
    derivative_u_forward = sparse.kron(d1_u_forward, identity_v, format="csr")
    derivative_u_backward = sparse.kron(d1_u_backward, identity_v, format="csr")
    derivative_v_forward = sparse.kron(identity_u, d1_v_forward, format="csr")
    derivative_v_backward = sparse.kron(identity_u, d1_v_backward, format="csr")

    electric_transverse = derivative_u_forward @ (eps_u * eu) + derivative_v_forward @ (eps_v * ev)
    electric_longitudinal = (beta / (float(k0) / _ETA0)) * (
        derivative_u_forward @ hv - derivative_v_forward @ hu
    )
    magnetic_transverse = derivative_u_backward @ (mu_u * hu) + derivative_v_backward @ (mu_v * hv)
    magnetic_longitudinal = (beta / (float(k0) * _ETA0)) * (
        derivative_u_backward @ ev - derivative_v_backward @ eu
    )
    electric_residual = _relative_vector_residual(
        electric_transverse - electric_longitudinal,
        electric_transverse,
        electric_longitudinal,
    )
    magnetic_residual = _relative_vector_residual(
        magnetic_transverse + magnetic_longitudinal,
        magnetic_transverse,
        magnetic_longitudinal,
    )
    return electric_residual, magnetic_residual


def _normalize_mode_profiles_to_unit_power(
    component_profiles: dict[str, torch.Tensor],
    *,
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
    normal_axis: str,
) -> dict[str, torch.Tensor]:
    """Scale a common-node modal profile to one watt on the transverse Yee grids."""
    power = _discrete_mode_profile_power(
        component_profiles,
        coords_u=coords_u,
        coords_v=coords_v,
        normal_axis=normal_axis,
    )
    abs_power = torch.abs(power)
    if float(abs_power.detach().item()) <= torch.finfo(abs_power.dtype).eps:
        raise RuntimeError("ModeSource eigenmode profile has zero integrated Poynting power.")
    scale = torch.rsqrt(abs_power)
    return {name: profile * scale for name, profile in component_profiles.items()}


def _discrete_mode_profile_power(
    component_profiles: dict[str, torch.Tensor],
    *,
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
    normal_axis: str,
) -> torch.Tensor:
    """Integrate signed modal power after interpolation to the transverse Yee grids."""
    if coords_u.ndim != 1 or coords_v.ndim != 1 or coords_u.numel() < 2 or coords_v.numel() < 2:
        raise ValueError("ModeSource profile coordinates must be one-dimensional with at least two samples.")
    expected_shape = (int(coords_u.numel()), int(coords_v.numel()))
    for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        profile = component_profiles[field_name]
        if tuple(int(dim) for dim in profile.shape) != expected_shape:
            raise ValueError(
                f"ModeSource {field_name} profile shape {tuple(profile.shape)} does not match "
                f"coordinate shape {expected_shape}."
            )

    axis_u, axis_v = _RIGHT_HANDED_TANGENTIAL_AXES[normal_axis]
    electric_u = component_profiles[f"E{axis_u}"]
    electric_v = component_profiles[f"E{axis_v}"]
    magnetic_u = component_profiles[f"H{axis_u}"]
    magnetic_v = component_profiles[f"H{axis_v}"]

    coords_u_half = 0.5 * (coords_u[:-1] + coords_u[1:])
    coords_v_half = 0.5 * (coords_v[:-1] + coords_v[1:])
    electric_u_half = 0.5 * (electric_u[:-1, :] + electric_u[1:, :])
    magnetic_v_half = 0.5 * (magnetic_v[:-1, :] + magnetic_v[1:, :])
    electric_v_half = 0.5 * (electric_v[:, :-1] + electric_v[:, 1:])
    magnetic_u_half = 0.5 * (magnetic_u[:, :-1] + magnetic_u[:, 1:])

    positive_density = 0.5 * torch.real(electric_u_half * torch.conj(magnetic_v_half))
    negative_density = 0.5 * torch.real(electric_v_half * torch.conj(magnetic_u_half))
    positive_power = torch.trapezoid(
        torch.trapezoid(positive_density, x=coords_v, dim=1),
        x=coords_u_half,
        dim=0,
    )
    negative_power = torch.trapezoid(
        torch.trapezoid(negative_density, x=coords_v_half, dim=1),
        x=coords_u,
        dim=0,
    )
    return positive_power - negative_power


def _select_and_normalize_vector_mode_numpy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    interior_u: int,
    interior_v: int,
    mode_index: int,
    field_names,
    preferred_field_name: str,
    operator=None,
    eps_planes=None,
    mu_planes=None,
    k0: float | None = None,
    du: float | None = None,
    dv: float | None = None,
    reject_spurious: bool = True,
):
    order = np.lexsort((np.abs(np.imag(eigenvalues)), -np.real(eigenvalues)))
    raw_candidates = []
    candidate_window = (
        max(2 * (int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING), 4)
        if reject_spurious
        else len(order)
    )
    for index in order:
        value = eigenvalues[index]
        if not np.isfinite(value) or np.real(value) <= 0.0:
            continue
        hu, hv, eu, ev = _split_vector_mode_components(
            eigenvectors[:, index],
            interior_u=interior_u,
            interior_v=interior_v,
            backend="numpy",
        )
        component_profiles = {
            field_names[0]: eu,
            field_names[1]: ev,
            field_names[2]: hu,
            field_names[3]: hv,
        }
        raw_candidates.append((index, value, component_profiles))
        if len(raw_candidates) >= candidate_window:
            break

    # A symmetric cross-section can produce an exactly degenerate polarization
    # pair.  Eigensolvers may return an arbitrary rotation of that subspace, so
    # selecting one raw vector makes the launched polarization jump with tiny
    # frequency or grid changes.  Diagonalize the requested-polarization energy
    # inside each degenerate subspace to obtain a deterministic polarized basis.
    independent_candidates = []
    cursor = 0
    while cursor < len(raw_candidates):
        group_stop = cursor + 1
        reference_value = raw_candidates[cursor][1]
        # Wide enough to capture ARPACK duplicate/conjugate-pair jitter and the
        # small numerical splitting of a symmetric polarization pair (observed
        # around 1e-5 relative), yet below distinct-mode spacings (>=1e-4
        # relative for the polarization pair of a mildly rectangular guide).
        degenerate_rtol = _VECTOR_DEGENERATE_RTOL if reject_spurious else 1.0e-7
        tolerance = degenerate_rtol * max(abs(reference_value), 1.0)
        while (
            group_stop < len(raw_candidates)
            and abs(raw_candidates[group_stop][1] - reference_value) <= tolerance
        ):
            group_stop += 1
        group = raw_candidates[cursor:group_stop]

        if len(group) > 1:
            # Sparse (ARPACK) solves can return duplicated eigenvectors and
            # complex-conjugate / arbitrarily phase-rotated copies of the same
            # real mode inside one degenerate group. The vector mode operator is
            # real here, so the group's invariant subspace has a real basis:
            # orthonormalize the stacked real and imaginary parts with an SVD
            # and keep only the numerically independent real directions. This
            # both removes duplicates and guarantees real rotated profiles.
            stacked = np.stack([eigenvectors[:, index] for index, _, _ in group], axis=1)
            stacked = np.concatenate([np.real(stacked), np.imag(stacked)], axis=1)
            basis, singular_values, _ = np.linalg.svd(stacked, full_matrices=False)
            rank = int(np.sum(singular_values > 1.0e-6 * singular_values[0]))
            raw_indices = tuple(int(entry[0]) for entry in group)
            group = []
            for column in range(rank):
                vector = basis[:, column]
                hu, hv, eu, ev = _split_vector_mode_components(
                    vector,
                    interior_u=interior_u,
                    interior_v=interior_v,
                    backend="numpy",
                )
                group.append(
                    (
                        raw_indices,
                        reference_value,
                        {
                            field_names[0]: eu,
                            field_names[1]: ev,
                            field_names[2]: hu,
                            field_names[3]: hv,
                        },
                        vector,
                    )
                )

        if len(group) == 1:
            entry = group[0]
            vector = entry[3] if len(entry) > 3 else eigenvectors[:, entry[0]]
            raw_indices = entry[0] if isinstance(entry[0], tuple) else (int(entry[0]),)
            rotated = [(raw_indices, entry[1], vector, entry[2])]
        else:
            count = len(group)
            preferred_gram = np.empty((count, count), dtype=np.complex128)
            electric_gram = np.empty((count, count), dtype=np.complex128)
            smoothness_gram = np.empty((count, count), dtype=np.complex128)
            for row, (_, _, row_profiles, _) in enumerate(group):
                for col, (_, _, col_profiles, _) in enumerate(group):
                    preferred_gram[row, col] = np.vdot(
                        row_profiles[preferred_field_name], col_profiles[preferred_field_name]
                    )
                    electric_gram[row, col] = sum(
                        np.vdot(row_profiles[name], col_profiles[name])
                        for name in row_profiles
                        if name.startswith("E")
                    )
                    smoothness_gram[row, col] = sum(
                        np.vdot(
                            np.diff(row_profiles[name], axis=axis),
                            np.diff(col_profiles[name], axis=axis),
                        )
                        for name in row_profiles
                        if name.startswith("E")
                        for axis in (0, 1)
                    )
            electric_scale = max(float(np.max(np.abs(np.diag(electric_gram)))), 1.0)
            smoothness_scale = max(
                float(np.max(np.abs(np.diag(smoothness_gram)))),
                np.finfo(np.float64).eps,
            )
            # Preferred-polarization energy can remain exactly degenerate between
            # centered-grid parity copies. A scale-free smoothness perturbation
            # orders that residual subspace deterministically without changing
            # distinct polarization families.
            preferred_gram -= 1.0e-9 * electric_scale * smoothness_gram / smoothness_scale
            electric_gram += np.eye(count) * (np.finfo(np.float64).eps * electric_scale)
            fractions, rotations = scipy_linalg.eigh(preferred_gram, electric_gram)
            rotated = []
            for rotation_index in np.argsort(fractions)[::-1]:
                coefficients = rotations[:, rotation_index]
                combined_vector = sum(
                    coefficient * vector
                    for coefficient, (_, _, _, vector) in zip(coefficients, group)
                )
                combined_profiles = {
                    name: sum(
                        coefficient * profiles[name]
                        for coefficient, (_, _, profiles, _) in zip(coefficients, group)
                    )
                    for name in group[0][2]
                }
                rotated.append((group[0][0], reference_value, combined_vector, combined_profiles))

        for raw_indices, value, selected_vector, component_profiles in rotated:
            component_profiles = _normalize_vector_mode_profiles_numpy(
                component_profiles,
                preferred_field_name=preferred_field_name,
            )
            independent_candidates.append(
                {
                    "raw_indices": raw_indices,
                    "beta": value,
                    "vector": selected_vector,
                    "profiles": component_profiles,
                }
            )
        cursor = group_stop

    candidate_count = len(independent_candidates)
    power_gram = np.zeros((candidate_count, candidate_count), dtype=np.complex128)
    for row, left in enumerate(independent_candidates):
        for col, right in enumerate(independent_candidates):
            power_gram[row, col] = _vector_mode_power_inner_product_numpy(
                left["profiles"],
                right["profiles"],
            )
    power_norm = np.sqrt(np.maximum(np.abs(np.real(np.diag(power_gram))), np.finfo(np.float64).eps))
    overlap_matrix = np.abs(power_gram / power_norm[:, None] / power_norm[None, :])

    retained_indices = []
    family_indices = []
    candidate_diagnostics = []
    for candidate_index, candidate in enumerate(independent_candidates):
        beta = candidate["beta"]
        profiles = candidate["profiles"]
        power = _vector_mode_power_sign_numpy(profiles)
        polarization_fraction = _vector_mode_polarization_fraction_numpy(
            profiles,
            preferred_field_name=preferred_field_name,
        )
        checkerboard_fraction = _vector_mode_checkerboard_fraction_numpy(profiles)
        eigenpair_residual = _vector_mode_eigenpair_residual_numpy(operator, beta, candidate["vector"])
        electric_divergence, magnetic_divergence = _vector_mode_divergence_residuals_numpy(
            candidate["vector"],
            beta=beta,
            interior_u=interior_u,
            interior_v=interior_v,
            eps_planes=eps_planes,
            mu_planes=mu_planes,
            k0=k0,
            du=du,
            dv=dv,
        )
        prior_overlaps = [float(overlap_matrix[candidate_index, prior]) for prior in retained_indices]
        max_overlap = max(prior_overlaps, default=0.0)
        near_duplicate_overlaps = [
            float(overlap_matrix[candidate_index, prior])
            for prior in retained_indices
            if abs(beta - independent_candidates[prior]["beta"])
            <= _VECTOR_DUPLICATE_BETA_RTOL * max(abs(beta), 1.0)
        ]
        max_near_duplicate_overlap = max(near_duplicate_overlaps, default=0.0)

        family_index = None
        if power <= 0.0:
            status = "backward_power"
        elif reject_spurious and checkerboard_fraction > _VECTOR_CHECKERBOARD_FRACTION_LIMIT:
            status = "checkerboard"
        elif reject_spurious and max_near_duplicate_overlap >= _VECTOR_DUPLICATE_OVERLAP_LIMIT:
            status = "duplicate"
        else:
            retained_indices.append(candidate_index)
            if polarization_fraction >= 0.5:
                family_index = len(family_indices)
                family_indices.append(candidate_index)
                status = "eligible"
            else:
                status = "orthogonal_polarization"

        candidate_diagnostics.append(
            {
                "candidate_index": candidate_index,
                "raw_indices": candidate["raw_indices"],
                "beta_real": float(np.real(beta)),
                "beta_imag": float(np.imag(beta)),
                "effective_index_real": None if k0 is None else float(np.real(beta) / max(float(k0), 1e-30)),
                "effective_index_imag": None if k0 is None else float(np.imag(beta) / max(float(k0), 1e-30)),
                "propagating": bool(np.real(beta) > 0.0 and abs(np.imag(beta)) <= 1.0e-7 * max(abs(beta), 1.0)),
                "eigenpair_residual": eigenpair_residual,
                "electric_divergence_residual": electric_divergence,
                "magnetic_divergence_residual": magnetic_divergence,
                "poynting_power": power,
                "polarization_fraction": polarization_fraction,
                "max_weighted_overlap": max_overlap,
                "checkerboard_fraction": checkerboard_fraction,
                "family_index": family_index,
                "status": status,
                "selected": False,
            }
        )

    # Match the scalar mode path: only modes dominated by the requested
    # tangential E polarization occupy indices in that family.
    if len(family_indices) <= int(mode_index):
        rejected = sum(entry["status"] in {"checkerboard", "duplicate"} for entry in candidate_diagnostics)
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only "
            f"{len(family_indices)} resolved forward modes in the requested polarization family were found "
            f"after rejecting {rejected} duplicate/checkerboard candidates."
        )
    selected_candidate_index = family_indices[int(mode_index)]
    candidate_diagnostics[selected_candidate_index]["selected"] = True
    selected = independent_candidates[selected_candidate_index]
    diagnostics = {
        "raw_candidate_count": len(raw_candidates),
        "independent_candidate_count": candidate_count,
        "selected_candidate_index": selected_candidate_index,
        "candidates": tuple(candidate_diagnostics),
        "overlap_matrix": overlap_matrix,
    }
    return selected["beta"], selected["vector"], selected["profiles"], diagnostics


def _solve_pec_vector_mode_eigenpair_torch(
    operator: torch.Tensor,
    active: torch.Tensor,
    *,
    interior_u: int,
    interior_v: int,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    """Select one forward PEC mode without leaving the material tensor device."""

    eigenvalues, eigenvectors = torch.linalg.eig(operator)
    order = torch.argsort(torch.real(eigenvalues), descending=True)
    full_size = 4 * int(active.numel())
    component_active = active.repeat(4)
    candidates = []
    for index_tensor in order:
        index = int(index_tensor.item())
        beta = eigenvalues[index]
        beta_real = float(torch.real(beta).item())
        beta_imag = float(torch.abs(torch.imag(beta)).item())
        if beta_real <= 0.0 or beta_imag > 1.0e-7 * max(abs(beta_real), 1.0):
            continue

        vector = torch.zeros(
            (full_size,),
            device=operator.device,
            dtype=eigenvectors.dtype,
        )
        vector[component_active] = eigenvectors[:, index]
        hu, hv, eu, ev = _split_vector_mode_components(
            vector,
            interior_u=interior_u,
            interior_v=interior_v,
            backend="torch",
        )
        profiles = {
            field_names[0]: eu,
            field_names[1]: ev,
            field_names[2]: hu,
            field_names[3]: hv,
        }
        preferred = profiles[preferred_field_name]
        peak = torch.max(torch.abs(preferred))
        if float(peak.item()) <= 0.0:
            continue
        peak_index = int(torch.argmax(torch.abs(preferred)).item())
        peak_value = preferred.reshape(-1)[peak_index]
        phase = torch.conj(peak_value) / torch.abs(peak_value)
        profiles = {name: profile * phase / peak for name, profile in profiles.items()}

        power_sign = torch.sum(torch.real(
            profiles[field_names[0]] * torch.conj(profiles[field_names[3]])
            - profiles[field_names[1]] * torch.conj(profiles[field_names[2]])
        ))
        if float(power_sign.item()) <= 0.0:
            continue
        electric_energy = sum(
            torch.sum(torch.abs(profile).square())
            for name, profile in profiles.items()
            if name.startswith("E")
        )
        preferred_energy = torch.sum(torch.abs(profiles[preferred_field_name]).square())
        polarization_fraction = preferred_energy / electric_energy
        if float(polarization_fraction.item()) < 0.5:
            continue
        candidates.append((beta, vector, profiles))

    if len(candidates) <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {len(candidates)} "
            "forward PEC modes in the requested polarization family were found."
        )
    return candidates[int(mode_index)]


def _full_field_component_profiles(component_profiles, shape, *, tangential_field_names):
    profiles = {name: torch.zeros(shape) for name in _FIELD_COMPONENTS + ("Hx", "Hy", "Hz")}
    for name in tangential_field_names:
        profiles[name] = component_profiles[name]
    return profiles

def _build_scalar_operator(index_sq: np.ndarray, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    main_u = np.full((interior_u,), -2.0 / (du * du), dtype=np.float64)
    off_u = np.full((max(interior_u - 1, 0),), 1.0 / (du * du), dtype=np.float64)
    d2_u = sparse.diags((off_u, main_u, off_u), offsets=(-1, 0, 1), format="csr")

    main_v = np.full((interior_v,), -2.0 / (dv * dv), dtype=np.float64)
    off_v = np.full((max(interior_v - 1, 0),), 1.0 / (dv * dv), dtype=np.float64)
    d2_v = sparse.diags((off_v, main_v, off_v), offsets=(-1, 0, 1), format="csr")

    laplacian = sparse.kron(sparse.eye(interior_v, format="csr"), d2_u, format="csr")
    laplacian = laplacian + sparse.kron(d2_v, sparse.eye(interior_u, format="csr"), format="csr")
    potential = sparse.diags(index_sq[1:-1, 1:-1].reshape(-1), offsets=0, format="csr")
    return laplacian + potential, interior_u, interior_v


def _build_scalar_operator_torch_dense(index_sq: torch.Tensor, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    dtype = index_sq.dtype
    device = index_sq.device

    main_u = torch.full((interior_u,), -2.0 / (du * du), device=device, dtype=dtype)
    off_u = torch.full((max(interior_u - 1, 0),), 1.0 / (du * du), device=device, dtype=dtype)
    d2_u = torch.diag(main_u)
    if off_u.numel() > 0:
        d2_u = d2_u + torch.diag(off_u, diagonal=1) + torch.diag(off_u, diagonal=-1)

    main_v = torch.full((interior_v,), -2.0 / (dv * dv), device=device, dtype=dtype)
    off_v = torch.full((max(interior_v - 1, 0),), 1.0 / (dv * dv), device=device, dtype=dtype)
    d2_v = torch.diag(main_v)
    if off_v.numel() > 0:
        d2_v = d2_v + torch.diag(off_v, diagonal=1) + torch.diag(off_v, diagonal=-1)

    identity_u = torch.eye(interior_u, device=device, dtype=dtype)
    identity_v = torch.eye(interior_v, device=device, dtype=dtype)
    laplacian = torch.kron(identity_v, d2_u) + torch.kron(d2_v, identity_u)
    potential = torch.diag(index_sq[1:-1, 1:-1].reshape(-1))
    return laplacian + potential, interior_u, interior_v


def _build_scalar_operator_torch_sparse(index_sq: torch.Tensor, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    device = index_sq.device
    dtype = index_sq.dtype
    flat_indices = torch.arange(unknowns, device=device, dtype=torch.int64).reshape((interior_v, interior_u))
    center = flat_indices.reshape(-1)

    row_chunks = [center]
    col_chunks = [center]
    value_chunks = [
        (
            index_sq[1:-1, 1:-1].reshape(-1)
            + float(-2.0 / (du * du) - 2.0 / (dv * dv))
        ).to(device=device, dtype=dtype)
    ]

    if interior_u > 1:
        lower = flat_indices[:, :-1].reshape(-1)
        upper = flat_indices[:, 1:].reshape(-1)
        coupling = torch.full((lower.numel(),), 1.0 / (du * du), device=device, dtype=dtype)
        row_chunks.extend((lower, upper))
        col_chunks.extend((upper, lower))
        value_chunks.extend((coupling, coupling))

    if interior_v > 1:
        lower = flat_indices[:-1, :].reshape(-1)
        upper = flat_indices[1:, :].reshape(-1)
        coupling = torch.full((lower.numel(),), 1.0 / (dv * dv), device=device, dtype=dtype)
        row_chunks.extend((lower, upper))
        col_chunks.extend((upper, lower))
        value_chunks.extend((coupling, coupling))

    indices = torch.stack((torch.cat(row_chunks), torch.cat(col_chunks)), dim=0)
    values = torch.cat(value_chunks)
    operator = torch.sparse_coo_tensor(indices, values, (unknowns, unknowns), device=device, dtype=dtype)
    return operator.coalesce(), interior_u, interior_v


def _solve_mode_eigenpair(operator, *, mode_index: int):
    dense = operator.toarray()
    eigenvalues, eigenvectors = scipy_linalg.eigh(dense)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    valid = np.where(eigenvalues > 0.0)[0]
    if valid.size <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {valid.size} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)])
    return float(eigenvalues[mode_slot]), eigenvectors[:, mode_slot]


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


def _real_profile_from_complex_eigenvector(eigenvector, *, interior_u: int, interior_v: int, shape):
    """Real transverse amplitude of a complex eigenmode, phase-aligned at its peak.

    A complex eigenvector carries an arbitrary global phase; removing the phase at
    the amplitude peak yields the real standing transverse shape used for soft
    injection (the complex propagation constant, not the transverse phase, carries
    the loss). Returns a peak-normalized, sign-oriented float64 numpy array on the
    full aperture grid with zeroed Dirichlet borders.
    """
    peak = int(np.argmax(np.abs(eigenvector)))
    peak_value = eigenvector[peak]
    if abs(peak_value) <= 0.0:
        raise RuntimeError("ModeSource complex eigenmode solve returned a zero profile.")
    aligned = eigenvector * (np.conjugate(peak_value) / abs(peak_value))
    profile = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float64)
    profile[1:-1, 1:-1] = np.real(aligned).reshape((int(interior_u), int(interior_v)))
    scale = float(np.max(np.abs(profile)))
    if scale <= 0.0:
        raise RuntimeError("ModeSource complex eigenmode solve returned a zero profile.")
    profile /= scale
    peak_index = np.unravel_index(np.argmax(np.abs(profile)), profile.shape)
    if profile[peak_index] < 0.0:
        profile *= -1.0
    return profile


def _solve_mode_eigenpair_torch(operator: torch.Tensor, *, mode_index: int):
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    valid = torch.nonzero(eigenvalues > 0.0, as_tuple=False).flatten()
    if valid.numel() <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {int(valid.numel())} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)].item())
    return eigenvalues[mode_slot], eigenvectors[:, mode_slot]


def _lobpcg_request_count(unknowns: int, *, mode_index: int) -> int:
    requested = max(int(mode_index) + _LOBPCG_REQUEST_PADDING, 1)
    max_requested = max(1, (int(unknowns) - 1) // 3)
    requested = min(requested, max_requested)
    if requested <= int(mode_index):
        raise ValueError(
            "ModeSource mode_index is too large for this aperture's resolution: the sparse "
            "LOBPCG eigensolver needs at least three interior aperture unknowns per requested "
            f"mode to span a valid iteration subspace, but mode_index={mode_index} on an "
            f"aperture with {unknowns} interior unknowns leaves fewer. Refine the aperture grid "
            "or request a lower mode_index."
        )
    return requested


def _lobpcg_initial_guess(interior_u: int, interior_v: int, *, k: int, device, dtype) -> torch.Tensor:
    coord_u = torch.arange(1, interior_u + 1, device=device, dtype=dtype)
    coord_v = torch.arange(1, interior_v + 1, device=device, dtype=dtype)
    basis_pairs: list[tuple[int, int]] = []
    total_order = 2
    while len(basis_pairs) < int(k):
        for order_u in range(1, total_order):
            order_v = total_order - order_u
            basis_pairs.append((order_u, order_v))
            if len(basis_pairs) >= int(k):
                break
        total_order += 1

    vectors = []
    for order_u, order_v in basis_pairs:
        candidate = torch.sin(math.pi * float(order_v) * coord_v / float(interior_v + 1))[:, None]
        candidate = candidate * torch.sin(math.pi * float(order_u) * coord_u / float(interior_u + 1))[None, :]
        vectors.append(candidate.reshape(-1))
    return torch.stack(vectors, dim=1).contiguous()


def _solve_mode_eigenpair_torch_sparse(operator: torch.Tensor, *, interior_u: int, interior_v: int, mode_index: int):
    unknowns = int(operator.shape[0])
    requested = _lobpcg_request_count(unknowns, mode_index=int(mode_index))
    initial_guess = _lobpcg_initial_guess(
        interior_u,
        interior_v,
        k=requested,
        device=operator.device,
        dtype=operator.dtype,
    )
    eigenvalues, eigenvectors = torch.lobpcg(
        operator,
        k=requested,
        X=initial_guess,
        niter=max(_LOBPCG_MAX_ITER, requested * 8),
        tol=_LOBPCG_TOL,
        largest=True,
    )
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    valid = torch.nonzero(eigenvalues > 0.0, as_tuple=False).flatten()
    if valid.numel() <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {int(valid.numel())} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)].item())
    return eigenvalues[mode_slot], eigenvectors[:, mode_slot]


def _orient_unit_eigenvector(eigenvector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(eigenvector)
    if float(norm.item()) <= 0.0:
        raise RuntimeError("ModeSource eigenmode solve returned a zero eigenvector.")
    normalized = eigenvector / norm
    peak_index = int(torch.argmax(torch.abs(normalized)).item())
    if float(normalized[peak_index].item()) < 0.0:
        normalized = -normalized
    return normalized


def _sparse_matvec(operator: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector for sparse matvec, got shape {tuple(vector.shape)}.")
    return torch.sparse.mm(operator, vector[:, None]).squeeze(1)


def _project_orthogonal(vector: torch.Tensor, eigenvector: torch.Tensor) -> torch.Tensor:
    return vector - eigenvector * torch.dot(eigenvector, vector)


def _apply_shifted_square_operator(
    operator: torch.Tensor,
    vector: torch.Tensor,
    *,
    eigenvalue: torch.Tensor,
    eigenvector: torch.Tensor,
) -> torch.Tensor:
    shifted = _sparse_matvec(operator, vector) - eigenvalue * vector
    shifted = _sparse_matvec(operator, shifted) - eigenvalue * shifted
    return shifted + eigenvector * torch.dot(eigenvector, vector)


def _conjugate_gradient_solve(
    apply_operator,
    rhs: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = torch.dot(r, r)
    rhs_norm = torch.sqrt(rr)
    threshold = max(float(rhs_norm.item()) * float(tol), float(tol))
    if float(rhs_norm.item()) <= threshold:
        return x

    for _ in range(int(max_iter)):
        ap = apply_operator(p)
        denom = torch.dot(p, ap)
        if abs(float(denom.item())) <= 1.0e-30:
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * ap
        r_norm = torch.linalg.vector_norm(r)
        if float(r_norm.item()) <= threshold:
            return x
        rr_new = torch.dot(r, r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return x


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


def _solve_vector_mode_eigenpair_sparse(
    eps_planes,
    mu_planes,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    operator, interior_u, interior_v = _build_vector_operator_sparse(eps_planes, mu_planes, k0=k0, du=du, dv=dv)
    matrix_size = int(operator.shape[0])
    requested = _vector_mode_request_count(matrix_size, mode_index=int(mode_index))
    initial_vector = np.random.default_rng(0).standard_normal(matrix_size)
    try:
        eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
            operator,
            k=requested,
            which="LR",
            tol=_VECTOR_EIGS_TOL,
            maxiter=_VECTOR_EIGS_MAX_ITER,
            v0=initial_vector,
        )
    except scipy_sparse_linalg.ArpackNoConvergence as error:
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors
        minimum_candidates = max(2 * (int(mode_index) + 1), 4)
        if eigenvalues is None or eigenvectors is None or len(eigenvalues) < minimum_candidates:
            raise
    return _select_and_normalize_vector_mode_numpy(
        eigenvalues,
        eigenvectors,
        interior_u=interior_u,
        interior_v=interior_v,
        mode_index=int(mode_index),
        field_names=field_names,
        preferred_field_name=preferred_field_name,
        operator=operator,
        eps_planes=eps_planes,
        mu_planes=mu_planes,
        k0=k0,
        du=du,
        dv=dv,
        reject_spurious=not _is_uniform_isotropic_vector_plane(eps_planes, mu_planes),
    )


def _solve_vector_mode_eigenpair_dense(
    eps_planes,
    mu_planes,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    operator, interior_u, interior_v = _build_vector_operator_sparse(eps_planes, mu_planes, k0=k0, du=du, dv=dv)
    dense = operator.toarray()
    eigenvalues, eigenvectors = scipy_linalg.eig(dense)
    return _select_and_normalize_vector_mode_numpy(
        eigenvalues,
        eigenvectors,
        interior_u=interior_u,
        interior_v=interior_v,
        mode_index=int(mode_index),
        field_names=field_names,
        preferred_field_name=preferred_field_name,
        operator=operator,
        eps_planes=eps_planes,
        mu_planes=mu_planes,
        k0=k0,
        du=du,
        dv=dv,
        reject_spurious=not _is_uniform_isotropic_vector_plane(eps_planes, mu_planes),
    )


def _mode_slice(tensor: torch.Tensor, *, axis: str, plane_index: int, tangential_bounds):
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    if axis == "x":
        return tensor[plane_index, u_lo : u_hi + 1, v_lo : v_hi + 1]
    if axis == "y":
        return tensor[u_lo : u_hi + 1, plane_index, v_lo : v_hi + 1]
    return tensor[u_lo : u_hi + 1, v_lo : v_hi + 1, plane_index]


def _bend_conformal_factor(source, *, tangential_axes, tangential_bounds, coord_map):
    """Heiblum-Harris conformal index factor over a bent-port mode plane.

    Returns ``None`` for a straight port. A curved (bent) waveguide keeps its
    cross-section plane axis-aligned but its guided mode is solved on the
    equivalent straight guide produced by the conformal map: for a bend of signed
    radius ``R`` about the cylinder axis ``bend_axis``, the transformed
    permittivity is ``eps_eq(u, v) = eps(u, v) * (1 + r/R)**2`` where ``r`` is the
    signed offset of each node from the port centre along the in-plane radial axis
    (the tangential axis that is not the cylinder axis). ``R > 0`` grades the index
    up on the ``+r`` side, shifting the mode outward and raising the effective
    index, exactly as a physical bend does. The returned float64 tensor has shape
    ``(nu, 1)`` or ``(1, nv)`` so it broadcasts against a mode-plane aperture slice.
    """
    bend_radius = source.get("bend_radius")
    if bend_radius is None:
        return None
    bend_axis = str(source["bend_axis"])
    axis_u, axis_v = tangential_axes
    radial_is_u = axis_v == bend_axis
    radial_axis = axis_u if radial_is_u else axis_v
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    lo, hi = (u_lo, u_hi) if radial_is_u else (v_lo, v_hi)
    coords_radial = coord_map[radial_axis][lo : hi + 1].to(dtype=torch.float64)
    radial_center = float(source["position"][_AXIS_TO_INDEX[radial_axis]])
    ratio = 1.0 + (coords_radial - radial_center) / float(bend_radius)
    if float(torch.min(ratio).item()) <= 0.0:
        raise ValueError(
            "Bent mode-plane aperture spans the bend centre of curvature "
            f"(1 + r/R <= 0 for bend_radius={float(bend_radius):g} about axis {bend_axis!r}); "
            "the conformal transform is singular there. Increase |bend_radius| or shrink the aperture."
        )
    factor_1d = ratio * ratio
    return factor_1d[:, None] if radial_is_u else factor_1d[None, :]


def _apply_bend_factor(tensor: torch.Tensor, factor) -> torch.Tensor:
    """Multiply a mode-plane eps slice by the conformal bend factor (identity if None)."""
    if factor is None:
        return tensor
    real_dtype = torch.real(tensor).dtype if torch.is_complex(tensor) else tensor.dtype
    return tensor * factor.to(device=tensor.device, dtype=real_dtype)


def _regular_grid_sample(coords: torch.Tensor, values: torch.Tensor):
    if values.numel() < 2:
        raise ValueError("ModeSource profile sampling requires at least two grid points per axis.")
    spacing = float(coords[1].item() - coords[0].item())
    if spacing <= 0.0:
        raise ValueError("ModeSource tangential coordinates must be strictly increasing.")
    return float(coords[0].item()), spacing


def _bilinear_sample(profile: torch.Tensor, coords_u: torch.Tensor, coords_v: torch.Tensor, positions: torch.Tensor):
    u_axis = int(profile.shape[0])
    v_axis = int(profile.shape[1])
    u0, du = _regular_grid_sample(coords_u, profile[:, 0])
    v0, dv = _regular_grid_sample(coords_v, profile[0, :])

    pos_u = positions[..., 0]
    pos_v = positions[..., 1]
    mask = (
        (pos_u >= float(coords_u[0].item()))
        & (pos_u <= float(coords_u[-1].item()))
        & (pos_v >= float(coords_v[0].item()))
        & (pos_v <= float(coords_v[-1].item()))
    )

    u_coord = torch.clamp((pos_u - u0) / du, min=0.0, max=max(u_axis - 1, 0))
    v_coord = torch.clamp((pos_v - v0) / dv, min=0.0, max=max(v_axis - 1, 0))
    u_lower = torch.floor(u_coord).to(torch.int64)
    v_lower = torch.floor(v_coord).to(torch.int64)
    u_upper = torch.clamp(u_lower + 1, max=u_axis - 1)
    v_upper = torch.clamp(v_lower + 1, max=v_axis - 1)
    u_frac = u_coord - u_lower.to(dtype=profile.dtype)
    v_frac = v_coord - v_lower.to(dtype=profile.dtype)

    f00 = profile[u_lower, v_lower]
    f10 = profile[u_upper, v_lower]
    f01 = profile[u_lower, v_upper]
    f11 = profile[u_upper, v_upper]
    sampled = (
        (1.0 - u_frac) * (1.0 - v_frac) * f00
        + u_frac * (1.0 - v_frac) * f10
        + (1.0 - u_frac) * v_frac * f01
        + u_frac * v_frac * f11
    )
    return torch.where(mask, sampled, torch.zeros_like(sampled))


def _mode_source_component_permittivity(solver, source, *, frequency: float) -> torch.Tensor:
    field_name = _mode_source_field_name(source)
    component_name = f"eps_{field_name}"
    field_tensor = getattr(solver, component_name, None)
    use_component_fields = bool(getattr(solver, "_mode_source_rebuild_from_fields", False))
    use_component_fields = use_component_fields or (
        isinstance(field_tensor, torch.Tensor) and bool(field_tensor.requires_grad)
    )

    if use_component_fields:
        if field_tensor is None:
            raise RuntimeError(
                f"ModeSource differentiable rebuild requires {component_name} on the temporary solver context."
            )
        eps0 = float(getattr(solver, "eps0", VACUUM_PERMITTIVITY))
        return field_tensor / eps0

    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is not None:
        # A diagonal-anisotropic aperture must be solved with the permittivity
        # component that matches the injected polarization (E_p sees eps_pp), the
        # same per-axis component the forward Yee update uses for that field
        # (solver.eps_E{p}). Averaging the three diagonal components would inject a
        # mode computed for an isotropic medium the forward solve never sees. For an
        # isotropic plane the three components coincide and this is unchanged.
        polarization_axis = field_name[1].lower()
        eps_components, _ = evaluate_material_components(compiled_material_model, float(frequency))
        return _average_node_tensor_to_component(eps_components[polarization_axis], field_name)

    if field_tensor is None:
        raise RuntimeError(
            "ModeSource requires either solver._compiled_material_model or component permittivity tensors."
        )
    eps0 = float(getattr(solver, "eps0", VACUUM_PERMITTIVITY))
    return field_tensor / eps0


def _normalize_profile_torch(profile: torch.Tensor) -> torch.Tensor:
    peak = torch.max(torch.abs(profile))
    if float(peak.item()) <= 0.0:
        raise RuntimeError("ModeSource eigenmode solve returned a zero profile.")
    normalized = profile / peak
    peak_index = int(torch.argmax(torch.abs(normalized).reshape(-1)).item())
    if float(normalized.reshape(-1)[peak_index].item()) < 0.0:
        normalized = -normalized
    return normalized


def _vector_mode_supported(
    solver,
    *,
    eps_by_axis: dict,
    mu_by_axis: dict,
    unknowns: int,
) -> bool:
    if getattr(solver, "_mode_source_rebuild_from_fields", False):
        return False
    if getattr(solver, "_compiled_material_model", None) is None:
        return False
    if any(component.requires_grad for component in eps_by_axis.values()):
        return False
    if any(component.requires_grad for component in mu_by_axis.values()):
        return False
    if unknowns <= 0:
        return False
    return True


def _real_plane_numpy(component: torch.Tensor) -> np.ndarray:
    real = torch.real(component) if torch.is_complex(component) else component
    return real.detach().cpu().numpy().astype(np.float64, copy=False)


def _boundary_connected_conductor(conductor: torch.Tensor) -> torch.Tensor:
    connected = torch.zeros_like(conductor)
    connected[0, :] = conductor[0, :]
    connected[-1, :] = conductor[-1, :]
    connected[:, 0] |= conductor[:, 0]
    connected[:, -1] |= conductor[:, -1]
    for _ in range(int(conductor.shape[0] + conductor.shape[1])):
        adjacent = torch.zeros_like(conductor)
        adjacent[1:, :] |= connected[:-1, :]
        adjacent[:-1, :] |= connected[1:, :]
        adjacent[:, 1:] |= connected[:, :-1]
        adjacent[:, :-1] |= connected[:, 1:]
        connected |= adjacent & conductor
    return connected


def _solve_pec_tem_mode_torch(
    eps_planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mu_planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pec_occupancy: torch.Tensor,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
):
    if int(mode_index) != 0:
        raise ValueError("A TEM WavePort currently supports one common conductor mode.")
    conductor = pec_occupancy >= _PEC_OCCUPANCY_THRESHOLD
    grounded = _boundary_connected_conductor(conductor)
    signal = conductor & ~grounded
    if not bool(torch.any(signal)):
        raise ValueError(
            "A TEM WavePort requires at least one isolated signal conductor and a "
            "grounded aperture boundary."
        )

    dielectric = ~conductor
    eps_u, eps_v, _ = eps_planes
    mu_u, mu_v, _ = mu_planes
    eps_relative = 0.5 * (torch.real(eps_u) + torch.real(eps_v))
    mu_relative = 0.5 * (torch.real(mu_u) + torch.real(mu_v))
    eps_values = eps_relative[dielectric]
    mu_values = mu_relative[dielectric]
    if not torch.allclose(
        eps_values,
        eps_values[0].expand_as(eps_values),
        rtol=1.0e-4,
        atol=1.0e-6,
    ) or not torch.allclose(
        mu_values,
        mu_values[0].expand_as(mu_values),
        rtol=1.0e-4,
        atol=1.0e-6,
    ):
        raise NotImplementedError(
            "TEM WavePort electrostatic normalization requires a uniformly filled "
            "cross-section; use a hybrid mode for inhomogeneous transmission lines."
        )

    potential = signal.to(dtype=torch.float64)
    free = dielectric.clone()
    free[0, :] = False
    free[-1, :] = False
    free[:, 0] = False
    free[:, -1] = False
    weight_u = 1.0 / (float(du) * float(du))
    weight_v = 1.0 / (float(dv) * float(dv))
    denominator = 2.0 * (weight_u + weight_v)
    for _ in range(2048):
        update = potential.clone()
        update[1:-1, 1:-1] = (
            weight_u * (potential[2:, 1:-1] + potential[:-2, 1:-1])
            + weight_v * (potential[1:-1, 2:] + potential[1:-1, :-2])
        ) / denominator
        potential = torch.where(free, update, potential)

    electric_u = torch.zeros_like(potential)
    electric_v = torch.zeros_like(potential)
    electric_u[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2.0 * float(du))
    electric_u[0, :] = -(potential[1, :] - potential[0, :]) / float(du)
    electric_u[-1, :] = -(potential[-1, :] - potential[-2, :]) / float(du)
    electric_v[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2.0 * float(dv))
    electric_v[:, 0] = -(potential[:, 1] - potential[:, 0]) / float(dv)
    electric_v[:, -1] = -(potential[:, -1] - potential[:, -2]) / float(dv)

    impedance = _ETA0 * torch.sqrt(mu_relative / eps_relative)
    magnetic_u = -electric_v / impedance
    magnetic_v = electric_u / impedance
    beta = torch.as_tensor(
        float(k0),
        device=potential.device,
        dtype=torch.float64,
    ) * torch.sqrt(eps_values[0] * mu_values[0])
    return beta, {
        field_names[0]: electric_u,
        field_names[1]: electric_v,
        field_names[2]: magnetic_u,
        field_names[3]: magnetic_v,
    }, int(torch.count_nonzero(conductor).item())


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
    du = _local_uniform_plane_spacing(solver.scene, axis_u, u_lo, u_hi)
    dv = _local_uniform_plane_spacing(solver.scene, axis_v, v_lo, v_hi)
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
    pec_node_count = 0
    has_interior_pec = pec_occupancy is not None and bool(
        torch.any(pec_occupancy[1:-1, 1:-1] >= _PEC_OCCUPANCY_THRESHOLD)
    )
    if has_interior_pec and source.get("wave_family") == "tem":
        beta_value, component_arrays, pec_node_count = _solve_pec_tem_mode_torch(
            tensor_eps_planes,
            tensor_mu_planes,
            pec_occupancy,
            k0=k0,
            du=du,
            dv=dv,
            mode_index=int(source["mode_index"]),
            field_names=field_names,
        )
        solver_kind = "tem_electrostatic_torch"
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
        if unknowns <= _FULL_VECTOR_DENSE_LIMIT:
            beta_value, _eigenvector, component_arrays, candidate_diagnostics = _solve_vector_mode_eigenpair_dense(
                eps_planes,
                mu_planes,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
                field_names=field_names,
                preferred_field_name=preferred_field_name,
            )
            solver_kind = "vector_dense"
        else:
            beta_value, _eigenvector, component_arrays, candidate_diagnostics = _solve_vector_mode_eigenpair_sparse(
                eps_planes,
                mu_planes,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(source["mode_index"]),
                field_names=field_names,
                preferred_field_name=preferred_field_name,
            )
            solver_kind = "vector_sparse"

    beta_real = (
        float(torch.real(beta_value).item())
        if isinstance(beta_value, torch.Tensor)
        else float(np.real(beta_value))
    )
    effective_index_value = beta_real / max(k0, 1e-30)
    target_device = torch.device(getattr(solver, "device", solver.scene.device))
    target_dtype = solver.Ex.dtype
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
    imag_bound = 1e-3 * max(profile_scale, 1e-30)
    plane_shape = tuple(int(dim) for dim in tensor_eps_planes[0].shape)
    interior_shape = (plane_shape[0] - 2, plane_shape[1] - 2)
    component_profiles = {}
    for name, array in component_arrays.items():
        if isinstance(array, torch.Tensor):
            resolved_array = array
            if resolved_array.is_complex():
                imag_max = float(torch.max(torch.abs(torch.imag(resolved_array))).item())
                if imag_max > imag_bound:
                    raise RuntimeError(
                        "Full-vector ModeSource forward solve returned a materially complex field profile."
                    )
                resolved_array = torch.real(resolved_array)
            resolved_profile = resolved_array.to(
                device=target_device,
                dtype=target_dtype,
            ).contiguous()
        else:
            resolved_array = np.asarray(array)
            if np.iscomplexobj(resolved_array):
                imag_max = float(np.max(np.abs(np.imag(resolved_array))))
                if imag_max > imag_bound:
                    raise RuntimeError(
                        "Full-vector ModeSource forward solve returned a materially complex field profile."
                    )
                resolved_array = np.real(resolved_array)
            resolved_profile = torch.as_tensor(
                resolved_array,
                device=target_device,
                dtype=target_dtype,
            ).contiguous()

        resolved_shape = tuple(int(dim) for dim in resolved_profile.shape)
        if resolved_shape == interior_shape:
            component_profile = torch.zeros(
                plane_shape,
                device=target_device,
                dtype=target_dtype,
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
        component_profiles.setdefault(field_name, torch.zeros_like(reference_profile))
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
        "effective_index": float(effective_index_value),
        "beta": float(beta_real),
        "effective_index_tensor": None,
        "beta_tensor": None,
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


def solve_mode_source_profile(solver, source) -> dict[str, object]:
    if source.get("injection", {}).get("kind") != "soft":
        raise ValueError("ModeSource currently supports soft injection only.")
    source_time = source["source_time"]
    if source_time["kind"] == "custom":
        raise ValueError(
            "ModeSource does not support CustomSourceTime; the native time-shifted surface "
            "kernel evaluates only the analytic CW/Gaussian/Ricker forms. Use PointDipole "
            "for arbitrary custom waveforms."
        )
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
        and not rebuild_from_fields
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
        eps_node_by_axis, mu_node_by_axis = _mode_source_relative_material_slices(
            solver,
            frequency=frequency,
            normal_axis=normal_axis,
            plane_index=plane_index,
            tangential_bounds=node_tangential_bounds,
        )
        eps_node_imag = _max_component_imag(eps_node_by_axis)
        mu_node_imag = _max_component_imag(mu_node_by_axis)
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
            raise NotImplementedError(
                "Lossy PEC-aware modes require a device-resident complex vector eigensolve; "
                "the scalar complex mode path cannot enforce internal conductors."
            )
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
        effective_index_tensor / _ETA0
        if effective_index_tensor is not None
        else effective_index_value / _ETA0
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
    for name, component in electric_vector.items():
        if abs(float(component)) > 1e-12:
            component_profiles[name] = float(component) * scalar_profile
    for name, component in resolved_magnetic_vector.items():
        if isinstance(component, torch.Tensor):
            component_profiles[name] = component * scalar_profile
        elif abs(float(component)) > 1e-12:
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
    return _bilinear_sample(
        profile,
        mode_data["coords_u"],
        mode_data["coords_v"],
        sample_positions,
    )


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
