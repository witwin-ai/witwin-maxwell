from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as scipy_sparse_linalg
from witwin.core.material import VACUUM_PERMITTIVITY

from ...compiler.materials import evaluate_material_permittivity
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
    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is None:
        raise RuntimeError("Full-vector ModeSource currently requires solver._compiled_material_model.")

    eps_r = evaluate_material_permittivity(compiled_material_model, float(frequency))
    mu_r = compiled_material_model["mu_r"]
    eps_slice = _mode_slice(
        eps_r,
        axis=normal_axis,
        plane_index=plane_index,
        tangential_bounds=tangential_bounds,
    )
    mu_slice = _mode_slice(
        mu_r,
        axis=normal_axis,
        plane_index=plane_index,
        tangential_bounds=tangential_bounds,
    )
    return eps_slice, mu_slice


def _build_first_difference_sparse(count: int, spacing: float):
    if count <= 0:
        raise ValueError("count must be > 0 for first-difference assembly.")
    off = np.full((max(count - 1, 0),), 0.5 / float(spacing), dtype=np.float64)
    return sparse.diags((-off, off), offsets=(-1, 1), shape=(count, count), format="csr")


def _build_first_difference_torch_dense(count: int, spacing: float, *, device, dtype):
    if count <= 0:
        raise ValueError("count must be > 0 for first-difference assembly.")
    operator = torch.zeros((count, count), device=device, dtype=dtype)
    if count > 1:
        off = torch.full((count - 1,), 0.5 / float(spacing), device=device, dtype=dtype)
        operator = operator + torch.diag(off, diagonal=1) - torch.diag(off, diagonal=-1)
    return operator


def _build_vector_operator_sparse(eps_r: np.ndarray, mu_r: np.ndarray, *, k0: float, du: float, dv: float):
    nu = int(eps_r.shape[0])
    nv = int(eps_r.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    d1_u = _build_first_difference_sparse(interior_u, du)
    d1_v = _build_first_difference_sparse(interior_v, dv)
    identity_u = sparse.eye(interior_u, format="csr", dtype=np.float64)
    identity_v = sparse.eye(interior_v, format="csr", dtype=np.float64)
    derivative_u = sparse.kron(d1_u, identity_v, format="csr")
    derivative_v = sparse.kron(identity_u, d1_v, format="csr")

    eps_flat = np.asarray(eps_r[1:-1, 1:-1], dtype=np.float64).reshape(-1)
    mu_flat = np.asarray(mu_r[1:-1, 1:-1], dtype=np.float64).reshape(-1)
    eps_inv = sparse.diags(1.0 / eps_flat, offsets=0, format="csr")
    mu_inv = sparse.diags(1.0 / mu_flat, offsets=0, format="csr")
    eps_diag = sparse.diags(eps_flat, offsets=0, format="csr")
    mu_diag = sparse.diags(mu_flat, offsets=0, format="csr")
    k0_sq = float(k0) * float(k0)

    a_hu_hu = derivative_v @ eps_inv @ derivative_v + k0_sq * mu_diag
    a_hu_hv = -derivative_v @ eps_inv @ derivative_u
    a_hv_hu = -derivative_u @ eps_inv @ derivative_v
    a_hv_hv = derivative_u @ eps_inv @ derivative_u + k0_sq * mu_diag

    a_eu_eu = -derivative_v @ mu_inv @ derivative_v - k0_sq * eps_diag
    a_eu_ev = derivative_v @ mu_inv @ derivative_u
    a_ev_eu = derivative_u @ mu_inv @ derivative_v
    a_ev_ev = -derivative_u @ mu_inv @ derivative_u - k0_sq * eps_diag

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


def _build_vector_operator_torch_dense(eps_r: torch.Tensor, mu_r: torch.Tensor, *, k0: float, du: float, dv: float):
    nu = int(eps_r.shape[0])
    nv = int(eps_r.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )
    if unknowns > _FULL_VECTOR_DENSE_LIMIT:
        raise NotImplementedError(
            "Full-vector differentiable ModeSource currently supports at most "
            f"{_FULL_VECTOR_DENSE_LIMIT} interior nodes per source-plane solve."
        )

    device = eps_r.device
    dtype = eps_r.dtype
    d1_u = _build_first_difference_torch_dense(interior_u, du, device=device, dtype=dtype)
    d1_v = _build_first_difference_torch_dense(interior_v, dv, device=device, dtype=dtype)
    identity_u = torch.eye(interior_u, device=device, dtype=dtype)
    identity_v = torch.eye(interior_v, device=device, dtype=dtype)
    derivative_u = torch.kron(d1_u, identity_v)
    derivative_v = torch.kron(identity_u, d1_v)

    eps_flat = eps_r[1:-1, 1:-1].reshape(-1)
    mu_flat = mu_r[1:-1, 1:-1].reshape(-1)
    eps_inv = torch.diag(torch.reciprocal(eps_flat))
    mu_inv = torch.diag(torch.reciprocal(mu_flat))
    eps_diag = torch.diag(eps_flat)
    mu_diag = torch.diag(mu_flat)
    k0_sq = float(k0) * float(k0)

    a_hu_hu = derivative_v @ eps_inv @ derivative_v + k0_sq * mu_diag
    a_hu_hv = -(derivative_v @ eps_inv @ derivative_u)
    a_hv_hu = -(derivative_u @ eps_inv @ derivative_v)
    a_hv_hv = derivative_u @ eps_inv @ derivative_u + k0_sq * mu_diag

    a_eu_eu = -(derivative_v @ mu_inv @ derivative_v) - k0_sq * eps_diag
    a_eu_ev = derivative_v @ mu_inv @ derivative_u
    a_ev_eu = derivative_u @ mu_inv @ derivative_v
    a_ev_ev = -(derivative_u @ mu_inv @ derivative_u) - k0_sq * eps_diag

    electric_scale = float(k0) / _ETA0
    magnetic_scale = float(k0) * _ETA0
    zero = torch.zeros_like(a_hu_hu)
    operator = torch.cat(
        (
            torch.cat((zero, zero, a_ev_eu / magnetic_scale, a_ev_ev / magnetic_scale), dim=1),
            torch.cat((zero, zero, -a_eu_eu / magnetic_scale, -a_eu_ev / magnetic_scale), dim=1),
            torch.cat((a_hv_hu / electric_scale, a_hv_hv / electric_scale, zero, zero), dim=1),
            torch.cat((-a_hu_hu / electric_scale, -a_hu_hv / electric_scale, zero, zero), dim=1),
        ),
        dim=0,
    )
    return operator, interior_u, interior_v


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
    order = torch.argsort(torch.real(eigenvalues), descending=True)
    positive = [int(index) for index in order.tolist() if float(torch.real(eigenvalues[index]).item()) > 0.0]
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
    sign = 1.0 if float(np.real(preferred_value)) >= 0.0 else -1.0
    for name in tuple(component_profiles):
        component_profiles[name] = sign * component_profiles[name] / peak
    return component_profiles


def _vector_mode_power_sign_numpy(component_profiles: dict[str, np.ndarray]) -> float:
    tangential_e = [component_profiles[name] for name in component_profiles if name.startswith("E")]
    tangential_h = [component_profiles[name] for name in component_profiles if name.startswith("H")]
    if len(tangential_e) != 2 or len(tangential_h) != 2:
        return 0.0
    eu, ev = tangential_e
    hu, hv = tangential_h
    return float(np.sum(np.real(eu * np.conj(hv) - ev * np.conj(hu))))


def _select_and_normalize_vector_mode_numpy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    interior_u: int,
    interior_v: int,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    order = np.lexsort((np.abs(np.imag(eigenvalues)), -np.real(eigenvalues)))
    positive_candidates = []
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
        component_profiles = _normalize_vector_mode_profiles_numpy(
            component_profiles,
            preferred_field_name=preferred_field_name,
        )
        power_sign = _vector_mode_power_sign_numpy(component_profiles)
        positive_candidates.append((index, value, component_profiles, power_sign))

    preferred_candidates = [entry for entry in positive_candidates if entry[3] > 0.0]
    selected_candidates = preferred_candidates if len(preferred_candidates) > int(mode_index) else positive_candidates
    if len(selected_candidates) <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {len(selected_candidates)} forward modes were found."
        )
    selected_index, selected_beta, selected_profiles, _ = selected_candidates[int(mode_index)]
    return selected_beta, eigenvectors[:, selected_index], selected_profiles


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
    if unknowns > _DENSE_EIGEN_LIMIT:
        raise NotImplementedError(
            "Differentiable ModeSource currently supports at most "
            f"{_DENSE_EIGEN_LIMIT} interior unknowns per source-plane solve."
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
        raise NotImplementedError(
            "Sparse ModeSource iterative solve requires unknowns >= 3 * requested_modes. "
            f"mode_index={mode_index} is too large for an aperture solve with {unknowns} interior unknowns."
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
    eps_r: np.ndarray,
    mu_r: np.ndarray,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    operator, interior_u, interior_v = _build_vector_operator_sparse(eps_r, mu_r, k0=k0, du=du, dv=dv)
    matrix_size = int(operator.shape[0])
    requested = _vector_mode_request_count(matrix_size, mode_index=int(mode_index))
    eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
        operator,
        k=requested,
        which="LR",
        tol=_VECTOR_EIGS_TOL,
        maxiter=_VECTOR_EIGS_MAX_ITER,
    )
    return _select_and_normalize_vector_mode_numpy(
        eigenvalues,
        eigenvectors,
        interior_u=interior_u,
        interior_v=interior_v,
        mode_index=int(mode_index),
        field_names=field_names,
        preferred_field_name=preferred_field_name,
    )


def _solve_vector_mode_eigenpair_dense(
    eps_r: np.ndarray,
    mu_r: np.ndarray,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
):
    operator, interior_u, interior_v = _build_vector_operator_sparse(eps_r, mu_r, k0=k0, du=du, dv=dv)
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
    )


def _mode_slice(tensor: torch.Tensor, *, axis: str, plane_index: int, tangential_bounds):
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    if axis == "x":
        return tensor[plane_index, u_lo : u_hi + 1, v_lo : v_hi + 1]
    if axis == "y":
        return tensor[u_lo : u_hi + 1, plane_index, v_lo : v_hi + 1]
    return tensor[u_lo : u_hi + 1, v_lo : v_hi + 1, plane_index]


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
        eps_r = evaluate_material_permittivity(compiled_material_model, float(frequency))
        return _average_node_tensor_to_component(eps_r, field_name)

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
    eps_slice: torch.Tensor,
    mu_slice: torch.Tensor,
    unknowns: int,
) -> bool:
    if getattr(solver, "_mode_source_rebuild_from_fields", False):
        return False
    if getattr(solver, "_compiled_material_model", None) is None:
        return False
    if eps_slice.requires_grad or mu_slice.requires_grad:
        return False
    if unknowns <= 0:
        return False
    return True


def _assemble_vector_mode_data(
    solver,
    source,
    *,
    normal_axis: str,
    tangential_axes,
    tangential_bounds,
    tangential_coord_map,
    plane_index: int,
    eps_slice: torch.Tensor,
    mu_slice: torch.Tensor,
    frequency: float,
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
    du = _local_uniform_plane_spacing(solver.scene, axis_u, u_lo, u_hi)
    dv = _local_uniform_plane_spacing(solver.scene, axis_v, v_lo, v_hi)
    k0 = 2.0 * math.pi * float(frequency) / float(solver.c)
    eps_real = torch.real(eps_slice) if torch.is_complex(eps_slice) else eps_slice
    mu_real = torch.real(mu_slice) if torch.is_complex(mu_slice) else mu_slice
    eps_np = eps_real.detach().cpu().numpy().astype(np.float64, copy=False)
    mu_np = mu_real.detach().cpu().numpy().astype(np.float64, copy=False)

    unknowns = max((int(eps_np.shape[0]) - 2) * (int(eps_np.shape[1]) - 2), 0)
    if unknowns <= _FULL_VECTOR_DENSE_LIMIT:
        beta_value, _eigenvector, component_arrays = _solve_vector_mode_eigenpair_dense(
            eps_np,
            mu_np,
            k0=k0,
            du=du,
            dv=dv,
            mode_index=int(source["mode_index"]),
            field_names=field_names,
            preferred_field_name=preferred_field_name,
        )
        solver_kind = "vector_dense"
    else:
        beta_value, _eigenvector, component_arrays = _solve_vector_mode_eigenpair_sparse(
            eps_np,
            mu_np,
            k0=k0,
            du=du,
            dv=dv,
            mode_index=int(source["mode_index"]),
            field_names=field_names,
            preferred_field_name=preferred_field_name,
        )
        solver_kind = "vector_sparse"

    beta_real = float(np.real(beta_value))
    effective_index_value = beta_real / max(k0, 1e-30)
    target_device = torch.device(getattr(solver, "device", solver.scene.device))
    target_dtype = solver.Ex.dtype
    component_profiles = {}
    for name, array in component_arrays.items():
        resolved_array = np.asarray(array)
        if np.iscomplexobj(resolved_array):
            imag_max = float(np.max(np.abs(np.imag(resolved_array))))
            if imag_max > 1e-6:
                raise RuntimeError("Full-vector ModeSource forward solve returned a materially complex field profile.")
            resolved_array = np.real(resolved_array)
        component_profiles[name] = torch.as_tensor(
            resolved_array,
            device=target_device,
            dtype=target_dtype,
        ).contiguous()
    if int(source["direction_sign"]) < 0:
        for field_name in field_names[2:]:
            component_profiles[field_name] = -component_profiles[field_name]

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
        "plane_index": int(plane_index),
        "box_lower": tuple(int(value) for value in lower),
        "box_upper": tuple(int(value) for value in upper),
    }


def solve_mode_source_profile(solver, source) -> dict[str, object]:
    if source.get("injection", {}).get("kind") != "soft":
        raise ValueError("ModeSource currently supports soft injection only.")
    source_time = source["source_time"]
    if source_time["kind"] != "cw":
        raise ValueError("ModeSource currently supports CW source_time only.")
    if solver.scene.boundary.uses_kind("periodic") or solver.scene.boundary.uses_kind("bloch"):
        raise NotImplementedError("ModeSource currently supports only none, pml, pec, or pmc boundaries.")

    normal_axis, tangential_axes = _mode_source_axes(source)
    normal_axis_index = _AXIS_TO_INDEX[normal_axis]
    plane_index = nearest_index(getattr(solver.scene, normal_axis), float(source["position"][normal_axis_index]))
    _validate_mode_source_position(solver.scene, source, plane_index=plane_index)

    field_name = _mode_source_field_name(source)
    frequency = float(source_time["frequency"])

    if (
        getattr(solver, "_compiled_material_model", None) is not None
        and not getattr(solver, "_mode_source_rebuild_from_fields", False)
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
        eps_node_slice, mu_node_slice = _mode_source_relative_material_slices(
            solver,
            frequency=frequency,
            normal_axis=normal_axis,
            plane_index=plane_index,
            tangential_bounds=node_tangential_bounds,
        )
        eps_node_imag = (
            float(torch.max(torch.abs(torch.imag(eps_node_slice))).item()) if torch.is_complex(eps_node_slice) else 0.0
        )
        mu_node_imag = (
            float(torch.max(torch.abs(torch.imag(mu_node_slice))).item()) if torch.is_complex(mu_node_slice) else 0.0
        )
        if eps_node_imag > 1e-7 or mu_node_imag > 1e-7:
            raise NotImplementedError(
                "ModeSource full-vector eigenmode solve currently requires real-valued epsilon and mu."
            )

        eps_node_real = torch.real(eps_node_slice) if torch.is_complex(eps_node_slice) else eps_node_slice
        mu_node_real = torch.real(mu_node_slice) if torch.is_complex(mu_node_slice) else mu_node_slice
        if torch.min(eps_node_real).item() <= 0.0 or torch.min(mu_node_real).item() <= 0.0:
            raise ValueError("ModeSource requires positive epsilon and mu on the source plane.")
        node_unknowns = max((int(eps_node_real.shape[0]) - 2) * (int(eps_node_real.shape[1]) - 2), 0)
        if _vector_mode_supported(
            solver,
            eps_slice=eps_node_real,
            mu_slice=mu_node_real,
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
                eps_slice=eps_node_real,
                mu_slice=mu_node_real,
                frequency=frequency,
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
    if eps_max_imag > 1e-7 or mu_max_imag > 1e-7:
        raise NotImplementedError("ModeSource scalar eigenmode solve currently requires real-valued epsilon and mu.")

    eps_real = torch.real(eps_slice) if torch.is_complex(eps_slice) else eps_slice
    mu_real = torch.real(mu_slice) if torch.is_complex(mu_slice) else mu_slice
    if torch.min(eps_real).item() <= 0.0 or torch.min(mu_real).item() <= 0.0:
        raise ValueError("ModeSource requires positive epsilon and mu on the source plane.")
    if not torch.allclose(mu_real, torch.ones_like(mu_real), atol=1e-6, rtol=0.0):
        raise NotImplementedError("ModeSource scalar eigenmode solve currently requires mu_r == 1 on the source plane.")

    axis_u, axis_v = tangential_axes
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    coords_u = tangential_coord_map[axis_u][u_lo : u_hi + 1]
    coords_v = tangential_coord_map[axis_v][v_lo : v_hi + 1]
    du = _local_uniform_plane_spacing(solver.scene, axis_u, u_lo, u_hi)
    dv = _local_uniform_plane_spacing(solver.scene, axis_v, v_lo, v_hi)
    k0 = 2.0 * math.pi * frequency / float(solver.c)
    unknowns = max((int(eps_real.shape[0]) - 2) * (int(eps_real.shape[1]) - 2), 0)

    if eps_real.requires_grad:
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
    if eps_real.requires_grad:
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
