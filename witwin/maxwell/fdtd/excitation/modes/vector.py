"""Nodal full-vector transverse mode family: operator, selection, diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as scipy_sparse_linalg

from ....constants import ETA_0
from .common import (
    _PEC_OCCUPANCY_THRESHOLD,
    _PEC_VECTOR_MATRIX_LIMIT,
    _SPURIOUS_NEAR_K0_BETA_LIMIT,
    _SPURIOUS_UNIFORMITY_LIMIT,
    _VECTOR_CHECKERBOARD_FRACTION_LIMIT,
    _VECTOR_DEGENERATE_RTOL,
    _VECTOR_DUPLICATE_BETA_RTOL,
    _VECTOR_DUPLICATE_OVERLAP_LIMIT,
    _VECTOR_EIGEN_REQUEST_PADDING,
    _VECTOR_EIGS_MAX_ITER,
    _VECTOR_EIGS_TOL,
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

    electric_scale = float(k0) / ETA_0
    magnetic_scale = float(k0) * ETA_0
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
    electric_scale = float(k0) / ETA_0
    magnetic_scale = float(k0) * ETA_0
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


def _vector_mode_envelope_variation_numpy(
    component_profiles: dict[str, np.ndarray],
    *,
    preferred_field_name: str,
) -> float | None:
    """Anti-checkerboard transverse-envelope variation of the preferred field.

    A raw one-cell gradient of a uniform-plane mode is dominated by the
    centered-grid Nyquist/checkerboard content (present on both spurious and
    guided candidates), so it cannot separate them. Block-averaging ``|profile|``
    over 2x2 cells removes that Nyquist content and leaves the smooth transverse
    envelope; the normalized squared first-difference of that envelope is ~0 for a
    transverse-uniform (plane-wave-like / null-space) mode and O(1e-2) or larger
    for a genuine guided mode with a half-wave envelope. This is a profile-based
    structural signal independent of the returned eigenvalue. Returns ``None``
    when the profile is empty.
    """
    profile = np.abs(np.asarray(component_profiles[preferred_field_name]))
    if profile.size == 0:
        return None
    nu, nv = profile.shape
    nu2, nv2 = nu - (nu % 2), nv - (nv % 2)
    if nu2 < 2 or nv2 < 2:
        return None
    blocks = profile[:nu2, :nv2].reshape(nu2 // 2, 2, nv2 // 2, 2).mean(axis=(1, 3))
    energy = float(np.sum(blocks ** 2))
    if energy <= 0.0:
        return None
    variation = 0.0
    if blocks.shape[0] > 1:
        variation += float(np.sum(np.diff(blocks, axis=0) ** 2))
    if blocks.shape[1] > 1:
        variation += float(np.sum(np.diff(blocks, axis=1) ** 2))
    return variation / energy


def _vector_mode_transverse_uniformity_numpy(
    component_profiles: dict[str, np.ndarray],
    *,
    preferred_field_name: str,
) -> float | None:
    """Absolute (dx-independent) transverse-uniformity of the preferred field.

    Returns ``min(|env|) / max(|env|)`` of the 2x2 block-averaged (anti-
    checkerboard) preferred-field envelope. The discrete transverse null-space
    branch (beta -> k0, E ~ const) is transverse-UNIFORM: its envelope barely
    varies, so this ratio is ~1. A genuine guided mode has a half-wave envelope
    that decays toward the metallic walls, so the ratio is well below 1 (sin(pi
    y/a) reaches sin(pi h/a) at the first interior node -> ratio < 0.3 across grid
    tiers, essentially independent of dx). Unlike the squared-difference envelope
    variation, which scales as (dx)^2 and false-triggers on legitimate fine-grid
    modes above ~5 fc (audit S1, executed), this magnitude ratio is an absolute
    structural signature. Returns ``None`` when the profile is empty.
    """
    profile = np.abs(np.asarray(component_profiles[preferred_field_name]))
    if profile.size == 0:
        return None
    nu, nv = profile.shape
    nu2, nv2 = nu - (nu % 2), nv - (nv % 2)
    if nu2 < 2 or nv2 < 2:
        return None
    blocks = profile[:nu2, :nv2].reshape(nu2 // 2, 2, nv2 // 2, 2).mean(axis=(1, 3))
    peak = float(np.max(blocks))
    if peak <= 0.0:
        return None
    return float(np.min(blocks)) / peak


def _vector_mode_wall_peak_fraction_numpy(
    component_profiles: dict[str, np.ndarray],
    *,
    preferred_field_name: str,
    field_names,
) -> float | None:
    """Boundary-consistency check (F1b): |preferred| on the tangential metallic wall.

    A tangential electric field must vanish on a PEC wall. The preferred component
    ``E_p`` is tangential to the aperture edge whose in-plane normal is the OTHER
    tangential axis, so its magnitude on the first/last interior rows adjacent to
    that edge must be small relative to the profile peak. A checkerboard/wall-peaked
    spurious candidate carries its maximum |E_p| ON those wall-adjacent rows. The
    edge along the preferred axis itself (where ``E_p`` is normal to the wall) is
    left free -- TE10 ``E_z`` is uniform along z and full-amplitude at the z-walls.
    Returns ``None`` when the preferred profile is absent or empty.
    """
    profile = np.abs(np.asarray(component_profiles.get(preferred_field_name)))
    if profile.size == 0 or profile.ndim != 2:
        return None
    peak = float(np.max(profile))
    if peak <= 0.0:
        return None
    # field_names = (E_u, E_v, H_u, H_v). Axis 0 of the interior profile is u,
    # axis 1 is v. E_u is tangential to the v-walls (axis-1 edges); E_v is
    # tangential to the u-walls (axis-0 edges).
    if preferred_field_name == field_names[0]:
        wall = np.concatenate([profile[:, 0], profile[:, -1]])
    elif preferred_field_name == field_names[1]:
        wall = np.concatenate([profile[0, :], profile[-1, :]])
    else:
        return None
    return float(np.max(wall)) / peak


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
    electric_longitudinal = (beta / (float(k0) / ETA_0)) * (
        derivative_u_forward @ hv - derivative_v_forward @ hu
    )
    magnetic_transverse = derivative_u_backward @ (mu_u * hu) + derivative_v_backward @ (mu_v * hv)
    magnetic_longitudinal = (beta / (float(k0) * ETA_0)) * (
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
    wave_family: str | None = None,
):
    # A closed uniform-isotropic aperture (reject_spurious is False -> a hollow
    # metallic guide with Dirichlet walls) hosts a discrete transverse null-space
    # branch at beta ~ k0. That branch is the correct TEM answer for a
    # doubly-connected line (handled on the separate electrostatic path), but for
    # a guided (non-TEM) mode request it is spurious and must be rejected.
    reject_near_k0 = (
        not reject_spurious
        and wave_family is not None
        and str(wave_family).lower() != "tem"
        and k0 is not None
        and float(k0) > 0.0
    )
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
        degenerate_rtol = _VECTOR_DEGENERATE_RTOL if (reject_spurious or reject_near_k0) else 1.0e-7
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
        envelope_variation = _vector_mode_envelope_variation_numpy(
            profiles,
            preferred_field_name=preferred_field_name,
        )
        transverse_uniformity = _vector_mode_transverse_uniformity_numpy(
            profiles,
            preferred_field_name=preferred_field_name,
        )
        wall_peak_fraction = _vector_mode_wall_peak_fraction_numpy(
            profiles,
            preferred_field_name=preferred_field_name,
            field_names=field_names,
        )
        near_k0 = (
            reject_near_k0
            and abs(np.imag(beta)) <= 1.0e-6 * max(abs(beta), 1.0)
            and np.real(beta) >= _SPURIOUS_NEAR_K0_BETA_LIMIT * float(k0)
        )
        # F1c: the k0 null branch is identified by its ABSOLUTE transverse
        # uniformity (min/max envelope near 1), not the dx-dependent squared
        # envelope variation that over-rejected legitimate fine-grid / high-f modes.
        planewave_like = (
            transverse_uniformity is not None
            and transverse_uniformity > _SPURIOUS_UNIFORMITY_LIMIT
        )
        # The checkerboard filter applies on the graded (reject_spurious) path. It
        # is NOT enabled on the uniform-isotropic path generically, because that
        # path also serves FREE-SPACE / open TE WavePorts (no metallic wall) whose
        # fundamental transverse mode is legitimately plane-wave-like; the selector
        # cannot tell a closed metallic guide from a free-space aperture from the
        # eps/mu planes alone (audit S1 round-4, executed: enabling the checkerboard
        # filter here rejected the legitimate free-space array-port modes).
        enforce_structure = reject_spurious
        # wall_peak_fraction is computed and recorded purely as a diagnostic; it is
        # NOT a rejection gate. Using it would require knowing that the aperture edge
        # is a metallic (Dirichlet) wall, which holds only for a closed metallic
        # guide -- a signal not available at the selector level. On a graded mode
        # source or a free-space port the aperture edge is a computational truncation,
        # so a legitimate higher-order mode may carry amplitude there. Closed-guide
        # enforcement is deferred with the transverse-operator redesign (open item).
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
        elif near_k0 and planewave_like:
            status = "spurious_near_k0"
        elif enforce_structure and checkerboard_fraction > _VECTOR_CHECKERBOARD_FRACTION_LIMIT:
            status = "checkerboard"
        elif enforce_structure and max_near_duplicate_overlap >= _VECTOR_DUPLICATE_OVERLAP_LIMIT:
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
                "envelope_variation": envelope_variation,
                "transverse_uniformity": transverse_uniformity,
                "wall_peak_fraction": wall_peak_fraction,
                "family_index": family_index,
                "status": status,
                "selected": False,
            }
        )

    # Match the scalar mode path: only modes dominated by the requested
    # tangential E polarization occupy indices in that family.
    if len(family_indices) <= int(mode_index):
        # F1d: never silently substitute another mode. If every candidate for the
        # requested index was rejected, raise a diagnostic error listing why.
        reject_reasons = {"checkerboard", "duplicate", "spurious_near_k0"}
        rejected = sum(entry["status"] in reject_reasons for entry in candidate_diagnostics)
        breakdown = ", ".join(
            f"{status}={sum(1 for e in candidate_diagnostics if e['status'] == status)}"
            for status in sorted(reject_reasons)
        )
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only "
            f"{len(family_indices)} structurally-valid forward modes in the requested polarization "
            f"family were found after rejecting {rejected} spurious candidates ({breakdown}). "
            "The guided mode selector will not substitute a spurious eigenvector; refine the "
            "aperture grid or check the requested mode/polarization."
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
    wave_family: str | None = None,
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
            # ARPACK's iterative "LR" search can fail to converge on fine
            # transverse grids (e.g. refined hollow-waveguide TE cross-sections).
            # Fall back to a dense eigen-decomposition, which is deterministic and
            # tractable for benchmark-scale cross-sections. This changes nothing
            # when ARPACK converges; it only unblocks the fine grid tiers that the
            # wave-level convergence study (audit S1) requires.
            return _solve_vector_mode_eigenpair_dense(
                eps_planes,
                mu_planes,
                k0=k0,
                du=du,
                dv=dv,
                mode_index=int(mode_index),
                field_names=field_names,
                preferred_field_name=preferred_field_name,
                wave_family=wave_family,
            )
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
        wave_family=wave_family,
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
    wave_family: str | None = None,
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
        wave_family=wave_family,
    )
