"""TEM / quasistatic line mode family (electrostatic potentials on PEC lines)."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as scipy_sparse_linalg

from ....constants import ETA_0
from .common import (
    _PEC_OCCUPANCY_THRESHOLD,
    _boundary_connected_conductor,
    _label_connected_components,
)


def _quasistatic_laplace_energy(
    eps_node: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_value: np.ndarray,
    *,
    du: float,
    dv: float,
):
    """Solve ``div(eps grad phi) = 0`` (Dirichlet on ``fixed_mask``) and return energy.

    Variable-coefficient 5-point Laplace on the node grid with face permittivities
    (arithmetic node average -- the Yee dielectric average). Assembled as a sparse
    symmetric-positive-definite system over the free interior nodes and solved directly
    with the same SciPy sparse path the transverse mode operators already use (a one-time
    mode-setup solve, not the FDTD hot loop). Returns ``(phi, energy)`` with ``energy =
    sum_faces eps_face (grad phi)^2 * du * dv`` (proportional to the per-length
    electrostatic energy; the ``1/2`` and area cancel in the ``eps_eff = C / C0`` ratio).
    """
    eps_node = np.asarray(eps_node, dtype=np.float64)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    fixed_value = np.asarray(fixed_value, dtype=np.float64)
    nu, nv = eps_node.shape
    phi = np.where(fixed_mask, fixed_value, 0.0)
    free = ~fixed_mask.copy()
    free[0, :] = False
    free[-1, :] = False
    free[:, 0] = False
    free[:, -1] = False

    eps_up = 0.5 * (eps_node[:-1, :] + eps_node[1:, :])   # u faces  (nu-1, nv)
    eps_vp = 0.5 * (eps_node[:, :-1] + eps_node[:, 1:])   # v faces  (nu, nv-1)
    wu = 1.0 / (float(du) * float(du))
    wv = 1.0 / (float(dv) * float(dv))

    free_indices = np.nonzero(free.reshape(-1))[0]
    if free_indices.size == 0:
        raise ValueError("Quasi-static line-mode solve has no free nodes; the plane is fully constrained.")
    index_of = -np.ones((nu * nv,), dtype=np.int64)
    index_of[free_indices] = np.arange(free_indices.size, dtype=np.int64)

    flat = np.arange(nu * nv, dtype=np.int64).reshape(nu, nv)
    rows = []
    cols = []
    data = []
    rhs = np.zeros((free_indices.size,), dtype=np.float64)

    def _add_coupling(coeff2d, node_flat, neigh_flat, node_mask):
        # coeff2d, node_flat, neigh_flat, node_mask all share one 2D shape.
        sel = node_mask
        c = coeff2d[sel]
        n_idx = index_of[node_flat[sel]]
        m_idx = index_of[neigh_flat[sel]]
        # diagonal for the free node
        rows.append(n_idx)
        cols.append(n_idx)
        data.append(c)
        neighbour_free = m_idx >= 0
        rows.append(n_idx[neighbour_free])
        cols.append(m_idx[neighbour_free])
        data.append(-c[neighbour_free])
        neighbour_fixed = ~neighbour_free
        np.add.at(rhs, n_idx[neighbour_fixed], c[neighbour_fixed] * phi.reshape(-1)[neigh_flat[sel][neighbour_fixed]])

    # For each free node, add its four face couplings. Build per-direction on the
    # interior slice so the neighbour indices stay in-bounds.
    # -u neighbour (i-1): face eps_up[i-1, j], present for i>=1.
    node = flat[1:, :]; neigh = flat[:-1, :]; coeff = eps_up * wu; mask = free[1:, :]
    _add_coupling(coeff, node, neigh, mask)
    # +u neighbour (i+1): face eps_up[i, j], present for i<=nu-2.
    node = flat[:-1, :]; neigh = flat[1:, :]; coeff = eps_up * wu; mask = free[:-1, :]
    _add_coupling(coeff, node, neigh, mask)
    # -v neighbour (j-1): face eps_vp[i, j-1], present for j>=1.
    node = flat[:, 1:]; neigh = flat[:, :-1]; coeff = eps_vp * wv; mask = free[:, 1:]
    _add_coupling(coeff, node, neigh, mask)
    # +v neighbour (j+1): face eps_vp[i, j], present for j<=nv-2.
    node = flat[:, :-1]; neigh = flat[:, 1:]; coeff = eps_vp * wv; mask = free[:, :-1]
    _add_coupling(coeff, node, neigh, mask)

    matrix = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(free_indices.size, free_indices.size),
    )
    solution = scipy_sparse_linalg.spsolve(matrix, rhs)
    phi.reshape(-1)[free_indices] = solution

    grad_u = (phi[1:, :] - phi[:-1, :]) / float(du)
    grad_v = (phi[:, 1:] - phi[:, :-1]) / float(dv)
    energy = float(
        (np.sum(eps_up * grad_u * grad_u) + np.sum(eps_vp * grad_v * grad_v)) * float(du) * float(dv)
    )
    return phi, energy


def _tem_signal_potentials(pec_occupancy: torch.Tensor, mode_index: int) -> list[float]:
    """Drive potentials for the quasi-static line mode selected by ``mode_index``.

    Counts the isolated (non-grounded) signal conductors on the aperture and maps the
    requested ``mode_index`` to a driving-potential vector for
    :func:`_solve_quasistatic_line_modes`. A single-signal line (coax, single-strip
    microstrip port) carries one mode driven ``[1.0]``. A two-signal coupled line
    (a differential pair whose aperture spans both strips) carries an even/common mode
    (``mode_index 0`` -> ``[1, 1]``) and an odd/differential mode (``mode_index 1`` ->
    ``[1, -1]``). More than two coupled signal conductors need explicit user-supplied
    potentials and are rejected (a routing boundary, not a solver capability).
    """
    conductor = pec_occupancy >= _PEC_OCCUPANCY_THRESHOLD
    grounded = _boundary_connected_conductor(conductor)
    isolated = (conductor & ~grounded).detach().to(dtype=torch.bool).cpu().numpy()
    _labels, count = _label_connected_components(isolated)
    index = int(mode_index)
    if count == 0:
        raise ValueError(
            "A quasi-TEM WavePort requires at least one isolated signal conductor and a "
            "grounded aperture boundary; none was found on the mode plane."
        )
    if count == 1:
        if index != 0:
            raise ValueError(
                "A single-signal quasi-TEM line carries one mode; mode_index must be 0."
            )
        return [1.0]
    if count == 2:
        if index == 0:
            return [1.0, 1.0]
        if index == 1:
            return [1.0, -1.0]
        raise ValueError(
            "A two-signal coupled quasi-TEM line carries two modes (mode_index 0 = even/"
            "common, 1 = odd/differential)."
        )
    raise ValueError(
        f"Quasi-TEM drive for {count} coupled signal conductors is not defined; a "
        "cross-section with more than two signal conductors needs explicit driving "
        "potentials."
    )


def _solve_quasistatic_line_modes(
    eps_planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pec_occupancy: torch.Tensor,
    *,
    k0: float,
    du: float,
    dv: float,
    field_names,
    signal_potentials,
    threshold: float = _PEC_OCCUPANCY_THRESHOLD,
):
    """Quasi-static (electrostatic) TEM/quasi-TEM line mode on an interior-PEC plane.

    ``eps_planes`` is the ``(eps_uu, eps_vv, eps_ww)`` node-grid permittivity (walls
    included); ``pec_occupancy`` is the node conductor occupancy. Conductors touching the
    aperture boundary are grounded (potential 0); each isolated interior conductor is a
    signal conductor. ``signal_potentials`` sets the driving potential on the isolated
    conductors ordered by ascending ``(u, v)`` centroid -- ``[1.0]`` for a single-signal
    line (coax, microstrip), ``[1.0, 1.0]`` for the even (common) and ``[1.0, -1.0]`` for
    the odd (differential) mode of a two-signal pair.

    The effective permittivity is the capacitance ratio ``eps_eff = C / C0`` where ``C``
    is solved with the true (inhomogeneous) permittivity and ``C0`` with vacuum, and the
    propagation constant is ``beta = k0 sqrt(eps_eff)``. The transverse fields are the
    electrostatic ``E = -grad phi`` with ``H`` from the effective wave impedance
    ``Z_eff = eta0 / sqrt(eps_eff)`` (so the forward Poynting power is positive). Returns
    a dict with ``beta``, ``eps_eff``, ``potential``, ``component_profiles``, and the
    conductor / isolated-conductor counts.
    """
    eps_uu_node, eps_vv_node, eps_ww_node = eps_planes
    eps_uu_real = torch.real(eps_uu_node) if eps_uu_node.is_complex() else eps_uu_node
    eps_vv_real = torch.real(eps_vv_node) if eps_vv_node.is_complex() else eps_vv_node
    target_device = eps_uu_node.device
    eps_node = (0.5 * (eps_uu_real + eps_vv_real)).detach().to(dtype=torch.float64).cpu().numpy()
    occupancy = np.asarray(pec_occupancy.detach().to(dtype=torch.float64).cpu().numpy())
    conductor = occupancy >= threshold
    if not bool(np.any(conductor)):
        raise ValueError(
            "Quasi-static line-mode solve found no conductor on the mode plane; the PEC "
            "structure is under-resolved or absent."
        )
    grounded = _boundary_connected_conductor(torch.as_tensor(conductor)).cpu().numpy()
    isolated = conductor & ~grounded
    labels, count = _label_connected_components(isolated)
    signal_potentials = list(signal_potentials)
    if count == 0:
        raise ValueError(
            "Quasi-static line-mode solve requires at least one isolated signal conductor and a "
            "grounded aperture boundary; none was found (single-conductor / hollow cross-section)."
        )
    if count != len(signal_potentials):
        raise ValueError(
            f"Quasi-static line-mode solve found {count} isolated signal conductor(s) but "
            f"{len(signal_potentials)} driving potential(s) were supplied."
        )
    # Order isolated conductors by ascending (u, v) centroid for a deterministic assignment.
    centroids = []
    for label in range(1, count + 1):
        rows, cols = np.nonzero(labels == label)
        centroids.append((float(rows.mean()), float(cols.mean()), label))
    centroids.sort()

    fixed_mask = conductor.copy()
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    fixed_value = np.zeros_like(eps_node)
    for potential, (_, _, label) in zip(signal_potentials, centroids):
        fixed_value = np.where(labels == label, float(potential), fixed_value)

    phi_eps, energy_eps = _quasistatic_laplace_energy(eps_node, fixed_mask, fixed_value, du=du, dv=dv)
    _, energy_vac = _quasistatic_laplace_energy(np.ones_like(eps_node), fixed_mask, fixed_value, du=du, dv=dv)
    if energy_vac <= 0.0:
        raise RuntimeError("Quasi-static line-mode solve produced a non-positive vacuum capacitance.")
    eps_eff = float(energy_eps / energy_vac)
    if eps_eff <= 0.0:
        raise RuntimeError("Quasi-static line-mode solve produced a non-positive effective permittivity.")
    beta_value = float(k0) * math.sqrt(eps_eff)

    du_f = float(du)
    dv_f = float(dv)
    electric_u = np.zeros_like(phi_eps)
    electric_v = np.zeros_like(phi_eps)
    electric_u[1:-1, :] = -(phi_eps[2:, :] - phi_eps[:-2, :]) / (2.0 * du_f)
    electric_u[0, :] = -(phi_eps[1, :] - phi_eps[0, :]) / du_f
    electric_u[-1, :] = -(phi_eps[-1, :] - phi_eps[-2, :]) / du_f
    electric_v[:, 1:-1] = -(phi_eps[:, 2:] - phi_eps[:, :-2]) / (2.0 * dv_f)
    electric_v[:, 0] = -(phi_eps[:, 1] - phi_eps[:, 0]) / dv_f
    electric_v[:, -1] = -(phi_eps[:, -1] - phi_eps[:, -2]) / dv_f
    impedance = ETA_0 / math.sqrt(eps_eff)
    magnetic_u = -electric_v / impedance
    magnetic_v = electric_u / impedance

    def _to_device(array):
        return torch.as_tensor(array, dtype=torch.float64, device=target_device)

    component_profiles = {
        field_names[0]: _to_device(electric_u),
        field_names[1]: _to_device(electric_v),
        field_names[2]: _to_device(magnetic_u),
        field_names[3]: _to_device(magnetic_v),
    }
    return {
        "beta": _to_device(np.asarray(beta_value)),
        "eps_eff": eps_eff,
        "potential": _to_device(phi_eps),
        "component_profiles": component_profiles,
        "conductor_count": int(np.count_nonzero(conductor)),
        "isolated_conductor_count": int(count),
    }


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

    impedance = ETA_0 * torch.sqrt(mu_relative / eps_relative)
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
