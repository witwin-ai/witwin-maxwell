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



def _native_tem_metrics(eps_node, *, du, dv, coords_u=None, coords_v=None):
    """Actual primal/dual transverse metrics; scalar callers retain their path."""
    if (coords_u is None)!=(coords_v is None):
        raise ValueError("Native TEM coordinates require both transverse axes.")
    if coords_u is None:
        return None
    steps=[];duals=[]
    uniform=True
    for axis,coordinates in enumerate((coords_u,coords_v)):
        coordinates=torch.as_tensor(coordinates,device=eps_node.device,dtype=torch.float64)
        if coordinates.ndim!=1 or coordinates.numel()!=eps_node.shape[axis]:
            raise ValueError("Native TEM coordinate lengths must match the node potential plane.")
        delta=torch.diff(coordinates)
        if not bool(torch.all(torch.isfinite(coordinates))) or not bool(torch.all(delta>0)):
            raise ValueError("Native TEM coordinates must be finite and strictly increasing.")
        uniform=uniform and bool(torch.allclose(delta,delta[:1].expand_as(delta),rtol=1e-10,atol=0.))
        steps.append(delta)
        duals.append(torch.cat((delta[:1]/2,(delta[:-1]+delta[1:])/2,delta[-1:]/2)))
    if uniform and du is not None and dv is not None:
        return None
    return (*steps,*duals)


def _quasistatic_laplace_energy_torch(eps_node, fixed_mask, fixed_value, *, du, dv,
                                       coords_u=None, coords_v=None):
    """Device-resident version of the same node/face Dirichlet energy solve."""
    from ....electrostatic.runtime import _pcg

    eps_node = eps_node.to(dtype=torch.float64)
    fixed_mask = torch.as_tensor(fixed_mask, device=eps_node.device, dtype=torch.bool)
    fixed_value = torch.as_tensor(fixed_value, device=eps_node.device, dtype=eps_node.dtype)
    free = ~fixed_mask
    free = free.clone()
    free[0, :] = free[-1, :] = False
    free[:, 0] = free[:, -1] = False
    if not bool(free.any()):
        raise ValueError("Quasi-static line-mode solve has no free nodes.")
    metrics=_native_tem_metrics(eps_node,du=du,dv=dv,coords_u=coords_u,coords_v=coords_v)
    if metrics is None:
        gu = .5 * (eps_node[:-1, :] + eps_node[1:, :]) / du**2
        gv = .5 * (eps_node[:, :-1] + eps_node[:, 1:]) / dv**2
    else:
        primal_u,primal_v,dual_u,dual_v=metrics
        # Finite-volume face conductances form a symmetric positive operator.
        # Each face uses its own physical normal length and transverse dual width.
        gu=.5*(eps_node[:-1,:]+eps_node[1:,:])*dual_v[None,:]/primal_u[:,None]
        gv=.5*(eps_node[:,:-1]+eps_node[:,1:])*dual_u[:,None]/primal_v[None,:]

    def laplace(phi):
        fu = gu * (phi[:-1, :] - phi[1:, :])
        fv = gv * (phi[:, :-1] - phi[:, 1:])
        return (torch.nn.functional.pad(fu, (0, 0, 0, 1))
                - torch.nn.functional.pad(fu, (0, 0, 1, 0))
                + torch.nn.functional.pad(fv, (0, 1, 0, 0))
                - torch.nn.functional.pad(fv, (1, 0, 0, 0)))

    phi = torch.where(fixed_mask, fixed_value, 0.)
    rhs = -laplace(phi)[free]
    diagonal = (torch.nn.functional.pad(gu, (0, 0, 0, 1)) + torch.nn.functional.pad(gu, (0, 0, 1, 0))
                + torch.nn.functional.pad(gv, (0, 1, 0, 0)) + torch.nn.functional.pad(gv, (1, 0, 0, 0)))

    def apply(value):
        full = torch.zeros_like(phi).masked_scatter(free, value)
        return laplace(full)[free]

    solution, report = _pcg(apply, rhs, diagonal[free].reciprocal(), tol=1e-11,
                            max_iterations=max(500, 8 * sum(eps_node.shape)))
    if not report.converged:
        raise RuntimeError(f"Quasi-static CUDA potential did not converge: residual={report.residual}.")
    phi = phi.masked_scatter(free, solution)
    energy = ((gu * (phi[1:, :] - phi[:-1, :]).square()).sum()
              + (gv * (phi[:, 1:] - phi[:, :-1]).square()).sum())
    if metrics is None:
        energy=energy*du*dv
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
    coords_u=None,
    coords_v=None,
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
    native edge differences of the dielectric potential for E and of the vacuum
    potential for H, scaled by ``sqrt(eps_eff)/eta0``. This preserves the vacuum
    capacitance contour current and the forward power identity. Returns
    a dict with ``beta``, ``eps_eff``, ``potential``, ``component_profiles``, and the
    conductor / isolated-conductor counts.
    """
    eps_uu_node, eps_vv_node, eps_ww_node = eps_planes
    eps_uu_real = torch.real(eps_uu_node) if eps_uu_node.is_complex() else eps_uu_node
    eps_vv_real = torch.real(eps_vv_node) if eps_vv_node.is_complex() else eps_vv_node
    target_device = eps_uu_node.device
    eps_node = (0.5 * (eps_uu_real + eps_vv_real)).detach().to(dtype=torch.float64)
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
    fixed_value = np.zeros(tuple(eps_node.shape), dtype=np.float64)
    for potential, (_, _, label) in zip(signal_potentials, centroids):
        fixed_value = np.where(labels == label, float(potential), fixed_value)

    phi_eps, energy_eps = _quasistatic_laplace_energy_torch(eps_node, fixed_mask, fixed_value, du=du, dv=dv,coords_u=coords_u,coords_v=coords_v)
    phi_vac, energy_vac = _quasistatic_laplace_energy_torch(torch.ones_like(eps_node), fixed_mask, fixed_value, du=du, dv=dv,coords_u=coords_u,coords_v=coords_v)
    if energy_vac <= 0.0:
        raise RuntimeError("Quasi-static line-mode solve produced a non-positive vacuum capacitance.")
    eps_eff = float(energy_eps / energy_vac)
    if eps_eff <= 0.0:
        raise RuntimeError("Quasi-static line-mode solve produced a non-positive effective permittivity.")
    beta_value = float(k0) * math.sqrt(eps_eff)

    metrics=_native_tem_metrics(eps_node,du=du,dv=dv,coords_u=coords_u,coords_v=coords_v)
    if metrics is None:
        du_f,dv_f=float(du),float(dv)
    else:
        du_f,dv_f=metrics[0][:,None],metrics[1][None,:]
    electric_u = -(phi_eps[1:, :] - phi_eps[:-1, :]) / du_f
    electric_v = -(phi_eps[:, 1:] - phi_eps[:, :-1]) / dv_f
    vacuum_u = -(phi_vac[1:, :] - phi_vac[:-1, :]) / du_f
    vacuum_v = -(phi_vac[:, 1:] - phi_vac[:, :-1]) / dv_f
    impedance = ETA_0 / math.sqrt(eps_eff)
    magnetic_u = -vacuum_v / impedance
    magnetic_v = vacuum_u / impedance

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
        "vacuum_potential": _to_device(phi_vac),
        "loaded_energy": energy_eps,
        "vacuum_energy": energy_vac,
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
    time_step=None,
    normal_spacing=None,
    coords_u=None,
    coords_v=None,
):
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
    eps_relative = 0.5 * (eps_u + eps_v)
    mu_relative = 0.5 * (mu_u + mu_v)
    eps_values = eps_relative[dielectric]
    mu_values = mu_relative[dielectric]
    complex_medium = any(value.is_complex() and bool(torch.any(value.imag != 0)) for value in (*eps_planes,*mu_planes))
    if complex_medium:
        for component, reference in (
            *((value, eps_values[0]) for value in eps_planes),
            *((value, mu_values[0]) for value in mu_planes),
        ):
            if not torch.allclose(component[dielectric],reference.expand_as(component[dielectric]),rtol=1e-4,atol=1e-6):
                raise NotImplementedError("Lossy TEM requires a homogeneous isotropic cross-section.")
        if bool(eps_values[0].real <= 0) or bool(mu_values[0].real <= 0):
            raise ValueError("Lossy TEM requires positive real epsilon and mu.")
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

    potentials = _tem_signal_potentials(pec_occupancy, int(mode_index))
    # A constant complex medium multiplies the Laplace equation by a scalar.
    # Both E and H therefore use exactly the same real vacuum potential.
    vacuum = tuple(torch.ones_like(torch.real(component)) for component in eps_planes)
    line = _solve_quasistatic_line_modes(vacuum, pec_occupancy, k0=k0,
        du=du, dv=dv, field_names=field_names, signal_potentials=potentials,
        coords_u=coords_u,coords_v=coords_v)
    profiles = line["component_profiles"]
    # Propagation constants retain RF phase precision independently of the
    # field storage dtype; promoting an already rounded beta cannot restore it.
    eps_value = eps_values[0].to(
        dtype=torch.complex128 if eps_values.is_complex() else torch.float64
    )
    mu_value = mu_values[0].to(
        dtype=torch.complex128 if mu_values.is_complex() else torch.float64
    )
    magnetic_scale = torch.sqrt(eps_value / mu_value)
    profiles[field_names[2]] = profiles[field_names[2]] * magnetic_scale
    profiles[field_names[3]] = profiles[field_names[3]] * magnetic_scale
    beta = float(k0) * torch.sqrt(eps_value * mu_value)
    if time_step is not None and normal_spacing is not None:
        from ....constants import C_0
        angle = .5 * float(k0) * C_0 * time_step
        beta = (2 / normal_spacing) * torch.asin(
            (normal_spacing / (C_0 * time_step)) * math.sin(angle) * torch.sqrt(eps_value * mu_value))
    return beta, profiles, int(torch.count_nonzero(conductor).item())


# --------------------------------------------------------------------------- #
# Multi-conductor terminal line basis (TerminalWavePort).                       #
# --------------------------------------------------------------------------- #
def _terminal_conductor_labels(pec_occupancy, *, terminal_masks, reference_mask, plane_shape,
                               threshold: float = _PEC_OCCUPANCY_THRESHOLD) -> np.ndarray:
    """Assign every conductor component on the port plane to a terminal or the reference.

    ``terminal_masks`` holds, per terminal, the flat plane-local node indices covered
    by that terminal's declared structure; ``reference_mask`` does the same for the
    reference structures. Returns an ``int64`` map with ``-1`` for dielectric nodes,
    ``0`` for reference conductor nodes and ``k + 1`` for terminal ``k``. Components
    touching the aperture boundary are grounded by the potential solve and therefore
    belong to the reference. Ambiguity is rejected: a component that touches two
    terminals, a terminal and the reference, or no declared conductor at all.
    """
    occupancy = pec_occupancy.detach().to(dtype=torch.float64).cpu().numpy()
    if tuple(occupancy.shape) != tuple(int(value) for value in plane_shape):
        raise ValueError(
            f"Terminal port plane {tuple(occupancy.shape)} does not match the compiled "
            f"aperture {tuple(plane_shape)}."
        )
    conductor = occupancy >= threshold
    grounded = _boundary_connected_conductor(torch.as_tensor(conductor)).cpu().numpy()
    labels, count = _label_connected_components(conductor)
    flat_labels = labels.reshape(-1)
    terminal_sets = [np.zeros(flat_labels.shape, dtype=bool) for _ in terminal_masks]
    for target, mask in zip(terminal_sets, terminal_masks):
        target[np.asarray(mask, dtype=np.int64)] = True
    reference = np.zeros(flat_labels.shape, dtype=bool)
    reference[np.asarray(reference_mask, dtype=np.int64)] = True
    grounded_flat = grounded.reshape(-1)
    label_map = np.full(flat_labels.shape, -1, dtype=np.int64)
    for component in range(1, count + 1):
        nodes = flat_labels == component
        hits = [index for index, target in enumerate(terminal_sets) if bool(np.any(target & nodes))]
        on_reference = bool(np.any(reference & nodes)) or bool(np.any(grounded_flat & nodes))
        if len(hits) > 1:
            raise ValueError(
                f"Terminals {tuple(index + 1 for index in hits)} are electrically connected on "
                "the port plane; each terminal must be a distinct conductor."
            )
        if hits and on_reference:
            raise ValueError(
                f"Terminal {hits[0] + 1} is connected to the reference conductor or the "
                "aperture boundary on the port plane."
            )
        if hits:
            label_map[nodes] = hits[0] + 1
        elif on_reference:
            label_map[nodes] = 0
        else:
            raise ValueError(
                "The port plane contains a floating conductor that is neither a declared "
                "terminal nor part of the reference; its potential is ambiguous."
            )
    for index in range(len(terminal_masks)):
        if not bool(np.any(label_map == index + 1)):
            raise ValueError(
                f"Terminal {index + 1} has no resolved conductor on the port plane."
            )
    return label_map.reshape(occupancy.shape)


def _terminal_face_conductances(eps_node, *, du, dv, coords_u=None, coords_v=None):
    """Face conductances of the node/face potential operator and its energy scale."""
    metrics = _native_tem_metrics(eps_node, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    if metrics is None:
        gu = .5 * (eps_node[:-1, :] + eps_node[1:, :]) / du**2
        gv = .5 * (eps_node[:, :-1] + eps_node[:, 1:]) / dv**2
        return gu, gv, float(du) * float(dv)
    primal_u, primal_v, dual_u, dual_v = metrics
    gu = .5 * (eps_node[:-1, :] + eps_node[1:, :]) * dual_v[None, :] / primal_u[:, None]
    gv = .5 * (eps_node[:, :-1] + eps_node[:, 1:]) * dual_u[:, None] / primal_v[None, :]
    return gu, gv, 1.0


def _terminal_fixed_values(label_map, potentials):
    fixed_mask = label_map >= 0
    fixed_mask[0, :] = fixed_mask[-1, :] = True
    fixed_mask[:, 0] = fixed_mask[:, -1] = True
    fixed_value = np.zeros(label_map.shape, dtype=np.float64)
    for index, potential in enumerate(potentials):
        fixed_value[label_map == index + 1] = float(potential)
    return fixed_mask, fixed_value


def _solve_terminal_line_basis(eps_node, label_map, *, du, dv, coords_u=None, coords_v=None):
    """Per-unit-length terminal capacitance matrices and the quasi-TEM modal basis.

    One potential solve per terminal (potential 1 on terminal ``k``, 0 elsewhere)
    is run with the true permittivity and with vacuum. ``C`` and ``C0`` are the
    discrete bilinear energy forms of those solutions (F/m). The quasi-TEM modes
    solve ``C v = eps_eff C0 v`` (``L = mu0 eps0 C0^-1``); the eigenvectors are
    therefore both ``C``- and ``C0``-orthogonal. Modes are ordered by descending
    ``eps_eff``. Degenerate clusters are rotated onto eigenvectors of ``C0`` so the
    basis is deterministic, and each column is scaled to a unit, positive peak entry.
    """
    from ....constants import EPSILON_0

    count = int(label_map.max())
    eps_node = eps_node.to(dtype=torch.float64)
    vacuum = torch.ones_like(eps_node)
    forms = []
    for medium in (eps_node, vacuum):
        gu, gv, scale = _terminal_face_conductances(medium, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
        potentials = []
        for index in range(count):
            unit = [0.0] * count
            unit[index] = 1.0
            fixed_mask, fixed_value = _terminal_fixed_values(label_map, unit)
            phi, _energy = _quasistatic_laplace_energy_torch(
                medium, fixed_mask, fixed_value, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
            potentials.append(phi)
        grad_u = torch.stack([phi[1:, :] - phi[:-1, :] for phi in potentials])
        grad_v = torch.stack([phi[:, 1:] - phi[:, :-1] for phi in potentials])
        form = (torch.einsum("aij,bij,ij->ab", grad_u, grad_u, gu)
                + torch.einsum("aij,bij,ij->ab", grad_v, grad_v, gv)) * scale * EPSILON_0
        forms.append(.5 * (form + form.transpose(0, 1)))
    capacitance, vacuum_capacitance = forms
    vacuum_values, vacuum_vectors = torch.linalg.eigh(vacuum_capacitance)
    if not bool(torch.all(vacuum_values > 0)):
        raise RuntimeError("Terminal vacuum capacitance matrix is not positive definite.")
    inverse_root = vacuum_vectors @ torch.diag(vacuum_values.rsqrt()) @ vacuum_vectors.transpose(0, 1)
    reduced = inverse_root @ capacitance @ inverse_root
    reduced = .5 * (reduced + reduced.transpose(0, 1))
    eps_eff, rotated = torch.linalg.eigh(reduced)
    order = torch.argsort(eps_eff, descending=True)
    eps_eff, rotated = eps_eff[order], rotated[:, order]
    if not bool(torch.all(eps_eff > 0)):
        raise RuntimeError("Terminal quasi-TEM basis produced a non-positive effective permittivity.")
    tolerance = 1.0e-6 * float(eps_eff.abs().max())
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and float(eps_eff[stop - 1] - eps_eff[stop]) <= tolerance:
            stop += 1
        if stop - start > 1:
            block = rotated[:, start:stop]
            tie = block.transpose(0, 1) @ vacuum_capacitance @ block
            _values, local = torch.linalg.eigh(.5 * (tie + tie.transpose(0, 1)))
            rotated[:, start:stop] = block @ local
        start = stop
    modal = inverse_root @ rotated
    for column in range(count):
        values = modal[:, column]
        magnitude = values.abs()
        # First entry within roundoff of the peak: symmetric pairs get a stable sign.
        peak = int(torch.nonzero(magnitude >= (1.0 - 1.0e-6) * magnitude.max())[0])
        modal[:, column] = values / values[peak]
    return {
        "capacitance": capacitance,
        "vacuum_capacitance": vacuum_capacitance,
        "modal_potentials": modal,
        "eps_eff": eps_eff,
    }


def _solve_terminal_line_mode(eps_planes, label_map, potentials, *, k0, du, dv, field_names,
                              coords_u=None, coords_v=None):
    """Quasi-TEM transverse fields of one terminal modal potential vector.

    For a generalized eigenvector of ``(C, C0)`` the vacuum potential that carries
    the modal currents is proportional to the dielectric potential, so the fields
    use the same native edge differences and scaling as the two-conductor
    quasi-static line mode: ``E = -grad(phi_eps)`` and
    ``H = z x (-grad(phi_vac)) sqrt(eps_eff) / eta0``.
    """
    eps_uu, eps_vv, _ = eps_planes
    eps_uu = torch.real(eps_uu) if eps_uu.is_complex() else eps_uu
    eps_vv = torch.real(eps_vv) if eps_vv.is_complex() else eps_vv
    target_device = eps_uu.device
    eps_node = (0.5 * (eps_uu + eps_vv)).detach().to(dtype=torch.float64)
    fixed_mask, fixed_value = _terminal_fixed_values(label_map, potentials)
    phi_eps, energy_eps = _quasistatic_laplace_energy_torch(
        eps_node, fixed_mask, fixed_value, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    phi_vac, energy_vac = _quasistatic_laplace_energy_torch(
        torch.ones_like(eps_node), fixed_mask, fixed_value, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    if not bool(energy_vac > 0.0):
        raise RuntimeError("Terminal quasi-TEM mode produced a non-positive vacuum energy.")
    eps_eff = float(energy_eps / energy_vac)
    if eps_eff <= 0.0:
        raise RuntimeError("Terminal quasi-TEM mode produced a non-positive effective permittivity.")
    metrics = _native_tem_metrics(eps_node, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    if metrics is None:
        du_f, dv_f = float(du), float(dv)
    else:
        du_f, dv_f = metrics[0][:, None], metrics[1][None, :]
    electric_u = -(phi_eps[1:, :] - phi_eps[:-1, :]) / du_f
    electric_v = -(phi_eps[:, 1:] - phi_eps[:, :-1]) / dv_f
    vacuum_u = -(phi_vac[1:, :] - phi_vac[:-1, :]) / du_f
    vacuum_v = -(phi_vac[:, 1:] - phi_vac[:, :-1]) / dv_f
    impedance = ETA_0 / math.sqrt(eps_eff)
    profiles = {
        field_names[0]: electric_u.to(device=target_device, dtype=torch.float64),
        field_names[1]: electric_v.to(device=target_device, dtype=torch.float64),
        field_names[2]: (-vacuum_v / impedance).to(device=target_device, dtype=torch.float64),
        field_names[3]: (vacuum_u / impedance).to(device=target_device, dtype=torch.float64),
    }
    beta = torch.as_tensor(float(k0) * math.sqrt(eps_eff), device=target_device, dtype=torch.float64)
    wall_fraction = _terminal_wall_charge_fraction(
        eps_node, phi_eps, label_map, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    return beta, profiles, int(np.count_nonzero(label_map >= 0)), eps_eff, wall_fraction


def _terminal_wall_charge_fraction(eps_node, phi, label_map, *, du, dv, coords_u=None, coords_v=None) -> float:
    """Share of the terminal charge that terminates on the artificial aperture walls.

    The potential solve grounds the whole aperture boundary. Boundary nodes that are
    not conductor are an artificial shield; the discrete electric flux they absorb
    (summed node-wise in magnitude), relative to the total terminal charge
    ``sum_k |Q_k|``, measures how strongly the
    aperture truncates the modal field (grid convergent, independent of the band
    choice of an energy fraction).
    """
    gu, gv, _scale = _terminal_face_conductances(eps_node, du=du, dv=dv, coords_u=coords_u, coords_v=coords_v)
    flux_u = gu * (phi[:-1, :] - phi[1:, :])
    flux_v = gv * (phi[:, :-1] - phi[:, 1:])
    outflow = torch.zeros_like(phi)
    outflow[:-1, :] += flux_u
    outflow[1:, :] -= flux_u
    outflow[:, :-1] += flux_v
    outflow[:, 1:] -= flux_v
    labels = torch.as_tensor(label_map, device=phi.device)
    wall = torch.zeros_like(labels, dtype=torch.bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    wall &= labels < 0
    terminal_charge = sum(abs(float(outflow[labels == index].sum())) for index in range(1, int(labels.max()) + 1))
    if terminal_charge <= 0.0:
        return 0.0
    # Node-wise magnitudes: odd modes put opposite charges on opposite walls.
    return float(outflow[wall].abs().sum()) / terminal_charge
