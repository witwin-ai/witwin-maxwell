"""Yee-staggered transverse vector mode family, including the PEC staircase path."""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as scipy_sparse_linalg

from ....constants import ETA_0
from .common import (
    _FULL_VECTOR_DENSE_LIMIT,
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
    _label_connected_components,
)
from .vector import (
    _normalize_vector_mode_profiles_numpy,
    _vector_mode_checkerboard_fraction_numpy,
    _vector_mode_polarization_fraction_numpy,
    _vector_mode_power_inner_product_numpy,
    _vector_mode_power_sign_numpy,
    _vector_mode_transverse_uniformity_numpy,
)


def _yee_half_to_node_first_difference(cells: int, spacing: float):
    """Backward first difference from a Yee half grid to the interior nodes.

    ``cells`` is the number of Yee cells across the aperture along one transverse
    axis (so the wall-to-wall extent is ``cells * spacing``). The half grid holds
    ``cells`` samples at ``(i + 1/2) * spacing`` (``i = 0 .. cells - 1``); the node
    grid holds ``cells + 1`` samples at ``i * spacing`` with the two end nodes lying
    ON the metallic walls. This returns the ``(cells - 1) x cells`` operator ``G``
    that maps a half-grid field to the ``cells - 1`` INTERIOR nodes only::

        (G f)_i = (f_{i+1/2} - f_{i-1/2}) / spacing ,   i = 1 .. cells - 1

    Both half neighbours of every interior node exist, so ``G`` needs no boundary
    ghost. Its negative transpose ``S = -G.T`` is the matching forward difference
    from the interior nodes back to the half grid, with the wall nodes eliminated
    (Dirichlet 0). The two composites reproduce the staggered transverse Laplacians:
    ``G @ S`` is the Dirichlet second difference on the interior nodes and ``S @ G``
    is the Neumann second difference on the half grid (a constant half field lies in
    its null space, which is exactly the ``TE_{m0}`` transverse-uniform direction).
    """
    if cells <= 1:
        raise ValueError("Transverse Yee operator needs at least two cells per axis.")
    interior = cells - 1
    scale = 1.0 / float(spacing)
    rows = np.repeat(np.arange(interior, dtype=np.int64), 2)
    cols = np.empty(2 * interior, dtype=np.int64)
    data = np.empty(2 * interior, dtype=np.float64)
    cols[0::2] = np.arange(interior, dtype=np.int64)      # half index i - 1/2
    cols[1::2] = np.arange(1, interior + 1, dtype=np.int64)  # half index i + 1/2
    data[0::2] = -scale
    data[1::2] = scale
    return sparse.csr_matrix((data, (rows, cols)), shape=(interior, cells))


def _yee_transverse_grids(nu_cells: int, nv_cells: int, du: float, dv: float) -> dict:
    """Yee sample coordinates for the staggered transverse mode operator.

    ``Eu`` lives on the ``u`` half grid and the interior ``v`` nodes; ``Ev`` lives
    on the interior ``u`` nodes and the ``v`` half grid. Returns the 1D coordinate
    arrays and the block shapes so a caller can reshape an eigenvector into its two
    component profiles. Wall-to-wall extents are ``a = nu_cells * du`` (``u``) and
    ``b = nv_cells * dv`` (``v``); the wall nodes sit at ``0`` and the extent.
    """
    u_half = (np.arange(nu_cells, dtype=np.float64) + 0.5) * du
    v_half = (np.arange(nv_cells, dtype=np.float64) + 0.5) * dv
    u_interior_node = np.arange(1, nu_cells, dtype=np.float64) * du
    v_interior_node = np.arange(1, nv_cells, dtype=np.float64) * dv
    return {
        "nu_cells": nu_cells,
        "nv_cells": nv_cells,
        "shape_eu": (nu_cells, nv_cells - 1),          # (u half, v interior node)
        "shape_ev": (nu_cells - 1, nv_cells),          # (u interior node, v half)
        "n_eu": nu_cells * (nv_cells - 1),
        "n_ev": (nu_cells - 1) * nv_cells,
        "eu_u": u_half,
        "eu_v": v_interior_node,
        "ev_u": u_interior_node,
        "ev_v": v_half,
        "extent_u": nu_cells * du,
        "extent_v": nv_cells * dv,
    }


def _yee_transverse_discrete_transverse_wavenumber(order: int, extent: float, spacing: float) -> float:
    """Discrete transverse wavenumber ``k~ = (2/dx) sin(m pi dx / (2 a))``.

    This is the exact eigen-wavenumber of the staggered transverse Laplacian for a
    half-wave of continuum wavenumber ``k = m pi / a`` on a mesh of spacing ``dx``;
    substituting ``sin(m pi u / a)`` (Dirichlet) or ``cos(m pi u / a)`` (Neumann)
    into the second-difference operator returns ``-k~**2`` to machine precision.
    """
    return (2.0 / float(spacing)) * math.sin(order * math.pi * float(spacing) / (2.0 * float(extent)))


def _build_yee_transverse_operator_sparse(
    *,
    nu_cells: int,
    nv_cells: int,
    du: float,
    dv: float,
    k0: float,
    eps_uu=None,
    eps_vv=None,
    eps_ww=None,
    pec_eu=None,
    pec_ev=None,
    pec_ww=None,
):
    """Genuinely Yee-staggered transverse full-vector mode operator ``P``.

    Solves ``P et = beta**2 et`` for the transverse electric field ``et = (Eu, Ev)``
    with the two components kept on their own Yee locations (``Eu`` at the ``u``
    half / ``v`` node grid, ``Ev`` at the ``u`` node / ``v`` half grid). It is the
    exact 2D restriction of the repo's 3D Yee cell (``Ex`` at ``(i+1/2, j)``, ``Ey``
    at ``(i, j+1/2)``, ``Ez`` at ``(i, j)``); the derivation is recorded in
    ``docs/assessments/e1-rf-modes-acceptance-2026-07-19.md``.

    Assumes a non-magnetic medium (``mu = 1``): the RF port families in scope are
    dielectric. ``eps_uu`` / ``eps_vv`` / ``eps_ww`` are the diagonal permittivity
    sampled at the ``Eu`` grid ``(nu_cells, nv_cells-1)``, the ``Ev`` grid
    ``(nu_cells-1, nv_cells)`` and the interior-node grid ``(nu_cells-1,
    nv_cells-1)`` respectively; ``None`` means vacuum (``eps = 1``). The eigenvalue
    is ``beta**2`` and the operator is real; for a homogeneous cross-section it is
    symmetric and the two blocks decouple into scalar transverse Helmholtz problems
    with the physically correct mixed walls (Dirichlet for the tangential component,
    natural/Neumann for the normal component) built in by the staggering.
    """
    if int(nu_cells) <= 1 or int(nv_cells) <= 1:
        raise ValueError("Transverse Yee mode operator needs at least two cells per axis.")
    nu_cells = int(nu_cells)
    nv_cells = int(nv_cells)
    meta = _yee_transverse_grids(nu_cells, nv_cells, du, dv)
    n_eu = meta["n_eu"]
    n_ev = meta["n_ev"]

    gu = _yee_half_to_node_first_difference(nu_cells, du)   # (nu-1) x nu  : u half -> u interior node
    gv = _yee_half_to_node_first_difference(nv_cells, dv)   # (nv-1) x nv  : v half -> v interior node
    su = (-gu.transpose()).tocsr()                          # nu x (nu-1)  : u interior node -> u half
    sv = (-gv.transpose()).tocsr()                          # nv x (nv-1)  : v interior node -> v half

    identity_u_half = sparse.eye(nu_cells, format="csr", dtype=np.float64)
    identity_v_half = sparse.eye(nv_cells, format="csr", dtype=np.float64)
    identity_u_node = sparse.eye(nu_cells - 1, format="csr", dtype=np.float64)
    identity_v_node = sparse.eye(nv_cells - 1, format="csr", dtype=np.float64)

    # Transverse Laplacian pieces (see _yee_half_to_node_first_difference).
    lap_u_neumann = (su @ gu).tocsr()      # nu x nu           (half grid)
    lap_v_neumann = (sv @ gv).tocsr()      # nv x nv           (half grid)
    lap_u_dirichlet = (gu @ su).tocsr()    # (nu-1) x (nu-1)   (interior nodes)
    lap_v_dirichlet = (gv @ sv).tocsr()    # (nv-1) x (nv-1)   (interior nodes)

    # Node <-> component maps used by the eps_ww divergence coupling.
    gu_big = sparse.kron(gu, identity_v_node, format="csr")   # n_d x n_eu : d/du of Eu -> node
    gv_big = sparse.kron(identity_u_node, gv, format="csr")   # n_d x n_ev : d/dv of Ev -> node
    su_big = sparse.kron(su, identity_v_node, format="csr")   # n_eu x n_d : d/du of node -> Eu
    sv_big = sparse.kron(identity_u_node, sv, format="csr")   # n_ev x n_d : d/dv of node -> Ev

    def _component_diag(values, shape) -> sparse.csr_matrix:
        if values is None:
            return sparse.eye(shape[0] * shape[1], format="csr", dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"Yee transverse eps slice shape {tuple(array.shape)} does not match the "
                f"expected component grid {tuple(shape)}."
            )
        return sparse.diags(array.reshape(-1), offsets=0, format="csr")

    shape_d = (nu_cells - 1, nv_cells - 1)
    eps_uu_diag = _component_diag(eps_uu, meta["shape_eu"])
    eps_vv_diag = _component_diag(eps_vv, meta["shape_ev"])
    if eps_ww is None:
        eps_ww_inv_values = np.ones((shape_d[0] * shape_d[1],), dtype=np.float64)
    else:
        eps_ww_array = np.asarray(eps_ww, dtype=np.float64)
        if tuple(eps_ww_array.shape) != shape_d:
            raise ValueError(
                f"Yee transverse eps_ww slice shape {tuple(eps_ww_array.shape)} does not match the "
                f"interior-node grid {shape_d}."
            )
        eps_ww_inv_values = 1.0 / eps_ww_array.reshape(-1)

    # Interior-PEC masking (F2a): a longitudinal node inside a conductor carries no
    # Ew, so it drops from the eps_ww divergence coupling. Zeroing its eps_ww^{-1}
    # diagonal entry removes its contribution to every ``su_big @ eps_ww^{-1} @
    # gu_big`` block symmetrically, exactly as the outer Dirichlet walls do -- no
    # penalty term, no asymmetry (the term is ``-gu_big^T D gu_big`` with D diagonal).
    pec_ww_mask = None
    if pec_ww is not None:
        pec_ww_mask = np.asarray(pec_ww, dtype=bool)
        if tuple(pec_ww_mask.shape) != shape_d:
            raise ValueError(
                f"Yee transverse pec_ww mask shape {tuple(pec_ww_mask.shape)} does not match the "
                f"interior-node grid {shape_d}."
            )
        eps_ww_inv_values = np.where(pec_ww_mask.reshape(-1), 0.0, eps_ww_inv_values)
    eps_ww_inv_diag = sparse.diags(eps_ww_inv_values, offsets=0, format="csr")

    k0_sq = float(k0) * float(k0)

    # P_uu = d/dv(d/dv) + k0^2 eps_uu + d/du eps_ww^{-1} d/du eps_uu
    p_uu = (
        sparse.kron(identity_u_half, lap_v_dirichlet, format="csr")
        + k0_sq * eps_uu_diag
        + su_big @ eps_ww_inv_diag @ gu_big @ eps_uu_diag
    )
    # P_vv = d/du(d/du) + k0^2 eps_vv + d/dv eps_ww^{-1} d/dv eps_vv
    p_vv = (
        sparse.kron(lap_u_dirichlet, identity_v_half, format="csr")
        + k0_sq * eps_vv_diag
        + sv_big @ eps_ww_inv_diag @ gv_big @ eps_vv_diag
    )
    # P_uv (Ev -> Eu) = -d/dv d/du + d/du eps_ww^{-1} d/dv eps_vv
    p_uv = -sparse.kron(su, gv, format="csr") + su_big @ eps_ww_inv_diag @ gv_big @ eps_vv_diag
    # P_vu (Eu -> Ev) = -d/du d/dv + d/dv eps_ww^{-1} d/du eps_uu
    p_vu = -sparse.kron(gu, sv, format="csr") + sv_big @ eps_ww_inv_diag @ gu_big @ eps_uu_diag

    operator = sparse.bmat([[p_uu, p_uv], [p_vu, p_vv]], format="csr")
    meta["operator"] = operator
    meta["block_uu"] = p_uu.tocsr()
    meta["block_vv"] = p_vv.tocsr()
    meta["block_uv"] = p_uv.tocsr()
    meta["block_vu"] = p_vu.tocsr()
    meta["k0"] = float(k0)
    meta["du"] = float(du)
    meta["dv"] = float(dv)
    # Longitudinal-field reconstruction (E1b) reuses the staggered first
    # differences: ``gu_big``/``gv_big`` take d/du(Eu)/d/dv(Ev) to interior nodes
    # (Gauss law for Ew); ``su_big``/``sv_big`` take a node field back to the Eu/Ev
    # half grids (the d/du Ew, d/dv Ew terms of the transverse H-curls).
    meta["gu_big"] = gu_big
    meta["gv_big"] = gv_big
    meta["su_big"] = su_big
    meta["sv_big"] = sv_big

    # Stacked active-unknown mask over (Eu, Ev). A transverse component sample whose
    # staggered location lies inside a conductor is eliminated (Dirichlet 0) just like
    # a wall node -- the caller reduces the operator to ``P[active][:, active]``.
    if pec_eu is None:
        eu_active = np.ones((n_eu,), dtype=bool)
    else:
        eu_mask = np.asarray(pec_eu, dtype=bool)
        if tuple(eu_mask.shape) != tuple(meta["shape_eu"]):
            raise ValueError(
                f"Yee transverse pec_eu mask shape {tuple(eu_mask.shape)} does not match the "
                f"Eu grid {tuple(meta['shape_eu'])}."
            )
        eu_active = ~eu_mask.reshape(-1)
    if pec_ev is None:
        ev_active = np.ones((n_ev,), dtype=bool)
    else:
        ev_mask = np.asarray(pec_ev, dtype=bool)
        if tuple(ev_mask.shape) != tuple(meta["shape_ev"]):
            raise ValueError(
                f"Yee transverse pec_ev mask shape {tuple(ev_mask.shape)} does not match the "
                f"Ev grid {tuple(meta['shape_ev'])}."
            )
        ev_active = ~ev_mask.reshape(-1)
    meta["pec_active"] = np.concatenate([eu_active, ev_active])
    meta["pec_ww_mask"] = pec_ww_mask
    return operator, meta


def _split_yee_transverse_eigenvector(eigenvector, meta: dict):
    """Reshape a stacked ``(Eu, Ev)`` eigenvector into its two component profiles."""
    vector = np.asarray(eigenvector).reshape(-1)
    n_eu = int(meta["n_eu"])
    eu = vector[:n_eu].reshape(meta["shape_eu"])
    ev = vector[n_eu:].reshape(meta["shape_ev"])
    return eu, ev


def _yee_stagger_eps_from_nodes(eps_node_planes, *, nu_cells: int, nv_cells: int):
    """Sample the diagonal node permittivity on the three staggered Yee grids.

    ``eps_node_planes`` is the ``(eps_uu, eps_vv, eps_ww)`` node-grid triple of the
    aperture (shape ``(nu_cells+1, nv_cells+1)``, walls included). The staggered
    Yee transverse operator wants ``eps_uu`` at the ``Eu`` grid ``(u half, v interior
    node)``, ``eps_vv`` at the ``Ev`` grid ``(u interior node, v half)`` and
    ``eps_ww`` at the interior nodes. Node -> half is the arithmetic average of the
    two bracketing nodes (the Yee dielectric average), interior-node selection drops
    the two wall rows/columns. Returns arrays whose shapes match the builder.
    """
    eps_uu_node, eps_vv_node, eps_ww_node = (
        np.asarray(component, dtype=np.float64) for component in eps_node_planes
    )
    # eps_uu on (u half, v interior node)
    eps_uu = 0.5 * (eps_uu_node[:-1, 1:-1] + eps_uu_node[1:, 1:-1])
    # eps_vv on (u interior node, v half)
    eps_vv = 0.5 * (eps_vv_node[1:-1, :-1] + eps_vv_node[1:-1, 1:])
    # eps_ww on interior nodes
    eps_ww = eps_ww_node[1:-1, 1:-1]
    return eps_uu, eps_vv, eps_ww


def _yee_half_to_node_neumann(values: np.ndarray, axis: int) -> np.ndarray:
    """Interpolate a Yee half-grid axis (``cells`` samples) to nodes (``cells+1``).

    Interior nodes are the average of the two bracketing half samples; the two wall
    nodes take the nearest half sample (a zeroth-order Neumann extrapolation, correct
    for a field whose normal derivative vanishes at the metallic wall).
    """
    values = np.moveaxis(values, axis, 0)
    cells = values.shape[0]
    node = np.empty((cells + 1,) + values.shape[1:], dtype=values.dtype)
    node[0] = values[0]
    node[-1] = values[-1]
    node[1:-1] = 0.5 * (values[:-1] + values[1:])
    return np.moveaxis(node, 0, axis)


def _yee_interior_to_node_dirichlet(values: np.ndarray, axis: int, *, cells: int) -> np.ndarray:
    """Place an interior-node axis (``cells-1`` samples) on nodes (``cells+1``).

    The two wall nodes are zero (Dirichlet) -- the component is tangential to that
    wall -- and the interior nodes carry the operator's samples verbatim.
    """
    values = np.moveaxis(values, axis, 0)
    node = np.zeros((cells + 1,) + values.shape[1:], dtype=values.dtype)
    node[1:-1] = values
    return np.moveaxis(node, 0, axis)


def _yee_reconstruct_node_profiles(eu_stag, ev_stag, *, beta, meta, eps_stag, field_names, pec_ww_mask=None):
    """Rebuild the transverse mode fields on the aperture node grid.

    From the transverse electric eigenvector ``(Eu, Ev)`` (each on its own Yee
    half/interior-node grid) reconstruct the longitudinal ``Ew`` (Gauss law) and the
    transverse magnetic field ``(Hu, Hv)`` from the discrete curl-E relations, then
    interpolate every component onto the common aperture node grid so the downstream
    power integrator and injection consume one collocated plane. All fields are real:
    the eigenvector is real, ``Ew`` is 90 deg out of phase (absorbed into a real
    amplitude), and the two ``i`` factors of the H-curls cancel it back to real.
    """
    nu_cells = int(meta["nu_cells"])
    nv_cells = int(meta["nv_cells"])
    du = float(meta["du"])
    dv = float(meta["dv"])
    k0 = float(meta["k0"])
    eps_uu, eps_vv, eps_ww = eps_stag
    eu = np.asarray(eu_stag).reshape(-1)
    ev = np.asarray(ev_stag).reshape(-1)

    # Longitudinal E at interior nodes from Gauss' law:
    # i beta eps_w Ew = -(d-_u(eps_u Eu) + d-_v(eps_v Ev)); factoring out i gives
    # a real interior-node amplitude ew (Ew = i * ew).
    eps_u_flat = np.asarray(eps_uu, dtype=np.float64).reshape(-1)
    eps_v_flat = np.asarray(eps_vv, dtype=np.float64).reshape(-1)
    eps_w_flat = np.asarray(eps_ww, dtype=np.float64).reshape(-1)
    beta_val = complex(beta)
    beta_mag = abs(beta_val)
    if beta_mag <= 0.0:
        ew_node = np.zeros_like(eps_w_flat)
    else:
        divergence = meta["gu_big"] @ (eps_u_flat * eu) + meta["gv_big"] @ (eps_v_flat * ev)
        ew_node = -np.real(divergence) / (np.real(beta_val) if abs(np.real(beta_val)) > 0 else beta_mag) / eps_w_flat
    if pec_ww_mask is not None:
        # A longitudinal node inside a conductor carries no Ew (its eps_ww coupling
        # was dropped from the operator); keep the reconstruction consistent.
        ew_node = np.where(np.asarray(pec_ww_mask, dtype=bool).reshape(-1), 0.0, ew_node)

    eta_scale = k0 * ETA_0  # omega * mu0 = k0 * eta0
    # Hu shares the Ev grid, Hv shares the Eu grid (standard Yee co-location).
    hu = -(np.real(beta_val) * ev + meta["sv_big"] @ ew_node) / eta_scale
    hv = (np.real(beta_val) * eu + meta["su_big"] @ ew_node) / eta_scale

    eu_grid = np.asarray(eu_stag).reshape(meta["shape_eu"])
    ev_grid = np.asarray(ev_stag).reshape(meta["shape_ev"])
    hu_grid = hu.reshape(meta["shape_ev"])
    hv_grid = hv.reshape(meta["shape_eu"])

    def _eu_grid_to_node(values):
        # (u half, v interior node) -> node grid: u half->node (Neumann), v Dirichlet.
        stage = _yee_half_to_node_neumann(values, axis=0)
        return _yee_interior_to_node_dirichlet(stage, axis=1, cells=nv_cells)

    def _ev_grid_to_node(values):
        # (u interior node, v half) -> node grid: u Dirichlet, v half->node (Neumann).
        stage = _yee_interior_to_node_dirichlet(values, axis=0, cells=nu_cells)
        return _yee_half_to_node_neumann(stage, axis=1)

    eu_node = _eu_grid_to_node(eu_grid)
    ev_node = _ev_grid_to_node(ev_grid)
    hu_node = _ev_grid_to_node(hu_grid)
    hv_node = _eu_grid_to_node(hv_grid)

    return {
        field_names[0]: eu_node,
        field_names[1]: ev_node,
        field_names[2]: hu_node,
        field_names[3]: hv_node,
    }


def _select_yee_transverse_mode_numpy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    meta: dict,
    eps_stag,
    mode_index: int,
    field_names,
    preferred_field_name: str,
    reject_spurious: bool,
    wave_family: str | None,
    pec_ww_mask=None,
):
    """Select one forward guided mode of the Yee-staggered transverse operator.

    ``eigenvalues`` are ``beta**2``; ``eigenvectors`` columns hold the stacked
    ``(Eu, Ev)`` transverse electric field. Candidates are ordered by descending
    ``beta**2`` (the fundamental has the largest, closest to ``k0**2``), degenerate
    subspaces are rotated to the requested polarization, and the same hardened
    filters as the legacy selector apply -- the ``beta -> k0`` transverse null-space
    branch is rejected for guided requests, and on a structure-enforcing (graded)
    aperture the checkerboard/duplicate gates apply. Returns ``(beta, node_profiles,
    diagnostics)`` with ``node_profiles`` collocated on the aperture node grid.
    """
    k0 = float(meta["k0"])
    reject_near_k0 = (
        not reject_spurious
        and wave_family is not None
        and str(wave_family).lower() != "tem"
        and k0 > 0.0
    )

    order = np.lexsort((np.abs(np.imag(eigenvalues)), -np.real(eigenvalues)))
    candidate_window = (
        max(2 * (int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING), 4)
        if reject_spurious
        else len(order)
    )
    raw_candidates = []
    for index in order:
        value = eigenvalues[index]
        if not np.isfinite(value) or np.real(value) <= 0.0:
            continue
        if abs(np.imag(value)) > 1.0e-6 * max(abs(np.real(value)), 1.0):
            # A materially complex beta**2 is not a lossless propagating mode.
            continue
        beta2 = float(np.real(value))
        beta = math.sqrt(beta2)
        vector = eigenvectors[:, index]
        raw_candidates.append((int(index), beta2, beta, vector))
        if len(raw_candidates) >= candidate_window:
            break

    # Group numerically-degenerate beta**2 values, orthogonalize each subspace with
    # an SVD (removes ARPACK duplicate/conjugate copies), and rotate the residual
    # subspace to diagonalize the requested-polarization energy so the launched mode
    # is a deterministic function of frequency/grid.
    independent = []
    cursor = 0
    while cursor < len(raw_candidates):
        reference = raw_candidates[cursor][1]
        stop = cursor + 1
        tolerance = _VECTOR_DEGENERATE_RTOL * max(abs(reference), 1.0)
        while stop < len(raw_candidates) and abs(raw_candidates[stop][1] - reference) <= tolerance:
            stop += 1
        group = raw_candidates[cursor:stop]
        vectors = [entry[3] for entry in group]
        if len(group) > 1:
            stacked = np.stack(vectors, axis=1)
            stacked = np.concatenate([np.real(stacked), np.imag(stacked)], axis=1)
            basis, singular_values, _ = np.linalg.svd(stacked, full_matrices=False)
            rank = int(np.sum(singular_values > 1.0e-6 * max(singular_values[0], 1e-30)))
            vectors = [basis[:, column] for column in range(max(rank, 1))]

        rotated_vectors = vectors
        if len(vectors) > 1:
            profiles_list = [
                _yee_reconstruct_node_profiles(
                    *_split_yee_transverse_eigenvector(vector, meta),
                    beta=raw_candidates[cursor][2],
                    meta=meta,
                    eps_stag=eps_stag,
                    field_names=field_names,
                    pec_ww_mask=pec_ww_mask,
                )
                for vector in vectors
            ]
            count = len(vectors)
            preferred_gram = np.empty((count, count), dtype=np.float64)
            electric_gram = np.empty((count, count), dtype=np.float64)
            for row in range(count):
                for col in range(count):
                    preferred_gram[row, col] = np.real(
                        np.vdot(profiles_list[row][preferred_field_name], profiles_list[col][preferred_field_name])
                    )
                    electric_gram[row, col] = np.real(
                        sum(
                            np.vdot(profiles_list[row][name], profiles_list[col][name])
                            for name in field_names[:2]
                        )
                    )
            electric_scale = max(float(np.max(np.abs(np.diag(electric_gram)))), 1.0)
            electric_gram += np.eye(count) * (np.finfo(np.float64).eps * electric_scale)
            fractions, rotations = scipy_linalg.eigh(preferred_gram, electric_gram)
            rotated_vectors = [
                sum(rotations[j, col] * vectors[j] for j in range(count))
                for col in np.argsort(fractions)[::-1]
            ]

        for vector in rotated_vectors:
            eu, ev = _split_yee_transverse_eigenvector(vector, meta)
            profiles = _yee_reconstruct_node_profiles(
                eu, ev, beta=raw_candidates[cursor][2], meta=meta, eps_stag=eps_stag,
                field_names=field_names, pec_ww_mask=pec_ww_mask,
            )
            profiles = _normalize_vector_mode_profiles_numpy(profiles, preferred_field_name=preferred_field_name)
            independent.append({"beta": raw_candidates[cursor][2], "vector": vector, "profiles": profiles})
        cursor = stop

    candidate_count = len(independent)
    power_gram = np.zeros((candidate_count, candidate_count), dtype=np.complex128)
    for row, left in enumerate(independent):
        for col, right in enumerate(independent):
            power_gram[row, col] = _vector_mode_power_inner_product_numpy(left["profiles"], right["profiles"])
    power_norm = np.sqrt(np.maximum(np.abs(np.real(np.diag(power_gram))), np.finfo(np.float64).eps))
    overlap_matrix = np.abs(power_gram / power_norm[:, None] / power_norm[None, :])

    retained_indices = []
    family_indices = []
    diagnostics_list = []
    for candidate_index, candidate in enumerate(independent):
        beta = candidate["beta"]
        profiles = candidate["profiles"]
        power = _vector_mode_power_sign_numpy(profiles)
        polarization_fraction = _vector_mode_polarization_fraction_numpy(
            profiles, preferred_field_name=preferred_field_name
        )
        checkerboard_fraction = _vector_mode_checkerboard_fraction_numpy(profiles)
        transverse_uniformity = _vector_mode_transverse_uniformity_numpy(
            profiles, preferred_field_name=preferred_field_name
        )
        near_k0 = (
            reject_near_k0
            and beta >= _SPURIOUS_NEAR_K0_BETA_LIMIT * k0
        )
        planewave_like = (
            transverse_uniformity is not None and transverse_uniformity > _SPURIOUS_UNIFORMITY_LIMIT
        )
        prior_overlaps = [
            float(overlap_matrix[candidate_index, prior])
            for prior in retained_indices
            if abs(beta - independent[prior]["beta"]) <= _VECTOR_DUPLICATE_BETA_RTOL * max(abs(beta), 1.0)
        ]
        max_near_duplicate_overlap = max(prior_overlaps, default=0.0)

        family_index = None
        if power <= 0.0:
            status = "backward_power"
        elif near_k0 and planewave_like:
            status = "spurious_near_k0"
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

        diagnostics_list.append(
            {
                "candidate_index": candidate_index,
                "beta_real": float(beta),
                "beta_imag": 0.0,
                "effective_index_real": float(beta / max(k0, 1e-30)),
                "effective_index_imag": 0.0,
                "propagating": True,
                "poynting_power": power,
                "polarization_fraction": polarization_fraction,
                "max_weighted_overlap": max_near_duplicate_overlap,
                "checkerboard_fraction": checkerboard_fraction,
                "transverse_uniformity": transverse_uniformity,
                "family_index": family_index,
                "status": status,
                "selected": False,
            }
        )

    if len(family_indices) <= int(mode_index):
        reject_reasons = {"checkerboard", "duplicate", "spurious_near_k0"}
        rejected = sum(entry["status"] in reject_reasons for entry in diagnostics_list)
        breakdown = ", ".join(
            f"{status}={sum(1 for e in diagnostics_list if e['status'] == status)}"
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
    diagnostics_list[selected_candidate_index]["selected"] = True
    selected = independent[selected_candidate_index]
    diagnostics = {
        "raw_candidate_count": len(raw_candidates),
        "independent_candidate_count": candidate_count,
        "selected_candidate_index": selected_candidate_index,
        "candidates": tuple(diagnostics_list),
        "overlap_matrix": overlap_matrix,
    }
    return selected["beta"], selected["profiles"], diagnostics


def _solve_yee_transverse_vector_mode(
    eps_node_planes,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
    wave_family: str | None,
    uniform: bool,
    use_dense: bool,
):
    """Solve one transverse full-vector guided mode on the Yee-staggered operator.

    ``eps_node_planes`` is the ``(eps_uu, eps_vv, eps_ww)`` node-grid permittivity of
    the aperture (walls included). Returns ``(beta, component_arrays, diagnostics)``
    with ``component_arrays`` the four tangential fields on the aperture node grid.
    """
    eps_uu_node, eps_vv_node, eps_ww_node = eps_node_planes
    nu_nodes = int(np.asarray(eps_uu_node).shape[0])
    nv_nodes = int(np.asarray(eps_uu_node).shape[1])
    nu_cells = nu_nodes - 1
    nv_cells = nv_nodes - 1
    if nu_cells <= 1 or nv_cells <= 1:
        raise ValueError("Transverse Yee mode aperture needs at least two cells per axis.")

    eps_uu, eps_vv, eps_ww = _yee_stagger_eps_from_nodes(
        eps_node_planes, nu_cells=nu_cells, nv_cells=nv_cells
    )
    # Always assemble the operator with the aperture's actual per-component eps.
    # ``uniform`` selects the symmetric eigensolve path (eigsh) and disables the
    # structure-enforcing spurious/checkerboard filters; it does NOT mean vacuum.
    # For a homogeneous (uniform) non-magnetic cross-section the operator equals
    # the vacuum operator plus a scalar ``(eps - 1) * k0**2`` identity shift, so it
    # stays exactly symmetric while returning the correct filled-guide beta
    # (``beta**2 = eps * k0**2 - kc**2``). Passing the real eps here is what keeps a
    # uniformly dielectric-filled aperture from collapsing onto the vacuum
    # propagation constant.
    operator_eps = (eps_uu, eps_vv, eps_ww)
    operator, meta = _build_yee_transverse_operator_sparse(
        nu_cells=nu_cells,
        nv_cells=nv_cells,
        du=du,
        dv=dv,
        k0=k0,
        eps_uu=operator_eps[0],
        eps_vv=operator_eps[1],
        eps_ww=operator_eps[2],
    )
    eps_stag = (eps_uu, eps_vv, eps_ww)
    matrix_size = int(operator.shape[0])

    if use_dense:
        dense = operator.toarray()
        if uniform:
            eigenvalues, eigenvectors = scipy_linalg.eigh(dense)
        else:
            eigenvalues, eigenvectors = scipy_linalg.eig(dense)
    else:
        requested = min(max(2 * (int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING), 4), matrix_size - 2)
        initial_vector = np.random.default_rng(0).standard_normal(matrix_size)
        try:
            if uniform:
                eigenvalues, eigenvectors = scipy_sparse_linalg.eigsh(
                    operator, k=requested, which="LA", tol=_VECTOR_EIGS_TOL,
                    maxiter=_VECTOR_EIGS_MAX_ITER, v0=initial_vector,
                )
            else:
                eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
                    operator, k=requested, which="LR", tol=_VECTOR_EIGS_TOL,
                    maxiter=_VECTOR_EIGS_MAX_ITER, v0=initial_vector,
                )
        except scipy_sparse_linalg.ArpackNoConvergence:
            dense = operator.toarray()
            if uniform:
                eigenvalues, eigenvectors = scipy_linalg.eigh(dense)
            else:
                eigenvalues, eigenvectors = scipy_linalg.eig(dense)

    beta, component_arrays, diagnostics = _select_yee_transverse_mode_numpy(
        eigenvalues,
        eigenvectors,
        meta=meta,
        eps_stag=eps_stag,
        mode_index=int(mode_index),
        field_names=field_names,
        preferred_field_name=preferred_field_name,
        reject_spurious=not uniform,
        wave_family=wave_family,
    )
    return beta, component_arrays, diagnostics


def _yee_stagger_pec_from_nodes(pec_node_occupancy, *, nu_cells: int, nv_cells: int, threshold: float):
    """Rasterize node PEC occupancy onto the three staggered Yee component grids.

    ``pec_node_occupancy`` is the aperture node-grid conductor occupancy (shape
    ``(nu_cells + 1, nv_cells + 1)``, walls included) in ``[0, 1]``. A staggered
    component is declared inside the conductor when its rasterized occupancy reaches
    ``threshold`` -- the SAME placement the eps staggering uses
    (``_yee_stagger_eps_from_nodes``): node -> half is the arithmetic mean of the two
    bracketing nodes, interior-node selection drops the two wall rows/columns. Returns
    boolean masks ``(pec_eu, pec_ev, pec_ww)`` matching the ``Eu`` grid ``(nu_cells,
    nv_cells - 1)``, the ``Ev`` grid ``(nu_cells - 1, nv_cells)`` and the interior-node
    grid ``(nu_cells - 1, nv_cells - 1)``.
    """
    occupancy = np.asarray(pec_node_occupancy, dtype=np.float64)
    expected = (nu_cells + 1, nv_cells + 1)
    if tuple(occupancy.shape) != expected:
        raise ValueError(
            f"PEC node occupancy shape {tuple(occupancy.shape)} does not match the aperture "
            f"node grid {expected}."
        )
    eu_occ = 0.5 * (occupancy[:-1, 1:-1] + occupancy[1:, 1:-1])   # (u half, v interior node)
    ev_occ = 0.5 * (occupancy[1:-1, :-1] + occupancy[1:-1, 1:])   # (u interior node, v half)
    ww_occ = occupancy[1:-1, 1:-1]                                # interior nodes
    return eu_occ >= threshold, ev_occ >= threshold, ww_occ >= threshold


def _yee_pec_connectivity_check(pec_ww_mask: np.ndarray) -> dict:
    """Fail-closed connectivity check over the interior-node conductor-free region.

    ``pec_ww_mask`` is the interior-node conductor mask (True = inside a conductor).
    The transverse mode is supported on the conductor-free (dielectric) node region; a
    dielectric node that is fully surrounded by conductor -- a numerically degenerate
    pinch point -- carries a spurious isolated null and is rejected. Returns a
    diagnostics dict with the count of connected conductor-free regions and the count of
    distinct interior conductors so the caller can record how many independent line
    modes the geometry supports (``conductors - 1`` for a doubly/multiply-connected
    line).
    """
    pec_ww_mask = np.asarray(pec_ww_mask, dtype=bool)
    dielectric = ~pec_ww_mask
    active_count = int(np.count_nonzero(dielectric))
    if active_count < 2:
        raise ValueError(
            "Interior-PEC mode plane leaves fewer than two conductor-free interior nodes; "
            "refine the aperture grid or check the conductor geometry."
        )
    dielectric_labels, dielectric_regions = _label_connected_components(dielectric)
    # A dielectric node with no conductor-free 4-neighbour is a degenerate pinch.
    isolated = np.zeros_like(dielectric)
    for shift_axis in (0, 1):
        for shift in (1, -1):
            neighbour = np.roll(dielectric, shift=shift, axis=shift_axis)
            # np.roll wraps; clear the wrapped edge so borders are not treated as neighbours.
            if shift_axis == 0:
                if shift == 1:
                    neighbour[0, :] = False
                else:
                    neighbour[-1, :] = False
            else:
                if shift == 1:
                    neighbour[:, 0] = False
                else:
                    neighbour[:, -1] = False
            isolated |= dielectric & ~neighbour
    fully_isolated = dielectric.copy()
    for shift_axis in (0, 1):
        for shift in (1, -1):
            neighbour = np.roll(dielectric, shift=shift, axis=shift_axis)
            if shift_axis == 0:
                if shift == 1:
                    neighbour[0, :] = False
                else:
                    neighbour[-1, :] = False
            else:
                if shift == 1:
                    neighbour[:, 0] = False
                else:
                    neighbour[:, -1] = False
            fully_isolated &= ~neighbour
    if bool(np.any(fully_isolated)):
        raise ValueError(
            "Interior-PEC mode plane has a conductor-free node fully surrounded by conductor "
            "(degenerate pinch point); refine the aperture grid across the conductor gap."
        )
    _, conductor_regions = _label_connected_components(pec_ww_mask)
    return {
        "dielectric_regions": int(dielectric_regions),
        "conductor_regions": int(conductor_regions),
        "active_interior_nodes": active_count,
    }


def _solve_yee_transverse_pec_mode(
    eps_node_planes,
    pec_node_occupancy,
    *,
    k0: float,
    du: float,
    dv: float,
    mode_index: int,
    field_names,
    preferred_field_name: str,
    wave_family: str | None,
    uniform: bool,
    use_dense: bool,
    threshold: float = _PEC_OCCUPANCY_THRESHOLD,
):
    """Solve one transverse mode on the interior-PEC-masked Yee-staggered operator.

    ``eps_node_planes`` is the ``(eps_uu, eps_vv, eps_ww)`` node-grid permittivity of
    the aperture (walls included); ``pec_node_occupancy`` is the matching node-grid
    conductor occupancy. Conductor-interior transverse-component unknowns are eliminated
    (Dirichlet 0) with the same symmetric row/column removal as the outer walls, and the
    conductor-interior longitudinal nodes drop from the eps_ww divergence coupling. For a
    doubly/multiply-connected line (an isolated interior conductor) the transverse
    null-space branch at ``beta**2 = eps * k0**2`` is the physical TEM answer and is
    kept -- pass ``wave_family = "tem"``. Returns ``(beta, component_arrays,
    diagnostics)`` with ``component_arrays`` the four tangential fields on the aperture
    node grid; ``diagnostics`` carries the connectivity report under ``connectivity``.
    """
    eps_uu_node, eps_vv_node, eps_ww_node = eps_node_planes
    nu_nodes = int(np.asarray(eps_uu_node).shape[0])
    nv_nodes = int(np.asarray(eps_uu_node).shape[1])
    nu_cells = nu_nodes - 1
    nv_cells = nv_nodes - 1
    if nu_cells <= 1 or nv_cells <= 1:
        raise ValueError("Transverse Yee mode aperture needs at least two cells per axis.")

    eps_uu, eps_vv, eps_ww = _yee_stagger_eps_from_nodes(
        eps_node_planes, nu_cells=nu_cells, nv_cells=nv_cells
    )
    pec_eu, pec_ev, pec_ww = _yee_stagger_pec_from_nodes(
        pec_node_occupancy, nu_cells=nu_cells, nv_cells=nv_cells, threshold=threshold
    )
    if not bool(np.any(pec_eu) or np.any(pec_ev) or np.any(pec_ww)):
        raise ValueError(
            "Interior-PEC mode solve was requested but no staggered component falls inside a "
            "conductor: the PEC structure is under-resolved on the mode plane (raise the "
            "aperture resolution) or absent."
        )
    if wave_family is not None and str(wave_family).lower() == "tem":
        # The curl-curl beta**2 operator DOES carry the gradient TEM branch in its spectrum:
        # for a curl-free field et = -grad(phi) with div(eps grad phi) = 0 both the curl-curl
        # and the divergence-coupling terms vanish identically, leaving P et = eps*k0**2 * et.
        # The TEM branch is removed from THIS masked reduced operator by the shipped occupancy
        # rasterization (_PEC_OCCUPANCY_THRESHOLD = 0.5 with '>='), which eliminates the
        # conductor-surface straddling normal-E samples (occupancy exactly 0.5) where the TEM
        # surface charge and field energy concentrate -- a masking-rasterization choice, not a
        # structural property of the operator. A strictly-interior (keep-straddle, threshold
        # > 0.5) elimination recovers the exact TEM eigenvalue eps*k0**2 (see
        # test_masked_operator_tem_branch_is_a_masking_artifact), but would leave a one-cell-
        # thick conductor sheet entirely unmasked. Rather than trade that off on this path,
        # TEM/quasi-TEM interior-PEC lines are solved on the quasi-static electrostatic engine
        # (_solve_quasistatic_line_modes); this masked operator serves the guided (non-TEM,
        # hybrid) interior-PEC modes.
        raise ValueError(
            "The interior-PEC-masked staggered mode operator (as rasterized here, occupancy "
            "threshold 0.5) does not expose the TEM/quasi-TEM gradient branch: the surface-"
            "straddling normal-E samples that carry the TEM energy are eliminated by the "
            "masking. Route TEM lines through the quasi-static electrostatic engine "
            "(_solve_quasistatic_line_modes) instead."
        )
    connectivity = _yee_pec_connectivity_check(pec_ww)

    operator, meta = _build_yee_transverse_operator_sparse(
        nu_cells=nu_cells,
        nv_cells=nv_cells,
        du=du,
        dv=dv,
        k0=k0,
        eps_uu=eps_uu,
        eps_vv=eps_vv,
        eps_ww=eps_ww,
        pec_eu=pec_eu,
        pec_ev=pec_ev,
        pec_ww=pec_ww,
    )
    eps_stag = (eps_uu, eps_vv, eps_ww)
    active = np.asarray(meta["pec_active"], dtype=bool)
    full_size = int(active.size)
    reduced = operator[active][:, active].tocsr()
    reduced_size = int(reduced.shape[0])
    if reduced_size < 2:
        raise ValueError("Interior-PEC masking leaves fewer than two active transverse unknowns.")

    dense_cutoff = min(_FULL_VECTOR_DENSE_LIMIT * 4, _PEC_VECTOR_MATRIX_LIMIT)
    solve_dense = use_dense or reduced_size <= dense_cutoff
    if solve_dense:
        dense = reduced.toarray()
        if uniform:
            reduced_values, reduced_vectors = scipy_linalg.eigh(dense)
        else:
            reduced_values, reduced_vectors = scipy_linalg.eig(dense)
    else:
        requested = min(max(2 * (int(mode_index) + _VECTOR_EIGEN_REQUEST_PADDING), 4), reduced_size - 2)
        initial_vector = np.random.default_rng(0).standard_normal(reduced_size)
        try:
            if uniform:
                reduced_values, reduced_vectors = scipy_sparse_linalg.eigsh(
                    reduced, k=requested, which="LA", tol=_VECTOR_EIGS_TOL,
                    maxiter=_VECTOR_EIGS_MAX_ITER, v0=initial_vector,
                )
            else:
                reduced_values, reduced_vectors = scipy_sparse_linalg.eigs(
                    reduced, k=requested, which="LR", tol=_VECTOR_EIGS_TOL,
                    maxiter=_VECTOR_EIGS_MAX_ITER, v0=initial_vector,
                )
        except scipy_sparse_linalg.ArpackNoConvergence:
            dense = reduced.toarray()
            if uniform:
                reduced_values, reduced_vectors = scipy_linalg.eigh(dense)
            else:
                reduced_values, reduced_vectors = scipy_linalg.eig(dense)

    # Scatter each reduced eigenvector back onto the full (Eu, Ev) stacked grid; the
    # eliminated conductor unknowns stay zero (Dirichlet), exactly as the wall nodes.
    eigenvectors = np.zeros((full_size, reduced_vectors.shape[1]), dtype=reduced_vectors.dtype)
    eigenvectors[active, :] = reduced_vectors

    beta, component_arrays, diagnostics = _select_yee_transverse_mode_numpy(
        reduced_values,
        eigenvectors,
        meta=meta,
        eps_stag=eps_stag,
        mode_index=int(mode_index),
        field_names=field_names,
        preferred_field_name=preferred_field_name,
        reject_spurious=not uniform,
        wave_family=wave_family,
        pec_ww_mask=meta["pec_ww_mask"],
    )
    diagnostics["connectivity"] = connectivity
    diagnostics["pec_active_count"] = int(np.count_nonzero(~active))
    return beta, component_arrays, diagnostics
