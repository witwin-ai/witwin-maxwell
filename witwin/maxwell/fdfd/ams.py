"""Auxiliary-space Maxwell (Hiptmair-Xu) preconditioner for the FDFD system.

Structure (all GPU, CuPy sparse):

- Gradient auxiliary space: the PML-stretched discrete gradient
  ``P_g = D S_e^-1 G`` maps nodal potentials onto the null space of the
  scaled curl-curl part — exact to round-off in the interior; the outermost
  cell layer carries an inherent truncation defect. The Galerkin operator
  ``A_g = P_g^T A P_g ~= k0^2 P_g^T E P_g`` is a definite nodal Poisson-type
  operator.
- Vector auxiliary space: component-wise nodal-to-edge averaging
  ``P_c = D S_e^-1 Pi_c``; the Galerkin blocks are built from the
  complex-shifted operator ``A + i beta k0^2 E`` (shifted-Laplacian damping
  of the indefinite wave part) so multigrid smoothing is stable.
- Auxiliary solves: geometric multigrid V(1,1) cycles on the structured nodal
  grid (Galerkin RAP coarsening, damped-Jacobi smoothing, dense solve on the
  coarsest level).
- Composition: symmetrized multiplicative Schwarz
  ``smoother -> gradient -> vector -> gradient -> smoother``. Every stage is
  complex-symmetric, so the preconditioner is valid for SQMR.

This is the in-repo answer to the plan's 10.4 AMS item: hypre-AMS is
unavailable on native Windows, and the structured Yee grid lets geometric
multigrid replace AMG for the auxiliary problems.

STATUS: EXPERIMENTAL — measured outcome on the indefinite time-harmonic
operator (see plan 10.8): the gradient-space construction is exact (the
null-space identity holds to round-off) and its multigrid contracts, but
the overall Hiptmair-Xu composition is non-contractive at these
wavenumbers even with EXACT auxiliary solves — oblique subspace
corrections amplify each other on the non-Hermitian indefinite operator.
This matches hypre-AMS's own scope (definite curl-curl / eddy-current
regimes). For the indefinite regime use sqmr+ssor+precision='double'
(converges through 64^3) or the direct backend; large-scale indefinite
solves need sweeping/DDM-class methods, out of scope here.
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cupy_linalg

# Complex shift applied to the vector auxiliary operator (shifted-Laplacian
# damping); the gradient space is definite and needs no shift.
AMS_SHIFT_BETA = 0.5
# Damped-Jacobi weight for the fine smoother and the multigrid smoother.
AMS_JACOBI_OMEGA = 0.8
# Stop coarsening once every axis is at or below this many nodes.
AMS_COARSEST_NODES = 8


def _prolongation_1d(n, dtype):
    """Vertex-centered 1D linear prolongation (fine n -> coarse (n+1)//2)."""
    nc = (n + 1) // 2
    rows, cols, vals = [], [], []
    for i in range(n):
        if i % 2 == 0:
            rows.append(i); cols.append(i // 2); vals.append(1.0)
        else:
            left = i // 2
            right = left + 1
            if right < nc:
                rows.extend((i, i)); cols.extend((left, right)); vals.extend((0.5, 0.5))
            else:
                rows.append(i); cols.append(left); vals.append(1.0)
    return cpsp.csr_matrix(
        (cp.asarray(vals, dtype=dtype), (cp.asarray(rows), cp.asarray(cols))),
        shape=(n, nc),
    )


def _prolongation_3d(shape, dtype):
    px = _prolongation_1d(shape[0], dtype)
    py = _prolongation_1d(shape[1], dtype)
    pz = _prolongation_1d(shape[2], dtype)
    return cpsp.kron(cpsp.kron(px, py), pz).tocsr()


class _NodalMultigrid:
    """Geometric V(1,1) multigrid on a structured nodal grid with Galerkin
    coarsening and damped-Jacobi smoothing. Complex-symmetric throughout
    (transposes, never conjugates).

    The operator is symmetrically diagonal-scaled first (S A S with
    S = diag(A)^-1/2), which absorbs the orders-of-magnitude PML weight
    range into the transfers; the coarsest level is solved densely with a
    small Tikhonov term (the gradient-space operator has an exact constant
    null space)."""

    def __init__(self, A, shape, omega=AMS_JACOBI_OMEGA):
        self.omega = omega
        A = A.tocsr()
        self.scale = cp.reciprocal(cp.sqrt(A.diagonal()))
        scale_matrix = cpsp.diags(self.scale).tocsr()
        current_A = (scale_matrix @ (A @ scale_matrix)).tocsr()
        current_shape = tuple(shape)
        self.levels = []
        while min(current_shape) > AMS_COARSEST_NODES:
            P = _prolongation_3d(current_shape, current_A.dtype)
            R = P.T.tocsr()
            inv_diag = cp.reciprocal(current_A.diagonal())
            self.levels.append((current_A, inv_diag, P, R))
            current_A = (R @ (current_A @ P)).tocsr()
            current_shape = tuple((s + 1) // 2 for s in current_shape)
        coarse = current_A.toarray()
        regularization = 1e-10 * float(cp.abs(coarse.diagonal()).max())
        coarse = coarse + regularization * cp.eye(coarse.shape[0], dtype=coarse.dtype)
        self.coarse_dense = coarse

    def solve(self, r):
        return self.scale * self._cycle(0, self.scale * r)

    def _cycle(self, level, r):
        if level == len(self.levels):
            return cp.linalg.solve(self.coarse_dense, r)
        A, inv_diag, P, R = self.levels[level]
        z = self.omega * inv_diag * r
        z = z + P @ self._cycle(level + 1, R @ (r - A @ z))
        z = z + self.omega * inv_diag * (r - A @ z)
        return z


def _edge_shapes(scene):
    return (
        (scene.Nx_ex, scene.Ny_ex, scene.Nz_ex),
        (scene.Nx_ey, scene.Ny_ey, scene.Nz_ey),
        (scene.Nx_ez, scene.Ny_ez, scene.Nz_ez),
    )


def _stretch_half_factors(solver, dtype):
    s_x, s_y, s_z = solver._create_pml_3d()
    s_x = cp.asarray(s_x, dtype=dtype)
    s_y = cp.asarray(s_y, dtype=dtype)
    s_z = cp.asarray(s_z, dtype=dtype)
    return (
        (s_x[:-1] + s_x[1:]) / 2,
        (s_y[:-1] + s_y[1:]) / 2,
        (s_z[:-1] + s_z[1:]) / 2,
    )


def _edge_stretch_weights(solver, dtype):
    """1/s along each edge's own axis, per component block (flat)."""
    scene = solver.scene
    shapes = _edge_shapes(scene)
    sxh, syh, szh = _stretch_half_factors(solver, dtype)
    wx = cp.broadcast_to((1.0 / sxh)[:, None, None], shapes[0]).ravel()
    wy = cp.broadcast_to((1.0 / syh)[None, :, None], shapes[1]).ravel()
    wz = cp.broadcast_to((1.0 / szh)[None, None, :], shapes[2]).ravel()
    return wx, wy, wz


def _node_index(scene):
    return cp.arange(scene.Nx * scene.Ny * scene.Nz).reshape(scene.Nx, scene.Ny, scene.Nz)


def _difference_block(node, axis, dx, dtype):
    """Plain discrete gradient for one component: edges x nodes, +/- 1/dx."""
    if axis == 0:
        lo, hi = node[:-1], node[1:]
    elif axis == 1:
        lo, hi = node[:, :-1], node[:, 1:]
    else:
        lo, hi = node[:, :, :-1], node[:, :, 1:]
    n_edges = lo.size
    rows = cp.arange(n_edges)
    rows2 = cp.concatenate([rows, rows])
    cols = cp.concatenate([hi.ravel(), lo.ravel()])
    vals = cp.concatenate([
        cp.full(n_edges, 1.0 / dx, dtype=dtype),
        cp.full(n_edges, -1.0 / dx, dtype=dtype),
    ])
    return cpsp.csr_matrix((vals, (rows2, cols)), shape=(n_edges, node.size))


def _average_block(node, axis, dtype):
    """Nodal-to-edge tangential averaging for one component: weights 1/2."""
    if axis == 0:
        lo, hi = node[:-1], node[1:]
    elif axis == 1:
        lo, hi = node[:, :-1], node[:, 1:]
    else:
        lo, hi = node[:, :, :-1], node[:, :, 1:]
    n_edges = lo.size
    rows = cp.arange(n_edges)
    rows2 = cp.concatenate([rows, rows])
    cols = cp.concatenate([hi.ravel(), lo.ravel()])
    vals = cp.full(2 * n_edges, 0.5, dtype=dtype)
    return cpsp.csr_matrix((vals, (rows2, cols)), shape=(n_edges, node.size))


def _eps_face_vector(solver, dtype):
    ex = cp.asarray(solver.material_eps_components["x"], dtype=dtype)
    ey = cp.asarray(solver.material_eps_components["y"], dtype=dtype)
    ez = cp.asarray(solver.material_eps_components["z"], dtype=dtype)
    return cp.concatenate([
        ((ex[:-1] + ex[1:]) / 2).ravel(),
        ((ey[:, :-1] + ey[:, 1:]) / 2).ravel(),
        ((ez[:, :, :-1] + ez[:, :, 1:]) / 2).ravel(),
    ])


class AMSPreconditioner:
    def __init__(self, solver, shift_beta=AMS_SHIFT_BETA):
        A = solver._iteration_matrix()
        dtype = A.dtype
        scene = solver.scene
        dx = float(scene.dx)
        node = _node_index(scene)
        node_shape = (scene.Nx, scene.Ny, scene.Nz)
        n_nodes = node.size
        edge_counts = (scene.N_ex, scene.N_ey, scene.N_ez)

        scale = solver._ensure_symmetrization_scale().astype(dtype)
        stretch = _edge_stretch_weights(solver, dtype)
        # per-component edge weight: d * (1/s) restricted to that block
        offsets = (0, edge_counts[0], edge_counts[0] + edge_counts[1])
        block_weights = [
            scale[offsets[c]:offsets[c] + edge_counts[c]] * stretch[c]
            for c in range(3)
        ]

        # Shifted-Laplacian architecture: every internal stage (smoother,
        # residual refreshes, auxiliary Galerkin operators) works on the
        # DAMPED operator A_beta = A + i beta k0^2 E; the whole object then
        # approximates A_beta^-1, which is the classical robust
        # preconditioner for the indefinite wave operator A. A fine-level
        # smoother on the undamped A itself amplifies wave modes.
        eps_face = _eps_face_vector(solver, dtype)
        shift = cpsp.diags(1j * shift_beta * solver.k0**2 * eps_face).tocsr()
        self.A_shift = (A + shift).tocsr()
        self.inv_diag = cp.reciprocal(self.A_shift.diagonal())
        self.omega = AMS_JACOBI_OMEGA

        # --- gradient auxiliary space (exact null-space complement) ---
        grad_blocks = []
        for c in range(3):
            block = _difference_block(node, c, dx, dtype)
            grad_blocks.append(cpsp.diags(block_weights[c]).tocsr() @ block)
        self.P_grad = cpsp.vstack(grad_blocks).tocsr()
        A_grad = (self.P_grad.T.tocsr() @ (self.A_shift @ self.P_grad)).tocsr()
        self.mg_grad = _NodalMultigrid(A_grad, node_shape)

        # --- vector auxiliary spaces ---
        # The Galerkin block P_c^T A P_c only contains the TRANSVERSE
        # Laplacian (the Yee (c,c) curl-curl block has no d^2/dc^2 term),
        # which leaves whole line families unsmoothable. Following the
        # Hiptmair-Xu identity (curl curl = -Lap + grad div on the
        # divergence-free complement), use the FULL stretched nodal
        # Laplacian as the vector surrogate, with the same complex shift
        # on the mass term.
        laplacian = (self.P_grad.T.tocsr() @ self.P_grad).tocsr()  # Gs^T D^2 Gs
        eps_diag = cpsp.diags(eps_face).tocsr()
        k02 = dtype.type((1.0 + 1j * shift_beta)) * solver.k0**2
        self.P_vec = []
        self.mg_vec = []
        for c in range(3):
            averaging = _average_block(node, c, dtype)
            block = cpsp.diags(block_weights[c]).tocsr() @ averaging
            zero_above = cpsp.csr_matrix((offsets[c], n_nodes), dtype=dtype)
            below = A.shape[0] - offsets[c] - edge_counts[c]
            zero_below = cpsp.csr_matrix((below, n_nodes), dtype=dtype)
            P_c = cpsp.vstack([zero_above, block, zero_below]).tocsr()
            mass_c = (P_c.T.tocsr() @ (eps_diag @ P_c)).tocsr()
            A_c = (k02 * mass_c - laplacian).tocsr()
            self.P_vec.append(P_c)
            self.mg_vec.append(_NodalMultigrid(A_c, node_shape))

    def _smooth(self, z, r):
        return z + self.omega * self.inv_diag * (r - self.A_shift @ z)

    def _grad_correct(self, z, r):
        residual = r - self.A_shift @ z
        return z + self.P_grad @ self.mg_grad.solve(self.P_grad.T @ residual)

    def _vec_correct(self, z, r):
        residual = r - self.A_shift @ z
        for P_c, mg in zip(self.P_vec, self.mg_vec):
            z = z + P_c @ mg.solve(P_c.T @ residual)
        return z

    def apply(self, r):
        # Symmetrized multiplicative Schwarz: S, G, V, G, S
        z = self.omega * self.inv_diag * r
        z = self._grad_correct(z, r)
        z = self._vec_correct(z, r)
        z = self._grad_correct(z, r)
        z = self._smooth(z, r)
        return z

    def as_linear_operator(self):
        return cupy_linalg.LinearOperator(
            self.A_shift.shape, matvec=self.apply, dtype=self.A_shift.dtype
        )


def build_ams_preconditioner(solver):
    return AMSPreconditioner(solver).as_linear_operator()
