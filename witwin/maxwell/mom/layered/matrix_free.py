"""Matrix-free layered MPIE backend: sparse near-field LU right preconditioner and block GMRES.

The system is the dense backend's equilibrated loop/tree basis (lowfreq.py),

    A = D [S Q^T (-i mu0 V + R/w) Q S + i/(eps0 w0^2) C^T P C (tree block)] D,

with S = diag(1 loops, w/w0 trees), Q the sparse loop/tree transform and C the
tree face charge. Face pairs are split by the assembly's own near predicate
(pair_near for horizontal pairs, wall_near with walls), which partitions every
face-pair tile exactly:

* near: the near pair list is evaluated once per frequency with the identical
  pair kernels, gathered to the RWG pattern (V_near, with the local loss Gram
  folded in) and the face pattern (P_near), and transformed to the basis with
  deterministic expand/stable-sort/segment-sum products. The result M is both
  the near part of the action and the preconditioner (complex128 sparse LU,
  deterministic mode, right preconditioning).
* far: every action re-evaluates all far face-pair tiles source tile by source
  tile and contracts them with the face coefficients of all port columns at
  once. No N^2 matrix exists.

D is 1/sqrt|diag(M)|. Admittances Y = B^T X and currents are invariant under D,
so the near-field diagonal is as valid as the dense backend's full diagonal.

Stopping criterion. A is complex symmetric (canonical face-pair tiles, symmetric
Gram loss and congruence), so with X* = A^-1 B and R = B - A X,
dY = B^T (X* - X) = X*^T R, hence |dY|_F <= |X*|_F |R|_F. Block GMRES stops when
|X|_F |R|_F <= ADMITTANCE_TOLERANCE |B^T X|_F, first with the Arnoldi residual,
then confirmed with an explicit true residual (one extra action). The tolerance
1e-5 keeps a 10x margin under the 1e-4 network agreement target.
surface_admittance adds nonsymmetric per-body DSA blocks to the near part
(admittance_near.py); then dY = Xa^T R with Xa = A^-T B, and |X|_F stands for
|Xa|_F up to the relative asymmetry of A, which agreement with dense measures.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import time

import numpy as np
import torch

from ..operator_protocol import IterativeState, KrylovView
from .assembly import _adaptive_rule, _frequency_view, _outer_rule, assembly_operands
from .lowfreq import C0, _transpose_columns, port_drive


ADMITTANCE_TOLERANCE = 1e-5
RESTART = 48
MAX_BLOCK_STEPS = 960
NEAR_RATIO = 2.0
NEAR_CHUNK = 1 << 20
TILE_BYTES = 1 << 28
# Expansion entries (or merged entries) of one row chunk of the near basis products (~0.6 GB transient).
PRODUCT_CHUNK = 1 << 23
# The native kernels' constants (common.cuh): mu0 = 1/(eps0 c0^2), so the loss
# fold R/(-i mu0 w) and the magnetic factor -i mu0 cancel exactly.
EPS0 = 8.8541878128e-12
MU0 = 1 / (EPS0 * C0 * C0)


def _csr_offsets(rows, count):
    """CSR offsets of row-sorted indices (exact integer counts)."""
    return torch.cat((rows.new_zeros(1), torch.cumsum(torch.bincount(rows, minlength=count), 0)))


def _chunks(cumulative, limit):
    """Row ranges whose summed weight (host prefix sums [rows+1]) stays <= limit; a heavier row is its own range."""
    bounds, rows = [0], len(cumulative) - 1
    while bounds[-1] < rows:
        start = bounds[-1]
        stop = int(torch.searchsorted(cumulative, cumulative[start] + limit, right=True)) - 1
        bounds.append(min(rows, max(stop, start + 1)))
    return list(zip(bounds[:-1], bounds[1:]))


def sparse_product(x, y, rows, columns, limit=PRODUCT_CHUNK):
    """C = X Y of CSR X = (offsets, cols, values) [rows, K] and Y [K, columns], X row chunk by chunk.

    Products are expanded in X-entry then Y-row order, stably sorted by output
    key and summed per segment in that order: values are deterministic. Row
    chunks partition the sorted key order, so the segments and their summation
    order do not depend on ``limit`` (at most ``limit`` expansion entries live
    at a time). Without X values only the pattern is formed. Nothing is kept
    between calls: the symbolic product is replayed per frequency.
    Returns (offsets, cols, values or None, expansion count).
    """
    x_offsets, x_cols, x_values = x[0], x[1], x[2]
    y_offsets, y_cols, y_values = y[0], y[1], y[2]
    device = x_cols.device
    counts = y_offsets.diff()[x_cols]
    cumulative = torch.cat((counts.new_zeros(1), torch.cumsum(counts, 0)))
    host = x_offsets.cpu()
    out_rows, out_cols, out_values = [], [], []
    for r0, r1 in _chunks(cumulative[x_offsets].cpu(), limit):
        e0, e1 = int(host[r0]), int(host[r1])
        c = counts[e0:e1]
        x_rows = torch.repeat_interleave(torch.arange(r0, r1, device=device), x_offsets[r0:r1 + 1].diff())
        left = torch.repeat_interleave(torch.arange(e1 - e0, device=device), c)
        starts = torch.cumsum(c, 0) - c
        right = y_offsets[x_cols[e0:e1]][left] + torch.arange(len(left), device=device) - starts[left]
        keys = x_rows[left] * columns + y_cols[right]
        del x_rows, starts
        keys, order = torch.sort(keys, stable=True)
        unique, sizes = torch.unique_consecutive(keys, return_counts=True)
        del keys
        out_rows.append(unique // columns)
        out_cols.append(unique % columns)
        del unique
        if x_values is not None:
            segments = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)))
            out_values.append(torch.ops.witwin_maxwell_mom_layered.segment_sum(
                (x_values[left[order] + e0] * y_values[right[order]]).to(torch.complex128).contiguous(), segments))
        del left, right, order, sizes
    values = torch.cat(out_values) if x_values is not None else None
    return _csr_offsets(torch.cat(out_rows), rows), torch.cat(out_cols), values, int(cumulative[-1])


def merge_blocks(magnetic, charge, n, loops, limit=PRODUCT_CHUNK):
    """The basis pattern of the magnetic block [n, n] plus the tree charge block (rows/cols + loops).

    Entries of one key are summed magnetic first, then charge (a stable key sort
    of the magnetic entries followed by the charge entries), row chunk by row chunk.
    Returns (offsets, cols, values or None).
    """
    m_offsets, m_cols, m_values = magnetic
    c_offsets, c_cols, c_values = charge
    device = m_cols.device
    trees = len(c_offsets) - 1
    hm, hc = m_offsets.cpu(), c_offsets.cpu()
    weight = hm.clone()
    weight[loops:] += hc
    rows_out, cols_out, values_out = [], [], []
    for r0, r1 in _chunks(weight, limit):
        a0, a1 = int(hm[r0]), int(hm[r1])
        t0, t1 = min(max(r0 - loops, 0), trees), min(max(r1 - loops, 0), trees)
        b0, b1 = int(hc[t0]), int(hc[t1])
        m_rows = torch.repeat_interleave(torch.arange(r0, r1, device=device), m_offsets[r0:r1 + 1].diff())
        c_rows = torch.repeat_interleave(torch.arange(t0, t1, device=device), c_offsets[t0:t1 + 1].diff()) + loops
        keys = torch.cat((m_rows * n + m_cols[a0:a1], c_rows * n + c_cols[b0:b1] + loops))
        del m_rows, c_rows
        keys, order = torch.sort(keys, stable=True)
        unique, sizes = torch.unique_consecutive(keys, return_counts=True)
        del keys
        rows_out.append(unique // n)
        cols_out.append(unique % n)
        del unique
        if m_values is not None:
            segments = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)))
            values_out.append(torch.ops.witwin_maxwell_mom_layered.segment_sum(
                torch.cat((m_values[a0:a1], c_values[b0:b1]))[order].contiguous(), segments))
        del order, sizes
    values = torch.cat(values_out) if m_values is not None else None
    return _csr_offsets(torch.cat(rows_out), n), torch.cat(cols_out), values


@dataclass
class NearStructure:
    """Frequency-independent near-field pattern and solver plan of one layout.

    The basis products of the near values (V' Q, Q^T V' Q, C^T P, C^T P C and
    their merge) are not stored: near_values replays them per frequency in
    bounded row chunks (sparse_product), and only the merged pattern stays.
    """

    green: list
    geometry: list
    far_rule: torch.Tensor
    near_rule: torch.Tensor
    cache: list
    pairs: torch.Tensor
    pair_offsets: torch.Tensor
    edge_rows: torch.Tensor
    edge_offsets: torch.Tensor
    edge_cols: torch.Tensor
    offsets: torch.Tensor
    cols: torch.Tensor
    diagonal: torch.Tensor
    charge_rows: list
    source_tile: int
    stats: dict = field(default_factory=dict)
    # surface_admittance (admittance_near.py): per-body basis patterns and their H-matrix plans (body_compress.py).
    bodies: list = None
    body_plans: list = None
    # Position of entry (c, r) for every entry (r, c): the exact M^T of the (unfolded) action pattern.
    transpose: torch.Tensor = None


def _support_operands(layout):
    basis, mesh = layout.basis, layout.mesh
    from ..topology import INTERIOR
    supports = torch.stack((basis.plus_faces, basis.minus_faces), -1).contiguous()
    slots = (mesh.topology.incidence_slots[mesh.topology.kind == INTERIOR].contiguous()
             if layout.support_slots is None else layout.support_slots)
    areas2 = (mesh.areas * 2).contiguous()
    slopes = (basis.face_signs * basis.edge_lengths[basis.face_edges] / areas2[:, None]).contiguous()
    from .mesh import JunctionRWGBasis
    support_slopes = ((basis.edge_lengths[:, None] * basis.edge_lengths.new_tensor([1., -1.])
                       / areas2[supports]).contiguous() if isinstance(basis, JunctionRWGBasis)
                      else areas2.new_empty(0))
    return supports, slots.contiguous(), slopes, basis.face_edges.contiguous(), support_slopes, areas2


def build_near_structure(prepared, pairs=None, base=None):
    """Near pair list, RWG/face patterns and the merged basis pattern.

    ``pairs`` replaces the assembly's near list by an explicit row-sorted
    (t, s) list that contains it (a wider preconditioner pattern); ``base``
    (the action structure) then lends its Green operands, rules and assembly
    cache, which do not depend on the pair list.
    Surface-admittance body blocks never enter the pattern: the action applies
    them as compressed H-matrices (body_compress.py) and the preconditioner
    adds them on the pattern's own keys (body_block_lu.py).
    """
    op = torch.ops.witwin_maxwell_mom_layered
    layout, lt = prepared.layout, prepared.loop_tree
    device = layout.mesh.device
    table = _frequency_view(prepared.table, 0, 1)
    if base is None:
        green, geometry = assembly_operands(layout, table)
        far_rule, near_rule = _adaptive_rule(device, True), _outer_rule(device)
        cache = op.assembly_cache(green, geometry, far_rule, table.exact_mode, table.ground)
    else:
        green, geometry, far_rule, near_rule, cache = base.green, base.geometry, base.far_rule, base.near_rule, base.cache
    action = pairs is None
    if action:
        pairs = op.near_pairs(green, geometry, far_rule, NEAR_RATIO, table.exact_mode, table.ground)
    faces, edges = len(layout.mesh.faces), layout.basis.n_edges
    pair_offsets = _csr_offsets(pairs[:, 0], faces)
    # RWG pattern: every edge of a face times every edge of a near partner face.
    supports, slots, slopes, face_edges, support_slopes, _ = _support_operands(layout)
    valid = slopes != 0
    width = face_edges.shape[1]
    left = face_edges[pairs[:, 0]][:, :, None].expand(-1, width, width)
    right = face_edges[pairs[:, 1]][:, None, :].expand(-1, width, width)
    mask = valid[pairs[:, 0]][:, :, None] & valid[pairs[:, 1]][:, None, :]
    keys = torch.unique(left[mask] * edges + right[mask], sorted=True)
    del left, right, mask
    edge_rows, edge_cols = keys // edges, keys % edges
    edge_offsets = _csr_offsets(edge_rows, edges)
    del keys
    n, loops = len(lt.norms), lt.loops
    charge_rows = _transpose_columns(lt.sparse_charge, faces)
    offsets, cols, expansion = _basis_pattern(lt, (edge_offsets, edge_cols), (pair_offsets, pairs[:, 1]), charge_rows,
                                              n, edges, faces)
    bodies = None
    if prepared.admittance is not None:
        from .admittance_near import body_basis_blocks
        bodies = body_basis_blocks(prepared.admittance, lt)
    rows = torch.repeat_interleave(torch.arange(n, device=device), offsets.diff())
    # Every basis function has its self face pairs in the near list.
    diagonal = (rows == cols).nonzero().flatten()
    rule_points = far_rule.numel() // 4
    source_tile = max(1, min(faces, TILE_BYTES // (160 * faces)))
    structure = NearStructure(green, geometry, far_rule, near_rule, cache, pairs, pair_offsets, edge_rows,
                              edge_offsets, edge_cols, offsets, cols, diagonal, charge_rows, source_tile)
    structure.bodies = bodies
    if bodies is not None and action:
        from .body_compress import body_compression_plan
        structure.body_plans = [body_compression_plan(b, layout) for b in bodies]
        # A is not symmetric with DSA blocks; the true adjoint transposes the near part exactly.
        structure.transpose = torch.searchsorted(rows * n + cols, cols * n + rows)
    del rows
    structure.stats = dict(near_face_pairs=len(pairs), near_rwg_nnz=len(edge_cols),
                           preconditioner_nnz=len(cols), rule_points=rule_points,
                           product_expansion=expansion, source_tile=source_tile,
                           structure_bytes=sum(t.nbytes for t in (pairs, pair_offsets, edge_rows, edge_offsets,
                                                                  edge_cols, offsets, cols, diagonal))
                           + sum(t.nbytes for t in charge_rows)
                           + (sum(t.nbytes for t in cache) if base is None else 0))
    if bodies is not None:
        from .admittance_near import body_stats
        structure.stats.update(body_stats(bodies))
        if structure.transpose is not None:
            structure.stats["structure_bytes"] += structure.transpose.nbytes
    return structure


def _basis_pattern(lt, rwg, face, charge_rows, n, edges, faces):
    """Symbolic pass of the near basis products: merged pattern (offsets, cols) and the expansion count."""
    trees = n - lt.loops
    vq = sparse_product((*rwg, None), lt.sparse_q_rows, edges, n)
    qvq = sparse_product((lt.sparse_q[0], lt.sparse_q[1], None), vq, n, n)
    cp = sparse_product((lt.sparse_charge[0], lt.sparse_charge[1], None), (*face, None), trees, faces)
    cpc = sparse_product(cp, charge_rows, trees, trees)
    offsets, cols, _ = merge_blocks(qvq[:3], cpc[:3], n, lt.loops)
    return offsets, cols, vq[3] + qvq[3] + cp[3] + cpc[3]


class LayeredMatrixFreeOperator:
    """SurfaceOperatorProtocol action A = M + Far of one frequency in the equilibrated basis.

    ``precondition`` applies M^-1 (sparse LU); ``apply`` is the matrix-free
    action. ``adjoint`` is conj(A^T conj(y)); A^T = A except for surface-admittance blocks.
    Vectors use the last dimension.
    """

    supports_direct = False

    def __init__(self, prepared, structure, frequency_index, solver, preconditioner=None):
        self.prepared, self.structure, self.index = prepared, structure, frequency_index
        lt, layout = prepared.loop_tree, prepared.layout
        frequency = float(prepared.table.frequencies[frequency_index])
        self.frequency, self.omega = frequency, 2 * math.pi * frequency
        self.loops, n = lt.loops, len(lt.norms)
        self.shape, self.device, self.dtype = (n, n), layout.mesh.device, torch.complex128
        self.iterative_state = IterativeState()
        self.table = _frequency_view(prepared.table, frequency_index, frequency_index + 1)
        self.green, _ = assembly_operands(layout, self.table)
        self.supports = _support_operands(layout)
        s = structure
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # Basis entries: -i mu0 S Q^T V' Q S and i/(eps0 w0^2) C^T P C on trees.
        ratio = self.omega / lt.omega0
        self.basis_scale = torch.ones(n, device=self.device, dtype=torch.float64)
        self.basis_scale[self.loops:] = ratio
        # surface_admittance (body_compress.py): compressed W_b per body for the action, the dense
        # basis block only transiently, for its diagonal and its entries on the preconditioner keys.
        self.bodies, self.body_scales = None, ()
        if s.bodies is not None:
            from .body_compress import body_operators
            self.bodies, body_diagonals, body_entries, self.body_seconds = body_operators(
                s, prepared.admittance, frequency, self.basis_scale, solver)
        values = self.near_values(s)
        diagonal = values[s.diagonal]
        for b, block in zip(s.bodies or (), body_diagonals if self.bodies else ()):
            diagonal[b.columns] += block
        self.scale = diagonal.abs().rsqrt()
        del diagonal
        self.values = self._scaled(s, values)
        del values
        self.solver = solver
        # A wider preconditioner pattern changes only M^-1; the action keeps M.
        pattern = (self.values if preconditioner is None
                   else self._scaled(preconditioner, self.near_values(preconditioner)))
        if self.bodies is None:
            solver.factor(pattern)
        else:
            # Body block columns d_b = S D of the compressed action.
            self.body_scales = [self.basis_scale[b.columns] * self.scale[b.columns] for b in s.bodies]
            # PatternBodyLU (body_block_lu.py): body entries on the pattern keys, D-scaled there.
            solver.factor(pattern, body_entries, self.scale)
            del body_entries
        del pattern
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - started
        self._apply_calls = self._adjoint_calls = self._far_calls = 0
        self.far_seconds = 0.0

    def near_values(self, s):
        """Unscaled basis values of the near structure s at this frequency (loss folded)."""
        op = torch.ops.witwin_maxwell_mom_layered
        prepared, lt, n = self.prepared, self.prepared.loop_tree, self.shape[0]
        tiles = op.near_tiles(self.green, s.geometry, s.far_rule, s.near_rule, NEAR_RATIO, False,
                              self.table.exact_mode, self.table.ground, s.pairs, NEAR_CHUNK, s.cache)
        supports, slots, slopes, face_edges, support_slopes, areas2 = self.supports
        v = op.near_potential(s.edge_rows, s.edge_cols, s.pair_offsets, s.pairs[:, 1].contiguous(), tiles,
                              supports, slots, slopes, face_edges, support_slopes)
        if prepared.loss is not None:
            loss = prepared.loss
            op.near_loss(v, s.edge_offsets, s.edge_cols, loss.blocks.contiguous(), loss.columns.contiguous(),
                         loss.supports.contiguous(), loss.impedance[self.index].contiguous(), self.frequency)
        p = tiles[:, 9] * (4 / (areas2[s.pairs[:, 0]] * areas2[s.pairs[:, 1]]))
        del tiles
        edges, faces, trees = len(s.edge_offsets) - 1, len(s.pair_offsets) - 1, n - lt.loops
        # Basis products replayed per frequency in bounded row chunks (sparse_product).
        vq = sparse_product((s.edge_offsets, s.edge_cols, v), lt.sparse_q_rows, edges, n)
        del v
        qvq = sparse_product(lt.sparse_q, vq, n, n)
        del vq
        rows = torch.repeat_interleave(torch.arange(n, device=self.device), qvq[0].diff())
        magnetic = qvq[2] * ((-1j * MU0) * self.basis_scale[rows] * self.basis_scale[qvq[1]])
        del rows
        cp = sparse_product(lt.sparse_charge, (s.pair_offsets, s.pairs[:, 1], p), trees, faces)
        del p
        cpc = sparse_product(cp, s.charge_rows, trees, trees)
        del cp
        charge = cpc[2] * (1j / (EPS0 * lt.omega0 ** 2))
        _, _, values = merge_blocks((qvq[0], qvq[1], magnetic), (cpc[0], cpc[1], charge), n, lt.loops)
        del qvq, cpc, magnetic, charge
        return values

    def _scaled(self, s, values):
        """D values D on the CSR pattern of s."""
        rows = torch.repeat_interleave(torch.arange(self.shape[0], device=self.device), s.offsets.diff())
        return (values * (self.scale[rows] * self.scale[s.cols])).contiguous()

    def refresh(self):
        return None

    def read_access(self):
        return nullcontext()

    def _far(self, x):
        """Matrix-free far action on basis columns x [N, P]."""
        op = torch.ops.witwin_maxwell_mom_layered
        lt, s = self.prepared.loop_tree, self.structure
        supports, slots, slopes, face_edges, support_slopes, areas2 = self.supports
        z = self.scale[:, None] * x
        u = op.sparse_product((z * self.basis_scale[:, None])[None].contiguous(), lt.sparse_q_rows, True)[0]
        coefficients = op.face_coefficients(u.contiguous(), supports, slots, slopes, face_edges, support_slopes)
        charge = op.sparse_product(z[None, self.loops:].contiguous(), s.charge_rows, True)[0]
        coefficients[:, 3] = charge * (4 / areas2[:, None])
        moments = self.far_moments(coefficients)
        vu = op.edge_gather(moments, supports, slots, slopes, face_edges, support_slopes)
        y = op.sparse_product(vu[None].contiguous(), lt.sparse_q, True)[0] * ((-1j * MU0) * self.basis_scale[:, None])
        potential = (moments[:, 3] / areas2[:, None])[None].contiguous()
        y[self.loops:] += op.sparse_product(potential, lt.sparse_charge, True)[0] * (1j / (EPS0 * lt.omega0 ** 2))
        self._far_calls += 1
        return self.scale[:, None] * y

    def far_moments(self, coefficients):
        """Far face moments [F,4,P] of the identical far tiles, all columns per tile visit."""
        s = self.structure
        return torch.ops.witwin_maxwell_mom_layered.far_apply(
            self.green, s.geometry, s.far_rule, s.near_rule, NEAR_RATIO, False, self.table.exact_mode,
            self.table.ground, s.source_tile, coefficients, s.cache)

    def apply_columns(self, x, transpose=False):
        """A x (or A^T x) for basis columns x [N, P]; the far pass serves all P columns at once.

        With surface-admittance bodies A^T is exact: the CSR near part through the
        structure's transpose permutation and the compressed body blocks transposed.
        """
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        s = self.structure
        values = self.values[s.transpose] if transpose and s.transpose is not None else self.values
        out = torch.ops.witwin_maxwell_mom_layered.csr_matmul(s.offsets, s.cols, values, x.contiguous())
        for b, block, scale in zip(s.bodies or (), self.bodies or (), self.body_scales):
            out[b.columns] += block.apply(x[b.columns], scale, transpose)
        out = out + self._far(x)
        torch.cuda.synchronize(self.device)
        self.far_seconds += time.perf_counter() - started
        return out

    def apply(self, x):
        self._apply_calls += 1
        return self.apply_columns(x.reshape(-1, self.shape[0]).mT).mT.reshape(x.shape)

    def adjoint(self, y):
        self._adjoint_calls += 1
        return self.apply_columns(y.reshape(-1, self.shape[0]).mT.conj(), True).conj().mT.reshape(y.shape)

    def precondition(self, r):
        """M^-1 r for basis columns r [N, P]."""
        return self.solver.solve(r)

    def riesz_inverse(self, residual):
        return residual

    def preconditioner_scales(self, kind):
        if kind != "none":
            raise ValueError("Layered matrix-free operators support only none scaling; use the near-field LU")
        ones = torch.ones(self.shape[0], device=self.device, dtype=torch.float64)
        return ones, ones, False

    def drive(self):
        """Equilibrated port drive [N, P]."""
        sscale = self.basis_scale.clone()
        return (port_drive(self.prepared) * (self.scale * sscale)[:, None]).contiguous()

    def recovery(self):
        r = torch.full((self.shape[0],), 1 / self.prepared.loop_tree.omega0, device=self.device, dtype=torch.float64)
        r[:self.loops] = 1 / self.omega
        return self.scale * r

    def persistence_receipt(self):
        return {"version": "layered_matrix_free_operator_v1", "operator_backend": "matrix_free",
                "formulation_id": "mpie_layered_v1", "space": "equilibrated_loop_tree",
                "frequency_Hz": self.frequency, "unknowns": self.shape[0], "loops": self.loops,
                "near_ratio": NEAR_RATIO, "preconditioner": "near_field_sparse_lu_complex128",
                "executable_snapshot": False}

    def operator_stats(self):
        return {"operator_backend": "matrix_free", "apply_calls": self._apply_calls,
                "adjoint_calls": self._adjoint_calls, "far_actions": self._far_calls,
                "action_seconds": self.far_seconds, "setup_seconds": self.setup_seconds,
                **self.structure.stats, **self.solver.stats, "body_compression": self.body_stats()}

    def body_stats(self):
        """Per-body compression of W_b (body_compress.py): ranks, stored entries and dropped |W_b|_F."""
        if self.bodies is None:
            return {}
        return {"compression_seconds": tuple(self.body_seconds), **{key: tuple(b.stats[key] for b in self.bodies)
                for key in ("rank", "max_rank", "stored_entries", "dense_entries", "dropped_frobenius",
                            "w_frobenius")}}


class NearFieldLU:
    """complex128 sparse LU of the basis near operator; one symbolic plan per layout.

    The GPU direct-solver allocations lie outside torch's allocator. The
    deterministic mode makes factors and solves bitwise reproducible; it
    supports one right-hand side per solve (the default mode is not
    reproducible with several), so columns are solved one after another.

    The deterministic forward solve reads workspace the library never writes
    and so needs zero-initialised allocations. The device memory pool keeps
    freed blocks, so a later solver in the process would receive an earlier
    solver's dirty workspace and index out of range. Each instance therefore
    owns a library handle whose device allocator draws from the same pool and
    zero-fills every block on the call stream (cuda/cudss_workspace.cpp).
    """

    def __init__(self, structure, device):
        from nvmath.internal.memory import get_device_memory_resource
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
        import nvmath.bindings.cudss as cudss
        from .build import build_extension
        n = len(structure.offsets) - 1
        self.offsets = structure.offsets.to(torch.int32)
        self.cols = structure.cols.to(torch.int32)
        self.values = torch.zeros(len(structure.cols), device=device, dtype=torch.complex128)
        matrix = torch.sparse_csr_tensor(self.offsets, self.cols, self.values, size=(n, n))
        self.rhs = torch.empty(n, device=device, dtype=torch.complex128)
        index = self.values.device.index
        with torch.cuda.device(index):
            self.handle = cudss.create()
        pool = int(get_device_memory_resource(index).handle)
        cudss.set_device_mem_handler(self.handle, build_extension().cudss_zeroed_allocator(pool))
        self.solver = DirectSolver(matrix, self.rhs, options=DirectSolverOptions(handle=self.handle))
        flag = np.array([1], dtype=np.int32)
        cudss.config_set(self.solver.config_ptr, cudss.ConfigParam.DETERMINISTIC_MODE, flag.ctypes.data, 4)
        self.planned = False
        self.stats = {}

    def factor(self, values):
        self.values.copy_(values)
        if not self.planned:
            plan = self.solver.plan()
            peak = int(plan.memory_estimates.peak_device_memory)
            free, _ = torch.cuda.mem_get_info(self.values.device)
            if free < peak + 2 * 1024 ** 3:
                raise MemoryError(f"Near-field sparse LU needs {peak} device bytes plus a 2 GiB reserve; {free} free")
            self.stats["factor_peak_bytes_estimate"] = peak
            self.planned = True
        info = self.solver.factorize()
        self.stats["factor_nnz"] = int(info.lu_nnz)

    def solve(self, r):
        columns = []
        for column in r.mT:
            self.rhs.copy_(column)
            columns.append(self.solver.solve().clone())
        return torch.stack(columns, 1)

    def free(self):
        import nvmath.bindings.cudss as cudss
        self.solver.free()
        cudss.destroy(self.handle)
        self.handle = None


def block_gmres(operator, b, *, x0=None, tolerance=ADMITTANCE_TOLERANCE, restart=RESTART, max_steps=MAX_BLOCK_STEPS,
                monitor=None):
    """Right-preconditioned block GMRES for A X = B, B [N, P], driven through KrylovView.

    CGS2 block Arnoldi of A M^-1 with Householder block QR; the small
    least-squares problems use QR of the block Hessenberg matrix. Stops on the
    admittance bound |X|_F |R|_F <= tolerance |B^T X|_F (module docstring),
    confirmed by an explicit true residual. ``monitor(step, X, residual_norm)``
    observes every iterate. ``x0`` is an initial guess (one extra action for its
    residual), kept only if |B - A x0|_F < |B|_F; otherwise, and for None, the
    iteration starts from zero. Returns X and a log (``warm_start``: x0 kept).
    """
    view = KrylovView(operator)
    n, p = b.shape
    steps = actions = restarts = 0
    x, residual, warm = torch.zeros_like(b), b.clone(), False
    if x0 is not None:
        start = b - view.apply(x0.mT).mT
        actions += 1
        if float(torch.linalg.matrix_norm(start)) < float(torch.linalg.matrix_norm(b)):
            x, residual, warm = x0.clone(), start, True
    history = []
    while True:
        basis, s = torch.linalg.qr(residual)
        blocks = [basis]
        hessenberg = torch.zeros((p * (restart + 1), p * restart), device=b.device, dtype=b.dtype)
        rhs = torch.zeros((p * (restart + 1), p), device=b.device, dtype=b.dtype)
        rhs[:p] = s
        accepted = False
        for j in range(restart):
            image = view.apply(operator.precondition(blocks[-1]).mT).mT
            actions += 1
            stacked = torch.cat(blocks, 1)
            coefficients = stacked.mH @ image
            image = image - stacked @ coefficients
            correction = stacked.mH @ image
            image = image - stacked @ correction
            q, r = torch.linalg.qr(image)
            hessenberg[:p * (j + 1), p * j:p * (j + 1)] = coefficients + correction
            hessenberg[p * (j + 1):p * (j + 2), p * j:p * (j + 1)] = r
            blocks.append(q)
            steps += 1
            h = hessenberg[:p * (j + 2), :p * (j + 1)]
            qh, rh = torch.linalg.qr(h, mode="complete")
            projected = qh.mH @ rhs[:p * (j + 2)]
            y = torch.linalg.solve_triangular(rh[:p * (j + 1)], projected[:p * (j + 1)], upper=True)
            residual_norm = float(torch.linalg.matrix_norm(projected[p * (j + 1):]))
            candidate = x + operator.precondition(torch.cat(blocks[:j + 1], 1) @ y)
            bound = residual_norm * float(torch.linalg.matrix_norm(candidate))
            admittance = float(torch.linalg.matrix_norm(b.mT @ candidate))
            history.append(dict(step=steps, residual=residual_norm, bound=bound / admittance))
            if monitor is not None:
                monitor(steps, candidate, residual_norm)
            if bound <= tolerance * admittance:
                accepted = True
                break
            if steps >= max_steps:
                break
        x = candidate
        residual = b - view.apply(x.mT).mT
        actions += 1
        true_norm = float(torch.linalg.matrix_norm(residual))
        true_bound = true_norm * float(torch.linalg.matrix_norm(x)) / float(torch.linalg.matrix_norm(b.mT @ x))
        history.append(dict(step=steps, true_residual=true_norm, true_bound=true_bound))
        if accepted and true_bound <= tolerance:
            break
        if steps >= max_steps:
            raise RuntimeError(f"Layered block GMRES did not reach the admittance bound {tolerance:.3g} "
                               f"in {steps} block steps (bound {true_bound:.3g})")
        restarts += 1
    return x, dict(block_steps=steps, actions=actions, restarts=restarts, warm_start=warm, true_residual=true_norm,
                   admittance_bound=true_bound, residual_norm_B=float(torch.linalg.matrix_norm(b)),
                   history=history)


def solve_matrix_free(prepared):
    """Per-frequency near LU + block GMRES; returns (admittance [K,P,P], currents [K,N,P], info)."""
    from .body_preconditioner import body_preconditioner_structure
    from .network import recover_scaled_chunk
    device = prepared.excitation.device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    structure = build_near_structure(prepared)
    wide = body_preconditioner_structure(prepared, structure)
    torch.cuda.synchronize(device)
    structure.stats["structure_seconds"] = time.perf_counter() - started
    if wide is not None:
        structure.stats.update(body_preconditioner_nnz=wide.stats["preconditioner_nnz"],
                               body_wide_ratio=wide.stats["wide_ratio"])
    total, ports = len(prepared.table.frequencies), prepared.excitation.shape[1]
    currents = torch.empty((total, prepared.layout.basis.n_edges, ports), device=device, dtype=torch.complex128)
    admittance = torch.empty((total, ports, ports), device=device, dtype=torch.complex128)
    from .body_block_lu import near_preconditioner
    solver = near_preconditioner(structure, wide, device)
    logs, coefficients = [], None
    for index in range(total):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        operator = LayeredMatrixFreeOperator(prepared, structure, index, solver, wide)
        rhs = operator.drive()
        # R6-3 (D3): start from the previous frequency's loop/tree coefficients c = X r (recovery).
        recovery = operator.recovery()[:, None]
        solution, log = block_gmres(operator, rhs, x0=None if coefficients is None else coefficients / recovery)
        coefficients = solution * recovery
        y, x = recover_scaled_chunk(prepared, solution[None], rhs[None], operator.recovery()[None],
                                    frequency_start=index)
        admittance[index], currents[index] = y[0], x[0]
        torch.cuda.synchronize(device)
        log.update(operator.operator_stats(), seconds=time.perf_counter() - started)
        logs.append(log)
        del operator
    solver.free()
    return admittance, currents, dict(structure=structure.stats, frequencies=logs)
