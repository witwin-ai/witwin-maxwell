"""Compressed surface-admittance body blocks for the iterative layered backends.

conductor_model="surface_admittance" adds one block per lossy body to the
equilibrated system (admittance_near.py),

    B_b = d_b Q_b^T (c W_b) Q_b d_b,   c = -i mu0 * i/(mu0 w),  d_b = S D on the body's c_b basis columns,

with W_b [n_b, n_b] the exterior-RWG admittance block and Q_b the sparse
loop/tree rows of those RWGs. Stored dense, B_b costs 16 c_b^2 bytes and its
preconditioner (R4-1) a dense LU over every column coupled to a body, which
is all of N once bodies span the layout.

Action. W_b is compressed as an H-matrix on the exterior RWG centres: a
median-bisection cluster tree (LEAF points per leaf), standard admissibility
min(diam) <= ETA dist of the bounding boxes. Inadmissible leaf pairs stay
dense; every admissible block is replaced by a full-pivot cross approximation
(greedy rank-one crosses through the largest residual entry, batched over the
blocks of one shape) that stops once the explicit residual |R|_F of the block
is <= TOLERANCE |W_b|_F. In the DSA regime W is smooth away from the diagonal
(a single-layer-like interior response); in the low regime it is the local
Gram z <g, g>, whose admissible blocks are exactly zero (rank 0). The dropped
residuals give the exact error eps_b = sqrt(sum |R|_F^2) = |W_b - W~_b|_F,
reported per body. (A truncated SVD per block gives ~25% lower ranks but the
looped cuSOLVER SVDs took minutes per body; the cross is a few batched kernels
per term.)

The compressed action is three row-owner CSR products, no atomics:

    u = Q_b (d_b x),  t = V^H u,  v = [N | U] [u; t],  y = d_b (Q_b^T v),

N the dense leaf entries and U, V^H the stacked cross factors (c folded
into U). A^T uses the transposed CSRs, built on first use.

Preconditioner. B_b enters the sparse near LU only on the keys the near
pattern already has (body_block_lu.PatternBodyLU), so the LU fill is that of
the body-free pattern; the rest of B_b stays in the exact action.
"""

from dataclasses import dataclass, field
import math
import time

import torch

from .admittance import MU0 as ADMITTANCE_MU0


LEAF = 32
ETA = 1.0
TOLERANCE = 1e-10


def _csr(rows, cols, values, count, width):
    """Row-sorted CSR (offsets, cols, values) of unique (row, col) triplets."""
    order = torch.argsort(rows * width + cols)
    rows, cols, values = rows[order], cols[order], values[order]
    offsets = torch.cat((rows.new_zeros(1), torch.cumsum(torch.bincount(rows, minlength=count), 0)))
    return offsets, cols.contiguous(), values.contiguous()


def _cluster(points, leaf):
    """Median-bisection cluster tree on CPU: nodes [index tensor, lo, hi, children]."""
    nodes = []

    def build(index):
        p = points[index]
        lo, hi = p.min(0).values, p.max(0).values
        me = len(nodes)
        nodes.append([index, lo, hi, None])
        if len(index) > leaf:
            order = torch.argsort(p[:, int(torch.argmax(hi - lo))], stable=True)
            half = len(index) // 2
            nodes[me][3] = (build(index[order[:half]]), build(index[order[half:]]))
        return me

    build(torch.arange(len(points)))
    return nodes


def _partition(nodes, eta):
    """Leaf block pairs (t, s, admissible) of the block cluster tree, in a fixed recursion order."""
    out = []

    def diameter(n):
        return float(torch.linalg.vector_norm(nodes[n][2] - nodes[n][1]))

    def distance(t, s):
        gap = torch.clamp(torch.maximum(nodes[t][1] - nodes[s][2], nodes[s][1] - nodes[t][2]), min=0)
        return float(torch.linalg.vector_norm(gap))

    def visit(t, s):
        if min(diameter(t), diameter(s)) <= eta * distance(t, s):
            out.append((t, s, True))
            return
        ct, cs = nodes[t][3], nodes[s][3]
        if ct is None and cs is None:
            out.append((t, s, False))
        elif ct is None:
            for c in cs:
                visit(t, c)
        elif cs is None:
            for c in ct:
                visit(c, s)
        else:
            for x in ct:
                for y in cs:
                    visit(x, y)

    visit(0, 0)
    return out


@dataclass
class BodyCompressionPlan:
    """Frequency-independent H-matrix plan of one body block (device tensors)."""

    n: int
    near_rows: torch.Tensor
    near_cols: torch.Tensor
    far: list                 # [(rows [G, m], cols [G, k])] grouped by block shape, recursion order inside a group
    q: tuple                  # CSR of Q_b [n_b, c_b] (complex values)
    qt: tuple                 # CSR of Q_b^T [c_b, n_b]
    stats: dict = field(default_factory=dict)


def body_compression_plan(block, layout):
    """H-matrix plan of one admittance_near.BodyBasisBlock on its exterior RWG centres."""
    basis, centroids = layout.basis, layout.mesh.centroids
    ids = block.ids
    device = ids.device
    centres = ((centroids[basis.plus_faces[ids]] + centroids[basis.minus_faces[ids]]) / 2).cpu()
    nodes = _cluster(centres, LEAF)
    pairs = _partition(nodes, ETA)
    near_rows, near_cols, groups = [], [], {}
    for t, s, admissible in pairs:
        it, js = nodes[t][0], nodes[s][0]
        if admissible:
            groups.setdefault((len(it), len(js)), []).append((it, js))
        else:
            near_rows.append(it[:, None].expand(-1, len(js)).reshape(-1))
            near_cols.append(js[None].expand(len(it), -1).reshape(-1))
    n = len(ids)
    near_rows, near_cols = torch.cat(near_rows), torch.cat(near_cols)
    order = torch.argsort(near_rows * n + near_cols)
    far = [(torch.stack([g[0] for g in group]).to(device), torch.stack([g[1] for g in group]).to(device))
           for _, group in sorted(groups.items())]
    q_values = block.q_values.to(torch.complex128)
    c = len(block.columns)
    plan = BodyCompressionPlan(n, near_rows[order].to(device), near_cols[order].to(device), far,
                               _csr(block.q_rows, block.q_cols, q_values, n, c),
                               _csr(block.q_cols, block.q_rows, q_values, c, n))
    plan.stats = dict(exterior=n, columns=c, near_entries=len(near_rows),
                      far_blocks=sum(len(r) for r, _ in far), leaf=LEAF, eta=ETA)
    return plan


def _cross(blocks, threshold):
    """Batched full-pivot cross approximation of blocks [G, m, n] to |residual|_F <= threshold per block.

    Each step takes the largest residual entry (first in row-major order on ties) of every
    active block as pivot and subtracts the rank-one cross through it; a block stops when its
    explicit residual Frobenius norm is <= threshold. Returns U [G, m, K], V [G, K, n] (prefix
    of counts[g] terms per block), counts [G] and the squared residual norms [G] that are dropped.
    """
    residual = blocks.clone()
    batch, m, n = residual.shape
    arange = torch.arange(batch, device=residual.device)
    norm2 = residual.abs().square().sum((1, 2))
    limit = threshold * threshold
    active = norm2 > limit
    counts = torch.zeros(batch, device=residual.device, dtype=torch.int64)
    us, vs = [], []
    for _ in range(min(m, n)):
        if not bool(active.any()):
            break
        flat = residual.abs().reshape(batch, -1).argmax(1)
        i, j = flat // n, flat % n
        pivot = residual[arange, i, j]
        pivot = torch.where(active, pivot, torch.ones_like(pivot))
        column = residual[arange, :, j] * active[:, None]
        row = residual[arange, i, :] / pivot[:, None]
        residual -= column[:, :, None] * row[:, None, :]
        us.append(column)
        vs.append(row)
        counts += active
        norm2 = torch.where(active, residual.abs().square().sum((1, 2)), norm2)
        active = active & (norm2 > limit)
    if not us:
        return blocks.new_zeros((batch, m, 0)), blocks.new_zeros((batch, 0, n)), counts, norm2
    return torch.stack(us, 2), torch.stack(vs, 1), counts, norm2


class CompressedBodyBlock:
    """B_b = d Q^T W~ Q d of one body and frequency; W~ the truncated H-matrix of c W_b."""

    def __init__(self, plan, w, constant):
        n = plan.n
        device = w.device
        self.plan = plan
        threshold = TOLERANCE * float(torch.linalg.matrix_norm(w))
        rows = [plan.near_rows]
        cols = [plan.near_cols]
        values = [w[plan.near_rows, plan.near_cols] * constant]
        v_rows, v_cols, v_values = [], [], []
        rank, dropped, ranks = 0, 0.0, []
        for block_rows, block_cols in plan.far:
            u, v, counts, residual = _cross(w[block_rows[:, :, None], block_cols[:, None, :]], threshold)
            dropped += float(residual.sum())
            ranks.append(counts)
            # Kept terms are a prefix per block (a block stops once its residual is below the threshold).
            g, k = (torch.arange(u.shape[2], device=device)[None] < counts[:, None]).nonzero().unbind(-1)
            # Rank index of each kept (block, term) in recursion order.
            index = rank + torch.arange(len(g), device=device)
            rank += len(g)
            m, width = block_rows.shape[1], block_cols.shape[1]
            # U columns: row block_rows[g, i], column index, value u[g, i, k] c.
            rows.append(block_rows[g].reshape(-1))
            cols.append((n + index)[:, None].expand(-1, m).reshape(-1))
            values.append((u[g, :, k] * constant).reshape(-1))
            v_rows.append(index[:, None].expand(-1, width).reshape(-1))
            v_cols.append(block_cols[g].reshape(-1))
            v_values.append(v[g, k, :].reshape(-1))
            del u, v
        empty_i = plan.near_rows.new_empty(0)
        self.rank = rank
        self.left = _csr(torch.cat(rows), torch.cat(cols), torch.cat(values), n, n + rank)
        self.right = _csr(torch.cat(v_rows) if v_rows else empty_i, torch.cat(v_cols) if v_cols else empty_i,
                          torch.cat(v_values) if v_values else w.new_empty(0), rank, n)
        self.transposed = None
        ranks = torch.cat(ranks) if ranks else empty_i
        self.stats = dict(rank=rank, max_rank=int(ranks.max()) if len(ranks) else 0,
                          stored_entries=len(self.left[1]) + len(self.right[1]),
                          dense_entries=n * n, dropped_frobenius=math.sqrt(dropped) * abs(constant),
                          w_frobenius=threshold / TOLERANCE * abs(constant))

    def _transpose(self):
        n, rank = self.plan.n, self.rank

        def rows_of(csr, count):
            return torch.repeat_interleave(torch.arange(count, device=csr[1].device), csr[0].diff())

        left_rows, right_rows = rows_of(self.left, n), rows_of(self.right, rank)
        near = self.left[1] < n
        # W~^T = N^T + V U^T: t = U^T u (rank rows), v = [N^T | V] [u; t].
        ut = _csr(self.left[1][~near] - n, left_rows[~near], self.left[2][~near], rank, n)
        cols = torch.cat((left_rows[near], n + right_rows))
        rows = torch.cat((self.left[1][near], self.right[1]))
        values = torch.cat((self.left[2][near], self.right[2]))
        self.transposed = (_csr(rows, cols, values, n, n + rank), ut)

    def apply(self, x, scale, transpose=False):
        """B_b x (or B_b^T x) for the body's basis columns x [c_b, P]; scale = d_b [c_b]."""
        op = torch.ops.witwin_maxwell_mom_layered
        plan = self.plan
        if transpose and self.transposed is None:
            self._transpose()
        left, right = self.transposed if transpose else (self.left, self.right)
        u = op.csr_matmul(*plan.q, (scale[:, None] * x).contiguous())
        # Rank 0 (every admissible block dropped, e.g. the local low-regime W): only the dense leaves.
        v = op.csr_matmul(*left, torch.cat((u, op.csr_matmul(*right, u))) if self.rank else u)
        return scale[:, None] * op.csr_matmul(*plan.qt, v)

    @property
    def nbytes(self):
        return sum(t.nbytes for csr in (self.left, self.right) for t in csr)


def fold_constant(frequency):
    """c = -i mu0 * i/(mu0_admittance w): the dense fold of W into the native basis units."""
    from .matrix_free import MU0
    return (-1j * MU0) * 1j / (ADMITTANCE_MU0 * 2 * math.pi * frequency)


def body_operators(structure, admittance, frequency, basis_scale, solver):
    """Per body: the compressed action block, the unscaled B_b diagonal and B_b on the preconditioner keys.

    Bodies stream one at a time: W_b (SurfaceAdmittance.body_block) is formed, folded and
    compressed, then released before the next body's interior operators exist.
    """
    from .admittance_near import body_values
    constant = fold_constant(frequency)
    compressed, diagonals, entries, seconds = [], [], [], []
    for index, (b, plan) in enumerate(zip(structure.bodies, structure.body_plans)):
        ids, w = admittance.body_block(index, frequency)
        block = body_values([b], {b.owner: (ids, w)}, frequency, basis_scale)[0]
        diagonals.append(block.diagonal().clone())
        entries.append(solver.body_entries(index, block))
        del block
        torch.cuda.synchronize(w.device)
        started = time.perf_counter()
        compressed.append(CompressedBodyBlock(plan, w, constant))
        torch.cuda.synchronize(w.device)
        seconds.append(time.perf_counter() - started)
        del w
    return compressed, diagonals, entries, seconds
