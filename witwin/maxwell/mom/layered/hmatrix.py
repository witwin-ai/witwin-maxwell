"""H-matrix far field for the layered MPIE backend: operator_backend="hmatrix".

The system, near-field structure, sparse-LU right preconditioner and block
GMRES are those of the matrix-free backend (matrix_free.py). Only the far
action changes: matrix_free re-integrates every far face pair in every action;
hmatrix compresses the far face-pair operator once per frequency.

Far operator. far_apply contracts face coefficients x [F,4,P] with face-pair
tiles: a vector channel (3 vertex moments per face, a 3F x 3F matrix with
entries tile(t,s)[3a+b]) and a scalar channel (F x F, tile(t,s)[9]). Canonical
tiles make both channels complex symmetric: tile(s,t) is the transposed
tile(t,s), bitwise.

Clusters. One support octree (shared cluster.py) per z class: each metal plane
and the vertical walls. Blocks are unordered cluster pairs over all class
pairs, so same-plane and cross-plane interactions are partitioned alike.

Admissibility is lateral, for every class pair. A layered Green function
depends on the lateral distance rho and on (z, z') only through its tabulated
remainder and images, some of which approach the observer plane (reflection
across an interface between the planes), so a vertical separation is never
used to admit a block. A block (T, S) is admissible when

    max(diam_xy T, diam_xy S) <= ETA * gap_xy  and  gap_xy^2 >= NEAR_RATIO^2 h^2 (1 + 1e-6),

with h^2 the largest squared face edge in T and S. The second condition puts
every face pair of the block in the far class of the assembly's own predicate
(pair_near / wall_near need a centroid lateral distance below NEAR_RATIO h), so
near pairs exist only in inadmissible leaf blocks and the near part of S2a
stays exact. The build checks this with integer counts (near_pairs_in_leaves).

Entries. Admissible blocks are sampled with the S2a list kernel (near_tiles)
over explicit (t, s) lists: the identical pair kernels of the dense assembly,
so every sampled entry is bitwise the dense tile entry.

Compression. Lockstep cross approximation over batches of blocks, separately
for the two channels, with face pivots: one sampled face row and face column
give a w x w pivot block (w = 3 vertex rows in the vector channel, w = 1 for
the scalar channel), and the update C W^+ R adds up to w ranks per sample
(W^+ by complete pivoting, pivots below PIVOT_RTOL times the first skipped). Pivots are the largest residual
magnitude with ties broken by the lowest index. When the update estimate
|dS_k|_F <= ACA_TOLERANCE |S_k|_F holds, a pseudo-random unused check face
(integer hash) must satisfy sqrt(m) |R_t|_F <= ACA_TOLERANCE |S_k|_F, otherwise
it becomes the next pivot. Gram/eigh recompression truncates at the same
relative tolerance. Each block is stored once; its transpose serves (S, T),
so the compressed operator is exactly complex symmetric, as the GMRES
stopping rule requires.

Leaf blocks. Far pairs of inadmissible leaf blocks (near pairs removed) keep
their exact complex128 tiles, stored once per unordered pair.

Tolerance. ACA_TOLERANCE = 1e-6 is the relative Frobenius error per block
and channel. Per entry that is about 1e-6 max|block|, inside the numerical
equivalence standard's float32 setup bound 64 eps32 max|ref| (7.6e-6) and at
the rounding floor of the dense reference itself, whose regular far vector
moments are accumulated in float32: a tighter tolerance would only chase
rounding noise in the ranks. Acceptance is measured agreement with dense,
|dY|_F/|Y|_F <= 1e-4 (the run-level bound), next to the GMRES stopping bound
1e-5.

Diagnostics (metadata, not gates). The compressed far operator F~ = F + dF
changes the admittance by dY = -X^T dA X~ (A complex symmetric). In face
coefficients c of the solution, with e_b >= |dF_b|_2 the block estimate (ACA
check plus truncation tail), the block terms are

    t_b = 2 (mu0 e_b^v |c^v_T|_F |c^v_S|_F + e_b^s |c^s_T|_F |c^s_S|_F / (4 eps0 w0^2)).

compression_bound_worst_case = sum_b t_b / |B^T X|_F is the triangle-inequality
bound; it is 2-3 orders above the measured error, because block errors do
not align. compression_estimate_quadrature = |t|_2 / |B^T X|_F treats them as
independent. hmatrix_exact_check=True (LayeredSurfaceConfig) measures the
first-order dY with one exact far action per frequency (O(N^2)) and refuses
a solve above COMPRESSION_TOLERANCE = 1e-5.
"""

from dataclasses import dataclass, field
import time

import torch

from ..cluster import build_support_octree
from .matrix_free import (EPS0, MU0, NEAR_CHUNK, NEAR_RATIO, LayeredMatrixFreeOperator,
                          _csr_offsets, block_gmres, build_near_structure)


ETA = 1.0
LEAF_SIZE = 32
ACA_TOLERANCE = 1e-6
COMPRESSION_TOLERANCE = 1e-5
NEAR_MARGIN = 1e-6
BATCH_BYTES = 1 << 27
RECOMPRESS_BYTES = 1 << 27
EIGH_BATCH = 256
LEAF_CHUNK = 1 << 22
INITIAL_RANK = 32
PIVOT_RTOL = 1e-2
GRAM_RTOL = 1e-14
CHANNELS = ((3, 0), (1, 3))


def _first_argmax(values, mask):
    """Index of the largest masked value per row; ties go to the lowest index."""
    values = values.masked_fill(~mask, -1.0)
    best = values.amax(-1, keepdim=True)
    index = torch.arange(values.shape[-1], device=values.device).expand_as(values)
    return torch.where(values == best, index, values.shape[-1]).amin(-1)


@dataclass
class ClusterBlocks:
    """Frequency-independent cluster partition of one layout."""

    faces: torch.Tensor          # [F] face ids in cluster order (classes concatenated)
    blocks: torch.Tensor         # [B,4] admissible (row start, m, column start, n) positions
    cross: torch.Tensor          # [B] bool: row and column clusters in different z classes
    batches: list                # admissible block ids per ACA batch
    entry_offsets: torch.Tensor  # [F+1] CSR of low-rank entries per face
    entries: torch.Tensor        # [E,3] (block, local index, side)
    leaf_pairs: torch.Tensor     # [L,2] far face pairs of inadmissible leaf blocks
    forward_offsets: torch.Tensor
    forward: torch.Tensor
    reverse_offsets: torch.Tensor
    reverse: torch.Tensor
    stats: dict = field(default_factory=dict)


def _node_tables(layout, edge2):
    """Per-class octrees concatenated: node ranges, lateral boxes, max edge^2, children."""
    triangles = layout.mesh.triangles
    lower, upper = triangles.amin(1), triangles.amax(1)
    planes = layout.triangle_planes
    faces, starts, stops, lows, highs, children, h2, leaf, cls, roots = [], [], [], [], [], [], [], [], [], []
    base = node_base = 0
    # Setup-only transfer of the few z-class labels.
    for index, z in enumerate(torch.unique(planes).tolist()):
        members = (planes == z).nonzero().flatten()
        tree = build_support_octree(lower[members], upper[members], leaf_size=LEAF_SIZE)
        ordered = members[tree.permutation]
        is_leaf = (tree.children < 0).all(1)
        # Leaves partition the positions; internal maxima follow bottom-up by depth.
        values = edge2[ordered]
        node_h2 = torch.zeros(len(tree.starts), device=values.device, dtype=values.dtype)
        leaves = is_leaf.nonzero().flatten()
        order = leaves[torch.argsort(tree.starts[leaves])]
        node_h2[order] = torch.segment_reduce(values, "max", lengths=tree.stops[order] - tree.starts[order])
        for depth in range(int(tree.depth.max()), -1, -1):
            internal = ((tree.depth == depth) & ~is_leaf).nonzero().flatten()
            if len(internal):
                child = tree.children[internal]
                node_h2[internal] = torch.where(child >= 0, node_h2[child.clamp_min(0)], 0.).amax(1)
        faces.append(ordered)
        starts.append(tree.starts + base)
        stops.append(tree.stops + base)
        lows.append(tree.lower[:, :2])
        highs.append(tree.upper[:, :2])
        children.append(torch.where(tree.children >= 0, tree.children + node_base, -1))
        h2.append(node_h2)
        leaf.append(is_leaf)
        cls.append(torch.full_like(tree.starts, index))
        roots.append(node_base)
        base += len(members)
        node_base += len(tree.starts)
    return (torch.cat(faces), torch.cat(starts), torch.cat(stops), torch.cat(lows), torch.cat(highs),
            torch.cat(children), torch.cat(h2), torch.cat(leaf), torch.cat(cls), roots)


def _partition(starts, stops, low, high, children, h2, leaf, roots):
    """Unordered block pairs: admissible [A,2] and inadmissible leaf [L,2] node pairs.

    Each unordered face pair lies in exactly one block. A diagonal pair (a, a)
    splits into child pairs (c_i, c_j), i <= j; other pairs split the node of
    larger lateral diameter (cluster.py's rule). Depth-first, deterministic.
    """
    device = starts.device
    pending = [torch.tensor([(a, b) for i, a in enumerate(roots) for b in roots[i:]], device=device)]
    upper = torch.triu_indices(8, 8, device=device)
    admissible, leaves = [], []
    diameter = torch.linalg.vector_norm(high - low, dim=1)
    while pending:
        pairs = pending.pop()
        a, b = pairs.unbind(1)
        diagonal = a == b
        gap = torch.linalg.vector_norm(torch.maximum(low[a] - high[b], low[b] - high[a]).clamp_min(0), dim=1)
        da, db = diameter[a], diameter[b]
        separated = gap * gap >= NEAR_RATIO ** 2 * torch.maximum(h2[a], h2[b]) * (1 + NEAR_MARGIN)
        good = ~diagonal & separated & (torch.maximum(da, db) <= ETA * gap)
        terminal = good | (leaf[a] & leaf[b])
        admissible.append(pairs[good])
        leaves.append(pairs[terminal & ~good])
        split = ~terminal & ~diagonal
        a, b = a[split], b[split]
        split_a = ~leaf[a] & (leaf[b] | (da[split] >= db[split]))
        child = torch.where(split_a[:, None], children[a], children[b])
        left = torch.where(split_a[:, None], child, a[:, None].expand(-1, 8))
        right = torch.where(split_a[:, None], b[:, None].expand(-1, 8), child)
        off = torch.stack((left, right), -1)[child >= 0]
        node = pairs[~terminal & diagonal, 0]
        child = children[node][:, upper]
        on = child.permute(0, 2, 1)[(child >= 0).all(1)]
        following = torch.cat((on, off))
        if len(following):
            # Reverse push keeps a deterministic depth-first order.
            pending.extend(reversed(following.split(4096)))
    return torch.cat(admissible), torch.cat(leaves)


def _ranges(starts, stops, nodes):
    return starts[nodes], stops[nodes] - starts[nodes]


def _pair_positions(row_start, m, col_start, n, diagonal):
    """All position pairs of node blocks; diagonal blocks keep i < j only."""
    sizes = m * n
    block = torch.repeat_interleave(torch.arange(len(m), device=m.device), sizes)
    local = torch.arange(len(block), device=m.device) - torch.repeat_interleave(torch.cumsum(sizes, 0) - sizes, sizes)
    i, j = local // n[block], local % n[block]
    keep = ~diagonal[block] | (i < j)
    return (row_start[block] + i)[keep], (col_start[block] + j)[keep]


def build_cluster_blocks(prepared, structure):
    """Cluster trees, admissible blocks, ACA batches and the exact leaf far-pair lists."""
    layout = prepared.layout
    F = len(layout.mesh.faces)
    edge2 = structure.geometry[4]
    faces, starts, stops, low, high, children, h2, leaf, cls, roots = _node_tables(layout, edge2)
    admissible, leaves = _partition(starts, stops, low, high, children, h2, leaf, roots)
    row_start, m = _ranges(starts, stops, admissible[:, 0])
    col_start, n = _ranges(starts, stops, admissible[:, 1])
    blocks = torch.stack((row_start, m, col_start, n), 1).contiguous()
    cross = cls[admissible[:, 0]] != cls[admissible[:, 1]]
    # Low-rank entries: face rows of T (side 0) and columns of S (side 1), sorted by face.
    count = len(blocks)
    ids = torch.arange(count, device=blocks.device)
    rows = torch.repeat_interleave(ids, m)
    cols = torch.repeat_interleave(ids, n)
    local_rows = torch.arange(len(rows), device=rows.device) - torch.repeat_interleave(torch.cumsum(m, 0) - m, m)
    local_cols = torch.arange(len(cols), device=cols.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
    entry_face = torch.cat((faces[row_start[rows] + local_rows], faces[col_start[cols] + local_cols]))
    entries = torch.cat((torch.stack((rows, local_rows, torch.zeros_like(rows)), 1),
                         torch.stack((cols, local_cols, torch.ones_like(cols)), 1)))
    order = torch.argsort(entry_face, stable=True)
    entries, entry_offsets = entries[order].contiguous(), _csr_offsets(entry_face[order], F)
    # ACA batches: blocks sorted by size, chunked greedily under a factor
    # workspace budget (both channels, 4 (m + n) INITIAL_RANK complex128 per
    # block, doubled for growth). Few batches mean few lockstep steps.
    order = torch.argsort(torch.maximum(m, n) * (1 << 32) + torch.minimum(m, n), stable=True)
    sizes = torch.stack((m[order], n[order]), 1).tolist()
    batches, first, mx, nx = [], 0, 0, 0
    for index, (a, b) in enumerate(sizes):
        wide, tall = max(mx, a), max(nx, b)
        if index > first and (index + 1 - first) * 16 * 4 * (wide + tall) * 2 * INITIAL_RANK > BATCH_BYTES:
            batches.append(order[first:index])
            first, wide, tall = index, a, b
        mx, nx = wide, tall
    if sizes:
        batches.append(order[first:])
    # Inadmissible leaf blocks: every pair, minus the near pairs (exact key search).
    pairs = structure.pairs
    near_keys = pairs[:, 0] * F + pairs[:, 1]
    leaf_row, leaf_m = _ranges(starts, stops, leaves[:, 0])
    leaf_col, leaf_n = _ranges(starts, stops, leaves[:, 1])
    diagonal = leaves[:, 0] == leaves[:, 1]
    found = covered = 0
    kept = []
    sizes = (leaf_m * leaf_n).tolist()
    first = 0
    while first < len(sizes):
        last, total = first, 0
        while last < len(sizes) and (total == 0 or total + sizes[last] <= LEAF_CHUNK):
            total += sizes[last]
            last += 1
        part = slice(first, last)
        i, j = _pair_positions(leaf_row[part], leaf_m[part], leaf_col[part], leaf_n[part], diagonal[part])
        t, s = faces[i], faces[j]
        keys = t * F + s
        place = torch.searchsorted(near_keys, keys).clamp_max(len(near_keys) - 1)
        near = near_keys[place] == keys
        found += int(near.sum())
        covered += len(keys)
        kept.append(torch.stack((t[~near], s[~near]), 1))
        first = last
    leaf_pairs = torch.cat(kept).contiguous()
    forward = torch.argsort(leaf_pairs[:, 0], stable=True)
    reverse = torch.argsort(leaf_pairs[:, 1], stable=True)
    clusters = ClusterBlocks(faces, blocks, cross, batches, entry_offsets, entries, leaf_pairs,
                             _csr_offsets(leaf_pairs[forward, 0], F), forward,
                             _csr_offsets(leaf_pairs[reverse, 1], F), reverse)
    self_pairs = F
    clusters.stats = dict(
        z_classes=len(roots), cluster_nodes=len(starts), admissible_blocks=count,
        cross_class_blocks=int(cross.sum()), leaf_blocks=len(leaves), aca_batches=len(batches),
        admissible_face_pairs=int((m * n).sum()), leaf_far_pairs=len(leaf_pairs),
        # Exact coverage: every unordered face pair once (diagonal leaves count i < j plus F self pairs).
        unordered_pairs_covered=int((m * n).sum()) + covered + self_pairs,
        unordered_pairs=F * (F + 1) // 2,
        near_pairs_in_leaves=found + self_pairs, near_pairs_unordered=(len(pairs) + self_pairs) // 2,
        leaf_bytes=leaf_pairs.nbytes + 160 * len(leaf_pairs) + 16 * len(leaf_pairs) + 16 * (F + 1),
        entry_bytes=entries.nbytes + entry_offsets.nbytes)
    return clusters


class TileSampler:
    """Face-pair tiles [K,10] of explicit (t, s) lists for one frequency.

    The S2a list kernel runs the identical pair kernels of the dense assembly,
    so each sampled tile is bitwise the dense tile of that pair.
    """

    def __init__(self, green, structure, table):
        self.green, self.structure, self.table = green, structure, table
        self.calls = self.pairs = 0

    def __call__(self, rows, cols):
        s = self.structure
        self.calls += 1
        self.pairs += len(rows)
        return torch.ops.witwin_maxwell_mom_layered.near_tiles(
            self.green, s.geometry, s.far_rule, s.near_rule, NEAR_RATIO, False, self.table.exact_mode,
            self.table.ground, torch.stack((rows, cols), 1).contiguous(), NEAR_CHUNK, s.cache)


class _ChannelACA:
    """Lockstep face-pivot cross approximation of one channel over a batch of blocks.

    A pivot is a face pair: the sampled face row t gives the w residual rows R
    (w x w n), the face column s of its largest residual entry gives the w
    residual columns C (w m x w), and W = R[:, s] = C[t] is the w x w pivot
    block. The update C W^+ R is formed by complete-pivoting eliminations in
    W, skipping pivots below PIVOT_RTOL times the first, so one face sample adds
    up to w ranks; w = 1 is classical partially pivoted ACA. Ties go to the
    lowest index throughout.
    """

    def __init__(self, width, component, m, n, ids):
        self.w, self.component, self.ids = width, component, ids
        device = m.device
        self.m, self.n = m, n
        self.rows, self.cols = width * m, width * n
        self.M, self.N = int(m.max()), int(n.max())
        self.R, self.C = width * self.M, width * self.N
        count = len(m)
        self.face_valid = torch.arange(self.M, device=device)[None] < m[:, None]
        self.row_valid = torch.arange(self.R, device=device)[None] < self.rows[:, None]
        self.col_valid = torch.arange(self.C, device=device)[None] < self.cols[:, None]
        self.U = torch.zeros((count, self.R, INITIAL_RANK), device=device, dtype=torch.complex128)
        self.V = torch.zeros((count, INITIAL_RANK, self.C), device=device, dtype=torch.complex128)
        self.k = 0
        self.norm2 = torch.zeros(count, device=device, dtype=torch.float64)
        self.estimate = torch.zeros_like(self.norm2)
        self.used = torch.zeros((count, self.M), device=device, dtype=torch.bool)
        self.face = torch.zeros(count, device=device, dtype=torch.long)
        self.check = torch.zeros(count, device=device, dtype=torch.bool)
        self.active = torch.ones(count, device=device, dtype=torch.bool)
        self.rank = torch.zeros(count, device=device, dtype=torch.long)
        self.cap = torch.minimum(self.rows, self.cols)

    def _random_face(self, b, open_faces):
        """Pseudo-random open face per block from an integer hash (deterministic)."""
        hashed = ((torch.arange(self.M, device=open_faces.device)[None] + 1) * 2654435761
                  + (self.k + 1) * 40503 + (self.ids[b, None] + 1) * 97) % 2147483647
        return _first_argmax(hashed.double(), open_faces)

    def _rows(self, b):
        return self.face[b, None] * self.w + torch.arange(self.w, device=b.device)

    def step_rows(self, row_tiles, b):
        """Residual face rows of active blocks b; returns (blocks, face column, residual rows) to update."""
        k, count = self.k, len(b)
        if self.w == 3:
            # tile(t,s)[3a+j] lands at row a, column 3 l + j.
            values = row_tiles[..., :9].reshape(count, -1, 3, 3).permute(0, 2, 1, 3).reshape(count, 3, -1)
        else:
            values = row_tiles[..., 9][:, None]
        values = values * self.col_valid[b, None]
        rows = self._rows(b)
        residual = values - self.U[b[:, None], rows, :k] @ self.V[b, :k]
        norm = torch.linalg.vector_norm(residual, dim=(1, 2))
        faces = torch.sqrt(self.m[b].double())
        accept = self.check[b] & (faces * norm <= ACA_TOLERANCE * torch.sqrt(self.norm2[b]))
        self.estimate[b] = torch.where(accept, faces * norm, self.estimate[b])
        column = _first_argmax(residual.abs().amax(1), self.col_valid[b])
        # A zero residual face row adds no rank: exhaust it and check a random face next.
        zero = ~accept & (residual.abs().amax((1, 2)) == 0)
        self.used[b, self.face[b]] |= zero
        open_faces = self.face_valid[b] & ~self.used[b]
        self.active[b] &= ~accept & ~(zero & ~open_faces.any(1))
        self.face[b] = torch.where(zero, self._random_face(b, open_faces), self.face[b])
        self.check[b] = zero
        update = (~accept & ~zero).nonzero().flatten()
        return b[update], column[update] // self.w, residual[update]

    def step_columns(self, b, face, residual, column_tiles):
        """Rank <= w update C W^+ R from the sampled pivot face columns of blocks b."""
        k, count, w = self.k, len(b), self.w
        if k + w > self.U.shape[2]:
            self.U = torch.cat((self.U, torch.zeros_like(self.U)), 2)
            self.V = torch.cat((self.V, torch.zeros_like(self.V)), 1)
        if w == 3:
            # tile(t,s)[3a+j] lands at row 3 l + a, column j.
            values = column_tiles[..., :9].reshape(count, -1, 3)
        else:
            values = column_tiles[..., 9][:, :, None]
        values = values * self.row_valid[b, :, None]
        cols = face[:, None] * w + torch.arange(w, device=b.device)
        U, V = self.U[b, :, :k], self.V[b, :k]
        columns = values - U @ V.gather(2, cols[:, None, :].expand(-1, k, -1))
        # C W^+ R by complete pivoting inside the sampled cross: up to w rank-one
        # eliminations, each on the largest remaining pivot-block entry (lowest
        # index on ties), skipped below PIVOT_RTOL times the first pivot.
        rows = self._rows(b)
        local = torch.arange(count, device=b.device)
        us, vs, kept = [], [], []
        first = None
        for _ in range(w):
            block = residual.gather(2, cols[:, None, :].expand(-1, w, -1))
            index = _first_argmax(block.abs().reshape(count, w * w), torch.ones_like(block.real, dtype=torch.bool).reshape(count, -1))
            i, j = index // w, index % w
            pivot = block[local, i, j]
            first = pivot.abs() if first is None else first
            ok = (pivot != 0) & (pivot.abs() > PIVOT_RTOL * first)
            u = torch.where(ok[:, None], columns[local, :, j] / torch.where(ok, pivot, 1)[:, None], 0)
            v = torch.where(ok[:, None], residual[local, i], 0)
            residual = residual - u.gather(1, rows)[:, :, None] * v[:, None, :]
            columns = columns - u[:, :, None] * v.gather(1, cols)[:, None, :]
            us.append(u)
            vs.append(v)
            kept.append(ok)
        u, v, keep = torch.stack(us, 2), torch.stack(vs, 1), torch.stack(kept, 1)
        gram = (u.mH @ u) * (v @ v.mH).mT
        size2 = gram.sum((1, 2)).real.clamp_min(0)
        cross = 2 * ((U.mH @ u) * (v @ V.mH).mT).sum((1, 2)).real
        self.U[b, :, k:k + w] = u
        self.V[b, k:k + w] = v
        self.norm2[b] = (self.norm2[b] + cross + size2).clamp_min(0)
        # A face is exhausted only when its whole pivot block entered the update;
        # truncated directions stay in its residual rows and may pivot again.
        self.used[b, self.face[b]] |= keep.all(1)
        self.rank[b] += keep.sum(1)
        small = torch.sqrt(size2) <= ACA_TOLERANCE * torch.sqrt(self.norm2[b])
        open_faces = self.face_valid[b] & ~self.used[b]
        exhausted = (self.rank[b] >= self.cap[b]) | ~open_faces.any(1)
        self.active[b] &= ~exhausted
        self.estimate[b] = torch.where(exhausted, 0, self.estimate[b])
        following = _first_argmax(u.abs().amax(2).reshape(count, self.M, w).amax(2), open_faces)
        self.face[b] = torch.where(small, self._random_face(b, open_faces), following)
        self.check[b] = small
        self.k += w

    def recompress(self):
        """Gram/eigh truncation in block chunks; returns compact factors, ranks and errors.

        With U^H U = Q L Q^H, S = U V = P N for orthonormal P = U Q L^-1/2 and
        N = L^1/2 Q^H V, so S and N share singular values; N N^H = W s^2 W^H
        gives U' = P W_r s_r and V' = s_r^-1 W_r^H N. Batched Hermitian
        eigensolvers keep this on the GPU (batched small SVD/QR are orders
        slower there). Squaring limits singular values to ~1e-8 s_max relative,
        far below the ACA_TOLERANCE truncation, once every rank-one term is
        balanced; Gram directions below GRAM_RTOL of the largest (terms below
        ~1e-14 |S|, and the lockstep zero columns) are dropped. The factors are
        compacted block by block (U_b [w m, r_b] then V_b [r_b, w n], row-major).
        """
        k, count = self.k, len(self.rank)
        if k == 0:
            empty = torch.zeros(count, device=self.rank.device, dtype=torch.long)
            nothing = self.U.new_empty(0)
            return nothing, nothing, empty, self.estimate
        # Bounded factor workspace, and a bounded matrix count per batched
        # eigensolver call: its workspace grows with the batch (2 GiB at ~1200 cores).
        chunk = max(1, min(EIGH_BATCH, RECOMPRESS_BYTES // (16 * (self.R + self.C) * k)))
        us, vs, all_ranks, all_errors = [], [], [], []
        for first in range(0, count, chunk):
            part = slice(first, first + chunk)
            u, v, ranks, errors = self._recompress(self.U[part, :, :k], self.V[part, :k], self.estimate[part])
            index = torch.arange(u.shape[2], device=u.device)
            us.append(u[self.row_valid[part, :, None] & (index[None, None] < ranks[:, None, None])])
            vs.append(v[(index[None, :, None] < ranks[:, None, None]) & self.col_valid[part, None, :]])
            all_ranks.append(ranks)
            all_errors.append(errors)
        return torch.cat(us), torch.cat(vs), torch.cat(all_ranks), torch.cat(all_errors)

    @staticmethod
    def _recompress(U, V, estimate):
        # Balance every rank-one term to |u_i| = |v_i|: elimination pivots leave
        # columns of very different scale, which would ruin the Gram spectrum.
        nu, nv = torch.linalg.vector_norm(U, dim=1), torch.linalg.vector_norm(V, dim=2)
        live = (nu > 0) & (nv > 0)
        factor = torch.where(live, nv / torch.where(live, nu, 1), 1).sqrt()
        U = U * torch.where(live, factor, 0)[:, None, :].to(U.dtype)
        V = V * torch.where(live, 1 / factor, 0)[:, :, None].to(V.dtype)
        lam, q = torch.linalg.eigh(U.mH @ U)
        lam = lam.clamp_min(0)
        good = lam > GRAM_RTOL * lam[:, -1:]
        root = torch.where(good, lam, 1).sqrt()
        n = torch.where(good, root, 0)[:, :, None].to(V.dtype) * (q.mH @ V)
        del V
        mu, w = torch.linalg.eigh(n @ n.mH)
        sigma, w = mu.flip(1).clamp_min(0).sqrt(), w.flip(2)
        tails = torch.cat((sigma.square().flip(1).cumsum(1).flip(1).sqrt(), sigma.new_zeros(len(sigma), 1)), 1)
        total = torch.linalg.vector_norm(sigma, dim=1, keepdim=True)
        ranks = (tails > ACA_TOLERANCE * total).sum(1)
        errors = estimate + tails.gather(1, ranks[:, None])[:, 0]
        r = int(ranks.max())
        positive = sigma[:, :r] > 0
        q = q * torch.where(good, 1 / root, 0)[:, None, :].to(q.dtype)
        u = U @ (q @ (w[:, :, :r] * sigma[:, None, :r].to(w.dtype)))
        v = (w[:, :, :r].mH @ n) * torch.where(positive, 1 / torch.where(positive, sigma[:, :r], 1), 0)[:, :, None]
        return u, v, ranks, errors


def _compress_batch(sampler, clusters, ids):
    """Both channels of one ACA batch in lockstep; returns per-channel compact factors."""
    blocks = clusters.blocks[ids]
    row_start, m, col_start, n = blocks.unbind(1)
    device = blocks.device
    mx, nx = int(m.max()), int(n.max())
    row_faces = clusters.faces[(row_start[:, None] + torch.arange(mx, device=device)).clamp_max(len(clusters.faces) - 1)]
    col_faces = clusters.faces[(col_start[:, None] + torch.arange(nx, device=device)).clamp_max(len(clusters.faces) - 1)]
    row_ok = torch.arange(mx, device=device)[None] < m[:, None]
    col_ok = torch.arange(nx, device=device)[None] < n[:, None]
    channels = [_ChannelACA(w, c, m, n, ids) for w, c in CHANNELS]
    while bool(torch.stack([ch.active.any() for ch in channels]).any()):
        # Rows: one face row per active block and channel, one sampler call.
        requests, rows, cols = [], [], []
        for ch in channels:
            b = ch.active.nonzero().flatten()
            pair = col_ok[b].nonzero()
            rows.append(row_faces[b[pair[:, 0]], ch.face[b[pair[:, 0]]]])
            cols.append(col_faces[b[pair[:, 0]], pair[:, 1]])
            requests.append((b, pair))
        tiles = sampler(torch.cat(rows), torch.cat(cols))
        updates, start = [], 0
        for ch, (b, pair) in zip(channels, requests):
            dense = tiles.new_zeros((len(b), nx, 10))
            dense[pair[:, 0], pair[:, 1]] = tiles[start:start + len(pair)]
            start += len(pair)
            updates.append(ch.step_rows(dense, b) if len(b) else None)
        # Columns: one face column per updating block and channel.
        requests, rows, cols = [], [], []
        for ch, update in zip(channels, updates):
            if update is None or not len(update[0]):
                requests.append(None)
                continue
            b, face = update[0], update[1]
            pair = row_ok[b].nonzero()
            rows.append(row_faces[b[pair[:, 0]], pair[:, 1]])
            cols.append(col_faces[b[pair[:, 0]], face[pair[:, 0]]])
            requests.append(pair)
        if rows:
            tiles = sampler(torch.cat(rows), torch.cat(cols))
        start = 0
        for ch, update, pair in zip(channels, updates, requests):
            if pair is None:
                continue
            dense = tiles.new_zeros((len(update[0]), mx, 10))
            dense[pair[:, 0], pair[:, 1]] = tiles[start:start + len(pair)]
            start += len(pair)
            ch.step_columns(*update, dense)
    out = []
    for ch in channels:
        u, v, ranks, errors = ch.recompress()
        out.append((u, v, ranks, errors, ch.rank))
        del ch.U, ch.V
    return out


class HMatrix:
    """Compressed far operator of one frequency: low-rank channels plus exact leaf tiles."""

    def __init__(self, clusters, sampler):
        device = clusters.blocks.device
        self.clusters = clusters
        started = time.perf_counter()
        torch.cuda.synchronize(device)
        pairs = clusters.leaf_pairs
        self.leaf_tiles = (torch.cat([sampler(chunk[:, 0].contiguous(), chunk[:, 1].contiguous())
                                      for chunk in pairs.split(LEAF_CHUNK)])
                           if len(pairs) else torch.empty((0, 10), device=device, dtype=torch.complex128))
        count = len(clusters.blocks)
        self.channels = []
        parts = [[] for _ in CHANNELS]
        for ids in clusters.batches:
            for index, (u, v, *rest) in enumerate(_compress_batch(sampler, clusters, ids)):
                # Batch factors wait in host memory until every rank is known, so the device never
                # holds them next to the final factor array (twice the factor bytes at F-C scale);
                # the round trip copies values exactly.
                parts[index].append((ids, u.cpu(), v.cpu(), *rest))
        _, m, _, n = clusters.blocks.unbind(1)
        for (w, component), results in zip(CHANNELS, parts):
            ranks = torch.zeros(count, device=device, dtype=torch.long)
            errors = torch.zeros(count, device=device, dtype=torch.float64)
            aca = torch.zeros_like(ranks)
            for ids, _, _, r, e, a in results:
                ranks[ids], errors[ids], aca[ids] = r, e, a
            # Factor layout in block-id order: U_b then V_b.
            sizes_u, sizes_v = w * m * ranks, w * n * ranks
            ends = torch.cumsum(sizes_u + sizes_v, 0)
            layout = torch.stack((ends - sizes_u - sizes_v, ends - sizes_v), 1).contiguous()
            factors = torch.empty(int(ends[-1]) if count else 0, device=device, dtype=torch.complex128)
            for ids, u, v, r, _, _ in results:
                # Batch-compact factors are block-contiguous in batch order.
                su, sv = sizes_u[ids], sizes_v[ids]
                cu = torch.cumsum(su, 0) - su
                cv = torch.cumsum(sv, 0) - sv
                factors[_spread(layout[ids, 0], su, cu)] = u.to(device)
                factors[_spread(layout[ids, 1], sv, cv)] = v.to(device)
            rank_offsets = torch.cat((ranks.new_zeros(1), torch.cumsum(ranks, 0)))
            rank_block = torch.repeat_interleave(torch.arange(count, device=device), ranks)
            self.channels.append(dict(width=w, component=component, factors=factors, layout=layout,
                                      rank_offsets=rank_offsets, rank_block=rank_block, ranks=ranks,
                                      errors=errors, aca_ranks=aca))
        torch.cuda.synchronize(device)
        self.seconds = time.perf_counter() - started

    def apply(self, coefficients):
        """Far moments [F,4,P] of face coefficients [F,4,P]."""
        op = torch.ops.witwin_maxwell_mom_layered
        c = self.clusters
        x = coefficients.contiguous()
        out = op.hmatrix_leaf(self.leaf_tiles, c.leaf_pairs, c.forward_offsets, c.forward, c.reverse_offsets,
                              c.reverse, x)
        for ch in self.channels:
            op.hmatrix_lowrank(ch["factors"], c.blocks, ch["layout"], ch["rank_offsets"], ch["rank_block"],
                               c.faces, c.entry_offsets, c.entries, x, out, ch["width"], ch["component"])
        return out

    def error_bound(self, coefficients, omega0):
        """Per channel (worst case, quadrature) of the block terms 2 e_b |c_T|_F |c_S|_F with the prefactor.

        The worst case is their sum (triangle inequality, |dF_b|_2 <= e_b); the
        quadrature estimate combines them as independent block errors.
        """
        c = self.clusters
        row_start, m, col_start, n = c.blocks.unbind(1)
        parts = []
        for ch, factor in zip(self.channels, (MU0, 1 / (4 * EPS0 * omega0 ** 2))):
            comp = slice(ch["component"], ch["component"] + ch["width"])
            norms = coefficients[c.faces][:, comp].abs().square().sum((1, 2))
            prefix = torch.cat((norms.new_zeros(1), torch.cumsum(norms, 0)))
            rows = (prefix[row_start + m] - prefix[row_start]).clamp_min(0).sqrt()
            cols = (prefix[col_start + n] - prefix[col_start]).clamp_min(0).sqrt()
            terms = 2 * ch["errors"] * rows * cols
            parts.append((factor * float(terms.sum()), factor * float(torch.linalg.vector_norm(terms))))
        return parts

    @property
    def nbytes(self):
        return (self.leaf_tiles.nbytes + sum(ch["factors"].nbytes + ch["layout"].nbytes + ch["rank_offsets"].nbytes
                                             + ch["rank_block"].nbytes for ch in self.channels))

    def stats(self):
        c = self.clusters
        _, m, _, n = c.blocks.unbind(1)
        out = dict(hmatrix_seconds=self.seconds, leaf_tile_bytes=self.leaf_tiles.nbytes,
                   hmatrix_bytes=self.nbytes + c.stats["leaf_bytes"] - self.leaf_tiles.nbytes + c.stats["entry_bytes"])
        for name, ch in zip(("vector", "scalar"), self.channels):
            w = ch["width"]
            ranks = ch["ranks"]
            out[f"{name}_factor_bytes"] = ch["factors"].nbytes
            out[f"{name}_dense_equivalent_bytes"] = int((16 * w * w * m * n).sum())
            for kind, mask in (("same_plane", ~c.cross), ("cross_plane", c.cross)):
                r = ranks[mask]
                out[f"{name}_rank_{kind}"] = (dict(blocks=len(r), max=int(r.max()), mean=float(r.double().mean()),
                                                   p90=float(r.double().quantile(.9)),
                                                   aca_max=int(ch["aca_ranks"][mask].max()))
                                              if len(r) else dict(blocks=0))
            out[f"{name}_rank_histogram"] = torch.bincount(ranks).tolist()
        return out


def _spread(starts, sizes, compact_starts):
    """Destination indices of block-contiguous segments (start_b, size_b) in compact order."""
    block = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), sizes)
    local = torch.arange(int(sizes.sum()), device=sizes.device) - compact_starts[block]
    return starts[block] + local


class LayeredHMatrixOperator(LayeredMatrixFreeOperator):
    """matrix_free operator whose far action is the frequency's compressed H-matrix."""

    def __init__(self, prepared, structure, clusters, frequency_index, solver, preconditioner=None):
        super().__init__(prepared, structure, frequency_index, solver, preconditioner)
        self.sampler = TileSampler(self.green, structure, self.table)
        self.hmatrix = HMatrix(clusters, self.sampler)

    def far_moments(self, coefficients):
        return self.hmatrix.apply(coefficients)

    def face_coefficients(self, x):
        """The far action's face coefficients [F,4,P] of basis columns x [N,P]."""
        op = torch.ops.witwin_maxwell_mom_layered
        lt, s = self.prepared.loop_tree, self.structure
        supports, slots, slopes, face_edges, support_slopes, areas2 = self.supports
        z = self.scale[:, None] * x
        u = op.sparse_product((z * self.basis_scale[:, None])[None].contiguous(), lt.sparse_q_rows, True)[0]
        coefficients = op.face_coefficients(u.contiguous(), supports, slots, slopes, face_edges, support_slopes)
        coefficients[:, 3] = op.sparse_product(z[None, self.loops:].contiguous(), s.charge_rows, True)[0] * (
            4 / areas2[:, None])
        return coefficients

    def compression_bound(self, x, b):
        """Diagnostics of the compression error in |dY|_F/|Y|_F for the solution x.

        Returns (worst case, quadrature estimate), each summed over the channels.
        """
        admittance = float(torch.linalg.matrix_norm(b.mT @ x))
        parts = self.hmatrix.error_bound(self.face_coefficients(x), self.prepared.loop_tree.omega0)
        return sum(p[0] for p in parts) / admittance, sum(p[1] for p in parts) / admittance

    def exact_check(self, x, b):
        """First-order |dY|_F/|Y|_F of the compression from one exact far action.

        dY = -X^T (A~ - A) X, and the far part of x^T A y is
        -i mu0 c_v^T F_v c_v' + i/(4 eps0 w0^2) c_s^T F_s c_s' in face coefficients.
        """
        c = self.face_coefficients(x)
        difference = self.far_moments(c) - LayeredMatrixFreeOperator.far_moments(self, c)
        omega0 = self.prepared.loop_tree.omega0
        dy = ((-1j * MU0) * torch.einsum("fkp,fkq->pq", c[:, :3], difference[:, :3])
              + (1j / (4 * EPS0 * omega0 ** 2)) * (c[:, 3].mT @ difference[:, 3]))
        return float(torch.linalg.matrix_norm(dy) / torch.linalg.matrix_norm(b.mT @ x))

    def persistence_receipt(self):
        return {**super().persistence_receipt(), "version": "layered_hmatrix_operator_v1",
                "operator_backend": "hmatrix", "eta": ETA, "leaf_size": LEAF_SIZE,
                "aca_tolerance": ACA_TOLERANCE}

    def operator_stats(self):
        return {**super().operator_stats(), "operator_backend": "hmatrix", **self.hmatrix.stats(),
                "sampler_calls": self.sampler.calls, "sampled_face_pairs": self.sampler.pairs}


def solve_hmatrix(prepared):
    """Per-frequency near LU + compressed far operator + block GMRES; as solve_matrix_free.

    With metadata hmatrix_exact_check (LayeredSurfaceConfig), every solve is
    verified by one exact far action and refused above COMPRESSION_TOLERANCE.
    """
    exact_check = prepared.metadata.get("hmatrix_exact_check", False)
    from .body_preconditioner import body_preconditioner_structure
    from .network import recover_scaled_chunk
    device = prepared.excitation.device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    structure = build_near_structure(prepared)
    wide = body_preconditioner_structure(prepared, structure)
    clusters = build_cluster_blocks(prepared, structure)
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
        operator = LayeredHMatrixOperator(prepared, structure, clusters, index, solver, wide)
        rhs = operator.drive()
        # R6-3 (D3): warm start from the previous frequency (matrix_free.solve_matrix_free).
        recovery = operator.recovery()[:, None]
        solution, log = block_gmres(operator, rhs, x0=None if coefficients is None else coefficients / recovery)
        coefficients = solution * recovery
        worst, quadrature = operator.compression_bound(solution, rhs)
        exact = None
        if exact_check:
            exact = operator.exact_check(solution, rhs)
            if not exact <= COMPRESSION_TOLERANCE:
                raise RuntimeError(f"Layered H-matrix compression error {exact:.3g} exceeds {COMPRESSION_TOLERANCE:.3g} "
                                   f"at {operator.frequency:.6g} Hz (exact far-action check)")
        y, x = recover_scaled_chunk(prepared, solution[None], rhs[None], operator.recovery()[None],
                                    frequency_start=index)
        admittance[index], currents[index] = y[0], x[0]
        torch.cuda.synchronize(device)
        log.update(operator.operator_stats(), compression_bound_worst_case=worst, compression_estimate_quadrature=quadrature,
                   compression_exact_check=exact, seconds=time.perf_counter() - started)
        logs.append(log)
        del operator
    solver.free()
    return admittance, currents, dict(structure={**structure.stats, **clusters.stats}, frequencies=logs)
