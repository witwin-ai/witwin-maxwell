"""Surface-admittance body blocks in the iterative layered backends.

conductor_model="surface_admittance" adds one dense block W_b per lossy body,
coupling only the exterior RWGs of that body (admittance.py). The dense
backend folds it before the loop/tree congruence. The matrix_free and hmatrix
backends add it to their near operator (matrix_free.py):

    A = M_mpie + sum_b D S Q_b^T (W_b/w) Q_b S D + Far,

which is the dense system with the exact near/far split of the exterior MPIE;
Far is unchanged. W enters Z in the dense backend's units, W * i/(mu0 w)
folded into V' and multiplied by the native -i mu0, with mu0 = 4e-7 pi as in
the dense fold.

Basis blocks. Q_b = Q[ids_b, :] keeps only the c_b basis columns that touch
the body's exterior RWGs, so Q_b^T W_b Q_b is a dense c_b x c_b block. It is
formed with two dense real-by-complex products per frequency (a symbolic
sparse product of a dense block would expand about nnz(Q row)^2 entries per
W entry).

Storage. The dense basis block of this module exists only transiently, one
body at a time (body_compress.body_operators): the action applies W_b as an
H-matrix on the exterior RWGs (body_compress.py), and the preconditioner adds
the block only on the near pattern's keys (body_block_lu.py).

Symmetry. Low-regime blocks are symmetric; DSA blocks are not (A1: 3-5% of
|W|), so A is complex symmetric only up to W - W^T, and A^T transposes the
dense blocks.
"""

from dataclasses import dataclass
import math

import torch

from .admittance import MU0 as ADMITTANCE_MU0


@dataclass
class BodyBasisBlock:
    """Frequency-independent basis pattern of one body block."""

    owner: int
    ids: torch.Tensor        # [n_b] exterior RWG ids (ascending)
    q_rows: torch.Tensor     # Q_b entries: local row, local column, value
    q_cols: torch.Tensor
    q_values: torch.Tensor
    columns: torch.Tensor    # [c_b] basis columns touching the body (ascending)

    @property
    def keys_count(self):
        return len(self.columns) ** 2


def body_basis_blocks(admittance, loop_tree):
    """One BodyBasisBlock per admittance body, from the CSC of Q^T (edge rows)."""
    offsets, indices, values = loop_tree.sparse_q_rows
    blocks = []
    for body in admittance.bodies:
        ids = body.exterior_ids
        counts = offsets[ids + 1] - offsets[ids]
        local = torch.repeat_interleave(torch.arange(len(ids), device=ids.device), counts)
        starts = torch.cumsum(counts, 0) - counts
        entries = offsets[ids][local] + torch.arange(len(local), device=ids.device) - starts[local]
        columns, inverse = torch.unique(indices[entries], return_inverse=True)
        blocks.append(BodyBasisBlock(body.owner, ids, local, inverse, values[entries], columns))
    return blocks


def body_values(blocks, body_blocks, frequency, basis_scale):
    """Dense basis blocks -i mu0 S Q_b^T (W_b i/(mu0 w)) Q_b S [c_b, c_b] per body .

    body_blocks is SurfaceAdmittance.body_blocks(frequency): {owner: (ids, W_ext)}.
    """
    from .matrix_free import MU0
    fold = 1j / (ADMITTANCE_MU0 * 2 * math.pi * frequency)
    out = []
    for b in blocks:
        ids, w = body_blocks[b.owner]
        q = torch.zeros((len(b.ids), len(b.columns)), device=w.device, dtype=torch.float64)
        q[b.q_rows, b.q_cols] = b.q_values
        w = w * fold
        wq = torch.complex(w.real @ q, w.imag @ q)
        del w
        block = torch.complex(q.mT @ wq.real, q.mT @ wq.imag)
        del wq, q
        scale = basis_scale[b.columns]
        block *= (-1j * MU0) * scale[:, None] * scale[None]
        out.append(block)
    return out


def body_stats(blocks):
    return dict(admittance_bodies=len(blocks),
                admittance_basis_columns=tuple(len(b.columns) for b in blocks),
                admittance_block_nnz=sum(b.keys_count for b in blocks))
