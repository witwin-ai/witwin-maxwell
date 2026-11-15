"""Sparse near LU preconditioner for surface_admittance on the iterative backends.

conductor_model="surface_admittance" adds one block B_b per body
(admittance_near.py), applied in the action as a compressed H-matrix of W_b
(body_compress.py). The right preconditioner M is the D-scaled near pattern
(or the D2-widened pattern) with B_b added only on the keys that pattern
already holds:

    M = P_pattern(near) + sum_b P_pattern(D B_b D),

factored by the sparse near LU (NearFieldLU) with the fill of the body-free
pattern. The rest of B_b stays in the exact action; GMRES absorbs it.

History. G2 folded every B_b entry into the sparse LU (34M LU nnz on F-B
80/20, about 6x dense memory). R4-1 factored the body-coupled column set D
densely with a Schur step: exact M^-1 of the full blocks, but |D| = N once
bodies span the layout, so 32 |D|^2 bytes (N^2). Restricting B_b to the
pattern keeps M sparse: on the 40/4 strip the B_b mass outside the D2
pattern is 2e-4 (window) and 2e-5 (DSA) of |B_b|_F, and the block GMRES
steps stay those of the exact dense-set inverse (6/7/9 at 1e6/2e8/1e10 Hz).

Values are gathered and added in body order (no atomics); the LU is cuDSS
in deterministic mode, so factors and solves are torch.equal on repeat.
"""

import torch

from .matrix_free import NearFieldLU


class PatternBodyLU(NearFieldLU):
    """Sparse LU of the near pattern plus every body block restricted to the pattern keys."""

    def __init__(self, structure, bodies, device):
        super().__init__(structure, device)
        n = len(structure.offsets) - 1
        rows = torch.repeat_interleave(torch.arange(n, device=device), structure.offsets.diff())
        cols = structure.cols
        self.body_keys = []
        for b in bodies:
            inside = torch.zeros(n, device=device, dtype=torch.bool)
            inside[b.columns] = True
            local = torch.full((n,), -1, device=device, dtype=torch.int64)
            local[b.columns] = torch.arange(len(b.columns), device=device)
            positions = (inside[rows] & inside[cols]).nonzero().flatten()
            # Pattern positions, their rows and columns, and their row-major keys into the dense [c_b, c_b] block.
            r, c = rows[positions], cols[positions]
            self.body_keys.append((positions, r, c, local[r] * len(b.columns) + local[c]))
        del rows
        self.stats["body_pattern_entries"] = tuple(len(k[0]) for k in self.body_keys)

    def body_entries(self, index, block):
        """Entries of body ``index``'s dense unscaled block on the pattern keys."""
        return block.reshape(-1)[self.body_keys[index][3]]

    def factor(self, values, bodies=(), scale=None):
        """values: D-scaled pattern values; bodies: unscaled pattern entries per body (body_entries)."""
        if bodies:
            values = values.clone()
            for (positions, rows, cols, _), entries in zip(self.body_keys, bodies):
                values[positions] += entries * (scale[rows] * scale[cols])
            # The per-body DSA transients (several GB at F-B 40/10) stay in torch's cache, which the
            # sparse LU's own pool cannot reuse: return them before the factorization allocates.
            torch.cuda.empty_cache()
        super().factor(values)


def near_preconditioner(structure, wide, device):
    """The right preconditioner of the iterative backends: the sparse near LU, with body blocks on its keys."""
    pattern = structure if wide is None else wide
    if structure.bodies is None:
        return NearFieldLU(pattern, device)
    return PatternBodyLU(pattern, structure.bodies, device)
