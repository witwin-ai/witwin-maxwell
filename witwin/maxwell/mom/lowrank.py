"""Verified complete or adaptive cross approximation for fixed CUDA blocks.

Every admitted block checks all residual entries in bounded row batches.
Adaptive pivots avoid scanning at every rank, but the verification still costs
quadratic work. This is a balanced block Frobenius test, not a global Riesz norm.
"""
from dataclasses import dataclass, field
import math
import torch


class CompressionFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class LowRankBlock:
    left: torch.Tensor
    right: torch.Tensor
    receipt: dict
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_versions', tuple((id(t), t._version) for t in (self.left, self.right)))

    def assert_current(self):
        if tuple((id(t), t._version) for t in (self.left, self.right)) != self._versions:
            raise ValueError('Compressed block factors changed after verification')

    def apply(self, x):
        self.assert_current()
        return (x @ self.right.mT) @ self.left.mT

    def adjoint(self, y):
        self.assert_current()
        return ((y.conj() @ self.left) @ self.right).conj()


def streamed_cross_approximation(sample_rows, sample_column, shape, *, device,
        dtype=torch.complex128, max_rank=32, rtol=1e-8, atol=0., row_batch=16,
        row_scale=None, column_scale=None, pivoting='complete'):
    """Approximate A = left @ right, then verify every balanced residual entry.

    Callbacks supply rows [batch,n] and a column [m] in the physical ordering.
    Positive diagonal scales define diag(row_scale) A diag(column_scale).
    Returned factors act in physical units. Sampled kernel/material gradients
    are deliberately rejected; current-vector derivatives remain native Torch.
    Adaptive pivots sample one row and column per update, using an update-size
    estimate only to schedule a full residual scan. Failed verification restarts
    from its largest residual. Every returned block passes the same full check.
    The final scan still costs O(m*n), so this is not linear-time setup.
    Rank exhaustion fails with the quantitative receipt and never falls back
    to an unverified approximation or a dense block.
    """
    if pivoting not in ('complete', 'adaptive'):
        raise ValueError('Pivoting must be complete or adaptive')
    device = torch.device(device)
    if device.type != 'cuda' or dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Cross approximation requires complex CUDA data')
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)):
        raise ValueError('Block shape must contain two positive integers')
    for name, value in (('max_rank', max_rank), ('row_batch', row_batch)):
        if type(value) is not int or value < 1:
            raise ValueError(name+' must be a positive integer')
    for name, value in (('rtol', rtol), ('atol', atol)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(name+' must be finite and nonnegative')
    m, n = shape
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    # Resolve a bare CUDA device to its concrete index before callback checks.
    anchor = torch.empty(0, device=device, dtype=dtype)
    device = anchor.device
    def scale(value, size):
        if value is None:
            return torch.ones(size, device=device, dtype=real_dtype)
        if (not isinstance(value, torch.Tensor) or value.shape != (size,)
                or value.device != device or value.dtype != real_dtype or value.requires_grad
                or not bool(torch.isfinite(value).all() & (value > 0).all())):
            raise ValueError('Block scales require positive finite fixed real CUDA vectors')
        return value.clone()
    rs, cs = scale(row_scale, m), scale(column_scale, n)
    def sampled(value, expected):
        if (not isinstance(value, torch.Tensor) or value.shape != expected
                or value.device != device or value.dtype != dtype or value.requires_grad
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Block sampler must return finite fixed complex CUDA tensors of the declared shape')
        return value
    left = torch.empty((m, 0), device=device, dtype=dtype)
    right = torch.empty((0, n), device=device, dtype=dtype)
    batch = min(row_batch, max(1, m//2))
    ceiling = min(max_rank, m, n)
    scans, row_calls, column_calls = 0, 0, 0
    matrix_norm = None
    next_row = 0
    used_rows = torch.zeros(m, device=device, dtype=torch.bool)
    require_scan = pivoting == 'complete'
    recompressed = False
    while True:
        rank = left.shape[1]
        scan_now = require_scan or rank == ceiling or pivoting == 'complete'
        if not scan_now:
            ids = torch.tensor([next_row], device=device, dtype=torch.int64)
            best_row = sampled(sample_rows(ids), (1, n))[0]*rs[next_row]*cs-left[next_row]@right
            row_calls += 1
            if not bool(torch.isfinite(best_row).all()):
                raise ValueError('Adaptive residual row overflowed native precision')
            pivot_row = next_row
            pivot_column = int(best_row.abs().argmax())
            # A zero or numerically exhausted row cannot establish convergence.
            # Check every entry, and restart from its largest residual pivot.
            scan_now = not bool(best_row[pivot_column].abs() > 0)
        if scan_now:
            residual_norm = torch.zeros((), device=device, dtype=real_dtype)
            original_norm = torch.zeros_like(residual_norm)
            largest = -1.
            pivot_row = pivot_column = None
            best_row = None
            for start in range(0, m, batch):
                rows = torch.arange(start, min(start+batch, m), device=device)
                original = sampled(sample_rows(rows), (len(rows), n))*rs[rows, None]*cs[None, :]
                row_calls += 1
                residual = original-left[rows]@right
                if not bool(torch.isfinite(residual).all()):
                    raise ValueError('Balanced block residual overflowed native precision')
                residual_norm = torch.hypot(residual_norm, torch.linalg.vector_norm(residual))
                if matrix_norm is None:
                    original_norm = torch.hypot(original_norm, torch.linalg.vector_norm(original))
                flat_index = int(residual.abs().argmax())
                local_row, local_col = divmod(flat_index, n)
                magnitude = float(residual[local_row, local_col].abs())
                if magnitude > largest:
                    largest = magnitude
                    pivot_row, pivot_column = start+local_row, local_col
                    best_row = residual[local_row].clone()
            scans += 1
            if matrix_norm is None:
                matrix_norm = float(original_norm)
            error = float(residual_norm)
            threshold = max(atol, rtol*matrix_norm)
            receipt = {'version': ('streamed_complete_pivot_cross_v1' if pivoting == 'complete' else 'verified_adaptive_cross_v1'), 'shape': list(shape),
                'rank': rank, 'max_rank': max_rank, 'rtol': rtol, 'atol': atol,
                'balanced_frobenius_norm': matrix_norm, 'balanced_residual_norm': error,
                'threshold': threshold, 'converged': math.isfinite(error) and math.isfinite(threshold) and error <= threshold,
                'residual_scans': scans, 'row_calls': row_calls, 'column_calls': column_calls,
                'maximum_sample_rows': batch, 'maximum_sample_elements': batch*n,
                'factor_bytes': (m+n)*rank*anchor.element_size(),
                'dense_bytes': m*n*anchor.element_size(),
                'storage_reduced': (m+n)*rank < m*n,
                'full_residual_verified': True, 'setup_complexity_qualified': False,
                'factor_recompressed': recompressed,
                'global_riesz_certified': False, 'physical_qualification': False}
            if not math.isfinite(error) or not math.isfinite(matrix_norm) or not math.isfinite(threshold):
                raise CompressionFailure('Block norm exceeds native precision', receipt)
            if error <= threshold:
                if pivoting == 'adaptive' and rank and not recompressed:
                    # QR and a small-core SVD remove redundant adaptive pivots.
                    # A conservative triangle budget proposes a truncation;
                    # the next full scan must verify the actual new factors.
                    ql, rl = torch.linalg.qr(left, mode='reduced')
                    qr, rr = torch.linalg.qr(right.mH, mode='reduced')
                    u, singular, vh = torch.linalg.svd(rl@rr.mH, full_matrices=False)
                    tails = torch.cat((singular.square().flip(0).cumsum(0).flip(0).sqrt(), singular.new_zeros(1)))
                    keep = int((tails <= .5*max(0., threshold-error)).nonzero()[0])
                    recompressed = True
                    if keep < rank:
                        left = ql@(u[:, :keep]*singular[:keep])
                        right = vh[:keep]@qr.mH
                        require_scan = True
                        continue
                    receipt['factor_recompressed'] = True
                if not receipt['storage_reduced']:
                    raise CompressionFailure('Verified block factors do not reduce storage; subdivide or store an explicit exact leaf', receipt)
                return LowRankBlock(left/rs[:, None], right/cs[None, :], receipt)
            if rank == ceiling:
                raise CompressionFailure(f'Block compression exhausted rank {rank}: residual {error:.6g} > {threshold:.6g}', receipt)
        column = sampled(sample_column(pivot_column), (m,))*rs*cs[pivot_column]
        column_calls += 1
        column = column-left@right[:, pivot_column]
        pivot = best_row[pivot_column]
        if not bool(torch.isfinite(column).all()) or not bool(pivot.abs() > 0):
            raise ValueError('Block pivot is zero or nonfinite before convergence')
        update_left = column/pivot
        left = torch.cat((left, update_left[:, None]), dim=1)
        right = torch.cat((right, best_row[None, :]), dim=0)
        if pivoting == 'adaptive':
            used_rows[pivot_row] = True
            candidates = column.abs().masked_fill(used_rows, 0.)
            next_row = int(candidates.argmax())
            # This estimate only schedules a complete residual verification;
            # it never admits a factorization, even when it underestimates error.
            update_norm = torch.linalg.vector_norm(update_left)*torch.linalg.vector_norm(best_row)
            approximation_norm = ((left.mH@left)*(right@right.mH).mT).sum().real.clamp_min(0).sqrt()
            if not bool(torch.isfinite(update_norm) & torch.isfinite(approximation_norm)):
                raise ValueError('Adaptive factor norm overflowed native precision')
            require_scan = (not bool(candidates[next_row] > 0)
                            or bool(update_norm <= .1*max(atol, rtol*float(approximation_norm))))
    raise AssertionError('Unreachable compression state')
