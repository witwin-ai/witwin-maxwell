"""Small batched CUDA eigenspaces with residual-checked native fallback."""
from __future__ import annotations
import torch


def batched_eig4(matrix):
    """Diagonalize fixed 4x4 samples using shifted QR on their CUDA batch.

    Hessenberg QR deflation determines eigenvalues; adjugate polynomial columns
    and residual-triggered SVD repair determine right eigenvectors. Unconverged, repeated, ill-conditioned or inaccurate
    candidates use torch.linalg.eig on only that subset, on the original device.
    Differentiable inputs retain the native eigensolver's autograd behavior.
    No constitutive damping, eigenvalue perturbation, or CPU tensor path is used.
    """
    if (not isinstance(matrix, torch.Tensor) or matrix.device.type != 'cuda'
            or matrix.ndim != 3 or matrix.shape[-2:] != (4, 4)
            or matrix.dtype not in (torch.complex64, torch.complex128)
            or not bool(torch.isfinite(matrix).all())):
        raise ValueError('Batched eig4 requires finite complex CUDA [N,4,4] matrices')
    count = len(matrix)
    receipt = {'version': 'cuda_shifted_qr_adjugate_checked_v1', 'matrix_count': count,
               'fallback_count': 0, 'qr_iterations': [], 'residual_certifies_eigenvalue_error': False}
    if matrix.requires_grad or count == 0:
        roots, vectors = torch.linalg.eig(matrix)
        receipt.update(fallback_count=count, native_reason='autograd' if count else 'empty')
        return roots, vectors, receipt
    eps = torch.finfo(matrix.real.dtype).eps
    eye = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
    scale = matrix.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(matrix.real.dtype).tiny)
    normalized = matrix/scale[:, None, None]
    schur = normalized.clone()
    # Unitary Householder similarity gives upper Hessenberg form before QR;
    # no matrix balancing, physical damping or change of eigenvalues is used.
    for column in (0, 1):
        tail = schur[:, column+1:, column].clone()
        length = torch.linalg.vector_norm(tail, dim=-1)
        leading = tail[:, 0]
        phase = torch.where(leading.abs() > 0, leading/leading.abs().clamp_min(torch.finfo(matrix.real.dtype).tiny),
                            torch.ones_like(leading))
        tail[:, 0] += phase*length
        direction = tail/torch.linalg.vector_norm(tail, dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.real.dtype).tiny)
        schur[:, column+1:, :] -= 2*direction[:, :, None]*(direction.conj()[:, None, :]@schur[:, column+1:, :])
        schur[:, :, column+1:] -= 2*(schur[:, :, column+1:]@direction[:, :, None])*direction.conj()[:, None, :]
        schur[:, column+2:, column] = 0
    failed = torch.zeros(count, device=matrix.device, dtype=torch.bool)
    for size in (4, 3):
        identity = eye[:size, :size]
        iterations = 0
        for _ in range(64):
            active = torch.linalg.vector_norm(schur[:, size-1, :size-1], dim=-1) > 32*eps
            if not bool(active.any()):
                break
            a, b = schur[:, size-2, size-2], schur[:, size-2, size-1]
            c, d = schur[:, size-1, size-2], schur[:, size-1, size-1]
            middle = (a+d)/2
            discriminant = torch.sqrt(((a-d)/2)**2+b*c)
            left, right = middle+discriminant, middle-discriminant
            shift = torch.where((left-d).abs() < (right-d).abs(), left, right)
            q, _ = torch.linalg.qr(schur[:, :size, :size]-shift[:, None, None]*identity)
            q = torch.where(active[:, None, None], q, identity)
            updated = schur.clone()
            updated[:, :size, :size] = q.mH@schur[:, :size, :size]@q
            updated[:, :size, size:] = q.mH@schur[:, :size, size:]
            updated[:, size:, :size] = schur[:, size:, :size]@q
            schur = updated
            iterations += 1
        receipt['qr_iterations'].append(iterations)
        failed |= torch.linalg.vector_norm(schur[:, size-1, :size-1], dim=-1) > 32*eps
        schur[:, size-1, :size-1] = 0
    roots = schur.diagonal(dim1=-2, dim2=-1).clone()
    a, b, c, d = schur[:, 0, 0], schur[:, 0, 1], schur[:, 1, 0], schur[:, 1, 1]
    middle, discriminant = (a+d)/2, torch.sqrt(((a-d)/2)**2+b*c)
    roots[:, 0], roots[:, 1] = middle+discriminant, middle-discriminant
    polynomial = eye.expand(count, 4, 4, 4).clone()
    targets = torch.arange(4, device=matrix.device)
    for index in range(4):
        factor = normalized[:, None]-roots[:, index, None, None, None]*eye
        factor = torch.where((targets == index)[None, :, None, None], eye, factor)
        polynomial = polynomial@factor
    columns = torch.linalg.vector_norm(polynomial, dim=-2).argmax(-1)
    vectors = polynomial.gather(-1, columns[:, :, None, None].expand(-1, -1, 4, 1)).squeeze(-1).mT
    vectors = vectors/torch.linalg.vector_norm(vectors, dim=-2, keepdim=True)
    finite = torch.isfinite(vectors).all(dim=(-2, -1)) & torch.isfinite(roots).all(-1)
    vectors = torch.where(finite[:, None, None], vectors, eye)
    pair_residual = torch.linalg.vector_norm(normalized@vectors-vectors*roots[:, None, :], dim=-2)
    gaps = (roots[:, :, None]-roots[:, None, :]).abs()
    separation = gaps.masked_fill(torch.eye(4, device=matrix.device, dtype=torch.bool), float('inf')).amin(dim=(-2, -1))
    # Root separation already mandates native fallback; repairing vectors
    # cannot change that decision and those vectors would be overwritten.
    separated = separation >= eps**.5
    repair = (pair_residual > 512*eps) & finite[:, None] & ~failed[:, None] & separated[:, None]
    receipt['svd_vector_repairs'] = int(repair.sum())
    if receipt['svd_vector_repairs']:
        owner, column = torch.nonzero(repair, as_tuple=True)
        shifted = normalized[owner]-roots[owner, column, None, None]*eye
        try:
            _, _, adjoint = torch.linalg.svd(shifted, full_matrices=False, driver='gesvdj')
            vectors[owner, :, column] = adjoint[:, -1, :].conj()
        except torch.linalg.LinAlgError:
            failed[owner] = True
            receipt['svd_repair_failed'] = True
    finite = torch.isfinite(vectors).all(dim=(-2, -1)) & torch.isfinite(roots).all(-1)
    residual = torch.linalg.vector_norm(normalized@vectors-vectors*roots[:, None, :], dim=-2).amax(-1)
    singular_values = torch.linalg.svdvals(vectors)
    condition = singular_values[:, 0]/singular_values[:, -1].clamp_min(torch.finfo(matrix.real.dtype).tiny)
    causes = {'qr_not_deflated': failed.clone(), 'nonfinite': ~finite,
              'eigenpair_residual': residual > 512*eps, 'small_root_separation': separation < eps**.5,
              'eigenvector_condition': condition > eps**(-.25)}
    receipt['fallback_causes_nonexclusive'] = {name: int(mask.sum()) for name, mask in causes.items()}
    for mask in causes.values():
        failed |= mask
    roots = roots*scale[:, None]
    receipt['fallback_count'] = int(failed.sum())
    if receipt['fallback_count']:
        exact_roots, exact_vectors = torch.linalg.eig(matrix[failed])
        roots[failed], vectors[failed] = exact_roots, exact_vectors
    return roots, vectors, receipt
