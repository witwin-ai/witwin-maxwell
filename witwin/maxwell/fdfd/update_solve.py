"""Certified CUDA small-system solves for nondifferentiable design proposals."""
import torch


def solve_update(matrix, product, rhs, tolerance):
    n=matrix.shape[-1]
    if n<=32 and matrix.is_contiguous() and product.is_contiguous() and rhs.is_contiguous():
        from .update_cuda import solve_small
        solution,residual,inverse,safe=solve_small(matrix,product,rhs,tolerance)
        if safe:
            return solution,residual,inverse
    identity=torch.eye(n,device=matrix.device,dtype=matrix.dtype)
    solved,info=torch.linalg.solve_ex(matrix,torch.cat((rhs[:,None],identity),dim=1))
    solution,inverse=solved[:,0],solved[:,1:]
    # If ||I-SX|| < 1 then sigma_min(S) >= (1-||I-SX||)/||X||.
    # Frobenius norms conservatively bound the corresponding spectral norms.
    defect=torch.linalg.matrix_norm(identity-matrix@inverse)
    lower=(1-defect)/torch.linalg.matrix_norm(inverse)
    threshold=16*torch.finfo(rhs.real.dtype).eps*n*torch.linalg.matrix_norm(product).clamp_min(1)
    residual=torch.linalg.vector_norm(matrix@solution-rhs)/torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.real.dtype).tiny)
    safe=(info==0)&torch.isfinite(solved).all()&torch.isfinite(lower)&(lower>threshold)
    valid=(info==0)&torch.isfinite(residual)&(residual<=tolerance)
    safe_host,valid_host,residual_host=torch.stack((safe,valid,residual)).detach().cpu().tolist()
    if not safe_host:
        # Keep the established singularity decision when the cheap bound is inconclusive.
        singular=torch.linalg.svdvals(matrix)
        if not bool(torch.isfinite(singular).all()) or bool(singular[-1]<=threshold):
            raise RuntimeError('Candidate update is numerically singular; accepted state was not changed.')
    if not valid_host:
        raise RuntimeError(f'Local response residual {residual_host} exceeds {tolerance}.')
    return solution,residual_host,inverse
