"""Checked matched/proportional SPD modes and Weyl branch quadrature."""
import math
import torch


def proportional_spd_spatial_green(displacements,medium,*,regular_part=False):
    """Exact coordinate transform for fixed real SPD mu = s epsilon.

    Return None outside strict equality, without projecting constitutive data.
    Point kernels have no determinant multiplier; physical volume integration
    supplies its own source measure. No transform or new mutable cache is kept.
    """
    medium.assert_current()
    eps,mu=medium.epsilon_r,medium.mu_r
    if (eps.requires_grad or mu.requires_grad or bool((eps.imag!=0).any())
            or bool((mu.imag!=0).any()) or not torch.equal(eps,eps.T)):
        return None
    a=eps.real
    pivot=a.diagonal().abs().argmax()
    denominator=a[pivot,pivot]
    if not bool(denominator>0):return None
    scale=mu.real[pivot,pivot]/denominator
    if not bool(torch.isfinite(scale)) or not bool(scale>0) or not torch.equal(scale*a,mu.real):
        return None
    factor,info=torch.linalg.cholesky_ex(a)
    if bool(info!=0):return None
    determinant=factor.diagonal().prod().square()
    if not bool(torch.isfinite(determinant)) or not bool(determinant>0):return None
    transform=factor/determinant.sqrt()
    if not bool(torch.isfinite(transform).all()):return None
    inverse,info=torch.linalg.inv_ex(transform)
    if bool(info!=0) or not bool(torch.isfinite(inverse).all()):return None
    if (not isinstance(displacements,torch.Tensor) or displacements.device!=a.device
            or displacements.dtype!=a.dtype or displacements.ndim!=2 or displacements.shape[1]!=3
            or not bool(torch.isfinite(displacements).all())):
        raise ValueError('Proportional SPD queries must be finite matching CUDA real [N,3]')
    from .tensor_medium import TensorMedium
    from .dyadic_green import isotropic_dyadic_green
    eye=torch.eye(3,device=a.device,dtype=eps.dtype)
    scalar=TensorMedium(medium.frequency,eye,scale*eye)
    value,receipt=isotropic_dyadic_green(displacements@inverse.T,scalar,regular_part=regular_part)
    dual=torch.block_diag(inverse,inverse).to(eps.dtype)
    value=dual.T@value@dual
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite proportional SPD Green value')
    medium.assert_current()
    receipt=dict(receipt,kernel_version='analytic_proportional_spd_spatial_v1',
        physical_qualification=False,constitutive_projection=False)
    return value,receipt


def matched_spd_plane_modes(generator,transverse,medium):
    """Return None outside exact matched fixed real-SPD relative coefficients."""
    medium.assert_current()
    eps,mu=medium.epsilon_r,medium.mu_r
    if (eps.requires_grad or mu.requires_grad or not torch.equal(eps,mu)
            or bool((eps.imag!=0).any()) or not torch.equal(eps,eps.T)):
        return None
    return _transformed_spd_plane_modes(generator,transverse,medium,eps.real,1.,1.,
        'analytic_matched_spd_transformation_modes_v1')


def proportional_spd_plane_modes(generator,transverse,medium):
    """Exact fixed complex scalar multiples of one real SPD metric only."""
    medium.assert_current()
    eps,mu=medium.epsilon_r,medium.mu_r
    if eps.requires_grad or mu.requires_grad or not torch.equal(eps,eps.T):
        return None
    metric=eps.real
    pivot=metric.diagonal().abs().argmax()
    denominator=metric[pivot,pivot]
    if not bool(denominator>0):return None
    electric=1+1j*(eps[pivot,pivot].imag/denominator)
    magnetic=mu[pivot,pivot]/denominator
    if (not bool(torch.isfinite(electric) & torch.isfinite(magnetic)) or bool(magnetic==0)
            or not torch.equal(electric*metric,eps) or not torch.equal(magnetic*metric,mu)):
        return None
    return _transformed_spd_plane_modes(generator,transverse,medium,metric,electric,magnetic,
        'analytic_proportional_spd_transformation_modes_v1')


def _transformed_spd_plane_modes(generator,transverse,medium,a,electric_scale,magnetic_scale,version):
    eps=medium.epsilon_r
    factor,info=torch.linalg.cholesky_ex(a)
    if bool(info!=0):return None
    determinant=factor.diagonal().prod().square()
    if not bool(torch.isfinite(determinant)) or not bool(determinant>0):return None
    transform=factor/determinant.sqrt()
    if not bool(torch.isfinite(transform).all()):return None
    inverse,info=torch.linalg.inv_ex(transform)
    if bool(info!=0) or not bool(torch.isfinite(inverse).all()):return None
    inverse=inverse.to(generator.dtype)
    qt=transverse
    b=qt@a[:2,2]
    c=torch.einsum('ni,ij,nj->n',qt,a[:2,:2],qt)-determinant*electric_scale*magnetic_scale
    discriminant=(b.square()-a[2,2]*c).to(generator.dtype).sqrt()
    positive=(-b+discriminant)/a[2,2]
    negative=(-b-discriminant)/a[2,2]
    roots=torch.stack((positive,positive,negative,negative),-1)
    vectors=[]
    identity=torch.eye(3,device=eps.device,dtype=generator.dtype)
    for longitudinal in (positive,negative):
        qx=torch.cat((qt.to(generator.dtype),longitudinal[:,None]),-1)
        qy=qx@transform.to(generator.dtype)
        reference=identity[qy.abs().argmin(-1)]
        electric=torch.linalg.cross(qy,reference)
        magnetic=torch.linalg.cross(qy,electric)/magnetic_scale
        # qy.qy=e*m and qy.E=0; all products are bilinear, not Hermitian.
        for ey,hy in ((electric,magnetic),(magnetic,-(electric_scale/magnetic_scale)*electric)):
            ex,hx=ey@inverse,hy@inverse
            vectors.append(torch.cat((ex[:,:2],hx[:,:2]),-1))
    vectors=torch.stack(vectors,-1)
    length=vectors.norm(dim=-2)
    tiny=torch.finfo(length.dtype).tiny
    vectors=vectors/length.clamp_min(tiny)[:,None,:]
    finite=torch.isfinite(vectors).all(dim=(-2,-1)) & torch.isfinite(roots).all(-1)
    safe=torch.where(finite[:,None,None],vectors,torch.eye(4,device=eps.device,dtype=eps.dtype))
    residual=(generator@safe-safe*roots[:,None,:]).norm(dim=-2)
    scale=torch.linalg.matrix_norm(generator,ord='fro')[:,None]+roots.abs()
    singular=torch.linalg.svdvals(safe)
    tolerance=512*torch.finfo(length.dtype).eps
    metrics_finite=(torch.isfinite(residual).all(-1) & torch.isfinite(scale).all(-1)
                    & torch.isfinite(singular).all(-1) & torch.isfinite(length).all(-1))
    failed=(~finite | ~metrics_finite | (length==0).any(-1) | (residual>tolerance*scale).any(-1)
            | (singular[:,-1]<=tolerance*singular[:,0]))
    from witwin.maxwell.mom.spectral_linalg import batched_eig4
    fallback_count=0;analytic_fallback_count=int(failed.sum())
    if analytic_fallback_count:
        r,v,receipt=batched_eig4(generator[failed])
        roots=roots.clone();vectors=vectors.clone()
        roots[failed],vectors[failed]=r,v
        fallback_count=receipt['fallback_count']
    return roots,vectors,dict(version=version,
        matrix_count=len(generator),analytic_fallback_count=analytic_fallback_count,
        fallback_count=fallback_count,residual_certifies_eigenvalue_error=False)


def matched_spd_transverse_rule(nodes,weights,k0,reference,medium,circle):
    """Integrate rho d rho with the exact angle-dependent branch ellipse.

    Relative epsilon and mu must be equal, fixed real SPD tensors. Below the
    branch rho=b sin(theta); above rho=b sqrt(1+t^2). Both Jacobians cancel
    the outgoing square-root singularity. The tail remains infinite.
    """
    medium.assert_current()
    a=medium.epsilon_r
    if (a.requires_grad or medium.mu_r.requires_grad or not torch.equal(a,medium.mu_r)
            or bool((a.imag!=0).any()) or not torch.equal(a,a.T)):
        raise ValueError('Matched-SPD branch mapping requires fixed exact matched real SPD tensors')
    return _spd_transverse_rule(nodes,weights,k0,reference,a.real,circle,1.)


def proportional_spd_transverse_rule(nodes,weights,k0,reference,medium,circle):
    """Split the real branch ellipse for fixed exact mu = scale * epsilon.

    Epsilon is real SPD and scale is positive. No approximate proportionality,
    loss removal or constitutive projection is permitted.
    """
    medium.assert_current()
    eps,mu=medium.epsilon_r,medium.mu_r
    if (eps.requires_grad or mu.requires_grad or bool((eps.imag!=0).any())
            or bool((mu.imag!=0).any()) or not torch.equal(eps,eps.T)):
        raise ValueError('Proportional-SPD mapping requires fixed real symmetric tensors')
    a=eps.real
    pivot=a.diagonal().abs().argmax()
    denominator=a[pivot,pivot]
    if not bool(denominator>0):raise ValueError('Proportional-SPD mapping requires positive epsilon')
    scale=mu.real[pivot,pivot]/denominator
    if not bool(torch.isfinite(scale)) or not bool(scale>0) or not torch.equal(scale*a,mu.real):
        raise ValueError('Proportional-SPD mapping requires exact positive proportionality')
    return _spd_transverse_rule(nodes,weights,k0,reference,a,circle,scale)


def spd_reference_transverse_rule(nodes,weights,k0,reference,medium,circle):
    """Place nodes using a proportional reference; retain the actual medium.

    This is only a radial change of variables onto [0,infinity). The reference
    branch need not be a physical modal branch. Actual spectral modes and all
    adaptive convergence checks use the original epsilon and mu.
    """
    medium.assert_current()
    eps,mu=medium.epsilon_r,medium.mu_r
    for value in (eps,mu):
        if (value.requires_grad or bool((value.imag!=0).any())
                or not torch.equal(value,value.T)):
            raise ValueError('SPD reference mapping requires fixed real symmetric tensors')
        _,info=torch.linalg.cholesky_ex(value.real)
        if bool(info!=0):raise ValueError('SPD reference mapping requires positive definite tensors')
    a=eps.real
    pivot=a.diagonal().argmax()
    scale=mu.real[pivot,pivot]/a[pivot,pivot]
    if not bool(torch.isfinite(scale)) or not bool(scale>0):
        raise ValueError('SPD reference scale is not positive and representable')
    return _spd_transverse_rule(nodes,weights,k0,reference,a,circle,scale)


def paired_spd_reference_transverse_rule(nodes, weights, k0, reference, exterior, interior, circle):
    """Common complete radial coordinates split at both reference ellipses.

    Reference branches place quadrature nodes; the original constitutive bits
    still enter every spectral evaluation. General SPD reference branches need
    not be exact physical modal branches. No transverse tail is truncated.
    """
    from .dyadic_green import _validate_factored_plane_medium
    branches = []
    for medium in (exterior, interior):
        _validate_factored_plane_medium(medium)
        a = medium.epsilon_r.real
        pivot = a.diagonal().argmax()
        scale = medium.mu_r.real[pivot, pivot]/a[pivot, pivot]
        factor, info = torch.linalg.cholesky_ex(a)
        if bool(info != 0):
            raise ValueError('Paired SPD reference requires positive definite tensors')
        determinant = factor.diagonal().prod().square()
        schur = a[:2, :2]-a[:2, 2, None]*a[None, 2, :2]/a[2, 2]
        denominator = torch.einsum('ni,ij,nj->n', circle, schur, circle)
        if (not bool(torch.isfinite(determinant) & torch.isfinite(scale))
                or not bool(determinant > 0) or not bool(scale > 0)
                or not bool((torch.isfinite(denominator) & (denominator > 0)).all())):
            raise ValueError('Paired SPD reference ellipses are not representable')
        branches.append((scale*determinant/denominator).sqrt())
    low, high = torch.minimum(*branches), torch.maximum(*branches)
    theta = nodes[:, None]*(math.pi/2)
    sine, cosine = theta.sin(), theta.cos()
    propagating = low[None]*sine
    propagating_weights = weights[:, None]*low[None].square()*sine*cosine*(math.pi/2)
    span = high.square()-low.square()
    between = (low[None].square()+span[None]*sine.square()).sqrt()
    # An empty middle interval has zero measure. Keep its placeholders away
    # from the coincident reference branch instead of compiling undefined poles.
    between = torch.where(span[None] > 0, between, low[None]*nodes[:, None])
    between_weights = weights[:, None]*span[None]*sine*cosine*(math.pi/2)
    t = nodes[:, None]/(1-nodes[:, None])/(k0*high[None]*reference)
    evanescent = high[None]*torch.sqrt(1+t.square())
    evanescent_weights = weights[:, None]*high[None]*t/(k0*reference*(1-nodes[:, None]).square())
    rho = torch.cat((propagating, between, evanescent))
    radial_weights = torch.cat((propagating_weights, between_weights, evanescent_weights))
    if not bool(torch.isfinite(rho).all() & torch.isfinite(radial_weights).all()):
        raise ValueError('Paired SPD radial quadrature is not representable')
    return (rho[:, :, None]*circle[None]).reshape(-1, 2), radial_weights.flatten()*k0**2/(2*math.pi*len(circle))


def _spd_transverse_rule(nodes,weights,k0,reference,a,circle,magnetic_scale):
    factor,info=torch.linalg.cholesky_ex(a)
    if bool(info!=0):raise ValueError('Matched-SPD branch mapping requires positive definite tensors')
    determinant=factor.diagonal().prod().square()
    schur=a[:2,:2]-a[:2,2,None]*a[None,2,:2]/a[2,2]
    denominator=torch.einsum('ni,ij,nj->n',circle,schur,circle)
    if (not bool(torch.isfinite(determinant)) or not bool(determinant>0)
            or not bool((torch.isfinite(denominator)&(denominator>0)).all())):
        raise ValueError('Matched-SPD branch ellipse is not representable')
    branch=(magnetic_scale*determinant/denominator).sqrt()
    theta=nodes[:,None]*(math.pi/2)
    prop=branch[None]*theta.sin()
    pw=weights[:,None]*branch[None].square()*theta.sin()*theta.cos()*(math.pi/2)
    t=nodes[:,None]/(1-nodes[:,None])/(k0*branch[None]*reference)
    evan=branch[None]*torch.sqrt(1+t.square())
    ew=weights[:,None]*branch[None]*t/(k0*reference*(1-nodes[:,None]).square())
    rho=torch.cat((prop,evan));radial_weights=torch.cat((pw,ew))
    if not bool(torch.isfinite(rho).all() & torch.isfinite(radial_weights).all()):
        raise ValueError('Matched-SPD radial quadrature is not representable')
    return (rho[:,:,None]*circle[None]).reshape(-1,2),radial_weights.flatten()*k0**2/(2*math.pi*len(circle))
