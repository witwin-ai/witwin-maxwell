"""Explicit projected quantities at 3D scalar Bloch eigenvalues.

These are not an admission path for an ordinary singular forced solve.
"""
import math
import torch
from .periodic import EwaldHelmholtzKernel


def projected_bloch_green(kernel,displacement,*,eigen_order):
    """Return the reduced resolvent and z-residue at z=|beta+B n|^2.

    The full degenerate eigenspace is projected out. With R(z)=(A-z)^-1,
    residue_z R=-P and the reduced kernel obeys (Delta+z)G=-delta+P.
    Ewald's real-space part still contains P/(4 eta^2); removing only the
    reciprocal zero denominator would therefore give the wrong finite part.
    """
    if not isinstance(kernel,EwaldHelmholtzKernel) or kernel.lattice.dimension!=3:
        raise ValueError('Projected Bloch resolvents require a three-dimensional scalar Ewald kernel')
    lattice=kernel.lattice
    lattice.assert_current()
    sample=lattice.vectors
    if (not isinstance(eigen_order,torch.Tensor) or eigen_order.shape!=(3,)
            or eigen_order.device!=sample.device or eigen_order.dtype!=torch.int64):
        raise ValueError('Eigenvalue selection requires a CUDA int64 [3] reciprocal order')
    if (not isinstance(displacement,torch.Tensor) or displacement.ndim<1 or displacement.shape[-1]!=3
            or displacement.device!=sample.device or displacement.dtype!=sample.dtype
            or not bool(torch.isfinite(displacement).all())):
        raise ValueError('Projected queries require finite matching real CUDA coordinates')
    selected=eigen_order.to(sample.dtype)@lattice.reciprocal+lattice.bloch
    eigenvalue=selected.square().sum()
    k=math.sqrt(float(eigenvalue))
    smallest=float(torch.linalg.svdvals(lattice.reciprocal)[-1])
    if kernel.reciprocal_shells<math.ceil((k+float(lattice.bloch.norm()))/smallest):
        raise ValueError('Reciprocal cutoff does not cover the complete degenerate eigenspace search')
    roundoff=128*torch.finfo(sample.dtype).eps
    projected=[]
    for indices in kernel._indices(kernel.reciprocal_shells,sample.device):
        q=indices.to(sample.dtype)@lattice.reciprocal+lattice.bloch
        q2=q.square().sum(-1)
        gap=q2-eigenvalue
        scale=torch.maximum(q2,q2.new_full(q2.shape,max(float(eigenvalue),smallest**2)))
        degenerate=gap.abs()<=roundoff*scale
        unresolved=(gap.abs()<=max(roundoff,kernel.pole_relative_tolerance)*scale)&~degenerate
        if bool(unresolved.any()):
            raise ValueError('Nearby distinct Bloch eigenvalues are unresolved; projection is ambiguous')
        projected.append(indices[degenerate])
    projected=torch.cat(projected)
    if not len(projected):
        raise ValueError('Selected Bloch eigenspace was not resolved')
    projected_q=projected.to(sample.dtype)@lattice.reciprocal+lattice.bloch
    points=displacement.reshape(-1,3)
    shifts=torch.floor(points@lattice.reciprocal.T/(2*math.pi)+0.5)
    translations=shifts@lattice.vectors
    wrapped=points-translations
    phase=torch.exp(1j*(translations@lattice.bloch))
    dtype=torch.complex128 if sample.dtype==torch.float64 else torch.complex64
    values,gradients,projectors,projector_gradients=[],[],[],[]
    for query in wrapped.split(kernel.query_batch):
        value,gradient=kernel._real(query,complex(k))
        projector=query.new_zeros(len(query),dtype=dtype)
        projector_gradient=query.new_zeros(query.shape,dtype=dtype)
        for begin in range(0,len(projected_q),kernel.image_batch):
            q=projected_q[begin:begin+kernel.image_batch]
            term=torch.exp(1j*(query@q.T))/lattice.cell_measure
            projector=projector+term.sum(-1)
            projector_gradient=projector_gradient+(term[...,None]*(1j*q)).sum(-2)
        for indices in kernel._indices(kernel.reciprocal_shells,sample.device):
            q=indices.to(sample.dtype)@lattice.reciprocal+lattice.bloch
            gap=q.square().sum(-1)-eigenvalue
            omitted=(indices[:,None,:]==projected[None]).all(-1).any(-1)
            q,gap=q[~omitted],gap[~omitted]
            amplitude=torch.exp(-gap/(4*kernel.eta**2))/(lattice.cell_measure*gap)
            term=torch.exp(1j*(query@q.T))*amplitude
            value=value+term.sum(-1)
            gradient=gradient+(term[...,None]*(1j*q)).sum(-2)
        values.append(value-projector/(4*kernel.eta**2))
        gradients.append(gradient-projector_gradient/(4*kernel.eta**2))
        projectors.append(projector)
        projector_gradients.append(projector_gradient)
    def collect(parts,vector=False):
        shape=displacement.shape if vector else displacement.shape[:-1]
        if not parts:
            return displacement.new_empty(shape,dtype=dtype)
        output=torch.cat(parts)*(phase[:,None] if vector else phase)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Projected Ewald sum is nonfinite; change the splitting parameter')
        return output.reshape(shape)
    value,gradient=collect(values),collect(gradients,True)
    projector,projector_gradient=collect(projectors),collect(projector_gradients,True)
    return {'green':value,'gradient':gradient,'projector_kernel':projector,
        'projector_gradient':projector_gradient,'residue_z':-projector,
        'residue_z_gradient':-projector_gradient,'projected_orders':projected,
        'eigenvalue_per_m2':eigenvalue,'spectral_parameter':'z=k_squared',
        'quantity':'reduced_resolvent_on_complement_of_complete_bloch_eigenspace',
        'ordinary_forced_solution':False,'kernel':kernel.receipt(),
        'eigenspace_relative_resolution':roundoff,
        'support_domain':'scalar_3d_bloch_laplacian_complete_resolved_eigenspace',
        'cutoff_converged':False,'physical_qualification':False}
