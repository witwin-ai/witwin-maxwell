"""Off-spectrum 3D Bloch coefficients of exterior equivalent-current potentials."""
import torch
from .lifting_fields import system_local_coefficients
from .operator import local_geometry
from .quadrature import triangle_rule
from .periodic import EwaldHelmholtzKernel
from .periodic_operator import validate_periodic_cell


def surface_current_fourier(handle,coefficients,wavevectors,order,*,phase=None):
    """Integrate actual RWG currents in bounded face/mode batches.

    A custom phase supports stably propagated Rayleigh planes. Otherwise the
    convention is the unnormalized integral of current times exp(-i q.r).
    """
    electric=coefficients.new_zeros(len(wavevectors),3)
    magnetic=torch.zeros_like(electric)
    for index,body in enumerate(handle.system.bodies):
        basis=body.basis
        local=system_local_coefficients(handle,coefficients,index)
        edges,free,slopes=local_geometry(basis)
        j,m=local[:basis.n_edges][edges]*slopes,local[basis.n_edges:][edges]*slopes
        for start in range(0,len(body.mesh.faces),handle.integration.pair_batch):
            stop=min(start+handle.integration.pair_batch,len(body.mesh.faces))
            points,weights=triangle_rule(body.mesh.triangles[start:stop],order)
            displacement=(points[:,:,None]-free[start:stop,None]).to(coefficients.dtype)
            current_j=torch.einsum('fl,fqlc->fqc',j[start:stop],displacement)
            current_m=torch.einsum('fl,fqlc->fqc',m[start:stop],displacement)
            for begin in range(0,len(wavevectors),32):
                end=min(begin+32,len(wavevectors))
                exponential=(torch.exp(-1j*torch.einsum('nc,fqc->nfq',wavevectors[begin:end],points.to(wavevectors.dtype)))
                    if phase is None else phase(points,begin,end))
                weighted=exponential*weights
                electric[begin:end]=electric[begin:end]+torch.einsum('nfq,fqc->nc',weighted,current_j)
                magnetic[begin:end]=magnetic[begin:end]+torch.einsum('nfq,fqc->nc',weighted,current_m)
    return electric,magnetic


def bloch_spectral_fields(handle,coefficients,*,kernel,orders,quadrature_order=6):
    """Return 3D Fourier coefficients of exterior current-generated potentials.

    These are distributional homogeneous-exterior fields over a cell, not the
    piecewise physical fields inside material bodies. No incident field or
    isolated-scatterer far field is included. Real resolvent poles reject even
    when the requested subset omits the resonant order; no projected-pole or
    eigenmode interpretation is silently substituted.
    """
    if not isinstance(kernel,EwaldHelmholtzKernel) or kernel.lattice.dimension!=3:
        raise ValueError('Bloch spectral fields require a three-dimensional Ewald kernel')
    lattice=kernel.lattice
    validate_periodic_cell(handle.system,lattice)
    kernel._guard(complex(handle.exterior.k))
    sample=lattice.vectors
    dtype=torch.complex128 if sample.dtype==torch.float64 else torch.complex64
    if (not isinstance(coefficients,torch.Tensor) or coefficients.shape!=(2*handle.system.total_edges,)
            or coefficients.dtype!=dtype or coefficients.device!=sample.device
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Bloch spectra require finite matching global CUDA RWG coefficients')
    if (not isinstance(orders,torch.Tensor) or orders.ndim!=2 or orders.shape[1]!=3
            or orders.dtype!=torch.int64 or orders.device!=sample.device
            or len(torch.unique(orders,dim=0))!=len(orders)):
        raise ValueError('Bloch orders require unique CUDA int64 [N,3] indices')
    if type(quadrature_order) is not int or quadrature_order<2:
        raise ValueError('Bloch quadrature order must be an integer >=2')
    q=orders.to(sample.dtype)@lattice.reciprocal+lattice.bloch
    j,m=surface_current_fourier(handle,coefficients,q,quadrature_order)
    j,m=j/lattice.cell_measure,m/lattice.cell_measure
    medium=handle.exterior
    denominator=q.square().sum(-1)-medium.k**2
    p=q.to(dtype)
    def transverse(value):
        return value-p*(p*value).sum(-1,keepdim=True)/medium.k**2
    electric=(1j*medium.omega*medium.mu*transverse(j)-1j*torch.linalg.cross(p,m))/denominator[:,None]
    magnetic=(1j*torch.linalg.cross(p,j)+1j*medium.omega*medium.epsilon*transverse(m))/denominator[:,None]
    if not bool(torch.isfinite(electric).all() and torch.isfinite(magnetic).all()):
        raise ValueError('Nonfinite Bloch spectral fields')
    return {'orders':orders.clone(),'wavevectors':q,'electric':electric,'magnetic':magnetic,
        'electric_current_density':j,'magnetic_current_density':m,
        'spectral_denominators':denominator,'cell_volume_m3':lattice.cell_measure,
        'field_kind':'homogeneous_exterior_equivalent_current_potential',
        'normalization':'cell_volume_normalized_fourier_coefficients_exp_positive_i_q_r',
        'kernel':kernel.receipt(),'quadrature_order':quadrature_order,
        'quadrature_converged':False,'order_truncation_converged':False,
        'physical_qualification':False,'physical_body_interior_fields':False}
