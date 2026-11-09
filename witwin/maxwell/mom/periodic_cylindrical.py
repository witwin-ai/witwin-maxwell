"""One-dimensional periodic outgoing cylindrical scattering amplitudes."""
import math
import torch
from .periodic_operator import validate_periodic_cell,validate_periodic_sources
from .periodic_spectral import surface_current_fourier


def cylindrical_orders(handle,coefficients,*,lattice,orders,directions,quadrature_order=6,
                       grazing_relative_tolerance=1e-10):
    """Return F such that E/H ~ sum_n F_n exp(i q_n.r+i gamma_n rho)/sqrt(rho).

    Directions are unit vectors normal to the lattice axis. Power is per radian
    and primitive axial period for propagating orders in a lossless exterior.
    Evanescent coefficients retain their radial exponential but carry no power
    at infinity. These are scattered cylindrical amplitudes, not isolated RCS.
    """
    if lattice.dimension!=1:
        raise ValueError('Cylindrical orders require a one-dimensional lattice')
    validate_periodic_cell(handle.system,lattice)
    validate_periodic_sources(handle.sources,lattice,handle.exterior)
    medium=handle.exterior
    if medium.k.imag!=0 or medium.epsilon_r.imag!=0 or medium.mu_r.imag!=0:
        raise ValueError('Cylindrical radiation power requires a lossless exterior')
    sample=lattice.vectors
    dtype=torch.complex128 if sample.dtype==torch.float64 else torch.complex64
    if (not isinstance(coefficients,torch.Tensor) or coefficients.shape!=(2*handle.system.total_edges,)
            or coefficients.dtype!=dtype or coefficients.device!=sample.device
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Cylindrical currents require finite matching global CUDA RWG coefficients')
    if (not isinstance(orders,torch.Tensor) or orders.ndim!=2 or orders.shape[1]!=1
            or orders.dtype!=torch.int64 or orders.device!=sample.device
            or len(torch.unique(orders,dim=0))!=len(orders)):
        raise ValueError('Cylindrical orders require unique CUDA int64 [N,1] indices')
    axis=sample[0]/lattice.cell_measure
    tolerance=512*torch.finfo(sample.dtype).eps
    if (not isinstance(directions,torch.Tensor) or directions.ndim!=2 or directions.shape[1]!=3
            or directions.device!=sample.device or directions.dtype!=sample.dtype
            or not bool(torch.isfinite(directions).all())
            or bool(((directions.square().sum(-1)-1).abs()>tolerance).any())
            or bool(((directions@axis).abs()>tolerance).any())):
        raise ValueError('Cylindrical directions must be finite unit vectors perpendicular to the lattice axis')
    if type(quadrature_order) is not int or quadrature_order<2:
        raise ValueError('Cylindrical quadrature order must be an integer >=2')
    if type(grazing_relative_tolerance) not in (int,float) or not math.isfinite(grazing_relative_tolerance) or grazing_relative_tolerance<=0:
        raise ValueError('Grazing tolerance must be finite and positive')
    q=orders.to(sample.dtype)@lattice.reciprocal+lattice.bloch
    q2=q.square().sum(-1)
    gap=medium.k**2-q2
    scale=torch.maximum(q2,q2.new_full(q2.shape,abs(medium.k)**2))
    if bool((gap.abs()<=max(tolerance,grazing_relative_tolerance)*scale).any()):
        raise ValueError('Unresolved grazing cylindrical order requires an explicit anomaly treatment')
    gamma=gap.to(dtype).sqrt()
    gamma=torch.where((gamma.imag==0)&(gamma.real*medium.k.real<0),-gamma,gamma)
    p=q[:,None,:].to(dtype)+gamma[:,None,None]*directions[None]
    j,m=surface_current_fourier(handle,coefficients,p.reshape(-1,3),quadrature_order)
    j,m=j.reshape_as(p),m.reshape_as(p)
    def transverse(value):
        return value-p*(p*value).sum(-1,keepdim=True)/medium.k**2
    # i H0^(1)/(4 L), with H0^(1) ~ sqrt(2/(pi gamma rho))*exp(i gamma rho-i pi/4).
    # Negative-index propagating modes use the conjugate outgoing branch.
    positive=complex(math.cos(math.pi/4),math.sin(math.pi/4))
    factor=positive/gamma.sqrt()
    factor=torch.where((gamma.imag==0)&(gamma.real<0),positive.conjugate()/gamma.abs().sqrt(),factor)
    factor=factor/(lattice.cell_measure*math.sqrt(8*math.pi))
    e=factor[:,None,None]*(1j*medium.omega*medium.mu*transverse(j)-1j*torch.linalg.cross(p,m))
    h=factor[:,None,None]*(1j*torch.linalg.cross(p,j)+1j*medium.omega*medium.epsilon*transverse(m))
    propagating=gamma.imag==0
    power=.5*lattice.cell_measure*(torch.linalg.cross(e,h.conj()).real*directions[None]).sum(-1)
    power=torch.where(propagating[:,None],power,torch.zeros_like(power))
    if not bool(torch.isfinite(e).all() and torch.isfinite(h).all() and torch.isfinite(power).all()):
        raise ValueError('Nonfinite cylindrical amplitudes; evanescent source exponential overflow')
    return {'orders':orders.clone(),'directions':directions.clone(),'axial_wavevectors':q,
        'wavevectors':p,'radial_wavenumbers':gamma,'electric':e,'magnetic':h,
        'radiated_power_W_per_radian':power,'propagating_lossless':propagating,
        'period_m':lattice.cell_measure,'axis':axis,'field_kind':'scattered',
        'normalization':'outgoing_exp_i_gamma_rho_over_sqrt_rho_per_axial_cell',
        'quadrature_order':quadrature_order,'quadrature_converged':False,
        'order_truncation_converged':False,'angular_quadrature_converged':False,
        'physical_qualification':False}
