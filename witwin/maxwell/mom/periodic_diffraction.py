"""Rayleigh amplitudes of scattered RWG fields on 2D periodic exterior planes."""
import math
import torch
from .periodic_operator import validate_periodic_cell,validate_periodic_sources
from .periodic_spectral import surface_current_fourier


def rayleigh_orders(handle,coefficients,*,lattice,orders,side,plane_offset,quadrature_order=6,
                    grazing_relative_tolerance=1e-10):
    """Return scattered E/H Fourier amplitudes and outward power per unit cell.

    Lattice orientation defines n=(a1 x a2)/area. `side` is +/-1 and the
    plane n.r=plane_offset must be strictly outside every body on that side.
    Amplitudes include propagation to that plane but not exp(i q.r_parallel).
    Incident amplitudes are not included; this is not transmission efficiency.
    """
    if lattice.dimension!=2:
        raise ValueError('Discrete Rayleigh orders require a two-dimensional lattice')
    validate_periodic_cell(handle.system,lattice)
    validate_periodic_sources(handle.sources,lattice,handle.exterior)
    sample=lattice.vectors
    dtype=torch.complex128 if sample.dtype==torch.float64 else torch.complex64
    if (not isinstance(coefficients,torch.Tensor) or coefficients.shape!=(2*handle.system.total_edges,)
            or coefficients.dtype!=dtype or coefficients.device!=sample.device
            or not bool(torch.isfinite(coefficients).all())):
        raise ValueError('Rayleigh currents must match finite global CUDA RWG coefficients')
    if (not isinstance(orders,torch.Tensor) or orders.ndim!=2 or orders.shape[1]!=2
            or orders.dtype!=torch.int64 or orders.device!=sample.device):
        raise ValueError('Diffraction orders require CUDA int64 [N,2] indices')
    if len(torch.unique(orders,dim=0))!=len(orders):
        raise ValueError('Diffraction orders must be unique')
    if type(side) is not int or side not in (-1,1):
        raise ValueError('Rayleigh side must be +1 or -1')
    if type(plane_offset) not in (int,float) or not math.isfinite(plane_offset):
        raise ValueError('Rayleigh plane offset must be finite')
    if type(quadrature_order) is not int or quadrature_order<2:
        raise ValueError('Rayleigh quadrature order must be an integer >=2')
    if not math.isfinite(grazing_relative_tolerance) or grazing_relative_tolerance<=0:
        raise ValueError('Grazing tolerance must be finite and positive')
    normal=torch.linalg.cross(sample[0],sample[1])/lattice.cell_measure
    coordinates=torch.cat([body.mesh.vertices@normal for body in handle.system.bodies])
    if not bool((side*(plane_offset-coordinates)>0).all()):
        raise ValueError('Rayleigh plane must lie strictly outside every body')
    medium=handle.exterior
    q=orders.to(sample.dtype)@lattice.reciprocal+lattice.bloch
    q2=q.square().sum(-1)
    gap=medium.k**2-q2
    gamma=gap.to(dtype).sqrt()
    flip=(gamma.imag<0)|((gamma.imag==0)&(gamma.real*medium.k.real<0))
    gamma=torch.where(flip,-gamma,gamma)
    scale=torch.maximum(q2.sqrt(),q2.new_full(q2.shape,abs(medium.k)))
    resolved=max(grazing_relative_tolerance,512*torch.finfo(sample.dtype).eps)
    if bool((gap.abs()<=resolved*scale.square()).any()):
        raise ValueError('Unresolved grazing Rayleigh order requires an explicit anomaly treatment')
    wavevector=q.to(dtype)+side*gamma[:,None]*normal
    def propagated_phase(points,begin,end):
        height=plane_offset-points@normal
        return torch.exp(1j*(side*gamma[begin:end,None,None]*height
            -torch.einsum('nc,fqc->nfq',q[begin:end],points)))
    j_hat,m_hat=surface_current_fourier(handle,coefficients,q,quadrature_order,phase=propagated_phase)
    def transverse(value):
        return value-wavevector*(wavevector*value).sum(-1,keepdim=True)/medium.k**2
    factor=1j/(2*lattice.cell_measure*gamma)
    electric=factor[:,None]*(1j*medium.omega*medium.mu*transverse(j_hat)-1j*torch.linalg.cross(wavevector,m_hat))
    magnetic=factor[:,None]*(1j*torch.linalg.cross(wavevector,j_hat)+1j*medium.omega*medium.epsilon*transverse(m_hat))
    outward=.5*lattice.cell_measure*(torch.linalg.cross(electric,magnetic.conj()).real*(side*normal)).sum(-1)
    if not bool(torch.isfinite(electric).all() and torch.isfinite(magnetic).all() and torch.isfinite(outward).all()):
        raise ValueError('Nonfinite Rayleigh amplitudes; reduce the evanescent propagation range')
    return {'orders':orders.clone(),'tangential_wavevectors':q,'wavevectors':wavevector,
        'normal_wavenumbers':gamma,'electric':electric,'magnetic':magnetic,'outward_power_W':outward,
        'propagating_lossless':(gamma.imag==0)&(gamma.real!=0),'normal':normal,'plane_offset_m':plane_offset,
        'side':side,'cell_area_m2':lattice.cell_measure,'field_kind':'scattered',
        'quadrature_order':quadrature_order,'quadrature_converged':False,
        'order_truncation_converged':False,'physical_qualification':False,
        'normalization':'outward_time_average_power_per_primitive_cell_at_plane'}


def rayleigh_total(handle,coefficients,*,lattice,orders,side,plane_offset,
                   quadrature_order=6,grazing_relative_tolerance=1e-10):
    """Combine incident and scattered plane amplitudes coherently by order.

    Total fields can contain both normal propagation directions. Only the nested
    scattered record has a single outgoing wavevector. Signed outward flux and
    interference are plane observables, not a certified absorption efficiency.
    """
    from .physics import incident_fields
    scattered=rayleigh_orders(handle,coefficients,lattice=lattice,orders=orders,
        side=side,plane_offset=plane_offset,quadrature_order=quadrature_order,
        grazing_relative_tolerance=grazing_relative_tolerance)
    electric=torch.zeros_like(scattered['electric'])
    magnetic=torch.zeros_like(scattered['magnetic'])
    point=(plane_offset*scattered['normal'])[None]
    for source in handle.sources:
        direction=lattice.vectors.new_tensor(source.direction)
        cycles=((handle.exterior.k.real*direction-lattice.bloch)@lattice.vectors.T)/(2*math.pi)
        index=cycles.round().to(torch.int64)
        matches=(orders==index).all(-1)
        if not bool(matches.any()):
            raise ValueError('Requested Rayleigh orders omit an incident diffraction order')
        e,h=incident_fields(source,point,handle.exterior)
        electric=electric+matches[:,None]*e
        magnetic=magnetic+matches[:,None]*h
    total_e=electric+scattered['electric']
    total_h=magnetic+scattered['magnetic']
    def power(e,h):
        return .5*lattice.cell_measure*(torch.linalg.cross(e,h.conj()).real*(side*scattered['normal'])).sum(-1)
    incident_power=power(electric,magnetic)
    total_power=power(total_e,total_h)
    # Evaluate cross terms directly so weak scattering is not lost to subtraction
    # of two incident-dominated total powers.
    interference=power(electric,scattered['magnetic'])+power(scattered['electric'],magnetic)
    return {'scattered':scattered,'orders':scattered['orders'],
        'tangential_wavevectors':scattered['tangential_wavevectors'],
        'electric':total_e,'magnetic':total_h,
        'incident_electric':electric,'incident_magnetic':magnetic,
        'outward_power_W':total_power,'incident_outward_power_W':incident_power,
        'interference_outward_power_W':interference,
        'normal':scattered['normal'],'side':side,'plane_offset_m':plane_offset,
        'cell_area_m2':lattice.cell_measure,'field_kind':'total',
        'quadrature_converged':False,'order_truncation_converged':False,
        'physical_qualification':False,'normalization':scattered['normalization']}


def rayleigh_power_balance(handle,coefficients,*,lattice,orders,lower_plane,upper_plane,
                           quadrature_order=6,grazing_relative_tolerance=1e-10):
    """Two-plane scattering/extinction ledger in a lossless exterior.

    Flux-inferred absorption is a diagnostic, not an independent material-work
    integral. Its normalization by scattered/extinction power exposes weak-
    scattering defects that incident-normalized transmission can conceal.
    """
    medium=handle.exterior
    if medium.epsilon_r.imag!=0 or medium.mu_r.imag!=0 or medium.k.imag!=0:
        raise ValueError('Two-plane scattering balance requires a lossless exterior')
    planes=[rayleigh_total(handle,coefficients,lattice=lattice,orders=orders,
        side=side,plane_offset=offset,quadrature_order=quadrature_order,
        grazing_relative_tolerance=grazing_relative_tolerance)
        for side,offset in ((-1,lower_plane),(1,upper_plane))]
    scattered=sum(p['scattered']['outward_power_W'].sum() for p in planes)
    extinction=-sum(p['interference_outward_power_W'].sum() for p in planes)
    absorption=extinction-scattered
    scale=torch.maximum(scattered.abs(),extinction.abs())
    relative=torch.where(scale>0,absorption.abs()/scale.clamp_min(torch.finfo(scale.dtype).tiny),
        torch.zeros_like(scale))
    return {'lower':planes[0],'upper':planes[1],
        'scattered_power_W':scattered,'extinction_power_W':extinction,
        'flux_inferred_absorption_W':absorption,
        'net_total_outward_power_W':sum(p['outward_power_W'].sum() for p in planes),
        'net_incident_outward_power_W':sum(p['incident_outward_power_W'].sum() for p in planes),
        'lossless_balance_relative_defect':relative,
        'lossless_balance_scale_W':scale,'physical_qualification':False,
        'independent_material_work_verified':False,'order_truncation_converged':False,
        'quadrature_converged':False}


class RayleighConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Rayleigh independent quadrature/order estimates failed')


def estimate_rayleigh_total(handle,coefficients,*,lattice,shells,side,plane_offset,
                            quadrature_order=6,relative_tolerance=1e-7,
                            field_absolute_tolerance=1e-10,power_absolute_tolerance=1e-14,
                            grazing_relative_tolerance=1e-10):
    """Independently double surface quadrature and square order range.

    Compare scattered E/eta-H modal vectors and scattered/interference power
    vectors, never incident-dominated total fields. Missing modes are zero-padded
    so evanescent tails count even when their outward power vanishes. Successive
    estimates are not bounds and do not qualify the underlying current solution.
    """
    if lattice.dimension!=2:
        raise ValueError('Rayleigh estimates require a two-dimensional lattice')
    lattice.assert_current()
    if type(shells) is not int or shells<1:
        raise ValueError('Rayleigh shells must be a positive integer')
    tolerances=(relative_tolerance,field_absolute_tolerance,power_absolute_tolerance)
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in tolerances)
            or relative_tolerance+field_absolute_tolerance==0
            or relative_tolerance+power_absolute_tolerance==0):
        raise ValueError('Rayleigh tolerances must be finite, nonnegative and nonzero for each quantity')
    minimum=math.ceil((abs(handle.exterior.k)+float(lattice.bloch.norm()))/
        float(torch.linalg.svdvals(lattice.reciprocal).min()))
    if shells<minimum:
        raise ValueError('Rayleigh order range must include all potentially propagating orders')
    def evaluate(count,order):
        axis=torch.arange(-count,count+1,device=lattice.vectors.device)
        indices=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        return rayleigh_total(handle,coefficients,lattice=lattice,orders=indices,side=side,
            plane_offset=plane_offset,quadrature_order=order,
            grazing_relative_tolerance=grazing_relative_tolerance)
    baseline=evaluate(shells,quadrature_order)
    eta=abs(handle.exterior.eta)
    def vectors(result):
        scattered=result['scattered']
        fields=torch.cat((scattered['electric'],eta*scattered['magnetic']),-1)
        powers=torch.stack((scattered['outward_power_W'],result['interference_outward_power_W']),-1)
        return fields,powers
    base_vectors=vectors(baseline)
    checks=[]
    for name,count,order in (('surface_quadrature',shells,2*quadrature_order),
                             ('diffraction_orders',2*shells,quadrature_order)):
        refined=evaluate(count,order)
        refined_vectors=vectors(refined)
        positions=(baseline['orders'][:,0]+count)*(2*count+1)+baseline['orders'][:,1]+count
        quantities=[]
        for quantity,base,target,atol in zip(('scattered_E_etaH','scattered_interference_power'),
                base_vectors,refined_vectors,(field_absolute_tolerance,power_absolute_tolerance)):
            padded=torch.zeros_like(target).index_copy(0,positions,base)
            difference=torch.linalg.vector_norm(target-padded)
            scale=torch.linalg.vector_norm(target)
            threshold=atol+relative_tolerance*scale
            quantities.append({'quantity':quantity,'difference':float(difference),
                'reference_norm':float(scale),'threshold':float(threshold),
                'passed':bool(torch.isfinite(difference)&(difference<=threshold))})
        checks.append({'axis':name,'baseline':quadrature_order if name=='surface_quadrature' else shells,
            'refined':order if name=='surface_quadrature' else count,'quantities':quantities,
            'passed':all(q['passed'] for q in quantities)})
    receipt={'checks':checks,'converged':all(c['passed'] for c in checks),
        'relative_tolerance':relative_tolerance,'field_absolute_tolerance':field_absolute_tolerance,
        'power_absolute_tolerance_W':power_absolute_tolerance,
        'certified_error_bound':False,'physical_qualification':False,
        'normalization':'modal_l2_scattered_E_etaH_and_scattered_interference_cell_power'}
    if not receipt['converged']:
        raise RayleighConvergenceError(receipt)
    baseline['quadrature_converged']=True
    baseline['order_truncation_converged']=True
    baseline['scattered']['quadrature_converged']=True
    baseline['scattered']['order_truncation_converged']=True
    baseline['convergence']=receipt
    return baseline
