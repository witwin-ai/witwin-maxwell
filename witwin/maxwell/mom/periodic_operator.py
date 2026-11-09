"""Periodic exterior correction to free-space multi-body PMCHWT assembly."""
import math
from dataclasses import dataclass,replace
import torch
from .operator import IntegrationConfig,assemble_system,_pair_integrals
from .periodic import EwaldHelmholtzKernel


class _RegularKernel:
    def __init__(self,kernel):
        self.kernel=kernel

    def evaluate(self,displacement,k):
        return self.kernel.regular_part(displacement,k)


def validate_periodic_cell(system,lattice):
    """Require each closed body strictly inside the centered primary cell.

    The initial cell convention is fractional coordinates (-1/2,1/2) on
    periodic axes; open axes are unrestricted. No body may cross a seam.
    """
    system.assert_current()
    lattice.assert_current()
    if system.device!=lattice.vectors.device or system.dtype!=lattice.vectors.dtype:
        raise ValueError('Periodic lattice and surface system must share CUDA device and precision')
    margin=512*torch.finfo(system.dtype).eps
    for body in system.bodies:
        fractional=body.mesh.vertices@lattice.reciprocal.T/(2*math.pi)
        if bool((fractional.abs()>=.5-margin).any()):
            raise ValueError('Periodic bodies must lie strictly inside the centered cell without crossing a seam')


def assemble_periodic_system(system,exterior,config=IntegrationConfig(),*,kernel,check=None,
                             interior_kernels=None):
    """Add only the regular exterior lattice contribution to every body pair.

    The primary image uses the existing free-space joint Duffy implementation.
    Each interior self term remains its original homogeneous free-space term;
    no interior term exists between different bodies. Smooth periodic terms
    use regular product rules even on coincident primary panels. Their magnetic
    self block need not vanish, unlike the radial free-space primary image.
    Fixed quadrature and Ewald budgets remain explicit acceptance obligations.
    """
    from .physics import Medium
    from .tensor_medium import TensorMedium
    if not isinstance(exterior,Medium):
        raise TypeError('Ewald exterior assembly requires a sampled scalar exterior')
    if not isinstance(kernel,EwaldHelmholtzKernel):
        raise TypeError('Periodic scalar assembly requires an EwaldHelmholtzKernel')
    if check is not None and not isinstance(check,PeriodicAssemblyCheck):
        raise TypeError('Periodic assembly check requires PeriodicAssemblyCheck')
    validate_periodic_cell(system,kernel.lattice)
    kernel._guard(complex(exterior.k))
    tensor_interiors=any(isinstance(body.interior,TensorMedium) for body in system.bodies)
    if tensor_interiors:
        from .anisotropic_operator import assemble_generalized_system
        matrix,base=assemble_generalized_system(system,exterior,config,
                                                interior_kernels=interior_kernels)
    else:
        if interior_kernels:
            raise ValueError('Interior dyadic kernels require tensor body regions')
        matrix,base=assemble_system(system,exterior,config)
    correction,records=_assemble_regular(system,exterior,config,kernel)
    convergence=None if check is None else _check_regular(system,exterior,config,kernel,correction,check)
    return matrix+correction,{'formulation':'periodic_exterior_free_space_interiors_pmchwt_v1',
        'kernel':kernel.receipt(),'primary_image':base,'regular_blocks':records,
        'regular_testing_order':config.order,'cell':'centered_fractional_open_interval',
        'physical_qualification':False,'regular_quadrature_converged':False,
        'ewald_truncation_converged':False,'interiors_periodic':False,'regular_convergence':convergence}


def _assemble_regular(system,exterior,config,kernel):
    regular=_RegularKernel(kernel)
    dtype=torch.complex128 if system.dtype==torch.float64 else torch.complex64
    correction=torch.zeros((2*system.total_edges,2*system.total_edges),device=system.device,dtype=dtype)
    omega=2*math.pi*exterior.frequency
    from ..constants import EPSILON_0,MU_0
    records=[]
    for test in system.bodies:
        for source in system.bodies:
            ni,nj=test.basis.n_edges,source.basis.n_edges
            fi,fj=len(test.mesh.faces),len(source.mesh.faces)
            block=correction.new_zeros(2*ni,2*nj)
            for start in range(0,fi*fj,config.pair_batch):
                ids=torch.arange(start,min(start+config.pair_batch,fi*fj),device=system.device)
                rows,cols=ids//fj,ids%fj
                d,magnetic=_pair_integrals(test.basis,rows,cols,exterior,config.order,config.order,
                    False,kernel=regular,source_basis=source.basis,zero_planar_self=False)
                ri=test.basis.face_edges[rows,:,None].expand(-1,-1,3).reshape(-1)
                ci=source.basis.face_edges[cols,None,:].expand(-1,3,-1).reshape(-1)
                for i,j,value in ((0,0,magnetic),(1,1,magnetic),
                    (0,1,1j*omega*EPSILON_0*exterior.epsilon_r*d),
                    (1,0,-1j*omega*MU_0*exterior.mu_r*d)):
                    block.index_put_((ri+i*ni,ci+j*nj),value.reshape(-1),accumulate=True)
            i,j=test.region_id-1,source.region_id-1
            for u,test_slice in enumerate((system.J_slices[i],system.M_slices[i])):
                for v,source_slice in enumerate((system.J_slices[j],system.M_slices[j])):
                    correction[test_slice,source_slice]=block[u*ni:(u+1)*ni,v*nj:(v+1)*nj]
            records.append({'test_region':test.region_id,'source_region':source.region_id,
                'contribution':'periodic_exterior_regular_part','panel_pairs':fi*fj})
    return correction,records


@dataclass(frozen=True)
class PeriodicAssemblyCheck:
    """Independent doubling estimates in impedance-balanced L2 RWG coordinates."""
    relative_tolerance: float = 1e-7
    absolute_tolerance: float = 1e-12

    def __post_init__(self):
        if (not all(type(v) in (int,float) and math.isfinite(v) and v>=0
                    for v in (self.relative_tolerance,self.absolute_tolerance))
                or self.relative_tolerance+self.absolute_tolerance==0):
            raise ValueError('Periodic assembly tolerances must be finite, nonnegative and nonzero')


class PeriodicAssemblyConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Periodic assembly refinement failed: '+receipt['failure_reason'])


def _check_regular(system,exterior,config,kernel,baseline,policy):
    if not isinstance(policy,PeriodicAssemblyCheck):
        raise TypeError('Periodic assembly check requires PeriodicAssemblyCheck')
    gram=torch.block_diag(*(b.basis.mass_matrix() for b in system.bodies))
    factor=torch.linalg.cholesky(gram).to(baseline.dtype)
    metric=torch.block_diag(factor,factor)
    n=system.total_edges
    eta=abs(exterior.eta)
    scale=baseline.real.new_ones(2*n)
    scale[n:]=eta
    def norm(matrix):
        balanced=matrix.detach()*scale[None,:]/scale[:,None]
        left=torch.linalg.solve_triangular(metric,balanced,upper=False)
        normalized=torch.linalg.solve_triangular(metric,left.T,upper=False).T
        return float(torch.linalg.vector_norm(normalized))
    receipt={'metric':'impedance_balanced_RWG_L2_Frobenius_proxy',
        'relative_tolerance':policy.relative_tolerance,'absolute_tolerance':policy.absolute_tolerance,
        'checks':[],'converged':False,'certified_error_bound':False,'physical_qualification':False,
        'primary_image_reassembled':False,'failure_reason':None}
    for axis in ('regular_testing_order','real_shells','reciprocal_shells','quadrature_order'):
        changed_config=replace(config,order=2*config.order) if axis=='regular_testing_order' else config
        changed_kernel=kernel if axis=='regular_testing_order' else replace(kernel,**{axis:2*getattr(kernel,axis)})
        refined,_=_assemble_regular(system,exterior,changed_config,changed_kernel)
        difference=norm(refined-baseline)
        reference=norm(refined)
        threshold=policy.absolute_tolerance+policy.relative_tolerance*reference
        receipt['checks'].append({'axis':axis,'absolute_difference':difference,'reference_norm':reference,
            'threshold':threshold,'passed':math.isfinite(difference) and difference<=threshold,
            'initial':config.order if axis=='regular_testing_order' else getattr(kernel,axis),
            'refined':changed_config.order if axis=='regular_testing_order' else getattr(changed_kernel,axis)})
    receipt['converged']=all(row['passed'] for row in receipt['checks'])
    if not receipt['converged']:
        receipt['failure_reason']='independent_regular_refinement_failed'
        raise PeriodicAssemblyConvergenceError(receipt)
    return receipt


def validate_periodic_sources(sources,lattice,exterior):
    """Require every incident plane wave to occupy the same real Bloch sector."""
    from ..sources import PlaneWave
    sources=tuple(sources)
    lattice.assert_current()
    for source in sources:
        if not isinstance(source,PlaneWave):
            raise NotImplementedError('Periodic incident fields currently require explicit plane waves')
        direction=lattice.vectors.new_tensor(source.direction)
        translations=lattice.vectors@direction
        imaginary_phase=exterior.k.imag*translations
        tolerance=512*torch.finfo(direction.dtype).eps*max(1.,abs(exterior.k)*float(lattice.vectors.norm()))
        if bool((imaginary_phase.abs()>tolerance).any()):
            raise ValueError('Incident attenuation along a periodic axis is incompatible with a real Bloch vector')
        if not lattice.matches_incident(exterior.k.real*direction):
            raise ValueError('All incident waves must share the lattice Bloch vector modulo reciprocal vectors')
    return {'source_count':len(sources),'bloch_sector':lattice.receipt(),
            'incident_phase_match':True,'physical_qualification':False}


def build_periodic_rhs(handle,*,kernel):
    """Lift a Bloch-compatible incident carrier using local medium differences.

    Regular image potentials of the exact incident Cauchy trace vanish by the
    closed-surface representation identity. No RWG projection of the carrier
    or duplicate lattice forcing is added to the existing local lifting RHS.
    """
    from .kernels import FREE_SPACE
    from .lifting import build_lifted_rhs
    if not isinstance(kernel,EwaldHelmholtzKernel):
        raise TypeError('Periodic RHS requires EwaldHelmholtzKernel')
    validate_periodic_cell(handle.system,kernel.lattice)
    kernel._guard(complex(handle.exterior.k))
    source_receipt=validate_periodic_sources(handle.sources,kernel.lattice,handle.exterior)
    if any(local.kernel.receipt()!=FREE_SPACE.receipt() for local in handle.body_handles):
        raise ValueError('Periodic lifting requires free-space local body kernels')
    dtype=torch.complex128 if handle.system.dtype==torch.float64 else torch.complex64
    rhs=torch.empty(2*handle.system.total_edges,device=handle.system.device,dtype=dtype)
    records=[]
    for index,local in enumerate(handle.body_handles):
        values,record=build_lifted_rhs(local)
        size=local.basis.n_edges
        rhs[handle.system.J_slices[index]]=values[:size]
        rhs[handle.system.M_slices[index]]=values[size:]
        records.append({'region_id':index+1,**record})
    return rhs,{'formulation':'periodic_analytic_incident_lift_v1','kernel':kernel.receipt(),
        'sources':source_receipt,'bodies':records,'unknown_order':'all_J_scattered_then_all_M_scattered',
        'carrier_representation':'exact_analytic_incident_trace','physical_qualification':False}
