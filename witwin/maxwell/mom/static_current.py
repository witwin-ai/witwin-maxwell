"""Native volume potential testing for an electrically isolated DC current block.

The weak equation is integral grad(v).J = 0, with E0 = -grad(phi).
Its natural boundary condition is zero normal total current. The constant
potential gauge is removed explicitly. Exterior electrostatics, free charge
and magnetostatic fields remain separate coupling obligations.
"""
from dataclasses import dataclass
import math
import torch
from .volume_mesh import TetrahedralVolumeMesh
from .nonlinear import instantaneous_harmonic_sources
from .fixed_tangent_inverse import _FixedTangentInverse,_streamed_inverse_bounds


@dataclass(frozen=True)
class StaticCurrentSolution:
    potential: torch.Tensor
    electric: torch.Tensor
    mean_current: torch.Tensor
    receipt: dict


class StaticCurrentConvergenceError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('DC current balance failed: '+receipt['failure_reason'])


class NodalCurrentBalance:
    """Continuous linear potential and cellwise total-current weak testing.

    One connected body has one arbitrary potential constant. Internal solves
    pin vertex zero, then return the exact volume-mean-zero representative.
    This numerical gauge does not impose an electrode or body neutrality.
    """
    def __init__(self, volume):
        if not isinstance(volume,TetrahedralVolumeMesh):
            raise TypeError('DC current balance requires a tetrahedral volume mesh')
        volume.assert_current()
        self.volume = volume
        tetrahedra = volume.vertices[volume.cells]
        gradients = torch.linalg.inv(tetrahedra[:,1:]-tetrahedra[:,:1]).transpose(-1,-2)
        self.gradients = torch.cat((-gradients.sum(1,keepdim=True),gradients),1)
        self.nodal_volume = volume.vertices.new_zeros(len(volume.vertices)).index_add(
            0,volume.cells.flatten(),(volume.volumes[:,None].expand(-1,4)/4).flatten())
        self._identity = (id(volume),id(self.gradients),self.gradients._version,
                          id(self.nodal_volume),self.nodal_volume._version)

    def _assert_current(self):
        self.volume.assert_current()
        if self._identity != (id(self.volume),id(self.gradients),self.gradients._version,
                              id(self.nodal_volume),self.nodal_volume._version):
            raise RuntimeError('DC current discretization was mutated')

    @property
    def cell_dofs(self):
        return self.volume.cells

    @property
    def n_dofs(self):
        return len(self.volume.vertices)

    def _real(self,value,shape,name):
        if (not isinstance(value,torch.Tensor) or value.device != self.volume.vertices.device
                or value.dtype != self.volume.vertices.dtype or value.shape != shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name+' requires matching finite real CUDA coordinates')
        return value

    def _coefficient(self,value,name):
        if type(value) in (int,float) and math.isfinite(value):
            value = self.volume.vertices.new_tensor(value)
        if isinstance(value,torch.Tensor) and value.ndim == 0:
            value = value.expand(len(self.volume.cells))
        self._real(value,(len(self.volume.cells),),name)
        if bool((value < 0).any()):
            raise ValueError(name+' must be nonnegative')
        return value

    def _conductivity(self,value):
        """Scalar or symmetric PSD cell conductivity in S/m.

        Tensor symmetry admits floating-point roundoff and explicitly uses
        the symmetric part. Negative eigenvalues beyond roundoff reject; no
        eigenvalue clipping supplies missing coercivity to the solve.
        """
        if not isinstance(value,torch.Tensor) or value.ndim<2:
            return self._coefficient(value,'Conductivity')
        if value.shape==(3,3):value=value.expand(len(self.volume.cells),3,3)
        self._real(value,(len(self.volume.cells),3,3),'Conductivity tensor')
        scale=value.abs().amax(dim=(-2,-1))
        tolerance=64*torch.finfo(value.dtype).eps*scale
        if bool(((value-value.mT).abs().amax(dim=(-2,-1))>tolerance).any()):
            raise ValueError('Conductivity tensor must be symmetric')
        value=.5*(value+value.mT)
        if bool((torch.linalg.eigvalsh(value)[:,0]<-tolerance).any()):
            raise ValueError('Conductivity tensor must be positive semidefinite')
        return value

    def electric(self,potential):
        self._assert_current()
        self._real(potential,(self.n_dofs,),'DC potential')
        return -torch.einsum('ta,tai->ti',potential[self.cell_dofs],self.gradients)

    def weak_current(self,current):
        self._assert_current()
        self._real(current,(len(self.volume.cells),3),'Mean current')
        local = torch.einsum('tai,ti,t->ta',self.gradients,current,self.volume.volumes)
        return current.new_zeros(self.n_dofs).index_add(0,self.cell_dofs.flatten(),local.flatten())

    def _matrix(self,tangent):
        local = torch.einsum('tai,tij,tbj,t->tab',self.gradients,tangent,self.gradients,self.volume.volumes)
        cells = self.cell_dofs
        size = self.n_dofs
        indices = (cells[:,:,None]*size+cells[:,None,:]).flatten()
        return local.new_zeros(size*size).index_add(0,indices,local.flatten()).reshape(size,size)

    def tangent_action(self,tangent):
        """Freeze a cell current derivative on the vertex/face-zero gauge."""
        return StaticCurrentTangentAction(self,tangent)

    def solve(self,ac_electric,*,conductivity=0.,tpa_scale=0.,source_current=None,
              initial_potential=None,relative_tolerance=1e-9,absolute_tolerance=1e-14,
              max_iterations=30,max_backtracks=12,tangent_rule=None,tangent_column_batch=8):
        """Balance Ohmic plus time-averaged public cubic TPA current.

        Conductivity may be a scalar, cell scalar array, a symmetric PSD 3x3
        tensor or a cell tensor array; TPA remains the existing isotropic law.
        ac_electric is positive peak [H,cell,3], excluding DC. Its exact second
        and third moments give <|E0+a|^2(E0+a)> without aliasing. Polarization
        has zero mean time derivative and is absent from this current law.
        Native first derivatives differentiate the converged potential equation.
        A singular current tangent fails, including an unexcited pure dielectric;
        it must use the electrostatic displacement formulation instead.
        An explicit tangent_rule selects bounded GMRES with complete streamed
        inverse-residual admission; None retains the dense eigensystem check.
        Both choices retain the original full weak-current residual gate.
        """
        self._assert_current()
        if tangent_rule is not None:
            from .linalg import StoppingRule
            if (type(tangent_rule) is not StoppingRule or tangent_rule.method!='gmres'
                    or tangent_rule.preconditioner!='none' or tangent_rule.recycle_size or tangent_rule.fallback!='none'):
                raise ValueError('DC tangent solve requires unpreconditioned unrecycled GMRES without fallback')
        if type(tangent_column_batch) is not int or tangent_column_batch<1:
            raise ValueError('DC tangent column batch must be a positive integer')
        dtype = torch.complex128 if self.volume.vertices.dtype == torch.float64 else torch.complex64
        if (not isinstance(ac_electric,torch.Tensor) or ac_electric.device != self.volume.vertices.device
                or ac_electric.dtype != dtype or ac_electric.ndim != 3
                or ac_electric.shape[1:] != (len(self.volume.cells),3)
                or not bool(torch.isfinite(ac_electric).all())):
            raise ValueError('AC fields require matching finite CUDA positive peak [H,cell,3] values')
        for value,name in ((relative_tolerance,'relative tolerance'),(absolute_tolerance,'absolute tolerance')):
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError(name+' must be finite and nonnegative')
        for value,name in ((max_iterations,'iteration budget'),(max_backtracks,'backtrack budget')):
            if type(value) is not int or value < 0:
                raise ValueError(name+' must be a nonnegative integer')
        sigma,tpa = self._conductivity(conductivity),self._coefficient(tpa_scale,'TPA scale')
        zero = self.volume.vertices.new_zeros(len(self.volume.cells),3)
        source = zero if source_current is None else self._real(source_current,zero.shape,'Source current')
        covariance = .5*torch.einsum('hti,htj->tij',ac_electric,ac_electric.conj()).real
        third = instantaneous_harmonic_sources(torch.cat((zero[None].to(dtype),ac_electric)),
                                               tpa_scale=1.).dissipative_current[0].real
        eye = torch.eye(3,device=zero.device,dtype=zero.dtype)
        trace = covariance.diagonal(dim1=-2,dim2=-1).sum(-1)
        def law(electric):
            cubic = (electric.square().sum(-1)+trace)[:,None]*electric
            cubic = cubic+2*torch.einsum('tij,tj->ti',covariance,electric)+third
            ohmic=torch.einsum('tij,tj->ti',sigma,electric) if sigma.ndim==3 else sigma[:,None]*electric
            linear_tangent=sigma if sigma.ndim==3 else sigma[:,None,None]*eye
            current = ohmic+tpa[:,None]*cubic+source
            tangent = linear_tangent+tpa[:,None,None]*(
                (electric.square().sum(-1)+trace)[:,None,None]*eye
                +2*electric[:,:,None]*electric[:,None,:]+2*covariance)
            return current,tangent
        def potential(vector):
            return torch.cat((vector.new_zeros(1),vector))
        def equation(vector):
            current,_ = law(self.electric(potential(vector)))
            return -self.weak_current(current)[1:]
        size = self.n_dofs
        initial = self.volume.vertices.new_zeros(size) if initial_potential is None else self._real(
            initial_potential,(size,),'Initial potential')
        vector = (initial[1:]-initial[0]).detach().clone()
        norm = lambda value: float(torch.linalg.vector_norm(value.detach()))
        threshold = absolute_tolerance+relative_tolerance*norm(equation(vector))
        if not math.isfinite(threshold):
            raise ValueError('DC current residual threshold must be finite')
        receipt = {'formulation':'isolated_nodal_dc_current_balance_v1','converged':False,
            'boundary_condition':'zero_normal_total_current_weak', 'gauge':'volume_mean_zero',
            'threshold_A_l2':threshold,'checks':[], 'failure_reason':None,
            'exterior_electrostatic_coupling':False,'neutral_body_charge_enforced':False,
            'magnetostatic_fields':False,'physical_qualification':False,
            'gradient':'first_order_implicit_native_torch'}
        def fail(reason):
            receipt['failure_reason'] = reason
            raise StaticCurrentConvergenceError(receipt)
        if not bool(sigma.any()) and not bool(covariance.any()) and not bool(source.any()):
            # A nonzero starting guess must not disguise the singular tangent
            # at the exact unexcited cubic root.
            fail('singular_current_tangent')
        def matrix(value):
            _,tangent = law(self.electric(potential(value)))
            reduced = self._matrix(tangent)[1:,1:].detach()
            if not bool(torch.isfinite(reduced).all()):
                fail('nonfinite_current_tangent')
            eigenvalues = torch.linalg.eigvalsh(reduced)
            if bool(eigenvalues[0] <= 64*torch.finfo(reduced.dtype).eps*len(reduced)*eigenvalues[-1]):
                fail('singular_current_tangent')
            return reduced
        def linear_solve(value,rhs,phase):
            if tangent_rule is None:
                return torch.linalg.solve(matrix(value),rhs)
            _,tangent=law(self.electric(potential(value)))
            inverse=FixedStaticCurrentInverse(self.tangent_action(tangent),tangent_rule)
            bound=_streamed_inverse_bounds(inverse,column_batch=tangent_column_batch)
            receipt.setdefault('tangent_checks',[]).append({'phase':phase,'bound':bound})
            if not bound['admitted']:fail('current_tangent_bound_inconclusive')
            try:result=inverse.apply(rhs)
            except StaticTangentConvergenceError as error:
                receipt['tangent_checks'][-1]['solve']=error.receipt
                fail('current_tangent_residual_failed')
            receipt['tangent_checks'][-1]['solve']=inverse.receipt['last_forward']
            return result
        if tangent_rule is not None:
            from dataclasses import asdict
            receipt.update(linear_solver='streamed_inverse_checked_gmres',tangent_rule=asdict(tangent_rule),
                tangent_column_batch=tangent_column_batch,global_current_jacobian_stored=False,
                fast_work_qualified=False)
        for iteration in range(max_iterations+1):
            residual = equation(vector)
            magnitude = norm(residual)
            receipt['checks'].append({'iteration':iteration,'residual_A_l2':magnitude})
            if not math.isfinite(magnitude):
                fail('nonfinite_residual')
            if magnitude <= threshold:
                root = (vector-linear_solve(vector,residual,'implicit')).detach()
                corrected = root-linear_solve(root,equation(root),'implicit')
                phi = potential(corrected)
                phi = phi-(phi*self.nodal_volume).sum()/self.nodal_volume.sum()
                electric = self.electric(phi)
                current,_ = law(electric)
                # Include the eliminated equation in the final conservation check.
                final = norm(self.weak_current(current))
                if not math.isfinite(final) or final > threshold:
                    fail('final_current_residual_failed')
                receipt.update(converged=True,iterations=iteration,final_residual_A_l2=final)
                return StaticCurrentSolution(phi,electric,current,receipt)
            if iteration == max_iterations:
                fail('iteration_budget_exhausted')
            direction = linear_solve(vector,-residual.detach(),'newton')
            for backtrack in range(max_backtracks+1):
                step = 2.**(-backtrack)
                candidate = vector+step*direction
                candidate_norm = norm(equation(candidate))
                if math.isfinite(candidate_norm) and candidate_norm <= (1-1e-4*step)*magnitude:
                    vector = candidate.detach()
                    receipt['checks'][-1].update(step=step,backtracks=backtrack)
                    break
            else:
                fail('line_search_exhausted')


class FaceCurrentBalance(NodalCurrentBalance):
    """Conservative cell-current balance using Crouzeix-Raviart face potentials.

    On a tetrahedron the basis for the face opposite vertex a is 1-3*lambda_a.
    Its face mean is one on its own face and zero on the others. With constant
    cell currents, weak testing therefore enforces each physical face flux sum,
    including zero boundary flux. Potential is continuous in face mean only;
    pointwise potential/trace convergence remains a separate spatial obligation.
    """
    def __init__(self,volume):
        super().__init__(volume)
        local = volume.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
        faces = volume.cells[:,local].reshape(-1,3).sort(-1).values
        self.faces,inverse = torch.unique(faces,dim=0,return_inverse=True)
        self.face_dofs = inverse.reshape(-1,4)
        self.gradients = -3*self.gradients
        self.nodal_volume = volume.vertices.new_zeros(len(self.faces)).index_add(
            0,self.face_dofs.flatten(),(volume.volumes[:,None].expand(-1,4)/4).flatten())
        self._identity = (id(volume),id(self.gradients),self.gradients._version,
                          id(self.nodal_volume),self.nodal_volume._version)
        self._face_versions = tuple((id(t),t._version) for t in (self.faces,self.face_dofs))

    @property
    def cell_dofs(self):
        return self.face_dofs

    @property
    def n_dofs(self):
        return len(self.faces)

    def _assert_current(self):
        super()._assert_current()
        if self._face_versions != tuple((id(t),t._version) for t in (self.faces,self.face_dofs)):
            raise RuntimeError('DC face-current discretization was mutated')

    def solve(self,*args,**kwargs):
        solution = super().solve(*args,**kwargs)
        from .magnetostatic import cell_current_continuity
        diagnostic = cell_current_continuity(self.volume,solution.mean_current)
        solution.receipt.update(formulation='isolated_face_dc_current_balance_v1',
            potential_basis='crouzeix_raviart_face_mean',
            normal_current_jump_rms_A_per_m2=float(diagnostic['area_weighted_jump_rms_A_per_m2'].detach()),
            pointwise_potential_continuity=False,spatial_convergence_certified=False)
        return solution


class _StaticTangentProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._product(value,adjoint)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        product=ctx.owner.apply if ctx.adjoint else ctx.owner.adjoint
        return product(gradient),None,None


class StaticCurrentTangentAction:
    """Fixed reduced G^T W T G action with linear cell storage and work.

    T is a frozen real cell current derivative, not assumed symmetric or
    positive definite. Both nodal and face-mean potential bases use their native
    gradients and connectivity. Pinning degree zero removes only the constant
    gauge; nonsingularity and conditioning are separate solver obligations.
    Native vector derivatives use the transposed action, including higher RHS
    derivatives. Material/geometry/linearization derivatives are not exposed.
    An action alone does not establish nonsingularity or nonlinear convergence.
    """
    def __init__(self,block,tangent):
        if type(block) not in (NodalCurrentBalance,FaceCurrentBalance):
            raise TypeError('DC tangent requires an exact native current discretization')
        block._assert_current()
        block._real(tangent,(len(block.volume.cells),3,3),'DC cell tangent')
        self.block=block
        self.tangent=tangent.detach().clone()
        self.shape=(block.n_dofs-1,)*2
        self.device,self.dtype=tangent.device,tangent.dtype
        self._binding=(id(block),id(self.tangent),self.tangent._version,self.shape,self.device,self.dtype)

    def assert_current(self):
        if self._binding!=(id(self.block),id(self.tangent),self.tangent._version,self.shape,self.device,self.dtype):
            raise ValueError('DC tangent binding changed')
        self.block._assert_current()

    def _product(self,value,adjoint):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('DC tangent vectors require matching finite reduced real coefficients')
        if value.numel()==0:return value.clone()
        block=self.block
        pinned=torch.cat((value.new_zeros(*value.shape[:-1],1),value),dim=-1)
        gradient=torch.einsum('...ta,tai->...ti',pinned[...,block.cell_dofs],block.gradients)
        tensor=self.tangent.mT if adjoint else self.tangent
        current=torch.einsum('tij,...tj->...ti',tensor,gradient)
        local=torch.einsum('tai,...ti,t->...ta',block.gradients,current,block.volume.volumes)
        result=value.new_zeros(*value.shape[:-1],block.n_dofs).index_add(
            -1,block.cell_dofs.flatten(),local.flatten(-2))[...,1:]
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite DC tangent product')
        self.assert_current()
        return result

    def apply(self,value):return _StaticTangentProduct.apply(value,self,False)
    def adjoint(self,value):return _StaticTangentProduct.apply(value,self,True)

    @torch.no_grad()
    def diagonal(self):
        self.assert_current()
        block=self.block
        local=torch.einsum('tai,tij,taj,t->ta',block.gradients,self.tangent,block.gradients,block.volume.volumes)
        result=local.new_zeros(block.n_dofs).index_add(0,block.cell_dofs.flatten(),local.flatten())[1:]
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite DC tangent diagonal')
        self.assert_current()
        return result


class StaticTangentConvergenceError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('DC tangent solve failed its original RHS residual tolerance')
        self.receipt=receipt


class FixedStaticCurrentInverse(_FixedTangentInverse):
    """Fixed DC tangent GMRES with implicit vector derivatives and no fallback."""
    def __init__(self,tangent,rule,*,refinement_steps=2):
        if type(tangent) is not StaticCurrentTangentAction:
            raise TypeError('DC inverse requires an exact native cellwise tangent')
        super().__init__(tangent,rule,refinement_steps=refinement_steps,
            error_type=StaticTangentConvergenceError,
            version='fixed_static_current_inverse_v1',residual_metric='reduced_current_Euclidean')
