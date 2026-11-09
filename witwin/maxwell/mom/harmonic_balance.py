"""Native dense harmonic-balance algebra for assembled volume-field responses.

Response matrices must come from a separately validated coupled discretization.
Convergence here establishes only the retained-harmonic algebraic equations.
"""
from dataclasses import dataclass
import math
import torch
from .nonlinear import instantaneous_harmonic_sources


class HarmonicBalanceConvergenceError(RuntimeError):
    def __init__(self, receipt):
        super().__init__(f"Harmonic balance failed: {receipt['failure_reason']}")
        self.receipt = receipt


@dataclass(frozen=True)
class HarmonicBalanceSolution:
    electric: torch.Tensor
    sources: object
    receipt: dict


class DenseHarmonicBalance:
    """Cellwise E unknowns with explicit DC and positive-frequency linear maps.

    polarization_response maps C/m^2 to V/m; current_response maps A/m^2 to
    V/m. The positive-frequency maps must obey G_P=-i*n*omega*G_J. DC current
    response is explicit: supplying a zero map is a declaration by the caller,
    not evidence that a material's DC conduction problem has been solved.
    """
    def __init__(self, background, polarization_response, current_response, cell_volumes,
                 fundamental_frequency, *, chi2=0., chi3=0., tpa_scale=0.,dc_current_closure=None,
                 chi2_tensor=None,chi3_tensor=None):
        self._validate_background(background,cell_volumes,fundamental_frequency)
        size = background.shape[1]*3
        shape = (len(background),size,size)
        for value in (polarization_response,current_response):
            if (not isinstance(value,torch.Tensor) or value.shape != shape or value.device != background.device
                    or value.dtype != background.dtype or not bool(torch.isfinite(value).all())
                    or bool((value[0].imag != 0).any())):
                raise ValueError('HB response matrices require matching finite CUDA complex maps and real DC')
        omega = 2*math.pi*fundamental_frequency*torch.arange(1,len(background),device=background.device,dtype=background.real.dtype)
        expected = -1j*omega[:,None,None]*current_response[1:]
        error = torch.linalg.vector_norm((polarization_response[1:]-expected).flatten(1),dim=1)
        scale = torch.linalg.vector_norm(expected.flatten(1),dim=1)
        if bool((error > 128*torch.finfo(background.real.dtype).eps*scale).any()):
            raise ValueError('Positive-frequency P/J response maps violate J=-i*n*omega*P')
        self.polarization_response=polarization_response.clone()
        self.current_response=current_response.clone()
        self._initialize(background,cell_volumes,fundamental_frequency,chi2=chi2,chi3=chi3,
            tpa_scale=tpa_scale,dc_current_closure=dc_current_closure,
            chi2_tensor=chi2_tensor,chi3_tensor=chi3_tensor,
            dc_maps_empty=not bool(polarization_response[0].any() or current_response[0].any()))

    @staticmethod
    def _validate_background(background,cell_volumes,fundamental_frequency):
        if (not isinstance(background,torch.Tensor) or background.device.type != 'cuda'
                or background.dtype not in (torch.complex64,torch.complex128)
                or background.ndim != 3 or background.shape[-1] != 3 or len(background) < 2
                or not background.shape[1] or not bool(torch.isfinite(background).all())
                or bool((background[0].imag != 0).any())):
            raise ValueError('HB background requires finite CUDA complex [H+1,cells,3] with real DC')
        if (isinstance(fundamental_frequency,bool) or not math.isfinite(fundamental_frequency)
                or fundamental_frequency <= 0):
            raise ValueError('HB fundamental frequency must be finite and positive')
        if (not isinstance(cell_volumes,torch.Tensor) or cell_volumes.device != background.device
                or cell_volumes.dtype != background.real.dtype or cell_volumes.shape != (background.shape[1],)
                or not bool(torch.isfinite(cell_volumes).all()) or not bool((cell_volumes > 0).all())):
            raise ValueError('HB cell volumes require matching positive real CUDA weights')

    def _initialize(self,background,cell_volumes,fundamental_frequency,*,chi2,chi3,tpa_scale,
                    dc_current_closure,dc_maps_empty,chi2_tensor=None,chi3_tensor=None):
        self.background,self.cell_volumes=background.clone(),cell_volumes.clone()
        self.frequency = float(fundamental_frequency)
        self.dc_current_closure = dc_current_closure
        if dc_current_closure is not None:
            from .harmonic_current import HarmonicCurrentClosure
            from .harmonic_mixed import MixedHarmonicClosure
            from .harmonic_dual_current import DualHarmonicCurrentClosure
            if not isinstance(dc_current_closure,(HarmonicCurrentClosure,MixedHarmonicClosure,DualHarmonicCurrentClosure)) or sum(dc_current_closure.counts) != background.shape[1]:
                raise ValueError('HB current closure must match the volume discretization')
            dc_current_closure.assert_current()
            actual_volumes = torch.cat([volume.volumes for volume in dc_current_closure.volumes])
            if (actual_volumes.device != cell_volumes.device or actual_volumes.dtype != cell_volumes.dtype
                    or not torch.equal(actual_volumes,cell_volumes)):
                raise ValueError('HB current closure must preserve the actual volume weights')
            if bool(background[0].any()) or not dc_maps_empty:
                raise ValueError('Current-balance DC closure requires empty DC background and response maps')
        self.chi2,self.chi3,self.tpa_scale = (value.clone() if isinstance(value,torch.Tensor) else value
                                            for value in (chi2,chi3,tpa_scale))
        self.chi2_tensor,self.chi3_tensor=(v.clone() if isinstance(v,torch.Tensor) else v
                                            for v in (chi2_tensor,chi3_tensor))
        # Validate coefficients with the same native constitutive contract.
        self.sources(background)
        self._versions = self._state()

    def _state(self):
        values = (self.background,self.polarization_response,self.current_response,self.cell_volumes,
                  self.chi2,self.chi3,self.tpa_scale,self.chi2_tensor,self.chi3_tensor)
        return (self.frequency,id(self.dc_current_closure))+tuple((id(v),v._version) if isinstance(v,torch.Tensor) else v for v in values)

    def _assert_current(self):
        if self._state() != self._versions:
            raise RuntimeError('HB response or constitutive state was mutated')
        if self.dc_current_closure is not None:
            self.dc_current_closure.assert_current()

    def sources(self, electric):
        return instantaneous_harmonic_sources(electric,chi2=self.chi2,chi3=self.chi3,tpa_scale=self.tpa_scale,
            chi2_tensor=self.chi2_tensor,chi3_tensor=self.chi3_tensor)

    def residual(self, electric, *, pump_scale=1.):
        self._assert_current()
        if electric.shape != self.background.shape or electric.dtype != self.background.dtype or electric.device != self.background.device:
            raise ValueError('HB electric fields must match the response discretization')
        sources = self.sources(electric)
        retained = len(electric)
        polarization = sources.polarization[:retained].flatten(1)
        current = sources.dissipative_current[:retained].flatten(1)
        response = self._response(polarization,current).reshape_as(electric)
        residual = electric-pump_scale*self.background-response
        if self.dc_current_closure is not None:
            dc = self.dc_current_closure.electric(electric[1:],self.tpa_scale,
                                                 polarization=sources.polarization[0].real)
            residual = torch.cat(((electric[0]-dc)[None],residual[1:]))
        return residual

    def _response(self,polarization,current):
        return (torch.einsum('hij,hj->hi',self.polarization_response,polarization)
                +torch.einsum('hij,hj->hi',self.current_response,current))

    @staticmethod
    def _pack(electric):
        return torch.cat((electric[0].real.flatten(),electric[1:].real.flatten(),electric[1:].imag.flatten()))

    def _unpack(self, vector):
        count = self.background.shape[1]*3
        dc = vector[:count].reshape(1,-1,3).to(self.background.dtype)
        ac_count = (len(self.background)-1)*count
        real = vector[count:count+ac_count].reshape(len(self.background)-1,-1,3)
        imaginary = vector[count+ac_count:].reshape_as(real)
        ac = torch.complex(real,imaginary)
        return torch.cat((dc,ac))

    def solve(self, initial=None, *, pump_scale=1., method='newton', relative_tolerance=1e-9,
              absolute_tolerance=1e-10, max_iterations=30, damping=1., max_backtracks=12,
              tangent_rule=None,implicit_solver='dense',tangent_column_batch=8):
        self._assert_current()
        if method not in ('newton','picard','newton_krylov'):
            raise ValueError('HB method must be newton, picard or newton_krylov')
        if method=='newton_krylov':
            from .linalg import StoppingRule
            if tangent_rule is None:
                tangent_rule=StoppingRule(preconditioner='none',rtol=1e-10,atol=1e-13)
            if (type(tangent_rule) is not StoppingRule or tangent_rule.method!='gmres'
                    or tangent_rule.preconditioner!='none' or tangent_rule.recycle_size
                    or tangent_rule.fallback!='none'):
                raise ValueError('HB Newton-Krylov requires unpreconditioned unrecycled GMRES without fallback')
        elif tangent_rule is not None:
            raise ValueError('HB tangent rule requires newton_krylov')
        if implicit_solver not in ('dense','streamed_krylov','streamed_inverse_krylov') or (implicit_solver!='dense' and method!='newton_krylov'):
            raise ValueError('HB streamed implicit solver requires newton_krylov')
        if type(tangent_column_batch) is not int or tangent_column_batch<1:
            raise ValueError('HB tangent column batch must be a positive integer')
        if (not all(math.isfinite(v) and v >= 0 for v in (pump_scale,relative_tolerance,absolute_tolerance))
                or relative_tolerance+absolute_tolerance == 0 or not math.isfinite(damping) or not 0 < damping <= 1
                or type(max_iterations) is not int or max_iterations < 0
                or type(max_backtracks) is not int or max_backtracks < 0):
            raise ValueError('Invalid HB continuation or iteration settings')
        field = pump_scale*self.background if initial is None else initial
        self.residual(field,pump_scale=pump_scale)
        vector = self._pack(field).detach()
        norm = lambda value: float(torch.linalg.vector_norm(value.detach()))
        threshold = absolute_tolerance+relative_tolerance*norm(self._pack(pump_scale*self.background))
        if not math.isfinite(threshold):
            raise ValueError('HB residual threshold must be finite')
        receipt = {'method': method, 'pump_scale': pump_scale, 'checks': [], 'converged': False,
            'physical_qualification': False, 'harmonic_truncation_converged': False,
            'stability_assessed': False, 'threshold_V_per_m_l2': threshold,
            'initialization': 'linear_background' if initial is None else 'explicit_state',
            'gradient': 'first_order_implicit_native_torch', 'failure_reason': None}
        if method=='newton_krylov':
            from dataclasses import asdict
            receipt.update(newton_direction='matrix_free_gmres',tangent_rule=asdict(tangent_rule),
                implicit_root_solver={'dense':'dense_svd_checked','streamed_krylov':'streamed_dominance_checked_gmres',
                    'streamed_inverse_krylov':'streamed_inverse_residual_checked_gmres'}[implicit_solver],
                fully_matrix_free=False,global_hb_jacobian_stored=implicit_solver=='dense')
        def equation(value):
            return self._pack(self.residual(self._unpack(value),pump_scale=pump_scale))
        def derivative(value, phase):
            jacobian = torch.autograd.functional.jacobian(equation,value,vectorize=False).detach()
            if not bool(torch.isfinite(jacobian).all()):
                receipt['failure_reason'] = f'nonfinite_{phase}_jacobian'
                raise HarmonicBalanceConvergenceError(receipt)
            try:
                singular_values = torch.linalg.svdvals(jacobian)
            except torch.linalg.LinAlgError as error:
                receipt['failure_reason'] = f'{phase}_jacobian_svd_failed'
                raise HarmonicBalanceConvergenceError(receipt) from error
            cutoff = 64*torch.finfo(value.dtype).eps*len(value)*singular_values[0]
            receipt['last_jacobian'] = {'phase': phase,
                'smallest_singular_value': float(singular_values[-1]),
                'largest_singular_value': float(singular_values[0]),
                'numerical_singularity_cutoff': float(cutoff)}
            if not bool(torch.isfinite(singular_values).all()) or bool(singular_values[-1] <= cutoff):
                receipt['failure_reason'] = f'singular_{phase}_jacobian'
                raise HarmonicBalanceConvergenceError(receipt)
            return jacobian
        def implicit_correction(value,residual):
            if implicit_solver=='dense':
                return torch.linalg.solve(derivative(value,'implicit'),residual)
            from .harmonic_jacobian_action import (HarmonicJacobianAction,FixedHarmonicTangentInverse,
                HarmonicTangentConvergenceError,streamed_tangent_bounds,streamed_inverse_tangent_bounds)
            tangent=HarmonicJacobianAction(self,self._unpack(value),pump_scale=pump_scale)
            bound=(streamed_inverse_tangent_bounds(tangent,tangent_rule,column_batch=tangent_column_batch)
                if implicit_solver=='streamed_inverse_krylov'
                else streamed_tangent_bounds(tangent,column_batch=tangent_column_batch))
            receipt.setdefault('implicit_tangent_checks',[]).append(bound)
            if not bound['admitted']:
                receipt['failure_reason']='implicit_tangent_bound_inconclusive'
                raise HarmonicBalanceConvergenceError(receipt)
            inverse=FixedHarmonicTangentInverse(tangent,tangent_rule)
            try:
                correction=inverse.apply(residual)
            except HarmonicTangentConvergenceError as error:
                receipt.update(failure_reason='implicit_krylov_residual_failed',implicit_solve=error.receipt)
                raise HarmonicBalanceConvergenceError(receipt) from error
            bound['solve']=inverse.receipt['last_forward']
            return correction
        for iteration in range(max_iterations+1):
            residual = equation(vector)
            magnitude = norm(residual)
            receipt['checks'].append({'iteration': iteration,'residual_V_per_m_l2': magnitude})
            if not math.isfinite(magnitude):
                receipt['failure_reason'] = 'nonfinite_residual'
                raise HarmonicBalanceConvergenceError(receipt)
            if magnitude <= threshold:
                # Differentiate the converged equation, not the iteration history.
                try:
                    corrected = vector-implicit_correction(vector,residual)
                except torch.linalg.LinAlgError as error:
                    receipt['failure_reason'] = 'singular_implicit_jacobian'
                    raise HarmonicBalanceConvergenceError(receipt) from error
                # Recheck at the corrected root: the pre-correction iterate
                # can be nonsingular even when the limiting root is degenerate.
                root = corrected.detach()
                corrected = root-implicit_correction(root,equation(root))
                final_norm = norm(equation(corrected))
                if not math.isfinite(final_norm) or final_norm > threshold:
                    receipt.update(failure_reason='implicit_correction_residual_failed',final_residual=final_norm)
                    raise HarmonicBalanceConvergenceError(receipt)
                electric = self._unpack(corrected)
                sources = self.sources(electric)
                retained = len(electric)
                receipt.update(converged=True,iterations=iteration,final_residual=final_norm,
                    generated_highest_harmonic=len(sources.polarization)-1,
                    omitted_polarization_l2_C_per_m2=norm(sources.polarization[retained:]),
                    omitted_dissipative_current_l2_A_per_m2=norm(sources.dissipative_current[retained:]),
                    volume_dissipation_W=float((sources.average_dissipation*self.cell_volumes).sum().detach()))
                from .nonlinear import harmonic_power_ledger
                power = harmonic_power_ledger(electric,sources,self.frequency,self.cell_volumes)
                receipt['nonlinear_power'] = {key:value.detach().cpu().tolist() if isinstance(value,torch.Tensor)
                                              else value for key,value in power.items()}
                if self.dc_current_closure is not None:
                    dc_states = self.dc_current_closure.solve(electric[1:],self.tpa_scale)
                    receipt['dc_current_closure'] = ([dc_states['nonlinear_receipt'],dc_states['receipt']] if isinstance(dc_states,dict) else [state.receipt for state in dc_states])
                    receipt['dc_formulation'] = self.dc_current_closure.formulation
                return HarmonicBalanceSolution(electric,sources,receipt)
            if iteration == max_iterations:
                receipt['failure_reason'] = 'iteration_budget_exhausted'
                raise HarmonicBalanceConvergenceError(receipt)
            if method == 'newton_krylov':
                from .harmonic_jacobian_action import (HarmonicJacobianAction,
                    FixedHarmonicTangentInverse,HarmonicTangentConvergenceError)
                tangent=HarmonicJacobianAction(self,self._unpack(vector),pump_scale=pump_scale)
                inverse=FixedHarmonicTangentInverse(tangent,tangent_rule)
                try:
                    direction=inverse.apply(-residual.detach())
                except HarmonicTangentConvergenceError as error:
                    receipt['checks'][-1]['tangent_solve']=error.receipt
                    receipt['failure_reason']='newton_krylov_residual_failed'
                    raise HarmonicBalanceConvergenceError(receipt) from error
                receipt['checks'][-1]['tangent_solve']=inverse.receipt['last_forward']
            elif method == 'newton':
                jacobian = derivative(vector,'newton')
                try:
                    direction = torch.linalg.solve(jacobian,-residual.detach())
                except torch.linalg.LinAlgError as error:
                    receipt['failure_reason'] = 'singular_newton_jacobian'
                    raise HarmonicBalanceConvergenceError(receipt) from error
            else:
                direction = -residual.detach()
            accepted = False
            for backtrack in range(max_backtracks+1):
                step = damping*2.**(-backtrack)
                candidate = vector+step*direction
                candidate_norm = norm(equation(candidate)) if bool(torch.isfinite(candidate).all()) else math.inf
                if math.isfinite(candidate_norm) and candidate_norm <= (1-1e-4*step)*magnitude:
                    vector = candidate.detach()
                    receipt['checks'][-1].update(damping=step,backtracks=backtrack)
                    accepted = True
                    break
            if not accepted:
                receipt['failure_reason'] = 'line_search_exhausted'
                raise HarmonicBalanceConvergenceError(receipt)

    def solve_ramp(self, pump_scales, *, branch_id, initial=None, **settings):
        """Follow an explicitly named warm-start path; no silent branch restarts."""
        from .static_current import StaticCurrentConvergenceError
        scales = tuple(pump_scales)
        if not scales or not isinstance(branch_id,str) or not branch_id.strip():
            raise ValueError('Continuation requires pump steps and an explicit branch identifier')
        solutions, state = [], initial
        for index,scale in enumerate(scales):
            try:
                solution = self.solve(state,pump_scale=scale,**settings)
            except (HarmonicBalanceConvergenceError,StaticCurrentConvergenceError) as error:
                error.receipt.update(branch_id=branch_id,failed_step=index,
                    completed_steps=[item.receipt for item in solutions])
                raise
            solution.receipt.update(branch_id=branch_id,continuation_step=index,
                                    branch_selection='explicit_initial_state_then_warm_start')
            solutions.append(solution)
            state = solution.electric.detach()
        return tuple(solutions)
