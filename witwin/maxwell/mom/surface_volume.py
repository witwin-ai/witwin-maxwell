"""Regional volume-current testing for the existing lifted surface equations.

Only induced-volume response building blocks are defined here. This is not a
new public prescribed interior-source API or a nonlinear field solve.
"""
import torch
from .operator import local_geometry
from .quadrature import triangle_rule
from .volume_mesh import BoundVolumeRegion
from .vie_operator import evaluate_cell_current_fields, cell_green_potentials
from .fields import FieldQueryConfig


def build_volume_source_rhs(surface_operator, bindings, currents, *, order=3, query_config=None):
    """Test each owning region's volume fields as [+<f,Hv>, -<f,Ev>].

    The exterior multi-body operator transports interaction to other bodies.
    Regional volume fields contribute only to their owning boundary equation.
    Uniform linear material response is already inside the regional Green and
    PMCHWT operator; no extra epsilon or mu contrast is multiplied here.
    """
    system = surface_operator.system
    system.assert_current()
    bindings, currents = tuple(bindings), tuple(currents)
    if len(bindings) != len(currents):
        raise ValueError('Each volume binding requires exactly one current array')
    if type(order) is not int or order < 2:
        raise ValueError('Volume-to-surface testing requires order >= 2')
    config = query_config or FieldQueryConfig()
    if not isinstance(config, FieldQueryConfig) or not config.adaptive:
        raise ValueError('Volume-to-surface testing requires adaptive field queries')
    output = torch.zeros(surface_operator.shape[0], device=surface_operator.device, dtype=surface_operator.dtype)
    records, used = [], set()
    for binding, current in zip(bindings, currents):
        if not isinstance(binding, BoundVolumeRegion) or binding.surface_system is not system:
            raise ValueError('Volume source must bind the exact owning surface system')
        binding.assert_current()
        index = binding.volume.region_id-1
        if index in used:
            raise ValueError('A body may have only one conforming volume discretization')
        used.add(index)
        body = system.bodies[index]
        basis = body.basis
        edges, free, slopes = local_geometry(basis)
        points, weights = triangle_rule(body.mesh.triangles, order)
        (electric, magnetic), receipt = evaluate_cell_current_fields(binding.volume, current,
            points.reshape(-1,3), body.interior, order=order, config=config, principal_value=True)
        values = (slopes[:, None, :, None]*(points[:, :, None]-free[:, None])).to(surface_operator.dtype)
        magnetic = magnetic.reshape_as(points)
        n = basis.n_edges
        local_rhs = output.new_zeros(2*n)
        tested_h = torch.einsum('tqli,tqi,tq->tl', values, magnetic, weights.to(surface_operator.dtype))
        # Transfer the electric potential's tangential derivative to RWG
        # divergence over its complete support. Edge terms cancel between the
        # two incident triangles; no hypersingular Hessian is tested directly.
        potential, gradient, _ = cell_green_potentials(binding.volume,points.reshape(-1,3),
            body.interior.k,order=receipt['final_order'],principal_value=True)
        vector = torch.einsum('qt,ti->qi',potential,current).reshape_as(points)
        scalar = torch.einsum('qti,ti->q',gradient,current).reshape(points.shape[:2])
        tested_e = 1j*body.interior.omega*body.interior.mu*torch.einsum(
            'tqli,tqi,tq->tl',values,vector,weights.to(surface_operator.dtype))
        tested_e -= 1j/(body.interior.omega*body.interior.epsilon)*torch.einsum(
            'tl,tq,tq->tl',(2*slopes).to(surface_operator.dtype),scalar,weights.to(surface_operator.dtype))
        local_rhs.index_add_(0,edges.flatten(),tested_h.flatten())
        local_rhs.index_add_(0,edges.flatten()+n,-tested_e.flatten())
        output[system.J_slices[index]] = local_rhs[:n]
        output[system.M_slices[index]] = local_rhs[n:]
        records.append({'region_id': body.region_id, 'field_query': receipt,
                        'surface_testing_order': order, 'global_rhs_error_certified': False})
    return output, {'representation': 'coupled_sie_vie_linear', 'source_role': 'induced_volume_current',
        'linear_background': 'regional_green_and_surface_operator_only', 'linear_contrast_added': False,
        'body_sources': records, 'physical_qualification': False, 'surface_testing_error_estimated': False,
        'electric_testing': 'full_rwg_support_divergence_volume_potential_v1'}


class VolumeFieldResponse:
    """Validated decode-only view of already recovered coupled unknowns."""
    def __init__(self,operator,bindings,currents,coefficients,*,order=3,query_config=None):
        operator.system.assert_current()
        if type(order) is not int or order < 2:
            raise ValueError('Volume field order must be an integer >= 2')
        bindings,currents = tuple(bindings),tuple(currents)
        if len(bindings) != len(currents):
            raise ValueError('Each field binding requires one current array')
        if (coefficients.shape != (operator.shape[0],) or coefficients.device != operator.device
                or coefficients.dtype != operator.dtype or not bool(torch.isfinite(coefficients).all())):
            raise ValueError('Coupled coefficients must match the surface operator')
        used = set()
        for binding,current in zip(bindings,currents):
            if not isinstance(binding,BoundVolumeRegion) or binding.surface_system is not operator.system:
                raise ValueError('Coupled fields require exact owning-system bindings')
            binding.assert_current()
            if binding.volume.region_id in used:
                raise ValueError('Duplicate coupled field volume region')
            used.add(binding.volume.region_id)
            if (current.shape != (len(binding.volume.cells),3) or current.device != operator.device
                    or current.dtype != operator.dtype or not bool(torch.isfinite(current).all())):
                raise ValueError('Coupled current arrays must match the volume discretization')
        self.operator,self.bindings = operator,bindings
        self.currents = tuple(value.clone() for value in currents)
        self.coefficients = coefficients.clone()
        self.order,self.config = order,query_config or FieldQueryConfig()
        if not isinstance(self.config,FieldQueryConfig):
            raise TypeError('Coupled fields require FieldQueryConfig')
        self._versions = self._field_state()

    def _field_state(self):
        from .tensor_medium import TensorMedium
        from .dyadic_green import DyadicGreenKernel
        def tensor_binding(handle):
            if not isinstance(handle.interior,TensorMedium):return None
            kernel=getattr(handle,'tensor_kernel',None)
            if type(kernel) is not DyadicGreenKernel or kernel.medium is not handle.interior:
                raise ValueError('Tensor volume queries require the owning regional kernel')
            handle.interior.assert_current()
            return (id(kernel),id(kernel.medium),kernel.integration,kernel.cache_max_bytes,
                    id(kernel.mode_cache),kernel._isotropic)
        operator=self.operator
        matrix=getattr(operator,'matrix',None)
        values=(*self.currents,self.coefficients)+(() if matrix is None else (matrix,))
        physics=tuple((id(h),id(h.exterior),h.integration,id(h.kernel),id(h.incident_carrier),
            id(h.basis) if local else None,id(h.interior) if local else None,
            tensor_binding(h) if local else None)
            for local,h in enumerate((operator,*operator.body_handles)))
        return (id(operator.system),physics,tuple((id(v),v._version) for v in values))

    def fields(self, points, region_id, *, query_config=None):
        """Decode complete off-boundary regional fields, preserving current gradients."""
        from .lifting_fields import decode_system
        operator = self.operator
        operator.system.assert_current()
        region_id = operator.system.region(region_id)
        if self._field_state() != self._versions:
            raise RuntimeError('Volume response state was mutated')
        for binding in self.bindings:
            binding.assert_current()
        config = query_config or self.config
        if not isinstance(config, FieldQueryConfig):
            raise TypeError('Coupled field settings require FieldQueryConfig')
        import copy
        from dataclasses import replace
        order = max(self.order, operator.integration.order, operator.integration.singular_order)
        receipt = {'region_id': region_id, 'complete_field_error_certified': False,
            'physical_qualification': False, 'checks': [], 'converged': None,
            'relative_tolerance': config.relative_tolerance,
            'absolute_tolerance_V_per_m': config.absolute_tolerance,
            'estimator': 'complete_surface_plus_volume_E_H_successive_order_differences'
                         if config.adaptive else 'not_requested'}
        self.last_field_receipt = receipt

        def evaluate(refinement_order=None):
            handle = operator
            volume_order = self.order
            if refinement_order is not None:
                handle = copy.copy(operator)
                handle.body_handles = tuple(copy.copy(h) for h in operator.body_handles)
                for local in handle.body_handles:
                    local.integration = replace(local.integration, order=refinement_order,
                                                singular_order=refinement_order,
                                                near_max_order=max(local.integration.near_max_order,
                                                                   refinement_order+2))
                volume_order = refinement_order
            fixed = replace(config, adaptive=False)
            result = decode_system(handle, self.coefficients, points, region_id, 'total', query_config=fixed)
            volume_log = None
            for binding, current in zip(self.bindings, self.currents):
                if binding.volume.region_id == region_id:
                    from .tensor_medium import TensorMedium
                    medium=operator.system.bodies[region_id-1].interior
                    if isinstance(medium,TensorMedium):
                        from .tensor_volume import cell_tensor_current_fields
                        volume,volume_log=cell_tensor_current_fields(binding.volume,current,points,
                            handle.body_handles[region_id-1].tensor_kernel,order=volume_order,
                            radial_order=volume_order)
                    else:
                        volume, volume_log = evaluate_cell_current_fields(binding.volume, current, points,
                            medium, order=volume_order, config=fixed)
                    result = tuple(a+b for a,b in zip(result,volume))
            receipt.update(surface=handle.last_field_query_log, volume=volume_log,
                           final_order=refinement_order)
            if any(not bool(torch.isfinite(value).all()) for value in result):
                receipt.update(converged=False, failure_reason='nonfinite_coupled_field')
                raise CoupledFieldConvergenceError(receipt)
            return result

        if not config.adaptive:
            return evaluate()
        receipt['converged'] = False
        if order >= config.max_order:
            receipt['failure_reason'] = 'initial_order_leaves_no_refinement'
            raise CoupledFieldConvergenceError(receipt)
        previous = tuple(value.detach() for value in evaluate(order))
        medium = operator.exterior if region_id == 0 else operator.system.bodies[region_id-1].interior
        from .tensor_medium import TensorMedium
        from ..constants import MU_0,EPSILON_0
        tensor_metric=isinstance(medium,TensorMedium)
        impedance=(MU_0/EPSILON_0)**.5 if tensor_metric else abs(medium.eta)
        receipt.update(field_error_metric='separate_E_and_vacuum_impedance_H' if tensor_metric
                       else 'separate_E_and_regional_impedance_H',
                       absolute_tolerance_H_A_per_m=config.absolute_tolerance/impedance)
        def field_norm(value):
            # Scale before squaring so finite large fields do not create an
            # artificial inf <= inf convergence decision.
            scale=value.abs().amax(dim=-1,keepdim=True)
            divisor=torch.where(scale==0,torch.ones_like(scale),scale)
            return torch.linalg.vector_norm(value/divisor,dim=-1)*scale.squeeze(-1)
        while order < config.max_order:
            order = min(2*order, config.max_order)
            result = evaluate(order)
            check, passed = {'order': order}, True
            for name, value, old, absolute in zip(('E','H'), result, previous,
                    (config.absolute_tolerance, config.absolute_tolerance/impedance)):
                error = field_norm(value-old)
                magnitude = field_norm(value)
                threshold=absolute+config.relative_tolerance*magnitude
                finite=torch.isfinite(error)&torch.isfinite(magnitude)&torch.isfinite(threshold)
                if not bool(finite.all()):
                    receipt.update(converged=False,failure_reason='nonfinite_coupled_error_metric',
                                   failed_field=name,failed_order=order)
                    raise CoupledFieldConvergenceError(receipt)
                success = bool((error <= threshold).all())
                passed = passed and success
                check[name] = {'passed': success,
                    'maximum_absolute_difference': float(error.max().detach()) if error.numel() else 0.,
                    'maximum_relative_difference': float((error/magnitude.clamp_min(
                        torch.finfo(magnitude.dtype).tiny)).max().detach()) if error.numel() else 0.}
            receipt['checks'].append(check)
            if passed:
                receipt.update(converged=True, failure_reason=None)
                return result
            previous = tuple(value.detach() for value in result)
        receipt['failure_reason'] = 'coupled_field_order_budget_exhausted'
        raise CoupledFieldConvergenceError(receipt)


class DenseVolumeResponse(VolumeFieldResponse):
    """Private differentiable affine response to induced regional currents.

    The dense matrix is snapshotted under its existing read lease. Native Torch
    LU and autograd implement the solve; no second boundary formulation is
    assembled. This candidate has no complete coupled error certificate.
    """
    def __init__(self, surface_operator, bindings, currents, *, order=3, query_config=None,
                 testing_max_order=32, testing_relative_tolerance=1e-4,
                 testing_absolute_tolerance=1e-12):
        from .runtime import SystemOperator
        if not isinstance(surface_operator, SystemOperator):
            raise TypeError('Dense volume response requires the existing SystemOperator')
        self.operator = surface_operator
        self.bindings = tuple(bindings)
        self.currents = tuple(current.clone() for current in currents)
        self.order = order
        self.config = query_config or FieldQueryConfig()
        addition, testing = adaptive_volume_source_rhs(surface_operator, self.bindings,
            self.currents, order=order, max_order=testing_max_order,
            relative_tolerance=testing_relative_tolerance,
            absolute_tolerance=testing_absolute_tolerance, query_config=self.config)
        receipt = {**testing['last_source_receipt'], 'outer_testing': testing,
                   'surface_testing_error_estimated': True}
        with surface_operator.read_access():
            surface_operator.refresh()
            matrix = surface_operator.matrix.clone()
            rhs = surface_operator.build_rhs()+addition
            self.coefficients = torch.linalg.solve(matrix, rhs)
            residual = rhs-matrix@self.coefficients
        norm_b = float(torch.linalg.vector_norm(rhs.detach()))
        norm_r = float(torch.linalg.vector_norm(residual.detach()))
        threshold = 512*torch.finfo(matrix.real.dtype).eps*norm_b
        if not bool(torch.isfinite(self.coefficients).all()) or norm_r > threshold:
            raise RuntimeError('Dense volume response failed its true surface residual check')
        self.receipt = {**receipt, 'surface_solver': 'native_torch_dense_autograd',
            'surface_absolute_residual': norm_r, 'surface_residual_threshold': threshold,
            'surface_relative_residual': norm_r/norm_b if norm_b else None,
            'complete_field_error_certified': False}
        self._versions = self._field_state()



class CoupledFieldConvergenceError(RuntimeError):
    def __init__(self, receipt):
        super().__init__('Complete surface-volume field quadrature did not converge')
        self.receipt = receipt


class VolumeTestingConvergenceError(RuntimeError):
    def __init__(self, receipt):
        super().__init__(f"Volume-to-surface testing exhausted its quadrature order budget: {receipt['checks']}")
        self.receipt = receipt


def adaptive_volume_source_rhs(surface_operator, bindings, currents, *, order=2,
                               max_order=32, relative_tolerance=1e-4,
                               absolute_tolerance=1e-12, query_config=None):
    """Refine outer RWG testing in the existing impedance-balanced dual proxy.

    Absolute tolerance is expressed in that dual norm, not pointwise V/m.
    Inner source-field tolerances are specified separately with query_config.
    Successive-order agreement is an estimator, not an error bound.
    """
    import math
    from .linalg import apply_test_riesz_inverse
    if type(order) is not int or type(max_order) is not int or order < 2 or max_order <= order:
        raise ValueError('Outer testing requires integer 2 <= order < max_order')
    if (not all(math.isfinite(v) and v >= 0 for v in (relative_tolerance,absolute_tolerance))
            or relative_tolerance+absolute_tolerance == 0):
        raise ValueError('Outer testing tolerances must be finite, nonnegative and not both zero')
    bindings, currents = tuple(bindings), tuple(currents)
    receipt = {'estimator': 'successive_outer_order_impedance_balanced_L2_dual_proxy',
        'relative_tolerance': relative_tolerance, 'absolute_tolerance_dual_proxy': absolute_tolerance,
        'checks': [], 'converged': False, 'global_rhs_error_certified': False,
        'physical_qualification': False}

    def norm(value):
        value = value.detach()
        inverse = apply_test_riesz_inverse(surface_operator,value)
        squared = torch.vdot(value,inverse).real
        return float(squared.clamp_min(0).sqrt())

    previous, _ = build_volume_source_rhs(surface_operator,bindings,currents,
        order=order,query_config=query_config)
    previous = previous.detach()
    while order < max_order:
        order = min(2*order,max_order)
        result, local_log = build_volume_source_rhs(surface_operator,bindings,currents,
            order=order,query_config=query_config)
        error, magnitude = norm(result-previous), norm(result)
        threshold = absolute_tolerance+relative_tolerance*magnitude
        receipt['checks'].append({'order': order, 'absolute_difference_dual_proxy': error,
            'norm_dual_proxy': magnitude, 'threshold': threshold})
        receipt.update(final_order=order, last_source_receipt=local_log)
        if math.isfinite(error) and math.isfinite(magnitude) and error <= threshold:
            receipt.update(converged=True, failure_reason=None)
            return result, receipt
        previous = result.detach()
    receipt['failure_reason'] = 'outer_testing_order_budget_exhausted'
    raise VolumeTestingConvergenceError(receipt)
