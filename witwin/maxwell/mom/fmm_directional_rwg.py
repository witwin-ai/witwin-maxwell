"""Independent source/receiver carrier RWG maps and internal scalar-medium EM."""
import math
import torch
from .fmm_expansion import _points, multi_indices
from .fmm_moments import _rwg_weighted_moments
from .nearfield import _selected_edges
from .operator import PreparedRWGGeometry
from .fmm_fft import FFTElectromagneticM2L
from .fmm_directional_fft import CartesianDirectionalFFTTranslation


class DirectionalRWGProjection:
    """Receiver maps are not source conjugates for a passive complex carrier."""
    def __init__(self, basis, edges, center, direction, wave_number, order, *,
                 quadrature_order, face_batch=32, geometry=None,
                 retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=256*1024**2):
        _points(center); _points(direction)
        mesh = basis.mesh
        if (center.shape != (3,) or direction.shape != (3,)
                or center.device != mesh.device or direction.device != mesh.device
                or center.dtype != torch.float64 or direction.dtype != torch.float64
                or mesh.dtype != torch.float64):
            raise ValueError('Directional RWG maps require matching fixed CUDA float64 geometry')
        if not bool((direction.norm()-1).abs() <= 512*torch.finfo(direction.dtype).eps):
            raise ValueError('Directional RWG carrier must be unit length without normalization')
        if type(order) is not int or not 0 <= order <= 12:
            raise ValueError('Directional RWG order must lie between zero and twelve')
        if any(type(v) is not int or v < 1 for v in
               (quadrature_order, face_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Directional RWG quadrature, batches and budgets must be positive integers')
        if quadrature_order < (order+4)//2:
            raise ValueError('Directional RWG quadrature cannot underresolve its polynomial factor')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Directional RWG carrier requires a fixed passive wave number')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
            raise ValueError('Directional RWG wave number must be finite and passive')
        _selected_edges(basis, edges)
        count = len(multi_indices(order))
        owns_geometry = geometry is None
        owned_geometry_bytes, expected, workspace = directional_rwg_projection_capacity(
            basis, len(edges), order, quadrature_order, face_batch, owns_geometry)
        if expected > retained_budget_bytes:
            raise MemoryError('Directional RWG retained budget exceeded before integration')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Directional RWG tensor workspace exceeded before integration')
        self.basis = basis
        self.owns_geometry, self.owned_geometry_bytes = owns_geometry, owned_geometry_bytes
        self.geometry = PreparedRWGGeometry(basis) if geometry is None else geometry
        self.geometry.assert_current(basis)
        self.edges, self.center, self.direction = edges.clone(), center.clone(), direction.clone()
        self.wave_number, self.order, self.count = wave, order, count
        self.quadrature_order, self.face_batch = quadrature_order, face_batch
        self.device, self.dtype = center.device, torch.complex128
        self.retained_budget_bytes = retained_budget_bytes
        self.tensor_workspace_budget_bytes = tensor_workspace_budget_bytes
        self.planned_retained_bytes = expected
        self.planned_tensor_workspace_bytes = workspace
        kwargs = dict(geometry=self.geometry, quadrature_order=quadrature_order, face_batch=face_batch)
        self.source_vector, self.source_divergence = _rwg_weighted_moments(
            basis, self.edges, self.center, order, phase=(-wave, self.direction), **kwargs)
        self.receiver_vector, self.receiver_divergence = _rwg_weighted_moments(
            basis, self.edges, self.center, order, phase=(wave, self.direction), **kwargs)
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected:
            raise RuntimeError('Directional RWG retained arrays differ from structural plan')
        self._binding = self._state()

    def tensors(self):
        return (self.edges, self.center, self.direction, self.source_vector,
                self.source_divergence, self.receiver_vector, self.receiver_divergence,
                *(self.geometry.tensors() if self.owns_geometry else ()))

    def _state(self):
        return (id(self.basis), id(self.geometry), self.owns_geometry, self.owned_geometry_bytes,
                self.wave_number, self.order, self.count,
                self.quadrature_order, self.face_batch, self.device, self.dtype,
                self.retained_budget_bytes, self.tensor_workspace_budget_bytes,
                self.planned_retained_bytes, self.planned_tensor_workspace_bytes, self.retained_bytes,
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Directional RWG projection changed')
        self.geometry.assert_current(self.basis)

    def _validate(self, value, tail):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < len(tail)
                or value.shape[-len(tail):] != tail or value.device != self.device
                or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Directional RWG action requires matching finite complex128 currents')

    @staticmethod
    def _finite(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Directional RWG action exceeds native precision')
        return value

    def source_moments(self, currents):
        self._validate(currents, (len(self.edges),))
        vector = torch.einsum('...e,eac->...ac', currents, self.source_vector)
        return self._finite(torch.cat((vector, (currents@self.source_divergence)[..., None]), dim=-1))

    def source_adjoint(self, moments):
        self._validate(moments, (self.count, 4))
        return self._finite(torch.einsum('eac,...ac->...e', self.source_vector.conj(), moments[..., :3])
                +moments[..., 3]@self.source_divergence.mH)

    def receiver_evaluate(self, coefficients):
        self._validate(coefficients, (self.count, 4))
        return self._finite(torch.einsum('eac,...ac->...e', self.receiver_vector, coefficients[..., :3])
                +coefficients[..., 3]@self.receiver_divergence.mT)

    def receiver_adjoint(self, currents):
        self._validate(currents, (len(self.edges),))
        vector = torch.einsum('...e,eac->...ac', currents, self.receiver_vector.conj())
        return self._finite(torch.cat((vector, (currents@self.receiver_divergence.conj())[..., None]), dim=-1))

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_directional_rwg_projection_v1',
            order=self.order, quadrature_order=self.quadrature_order, face_batch=self.face_batch,
            owned_retained_bytes=self.retained_bytes,
            planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            shared_basis_excluded=True, caller_shared_geometry_excluded=not self.owns_geometry,
            owned_geometry_included=self.owns_geometry, owned_geometry_bytes=self.owned_geometry_bytes,
            source_receiver_independently_integrated=True,
            polynomial_exact=False, quadrature_convergence_certified=False,
            physical_qualification=False, directional_tree_integrated=False, peak_memory_qualified=False)


def directional_rwg_admission_workspace_bytes(test, source):
    """Visible geometry, full-basis integral and selected-statistic allowance."""
    from .fmm_error import directional_bound_workspace_bytes
    return directional_bound_workspace_bytes(1, 1)+4096+sum(
        len(p.basis.mesh.faces)*4096+p.basis.n_edges*512+len(p.edges)*1024
        for p in (test, source))


def _directional_rwg_admission(test, source, medium, impedance, workspace):
    from .physics import Medium
    from .fmm_error import directional_multipole_truncation_bound
    from .rwg_interpolation_error import rwg_integral_logs, block_error_log, _receipt
    if not isinstance(test, DirectionalRWGProjection) or not isinstance(source, DirectionalRWGProjection):
        raise TypeError('Directional admission requires prescribed direct RWG projections')
    if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
        raise ValueError('Directional RWG admission requires a finite-frequency scalar medium')
    if type(impedance) not in (int, float) or not math.isfinite(impedance) or impedance <= 0:
        raise ValueError('Directional RWG balancing impedance must be positive and finite')
    for p in (test, source):
        p.assert_current()
        if p.wave_number != medium.k:
            raise ValueError('Directional RWG projection wave differs from medium')
    if test.device != source.device or test.order != source.order or not torch.equal(test.direction, source.direction):
        raise ValueError('Directional RWG projections require equal device, order and carrier')
    planned = directional_rwg_admission_workspace_bytes(test, source)
    if type(workspace) is not int or workspace < 1:
        raise ValueError('Directional RWG admission workspace must be a positive integer')
    if planned > workspace:
        raise MemoryError('Directional RWG admission workspace exceeded before support/integral tensors')
    radii, statistics = [], []
    for p in (test, source):
        _selected_edges(p.basis, p.edges)
        faces = torch.cat((p.basis.plus_faces[p.edges], p.basis.minus_faces[p.edges]))
        offsets = p.geometry.triangles[faces]-p.center
        radius = offsets.norm(dim=-1).amax()*(1+512*torch.finfo(torch.float64).eps)
        if not bool(torch.isfinite(radius) & (radius > 0)):
            raise ValueError('Directional complete RWG support radius exceeds native geometry range')
        radii.append(radius)
        del faces, offsets
        logs = rwg_integral_logs(p.basis, p.geometry)[p.edges]
        u, q = logs.unbind(-1)
        statistics.append(torch.logsumexp(torch.stack((2*u, 2*q, u+q), dim=-1), dim=0))
        del logs, u, q
    kernel = directional_multipole_truncation_bound(test.center-source.center, test.direction,
        *radii, medium.k, test.order, tensor_workspace_budget_bytes=workspace)
    logged = block_error_log(*statistics, medium,
        {'green_absolute_bound': kernel['green_absolute'],
         'gradient_component_absolute_bound': kernel['gradient_component_absolute']}, impedance)
    bound = _receipt(logged, impedance)
    bound.update(version='directional_rwg_ideal_truncation_frobenius_v1',
        test_support_radius=float(radii[0]), source_support_radius=float(radii[1]),
        complete_selected_plus_minus_support_verified=True,
        prescribed_projection_composition_only=True, phase_path_included=False,
        global_operator_certified=False, production_qualification=False,
        planned_admission_tensor_workspace_bytes=planned)
    for p in (test, source):p.assert_current()
    return radii, kernel, bound


def directional_nested_admission_workspace_bytes(test, source):
    """Full prescribed trees, provider geometry and log-statistic allowance."""
    from .nearfield import ScalarSystemBlocks
    from .fmm_directional_moments import directional_phase_path_workspace_bytes
    from .fmm_error import directional_bound_workspace_bytes
    planned = directional_bound_workspace_bytes(1, 1)+4096
    for projection in (test, source):
        basis = projection.basis
        faces = (sum(len(body.basis.mesh.faces) for body in basis.system.bodies)
                 if isinstance(basis, ScalarSystemBlocks) else len(basis.mesh.faces))
        planned += (4096*faces+1024*projection.tree.size+512*projection.tree.node_count
                    +directional_phase_path_workspace_bytes(projection.tree.node_count))
    return planned


def _directional_phase_perturbed_kernel(test, test_context, source, source_context,
                                         medium, radii, kernel, workspace):
    """Shared continuous-kernel majorant for ideal context-to-leaf phase paths."""
    from .rwg_interpolation_error import _log_magnitude
    displacement = test.centers[test_context]-source.centers[source_context]
    products = (test._ideal_phase_path_log_products(workspace), source._ideal_phase_path_log_products(workspace))
    lt, ls = products[0][test_context], products[1][source_context]
    combined_log = lt+ls
    delta = torch.expm1(combined_log)
    if not bool(torch.isfinite(delta) & ((combined_log == 0) | (delta > 0))):
        raise ValueError('Nested positive combined phase defect exceeds native range')
    separation = displacement.norm()-sum(radii)
    if not bool(torch.isfinite(separation) & (separation > 0)):
        raise ValueError('Nested phase perturbation requires positive real support separation')
    log_gmax = -medium.k.imag*separation-math.log(4*math.pi)-separation.log()
    log_gradmax = log_gmax+torch.logaddexp(separation.new_tensor(_log_magnitude(medium.k)), -separation.log())
    effective = dict(kernel)
    if bool(combined_log > 0):
        log_delta = delta.log()
        for name, log_name, majorant in (('green_absolute','green_log_absolute',log_gmax),
                ('gradient_component_absolute','gradient_component_log_absolute',log_gradmax)):
            truncation_log = separation.new_tensor(kernel[log_name])
            logged = torch.logaddexp(truncation_log, log_delta+torch.logaddexp(majorant,truncation_log))
            value = logged.clamp_min(math.log(torch.finfo(torch.float64).tiny)).exp()
            if not bool(torch.isfinite(logged) & torch.isfinite(value) & (value > 0)):
                raise ValueError('Nested phase-perturbed ideal kernel bound exceeds native range')
            effective[name], effective[log_name] = float(value), float(logged)
    effective.update(version='directional_kernel_with_ideal_phase_paths_v1',
        scope='ideal_continuous_kernel_and_prescribed_leaf_to_node_phase_products_only',
        test_phase_log_one_plus_relative_bound=float(lt), source_phase_log_one_plus_relative_bound=float(ls),
        combined_phase_relative_bound=float(delta), phase_path_included=True,
        phase_derivative_error_certified=False)
    return effective, lt, ls, delta


def _nested_directional_rwg_admission(test, test_node, source, source_node, medium,
                                      impedance, workspace):
    from .fmm_directional_moments import NestedDirectionalRWGProjection
    from .physics import Medium
    from .fmm_error import directional_multipole_truncation_bound
    from .rwg_interpolation_error import tree_integral_statistics, block_error_log, _receipt, _log_magnitude
    if not all(isinstance(p, NestedDirectionalRWGProjection) for p in (test, source)):
        raise TypeError('Nested directional admission requires prescribed complete RWG trees')
    if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
        raise ValueError('Nested directional admission requires a finite-frequency scalar medium')
    if type(impedance) not in (int, float) or not math.isfinite(impedance) or impedance <= 0:
        raise ValueError('Nested directional balancing impedance must be positive and finite')
    for p, node in ((test, test_node), (source, source_node)):
        p.assert_current()
        if type(node) is not int or not 0 <= node < p.tree.node_count:
            raise ValueError('Prescribed directional interaction node is outside its tree')
        if p.wave_number != medium.k:
            raise ValueError('Prescribed directional node wave differs from medium')
    directions = tuple(p.node_directions[node] if p.changing_carriers else p.direction
                       for p, node in ((test, test_node), (source, source_node)))
    order = test.node_orders[test_node]
    if (test.device != source.device or order != source.node_orders[source_node]
            or not torch.equal(*directions)):
        raise ValueError('Prescribed directional nodes require equal device, order and carrier')
    planned = directional_nested_admission_workspace_bytes(test, source)
    if type(workspace) is not int or workspace < 1:
        raise ValueError('Nested directional admission workspace must be a positive integer')
    if planned > workspace:
        raise MemoryError('Nested directional admission workspace exceeded before path/statistic tensors')
    displacement = test.centers[test_node]-source.centers[source_node]
    # Recheck corners about the actual stored center for fixed and changing
    # trees; never shrink an existing radius or rely on a rounded box midpoint.
    radii = _directional_node_support_radii(((test, test_node, test_node), (source, source_node, source_node)))
    kernel = directional_multipole_truncation_bound(displacement, directions[0], *radii,
        medium.k, order, tensor_workspace_budget_bytes=workspace)
    effective, lt, ls, delta = _directional_phase_perturbed_kernel(test, test_node,
        source, source_node, medium, radii, kernel, workspace)
    statistics = (tree_integral_statistics(test), tree_integral_statistics(source))
    logged = block_error_log(statistics[0][test_node], statistics[1][source_node], medium,
        {'green_absolute_bound': effective['green_absolute'],
         'gradient_component_absolute_bound': effective['gradient_component_absolute']}, impedance)
    bound = _receipt(logged, impedance)
    bound.update(version='directional_nested_node_ideal_truncation_phase_frobenius_v1',
        test_node=test_node, source_node=source_node,
        test_support_radius=float(radii[0]), source_support_radius=float(radii[1]),
        complete_selected_plus_minus_support_verified=True, prescribed_projection_composition_only=True,
        phase_path_included=True, combined_phase_relative_bound=float(delta),
        test_phase_log_one_plus_relative_bound=float(lt), source_phase_log_one_plus_relative_bound=float(ls),
        effective_green_absolute=effective['green_absolute'],
        effective_gradient_component_absolute=effective['gradient_component_absolute'],
        global_operator_certified=False, production_qualification=False,
        planned_admission_tensor_workspace_bytes=planned)
    for p in (test, source):p.assert_current()
    return radii, effective, bound


class DirectionalFFTElectromagneticM2L(FFTElectromagneticM2L):
    """Original PMCHWT curl/material channels with directional physical gradients."""
    def __init__(self, displacement, direction, medium, normalization_length, order, **kwargs):
        from .physics import Medium
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Directional EM M2L requires a finite-frequency scalar medium')
        self._bind_scalar(CartesianDirectionalFFTTranslation(displacement, direction, medium.k,
                          normalization_length, order, **kwargs), medium)

    def _bind_scalar(self, scalar, medium):
        self._rwg_context = None
        super()._bind_scalar(scalar, medium)

    @classmethod
    def from_projections(cls, test, source, medium, normalization_length, *,
                         kernel_error_budget, ideal_rwg_error_budget, balancing_impedance,
                         tensor_workspace_budget_bytes=128*1024**2, **kwargs):
        from dataclasses import asdict
        from .interpolation_error import KernelErrorBudget
        from .fmm_error import require_multipole_budget
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise ValueError('Directional RWG admission requires an explicit KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if (type(ideal_rwg_error_budget) not in (int, float)
                or not math.isfinite(ideal_rwg_error_budget) or ideal_rwg_error_budget <= 0):
            raise ValueError('Directional ideal RWG absolute budget must be positive and finite')
        if any(key in kwargs for key in ('test_radius', 'source_radius')):
            raise ValueError('Prescribed RWG radii are derived from complete supports')
        radii, kernel, bound = _directional_rwg_admission(test, source, medium,
            balancing_impedance, tensor_workspace_budget_bytes)
        require_multipole_budget(kernel, kernel_error_budget)
        if bound['absolute_bound'] > ideal_rwg_error_budget:
            raise ValueError('Directional ideal RWG truncation exceeds absolute budget')
        result = cls(test.center-source.center, test.direction, medium, normalization_length,
            test.order, test_radius=radii[0], source_radius=radii[1],
            kernel_error_budget=kernel_error_budget,
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, **kwargs)
        result._rwg_context = (test, source)
        result.ideal_rwg_error_budget = ideal_rwg_error_budget
        result.balancing_impedance = balancing_impedance
        result._rwg_bound = bound
        result._binding = result._state()
        result.assert_current()
        return result

    @classmethod
    def from_nested_nodes(cls, test, test_node, source, source_node, medium, normalization_length, *,
                          kernel_error_budget, ideal_rwg_error_budget, balancing_impedance,
                          tensor_workspace_budget_bytes=128*1024**2, **kwargs):
        from dataclasses import asdict
        from .interpolation_error import KernelErrorBudget
        from .fmm_error import require_multipole_budget
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise ValueError('Nested directional admission requires an explicit KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if (type(ideal_rwg_error_budget) not in (int, float)
                or not math.isfinite(ideal_rwg_error_budget) or ideal_rwg_error_budget <= 0):
            raise ValueError('Nested ideal RWG absolute budget must be positive and finite')
        if any(key in kwargs for key in ('test_radius', 'source_radius')):
            raise ValueError('Prescribed node radii are derived from complete tree supports')
        radii, effective, bound = _nested_directional_rwg_admission(test, test_node,
            source, source_node, medium, balancing_impedance, tensor_workspace_budget_bytes)
        require_multipole_budget(effective, kernel_error_budget)
        if bound['absolute_bound'] > ideal_rwg_error_budget:
            raise ValueError('Nested ideal RWG phase/truncation exceeds absolute budget')
        direction = test.node_directions[test_node] if test.changing_carriers else test.direction
        result = cls(test.centers[test_node]-source.centers[source_node], direction, medium,
            normalization_length, test.node_orders[test_node], test_radius=radii[0], source_radius=radii[1],
            kernel_error_budget=kernel_error_budget, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, **kwargs)
        result._rwg_context = (test, source)
        result.ideal_rwg_error_budget = ideal_rwg_error_budget
        result.balancing_impedance = balancing_impedance
        result._rwg_bound = bound
        result._binding = result._state()
        result.assert_current()
        return result

    @classmethod
    def from_sparse_contexts(cls, test, test_context, source, source_context, medium, normalization_length, *,
                             kernel_error_budget, ideal_rwg_error_budget, balancing_impedance,
                             tensor_workspace_budget_bytes=128*1024**2, **kwargs):
        """Admit actual sparse IDs; physical node IDs remain separate metadata."""
        return cls._from_contexts(test, test_context, source, source_context, medium, normalization_length,
            kernel_error_budget=kernel_error_budget, ideal_rwg_error_budget=ideal_rwg_error_budget,
            balancing_impedance=balancing_impedance, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,
            dag=False, **kwargs)

    @classmethod
    def from_dag_contexts(cls, test, test_context, source, source_context, medium, normalization_length, *,
                          kernel_error_budget, ideal_rwg_error_budget, balancing_impedance,
                          tensor_workspace_budget_bytes=128*1024**2, **kwargs):
        """Admit shared changing-carrier contexts including both ideal phase paths."""
        return cls._from_contexts(test, test_context, source, source_context, medium, normalization_length,
            kernel_error_budget=kernel_error_budget, ideal_rwg_error_budget=ideal_rwg_error_budget,
            balancing_impedance=balancing_impedance, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,
            dag=True, **kwargs)

    @classmethod
    def _from_contexts(cls, test, test_context, source, source_context, medium, normalization_length, *,
                       kernel_error_budget, ideal_rwg_error_budget, balancing_impedance, dag,
                       tensor_workspace_budget_bytes, **kwargs):
        from dataclasses import asdict
        from .interpolation_error import KernelErrorBudget
        from .fmm_error import require_multipole_budget
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise ValueError('Sparse directional admission requires an explicit KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if (type(ideal_rwg_error_budget) not in (int, float)
                or not math.isfinite(ideal_rwg_error_budget) or ideal_rwg_error_budget <= 0):
            raise ValueError('Sparse ideal RWG absolute budget must be positive and finite')
        if any(key in kwargs for key in ('test_radius', 'source_radius')):
            raise ValueError('Sparse context radii are derived from complete physical supports')
        radii, kernel, bound = _sparse_directional_rwg_admission(test, test_context,
            source, source_context, medium, balancing_impedance, tensor_workspace_budget_bytes, dag=dag)
        require_multipole_budget(kernel, kernel_error_budget)
        if bound['absolute_bound'] > ideal_rwg_error_budget:
            raise ValueError('Sparse ideal RWG truncation exceeds absolute budget')
        direction = test.carriers[test.context_keys[test_context][1]]
        result = cls(test.centers[test_context]-source.centers[source_context], direction, medium,
            normalization_length, test.node_orders[test_context], test_radius=radii[0], source_radius=radii[1],
            kernel_error_budget=kernel_error_budget, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, **kwargs)
        result._rwg_context = (test, source)
        result.ideal_rwg_error_budget = ideal_rwg_error_budget
        result.balancing_impedance = balancing_impedance
        result._rwg_bound = bound
        result._binding = result._state()
        result.assert_current()
        return result

    def _state(self):
        state = super()._state()
        if self._rwg_context is None:
            return state
        return (*state, tuple(id(p) for p in self._rwg_context),
            self.ideal_rwg_error_budget, self.balancing_impedance, tuple(self._rwg_bound.items()))

    def _assert_translation_current(self):
        super().assert_current()
        if (self._rwg_context is not None and 'test_context' in self._rwg_bound
                and any(t.requires_grad for t in self.scalar.tensors())):
            raise ValueError('Sparse material-bound translation requires fixed native configuration')

    def assert_current(self):
        self._assert_translation_current()
        if self._rwg_context is not None:
            for projection in self._rwg_context:
                projection.assert_current()

    def ideal_rwg_truncation_bound(self):
        self.assert_current()
        if self._rwg_context is None:
            raise ValueError('No prescribed RWG projection admission configured')
        return dict(self._rwg_bound)

    def receipt(self):
        record = super().receipt()
        record.update(version='experimental_directional_fft_pmchwt_m2l_v1',
            phase_projection_scope='Independent physical source/receiver RWG maps required',
            directional_tree_integrated=False, error_bound_certified=False)
        if self._rwg_context is not None:
            record.update(version='experimental_directional_fft_prescribed_rwg_budget_v1',
                ideal_rwg_truncation_admission=self.ideal_rwg_truncation_bound(),
                ideal_rwg_error_budget=self.ideal_rwg_error_budget,
                balancing_impedance=self.balancing_impedance,
                caller_prescribed_projection_storage_excluded=True,
                prescribed_projection_composition_only=True, production_qualification=False)
        return record


class DirectionalNestedFarActions:
    """Explicit disjoint FAR blocks; near entries are zero, not a full solver."""
    from .direct_system import DirectSystemActions as _LinearActions
    apply = _LinearActions.apply
    adjoint = _LinearActions.adjoint

    def __init__(self, test, source, medium, couplings, *, ideal_rwg_error_budget,
                 balancing_impedance=None, rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2):
        self._initialize(test, source, medium, couplings, ideal_rwg_error_budget=ideal_rwg_error_budget,
            balancing_impedance=balancing_impedance, rhs_batch=rhs_batch,
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, sparse=False)

    def _initialize(self, test, source, medium, couplings, *, ideal_rwg_error_budget,
                    balancing_impedance, rhs_batch, tensor_workspace_budget_bytes, sparse):
        from .fmm_directional_moments import (NestedDirectionalRWGProjection,
            SparseDirectionalRWGProjection, DirectionalRWGDAGProjection)
        from .physics import Medium
        from .rwg_interpolation_error import _receipt
        projection_type = (DirectionalRWGDAGProjection if sparse == 'dag'
            else SparseDirectionalRWGProjection if sparse else NestedDirectionalRWGProjection)
        if not all(isinstance(p, projection_type) for p in (test, source)):
            raise TypeError('Directional far actions require matching prescribed prototype contexts')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Directional far actions require a finite-frequency scalar medium')
        for projection in (test, source):
            projection.assert_current()
            if projection.wave_number != medium.k:
                raise ValueError('Directional far prototype wave differs from medium')
        if test.device != source.device or not isinstance(couplings, (tuple, list)):
            raise ValueError('Directional far actions require matching devices and an explicit coupling sequence')
        eta = abs(medium.eta) if balancing_impedance is None else balancing_impedance
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (eta, ideal_rwg_error_budget)):
            raise ValueError('Directional far impedance and ideal absolute budget must be positive and finite')
        if any(type(v) is not int or v < 1 for v in (rhs_batch, tensor_workspace_budget_bytes)):
            raise ValueError('Directional far RHS and workspace must be positive integers')
        records, targets, origins = [], {}, {}
        for coupling in couplings:
            if not isinstance(coupling, DirectionalFFTElectromagneticM2L):
                raise TypeError('Directional far blocks require admitted prescribed-node couplings')
            coupling.assert_current()
            if coupling._rwg_context is None or coupling.medium is not medium or coupling.balancing_impedance != eta:
                raise ValueError('Directional far couplings require a common medium and balancing impedance')
            bound = coupling.ideal_rwg_truncation_bound()
            names = ('test_context', 'source_context') if sparse else ('test_node', 'source_node')
            if not bound.get('phase_path_included') or any(name not in bound for name in names):
                raise ValueError('Directional far blocks require actual matching prescribed context admission')
            target, origin = coupling._rwg_context
            if sparse == 'dag' and (not bound.get('changing_carrier_DAG')
                    or not all(isinstance(p, projection_type) for p in (target, origin))):
                raise TypeError('DAG far blocks require phase-inclusive DAG coupling owners')
            if any(p.basis is not prototype.basis or p.tree is not prototype.tree
                   for p, prototype in ((target, test), (origin, source))):
                raise ValueError('Directional far maps must share physical basis and tree identities')
            a, b = (bound[name] for name in names)
            if sparse:
                an, bn = target.context_keys[a][0], origin.context_keys[b][0]
                records.append((int(target.tree.starts[an]), int(target.tree.stops[an]),
                                int(origin.tree.starts[bn]), int(origin.tree.stops[bn]), a, b))
            else:
                records.append((target.starts[a], target.stops[a], origin.starts[b], origin.stops[b], a, b))
            targets[id(target)], origins[id(origin)] = target, origin
        self._validate_disjoint_rectangles(records)
        self.test, self.source, self.medium, self.couplings = test, source, medium, tuple(couplings)
        self.targets, self.origins, self.rectangles = tuple(targets.values()), tuple(origins.values()), tuple(records)
        self.device, self.dtype = test.device, torch.complex128
        self.shape = (2*test.tree.size, 2*source.tree.size)
        self.rhs_batch, self.tensor_workspace_budget_bytes = rhs_batch, tensor_workspace_budget_bytes
        self.balancing_impedance, self.ideal_rwg_error_budget = eta, ideal_rwg_error_budget
        counts = sum(sum(p.node_counts) for p in (*self.targets, *self.origins))
        maps = max((p.planned_tensor_workspace_bytes for p in (*self.targets, *self.origins)), default=0)
        fields = max((c.planned_tensor_workspace_bytes for c in self.couplings), default=0)
        self.planned_tensor_workspace_bytes = (4096+maps+fields
            +16*rhs_batch*(16*(test.tree.size+source.tree.size)+16*counts)+32*len(couplings))
        if self.planned_tensor_workspace_bytes > tensor_workspace_budget_bytes:
            raise MemoryError('Directional far workspace exceeded before log bounds or current buffers')
        log_bounds = [c.ideal_rwg_truncation_bound()['log_absolute_bound'] for c in couplings]
        logs = torch.tensor([-math.inf if bound is None else bound for bound in log_bounds],
                            device=self.device, dtype=torch.float64)
        logged = .5*torch.logsumexp(2*logs, dim=0) if len(couplings) else logs.new_tensor(-torch.inf)
        self._ideal_bound = _receipt(logged, eta)
        self._ideal_bound.update(version='directional_disjoint_far_ideal_frobenius_v1',
            phase_paths_included=True, disjoint_physical_rectangles_checked=True,
            far_component_only=True, near_entries_zero=True, global_full_operator_certified=False,
            production_qualification=False)
        if self._ideal_bound['absolute_bound'] > ideal_rwg_error_budget:
            raise ValueError('Directional disjoint far ideal error exceeds regional absolute budget')
        self.retained_bytes, self.planned_retained_bytes = 0, 0
        self._binding = self._state()
        self.assert_current()

    @staticmethod
    def _validate_disjoint_rectangles(records):
        from bisect import bisect_left
        events = []
        for i, (start, stop, low, high, _, _) in enumerate(records):
            if not 0 <= start < stop or not 0 <= low < high:
                raise ValueError('Directional far rectangles must be nonempty half-open ranges')
            events.extend(((start, 1, low, high, i), (stop, 0, low, high, i)))
        active = []
        for _, add, low, high, i in sorted(events):
            key = (low, high, i);position = bisect_left(active, key)
            if not add:
                if position >= len(active) or active[position] != key:
                    raise ValueError('Directional far rectangle sweep metadata is inconsistent')
                active.pop(position)
            else:
                if ((position and active[position-1][1] > low)
                        or (position < len(active) and active[position][0] < high)):
                    raise ValueError('Directional far physical rectangles overlap or duplicate')
                active.insert(position, key)

    def tensors(self):
        return ()

    def _state(self):
        m = self.medium
        return (id(self.test), id(self.source), id(m), m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta,
                tuple(id(c) for c in self.couplings), tuple(id(p) for p in self.targets),
                tuple(id(p) for p in self.origins), self.rectangles, self.shape, self.device, self.dtype,
                self.rhs_batch, self.tensor_workspace_budget_bytes, self.planned_tensor_workspace_bytes,
                self.balancing_impedance, self.ideal_rwg_error_budget, self.retained_bytes,
                self.planned_retained_bytes, tuple(self._ideal_bound.items()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Directional far actions or admitted partition metadata changed')
        # All coupling contexts are in targets/origins. Validate each shared
        # owner once, then every distinct translation binding/table below.
        owners = {id(p):p for p in (self.test,self.source,*self.targets,*self.origins)}
        for projection in owners.values():projection.assert_current()
        for coupling in self.couplings:coupling._assert_translation_current()

    def _action(self, value, *, adjoint):
        from .fmm_fft import _rhs_chunks
        self.assert_current()
        ni, no = (self.shape[0], self.shape[1]) if adjoint else (self.shape[1], self.shape[0])
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != ni
                or value.device != self.device or value.dtype != self.dtype):
            raise ValueError('Directional far action requires matching physical complex128 currents')
        result = value.new_empty((math.prod(value.shape[:-1]), no))
        inputs, outputs = (self.targets, self.origins) if adjoint else (self.origins, self.targets)
        for start, chunk in _rhs_chunks(value, 1, self.rhs_batch):
            currents = torch.stack((chunk[..., :ni//2], chunk[..., ni//2:]), -2)
            moments = {id(p):p.receiver_upward(currents) if adjoint else p.source_upward(currents) for p in inputs}
            locals_by_owner = {id(p):[value.new_zeros((len(chunk), 2, count, 4)) for count in p.node_counts] for p in outputs}
            for coupling, rectangle in zip(self.couplings, self.rectangles, strict=True):
                target, origin = coupling._rwg_context;a, b = rectangle[-2:]
                if adjoint:
                    translated = coupling._action_after_validation(moments[id(target)][a], adjoint=True)
                    locals_by_owner[id(origin)][b] += translated
                else:
                    translated = coupling._action_after_validation(moments[id(origin)][b], adjoint=False)
                    locals_by_owner[id(target)][a] += translated
                del translated
            batch = value.new_zeros((len(chunk), 2, no//2))
            for projection in outputs:
                locals_ = tuple(locals_by_owner[id(projection)])
                batch += projection.source_downward(locals_) if adjoint else projection.receiver_downward(locals_)
                del locals_
            if not bool(torch.isfinite(batch).all()):
                raise ValueError('Directional far physical action exceeds native precision')
            result[start:start+len(chunk)] = torch.cat((batch[:, 0], batch[:, 1]), -1)
            del currents, moments, locals_by_owner, batch
        self.assert_current()
        return result.reshape(*value.shape[:-1], no)

    def ideal_rwg_truncation_bound(self):
        self.assert_current()
        return dict(self._ideal_bound)

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_disjoint_directional_far_actions_v1',
            far_blocks=len(self.couplings), target_projection_owners=len(self.targets),
            source_projection_owners=len(self.origins), shape=self.shape, rhs_batch=self.rhs_batch,
            ideal_rwg_error_budget=self.ideal_rwg_error_budget,
            ideal_rwg_truncation_admission=self.ideal_rwg_truncation_bound(),
            balancing_impedance=self.balancing_impedance, balancing_default='abs(medium.eta)',
            owned_retained_bytes=0, caller_couplings_maps_trees_excluded=True,
            planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            workspace_excludes=('caller input/output tensors','allocator cache/workspace','backend FFT library workspace'),
            current_backward='guarded_full_Hermitian_no_saved_internal_graph',
            far_component_only=True, near_entries_zero=True, complete_system_coverage=False,
            floating_point_certified=False, quadrature_certified=False, physical_qualification=False,
            complexity_qualification=False, peak_memory_qualified=False, production_qualification=False)


def directional_rwg_projection_capacity(basis, edge_count, order, quadrature_order, face_batch, owns_geometry):
    """Shared structural estimator; existing constructor validates its inputs."""
    count = len(multi_indices(order))
    geometry_bytes = len(basis.mesh.faces)*16*8 if owns_geometry else 0
    retained = 48+edge_count*8+2*edge_count*count*4*16+geometry_bytes
    workspace = (2*retained+face_batch*quadrature_order**2*(128*count+1024)
        +basis.n_edges*64+len(basis.mesh.faces)*16*8*3)
    return geometry_bytes, retained, workspace


class AutomaticDirectionalRWGBlock:
    """Actual selected-support automatic cone/order and ideal balanced admission.

    This selected direct block owns its maps and kernel. It does not duplicate
    full-tree maps per cone or claim complete system/scaling qualification.
    Geometry remains caller-owned and guarded. Kernel budgets are conservatively
    split to ensure a requested ideal RWG Frobenius budget before integration.
    """
    def __init__(self, test_basis, test_edges, test_center, test_geometry,
                 source_basis, source_edges, source_center, source_geometry, medium, *,
                 kernel_error_budget, ideal_rwg_error_budget, balancing_impedance,
                 quadrature_order, face_batch=32, min_order=0, max_order=12,
                 cone_subdivisions=(1, 2, 4, 8, 16, 32, 64),
                 retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=256*1024**2):
        from dataclasses import asdict
        from .physics import Medium
        from .interpolation_error import KernelErrorBudget
        from .fmm_error import directional_bound_workspace_bytes
        from .fmm_order import DirectionalConeOrderPlan
        from .fmm_storage import m2l_storage_bytes
        from .fmm_directional_fft import directional_fft_setup_workspace_bytes
        from .rwg_interpolation_error import rwg_integral_logs, block_error_log, _receipt
        if not isinstance(medium, Medium) or not math.isfinite(medium.frequency) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Automatic directional RWG requires a finite-frequency scalar medium')
        if not isinstance(kernel_error_budget, KernelErrorBudget):
            raise TypeError('Automatic directional RWG requires KernelErrorBudget')
        KernelErrorBudget(**asdict(kernel_error_budget))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (ideal_rwg_error_budget, balancing_impedance)):
            raise ValueError('Automatic directional RWG absolute budget and impedance must be positive finite')
        if any(type(v) is not int or v < 1 for v in
               (quadrature_order, face_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Automatic directional RWG quadrature, batches and capacities must be positive integers')
        if type(min_order) is not int or type(max_order) is not int or not 0 <= min_order <= max_order <= 12:
            raise ValueError('Automatic directional RWG order range must lie in0..12')
        sides = ((test_basis, test_edges, test_center, test_geometry),
                 (source_basis, source_edges, source_center, source_geometry))
        for basis, edges, center, geometry in sides:
            _points(center); _selected_edges(basis, edges)
            if center.shape != (3,) or center.dtype != torch.float64 or center.device != basis.mesh.device:
                raise ValueError('Automatic directional RWG requires matching fixed CUDA float64 centers')
            if basis.mesh.dtype != torch.float64 or not isinstance(geometry, PreparedRWGGeometry):
                raise ValueError('Automatic directional RWG requires prepared float64 physical geometry')
            geometry.assert_current(basis)
        if test_center.device != source_center.device:
            raise ValueError('Automatic directional RWG bases must share device')
        inputs = (test_edges, test_center, source_edges, source_center)
        versions = tuple((id(t), t._version) for t in inputs)
        admission_workspace = directional_bound_workspace_bytes(1, max_order-min_order+1)+8192+sum(
            len(b.mesh.faces)*4096+b.n_edges*512+len(e)*1024 for b, e, _, _ in sides)
        if admission_workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Automatic directional RWG admission workspace exceeded before support/statistics')
        radii, statistics = [], []
        for basis, edges, center, geometry in sides:
            faces = torch.cat((basis.plus_faces[edges], basis.minus_faces[edges]))
            radius = (geometry.triangles[faces]-center).norm(dim=-1).amax()*(1+512*torch.finfo(torch.float64).eps)
            if not bool(torch.isfinite(radius) & (radius > 0)):
                raise ValueError('Automatic directional RWG full support exceeds native geometry range')
            radii.append(radius.reshape(1))
            u, q = rwg_integral_logs(basis, geometry)[edges].unbind(-1)
            statistics.append(torch.logsumexp(torch.stack((2*u, 2*q, u+q), dim=-1), dim=0))
        # The balanced majorant squared is A^2 G^2 + B^2 H^2. Allocating half
        # the squared budget to each channel is conservative, not joint optimal.
        log_weights = torch.stack([block_error_log(*statistics, medium,
            dict(green_absolute_bound=g, gradient_component_absolute_bound=h), balancing_impedance)
            for g, h in ((1., 0.), (0., 1.))])
        limits = torch.minimum(torch.exp(math.log(ideal_rwg_error_budget)-.5*math.log(2)-log_weights),
            test_center.new_tensor([kernel_error_budget.green_absolute, kernel_error_budget.gradient_component_absolute]))
        if not bool(torch.isfinite(limits).all() & (limits > 0).all()):
            raise ValueError('Automatic directional RWG effective budget is not natively representable')
        effective_budget = KernelErrorBudget(*limits.tolist())
        displacement = (test_center-source_center).reshape(1, 3)
        plan = DirectionalConeOrderPlan(displacement, *radii, medium.k, effective_budget,
            min_order=min_order, max_order=max_order, cone_subdivisions=cone_subdivisions,
            retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        p = int(plan.orders[0]); count = len(multi_indices(p))
        if quadrature_order < (p+4)//2:
            raise ValueError('Automatic directional RWG quadrature underresolves selected polynomial order')
        kernel = dict(green_absolute_bound=float(plan.green_bounds[0]),
            gradient_component_absolute_bound=float(plan.gradient_bounds[0]))
        bound = _receipt(block_error_log(*statistics, medium, kernel, balancing_impedance), balancing_impedance)
        if bound['absolute_bound'] > ideal_rwg_error_budget:
            raise ValueError('Automatic directional RWG selected ideal bound exceeds budget')
        projection_capacities = [directional_rwg_projection_capacity(b, len(e), p, quadrature_order,
            face_batch, False) for b, e, _, _ in sides]
        scalar_bytes = m2l_storage_bytes(p, 8, 'fft_v1')+40
        # 144 owned planner bytes plus40 generated planner inputs owned here.
        retained = 184+sum(v[1] for v in projection_capacities)+scalar_bytes
        # All stages preflight before moments/coefficients. Summing independent
        # component allowances is conservative; this is not measured peak use.
        workspace = (admission_workspace+plan.planned_workspace_bytes+sum(v[2] for v in projection_capacities)
            +directional_fft_setup_workspace_bytes(p, 4, declared=True)+4*count*16*128)
        if retained > retained_budget_bytes:
            raise MemoryError('Automatic directional RWG total retained capacity exceeded before integration/coefficients')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Automatic directional RWG component workspace exceeded before integration/coefficients')
        if tuple((id(t), t._version) for t in inputs) != versions:
            raise ValueError('Actual selected support changed during automatic admission')
        for b, _, _, g in sides:g.assert_current(b)
        maps = [DirectionalRWGProjection(b, e, c, plan.directions[0], medium.k, p,
            quadrature_order=quadrature_order, face_batch=face_batch, geometry=g,
            retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
            for b, e, c, g in sides]
        coupling = DirectionalFFTElectromagneticM2L.from_projections(*maps, medium,
            plan.normalization_lengths[0], kernel_error_budget=effective_budget,
            ideal_rwg_error_budget=ideal_rwg_error_budget, balancing_impedance=balancing_impedance,
            retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        self.test, self.source, self.coupling, self.plan = *maps, coupling, plan
        self.kernel_error_budget, self.ideal_rwg_error_budget = kernel_error_budget, ideal_rwg_error_budget
        self.balancing_impedance = balancing_impedance
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.retained_bytes, self.planned_tensor_workspace_bytes = retained, workspace
        self._inputs, self._versions = inputs, versions
        self._binding = self._state()
        from .direct_capacity import storage_bytes
        if storage_bytes(self.tensors()) != retained:
            raise RuntimeError('Automatic directional RWG owned inventory differs from plan')
        self.assert_current()

    def tensors(self):
        from .fmm_order import _DIRECTIONAL_PLAN_TENSORS
        return (*(getattr(self.plan, name) for name in _DIRECTIONAL_PLAN_TENSORS),
            *self.plan._inputs, *self.test.tensors(), *self.source.tensors(), *self.coupling.tensors())

    def _state(self):
        return (id(self.test), id(self.source), id(self.coupling), id(self.plan),
            (self.kernel_error_budget.green_absolute, self.kernel_error_budget.gradient_component_absolute),
            self.ideal_rwg_error_budget, self.balancing_impedance,
            self.retained_budget_bytes, self.tensor_workspace_budget_bytes, self.retained_bytes,
            self.planned_tensor_workspace_bytes, tuple((id(t), t._version) for t in self._inputs), self._versions)

    def assert_current(self):
        if self._state() != self._binding or tuple((id(t), t._version) for t in self._inputs) != self._versions:
            raise ValueError('Automatic directional RWG support or configuration changed')
        self.plan.assert_current(); self.test.assert_current(); self.source.assert_current(); self.coupling.assert_current()

    def _action(self, value, *, adjoint):
        from .fmm_fft import _rhs_chunks
        self.assert_current()
        n = len(self.test.edges) if adjoint else len(self.source.edges)
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != 2*n
                or value.device != self.test.device or value.dtype != torch.complex128):
            raise ValueError('Automatic directional RWG requires matching physical J/M currents')
        no = 2*(len(self.source.edges) if adjoint else len(self.test.edges))
        result = value.new_empty((math.prod(value.shape[:-1]), no))
        for start, chunk in _rhs_chunks(value, 1, self.coupling.rhs_batch):
            currents = torch.stack((chunk[:, :n], chunk[:, n:]), dim=-2)
            if adjoint:
                moments = self.test.receiver_adjoint(currents)
                translated = self.coupling.adjoint(moments)
                output = self.source.source_adjoint(translated)
            else:
                moments = self.source.source_moments(currents)
                translated = self.coupling.apply(moments)
                output = self.test.receiver_evaluate(translated)
            result[start:start+len(chunk)] = torch.cat((output[:, 0], output[:, 1]), dim=-1)
            del currents, moments, translated, output
        self.assert_current()
        return result.reshape(*value.shape[:-1], no)

    def apply(self, value):
        from .direct_system import DirectSystemActions
        return DirectSystemActions.apply(self, value)

    def adjoint(self, value):
        from .direct_system import DirectSystemActions
        return DirectSystemActions.adjoint(self, value)

    def receipt(self):
        from dataclasses import asdict
        self.assert_current()
        return dict(version='automatic_actual_support_directional_rwg_block_v1',
            plan=self.plan.receipt(), coupling=self.coupling.receipt(),
            original_kernel_budget=asdict(self.kernel_error_budget),
            budget_split_rule='half_squared_ideal_balanced_budget_per_green_and_gradient_channel',
            owned_retained_bytes=self.retained_bytes, planned_component_workspace_bytes=self.planned_tensor_workspace_bytes,
            caller_geometry_excluded=True, generated_plan_inputs_included=True,
            complete_selected_plus_minus_support_verified=True,
            ideal_rwg_error_budget=self.ideal_rwg_error_budget, balancing_impedance=self.balancing_impedance,
            finite_quadrature_native_certified=False, complete_system_certified=False,
            sparse_directional_tree_integrated=False, physical_qualification=False, production_qualification=False)


def _directional_node_support_radii(selections):
    """Shared complete physical support corners about the selected map center."""
    from .nearfield import ScalarSystemBlocks
    radii = []
    for projection, node, physical_node in selections:
        if isinstance(projection.basis, ScalarSystemBlocks):
            lower, upper = projection.basis.support_boxes()
        else:
            basis, triangles = projection.basis, projection.geometry.triangles
            supports = torch.cat((triangles[basis.plus_faces], triangles[basis.minus_faces]), 1)
            lower, upper = supports.amin(1), supports.amax(1)
            del supports
        edges = projection.tree.permutation[int(projection.tree.starts[physical_node]):int(projection.tree.stops[physical_node])]
        corners = torch.maximum((lower[edges]-projection.centers[node]).abs(),
                                (upper[edges]-projection.centers[node]).abs())
        corner_norms = corners.norm(dim=-1)
        radius = torch.maximum(projection.radii[node], corner_norms.amax()*(1+512*torch.finfo(torch.float64).eps))
        if not bool(torch.isfinite(radius) & (radius > 0) & (corner_norms <= radius).all()):
            raise ValueError('Prescribed node radius must enclose all complete RWG support corners')
        radii.append(radius)
        del lower, upper, edges, corners, corner_norms
    return radii


def directional_sparse_admission_workspace_bytes(test, source, *, dag=False):
    """Component allowance for actual physical support and shared RWG statistics."""
    from .nearfield import ScalarSystemBlocks
    from .fmm_error import directional_bound_workspace_bytes
    planned = directional_bound_workspace_bytes(1, 1)+4096
    for projection in (test, source):
        basis = projection.basis
        faces = (sum(len(body.basis.mesh.faces) for body in basis.system.bodies)
                 if isinstance(basis, ScalarSystemBlocks) else len(basis.mesh.faces))
        planned += 4096*faces+1024*projection.tree.size+512*projection.tree.node_count
        if dag:
            from .fmm_directional_moments import directional_phase_path_workspace_bytes
            planned += directional_phase_path_workspace_bytes(len(projection.node_counts))
    return planned


def _sparse_directional_rwg_admission(test, test_context, source, source_context, medium,
                                       impedance, workspace, *, dag=False):
    from .fmm_directional_moments import SparseDirectionalRWGProjection, DirectionalRWGDAGProjection
    from .physics import Medium
    from .fmm_error import directional_multipole_truncation_bound
    from .rwg_interpolation_error import tree_integral_statistics, block_error_log, _receipt
    projection_type = DirectionalRWGDAGProjection if dag else SparseDirectionalRWGProjection
    if not all(isinstance(p, projection_type) for p in (test, source)):
        raise TypeError('Sparse directional admission requires sparse physical RWG contexts')
    if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
        raise ValueError('Sparse directional admission requires a finite-frequency scalar medium')
    if type(impedance) not in (int, float) or not math.isfinite(impedance) or impedance <= 0:
        raise ValueError('Sparse directional balancing impedance must be positive and finite')
    for p, context in ((test, test_context), (source, source_context)):
        p.assert_current()
        if type(context) is not int or not 0 <= context < len(p.context_keys):
            raise ValueError('Sparse interaction context is outside its active forest')
        if p.wave_number != medium.k:
            raise ValueError('Sparse context wave differs from medium')
    tn, tc = test.context_keys[test_context]; sn, sc = source.context_keys[source_context]
    if (test.device != source.device or test.node_orders[test_context] != source.node_orders[source_context]
            or not torch.equal(test.carriers[tc], source.carriers[sc])):
        raise ValueError('Sparse interaction contexts require equal device, order and carrier')
    if type(workspace) is not int or workspace < 1:
        raise ValueError('Sparse admission workspace must be a positive integer')
    planned = directional_sparse_admission_workspace_bytes(test, source, dag=dag)
    if planned > workspace:
        raise MemoryError('Sparse admission workspace exceeded before support/statistic tensors')
    radii = _directional_node_support_radii(((test, test_context, tn), (source, source_context, sn)))
    kernel = directional_multipole_truncation_bound(test.centers[test_context]-source.centers[source_context],
        test.carriers[tc], *radii, medium.k, test.node_orders[test_context], tensor_workspace_budget_bytes=workspace)
    lt = ls = delta = 0.
    if dag:
        kernel, lt, ls, delta = _directional_phase_perturbed_kernel(test, test_context,
            source, source_context, medium, radii, kernel, workspace)
    # Fixed sparse forests retain exact zero ideal phase truncation. DAGs
    # include both maximum path products before actual material weighting.
    statistics = (tree_integral_statistics(test), tree_integral_statistics(source))
    logged = block_error_log(statistics[0][tn], statistics[1][sn], medium,
        {'green_absolute_bound':kernel['green_absolute'],
         'gradient_component_absolute_bound':kernel['gradient_component_absolute']}, impedance)
    bound = _receipt(logged, impedance)
    bound.update(version='directional_sparse_context_ideal_frobenius_v1',
        test_context=test_context, source_context=source_context, test_physical_node=tn, source_physical_node=sn,
        test_support_radius=float(radii[0]), source_support_radius=float(radii[1]),
        complete_selected_plus_minus_support_verified=True, prescribed_projection_composition_only=True,
        phase_path_included=True, combined_phase_relative_bound=float(delta),
        test_phase_log_one_plus_relative_bound=float(lt), source_phase_log_one_plus_relative_bound=float(ls),
        effective_green_absolute=kernel['green_absolute'],
        effective_gradient_component_absolute=kernel['gradient_component_absolute'],
        global_operator_certified=False, production_qualification=False,
        planned_admission_tensor_workspace_bytes=planned)
    if dag:
        bound.update(version='directional_shared_dag_ideal_truncation_phase_frobenius_v1',
            changing_carrier_DAG=True, shared_child_indegree_not_multiplied=True,
            phase_derivative_error_certified=False)
    for p in (test, source):p.assert_current()
    return radii, kernel, bound


class DirectionalSparseFarActions:
    """Disjoint sparse contexts in full physical coordinates, sharing FAR actions."""
    _initialize = DirectionalNestedFarActions._initialize
    _validate_disjoint_rectangles = staticmethod(DirectionalNestedFarActions._validate_disjoint_rectangles)
    tensors = DirectionalNestedFarActions.tensors
    _state = DirectionalNestedFarActions._state
    assert_current = DirectionalNestedFarActions.assert_current
    _action = DirectionalNestedFarActions._action
    apply = DirectionalNestedFarActions.apply
    adjoint = DirectionalNestedFarActions.adjoint
    ideal_rwg_truncation_bound = DirectionalNestedFarActions.ideal_rwg_truncation_bound

    def __init__(self, test, source, medium, couplings, *, ideal_rwg_error_budget,
                 balancing_impedance=None, rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2):
        self._initialize(test, source, medium, couplings, ideal_rwg_error_budget=ideal_rwg_error_budget,
            balancing_impedance=balancing_impedance, rhs_batch=rhs_batch,
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, sparse=True)

    def receipt(self):
        record = DirectionalNestedFarActions.receipt(self)
        record.update(version='experimental_disjoint_sparse_directional_far_actions_v1',
            sparse_context_indices_separate_from_physical_nodes=True,
            caller_active_context_maps_only=True, automatic_global_cones=False)
        return record


class DirectionalDAGFarActions:
    """Disjoint material-bound DAG FAR blocks sharing physical action kernels."""
    _initialize = DirectionalNestedFarActions._initialize
    _validate_disjoint_rectangles = staticmethod(DirectionalNestedFarActions._validate_disjoint_rectangles)
    tensors = DirectionalNestedFarActions.tensors
    _state = DirectionalNestedFarActions._state
    assert_current = DirectionalNestedFarActions.assert_current
    _action = DirectionalNestedFarActions._action
    apply = DirectionalNestedFarActions.apply
    adjoint = DirectionalNestedFarActions.adjoint
    ideal_rwg_truncation_bound = DirectionalNestedFarActions.ideal_rwg_truncation_bound

    def __init__(self, test, source, medium, couplings, *, ideal_rwg_error_budget,
                 balancing_impedance=None, rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2):
        self._initialize(test, source, medium, couplings, ideal_rwg_error_budget=ideal_rwg_error_budget,
            balancing_impedance=balancing_impedance, rhs_batch=rhs_batch,
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes, sparse='dag')

    def receipt(self):
        record = DirectionalNestedFarActions.receipt(self)
        record.update(version='experimental_disjoint_shared_dag_directional_far_actions_v1',
            changing_carrier_DAG=True, phase_paths_included=True,
            sparse_context_indices_separate_from_physical_nodes=True,
            caller_active_context_maps_only=True, automatic_global_cones=False)
        return record
