"""Exact fixed-carrier center shifts with distinct source/receiver Hermitian maps."""
import math
import torch
from .fmm_expansion import _points, multi_indices
from .fmm_shift import SeparableMultipoleShift
from .fmm_fft import _rhs_chunks
from .direct_system import DirectSystemActions


class _DirectionalShiftMap:
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, owner, kind):
        self.owner, self.kind = owner, kind
        self._binding = (id(owner), kind)

    def assert_current(self):
        if (id(self.owner), self.kind) != self._binding:
            raise ValueError('Directional shift map changed')
        self.owner.assert_current()

    def _action(self, value, *, adjoint):
        self.assert_current()
        return self.owner._action(value, self.kind, adjoint=adjoint)


class DirectionalFixedCarrierShift:
    """d=child-parent: source=e^(-iku.d)S, receiver=e^(+iku.d)S.T.

    Fixed-current backward re-enters the complete Hermitian map without saving
    generated gathers/weights. Direction changes require separate admission.
    """
    def __init__(self, displacement, direction, wave_number, order, *, target_direction=None,
                 layout=None, rhs_batch=4, retained_budget_bytes=1024**2,
                 tensor_workspace_budget_bytes=128*1024**2):
        _points(displacement); _points(direction)
        target_direction = direction if target_direction is None else target_direction
        _points(target_direction)
        if (any(t.shape != (3,) or t.dtype != torch.float64 for t in
                (displacement, direction, target_direction))
                or displacement.device != direction.device or direction.device != target_direction.device):
            raise ValueError('Directional shifts require matching fixed CUDA float64 vectors')
        for carrier in (direction, target_direction):
            if not bool((carrier.norm()-1).abs() <= 512*torch.finfo(carrier.dtype).eps):
                raise ValueError('Directional shift carriers must be unit length without normalization')
        if not torch.equal(direction.contiguous().view(torch.int64), target_direction.contiguous().view(torch.int64)):
            raise ValueError('Fixed-carrier shifts reject direction changes before layout construction')
        if type(order) is not int or not 0 <= order <= 12:
            raise ValueError('Directional shift order must lie between zero and twelve')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Directional shift wave number must be a fixed passive scalar')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
            raise ValueError('Directional shift wave number must be finite and passive')
        if any(type(v) is not int or v < 1 for v in
               (rhs_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Directional shift batches and budgets must be positive integers')
        count = len(multi_indices(order)); layout_bytes = 6*count*(order+1)*24
        owns_layout = layout is None
        if not owns_layout:
            if (not isinstance(layout, SeparableMultipoleShift) or layout.order != order
                    or layout.device != displacement.device or layout.real_dtype != torch.float64):
                raise ValueError('Directional shift shared layout must match its native order/device/precision')
            layout.assert_current()
        expected = 104+(layout_bytes if owns_layout else 0)
        workspace = (2*layout_bytes+rhs_batch*count*(order+1)*4*16*8
                     +count*(order+1)*8*16+512)
        if expected > retained_budget_bytes:
            raise MemoryError('Directional shift retained budget exceeded before layout construction')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Directional shift tensor workspace exceeded before layout construction')
        phase_argument = 1j*wave*(displacement@direction)
        source_phase, receiver_phase = torch.exp(-phase_argument), torch.exp(phase_argument)
        if not bool(torch.isfinite(source_phase) & torch.isfinite(receiver_phase)
                    & (source_phase.abs() > 0) & (receiver_phase.abs() > 0)):
            raise ValueError('Directional center phases exceed native precision')
        self.displacement, self.direction = displacement.clone(), direction.clone()
        self.target_direction = target_direction.clone()
        self.source_phase, self.receiver_phase = source_phase, receiver_phase
        self.wave_number, self.order, self.count = wave, order, count
        self.device, self.dtype = displacement.device, torch.complex128
        self.rhs_batch, self.owns_layout = rhs_batch, owns_layout
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.planned_retained_bytes, self.planned_tensor_workspace_bytes = expected, workspace
        self.layout = (SeparableMultipoleShift(order, device=self.device, dtype=torch.float64)
                       if owns_layout else layout)
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected:
            raise RuntimeError('Directional shift retained inventory differs from plan')
        self.source_map = _DirectionalShiftMap(self, 'source')
        self.receiver_map = _DirectionalShiftMap(self, 'receiver')
        self._binding = self._state()

    def tensors(self):
        return (self.displacement, self.direction, self.target_direction, self.source_phase,
                self.receiver_phase, *(self.layout.tensors() if self.owns_layout else ()))

    def _state(self):
        return (self.wave_number, self.order, self.count, self.device, self.dtype, self.rhs_batch,
                self.owns_layout, id(self.layout), self.retained_budget_bytes,
                self.tensor_workspace_budget_bytes, self.planned_retained_bytes,
                self.planned_tensor_workspace_bytes, self.retained_bytes,
                id(self.source_map), self.source_map.kind, id(self.source_map.owner),
                id(self.receiver_map), self.receiver_map.kind, id(self.receiver_map.owner),
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Directional fixed-carrier shift changed')
        self.layout.assert_current()

    def _action(self, value, kind, *, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 2
                or value.shape[-2:] != (self.count, 4) or value.device != self.device
                or value.dtype != self.dtype):
            raise ValueError('Directional shift requires matching complex128 vector/divergence channels')
        leading = value.shape[:-2]
        rows = math.prod(leading)
        result = value.new_empty((rows, self.count, 4))
        phase = self.source_phase if kind == 'source' else self.receiver_phase
        phase = phase.conj() if adjoint else phase
        transposed = (kind == 'receiver') != adjoint
        for start, chunk in _rhs_chunks(value, 2, self.rhs_batch):
            shifted = self.layout.apply(self.displacement, chunk, adjoint=transposed)
            translated = shifted*phase
            if not bool(torch.isfinite(translated).all() & ~((shifted != 0) & (translated == 0)).any()):
                raise ValueError('Directional translated moments exceed native precision')
            result[start:start+len(chunk)] = translated
        return result.reshape(value.shape)

    def source(self, value, *, adjoint=False):
        if type(adjoint) is not bool:
            raise ValueError('Directional source adjoint flag must be boolean')
        return self.source_map.adjoint(value) if adjoint else self.source_map.apply(value)

    def receiver(self, value, *, adjoint=False):
        if type(adjoint) is not bool:
            raise ValueError('Directional receiver adjoint flag must be boolean')
        return self.receiver_map.adjoint(value) if adjoint else self.receiver_map.apply(value)

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_directional_fixed_carrier_shift_v1', order=self.order,
            center_displacement='child_minus_parent', source_shift='exp(-ik*u.d)*S(d)',
            receiver_shift='exp(+ik*u.d)*S(d).T', direction_changes_admitted=False,
            owned_layout_included=self.owns_layout, caller_shared_layout_excluded=not self.owns_layout,
            retained_bytes=self.retained_bytes, planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            rhs_batch=self.rhs_batch, no_generated_current_graph_saved=True,
            tensor_workspace_excludes=('allocator cache', 'caller input/output tensors', 'whole tree/system'),
            exact_arithmetic_polynomial_shift=True, floating_point_certified=False,
            directional_tree_integrated=False, physical_qualification=False,
            complexity_qualification=False, peak_memory_qualified=False)


class DirectionalChangingCarrierShift:
    """Padded phase Taylor transfer between fixed child and parent carriers.

    The ideal relative phase bound assumes complete support inside child_radius.
    It does not bound cancellation-relative currents, derivatives or EM kernels.
    Qminus maps child source moments to parent moments; Qplus.T maps physical
    parent receiver locals to child locals. Their complete Hermitian maps differ.
    """
    source = DirectionalFixedCarrierShift.source
    receiver = DirectionalFixedCarrierShift.receiver

    def __init__(self, displacement, direction, target_direction, wave_number,
                 source_order, target_order, phase_order, *, child_radius,
                 phase_relative_budget=1e-12, rhs_batch=4,
                 retained_budget_bytes=64*1024**2, tensor_workspace_budget_bytes=128*1024**2,
                 phase_storage='dense_v1',phase_layout=None,center_layout=None):
        for value in (displacement, direction, target_direction):
            _points(value)
        if (any(t.shape != (3,) or t.dtype != torch.float64 for t in
                (displacement, direction, target_direction))
                or any(t.device != displacement.device for t in (direction, target_direction))):
            raise ValueError('Changing carriers require matching fixed CUDA float64 vectors')
        if any(not bool((u.norm()-1).abs() <= 512*torch.finfo(u.dtype).eps)
               for u in (direction, target_direction)):
            raise ValueError('Changing carrier directions must be unit length without normalization')
        if (not isinstance(child_radius, torch.Tensor) or child_radius.shape != ()
                or child_radius.device != displacement.device or child_radius.dtype != torch.float64
                or child_radius.requires_grad or not bool(torch.isfinite(child_radius) & (child_radius > 0))):
            raise ValueError('Changing carrier support radius must be a positive fixed CUDA float64 scalar')
        if (any(type(p) is not int or not 0 <= p <= 12 for p in (source_order, target_order, phase_order))
                or source_order < target_order+phase_order):
            raise ValueError('Changing carrier source order must cover target plus phase order within zero to twelve')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Changing carrier wave number must be a fixed passive scalar')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
            raise ValueError('Changing carrier wave number must be finite and passive')
        if (type(phase_relative_budget) not in (int, float) or not math.isfinite(phase_relative_budget)
                or not 0 < phase_relative_budget < 1):
            raise ValueError('Changing carrier relative phase budget must be finite and between zero and one')
        if any(type(n) is not int or n < 1 for n in
               (rhs_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Changing carrier batches and budgets must be positive integers')
        source_count, target_count = len(multi_indices(source_order)), len(multi_indices(target_order))
        if phase_storage not in ('dense_v1','compact_phase_v1'):
            raise ValueError('Unsupported changing carrier phase storage')
        compact = phase_storage == 'compact_phase_v1'
        if not compact and (phase_layout is not None or center_layout is not None):
            raise ValueError('Dense changing carrier storage does not accept shared compact layouts')
        phase_count=len(multi_indices(phase_order))
        phase_bytes=8*target_count*phase_count+32*phase_count
        center_bytes=144*target_count*(target_order+1)
        if compact:
            if phase_layout is not None:
                if (not isinstance(phase_layout,ChangingCarrierPhaseLayout)
                        or (phase_layout.source_order,phase_layout.target_order,phase_layout.phase_order)
                        != (source_order,target_order,phase_order) or phase_layout.device != displacement.device):
                    raise ValueError('Shared changing phase layout must match native orders and device')
                phase_layout.assert_current()
            if center_layout is not None:
                if (not isinstance(center_layout,SeparableMultipoleShift) or center_layout.order != target_order
                        or center_layout.device != displacement.device or center_layout.real_dtype != torch.float64):
                    raise ValueError('Shared changing center layout must match native target order and device')
                center_layout.assert_current()
            expected=128+32*phase_count+(phase_bytes if phase_layout is None else 0)+(center_bytes if center_layout is None else 0)
            workspace=(3*(phase_bytes+center_bytes+expected)+rhs_batch*target_count*phase_count*4*16*8
                       +rhs_batch*target_count*(target_order+1)*4*16*8+target_count*phase_count*128+4096)
        else:
            expected = 128+32*source_count*target_count
            workspace = 3*expected+4096*1024+rhs_batch*(source_count+target_count)*4*16*8
        if expected > retained_budget_bytes:
            raise MemoryError('Changing carrier retained budget exceeded before transfer tables')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Changing carrier tensor workspace exceeded before transfer tables')
        delta = target_direction-direction
        scaled_delta = -1j*wave*delta
        if not bool(torch.isfinite(scaled_delta).all()) or (wave != 0 and
                bool(((delta != 0) & (scaled_delta == 0)).any())):
            raise ValueError('Changing carrier nonzero phase components exceed native precision')
        x = torch.linalg.vector_norm(delta)*abs(wave)*child_radius
        exact_phase = wave == 0 or torch.equal(direction, target_direction)
        if not bool(torch.isfinite(x)) or (not exact_phase and not bool(x > 0)):
            raise ValueError('Changing carrier phase argument overflows or underflows native precision')
        log_bound = torch.zeros_like(x) if exact_phase else 2*x+(phase_order+1)*x.log()-math.lgamma(phase_order+2)
        if not bool(torch.isfinite(log_bound)):
            raise ValueError('Changing carrier phase bound exceeds native precision')
        if not exact_phase and bool(log_bound > math.log(phase_relative_budget)):
            raise ValueError('Changing carrier ideal relative phase bound exceeds configured budget')
        argument = 1j*wave*(displacement@target_direction)
        source_phase, receiver_phase = torch.exp(-argument), torch.exp(argument)
        if not bool(torch.isfinite(source_phase) & torch.isfinite(receiver_phase)
                    & (source_phase.abs() > 0) & (receiver_phase.abs() > 0)):
            raise ValueError('Changing carrier center phases exceed native precision')
        self.displacement, self.direction, self.target_direction = (
            displacement.clone(), direction.clone(), target_direction.clone())
        self.child_radius, self.phase_argument, self.log_phase_bound = child_radius.clone(), x, log_bound
        self.source_phase, self.receiver_phase = source_phase, receiver_phase
        self.wave_number = wave
        self.source_order, self.target_order, self.phase_order = source_order, target_order, phase_order
        self.source_count, self.target_count = source_count, target_count
        self.phase_relative_budget, self.exact_phase = phase_relative_budget, exact_phase
        self.rhs_batch, self.device, self.dtype = rhs_batch, displacement.device, torch.complex128
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.planned_retained_bytes, self.planned_tensor_workspace_bytes = expected, workspace
        self.phase_storage,self.phase_count=phase_storage,phase_count
        self.owns_phase_layout,self.owns_center_layout=compact and phase_layout is None,compact and center_layout is None
        self.phase_layout,self.center_layout=phase_layout,center_layout
        if compact:
            self.phase_layout=ChangingCarrierPhaseLayout(source_order,target_order,phase_order,device=self.device) if self.owns_phase_layout else phase_layout
            self.center_layout=SeparableMultipoleShift(target_order,device=self.device,dtype=torch.float64) if self.owns_center_layout else center_layout
            self._build_compact()
        else:
            self._build_tables()
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected:
            raise RuntimeError('Changing carrier owned storage differs from preallocation plan')
        self.source_map, self.receiver_map = _DirectionalShiftMap(self, 'source'), _DirectionalShiftMap(self, 'receiver')
        self._binding = self._state()

    def _build_tables(self):
        # Host traversal contains polynomial integer metadata only. Physical
        # displacements, phases, coefficients and reductions stay on CUDA.
        lookup = {a:i for i,a in enumerate(multi_indices(self.source_order))}
        phase_indices = multi_indices(self.phase_order)
        targets = multi_indices(self.target_order)
        negative = torch.zeros((self.target_count*self.source_count,), device=self.device, dtype=self.dtype)
        positive = torch.zeros_like(negative)
        q = -1j*self.wave_number*(self.target_direction-self.direction)
        pending = []

        def flush():
            if not pending:
                return
            flat, powers_d, powers_q, factors = zip(*pending)
            flat = torch.tensor(flat, device=self.device, dtype=torch.int64)
            powers_d = torch.tensor(powers_d, device=self.device, dtype=torch.int64)
            powers_q = torch.tensor(powers_q, device=self.device, dtype=torch.int64)
            factors = torch.tensor(factors, device=self.device, dtype=torch.float64)
            structural = (((self.displacement != 0) | (powers_d == 0)).all(-1)
                          & (((self.target_direction != self.direction) & (self.wave_number != 0))
                             | (powers_q == 0)).all(-1))
            # Native complex tensor pow(0, 0) is nonfinite. Polynomial metadata
            # requires the algebraic value one, without evaluating that branch.
            phase_powers = torch.where(q == 0,torch.ones_like(q),q).pow(powers_q)
            phase_powers = torch.where((q == 0) & (powers_q != 0),torch.zeros_like(phase_powers),phase_powers)
            base = factors*self.displacement.pow(powers_d).prod(-1)*phase_powers.prod(-1)
            minus = base*self.source_phase
            plus = base*(1-2*(powers_q.sum(-1)%2))*self.receiver_phase
            if not bool(torch.isfinite(minus).all() & torch.isfinite(plus).all()
                        & ~(structural & ((base == 0) | (minus == 0) | (plus == 0))).any()):
                raise ValueError('Changing carrier nonzero coefficient terms exceed native precision')
            negative.index_add_(0, flat, minus)
            positive.index_add_(0, flat, plus)
            pending.clear()

        for row, alpha in enumerate(targets):
            for gamma in targets:
                if any(g > a for g,a in zip(gamma,alpha)):
                    continue
                powers_d = tuple(a-g for a,g in zip(alpha,gamma))
                binomial = math.prod(math.comb(a,g) for a,g in zip(alpha,gamma))
                for eta in phase_indices:
                    beta = tuple(g+e for g,e in zip(gamma,eta))
                    flat = row*self.source_count+lookup[beta]
                    factor = binomial/math.prod(math.factorial(e) for e in eta)
                    pending.append((flat,powers_d,eta,factor))
                    if len(pending) == 4096:
                        flush()
        flush()
        self.source_coefficients = negative.reshape(self.target_count,self.source_count)
        self.receiver_coefficients = positive.reshape(self.target_count,self.source_count)
        if not bool(torch.isfinite(self.source_coefficients).all() & torch.isfinite(self.receiver_coefficients).all()):
            raise ValueError('Changing carrier accumulated transfer tables exceed native precision')

    def tensors(self):
        return (self.displacement, self.direction, self.target_direction, self.child_radius,
                self.phase_argument, self.log_phase_bound, self.source_phase, self.receiver_phase,
                self.source_coefficients, self.receiver_coefficients,
                *(self.phase_layout.tensors() if self.owns_phase_layout else ()),
                *(self.center_layout.tensors() if self.owns_center_layout else ()))

    def _state(self):
        return (self.wave_number, self.source_order, self.target_order, self.phase_order,
                self.source_count, self.target_count, self.phase_relative_budget, self.exact_phase,
                self.phase_storage,self.phase_count,self.owns_phase_layout,self.owns_center_layout,
                id(self.phase_layout),id(self.center_layout),
                self.rhs_batch, self.device, self.dtype, self.retained_budget_bytes,
                self.tensor_workspace_budget_bytes, self.planned_retained_bytes,
                self.planned_tensor_workspace_bytes, self.retained_bytes,
                id(self.source_map), self.source_map.kind, id(self.source_map.owner),
                id(self.receiver_map), self.receiver_map.kind, id(self.receiver_map.owner),
                tuple((id(t),t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Directional changing-carrier shift changed')
        if self.phase_storage == 'compact_phase_v1':
            self.phase_layout.assert_current();self.center_layout.assert_current()

    def _action(self, value, kind, *, adjoint):
        self.assert_current()
        source = kind == 'source'
        input_count = self.source_count if source != adjoint else self.target_count
        output_count = self.target_count if source != adjoint else self.source_count
        if (not isinstance(value, torch.Tensor) or value.ndim < 2
                or value.shape[-2:] != (input_count,4) or value.device != self.device or value.dtype != self.dtype):
            raise ValueError('Changing carrier action requires matching complex128 vector/divergence channels')
        if self.phase_storage == 'compact_phase_v1':
            return self._compact_action(value,kind,adjoint=adjoint)
        matrix = self.source_coefficients if source else self.receiver_coefficients.T
        if adjoint:
            matrix = matrix.mH
        result = value.new_empty((math.prod(value.shape[:-2]),output_count,4))
        for start,chunk in _rhs_chunks(value,2,self.rhs_batch):
            translated = matrix@chunk
            if not bool(torch.isfinite(translated).all()):
                raise ValueError('Changing carrier translated moments exceed native precision')
            result[start:start+len(chunk)] = translated
        return result.reshape(*value.shape[:-2],output_count,4)

    def receipt(self):
        self.assert_current()
        record = dict(version='experimental_directional_changing_carrier_shift_v1',
            source_order=self.source_order,target_order=self.target_order,phase_order=self.phase_order,
            source_count=self.source_count,target_count=self.target_count,exact_phase=self.exact_phase,
            child_support_radius=float(self.child_radius), phase_argument=float(self.phase_argument),
            ideal_relative_phase_log_bound=None if self.exact_phase else float(self.log_phase_bound),
            phase_relative_budget=self.phase_relative_budget,
            admission_scope='ideal relative phase only assuming complete child support enclosed by radius',
            phase_derivative_error_certified=False,relative_current_error_certified=False,
            source_receiver_independent_signs=True,source_order_padding_required=True,
            retained_bytes=self.retained_bytes,planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            rhs_batch=self.rhs_batch,no_generated_current_graph_saved=True,
            tensor_workspace_excludes=('caller input/output','allocator workspace/cache','whole tree/system'),
            floating_point_certified=False,directional_tree_integrated=False,cone_tree_integrated=False,
            kernel_error_certified=False,physical_qualification=False,complexity_qualification=False,
            peak_memory_qualified=False,production_qualification=False)
        if self.phase_storage == 'compact_phase_v1':
            record.update(version='experimental_directional_compact_changing_carrier_shift_v1',
                phase_storage=self.phase_storage,owned_phase_layout_included=self.owns_phase_layout,
                owned_center_layout_included=self.owns_center_layout,
                phase_coefficients_per_sign=self.phase_count,phase_factorization_exact_arithmetic=True,
                native_order_or_admission_range_identical_to_dense=False)
        return record

    def _build_compact(self):
        powers=self.phase_layout.powers
        q=-1j*self.wave_number*(self.target_direction-self.direction)
        values=torch.where(q == 0,torch.ones_like(q),q).pow(powers)
        values=torch.where((q == 0) & (powers != 0),torch.zeros_like(values),values)
        minus=values.prod(-1)*self.phase_layout.inverse_factorials
        plus=minus*(1-2*(powers.sum(-1)%2))
        self.source_coefficients,self.receiver_coefficients=minus,plus
        # Check every structural center/phase coefficient product. A factorized
        # representation can have a different native range than a dense sum.
        dp=torch.tensor(multi_indices(self.target_order),device=self.device,dtype=torch.int64)
        real=self.displacement.pow(dp).prod(-1)
        structural_d=((self.displacement != 0) | (dp == 0)).all(-1)
        structural_q=(((self.target_direction != self.direction) & (self.wave_number != 0)) | (powers == 0)).all(-1)
        structural=structural_d[:,None] & structural_q[None]
        a=real[:,None]*minus[None]*self.source_phase
        b=real[:,None]*plus[None]*self.receiver_phase
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()
                    & ~(structural & ((a == 0) | (b == 0))).any()):
            raise ValueError('Compact changing carrier coefficient factors exceed native precision')

    def _phase_apply(self,value,coefficients,*,transpose):
        indices=self.phase_layout.indices
        if transpose:
            products=value[:,:,None,:]*coefficients[None,None,:,None]
            if bool(((value[:,:,None,:] != 0) & (coefficients[None,None,:,None] != 0) & (products == 0)).any()):
                raise ValueError('Compact nonzero phase products underflow native precision')
            result=value.new_zeros((len(value),self.source_count,4))
            result.index_add_(1,indices.flatten(),products.flatten(1,2))
        else:
            gathered=value[:,indices,:]
            products=gathered*coefficients[None,None,:,None]
            if bool(((gathered != 0) & (coefficients[None,None,:,None] != 0) & (products == 0)).any()):
                raise ValueError('Compact nonzero phase products underflow native precision')
            result=products.sum(-2)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Compact phase accumulation exceeds native precision')
        return result

    def _compact_action(self,value,kind,*,adjoint):
        source=kind == 'source'
        output_count=self.target_count if source != adjoint else self.source_count
        result=value.new_empty((math.prod(value.shape[:-2]),output_count,4))
        coefficients=self.source_coefficients if source else self.receiver_coefficients
        phase=self.source_phase if source else self.receiver_phase
        if adjoint:
            coefficients,phase=coefficients.conj(),phase.conj()
        for start,chunk in _rhs_chunks(value,2,self.rhs_batch):
            if source != adjoint:
                phased=self._phase_apply(chunk,coefficients,transpose=False)
                shifted=self.center_layout.apply(self.displacement,phased,reject_underflow=True)
            else:
                centered=self.center_layout.apply(self.displacement,chunk,adjoint=True,reject_underflow=True)
                shifted=self._phase_apply(centered,coefficients,transpose=True)
            translated=shifted*phase
            if not bool(torch.isfinite(translated).all() & ~((shifted != 0) & (translated == 0)).any()):
                raise ValueError('Compact changing carrier translated moments exceed native precision')
            result[start:start+len(chunk)]=translated
        return result.reshape(*value.shape[:-2],output_count,4)


class ChangingCarrierPhaseLayout:
    """Fixed padded polynomial gather indices, shared independently of carriers."""
    def __init__(self,source_order,target_order,phase_order,*,device):
        if (any(type(p) is not int or not 0 <= p <= 12 for p in (source_order,target_order,phase_order))
                or source_order < target_order+phase_order):
            raise ValueError('Changing phase layout requires supported padded native orders')
        device=torch.device(device)
        if device.type != 'cuda':raise ValueError('Changing phase layout requires CUDA')
        source=multi_indices(source_order);target=multi_indices(target_order);phase=multi_indices(phase_order)
        lookup={a:i for i,a in enumerate(source)}
        self.indices=torch.tensor([[lookup[tuple(a+e for a,e in zip(alpha,eta))] for eta in phase]
            for alpha in target],device=device,dtype=torch.int64)
        self.powers=torch.tensor(phase,device=device,dtype=torch.int64)
        self.inverse_factorials=torch.tensor([1/math.prod(math.factorial(e) for e in eta) for eta in phase],
            device=device,dtype=torch.float64)
        self.source_order,self.target_order,self.phase_order=source_order,target_order,phase_order
        self.source_count,self.target_count,self.phase_count=len(source),len(target),len(phase)
        self.device=self.indices.device
        self._binding=self._state()

    def tensors(self):return self.indices,self.powers,self.inverse_factorials

    def _state(self):
        return (self.source_order,self.target_order,self.phase_order,self.source_count,self.target_count,
            self.phase_count,self.device,tuple((id(t),t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:raise ValueError('Changing carrier phase layout changed')
