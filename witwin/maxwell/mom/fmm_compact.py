"""Cartesian M2L stored as Taylor coefficients with bounded row-table generation."""
import math
import torch
from .fmm_expansion import helmholtz_taylor, multi_indices, _points
from .physics import Medium
from .fmm_storage import m2l_storage_bytes
from .scalar_moment_action import scalar_moment_action
from .direct_system import DirectSystemActions


def _coefficient_index(alpha):
    degree = alpha.sum(-1)
    i, j = alpha[..., 0], alpha[..., 1]
    return degree*(degree+1)*(degree+2)//6+i*(degree+1)-i*(i-1)//2+j


class CompactElectromagneticM2L:
    """Same polynomial translation, retaining O(p^3) instead of O(p^6) values.

    Arithmetic remains dense expansion algebra, generated in bounded target
    row batches. This is a storage representation, not a faster asymptotic
    translation algorithm or a truncation/physical certificate.
    """
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, displacement, medium, order, *, row_batch=16, retained_budget_bytes=512*1024**2):
        _points(displacement)
        if displacement.shape != (3,):
            raise ValueError('Compact M2L requires one fixed center displacement')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Compact electromagnetic M2L requires a finite-frequency medium')
        if type(order) is not int or order < 0:
            raise ValueError('Compact M2L order must be a nonnegative integer')
        for value in (row_batch, retained_budget_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('Compact M2L row batch and retained budget must be positive integers')
        expected = m2l_storage_bytes(order, displacement.element_size(), 'compact')
        if expected > retained_budget_bytes:
            raise MemoryError('Compact M2L retained budget exceeded before coefficient construction')
        self._bind_coefficients(helmholtz_taylor(displacement, medium.k, 2*order+1), medium, order, row_batch)

    @classmethod
    def create_many(cls, displacements, medium, order, *, row_batch=16,
                    retained_budget_bytes=512*1024**2):
        """Bound coefficient workspace to32centers, retaining independent layouts.

        Each coefficient vector is copied into its own exact-size allocation;
        no hidden batch storage survives through a coupling's tensor views.
        Polynomial algebra, retained capacity and scalar construction are unchanged.
        """
        _points(displacements)
        if displacements.ndim != 2:
            raise ValueError('Compact M2L batch requires fixed center displacements')
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Compact electromagnetic M2L requires a finite-frequency medium')
        if type(order) is not int or order < 0:
            raise ValueError('Compact M2L order must be a nonnegative integer')
        for value in (row_batch, retained_budget_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('Compact M2L row batch and retained budget must be positive integers')
        expected = m2l_storage_bytes(order, displacements.element_size(), 'compact')
        if len(displacements)*expected > retained_budget_bytes:
            raise MemoryError('Compact M2L batch retained budget exceeded before coefficient construction')
        if not len(displacements):
            return ()
        layout = _coefficient_layout(order, displacements.device, displacements.dtype)
        result = []
        for start in range(0, len(displacements), 32):
            coefficients = helmholtz_taylor(displacements[start:start+32], medium.k, 2*order+1)
            for vector in coefficients:
                obj = cls.__new__(cls)
                obj._bind_coefficients(vector.clone(), medium, order, row_batch, layout=layout)
                result.append(obj)
            # Release the batch and its final vector view before the next call.
            del vector, coefficients
        return tuple(result)

    def _bind_coefficients(self, coefficients, medium, order, row_batch, *, layout=None):
        self.coefficients = coefficients
        if layout is None:
            self.indices, self.binomial = _coefficient_layout(order, coefficients.device, coefficients.real.dtype)
        else:
            # Preserve exact-size independent metadata ownership and mutation checks.
            self.indices, self.binomial = (tensor.clone() for tensor in layout)
        self.medium, self.order, self.count, self.row_batch = medium, order, math.comb(order+3, 3), row_batch
        self.device, self.dtype = coefficients.device, coefficients.dtype
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != m2l_storage_bytes(order, coefficients.real.element_size(), 'compact'):
            raise RuntimeError('Compact M2L storage differs from structural plan')
        self._binding = self._state()

    def tensors(self):
        return (self.coefficients, self.indices, self.binomial)

    def reversed_translation(self):
        """Borrow fixed coefficients for the opposite free-space displacement.

        Radial Green derivatives at -r have degree parity. Generated Green
        rows use that parity; gradient rows have one additional minus sign.
        This is ordinary displacement reversal, independent of the Hermitian
        current action. No numerical tensor is allocated or retained here.
        """
        self.assert_current()
        if hasattr(self, '_reverse_owner'):
            raise ValueError('Compact reverse translation requires an original owner')
        if self.dtype != torch.complex128 or any(t.requires_grad for t in self.tensors()):
            raise ValueError('Compact reverse translation requires fixed native complex128 tensors')
        result = type(self).__new__(type(self))
        for name in ('coefficients', 'indices', 'binomial', 'medium', 'order',
                     'count', 'row_batch', 'device', 'dtype', 'retained_bytes'):
            setattr(result, name, getattr(self, name))
        result._reverse_owner = self
        result._binding = result._state()
        return result

    def _state(self):
        return (id(self.medium), self.medium.frequency, self.medium.epsilon_r, self.medium.mu_r,
            self.medium.k, self.medium.eta, self.order, self.count, self.row_batch, self.device,
            self.dtype, self.retained_bytes, tuple((id(t), t._version) for t in self.tensors())) + (
                (id(self._reverse_owner), tuple(t.requires_grad for t in self.tensors()))
                if hasattr(self, '_reverse_owner') else ())

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Compact M2L changed')
        if hasattr(self, '_reverse_owner'):
            self._reverse_owner.assert_current()

    def _tables(self, start, stop):
        return compact_tables(self, start, stop)

    def _action(self, value, *, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 3 or value.shape[-3:] != (2, self.count, 4)
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Compact M2L requires finite matching complex moment channels')
        return compact_moment_action(self, self.medium, value, adjoint=adjoint)

    def receipt(self):
        self.assert_current()
        return dict(version=('compact_reciprocal_cartesian_m2l_coefficients_v1'
                            if hasattr(self, '_reverse_owner') else 'compact_cartesian_m2l_coefficients_v1'), order=self.order,
            **({} if not hasattr(self, '_reverse_owner') else dict(
                coefficient_owner='borrowed_original_translation',
                displacement_rule='degree_parity_green_extra_minus_gradient',
                additional_owned_tensor_bytes=0, production_qualification=False)),
            retained_bytes=self.retained_bytes, dense_table_bytes=4*self.count**2*self.coefficients.element_size(),
            row_batch=self.row_batch,
            generated_table_bytes=4*min(self.row_batch, self.count)*self.count*self.coefficients.element_size(),
            current_backward='fixed_operator_adjoint_no_saved_generated_tables',
            peak_excludes=['gather indices and weights', 'polynomial workspace', 'action intermediates', 'allocator overhead'],
            physical_qualification=False, complexity_qualification=False, peak_memory_qualified=False)


def _coefficient_layout(order, device, real_dtype):
    """Build fixed integer indexing and real binomial weights once per call."""
    indices = torch.tensor(multi_indices(order), device=device, dtype=torch.int64)
    binomial = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(order+1)]
                             for n in range(2*order+2)], device=device, dtype=real_dtype)
    if not bool(torch.isfinite(binomial).all()):
        raise ValueError('Compact M2L binomial weights exceed native precision')
    return indices, binomial


def compact_tables(operator, start, stop):
    alpha = operator.indices[None, :, :]
    total = operator.indices[start:stop, None, :]+alpha
    factor = operator.binomial[total, alpha].prod(-1)*(1-2*(alpha.sum(-1)%2))
    if hasattr(operator, '_reverse_owner'):
        factor = factor*(1-2*(total.sum(-1)%2))
    green = operator.coefficients[_coefficient_index(total)]*factor
    gradient = []
    for axis in range(3):
        shifted = total.clone()
        shifted[..., axis] += 1
        gradient.append(operator.coefficients[_coefficient_index(shifted)]*factor*(total[..., axis]+1))
    gradient = torch.stack(gradient, dim=-1)
    if hasattr(operator, '_reverse_owner'):
        gradient = -gradient
    if not bool(torch.isfinite(green).all() & torch.isfinite(gradient).all()):
        raise ValueError('Generated compact M2L table exceeds native precision')
    return green, gradient


def compact_moment_action(operator, medium, value, *, adjoint):
    result = torch.zeros_like(value) if adjoint else []
    for start in range(0, operator.count, operator.row_batch):
        stop = min(start+operator.row_batch, operator.count)
        green, gradient = operator._tables(start, stop)
        selected = value[..., start:stop, :] if adjoint else value
        block = scalar_moment_action(green, gradient, medium, selected, adjoint=adjoint)
        if adjoint:
            result = result+block
        else:
            result.append(block)
    return result if adjoint else torch.cat(result, dim=-2)


def compact_batch_workspace_bytes(order, pairs, rhs_count, row_batch):
    """Conservative visible tensor bound; allocator/library peaks are separate."""
    if (type(order) is not int or not 0 <= order <= 12
            or type(pairs) is not int or not 1 <= pairs <= 32
            or type(rhs_count) is not int or not 1 <= rhs_count <= 4
            or type(row_batch) is not int or row_batch < 1):
        raise ValueError('Compact FAR batch requires native order0..12, pairs1..32 and RHS1..4')
    count = math.comb(order+3, 3)
    rows = min(row_batch, 16, count)
    return (256*pairs*rows*count + 8*rhs_count*pairs*2*count*4*16
            + pairs*m2l_storage_bytes(order, 8, 'compact'))


def compact_moment_actions(operators, value, *, adjoint=False):
    """Apply<=32 original compact translations to<=4 physical RHS on CUDA.

    Coefficients/layouts remain owned by the original translations. All stacks,
    row tables and electromagnetic intermediates are temporary. This internal
    primitive is called inside the existing fixed-current autograd boundary.
    """
    operators = tuple(operators)
    if not operators or len(operators) > 32:
        raise ValueError('Compact FAR batch requires one through32 translations')
    if any(type(op) is not CompactElectromagneticM2L for op in operators):
        raise TypeError('Compact FAR batch requires original compact translations')
    first = operators[0]
    for op in operators:
        op.assert_current()
        if (op.medium is not first.medium or op.order != first.order
                or op.device != first.device or op.dtype != torch.complex128
                or op.row_batch != first.row_batch or any(t.requires_grad for t in op.tensors())):
            raise ValueError('Compact FAR batch requires fixed matching native complex128 operators')
    if (not isinstance(value, torch.Tensor) or value.ndim != 5
            or value.shape[1:] != (len(operators),2,first.count,4)
            or value.device != first.device or value.dtype != torch.complex128
            or value.device.type != 'cuda' or not 1 <= value.shape[0] <= 4):
        raise ValueError('Compact FAR batch requires finite matching CUDA current moments')
    required = compact_batch_workspace_bytes(first.order,len(operators),len(value),first.row_batch)
    if required > 128*1024**2:
        raise MemoryError('Compact FAR visible tensor workspace exceeds128MiB before stacks')
    if torch.is_grad_enabled() and value.requires_grad:
        raise ValueError('Compact FAR batching requires the fixed-current autograd boundary')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Compact FAR batch requires finite CUDA current moments')
    indices = torch.stack([op.indices for op in operators])
    binomial = torch.stack([op.binomial for op in operators])
    coefficients = torch.stack([op.coefficients for op in operators])
    reverse = torch.tensor([hasattr(op,'_reverse_owner') for op in operators],device=value.device)
    batch = torch.arange(len(operators),device=value.device)
    result = torch.zeros_like(value) if adjoint else []
    for start in range(0,first.count,min(first.row_batch,16)):
        stop = min(start+min(first.row_batch,16),first.count)
        alpha = indices[:,None,:,:]
        total = indices[:,start:stop,None,:]+alpha
        factor = binomial[batch[:,None,None,None],total,alpha].prod(-1)*(1-2*(alpha.sum(-1)%2))
        factor = factor*torch.where(reverse[:,None,None],1-2*(total.sum(-1)%2),1)
        green = coefficients[batch[:,None,None],_coefficient_index(total)]*factor
        gradient = []
        for axis in range(3):
            shifted = total.clone();shifted[...,axis] += 1
            gradient.append(coefficients[batch[:,None,None],_coefficient_index(shifted)]*factor*(total[...,axis]+1))
        gradient = torch.stack(gradient,dim=-1)
        gradient = gradient*torch.where(reverse[:,None,None,None],-1,1)
        if not bool(torch.isfinite(green).all() & torch.isfinite(gradient).all()):
            raise ValueError('Generated compact FAR batch exceeds native precision')
        selected = value[...,start:stop,:] if adjoint else value
        block = scalar_moment_action(green,gradient,first.medium,selected,adjoint=adjoint)
        result = result+block if adjoint else result+[block]
        del green,gradient,block,total,factor,shifted
    for op in operators:op.assert_current()
    return result if adjoint else torch.cat(result,dim=-2)


class SignedAxisCompactElectromagneticM2L:
    """Borrow one original radial jet under an exact signed axis permutation.

    Coordinate derivatives transform by multi-index permutation and parity.
    Gradient components use the same permutation and its extra axis sign.
    Physical moment channels stay in their original global coordinate frame.
    No coefficient/layout tensor is copied or retained by this view.
    """
    apply = CompactElectromagneticM2L.apply
    adjoint = CompactElectromagneticM2L.adjoint
    tensors = CompactElectromagneticM2L.tensors
    _action = CompactElectromagneticM2L._action

    def __init__(self, owner, permutation, signs):
        if type(owner) is not CompactElectromagneticM2L or hasattr(owner,'_reverse_owner'):
            raise TypeError('Signed-axis translation requires an original compact owner')
        owner.assert_current()
        if (owner.dtype!=torch.complex128 or owner.device.type!='cuda'
                or any(t.requires_grad for t in owner.tensors())):
            raise ValueError('Signed-axis translation requires fixed native CUDA complex128 tensors')
        if (type(permutation) is not tuple or len(permutation)!=3
                or any(type(axis) is not int for axis in permutation)
                or sorted(permutation)!=[0,1,2]
                or type(signs) is not tuple or len(signs)!=3
                or any(type(sign) is not int or sign not in (-1,1) for sign in signs)):
            raise ValueError('Signed-axis translation requires an exact axis permutation and signs')
        for name in ('coefficients','indices','binomial','medium','order','count',
                     'row_batch','device','dtype','retained_bytes'):
            setattr(self,name,getattr(owner,name))
        self._signed_axis_owner=owner
        self.axis_permutation=permutation
        self.axis_signs=signs
        self._binding=self._state()

    def _state(self):
        return CompactElectromagneticM2L._state(self)+(id(self._signed_axis_owner),self.axis_permutation,
            self.axis_signs,tuple(t.requires_grad for t in self.tensors()))

    def assert_current(self):
        CompactElectromagneticM2L.assert_current(self)
        self._signed_axis_owner.assert_current()

    def _tables(self, start, stop):
        alpha=self.indices[None,:,:]
        total=self.indices[start:stop,None,:]+alpha
        factor=self.binomial[total,alpha].prod(-1)*(1-2*(alpha.sum(-1)%2))
        for axis,sign in enumerate(self.axis_signs):
            if sign<0:factor=factor*(1-2*(total[...,axis]%2))
        inverse=[self.axis_permutation.index(axis) for axis in range(3)]
        transformed=total[...,inverse]
        green=self.coefficients[_coefficient_index(transformed)]*factor
        gradient=[]
        for axis,owner_axis in enumerate(self.axis_permutation):
            shifted=transformed.clone();shifted[...,owner_axis]+=1
            gradient.append(self.coefficients[_coefficient_index(shifted)]*factor
                *(total[...,axis]+1)*self.axis_signs[axis])
        gradient=torch.stack(gradient,dim=-1)
        if not bool(torch.isfinite(green).all() & torch.isfinite(gradient).all()):
            raise ValueError('Generated signed-axis M2L table exceeds native precision')
        return green,gradient

    def _temporary_original_translation(self, jet_indices):
        """Generate one temporary jet in actual axes for the existing batch action."""
        inverse=[self.axis_permutation.index(axis) for axis in range(3)]
        transformed=jet_indices[...,inverse]
        coefficients=self.coefficients[_coefficient_index(transformed)]
        for axis,sign in enumerate(self.axis_signs):
            if sign<0:coefficients=coefficients*(1-2*(jet_indices[...,axis]%2))
        result=CompactElectromagneticM2L.__new__(CompactElectromagneticM2L)
        for name in ('indices','binomial','medium','order','count','row_batch',
                     'device','dtype','retained_bytes'):
            setattr(result,name,getattr(self,name))
        result.coefficients=coefficients
        result._binding=result._state()
        return result

    def receipt(self):
        self.assert_current()
        return dict(version='signed_axis_compact_cartesian_m2l_view_v1',order=self.order,
            coefficient_owner='borrowed_original_physical_displacement',
            axis_permutation=self.axis_permutation,axis_signs=self.axis_signs,
            additional_owned_tensor_bytes=0,borrowed_native_bytes=self.retained_bytes,
            gradient_rule='permuted_coordinate_derivative_with_extra_axis_sign',
            moment_frame='unchanged_original_global_coordinates',
            current_backward='fixed_operator_adjoint_no_saved_generated_tables',
            public_integration_qualified=False,physical_qualification=False,
            complexity_qualification=False,peak_memory_qualified=False,
            production_qualification=False)


def signed_axis_compact_moment_actions(operators, value, *, adjoint=False,
                                      workspace_budget_bytes=128*1024**2):
    """Reuse the original bounded CUDA batch action with temporary actual jets.

    Original owners remain borrowed. A conservative visible workspace gate
    includes the temporary transformed jets before their construction; the
    existing action then enforces its original128MiB gate independently.
    Allocator, library workspace, host metadata and whole peaks are unqualified.
    """
    operators=tuple(operators)
    if (type(workspace_budget_bytes) is not int
            or not 1<=workspace_budget_bytes<=128*1024**2):
        raise ValueError('Signed-axis workspace budget must be positive and at most128MiB')
    if (not operators or len(operators)>32
            or any(type(op) is not SignedAxisCompactElectromagneticM2L for op in operators)):
        raise ValueError('Signed-axis batch requires one through32 signed-axis views')
    first=operators[0]
    for op in operators:
        op.assert_current()
        if (op.medium is not first.medium or op.order!=first.order
                or op.device!=first.device or op.dtype!=torch.complex128
                or op.row_batch!=first.row_batch):
            raise ValueError('Signed-axis batch requires matching fixed native translations')
    if (not isinstance(value,torch.Tensor) or value.ndim!=5
            or value.shape[1:]!=(len(operators),2,first.count,4)
            or value.device!=first.device or value.dtype!=torch.complex128
            or not 1<=value.shape[0]<=4):
        raise ValueError('Signed-axis batch requires finite matching CUDA moment channels')
    if torch.is_grad_enabled() and value.requires_grad:
        raise ValueError('Signed-axis batching requires the fixed-current autograd boundary')
    jets=math.comb(2*first.order+4,3)
    temporary_bytes=16*len(operators)*jets+64*jets*3
    required=compact_batch_workspace_bytes(first.order,len(operators),len(value),first.row_batch)+temporary_bytes
    if required>workspace_budget_bytes:
        raise MemoryError('Signed-axis visible tensor workspace budget exceeded before jets')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Signed-axis batch requires finite CUDA moment channels')
    jet_indices=torch.tensor(multi_indices(2*first.order+1),device=first.device,dtype=torch.int64)
    temporary=tuple(op._temporary_original_translation(jet_indices) for op in operators)
    result=compact_moment_actions(temporary,value,adjoint=adjoint)
    for op in operators:op.assert_current()
    return result


class _RadialSeriesM2LOwner:
    """Own the original univariate coefficients at one original native radius."""
    def __init__(self,displacement,medium,order,row_batch):
        from .fmm_expansion import _helmholtz_radial_series
        from .fmm_storage import radial_series_m2l_owner_bytes
        self.radius=torch.linalg.vector_norm(displacement).clone()
        radial=_helmholtz_radial_series(self.radius,complex(medium.k),2*order+1)
        self.radial=torch.stack(radial)
        self.indices,self.binomial=_coefficient_layout(order,displacement.device,torch.float64)
        self.medium,self.order,self.row_batch=medium,order,row_batch
        self.count=math.comb(order+3,3)
        self.device,self.dtype=displacement.device,torch.complex128
        self.retained_bytes=sum(t.untyped_storage().nbytes() for t in self.tensors())
        if self.retained_bytes!=radial_series_m2l_owner_bytes(order):
            raise RuntimeError('Radial series owner differs from its original native storage plan')
        if not bool(torch.isfinite(self.radial).all()):
            raise ValueError('Original radial series exceeds native precision')
        self._binding=self._state()

    def tensors(self):
        return self.radius,self.radial,self.indices,self.binomial

    def _state(self):
        m=self.medium
        return (id(m),m.frequency,m.epsilon_r,m.mu_r,m.k,m.eta,self.order,self.count,self.row_batch,
            self.device,self.dtype,self.retained_bytes,
            tuple((id(t),t._version,t.shape,t.device,t.dtype,t.requires_grad) for t in self.tensors()))

    def assert_current(self):
        if self._state()!=self._binding:
            raise ValueError('Radial series owner changed')


class RadialSeriesElectromagneticM2L:
    """Store original radial series and compose original native jets per action.

    Every view owns its exact original displacement. Radius owners share only
    equal native norm bits within one fixed medium/order/layout. Full Cartesian
    jets and row tables are temporary; no lower-order approximation is added.
    Setup, allocator/library peaks and complexity remain unqualified.
    """
    apply=CompactElectromagneticM2L.apply
    adjoint=CompactElectromagneticM2L.adjoint

    def __init__(self,owner,displacement):
        if type(owner) is not _RadialSeriesM2LOwner:
            raise TypeError('Radial view requires an original native radial owner')
        owner.assert_current();_points(displacement)
        if (displacement.shape!=(3,) or displacement.dtype!=torch.float64
                or displacement.device!=owner.device):
            raise ValueError('Radial view requires one original matching native displacement')
        radius=torch.linalg.vector_norm(displacement)
        if not torch.equal(radius.view(torch.int64),owner.radius.view(torch.int64)):
            raise ValueError('Radial view requires bit-identical original native radius')
        self.displacement=displacement.clone()
        self.owner=owner
        for name in ('medium','order','row_batch','count','device','dtype'):
            setattr(self,name,getattr(owner,name))
        self.retained_bytes=owner.retained_bytes+24
        self.workspace_budget_bytes=128*1024**2
        self._binding=self._state()

    @classmethod
    def create_many(cls,displacements,medium,order,*,row_batch=16,
                    retained_budget_bytes=512*1024**2):
        from .fmm_storage import radial_displacement_catalog,radial_series_m2l_owner_bytes
        _points(displacements)
        if displacements.ndim!=2 or displacements.dtype!=torch.float64:
            raise ValueError('Radial M2L requires original fixed native float64 vectors')
        if (not isinstance(medium,Medium) or medium.frequency<=0 or medium.k==0
                or not math.isfinite(complex(medium.k).real) or not math.isfinite(complex(medium.k).imag)
                or complex(medium.k).imag<0):
            raise ValueError('Radial M2L requires a finite passive scalar medium')
        native=radial_series_m2l_owner_bytes(order)
        if type(row_batch) is not int or not 1<=row_batch<=16:
            raise ValueError('Radial M2L row batch must lie in1..16')
        if type(retained_budget_bytes) is not int or retained_budget_bytes<1:
            raise ValueError('Radial M2L retained budget must be a positive integer')
        representatives,bindings=radial_displacement_catalog(displacements)
        required=len(representatives)*native+24*len(displacements)
        if required>retained_budget_bytes:
            raise MemoryError('Radial M2L retained budget exceeded before original radial coefficients')
        owners=tuple(_RadialSeriesM2LOwner(displacements[n],medium,order,row_batch) for n in representatives)
        result=tuple(cls(owners[index],delta) for index,delta in zip(bindings,displacements,strict=True))
        # Every view owns exactly three original real coordinates. The radius
        # owner metadata/series are shared; full Cartesian coefficients are absent.
        storages={(t.device,t.untyped_storage().data_ptr()):t.untyped_storage().nbytes()
                  for op in result for t in op.tensors()}
        if sum(storages.values())!=required:
            raise RuntimeError('Radial M2L native inventory differs from its preallocation plan')
        return result

    def tensors(self):
        return (*self.owner.tensors(),self.displacement)

    def _state(self):
        return (id(self.owner),id(self.medium),self.order,self.count,self.row_batch,self.device,self.dtype,
            self.retained_bytes,self.workspace_budget_bytes,
            tuple((id(t),t._version,t.shape,t.device,t.dtype,t.requires_grad) for t in self.tensors()))

    def assert_current(self):
        if self._state()!=self._binding:
            raise ValueError('Radial M2L view changed')
        self.owner.assert_current()

    def _action(self,value,*,adjoint):
        from .fmm_fft import _rhs_chunks
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<3 or value.shape[-3:]!=(2,self.count,4)
                or value.device!=self.device or value.dtype!=self.dtype):
            raise ValueError('Radial M2L requires matching native CUDA moment channels')
        rhs=math.prod(value.shape[:-3])
        radial_series_batch_plan(self.order,1,min(max(rhs,1),4),self.row_batch,
            workspace_budget_bytes=self.workspace_budget_bytes)
        output=torch.empty((rhs,2,self.count,4),device=self.device,dtype=self.dtype)
        for start,chunk in _rhs_chunks(value,3,4):
            result=radial_series_moment_actions((self,),chunk[:,None],adjoint=adjoint,
                workspace_budget_bytes=self.workspace_budget_bytes)
            output[start:start+len(chunk)]=result[:,0]
        self.assert_current()
        return output.reshape(value.shape)

    def receipt(self):
        self.assert_current()
        return dict(version='original_native_radial_series_cartesian_m2l_v1',order=self.order,
            owner_native_bytes=self.owner.retained_bytes,view_owned_native_bytes=24,
            standalone_retained_bytes=self.retained_bytes,
            owner_key='bit_identical_original_cuda_vector_norm_radius',
            displacement='exact_original_physical_vector',polynomial='original_radial_recurrence_and_composition',
            retained_cartesian_coefficient_bytes=0,row_batch=self.row_batch,
            rhs_chunk=4,requested_pair_batch_limit=32,workspace_budget_bytes=self.workspace_budget_bytes,
            action_chunk_admission='largest_pair_chunk_that_fits_before_composition',
            current_backward='original_full_Hermitian_no_saved_internal_graph',
            workspace_scope='conservative_visible_numerical_chunk; final_output_caller_and_whole_peaks_excluded',
            physical_qualification=False,complexity_qualification=False,peak_memory_qualified=False,
            public_integration_qualified=False,production_qualification=False)


def radial_series_batch_plan(order,pairs,rhs_count,row_batch,*,workspace_budget_bytes=128*1024**2):
    """Bound original composition plus the existing compact action before jets.

    Eight complex jet arrays cover original power/product/result, composition
    gathers/products, result accumulation and independent temporary vectors.
    All six original axis/degree integer shift pairs and index weights are
    included conservatively. This is not allocator/library or whole-process
    peak admission. A requested32-pair endpoint may require smaller chunks.
    """
    if (type(workspace_budget_bytes) is not int or not 1<=workspace_budget_bytes<=128*1024**2):
        raise ValueError('Radial M2L workspace must be positive and at most128MiB')
    compact_batch_workspace_bytes(order,pairs,rhs_count,row_batch)
    if row_batch>16:
        raise ValueError('Radial M2L row batch must lie in1..16')
    jets=math.comb(2*order+4,3)
    def required(n):
        return (compact_batch_workspace_bytes(order,n,rhs_count,row_batch)
                +8*16*n*jets+192*jets+32*n*(2*order+2)+64*n)
    chunk=pairs
    while chunk>0 and required(chunk)>workspace_budget_bytes:chunk-=1
    if not chunk:
        raise MemoryError('Radial M2L visible workspace exceeded before original composition')
    return dict(requested_pairs=pairs,admitted_pair_chunk=chunk,
        chunks=(pairs+chunk-1)//chunk,planned_visible_numerical_chunk_bytes=required(chunk),
        workspace_budget_bytes=workspace_budget_bytes,peak_memory_qualified=False,
        final_output_and_caller_inputs_excluded=True,production_qualification=False)


def _radial_series_temporary_translations(operators):
    """Compose exact original full jets for one already admitted native chunk."""
    from .fmm_expansion import _compose_radial_taylor
    displacements=torch.stack([op.displacement for op in operators])
    radii=torch.stack([op.owner.radius for op in operators])
    radial=torch.stack([op.owner.radial for op in operators])
    coefficients=_compose_radial_taylor(displacements,radii,radial.unbind(-1),2*operators[0].order+1)
    temporary=[]
    for op,vector in zip(operators,coefficients,strict=True):
        native=CompactElectromagneticM2L.__new__(CompactElectromagneticM2L)
        for name in ('medium','order','count','row_batch','device','dtype'):
            setattr(native,name,getattr(op,name))
        native.indices,native.binomial=op.owner.indices,op.owner.binomial
        native.coefficients=vector.clone()
        native.retained_bytes=m2l_storage_bytes(op.order,8,'compact')
        native._binding=native._state()
        temporary.append(native)
    return tuple(temporary)


def radial_series_moment_actions(operators,value,*,adjoint=False,workspace_budget_bytes=128*1024**2):
    """Reuse the original compact native action with admitted radial composition."""
    operators=tuple(operators)
    if (not operators or len(operators)>32
            or any(type(op) is not RadialSeriesElectromagneticM2L for op in operators)):
        raise ValueError('Radial batch requires one through32 original radial views')
    first=operators[0]
    for op in operators:
        op.assert_current()
        if (op.medium is not first.medium or op.order!=first.order or op.row_batch!=first.row_batch
                or op.device!=first.device or op.dtype!=torch.complex128):
            raise ValueError('Radial batch requires matching fixed native translations')
    if (not isinstance(value,torch.Tensor) or value.ndim!=5
            or value.shape[1:]!=(len(operators),2,first.count,4)
            or value.device!=first.device or value.dtype!=torch.complex128
            or not 1<=value.shape[0]<=4):
        raise ValueError('Radial batch requires matching original CUDA current moments')
    if torch.is_grad_enabled() and value.requires_grad:
        raise ValueError('Radial batching requires the fixed-current autograd boundary')
    plan=radial_series_batch_plan(first.order,len(operators),len(value),first.row_batch,
        workspace_budget_bytes=workspace_budget_bytes)
    chunk=plan['admitted_pair_chunk'];results=[]
    for start in range(0,len(operators),chunk):
        current=operators[start:start+chunk]
        selected=value[:,start:start+len(current)]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError('Radial batch requires finite CUDA current moments')
        temporary=_radial_series_temporary_translations(current)
        results.append(compact_moment_actions(temporary,selected,adjoint=adjoint))
        del temporary
    result=results[0] if len(results)==1 else torch.cat(results,dim=1)
    for op in operators:op.assert_current()
    return result
