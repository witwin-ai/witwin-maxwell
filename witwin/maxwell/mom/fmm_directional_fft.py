"""Internal centered directional coefficients and physical-gradient CUDA FFT."""
import math
from dataclasses import asdict
from fractions import Fraction
import torch
from .fmm_expansion import _points, multi_indices
from .fmm_fft import CartesianFFTTranslation
from .fmm_storage import m2l_storage_bytes
from .fmm_error import directional_multipole_truncation_bound, directional_multipole_truncation_bounds_many, directional_bound_workspace_bytes, MultipoleTruncationError
from .interpolation_error import KernelErrorBudget, KernelBoundUnavailable


def _validate_geometry(displacements, directions, wave_number, degree):
    _points(displacements)
    _points(directions)
    if (displacements.dtype != torch.float64 or directions.dtype != torch.float64
            or displacements.shape != directions.shape or displacements.ndim not in (1, 2)
            or displacements.device != directions.device):
        raise ValueError('Directional coefficients require matching fixed CUDA float64 centers and directions')
    if type(degree) is not int or not 0 <= degree <= 25:
        raise ValueError('Directional coefficient degree must lie between zero and twenty-five')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Directional wave number must be a fixed passive scalar')
    wave = complex(wave_number)
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
        raise ValueError('Directional wave number must be finite and passive')
    radius = torch.linalg.vector_norm(displacements, dim=-1)
    norm = torch.linalg.vector_norm(directions, dim=-1)
    if not bool((torch.isfinite(radius) & (radius > 0)).all()):
        raise ValueError('Directional centers must have finite positive separation')
    if not bool((norm-1).abs().le(512*torch.finfo(norm.dtype).eps).all()):
        raise ValueError('Directional carrier must be unit length; directions are never silently normalized')
    return wave, radius


def _coefficient_workspace_bytes(degree, batch):
    # Conservative visible coefficients, products, gathers, scales and shared
    # index layouts. Constants/layouts are setup only, never retained by objects.
    count = math.comb(degree+3, 3)
    return batch*count*16*24 + count*8*64


def _radial_constants(count):
    """Exact analytic constants of A_j(z), not a host numerical kernel path."""
    rows = [[Fraction(1)]]
    half = [Fraction(1)]
    for j in range(1, count+1):
        half.append(half[-1]*(Fraction(1, 2)-j+1)/j)
    for j in range(count):
        result = [Fraction(0)]*(j+2)
        for degree, coefficient in enumerate(rows[j]):
            result[degree] -= (1+2*j)*coefficient
            result[degree+1] += coefficient
        for m in range(1, j+1):
            for degree, coefficient in enumerate(rows[j-m]):
                result[degree+1] += half[m]*coefficient
        rows.append([c/(2*(j+1)) for c in result])
    return rows


def _shift_layout(indices, lookup, shift, degree, device):
    origins, destinations = [], []
    for index, alpha in enumerate(indices):
        beta = tuple(a+b for a, b in zip(alpha, shift))
        if sum(beta) <= degree:
            origins.append(index)
            destinations.append(lookup[beta])
    return (torch.tensor(origins, device=device, dtype=torch.long),
            torch.tensor(destinations, device=device, dtype=torch.long))


def _multiply_shifts(value, layouts, coefficients):
    result = torch.zeros_like(value)
    for (origin, destination), coefficient in zip(layouts, coefficients, strict=True):
        # A shift has unique destinations. Different monomials accumulate in
        # fixed order, without a coefficient-count squared map or atomic sum.
        result[:, destination] += value[:, origin]*coefficient[:, None]
    return result


def directional_helmholtz_taylor(displacements, directions, wave_number, degree, *,
                                 tensor_workspace_budget_bytes=128*1024**2):
    """Stable centered exp(-ik*u.delta)*G(R+delta) coefficients, at most32 centers.

    Exact radial carrier removal precedes a separable residual-direction phase.
    Native cancellation/phase-range accuracy is not certified by finite output.
    """
    wave, radius = _validate_geometry(displacements, directions, wave_number, degree)
    single = displacements.ndim == 1
    centers = displacements[None] if single else displacements
    carriers = directions[None] if single else directions
    radii = radius[None] if single else radius
    batch = len(centers)
    if batch > 32:
        raise ValueError('Directional native coefficient batches must not exceed thirty-two')
    if type(tensor_workspace_budget_bytes) is not int or tensor_workspace_budget_bytes < 1:
        raise ValueError('Directional coefficient workspace budget must be a positive integer')
    if _coefficient_workspace_bytes(degree, batch) > tensor_workspace_budget_bytes:
        raise MemoryError('Directional coefficient workspace exceeded before expansions')
    indices = multi_indices(degree)
    if not batch:
        return centers.new_empty((0, len(indices)), dtype=torch.complex128)
    degrees = torch.tensor([sum(alpha) for alpha in indices], device=centers.device)
    physical_powers = radii[:, None].pow(degrees)
    if not bool((torch.isfinite(physical_powers) & (physical_powers > 0)).all()):
        raise ValueError('Directional physical scale powers exceed native precision')
    lookup = {alpha: i for i, alpha in enumerate(indices)}
    unit = centers/radii[:, None]
    linear = [unit[:, axis] for axis in range(3)]
    shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    linear_layouts = [_shift_layout(indices, lookup, shift, degree, centers.device) for shift in shifts]
    quadratic_shifts = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    quadratic = [1-unit[:, axis].square() for axis in range(3)]
    quadratic += [-2*unit[:, a]*unit[:, b] for a, b in ((0, 1), (0, 2), (1, 2))]
    quadratic_layouts = [_shift_layout(indices, lookup, shift, degree, centers.device) for shift in quadratic_shifts]
    radial = _radial_constants(degree//2)
    z = 1j*wave*radii
    z_powers = [torch.ones_like(z)]
    for _ in range(degree//2):
        z_powers.append(z_powers[-1]*z)
    result = torch.zeros((batch, len(indices)), device=centers.device, dtype=torch.complex128)
    transverse = torch.zeros_like(result); transverse[:, 0] = 1
    for j, constants in enumerate(radial):
        polynomial = transverse
        for b in range(degree-2*j+1):
            factor = sum(float(c)*((-1)**b)*math.comb(2*j-l+b, b)*z_powers[l]
                         for l, c in enumerate(constants))
            result += polynomial*factor[:, None]
            if b < degree-2*j:
                polynomial = _multiply_shifts(polynomial, linear_layouts, linear)
        if j < degree//2:
            transverse = _multiply_shifts(transverse, quadratic_layouts, quadratic)
    residual = 1j*wave*radii[:, None]*(unit-carriers)
    for axis in range(3):
        if bool((residual[:, axis] == 0).all()):
            continue
        updated = torch.zeros_like(result)
        phase = torch.ones_like(z)
        for m in range(degree+1):
            shift = tuple(m if a == axis else 0 for a in range(3))
            origin, destination = _shift_layout(indices, lookup, shift, degree, centers.device)
            updated[:, destination] += result[:, origin]*phase[:, None]
            phase = phase*residual[:, axis]/(m+1)
        result = updated
    amplitude = torch.exp(z)/(4*math.pi*radii)
    numerator = result*amplitude[:, None]
    physical = numerator/physical_powers
    lost = ((numerator.real != 0) & (physical.real == 0)) | ((numerator.imag != 0) & (physical.imag == 0))
    if not bool(torch.isfinite(amplitude).all() & (amplitude.abs() > 0).all()
                & ~((result != 0) & (numerator == 0)).any() & ~lost.any()):
        raise ValueError('Directional physical coefficient scaling underflows native precision')
    result = physical
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Directional coefficient recurrence exceeds native precision')
    return result[0] if single else result


class CartesianDirectionalFFTTranslation(CartesianFFTTranslation):
    """Input moments already carry source phase; receiver carrier is not applied."""
    def __init__(self, displacement, direction, wave_number, normalization_length, order, *,
                 rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2,
                 retained_budget_bytes=1024**2,test_radius=None,source_radius=None,kernel_error_budget=None):
        self._prepare_directional(displacement, direction, wave_number, normalization_length, order,
            rhs_batch, tensor_workspace_budget_bytes, retained_budget_bytes, coefficient_batch=1,
            test_radius=test_radius,source_radius=source_radius,kernel_error_budget=kernel_error_budget)
        coefficients = directional_helmholtz_taylor(displacement, direction, wave_number, 2*order+1,
            tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
        self._bind_coefficients(coefficients, wave_number)

    def _prepare_directional(self, displacement, direction, wave_number, normalization_length,
                             order, rhs_batch, workspace, retained, *, coefficient_batch,
                             test_radius=None,source_radius=None,kernel_error_budget=None):
        _validate_geometry(displacement, direction, wave_number, 2*order+1 if type(order) is int else -1)
        if type(rhs_batch) is not int or rhs_batch < 1:
            raise ValueError('Directional RHS batch must be a positive integer')
        if displacement.shape != (3,):
            raise ValueError('Directional FFT requires one center and carrier')
        if normalization_length is None:
            raise ValueError('Directional FFT requires explicit independent normalization')
        declared=any(v is not None for v in (test_radius,source_radius,kernel_error_budget))
        if declared and (test_radius is None or source_radius is None or not isinstance(kernel_error_budget,KernelErrorBudget)):
            raise ValueError('Directional admission requires both support radii and KernelErrorBudget')
        if declared:KernelErrorBudget(**asdict(kernel_error_budget))
        extra=16 if declared else 0
        expected = m2l_storage_bytes(order, 8, 'fft_v1')+24+extra
        if type(retained) is not int or retained < 1:
            raise ValueError('Directional retained budget must be a positive integer')
        if expected > retained:
            raise MemoryError('Directional retained budget exceeded before coefficients')
        self.planned_setup_tensor_workspace_bytes = directional_fft_setup_workspace_bytes(
            order, rhs_batch, coefficient_batch=coefficient_batch, declared=declared)
        if type(workspace) is not int or workspace < 1:
            raise ValueError('Directional workspace budget must be a positive integer')
        if self.planned_setup_tensor_workspace_bytes > workspace:
            raise MemoryError('Directional setup workspace exceeded before coefficients')
        self.support_radii,self.kernel_error_budget,self._ideal_bound=None,None,None
        if declared:
            bound=directional_multipole_truncation_bound(displacement,direction,test_radius,source_radius,
                wave_number,order,tensor_workspace_budget_bytes=workspace)
            bound.update(admitted=(bound['green_absolute']<=kernel_error_budget.green_absolute
                and bound['gradient_component_absolute']<=kernel_error_budget.gradient_component_absolute))
            if not bound['admitted']:raise MultipoleTruncationError(bound)
            self.support_radii=torch.stack((test_radius,source_radius))
            self.kernel_error_budget,self._ideal_bound=kernel_error_budget,bound
        self.direction = direction.clone()
        self._prepare(displacement, order, rhs_batch, workspace, retained-24-extra,
            normalization_length=normalization_length)
        self.planned_retained_bytes += 24+extra
        self.planned_tensor_workspace_bytes = self.planned_setup_tensor_workspace_bytes

    @classmethod
    def create_many(cls, displacements, directions, wave_number, normalization_lengths, order, *,
                    rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2,
                    retained_budget_bytes=1024**2,test_radii=None,source_radii=None,kernel_error_budget=None):
        _validate_geometry(displacements, directions, wave_number, 2*order+1 if type(order) is int else -1)
        if any(type(value) is not int or value < 1 for value in
               (rhs_batch, tensor_workspace_budget_bytes, retained_budget_bytes)):
            raise ValueError('Directional batch and budgets must be positive integers')
        if displacements.ndim != 2:
            raise ValueError('Directional FFT batch requires center and carrier rows')
        if (not isinstance(normalization_lengths, torch.Tensor) or normalization_lengths.shape != (len(displacements),)
                or normalization_lengths.device != displacements.device or normalization_lengths.dtype != torch.float64
                or normalization_lengths.requires_grad or not bool((torch.isfinite(normalization_lengths) & (normalization_lengths > 0)).all())):
            raise ValueError('Directional FFT requires matching explicit positive CUDA float64 scales')
        declared=any(v is not None for v in (test_radii,source_radii,kernel_error_budget))
        if declared:
            if not isinstance(kernel_error_budget,KernelErrorBudget):
                raise ValueError('Directional batch admission requires KernelErrorBudget and both support radii')
            KernelErrorBudget(**asdict(kernel_error_budget))
            for radii in (test_radii,source_radii):
                if (not isinstance(radii,torch.Tensor) or radii.shape!=(len(displacements),)
                        or radii.dtype!=torch.float64 or radii.device!=displacements.device or radii.requires_grad
                        or not bool((torch.isfinite(radii) & (radii>=0)).all())):
                    raise ValueError('Directional batch support radii must match fixed native centers')
        expected = m2l_storage_bytes(order, 8, 'fft_v1')+24+(16 if declared else 0)
        if type(retained_budget_bytes) is not int or retained_budget_bytes < 1:
            raise ValueError('Directional retained budget must be a positive integer')
        if len(displacements)*expected > retained_budget_bytes:
            raise MemoryError('Directional batch retained budget exceeded before coefficients')
        if declared:
            # Check every configured row before allocating any translation.
            for start in range(0,len(displacements),32):
                stop=min(start+32,len(displacements))
                bounds=directional_multipole_truncation_bounds_many(displacements[start:stop],directions[start:stop],
                    test_radii[start:stop],source_radii[start:stop],wave_number,(order,),
                    tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                if not bool(bounds['available'].all()):
                    raise KernelBoundUnavailable('Directional batch has an unavailable Cauchy contour bound')
                admitted=(bounds['green_absolute']<=kernel_error_budget.green_absolute) & (
                    bounds['gradient_component_absolute']<=kernel_error_budget.gradient_component_absolute)
                if not bool(admitted.all()):
                    row=start+int((~admitted[:,0]).nonzero()[0,0])
                    bound=directional_multipole_truncation_bound(displacements[row],directions[row],test_radii[row],
                        source_radii[row],wave_number,order,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                    bound['admitted']=False
                    raise MultipoleTruncationError(bound)
                del bounds,admitted
        result = []
        for start in range(0, len(displacements), 32):
            stop = min(start+32, len(displacements))
            objects = []
            for row in range(start, stop):
                obj = cls.__new__(cls)
                obj._prepare_directional(displacements[row], directions[row], wave_number, normalization_lengths[row],
                    order, rhs_batch, tensor_workspace_budget_bytes, expected, coefficient_batch=stop-start,
                    test_radius=test_radii[row] if declared else None,source_radius=source_radii[row] if declared else None,
                    kernel_error_budget=kernel_error_budget)
                objects.append(obj)
            vectors = directional_helmholtz_taylor(displacements[start:stop], directions[start:stop], wave_number,
                2*order+1, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
            for obj, vector in zip(objects, vectors, strict=True):
                obj._bind_coefficients(vector.clone(), wave_number)
                result.append(obj)
            del vector, vectors
        return tuple(result)

    def tensors(self):
        return (*super().tensors(), self.direction,*((self.support_radii,) if self.support_radii is not None else ()))

    def _state(self):
        return (*super()._state(), self.planned_retained_bytes, self.planned_setup_tensor_workspace_bytes,
            None if self.kernel_error_budget is None else tuple(asdict(self.kernel_error_budget).items()),
            None if self._ideal_bound is None else tuple(self._ideal_bound.items()))

    def _fourier_fields(self):
        fields = super()._fourier_fields()
        fields[1:] += (1j*self.wave_number*self.direction)[:, None, None, None]*fields[0]
        if not bool(torch.isfinite(fields).all()):
            raise ValueError('Directional physical-gradient grid exceeds native precision')
        return fields

    def receipt(self):
        self.assert_current()
        record = super().receipt()
        record.update(version='experimental_directional_coefficient_fft_v1',
            input_scope='source_carrier_already_applied_to_moments',
            output_scope='physical_gradient_locals_before_receiver_carrier',
            carrier_gradient='grad_h_plus_ik_direction_h',
            coefficient_rule='centered_radial_carrier_then_separable_residual_v1',
            direction_normalized=False, planned_setup_tensor_workspace_bytes=self.planned_setup_tensor_workspace_bytes,
            error_bound_certified=False, material_gradient_qualified=False,
            directional_tree_integrated=False, infinite_periodic_certified=False)
        if self.support_radii is not None:
            record.update(version='experimental_directional_coefficient_fft_kernel_budget_v1',
                ideal_truncation_admission=dict(self._ideal_bound),kernel_error_budget=asdict(self.kernel_error_budget),
                support_radius_assumption=tuple(float(r) for r in self.support_radii),
                complete_support_radius_verified=False,production_qualification=False)
        return record


def directional_fft_setup_workspace_bytes(order, rhs_batch, *, coefficient_batch=1, declared=False):
    """Shared structural setup estimator; the constructor validates its inputs."""
    side = 1 << (2*order).bit_length()
    action = side**3*(16*(32+20*rhs_batch)+8*48)+3*side**3*16
    return (action+_coefficient_workspace_bytes(2*order+1, coefficient_batch)
        +(directional_bound_workspace_bytes(1, 1)+16 if declared else 0))
