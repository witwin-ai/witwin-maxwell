"""Exact-arithmetic Cartesian M2L truncation bounds on separated spheres.

Bivariate Cauchy bounds follow source/target total-degree truncation separately.
Native bound evaluation is not a floating-point or RWG integration certificate.
"""
import math
import torch
from .interpolation_error import KernelErrorBudget, KernelBoundUnavailable


class MultipoleTruncationError(ValueError):
    def __init__(self, receipt):
        super().__init__('Cartesian M2L truncation bound exceeds the declared kernel budget')
        self.receipt = receipt


def validate_kernel_bound_rule(rule):
    if rule not in ('bivariate_cauchy_v1', 'real_support_min_v1', 'positive_tail_min_v1', 'first_wave_min_v1', 'direct_positive_tail_min_v1', 'direct_first_wave_min_v1'):
        raise ValueError('Unsupported FMM kernel_bound_rule')


def multipole_truncation_bound(displacement, test_radius, source_radius, wave_number, order, *,
                                kernel_bound_rule='bivariate_cauchy_v1'):
    validate_kernel_bound_rule(kernel_bound_rule)
    if kernel_bound_rule != 'bivariate_cauchy_v1':
        if (not isinstance(displacement, torch.Tensor) or displacement.shape != (3,)
                or any(not isinstance(r, torch.Tensor) or r.shape != () for r in (test_radius, source_radius))):
            raise ValueError('M2L bound requires displacement and scalar radius tensors')
        bounds = multipole_truncation_bounds_many(displacement[None], test_radius[None], source_radius[None],
            wave_number, (order,), kernel_bound_rule=kernel_bound_rule)
        if not bool(bounds['available'][0, 0]):
            raise KernelBoundUnavailable('No finite separated real-support M2L bound')
        green, gradient = (float(bounds[name][0, 0]) for name in
                           ('green_absolute', 'gradient_component_absolute'))
        versions = {'real_support_min_v1': 'cartesian_real_support_min_bound_v1',
                    'positive_tail_min_v1': 'cartesian_positive_tail_min_bound_v1',
                    'first_wave_min_v1': 'cartesian_first_wave_min_bound_v1',
                    'direct_positive_tail_min_v1': 'cartesian_direct_positive_tail_min_bound_v1',
                    'direct_first_wave_min_v1': 'cartesian_direct_first_wave_min_bound_v1'}
        return dict(version=versions[kernel_bound_rule], order=order,
            kernel_bound_rule=kernel_bound_rule,
            radius_sum_over_distance=float((test_radius.double()+source_radius.double()) /
                                          torch.linalg.vector_norm(displacement.double())),
            active_expansion_sides=int((torch.stack((test_radius, source_radius)) > 0).sum()),
            green_absolute=green, gradient_component_absolute=gradient,
            scope='ideal_exact_arithmetic_kernel_truncation_only', floating_point_certified=False,
            quadrature_certified=False, implemented_operator_certified=False, physical_qualification=False)

    if (not isinstance(displacement, torch.Tensor) or displacement.shape != (3,)
            or displacement.device.type != 'cuda' or displacement.dtype not in (torch.float32, torch.float64)
            or displacement.requires_grad or not bool(torch.isfinite(displacement).all())):
        raise ValueError('M2L bound requires a finite fixed CUDA displacement')
    for radius in (test_radius, source_radius):
        if (not isinstance(radius, torch.Tensor) or radius.shape != () or radius.device != displacement.device
                or radius.dtype != displacement.dtype or radius.requires_grad
                or not bool(torch.isfinite(radius) & (radius >= 0))):
            raise ValueError('M2L radii must be finite nonnegative fixed matching CUDA scalars')
    if type(order) is not int or order < 0:
        raise ValueError('M2L bound order must be a nonnegative integer')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('M2L wave number must be a fixed scalar')
    wave = complex(wave_number)
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
        raise ValueError('M2L bound requires a finite passive wave number')
    distance = torch.linalg.vector_norm(displacement.double())
    if not bool(torch.isfinite(distance) & (distance > 0)):
        raise ValueError('M2L bound centers require finite positive separation')
    radii = torch.stack((test_radius, source_radius)).double()
    active = int((radii > 0).sum())
    ratio = radii.sum()/distance
    receipt = {'version': 'cartesian_bivariate_cauchy_bound_v1', 'order': order,
               'radius_sum_over_distance': float(ratio), 'active_expansion_sides': active,
               'scope': 'ideal_exact_arithmetic_kernel_truncation_only',
               'floating_point_certified': False, 'quadrature_certified': False,
               'implemented_operator_certified': False, 'physical_qualification': False}
    if not active:
        return {**receipt, 'green_absolute': 0., 'gradient_component_absolute': 0.,
                'green_log_absolute': None, 'gradient_component_log_absolute': None,
                'green_contour_fraction': None, 'gradient_contour_fraction': None}
    candidates = torch.tensor([.05, .1, .15, .2, .25, .3, .35, .4],
                              device=displacement.device, dtype=torch.float64)
    candidates = candidates[candidates > ratio]
    if not len(candidates):
        raise KernelBoundUnavailable('No analytic Cauchy contour for the requested support-sphere ratio')
    alpha = torch.sqrt(1-2*candidates-candidates.square())
    radial_lower = alpha*distance
    radial_change = distance*candidates*(2+candidates)/(1+alpha)
    # Work entirely in logs until final conversion: exponential decay may be
    # offset by the 1/r or gradient factors at small physical scales.
    log_green_max = abs(wave.real)*radial_change-wave.imag*radial_lower-math.log(4*math.pi)-radial_lower.log()
    log_rho = candidates.log()-ratio.log()
    tail_power_log = -(order+1)*log_rho
    log_denominator = torch.log(-torch.expm1(-log_rho))
    if active == 1:
        tail_log = tail_power_log-log_denominator
    else:
        tail_log = tail_power_log+torch.log(2-torch.exp(tail_power_log))-2*log_denominator
    log_wave = -math.inf if wave == 0 else math.log(abs(wave))
    log_gradient_max = (log_green_max+torch.logaddexp(radial_lower.new_tensor(log_wave), -radial_lower.log())
                        +torch.log1p(candidates)-alpha.log())
    green_logs, gradient_logs = log_green_max+tail_log, log_gradient_max+tail_log
    gi, hi = int(green_logs.argmin()), int(gradient_logs.argmin())
    def finite_bound(log_value):
        if not bool(torch.isfinite(log_value)):
            raise KernelBoundUnavailable('M2L bound exceeds finite log representation')
        # A positive mathematical upper bound must not be reported as zero.
        value = torch.exp(log_value.clamp_min(math.log(torch.finfo(torch.float64).tiny)))
        if not bool(torch.isfinite(value)):
            raise KernelBoundUnavailable('M2L bound exceeds finite native representation')
        return float(value)
    return {**receipt, 'green_absolute': finite_bound(green_logs[gi]),
            'gradient_component_absolute': finite_bound(gradient_logs[hi]),
            'green_log_absolute': float(green_logs[gi]),
            'gradient_component_log_absolute': float(gradient_logs[hi]),
            'green_contour_fraction': float(candidates[gi]),
            'gradient_contour_fraction': float(candidates[hi])}



def multipole_truncation_bounds(displacements, test_radii, source_radii, wave_number, order, *,
                                 kernel_bound_rule='bivariate_cauchy_v1'):
    """Single-order CUDA vector bounds with the original native batch contract."""
    values = multipole_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, (order,), kernel_bound_rule=kernel_bound_rule)
    return {name: value[:, 0] for name, value in values.items()}


def multipole_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders, *,
                                      kernel_bound_rule='bivariate_cauchy_v1'):
    """Evaluate at most sixteen orders over a batch of fixed CUDA spheres.

    Unavailable contours or finite representations are marked per block. This
    computes the same ideal bounds as the scalar receipt; it adds no enclosure
    for floating point, quadrature or the implemented operator.
    """
    validate_kernel_bound_rule(kernel_bound_rule)
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    radii = torch.stack((test_radii, source_radii), dim=-1).double()
    active = (radii > 0).sum(-1)
    ratio = radii.sum(-1)/distance
    candidates = torch.tensor([.05, .1, .15, .2, .25, .3, .35, .4],
                              device=displacements.device, dtype=torch.float64)
    contours = candidates[None, :] > ratio[:, None]
    alpha = torch.sqrt(1-2*candidates-candidates.square())
    radial_lower = distance[:, None]*alpha
    radial_change = distance[:, None]*candidates*(2+candidates)/(1+alpha)
    log_green_max = (abs(wave.real)*radial_change-wave.imag*radial_lower
                     -math.log(4*math.pi)-radial_lower.log())
    # Invalid contours are excluded before reduction. Safe log arguments avoid
    # NaNs on those excluded entries and on zero-radius exact blocks.
    safe_ratio = torch.where(active > 0, ratio, torch.ones_like(ratio))
    log_rho = candidates.log()-safe_ratio[:, None].log()
    log_rho = torch.where(contours & (active[:, None] > 0), log_rho, torch.ones_like(log_rho))
    order_values = torch.tensor(orders, device=displacements.device, dtype=torch.float64)
    tail_power_log = -(order_values[None, :, None]+1)*log_rho[:, None, :]
    log_denominator = torch.log(-torch.expm1(-log_rho))
    tail_log = torch.where(active[:, None, None] == 1, tail_power_log-log_denominator[:, None, :],
        tail_power_log+torch.log(2-torch.exp(tail_power_log))-2*log_denominator[:, None, :])
    log_wave = -math.inf if wave == 0 else math.log(abs(wave))
    log_gradient_max = (log_green_max+torch.logaddexp(radial_lower.new_tensor(log_wave), -radial_lower.log())
                        +torch.log1p(candidates)-alpha.log())
    green_log = torch.where(contours[:, None, :], log_green_max[:, None, :]+tail_log, torch.inf).amin(-1)
    gradient_log = torch.where(contours[:, None, :], log_gradient_max[:, None, :]+tail_log, torch.inf).amin(-1)
    green = torch.exp(green_log.clamp_min(math.log(torch.finfo(torch.float64).tiny)))
    gradient = torch.exp(gradient_log.clamp_min(math.log(torch.finfo(torch.float64).tiny)))
    available = ((torch.isfinite(green_log) & torch.isfinite(gradient_log)
                  & torch.isfinite(green) & torch.isfinite(gradient)) | (active[:, None] == 0))
    green = torch.where(available & (active[:, None] > 0), green, 0.)
    gradient = torch.where(available & (active[:, None] > 0), gradient, 0.)
    if kernel_bound_rule != 'bivariate_cauchy_v1':
        real = _real_support_bounds(distance, radii, wave, orders)
        # An unavailable zero placeholder must never enter the minimum.
        green = torch.minimum(torch.where(available, green, torch.inf),
                              torch.where(real['available'], real['green_absolute'], torch.inf))
        gradient = torch.minimum(torch.where(available, gradient, torch.inf),
                                 torch.where(real['available'], real['gradient_component_absolute'], torch.inf))
        available = available | real['available']
        if kernel_bound_rule in ('positive_tail_min_v1', 'first_wave_min_v1', 'direct_positive_tail_min_v1', 'direct_first_wave_min_v1'):
            positive = _positive_tail_bounds(distance, radii, wave, orders)
            green = torch.minimum(green, torch.where(positive['available'], positive['green_absolute'], torch.inf))
            gradient = torch.minimum(gradient, torch.where(positive['available'], positive['gradient_component_absolute'], torch.inf))
            # Existing unavailable entries are still infinity at this point.
            available = available | positive['available']
            if kernel_bound_rule in ('first_wave_min_v1', 'direct_positive_tail_min_v1', 'direct_first_wave_min_v1'):
                first = _positive_tail_bounds(distance, radii, wave, orders, first_wave_exact=True)
                green = torch.minimum(green, torch.where(first['available'], first['green_absolute'], torch.inf))
                gradient = torch.minimum(gradient, torch.where(first['available'], first['gradient_component_absolute'], torch.inf))
                available = available | first['available']
            if kernel_bound_rule in ('direct_positive_tail_min_v1', 'direct_first_wave_min_v1'):
                direct = _direct_positive_tail_bounds(distance, radii, wave, orders)
                green = torch.minimum(green, torch.where(direct['available'], direct['green_absolute'], torch.inf))
                gradient = torch.minimum(gradient, torch.where(direct['available'], direct['gradient_component_absolute'], torch.inf))
                available = available | direct['available']
                if kernel_bound_rule == 'direct_first_wave_min_v1':
                    corrected = _direct_positive_tail_bounds(distance, radii, wave, orders, first_wave_exact=True)
                    green = torch.minimum(green, torch.where(corrected['available'], corrected['green_absolute'], torch.inf))
                    gradient = torch.minimum(gradient, torch.where(corrected['available'], corrected['gradient_component_absolute'], torch.inf))
                    available = available | corrected['available']
        green = torch.where(available, green, 0.)
        gradient = torch.where(available, gradient, 0.)
    return {'green_absolute': green, 'gradient_component_absolute': gradient,
            'available': available}



def _bound_inputs(displacements, test_radii, source_radii, wave_number, orders):
    if (not isinstance(displacements, torch.Tensor) or displacements.ndim != 2
            or displacements.shape[1] != 3 or displacements.device.type != 'cuda'
            or displacements.dtype not in (torch.float32, torch.float64)
            or displacements.requires_grad or not bool(torch.isfinite(displacements).all())):
        raise ValueError('M2L bounds require finite fixed CUDA displacements')
    for radii in (test_radii, source_radii):
        if (not isinstance(radii, torch.Tensor) or radii.shape != displacements.shape[:1]
                or radii.device != displacements.device or radii.dtype != displacements.dtype
                or radii.requires_grad or not bool((torch.isfinite(radii) & (radii >= 0)).all())):
            raise ValueError('M2L radii must be finite nonnegative fixed matching CUDA vectors')
    if (not isinstance(orders, (tuple, list)) or not 1 <= len(orders) <= 16
            or any(type(order) is not int or not 0 <= order <= torch.iinfo(torch.int64).max for order in orders)):
        raise ValueError('M2L batch orders require one to sixteen nonnegative native integers')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('M2L wave number must be a fixed scalar')
    wave = complex(wave_number)
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
        raise ValueError('M2L bound requires a finite passive wave number')
    distance = torch.linalg.vector_norm(displacements.double(), dim=-1)
    if not bool((torch.isfinite(distance) & (distance > 0)).all()):
        raise ValueError('M2L bound centers require finite positive separation')
    return wave, distance


def real_support_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders):
    """Ideal double-total-degree Helmholtz tails on real separated supports.

    Uses a real symmetric derivative norm and a union of the two degree tails.
    It does not cover a screened Ewald kernel, coefficient roundoff or RWG
    quadrature. See docs/reference/mom_fmm_real_support_bound.md for the proof.
    Unavailable rows have zero placeholders and an independent rejection mask.
    """
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    return _real_support_bounds(distance, torch.stack((test_radii, source_radii), dim=-1).double(), wave, orders)


def _real_support_bounds(distance, radii, wave, orders):
    a, b = (radii/distance[:, None]).unbind(-1)
    q = a+b
    separated = torch.isfinite(q) & (q < 1)
    # Invalid rows are excluded after arithmetic on safe arguments.
    a, b, q = (torch.where(separated, x, 0.)[:, None] for x in (a, b, q))
    p1 = torch.tensor(orders, device=distance.device, dtype=torch.float64)[None, :]+1
    log_remaining = torch.log1p(-q)
    log_fs = p1*(b.log()-torch.log1p(-a))-log_remaining
    log_ft = p1*(a.log()-torch.log1p(-b))-log_remaining
    us, ut = p1/(1-a)+1/(1-q), p1/(1-b)+1/(1-q)
    vs = us*(us+1)+p1*a/(1-a).square()+q/(1-q).square()
    vt = ut*(ut+1)+p1*b/(1-b).square()+q/(1-q).square()
    log_f = torch.logaddexp(log_fs, log_ft)
    log_d1 = torch.logaddexp(log_fs+us.log(), log_ft+ut.log())
    log_d2 = torch.logaddexp(log_fs+vs.log(), log_ft+vt.log())
    # |k|-Im(k) = Re(k)^2/(|k|+Im(k)), evaluated without cancellation,
    # squaring large native scalars or forming an overflowing |k|+Im(k).
    scale = max(abs(wave.real), wave.imag)
    if scale:
        real_scaled, imag_scaled = wave.real/scale, wave.imag/scale
        norm_scaled = math.hypot(real_scaled, imag_scaled)
        log_magnitude_distance = distance.log()+math.log(scale)+math.log(norm_scaled)
        # The logarithm also retains small positive kD below native range.
        log_exponential_tail = torch.where(log_magnitude_distance < -36,
            log_magnitude_distance, torch.log(-torch.expm1(-torch.exp(log_magnitude_distance))))
        if wave.real:
            log_delta = 2*math.log(abs(wave.real))-math.log(scale)-math.log(norm_scaled+imag_scaled)
            delta_distance = torch.exp(distance.log()+log_delta)
        else:
            delta_distance = torch.zeros_like(distance)
        log_b1 = delta_distance+log_exponential_tail
    else:
        log_b1 = torch.full_like(distance, -torch.inf)
    log_b0 = -wave.imag*distance
    log_d = distance.log()[:, None]
    log_green = torch.logaddexp(log_b0[:, None]+log_f,
        log_b1[:, None]+log_d1-math.log(2))-math.log(4*math.pi)-log_d
    log_gradient = torch.logaddexp(log_b0[:, None]+log_d1,
        log_b1[:, None]+log_d2-math.log(2))-math.log(4*math.pi)-2*log_d
    floor = math.log(torch.finfo(torch.float64).tiny)
    green, gradient = torch.exp(log_green.clamp_min(floor)), torch.exp(log_gradient.clamp_min(floor))
    zero = (radii == 0).all(-1)[:, None]
    available = separated[:, None] & ((torch.isfinite(log_green) & torch.isfinite(log_gradient)
        & torch.isfinite(green) & torch.isfinite(gradient)) | zero)
    return dict(green_absolute=torch.where(available & ~zero, green, 0.),
                gradient_component_absolute=torch.where(available & ~zero, gradient, 0.), available=available)


def positive_tail_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders):
    """Positive-coefficient, one-sided-contour ideal Helmholtz kernel tails.

    The actual other-side radius is held fixed. A finite native contour family
    bounds each missing-degree tail independently; no Ewald or float certificate.
    See docs/reference/mom_fmm_positive_coefficient_tail.md for the derivation.
    """
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    return _positive_tail_bounds(distance, torch.stack((test_radii, source_radii), dim=-1).double(), wave, orders)


def first_wave_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders):
    """Ideal positive tails retaining the exact first-wave Legendre bound.

    Free-space real supports only; no coefficient, quadrature or float enclosure.
    See docs/reference/mom_fmm_first_wave_tail.md for the reviewed derivation.
    """
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    return _positive_tail_bounds(distance, torch.stack((test_radii, source_radii), dim=-1).double(),
                                 wave, orders, first_wave_exact=True)


def _log_second_exponential_remainder(log_x, x):
    """log((exp(x)-1-x)/exp(x)) for positive x, with analytic series tail.

    Below .1, retain R2 = sum_l 2*x^l/(l+2)! through l=8 and upper-bound
    its remaining positive tail geometrically. This is an analytic upper
    approximation, not a floating-point enclosure. log_x retains subnormals.
    """
    small_x = x.clamp(max=.1)
    polynomial = torch.full_like(x, 2/math.factorial(10))
    for degree in range(7, -1, -1):
        polynomial = polynomial*small_x+2/math.factorial(degree+2)
    tail = (2/math.factorial(11))*small_x.pow(9)/(1-small_x/12)
    small = 2*log_x-math.log(2)-small_x+torch.log(polynomial+tail)
    # Moderate x has no small second-remainder subtraction; at x>50,
    # factor <=1 is conservatively replaced by 1, before overflow.
    moderate_x = x.clamp(min=.1, max=50.)
    moderate = torch.log(-torch.expm1(torch.log1p(moderate_x)-moderate_x))
    return torch.where(log_x <= math.log(.1), small,
                       torch.where(log_x > math.log(50.), 0., moderate))


def _positive_tail_bounds(distance, radii, wave, orders, *, first_wave_exact=False):
    a, b = (radii/distance[:, None]).unbind(-1)
    separated = torch.isfinite(a+b) & (a+b < 1)
    a, b = (torch.where(separated, x, 0.) for x in (a, b))
    fractions = distance.new_tensor([.01,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.8,.9,.95])
    p1 = distance.new_tensor(orders)[None, :, None]+1
    # Each contour increases only the omitted-side radius, within the open disk.
    # The adaptive family guarantees a mathematical contour when q<1; rounding
    # back to an endpoint is excluded below rather than treated as valid.
    def tail(held, omitted):
        contours = omitted[:, None]+(1-held-omitted)[:, None]*fractions
        total = held[:, None]+contours
        valid = separated[:, None] & (contours > omitted[:, None]) & (total < 1)
        safe_t = torch.where(valid, total, .5)
        log_remaining = torch.log1p(-safe_t)
        log_c = (2-safe_t).log()-math.log(2)-2*log_remaining
        log_dc = (3-safe_t).log()-math.log(2)-3*log_remaining
        scale = max(abs(wave.real), wave.imag)
        if scale:
            norm = math.hypot(wave.real/scale, wave.imag/scale)
            log_k = math.log(scale)+math.log(norm)
            log_kd = distance.log()+log_k
            log_x = log_kd[:, None]+safe_t.log()
            x = torch.exp(log_x)
            # Combine attenuation with exponential growth before either can
            # overflow: |k|D t-kappa D, using signed scaled wave constants.
            coefficient = norm*safe_t-wave.imag/scale
            # Reject contours at unresolved growth/attenuation cancellation.
            # Signed logarithmic scaling avoids inf*0 and intermediate kD overflow.
            ambiguous = coefficient.abs() <= 8*torch.finfo(torch.float64).eps*(norm*safe_t+wave.imag/scale)
            valid = valid & ~ambiguous
            log_exponent_magnitude = distance.log()[:, None]+math.log(scale)+coefficient.abs().log()
            exponent = coefficient.sign()*torch.exp(log_exponent_magnitude)
            # log(1-exp(-x)), including positive x below native representation.
            log_remainder = torch.where(log_x < -36, log_x, torch.log(-torch.expm1(-x)))
            exponential_remainder = exponent+log_remainder
            exponential_derivative = exponent+log_kd[:, None]
        else:
            exponential_remainder = torch.full_like(safe_t, -torch.inf)
            exponential_derivative = torch.full_like(safe_t, -torch.inf)
            log_kd = torch.full_like(distance, -torch.inf)
            log_x = torch.full_like(safe_t, -torch.inf)
            x = torch.zeros_like(safe_t)
            exponent = torch.zeros_like(safe_t)
        log_attenuation = -wave.imag*distance[:, None]
        log_phi = torch.logaddexp(log_attenuation-log_remaining,
                                 exponential_remainder+log_c)
        log_derivative = torch.logaddexp(log_attenuation-2*log_remaining,
            torch.logaddexp(exponential_derivative+log_c, exponential_remainder+log_dc))
        if first_wave_exact:
            second_remainder = exponent+_log_second_exponential_remainder(log_x, x)
            log_phi = torch.logaddexp(log_attenuation+torch.logaddexp(torch.zeros_like(log_x), log_x)
                                     -log_remaining, second_remainder+log_c)
            log_derivative = torch.logaddexp(
                log_attenuation+torch.logaddexp(torch.zeros_like(log_kd), log_kd)[:, None]-2*log_remaining,
                torch.logaddexp(exponential_remainder+log_kd[:, None]+log_c, second_remainder+log_dc))
        log_ratio = omitted[:, None].log()-contours.log()
        log_power = p1*log_ratio[:, None, :]
        g = torch.where(valid[:, None, :], log_phi[:, None, :]+log_power, torch.inf).amin(-1)
        h = torch.where(valid[:, None, :], log_derivative[:, None, :]+log_power, torch.inf).amin(-1)
        # With zero omitted radius there is exactly no missing-degree tail.
        zero = omitted[:, None] == 0
        return torch.where(zero, -torch.inf, g), torch.where(zero, -torch.inf, h)
    sg, sh = tail(a, b)
    tg, th = tail(b, a)
    log_d = distance.log()[:, None]
    green_log = torch.logaddexp(sg, tg)-math.log(4*math.pi)-log_d
    gradient_log = torch.logaddexp(sh, th)-math.log(4*math.pi)-2*log_d
    floor = math.log(torch.finfo(torch.float64).tiny)
    green, gradient = torch.exp(green_log.clamp_min(floor)), torch.exp(gradient_log.clamp_min(floor))
    zero = (radii == 0).all(-1)[:, None]
    available = separated[:, None] & ((torch.isfinite(green_log) & torch.isfinite(gradient_log)
                                     & torch.isfinite(green) & torch.isfinite(gradient)) | zero)
    return dict(green_absolute=torch.where(available & ~zero, green, 0.),
                gradient_component_absolute=torch.where(available & ~zero, gradient, 0.), available=available)

def require_multipole_budget(receipt, budget):
    if not isinstance(budget, KernelErrorBudget):
        raise TypeError('M2L truncation admission requires KernelErrorBudget')
    if (receipt['green_absolute'] > budget.green_absolute
            or receipt['gradient_component_absolute'] > budget.gradient_component_absolute):
        raise MultipoleTruncationError(receipt)


def direct_positive_tail_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders):
    """Ideal missing-degree tails of a positive rational-exponential majorant.

    Native workspace admits p<=64 and |k|*omitted_radius<=32, otherwise this
    bound is unavailable. No contour, coefficient roundoff or Ewald enclosure.
    See docs/reference/mom_fmm_direct_positive_tail.md for the reviewed proof.
    """
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    return _direct_positive_tail_bounds(distance,
        torch.stack((test_radii, source_radii), dim=-1).double(), wave, orders)


def _log_positive_exponential_tail(log_x, orders):
    """Positive exp-series tail starting at p+1, plus analytic upper remainder.

    orders are bounded configuration values; log_x is a CUDA pair vector.
    Retain 97 terms, then bound all positive remaining terms geometrically.
    At x<=32 every subsequent ratio is at most x/(p+99)<1.
    """
    p1 = log_x.new_tensor(orders)+1
    valid = log_x <= math.log(32.)
    safe = torch.where(valid, log_x, -torch.inf)
    powers = p1[None, :, None]+torch.arange(97, device=log_x.device, dtype=torch.float64)[None, None, :]
    terms = powers*safe[:, None, None]-torch.lgamma(powers+1)
    next_term = terms[:, :, -1]+safe[:, None]-(p1+97).log()[None, :]
    ratio_log = safe[:, None]-(p1+98).log()[None, :]
    remainder = next_term-torch.log(-torch.expm1(ratio_log))
    return torch.logaddexp(terms.logsumexp(-1), remainder), valid


def direct_first_wave_truncation_bounds_many(displacements, test_radii, source_radii, wave_number, orders):
    """Ideal direct tails with the exact first-wave positive correction.

    Uses an analytic exact correction, not an upper-approximated subtrahend.
    Ambiguous native positive differences reject; no float or Ewald certificate.
    See docs/reference/mom_fmm_direct_first_wave_tail.md for the proof.
    """
    wave, distance = _bound_inputs(displacements, test_radii, source_radii, wave_number, orders)
    return _direct_positive_tail_bounds(distance,
        torch.stack((test_radii, source_radii), dim=-1).double(), wave, orders, first_wave_exact=True)


def _log_positive_tail_difference(minuend, subtrahend):
    """Stable positive logarithmic difference with conservative native rejection."""
    ratio = subtrahend-minuend
    exact_zero = torch.isneginf(subtrahend)
    tolerance = 64*torch.finfo(torch.float64).eps*(1+minuend.abs()+subtrahend.abs())
    valid = torch.isfinite(minuend) & (exact_zero | (torch.isfinite(subtrahend) & (ratio < -tolerance)))
    safe = torch.where(valid & ~exact_zero, ratio, -1.)
    result = torch.where(exact_zero, minuend, minuend+torch.log(-torch.expm1(safe)))
    return torch.where(valid, result, torch.inf), valid


def _rational_tail_log(m, threshold, log_l, log_q, log_remaining):
    """Exact geometric tail at a nonnegative b-degree threshold, in logs."""
    power = torch.where(threshold[None, :] == 0, 0., threshold[None, :]*log_q[:, None])
    lr, lq = log_remaining[:, None], log_q[:, None]
    if m == 1:
        aa = -lr
    elif m == 2:
        aa = torch.logaddexp((threshold+1).log()[None, :]-lr, lq-2*lr)
    else:
        aa = torch.logaddexp(((threshold+1).log()+(threshold+2).log()-math.log(2))[None, :]-lr,
            torch.logaddexp((threshold+2).log()[None, :]+lq-2*lr, 2*lq-3*lr))
    return power-m*log_l[:, None]+aa


def _direct_positive_tail_bounds(distance, radii, wave, orders, *, first_wave_exact=False):
    a, b = (radii/distance[:, None]).unbind(-1)
    separated = torch.isfinite(a+b) & (a+b < 1)
    a, b = (torch.where(separated, x, 0.) for x in (a, b))
    # Prefix workspace is bounded even when an old policy can evaluate a much
    # larger order. Such columns stay unavailable for this additional bound.
    bounded_orders = tuple(min(p, 64) for p in orders)
    order_valid = torch.tensor([p <= 64 for p in orders], device=distance.device, dtype=torch.bool)
    p1 = distance.new_tensor(bounded_orders)+1
    levels = torch.arange(max(bounded_orders)+1, device=distance.device, dtype=torch.float64)
    prefix = levels[None, :] < p1[:, None]
    h = (p1[:, None]-levels[None, :]).clamp_min(1)
    scale = max(abs(wave.real), wave.imag)
    if scale:
        norm = math.hypot(wave.real/scale, wave.imag/scale)
        log_kd = distance.log()+math.log(scale)+math.log(norm)
    else:
        log_kd = torch.full_like(distance, -torch.inf)

    def tail(held, omitted):
        # An omitted zero support has exactly no missing-degree terms; safe
        # arithmetic below is masked to this exact result before conversion.
        log_l = torch.log1p(-held)
        log_q = omitted.log()-log_l
        q = torch.exp(log_q)
        log_remaining = torch.log1p(-q)
        series, series_valid = _log_positive_exponential_tail(log_kd+omitted.log(), bounded_orders)
        valid = separated & series_valid
        if scale:
            coefficient = norm*held-wave.imag/scale
            ambiguous = coefficient.abs() <= 8*torch.finfo(torch.float64).eps*(norm*held+wave.imag/scale)
            exact_zero = (held == 0) & (wave.imag == 0)
            valid = valid & (~ambiguous | exact_zero)
            exponent = coefficient.sign()*torch.exp(distance.log()+math.log(scale)+coefficient.abs().log())
        else:
            exponent = torch.zeros_like(distance)
        # K=0 still has its exact l=0 term. Avoid selecting 0*log(0).
        log_terms = torch.where(levels[None, :] == 0, 0.,
            levels[None, :]*(log_kd+log_l)[:, None])-torch.lgamma(levels+1)[None, :]
        lr, lq = log_remaining[:, None, None], log_q[:, None, None]
        aa1 = -lr.expand(-1, len(orders), len(levels))
        aa2 = torch.logaddexp((h+1).log()[None, :, :]-lr, lq-2*lr)
        aa3 = torch.logaddexp(
            ((h+1).log()+(h+2).log()-math.log(2))[None, :, :]-lr,
            torch.logaddexp((h+2).log()[None, :, :]+lq-2*lr, 2*lq-3*lr))
        tails = []
        for m, aa in enumerate((aa1, aa2, aa3), 1):
            prefix_sum = torch.where(prefix[None, :, :], log_terms[:, None, :]+aa, -torch.inf).logsumexp(-1)
            finite = p1[None, :]*log_q[:, None]-m*log_l[:, None]+prefix_sum
            complete = series-m*(log_l+log_remaining)[:, None]
            tails.append(torch.logaddexp(finite, complete))
        t1, t2, t3 = tails
        green = exponent[:, None]+torch.logaddexp(t1, t2)-math.log(2)
        gradient = exponent[:, None]+torch.logaddexp(
            log_kd[:, None]+t1-math.log(2),
            torch.logaddexp(torch.logaddexp(log_kd, torch.zeros_like(log_kd))[:, None]+t2-math.log(2), t3))
        column_valid = valid[:, None]
        if first_wave_exact:
            u2 = [_rational_tail_log(2, (p1-shift).clamp_min(0), log_l, log_q, log_remaining)
                  for shift in range(3)]
            if wave == 0:
                # Exact Laplace tail; no subtraction of large positive terms.
                green = _rational_tail_log(1, p1, log_l, log_q, log_remaining)
                gradient = u2[0]
            else:
                u3 = [_rational_tail_log(3, (p1-shift).clamp_min(0), log_l, log_q, log_remaining)
                      for shift in range(3)]
                la, lb, lk = held.log()[:, None], omitted.log()[:, None], log_kd[:, None]
                correction_g = torch.logaddexp(
                    u2[0]+torch.logaddexp(la, lk+2*la),
                    torch.logaddexp(u2[1]+torch.logaddexp(lb, lk+math.log(2)+la+lb), u2[2]+lk+2*lb))-math.log(2)
                correction_h = torch.stack((u2[0]-math.log(2), u2[0]+lk+la, u2[1]+lk+lb,
                    u3[0]+la, u3[1]+lb, u3[0]+lk+2*la,
                    u3[1]+lk+math.log(2)+la+lb, u3[2]+lk+2*lb)).logsumexp(0)
                attenuation = -wave.imag*distance[:, None]
                green, green_valid = _log_positive_tail_difference(green, correction_g+attenuation)
                gradient, gradient_valid = _log_positive_tail_difference(gradient, correction_h+attenuation)
                column_valid = column_valid & green_valid & gradient_valid
        zero = omitted[:, None] == 0
        # No omitted degree exists on a zero side, independent of representation.
        if first_wave_exact:
            column_valid = torch.where(zero, separated[:, None], column_valid)
        return torch.where(zero, -torch.inf, green), torch.where(zero, -torch.inf, gradient), column_valid

    sg, sh, source_valid = tail(a, b)
    tg, th, target_valid = tail(b, a)
    log_d = distance.log()[:, None]
    green_log = torch.logaddexp(sg, tg)-math.log(4*math.pi)-log_d
    gradient_log = torch.logaddexp(sh, th)-math.log(4*math.pi)-2*log_d
    floor = math.log(torch.finfo(torch.float64).tiny)
    green, gradient = torch.exp(green_log.clamp_min(floor)), torch.exp(gradient_log.clamp_min(floor))
    zero = (radii == 0).all(-1)[:, None]
    available = (source_valid & target_valid & order_valid[None, :]
        & (torch.isfinite(green_log) & torch.isfinite(gradient_log)
           & torch.isfinite(green) & torch.isfinite(gradient))) | (separated[:, None] & zero)
    return dict(green_absolute=torch.where(available & ~zero, green, 0.),
                gradient_component_absolute=torch.where(available & ~zero, gradient, 0.), available=available)


def directional_bound_workspace_bytes(batch,order_count):
    """Conservative visible contour, order, logarithm and reduction tensors."""
    return batch*order_count*17*8*32+batch*17*8*32+4096


def directional_multipole_truncation_bounds_many(displacements,directions,test_radii,source_radii,
        wave_number,orders,*,tensor_workspace_budget_bytes=128*1024**2):
    """Ideal continuous directional Green and each physical-gradient tail.

    Cauchy contours enclose the carrier-removed kernel in a complex Euclidean
    ball. Restore the real-support carrier after separately truncating both
    source and target total degrees. No roundoff/quadrature/tree certificate.
    Unavailable entries have zero placeholders and MUST check available first.
    """
    wave,distance=_bound_inputs(displacements,test_radii,source_radii,wave_number,orders)
    if (displacements.dtype != torch.float64 or len(displacements)>32
            or any(p>12 for p in orders)):
        raise ValueError('Directional bounds require at most32 native float64 centers and orders zero to twelve')
    if (not isinstance(directions,torch.Tensor) or directions.shape != displacements.shape
            or directions.dtype != torch.float64 or directions.device != displacements.device
            or directions.requires_grad or not bool(torch.isfinite(directions).all())):
        raise ValueError('Directional bounds require matching fixed finite CUDA float64 carriers')
    carrier_norm=directions.norm(dim=-1)
    if not bool(((carrier_norm-1).abs()<=512*torch.finfo(torch.float64).eps).all()):
        raise ValueError('Directional bound carriers must be unit length without normalization')
    if type(tensor_workspace_budget_bytes) is not int or tensor_workspace_budget_bytes<1:
        raise ValueError('Directional bound workspace budget must be a positive integer')
    if directional_bound_workspace_bytes(len(displacements),len(orders))>tensor_workspace_budget_bytes:
        raise MemoryError('Directional bound workspace exceeded before contours')
    radii=torch.stack((test_radii,source_radii),-1)
    active=(radii>0).sum(-1)
    total_radius=radii.sum(-1)
    ratio=total_radius/distance
    unit=displacements/distance[:,None]
    mismatch=(unit-directions).norm(dim=-1)
    fixed=distance.new_tensor([.05,.1,.15,.2,.25,.3,.35,.4])
    adaptive=ratio[:,None]*distance.new_tensor([1.25,1.5,2.,3.,4.,8.,12.,16.,32.])[None]
    rho=torch.cat((fixed[None].expand(len(distance),-1),adaptive),-1)
    legal=(rho>0) & (rho<=.4)
    rho=torch.where(legal,rho,.05)
    alpha=torch.sqrt(1-2*rho-rho.square())
    wave_scale=max(abs(wave.real),wave.imag)
    log_wave=(math.log(wave_scale)+.5*math.log((wave.real/wave_scale)**2+(wave.imag/wave_scale)**2)
              if wave_scale else -math.inf)
    wave_distance=torch.exp(distance.log()+log_wave) if wave_scale else torch.zeros_like(distance)
    # Native loss of a nonzero physical scale must not yield a false zero tail.
    native=(torch.isfinite(ratio) & ((active==0) | (ratio>0))
            & torch.isfinite(wave_distance) & ((wave==0) | (wave_distance>0)))
    contours=(rho>ratio[:,None]) & legal & native[:,None]
    residual=rho*mismatch[:,None]+rho.square()/(alpha+1-rho)
    growth=wave_distance[:,None]*residual
    contours=contours & torch.isfinite(growth) & ((wave==0) | (growth>0))
    # Combine passive center decay and real-support carrier restoration before
    # scaling, to avoid subtracting two separately overflowing exponents.
    decay=-wave.imag*(distance-carrier_norm*total_radius)
    log_green_max=(decay[:,None]+growth-math.log(4*math.pi)
                   -distance.log()[:,None]-alpha.log())
    log_gradient_max=(log_green_max+torch.logaddexp(distance.new_tensor(log_wave),
        -distance.log()[:,None]-alpha.log())+torch.log1p(rho)-alpha.log())
    safe_ratio=torch.where((active>0) & native,ratio,torch.ones_like(ratio))
    log_contour=rho.log()-safe_ratio.log()[:,None]
    log_contour=torch.where(contours & (active[:,None]>0),log_contour,torch.ones_like(log_contour))
    p=distance.new_tensor(orders)
    log_power=-(p[None,:,None]+1)*log_contour[:,None]
    log_denominator=torch.log(-torch.expm1(-log_contour))[:,None]
    tail=torch.where(active[:,None,None]==1,log_power-log_denominator,
        log_power+torch.log(2-torch.exp(log_power))-2*log_denominator)
    green_logs=torch.where(contours[:,None],log_green_max[:,None]+tail,torch.inf)
    gradient_logs=torch.where(contours[:,None],log_gradient_max[:,None]+tail,torch.inf)
    green_log,green_index=green_logs.min(-1)
    gradient_log,gradient_index=gradient_logs.min(-1)
    floor=math.log(torch.finfo(torch.float64).tiny)
    green=torch.exp(green_log.clamp_min(floor));gradient=torch.exp(gradient_log.clamp_min(floor))
    zero=(active==0)[:,None]
    available=zero | (torch.isfinite(green_log) & torch.isfinite(gradient_log)
                     & torch.isfinite(green) & torch.isfinite(gradient) & native[:,None])
    return dict(available=available,green_absolute=torch.where(available & ~zero,green,0.),
        gradient_component_absolute=torch.where(available & ~zero,gradient,0.),
        green_log_absolute=torch.where(zero,-torch.inf,green_log),
        gradient_component_log_absolute=torch.where(zero,-torch.inf,gradient_log),
        green_contour_fraction=rho.gather(1,green_index),gradient_contour_fraction=rho.gather(1,gradient_index))


def directional_multipole_truncation_bound(displacement,direction,test_radius,source_radius,wave_number,
        order,*,tensor_workspace_budget_bytes=128*1024**2):
    """Single ideal directional kernel receipt; reject every unavailable entry."""
    if (not isinstance(displacement,torch.Tensor) or displacement.shape!=(3,)
            or not isinstance(direction,torch.Tensor) or direction.shape!=(3,)
            or any(not isinstance(r,torch.Tensor) or r.shape!=() for r in (test_radius,source_radius))):
        raise ValueError('Directional bound requires displacement, carrier and scalar support radii')
    values=directional_multipole_truncation_bounds_many(displacement[None],direction[None],
        test_radius[None],source_radius[None],wave_number,(order,),
        tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
    if not bool(values['available'][0,0]):
        raise KernelBoundUnavailable('No finite native directional Cauchy contour bound')
    zero=not bool((test_radius>0) | (source_radius>0))
    return dict(version='directional_bivariate_cauchy_bound_v1',order=order,
        green_absolute=float(values['green_absolute'][0,0]),
        gradient_component_absolute=float(values['gradient_component_absolute'][0,0]),
        green_log_absolute=None if zero else float(values['green_log_absolute'][0,0]),
        gradient_component_log_absolute=None if zero else float(values['gradient_component_log_absolute'][0,0]),
        green_contour_fraction=None if zero else float(values['green_contour_fraction'][0,0]),
        gradient_contour_fraction=None if zero else float(values['gradient_contour_fraction'][0,0]),
        gradient_scope='each Cartesian physical-gradient component; not full vector norm',
        scope='ideal_exact_arithmetic_continuous_directional_kernel_truncation_only',
        carrier_removed_before_Cauchy=True,real_support_carrier_restored=True,
        floating_point_certified=False,quadrature_certified=False,phase_path_certified=False,
        global_operator_certified=False,physical_qualification=False,production_qualification=False)
