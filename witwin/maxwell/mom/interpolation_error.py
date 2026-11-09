"""Analytic truncation bounds for free-space tensor Chebyshev interpolation.

The bounds concern exact arithmetic and exact kernel samples. They do not bound
floating-point evaluation, quadrature, RWG projection, or the system operator.
See docs/reference/mom_scalar_interpolation_bounds.md for the derivation.
"""
from dataclasses import dataclass, asdict
import math
import torch
from .interpolation import InterpolationBox


@dataclass(frozen=True)
class KernelErrorBudget:
    green_absolute: float
    gradient_component_absolute: float

    def __post_init__(self):
        for value in (self.green_absolute, self.gradient_component_absolute):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Absolute kernel truncation budgets must be positive finite scalars')


class KernelInterpolationError(ValueError):
    def __init__(self, receipt):
        super().__init__('Analytic kernel interpolation truncation bound exceeds its budget')
        self.receipt = receipt


class KernelBoundUnavailable(ValueError):
    """The chosen order's analytic bound exceeds finite native representation."""


def scalar_kernel_interpolation_bound(test, source, wave_number):
    """Bound |G-I G| and each component of |grad G-I grad G| uniformly.

    Six univariate ellipse bounds are combined by a tensor telescoping identity.
    Geometry arithmetic is CUDA float64, irrespective of input precision; the
    receipt deliberately does not claim directed-rounding certification.
    """
    if not isinstance(test, InterpolationBox) or not isinstance(source, InterpolationBox):
        raise TypeError('Interpolation bounds require interpolation boxes')
    test.assert_current()
    source.assert_current()
    if test.device != source.device or test.dtype != source.dtype:
        raise ValueError('Interpolation bound boxes must share device and precision')
    return scalar_box_interpolation_bound(test.lower, test.upper, source.lower, source.upper,
                                          wave_number, test.order, source.order)


def scalar_box_interpolation_bound(lower_a, upper_a, lower_b, upper_b, wave_number,
                                   test_order, source_order):
    """Evaluate a prospective order using only bounds, without allocating nodes."""
    bounds = (lower_a, upper_a, lower_b, upper_b)
    if (any(not isinstance(t, torch.Tensor) or t.shape != (3,) or t.device.type != 'cuda'
            or t.dtype not in (torch.float32, torch.float64) or t.requires_grad for t in bounds)
            or any(t.device != lower_a.device or t.dtype != lower_a.dtype for t in bounds)
            or not bool(torch.stack([torch.isfinite(t).all() for t in bounds]).all())
            or not bool((upper_a >= lower_a).all() & (upper_b >= lower_b).all())):
        raise ValueError('Prospective interpolation requires finite ordered fixed CUDA bounds')
    if any(type(v) is not int or v < 1 for v in (test_order, source_order)):
        raise ValueError('Prospective interpolation orders must be positive integers')
    device = lower_a.device
    # Match InterpolationBox extent rounding before promoting the bound arithmetic.
    widths = torch.cat(((upper_a-lower_a)*.5, (upper_b-lower_b)*.5)).double()
    if not bool(torch.isfinite(widths).all()):
        raise ValueError('Interpolation extent exceeds native precision')
    orders = [test_order]*3+[source_order]*3
    wave = complex(wave_number)
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
        raise ValueError('Analytic bound requires a finite outgoing passive wave number')
    lower_a, upper_a = lower_a.double(), upper_a.double()
    lower_b, upper_b = lower_b.double(), upper_b.double()
    distance = torch.linalg.vector_norm(torch.maximum(lower_a-upper_b, lower_b-upper_a).clamp_min(0))
    if not bool(torch.isfinite(distance) & (distance > 0)):
        raise ValueError('Analytic bound requires strictly separated finite boxes')
    # Every s < sqrt(2)-1 keeps the principal square root away from its cut.
    fractions = torch.tensor([.025, .05, .1, .15, .2, .25, .3, .35, .4], device=device, dtype=torch.float64)
    alpha = torch.sqrt(1-2*fractions-fractions.square())
    delta = fractions*distance
    radial_shift = delta*(2+fractions)/(1+alpha)
    log_green_max = (abs(wave.real)*radial_shift-wave.imag*alpha*distance
                     -math.log(4*math.pi)-torch.log(alpha)-torch.log(distance))
    wave_scale = max(abs(wave.real), abs(wave.imag))
    log_wave = (math.log(wave_scale)+.5*math.log((wave.real/wave_scale)**2+(wave.imag/wave_scale)**2)
                if wave_scale else -math.inf)
    log_gradient_max = (log_green_max+torch.logaddexp(torch.full_like(alpha, log_wave),
                        -torch.log(alpha)-torch.log(distance))+torch.log1p(fractions)-torch.log(alpha))
    log_factor = torch.full_like(fractions, -torch.inf)
    log_amplification = 0.
    active_axes = 0
    for width, count in zip(widths, orders):
        if not bool(width > 0):
            continue
        active_axes += 1
        log_ratio = torch.log(delta)-torch.log(width)
        log_rho = torch.logaddexp(torch.zeros_like(log_ratio), log_ratio)
        log_term = log_amplification+math.log(4)-(count-1)*log_rho-log_ratio
        log_factor = torch.logaddexp(log_factor, log_term)
        log_amplification += math.log1p(2/math.pi*math.log(count))
    green_candidates, gradient_candidates = log_factor+log_green_max, log_factor+log_gradient_max
    # Combine all scales before exponentiation: an underflowed exponential may
    # otherwise be multiplied by a large inverse distance or wave number.
    green_candidates = torch.where(torch.isnan(green_candidates), torch.inf, green_candidates)
    gradient_candidates = torch.where(torch.isnan(gradient_candidates), torch.inf, gradient_candidates)
    if active_axes:
        floor = math.log(torch.finfo(torch.float64).tiny)
        green = torch.exp(green_candidates.min().clamp_min(floor))
        gradient = torch.exp(gradient_candidates.min().clamp_min(floor))
    else:
        green = widths.new_zeros(())
        gradient = widths.new_zeros(())
    if not bool(torch.isfinite(green) & torch.isfinite(gradient)):
        raise KernelBoundUnavailable('No finite analytic kernel bound in native precision')
    return {'version': 'scalar_chebyshev_ellipse_truncation_v1',
            'green_absolute_bound': float(green), 'gradient_component_absolute_bound': float(gradient),
            'active_axes': active_axes, 'scope': 'exact_arithmetic_kernel_interpolation_only',
            'floating_point_certified': False, 'quadrature_included': False,
            'operator_error_certified': False, 'physical_qualification': False}


def require_kernel_budget(receipt, budget):
    if not isinstance(budget, KernelErrorBudget):
        raise TypeError('Kernel error admission requires KernelErrorBudget')
    if (receipt['green_absolute_bound'] > budget.green_absolute
            or receipt['gradient_component_absolute_bound'] > budget.gradient_component_absolute):
        raise KernelInterpolationError({**receipt, 'budget': asdict(budget)})
