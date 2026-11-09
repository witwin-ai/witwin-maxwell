"""Cartesian multipole/local Helmholtz primitives on native CUDA tensors.

These are polynomial multipole and local expansions, not interpolation bases.
Fixed order, fixed geometry and low electrical sizes are experimental; this
module alone is not a complete FMM backend or an error/complexity certificate.
"""
import math
import torch


def multi_indices(order):
    if type(order) is not int or order < 0:
        raise ValueError('Expansion order must be a nonnegative integer')
    return tuple((i, j, total-i-j) for total in range(order+1)
                 for i in range(total+1) for j in range(total-i+1))


def _points(value):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cuda'
            or value.dtype not in (torch.float32, torch.float64) or value.ndim < 1
            or value.shape[-1] != 3 or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError('Expansion geometry must be finite fixed real CUDA coordinates')


def monomials(offsets, order):
    """Return x**alpha in total-degree order, with no factorial normalization."""
    _points(offsets)
    indices = torch.tensor(multi_indices(order), device=offsets.device, dtype=torch.int64)
    return offsets.unsqueeze(-2).pow(indices).prod(-1)


def multipole_moments(offsets, charges, order):
    """M_alpha = sum_j q_j (y_j-c_source)**alpha, arbitrary leading RHS axes."""
    _points(offsets)
    dtype = torch.complex128 if offsets.dtype == torch.float64 else torch.complex64
    if (offsets.ndim != 2 or not isinstance(charges, torch.Tensor) or charges.ndim < 1
            or charges.shape[-1] != len(offsets) or charges.device != offsets.device
            or charges.dtype != dtype or not bool(torch.isfinite(charges).all())):
        raise ValueError('Multipole charges must match the fixed CUDA source points')
    return charges@monomials(offsets, order).to(dtype)


def shift_matrix(displacement, order):
    """S[a,b]=binom(a,b)*d**(a-b); M_parent=S M_child, L_child=S.T L_parent.

    d is child center minus parent center for both translations.
    """
    _points(displacement)
    if displacement.shape != (3,):
        raise ValueError('Expansion translation requires one displacement')
    indices = multi_indices(order)
    differences, factors = [], []
    for a in indices:
        for b in indices:
            valid = all(x >= y for x, y in zip(a, b))
            differences.append(tuple(max(0, x-y) for x, y in zip(a, b)))
            factors.append(math.prod(math.comb(x, y) for x, y in zip(a, b)) if valid else 0)
    powers = torch.tensor(differences, device=displacement.device, dtype=torch.int64)
    factors = torch.tensor(factors, device=displacement.device, dtype=displacement.dtype)
    return (factors*displacement.pow(powers).prod(-1)).reshape(len(indices), len(indices))


def helmholtz_taylor(displacement, wave_number, order):
    """Taylor coefficients g_alpha of exp(ik|R+d|)/(4*pi*|R+d|).

    Leading displacement dimensions are independent centers sharing one wave number.
    Dimensionless radial series F(s) obeys 4s F''+6F'+(k|R|)^2 F=0.
    Substitute s=1+2*(R/|R|).t+t.t, t=d/|R|, then rescale by |R|**|alpha|.
    """
    _points(displacement)
    indices = multi_indices(order)
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Helmholtz wave number must be a fixed scalar')
    try:
        wave = complex(wave_number)
    except (ValueError, TypeError) as error:
        raise ValueError('Helmholtz wave number must be finite and passive') from error
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
        raise ValueError('Helmholtz wave number must be finite and passive')
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    if not bool((torch.isfinite(radius) & (radius > 0)).all()):
        raise ValueError('M2L centers must be separated by a finite positive distance')
    dtype = torch.complex128 if displacement.dtype == torch.float64 else torch.complex64
    radial = _helmholtz_radial_series(radius, wave, order)
    return _compose_radial_taylor(displacement, radius, radial, order)


def _compose_radial_taylor(displacement, radius, radial, order):
    """Compose a series in s-1 with s=|R+h|^2/|R|^2 on fixed centers."""
    indices = multi_indices(order)
    dtype = radial[0].dtype
    lookup = {alpha: i for i, alpha in enumerate(indices)}
    shifts = []
    unit = displacement/radius[..., None]
    for axis in range(3):
        for degree in (1, 2):
            origins, destinations = [], []
            for i, alpha in enumerate(indices):
                beta = list(alpha)
                beta[axis] += degree
                if sum(beta) <= order:
                    origins.append(i)
                    destinations.append(lookup[tuple(beta)])
            shifts.append((torch.tensor(origins, device=displacement.device, dtype=torch.int64),
                           torch.tensor(destinations, device=displacement.device, dtype=torch.int64),
                           2*unit[..., axis, None] if degree == 1 else unit.new_ones(())))
    power = torch.zeros((*displacement.shape[:-1], len(indices)), device=displacement.device, dtype=dtype)
    power[..., 0] = 1
    result = radial[0][..., None]*power
    for n in range(1, order+1):
        product = torch.zeros_like(power)
        for origin, destination, coefficient in shifts:
            product.index_add_(-1, destination, power[..., origin]*coefficient)
        power = product
        result = result+radial[n][..., None]*power
    degrees = torch.tensor([sum(alpha) for alpha in indices], device=displacement.device)
    result = result/radius[..., None].pow(degrees)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Radial expansion exceeds native precision')
    return result


def multipole_to_local(displacement, wave_number, order, *, table_budget_bytes=512*1024**2):
    """Return batched Green and target-gradient M2L matrices [...,P,P], [...,3,P,P].

    L_beta=sum_alpha (-1)**|alpha| binom(alpha+beta,alpha) g_(alpha+beta) M_alpha.
    Source and target each retain total degree <= order. Derivatives of the
    kernel are expanded separately at the same source/target truncation.
    """
    _points(displacement)
    indices = multi_indices(order)
    itemsize = 16 if displacement.dtype == torch.float64 else 8
    if type(table_budget_bytes) is not int or table_budget_bytes < 1:
        raise ValueError('M2L table budget must be a positive integer')
    if 4*math.prod(displacement.shape[:-1])*len(indices)**2*itemsize > table_budget_bytes:
        raise MemoryError('M2L table budget exceeded before coefficient allocation')
    derivatives = helmholtz_taylor(displacement, wave_number, 2*order+1)
    lookup = {alpha: i for i, alpha in enumerate(multi_indices(2*order+1))}
    gather, factors, gradient_gather, gradient_factors = [], [], [], []
    for beta in indices:
        for alpha in indices:
            total = tuple(a+b for a, b in zip(alpha, beta))
            factor = (-1)**sum(alpha)*math.prod(math.comb(t, a) for t, a in zip(total, alpha))
            gather.append(lookup[total])
            factors.append(factor)
            for axis in range(3):
                shifted = list(total)
                shifted[axis] += 1
                gradient_gather.append(lookup[tuple(shifted)])
                gradient_factors.append(factor*(total[axis]+1))
    def table(ids, weights):
        ids = torch.tensor(ids, device=displacement.device, dtype=torch.int64)
        weights = torch.tensor(weights, device=displacement.device, dtype=displacement.dtype)
        return derivatives[..., ids]*weights
    size = len(indices)
    green = table(gather, factors).reshape(*displacement.shape[:-1], size, size)
    gradient = table(gradient_gather, gradient_factors).reshape(*displacement.shape[:-1], size, size, 3).movedim(-1, -3).contiguous()
    if not bool(torch.isfinite(green).all() & torch.isfinite(gradient).all()):
        raise ValueError('M2L coefficients exceed native precision')
    return green, gradient


def _helmholtz_radial_series(radius, wave, order):
    """Original radial recurrence on already admitted native radii and wave.

    Callers retain the same precision and finite positive-distance admission.
    This is the original polynomial series, with no new truncation or fitting.
    """
    dtype = torch.complex128 if radius.dtype == torch.float64 else torch.complex64
    kr = radius.to(dtype)*wave
    radial = [torch.exp(1j*kr)/(4*math.pi*radius)]
    if order:
        radial.append(radial[0]*(1j*kr-1)/2)
    for n in range(order-1):
        radial.append(-((n+1)*(4*n+6)*radial[-1]+kr.square()*radial[-2])/(4*(n+2)*(n+1)))
    return radial
