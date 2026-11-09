"""Taylor primitives for Ewald screened real images and the primary remainder."""
import math
import torch
from .periodic import EwaldHelmholtzKernel
from .fmm_expansion import _points, multi_indices, _compose_radial_taylor


def ewald_real_taylor(kernel, displacement, wave_number, order, *, coefficient_budget_bytes=512*1024**2):
    """Expand one screened primary real-space kernel at nonzero centers.

    For F(s)=G_real(|R| sqrt(s)), the radial Helmholtz equation is
    4s F''+6F'+(k|R|)^2 F = |R|^2 S exp(-eta^2 |R|^2 s),
    S=eta^3 exp(k^2/(4eta^2))/pi^(3/2). Seeds use the configured real-space
    Ewald value and radial derivative. Quadrature and recurrence roundoff are
    not certified. No lattice image sum or Bloch factor is included here.
    """
    if not isinstance(kernel, EwaldHelmholtzKernel):
        raise TypeError('Real Taylor expansion requires EwaldHelmholtzKernel')
    _points(displacement)
    if displacement.device != kernel.lattice.vectors.device or displacement.dtype != kernel.lattice.vectors.dtype:
        raise ValueError('Real Taylor centers must match lattice coordinates')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Real Taylor wave number must be a fixed scalar')
    wave = complex(wave_number)
    kernel._guard(wave)
    indices = multi_indices(order)
    if type(coefficient_budget_bytes) is not int or coefficient_budget_bytes < 1:
        raise ValueError('Coefficient budget must be a positive integer')
    required = math.prod(displacement.shape[:-1])*len(indices)*2*displacement.element_size()
    if required > coefficient_budget_bytes:
        raise MemoryError('Real Taylor returned coefficient budget exceeded before radial seeds')
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    if not bool(((radius > 0) & torch.isfinite(radius)).all()):
        raise ValueError('Screened real Taylor centers must have finite nonzero radius')
    value, gradient = kernel._real(displacement.reshape(-1, 3), wave, images='zero')
    radial = [value.reshape(radius.shape)]
    if order:
        radial.append(.5*(gradient.reshape_as(displacement)*displacement).sum(-1))
    x = (kernel.eta*radius).square()
    source = radius.square()*kernel.eta**3/math.pi**1.5*torch.exp(-x+(wave/(2*kernel.eta))**2)
    kr2 = (wave*radius).square()
    for n in range(order-1):
        radial.append((source-(n+1)*(4*n+6)*radial[-1]-kr2*radial[-2])/(4*(n+2)*(n+1)))
        source = source*(-x)/(n+1)
    result = _compose_radial_taylor(displacement, radius, radial, order)
    kernel._guard(wave)
    return result


def ewald_primary_regular_taylor(kernel, wave_number, order):
    """Entire primary G_real-G_free coefficients about zero, without subtraction."""
    if not isinstance(kernel, EwaldHelmholtzKernel):
        raise TypeError('Primary regular Taylor expansion requires EwaldHelmholtzKernel')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Primary regular wave number must be a fixed scalar')
    wave = complex(wave_number)
    kernel._guard(wave)
    indices = multi_indices(order)
    zero = kernel.lattice.vectors.new_zeros((1, 3))
    radial = kernel._regular_radial_coefficients(zero, wave, order//2+1)
    result = []
    for alpha in indices:
        if any(i % 2 for i in alpha):
            result.append(torch.zeros_like(radial[0]))
        else:
            half = tuple(i//2 for i in alpha)
            factor = math.factorial(sum(half))//math.prod(math.factorial(i) for i in half)
            result.append(radial[sum(half)]*factor)
    result = torch.stack(result)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Primary regular Taylor coefficients exceed native precision')
    kernel._guard(wave)
    return result


def ewald_regular_taylor(kernel, wave_number, order, *, coefficient_budget_bytes=512*1024**2):
    """Configured G_periodic-G_free coefficients about the primary origin.

    Valid as a local series up to the nearest nonzero image singularity. This
    function does not admit a support radius or remove near nonzero images.
    It is not yet a complete periodic FMM coupling or an Ewald cutoff bound.
    """
    from .fmm_ewald_spectral import ewald_spectral_taylor
    if not isinstance(kernel, EwaldHelmholtzKernel):
        raise TypeError('Regular Taylor expansion requires EwaldHelmholtzKernel')
    zero = kernel.lattice.vectors.new_zeros(3)
    result = ewald_spectral_taylor(kernel, zero, wave_number, order,
        coefficient_budget_bytes=coefficient_budget_bytes)
    primary = ewald_primary_regular_taylor(kernel, wave_number, order)
    result += primary
    del primary
    output_bytes = result.numel()*result.element_size()
    batch = min(kernel.image_batch, coefficient_budget_bytes//output_bytes-1)
    # The spectral gate already admits output plus at least one mode batch.
    if batch < 1:
        raise MemoryError('Regular Taylor budget must hold output and one image coefficient batch')
    for indices in kernel._indices(kernel.real_shells, zero.device):
        indices = indices[(indices != 0).any(-1)]
        for selected in indices.split(batch):
            if not len(selected):
                continue
            coefficients = ewald_real_taylor(kernel, -kernel.lattice.translations(selected), wave_number, order,
                coefficient_budget_bytes=coefficient_budget_bytes-output_bytes)
            result += torch.einsum('b,bc->c', kernel.lattice.phase(selected), coefficients)
            del coefficients
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Regular Ewald Taylor sum exceeds native precision')
    kernel.lattice.assert_current()
    return result
