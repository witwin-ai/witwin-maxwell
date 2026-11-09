"""Native Cartesian Taylor coefficients of the Ewald reciprocal contribution."""
import math
import torch
from .periodic import EwaldHelmholtzKernel
from .fmm_expansion import multi_indices, _points
from .quadrature import gauss_rule


def quadratic_exponential_taylor(linear, quadratic, order):
    """Coefficients of exp(L.h + h.Q.h), with symmetric Q and batched L/Q.

    Differentiating along any nonzero multi-index component gives
    alpha_i c_alpha = L_i c_(alpha-e_i) + 2 sum_j Q_ij c_(alpha-e_i-e_j).
    All polynomial algebra stays on the input CUDA device.
    """
    indices = multi_indices(order)
    if (linear.device.type != 'cuda' or linear.dtype not in (torch.complex64, torch.complex128)
            or linear.shape[-1:] != (3,) or quadratic.shape != (*linear.shape[:-1], 3, 3)
            or quadratic.device != linear.device or quadratic.dtype != linear.dtype
            or linear.requires_grad or quadratic.requires_grad
            or not bool(torch.isfinite(linear).all() & torch.isfinite(quadratic).all())
            or not torch.equal(quadratic, quadratic.transpose(-1, -2))):
        raise ValueError('Gaussian Taylor data require finite fixed CUDA complex L and symmetric Q')
    values = {(0, 0, 0): torch.ones_like(linear[..., 0])}
    for alpha in indices[1:]:
        axis = next(i for i, value in enumerate(alpha) if value)
        previous = list(alpha)
        previous[axis] -= 1
        value = linear[..., axis]*values[tuple(previous)]
        for j in range(3):
            if previous[j]:
                lower = previous.copy()
                lower[j] -= 1
                value = value+2*quadratic[..., axis, j]*values[tuple(lower)]
        values[alpha] = value/alpha[axis]
    result = torch.stack(tuple(values[a] for a in indices), -1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Gaussian Taylor coefficients exceed native precision')
    return result


def ewald_spectral_taylor(kernel, displacement, wave_number, order, *, coefficient_budget_bytes=512*1024**2):
    """Taylor coefficients of the configured reciprocal term about one point.

    Uses the same finite reciprocal cube and outgoing heat-contour quadrature
    as EwaldHelmholtzKernel. Real-space images and the primary singular/regular
    split are not included. No truncation or floating-point certificate follows.
    The budget covers output plus one returned batched coefficient array;
    recurrence, contour and reduction workspaces remain outside that gate.
    """
    if not isinstance(kernel, EwaldHelmholtzKernel):
        raise TypeError('Spectral Taylor expansion requires EwaldHelmholtzKernel')
    _points(displacement)
    if (displacement.shape != (3,) or displacement.device != kernel.lattice.vectors.device
            or displacement.dtype != kernel.lattice.vectors.dtype):
        raise ValueError('Spectral Taylor center must match lattice coordinates')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Spectral Taylor wave number must be a fixed scalar')
    wave = complex(wave_number)
    kernel._guard(wave)
    indices = multi_indices(order)
    if type(coefficient_budget_bytes) is not int or coefficient_budget_bytes < 1:
        raise ValueError('Coefficient budget must be a positive integer')
    dtype = torch.complex128 if displacement.dtype == torch.float64 else torch.complex64
    itemsize = 2*displacement.element_size()
    output_bytes = len(indices)*itemsize
    dimension = kernel.lattice.dimension
    samples = 1 if dimension == 3 else kernel.quadrature_order
    batch = min(kernel.image_batch, (coefficient_budget_bytes//output_bytes-1)//samples)
    if batch < 1:
        raise MemoryError('Spectral Taylor budget must hold output and one reciprocal mode coefficient batch')
    result = torch.zeros(len(indices), device=displacement.device, dtype=dtype)
    if dimension == 3:
        alpha = torch.tensor(indices, device=displacement.device, dtype=torch.int64)
        factorials = displacement.new_tensor([math.prod(math.factorial(i) for i in p) for p in indices])
        if not bool(torch.isfinite(factorials).all()):
            raise ValueError('Spectral Taylor factorial weights exceed native precision')
    else:
        u, weights = gauss_rule(kernel.quadrature_order, device=displacement.device, dtype=displacement.dtype)
        projector = torch.eye(3, device=displacement.device, dtype=displacement.dtype)-kernel.lattice.reciprocal.T@kernel.lattice.vectors/(2*math.pi)
        perpendicular = displacement@projector
        quadratic_metric = projector@projector.T
        rho2 = perpendicular.square().sum()
        normalization = 1/(2*math.pi**((3-dimension)/2)*kernel.lattice.cell_measure)
    for original in kernel._indices(kernel.reciprocal_shells, displacement.device):
        for selected in original.split(batch):
            q = selected.to(displacement.dtype)@kernel.lattice.reciprocal+kernel.lattice.bloch
            a = wave*wave-q.square().sum(-1)
            phase = torch.exp(1j*(q@displacement))
            if dimension == 3:
                coefficients = (1j*q[:, None, :]).pow(alpha).prod(-1)/factorials
                if not bool(torch.isfinite(coefficients).all()):
                    raise ValueError('Spectral Taylor plane-wave coefficients exceed native precision')
                amplitude = -torch.exp(a/(4*kernel.eta**2))/(kernel.lattice.cell_measure*a)
                result += torch.einsum('b,bc->c', phase*amplitude, coefficients)
                del coefficients
            else:
                angle = torch.angle(a)
                if wave.real < 0:
                    angle = torch.where(angle > 0, angle-2*math.pi, angle)
                    theta = (angle+math.pi)/2
                else:
                    theta = (angle-math.pi)/2
                endpoint = kernel.eta*torch.exp(1j*theta)
                for arc in (False, True):
                    t = kernel.eta*torch.exp(1j*theta[:, None]*(1-u)) if arc else endpoint[:, None]*u
                    dt = -1j*theta[:, None]*t if arc else endpoint[:, None].expand_as(t)
                    linear = 1j*q[:, None, :]-2*t.square()[..., None]*(perpendicular@projector.T)
                    quadratic = -t.square()[..., None, None]*quadratic_metric
                    coefficients = quadratic_exponential_taylor(linear.to(dtype), quadratic.to(dtype), order)
                    amplitude = phase[:, None]*normalization*torch.exp(-rho2*t.square()+a[:, None]/(4*t.square()))*t**(-dimension)*dt*weights
                    result += torch.einsum('bq,bqc->c', amplitude, coefficients)
                    del coefficients
    kernel._guard(wave)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Spectral Taylor sum exceeds native precision')
    return result
