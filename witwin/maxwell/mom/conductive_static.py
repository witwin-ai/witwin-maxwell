"""Homogeneous DC Maxwell Green blocks with electric and magnetic conduction.

curl E = -sigma_m H - K, curl H = sigma_e E + J. For positive scalar
conductivities, (curl curl + kappa**2) E = -sigma_m J - curl K,
kappa**2 = sigma_e*sigma_m. The decaying scalar kernel solves
(-Delta+kappa**2)g=delta. Its vector inverse is Dg=(I-Hessian/kappa**2)g.
These are source-separated kernels, not singular cell or interface quadrature.
"""
from __future__ import annotations
import math
import torch


def conductive_dc_green(displacement, *, sigma_e, sigma_m):
    """Return EE, EK, HJ, HK blocks for electric J and magnetic K densities.

    Every block has shape [...,3,3], and is applied to the source vector in
    its final axis. Only strictly positive scalar conductivities are supported;
    zero/singular material limits require their distinct static formulations.
    The coordinate and material operations stay on CUDA and retain autograd.
    """
    if (not isinstance(displacement, torch.Tensor) or displacement.device.type != 'cuda'
            or displacement.dtype not in (torch.float32, torch.float64)
            or displacement.ndim < 1 or displacement.shape[-1] != 3
            or not bool(torch.isfinite(displacement).all())):
        raise ValueError('Conductive DC displacements require finite real CUDA [...,3]')
    def coefficient(value):
        if isinstance(value, torch.Tensor):
            if value.device != displacement.device or value.dtype != displacement.dtype or value.ndim != 0:
                raise ValueError('Conductive DC coefficients require matching real CUDA scalars')
            result = value
        else:
            if type(value) not in (float, int):
                raise ValueError('Conductive DC coefficients must be positive real scalars')
            result = displacement.new_tensor(value)
        if not bool(torch.isfinite(result)) or not bool(result > 0):
            raise ValueError('Conductive DC coefficients must be finite and strictly positive')
        return result
    electric, magnetic = coefficient(sigma_e), coefficient(sigma_m)
    square = electric*magnetic
    if not bool(torch.isfinite(square)) or not bool(square > 0):
        raise ValueError('Conductive DC screening scale is numerically unresolved')
    kappa = torch.sqrt(square)
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    if not bool(torch.isfinite(radius).all()) or bool((radius == 0).any()):
        raise ValueError('Conductive DC coincident nodes require singular quadrature')
    unit = displacement/radius[..., None]
    inverse = 1/radius
    green = torch.exp(-kappa*radius)*inverse/(4*math.pi)
    first_over_r = -green*(kappa+inverse)*inverse
    second = green*(square+2*kappa*inverse+2*inverse.square())
    eye = torch.eye(3, device=displacement.device, dtype=displacement.dtype)
    hessian = first_over_r[..., None, None]*eye+(second-first_over_r)[..., None, None]*unit[..., :, None]*unit[..., None, :]
    dyadic = green[..., None, None]*eye-hessian/square
    gradient = (-green*(kappa+inverse))[..., None]*unit
    x, y, z = gradient.unbind(-1); zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*gradient.shape[:-1], 3, 3)
    blocks = (-magnetic*dyadic, -cross, cross, -electric*dyadic)
    if any(not bool(torch.isfinite(block).all()) for block in blocks):
        raise ValueError('Conductive DC Green blocks exceed resolved arithmetic')
    return blocks


def conductive_dc_cell_fields(mesh, electric_current, magnetic_current, points, *, sigma_e, sigma_m,
                              order=6, principal_value=False):
    """Integrate real piecewise-constant DC currents, including the self contact.

    Reuses native tetrahedral scalar potentials and distributional Hessians at
    k=i*kappa. Conductivities are fixed Python scalars because that quadrature
    currently freezes its wavenumber; source-current autograd remains available.
    Fixed order is not an adaptive error or interface convergence certificate.
    """
    from .volume_mesh import TetrahedralVolumeMesh
    from .vie_operator import cell_green_potentials
    if not isinstance(mesh, TetrahedralVolumeMesh):
        raise TypeError('Conductive DC cell fields require a native volume mesh')
    for value in (sigma_e, sigma_m):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Cell conductivity requires a fixed finite positive Python scalar')
    square = sigma_e*sigma_m
    if not math.isfinite(square) or square <= 0:
        raise ValueError('Conductive DC screening scale is numerically unresolved')
    for current in (electric_current, magnetic_current):
        if (not isinstance(current, torch.Tensor) or current.device != mesh.vertices.device
                or current.dtype != mesh.vertices.dtype or current.shape != (len(mesh.cells), 3)
                or not bool(torch.isfinite(current).all())):
            raise ValueError('DC currents require matching real CUDA [cells,3]')
    scalar, gradient, hessian = cell_green_potentials(mesh, points, 1j*math.sqrt(square),
                                                     order=order, principal_value=principal_value)
    if any(not bool(torch.isfinite(value).all()) or bool(value.imag.any()) for value in (scalar, gradient, hessian)):
        raise ValueError('Real screened cell potentials are nonfinite or unexpectedly complex')
    scalar, gradient, hessian = scalar.real, gradient.real, hessian.real
    def dyadic(current):
        return torch.einsum('qt,ti->qi', scalar, current)-torch.einsum('qtij,tj->qi', hessian, current)/square
    def curl(current):
        return torch.linalg.cross(gradient, current[None].expand_as(gradient)).sum(1)
    electric = -sigma_m*dyadic(electric_current)-curl(magnetic_current)
    magnetic = curl(electric_current)-sigma_e*dyadic(magnetic_current)
    if not bool(torch.isfinite(electric).all() & torch.isfinite(magnetic).all()):
        raise ValueError('Conductive DC cell fields exceed resolved arithmetic')
    return electric, magnetic
