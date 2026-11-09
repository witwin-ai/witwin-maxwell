"""Independent CUDA auxiliary electric-dipole reference for a smooth dielectric.

This benchmark-only fundamental-solution collocation is neither a volume dipole
approximation nor a surface-current discretization. It imports no MoM/FDTD kernels.
Qualification is empirical: source/order/rank sensitivity and independent holdout
checks are required; a small least-squares residual is not an error certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

C0 = 299792458.0
MU0 = 4 * math.pi * 1e-7
ETA0 = MU0 * C0


def angular_rule(order, *, offset=0.0, device="cuda"):
    """Positive Gauss-Legendre/azimuth rule on the unit sphere."""
    if not isinstance(order, int) or order < 2:
        raise ValueError("angular order must be an integer >= 2")
    mu, weights = np.polynomial.legendre.leggauss(order)
    phi = (np.arange(2 * order) + offset) * math.pi / order
    z, p = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(p), np.sqrt(1-z*z)*np.sin(p), z), -1).reshape(-1, 3)
    weights = np.repeat(weights, 2 * order) * math.pi / order
    return (torch.as_tensor(directions, device=device, dtype=torch.float64),
            torch.as_tensor(weights, device=device, dtype=torch.float64))


def ellipsoid_surface(axes, order, *, offset, device="cuda"):
    directions, weights = angular_rule(order, offset=offset, device=device)
    axes = torch.as_tensor(axes, device=device, dtype=torch.float64)
    raw_normal = directions / axes
    normal_length = torch.linalg.vector_norm(raw_normal, dim=-1)
    normal = raw_normal / normal_length[:, None]
    pole = torch.zeros_like(normal)
    pole[:, 2] = 1
    first = torch.linalg.cross(pole, normal, dim=-1)
    first /= torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    tangent = torch.stack((first, torch.linalg.cross(normal, first, dim=-1)), 1)
    area_weights = weights * torch.prod(axes) * normal_length
    return directions * axes, normal, tangent, area_weights


def electric_dipole_kernels(points, sources, k, eta):
    """Return [point, source, field-component, moment-component] E/H kernels.

    SI metres and exp(-i omega t), outgoing g=exp(ikr)/(4*pi*r).
    E=(I+grad grad/k^2)g p; H=curl(E)/(i*k*eta). The expansion moment p
    has units V so E is V/m and H is A/m; the physical electric-dipole
    prefactor has been absorbed into p. Queries at source locations are invalid.
    """
    if points.device.type != "cuda" or sources.device != points.device:
        raise ValueError("Dipole kernels require points and sources on one CUDA device")
    if points.dtype != torch.float64 or sources.dtype != torch.float64:
        raise ValueError("Dipole reference coordinates require float64")
    delta = points[:, None, :] - sources[None, :, :]
    radius = torch.linalg.vector_norm(delta, dim=-1)
    if bool(torch.any(radius == 0)):
        raise ValueError("A dipole field is undefined at its source")
    normal = delta / radius[:, :, None]
    kr = k * radius
    green = torch.exp(1j * kr) / (4 * math.pi * radius)
    identity = torch.eye(3, device=points.device, dtype=torch.complex128)
    electric = green[:, :, None, None] * (
        (1+1j/kr-1/kr**2)[:, :, None, None] * identity
        + (-1-3j/kr+3/kr**2)[:, :, None, None]
        * normal[:, :, :, None] * normal[:, :, None, :])
    cross = torch.zeros((*radius.shape, 3, 3), device=points.device, dtype=torch.complex128)
    cross[:, :, 0, 1] = -normal[:, :, 2]
    cross[:, :, 0, 2] = normal[:, :, 1]
    cross[:, :, 1, 0] = normal[:, :, 2]
    cross[:, :, 1, 2] = -normal[:, :, 0]
    cross[:, :, 2, 0] = -normal[:, :, 1]
    cross[:, :, 2, 1] = normal[:, :, 0]
    magnetic = (green / eta * (1+1j/kr))[:, :, None, None] * cross
    return electric, magnetic


def _tangent_kernel(tangent, kernel):
    return torch.einsum("pta,psab->ptsb", tangent.to(torch.complex128), kernel).reshape(2*len(tangent), -1)


def weighted_relative(actual, reference, weights):
    numerator = torch.sum(weights[:, None] * abs(actual-reference)**2)
    denominator = torch.sum(weights[:, None] * abs(reference)**2)
    return float(torch.sqrt(numerator/denominator)) if float(denominator) > 0 else None


@dataclass(frozen=True)
class AuxiliaryConfig:
    order: int = 16
    inner_factor: float = 0.6
    outer_factor: float = 2.25
    rcond: float = 1e-12
    collocation_order: int | None = None


@dataclass
class AuxiliaryReference:
    axes: torch.Tensor
    epsilon: complex
    frequency: float
    phase: float
    exterior_sources: torch.Tensor
    interior_sources: torch.Tensor
    exterior_moments: torch.Tensor
    interior_moments: torch.Tensor
    config: AuxiliaryConfig
    solve_log: dict

    @property
    def k0(self):
        return 2 * math.pi * self.frequency / C0

    @property
    def k1(self):
        return self.k0 * np.sqrt(self.epsilon)

    @property
    def eta1(self):
        return ETA0 / np.sqrt(self.epsilon)

    def incident(self, points):
        electric = torch.zeros_like(points, dtype=torch.complex128)
        electric[:, 0] = torch.exp(1j * (self.k0 * points[:, 2] - self.phase))
        magnetic = torch.zeros_like(electric)
        magnetic[:, 1] = electric[:, 0] / ETA0
        return electric, magnetic

    def _potential(self, points, interior):
        sources = self.interior_sources if interior else self.exterior_sources
        moments = self.interior_moments if interior else self.exterior_moments
        e, h = electric_dipole_kernels(points, sources,
            self.k1 if interior else self.k0, self.eta1 if interior else ETA0)
        return (torch.einsum("psab,sb->pa", e, moments),
                torch.einsum("psab,sb->pa", h, moments))

    def fields(self, points, *, region, kind="total"):
        if region not in ("interior", "exterior") or kind not in ("total", "scattered"):
            raise ValueError("Use an interior/exterior region and total/scattered field kind")
        points = torch.as_tensor(points, device=self.axes.device, dtype=torch.float64)
        level = torch.sum((points/self.axes)**2, dim=-1)
        invalid_region = level >= 1 if region == "interior" else level <= 1
        if bool(torch.any(invalid_region)):
            raise ValueError("Physical query points must lie strictly in the declared ellipsoid region")
        e, h = self._potential(points, region == "interior")
        ie, ih = self.incident(points)
        if region == "exterior" and kind == "total":
            e, h = e+ie, h+ih
        elif region == "interior" and kind == "scattered":
            e, h = e-ie, h-ih
        return e, h

    def far_field(self, directions):
        directions = torch.as_tensor(directions, device=self.axes.device, dtype=torch.float64)
        if not torch.allclose(torch.linalg.vector_norm(directions, dim=-1), torch.ones(len(directions),device=self.axes.device,dtype=torch.float64), rtol=1e-12, atol=1e-12):
            raise ValueError("Far-field directions must be unit vectors")
        phase = torch.exp(-1j*self.k0*(directions@self.exterior_sources.T))/(4*math.pi)
        moments = phase@self.exterior_moments
        return moments-directions*torch.sum(directions*moments, dim=-1, keepdim=True)

    def holdout(self, *, order=None, offset=0.217):
        order = max(31, 2*self.config.order+3) if order is None else order
        points, normal, tangent, weights = ellipsoid_surface(self.axes, order, offset=offset, device=self.axes.device)
        eo, ho = self._potential(points, False)
        ei, hi = self._potential(points, True)
        ie, ih = self.incident(points)
        eo, ho = eo+ie, ho+ih
        tangential = lambda value: torch.einsum("pta,pa->pt", tangent.to(torch.complex128), value)
        normal_part = lambda value: torch.sum(normal*value, dim=-1, keepdim=True)
        return dict(order=order, points=len(points), azimuth_offset=offset,
            tangential_E=weighted_relative(tangential(ei), tangential(eo), weights),
            tangential_H=weighted_relative(tangential(hi), tangential(ho), weights),
            normal_D=weighted_relative(self.epsilon*normal_part(ei), normal_part(eo), weights),
            normal_B=weighted_relative(normal_part(hi), normal_part(ho), weights))


def solve_reference(*, axes=(.1,.08,.065), epsilon=2.25+0j, frequency=3e8,
                    phase=0.0, config=AuxiliaryConfig(), device="cuda"):
    """Column-balanced CUDA complex128 SVD least squares of E_t and eta0 H_t.

    The vacuum exterior uses sources on the interior auxiliary shell; the
    dielectric interior uses sources on the exterior shell. Both fields satisfy
    their homogeneous Maxwell equations exactly away from their source points.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Auxiliary reference solves require CUDA; no CPU fallback")
    if len(axes) != 3 or any(not math.isfinite(float(a)) or float(a) <= 0 for a in axes):
        raise ValueError("Three finite positive ellipsoid semiaxes are required")
    epsilon = complex(epsilon)
    if not (math.isfinite(epsilon.real) and math.isfinite(epsilon.imag) and epsilon.real > 0 and epsilon.imag >= 0):
        raise ValueError("The reference supports a passive positive scalar dielectric")
    if not math.isfinite(frequency) or frequency <= 0 or not math.isfinite(phase):
        raise ValueError("Frequency must be positive and phase must be finite")
    if not (all(math.isfinite(value) for value in (config.inner_factor,config.outer_factor,config.rcond))
            and 0 < config.inner_factor < 1 < config.outer_factor and 0 < config.rcond < 1):
        raise ValueError("Require 0 < inner_factor < 1 < outer_factor and 0 < rcond < 1")
    collocation_order = 2*config.order+1 if config.collocation_order is None else config.collocation_order
    if collocation_order <= config.order:
        raise ValueError("Collocation must oversample the auxiliary source rule")
    start = time.perf_counter()
    directions, _ = angular_rule(config.order, device=device)
    axes = torch.tensor(axes, device=device, dtype=torch.float64)
    count = len(directions)
    reference = AuxiliaryReference(axes, epsilon, frequency, phase,
        config.inner_factor*directions*axes, config.outer_factor*directions*axes,
        torch.empty((count,3),device=device,dtype=torch.complex128),
        torch.empty((count,3),device=device,dtype=torch.complex128), config, {})
    points, _, tangent, weights = ellipsoid_surface(axes, collocation_order, offset=.371, device=device)
    rows = 2*len(points)
    matrix = torch.empty((2*rows, 6*count), device=device, dtype=torch.complex128)
    for side, (sources, k, eta) in enumerate(((reference.exterior_sources, reference.k0, ETA0),
                                             (reference.interior_sources, reference.k1, reference.eta1))):
        electric, magnetic = electric_dipole_kernels(points, sources, k, eta)
        columns = slice(3*count*side, 3*count*(side+1))
        sign = 1 if side == 0 else -1
        matrix[:rows, columns] = sign*_tangent_kernel(tangent, electric)
        matrix[rows:, columns] = sign*ETA0*_tangent_kernel(tangent, magnetic)
        del electric, magnetic
    ie, ih = reference.incident(points)
    rhs = -torch.cat((torch.einsum("pta,pa->pt",tangent.to(torch.complex128),ie).reshape(-1),
                      ETA0*torch.einsum("pta,pa->pt",tangent.to(torch.complex128),ih).reshape(-1)))
    row_scale = torch.sqrt(weights).repeat_interleave(2).repeat(2)
    matrix *= row_scale[:, None]
    rhs *= row_scale
    column_scale = torch.linalg.vector_norm(matrix, dim=0)
    matrix /= column_scale[None, :]
    torch.cuda.synchronize(device)
    svd_start = time.perf_counter()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
    keep = singular > config.rcond*singular[0]
    projection = u.conj().T@rhs
    projected = torch.where(keep, projection/singular, torch.zeros_like(projection))
    balanced_moments = vh.conj().T@projected
    moments = balanced_moments/column_scale
    residual = float(torch.linalg.vector_norm(matrix@balanced_moments-rhs)/torch.linalg.vector_norm(rhs))
    reference.exterior_moments = moments[:3*count].reshape(-1,3).clone()
    reference.interior_moments = moments[3*count:].reshape(-1,3).clone()
    torch.cuda.synchronize(device)
    reference.solve_log = dict(matrix_shape=list(matrix.shape), collocation_order=collocation_order,
        collocation_points=len(points), collocation_azimuth_offset=.371, sources_per_shell=count,
        rcond=config.rcond, svd_rank=int(keep.sum()), singular_values=singular.cpu().tolist(),
        condition_untruncated=float(singular[0]/singular[-1]),
        condition_retained=float(singular[0]/singular[keep][-1]),
        coefficient_norm=float(torch.linalg.vector_norm(moments)),
        weighted_collocation_residual=residual, svd_seconds=time.perf_counter()-svd_start,
        elapsed_seconds=time.perf_counter()-start, dtype="complex128",device=str(device),
        reference_certified=False, error_bound="none; independent empirical qualification required")
    return reference
