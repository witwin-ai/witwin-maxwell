"""Independent field-based power observations for a solved MoM current."""
from __future__ import annotations

import math
import torch

from .fields import decode, far_field


def integrate_volume_dissipation(medium, electric, magnetic, weights):
    """Material loss from independently supplied physical CUDA fields."""
    if (electric.shape != magnetic.shape or electric.shape[-1:] != (3,)
            or weights.shape != electric.shape[:-1] or electric.device.type != 'cuda'
            or magnetic.device != electric.device or weights.device != electric.device):
        raise ValueError('Volume weights and E/H must match on one CUDA device')
    if (not bool(torch.isfinite(weights).all() & (weights > 0).all())
            or not bool(torch.isfinite(electric).all() & torch.isfinite(magnetic).all())):
        raise ValueError('Volume fields and positive weights must be finite')
    from .tensor_medium import TensorMedium
    if isinstance(medium, TensorMedium):
        return (medium.dissipated_power_density(electric, magnetic)*weights).sum()
    density = .5*medium.omega*(medium.epsilon.imag*electric.abs().square().sum(-1)
                              + medium.mu.imag*magnetic.abs().square().sum(-1))
    return (density*weights).sum()


def volume_absorption(handle, coefficients, points, volume_weights, *, query_config=None):
    """Integrate physical interior dissipation over supplied volume quadrature.

    Weights have units m^3. This calculation uses independently sampled E/H,
    not the difference of extinction and scattering. Spatial quadrature error
    must be evaluated by refining the supplied volume rule.
    """
    if volume_weights.shape != points.shape[:-1] or volume_weights.device != points.device:
        raise ValueError("Volume weights must match the CUDA query points")
    if not bool(torch.isfinite(volume_weights).all() & (volume_weights > 0).all()):
        raise ValueError("Volume weights must be finite and positive in cubic metres")
    e,h = decode(handle,coefficients,points,region_id="interior",field_kind="total",query_config=query_config)
    return integrate_volume_dissipation(handle.interior, e, h, volume_weights)


def plane_wave_cross_sections(handle, coefficients, directions, solid_angle_weights, *, far_field_function=None):
    """Return scattered and extinguished power cross sections for one plane wave.

    Scattering uses angular quadrature of the complex far field. Extinction
    independently uses the forward optical theorem, including translation phase.
    """
    carrier = handle.incident_carrier
    carrier.validate_medium(handle.exterior)
    if len(carrier.waves) != 1 or carrier.dipoles:
        raise ValueError("Plane-wave cross sections require exactly one incident wave and no point dipoles")
    if solid_angle_weights.shape != directions.shape[:-1] or solid_angle_weights.device != directions.device:
        raise ValueError("Solid-angle weights must match the CUDA directions")
    if not bool(torch.isfinite(solid_angle_weights).all() & (solid_angle_weights > 0).all()):
        raise ValueError("Solid-angle weights must be finite and positive")
    if not torch.isclose(solid_angle_weights.sum(),torch.tensor(4*math.pi,device=directions.device,dtype=solid_angle_weights.dtype),rtol=1e-6,atol=1e-8):
        raise ValueError("Full-sphere solid-angle weights must sum to 4*pi")
    source = carrier.waves[0]
    amplitude = complex(*source.amplitude)
    if abs(amplitude) == 0:
        raise ValueError("Cross sections are undefined for zero incident amplitude")
    evaluate_far = (lambda directions: far_field(handle, coefficients, directions)) if far_field_function is None else far_field_function
    f,_ = evaluate_far(directions)
    scattering = (solid_angle_weights*f.abs().square().sum(-1)).sum()/abs(amplitude)**2
    forward_direction = torch.tensor([source.direction],device=directions.device,dtype=directions.dtype)
    forward,_ = evaluate_far(forward_direction)
    polarization = torch.tensor(source.polarization,device=directions.device,dtype=forward.dtype)
    extinction = 4*math.pi/handle.exterior.k.real*(amplitude.conjugate()*(forward[0]*polarization).sum()).imag/abs(amplitude)**2
    return {"scattering_m2":scattering,"extinction_m2":extinction,
            "absorption_by_balance_m2":extinction-scattering,
            "incident_intensity_W_per_m2":abs(amplitude)**2/(2*handle.exterior.eta.real)}
