from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from ..sources import CW, GaussianPulse, PlaneWave, RickerWavelet
from ..constants import resolve_real_dtype
from .stratton_chu import _resolve_tensor_device, _to_real_tensor


def _resolve_frequency(far_field, source=None):
    if "frequency" in far_field:
        return far_field["frequency"]
    if source is not None and source.source_time is not None:
        return float(source.source_time.frequency)
    raise ValueError("frequency is required in far_field or must be inferable from source.")


def _resolve_c(*, solver=None, c: float | None = None) -> float:
    if solver is not None:
        c = getattr(solver, "c", getattr(solver, "c0", c))
    if c is None:
        raise ValueError("c is required for RCS normalization. Pass solver=... or explicit c=...")
    return float(c)


def _plane_wave_sources(*, source=None, scene=None, result=None) -> list[PlaneWave]:
    if source is not None:
        if not isinstance(source, PlaneWave):
            raise TypeError("source must be a PlaneWave when provided.")
        return [source]

    if result is not None:
        scene = result.scene
    if scene is None:
        return []

    return [candidate for candidate in getattr(scene, "sources", []) if isinstance(candidate, PlaneWave)]


def infer_incident_plane_wave_amplitude(*, source=None, scene=None, result=None) -> float:
    plane_wave_sources = _plane_wave_sources(source=source, scene=scene, result=result)
    if not plane_wave_sources:
        raise ValueError("No PlaneWave source available to infer incident amplitude.")
    if len(plane_wave_sources) != 1:
        raise ValueError("Expected exactly one PlaneWave source when inferring incident amplitude.")

    plane_wave = plane_wave_sources[0]
    source_time = plane_wave.source_time
    if source_time is None:
        return 1.0
    if isinstance(source_time, (CW, GaussianPulse, RickerWavelet)):
        return float(source_time.amplitude)
    if isinstance(source_time, Mapping) and "amplitude" in source_time:
        return float(source_time["amplitude"])
    raise TypeError("Unsupported PlaneWave source_time type for amplitude inference.")


def _scattered_field_magnitude_sq(far_field) -> torch.Tensor:
    device = _resolve_tensor_device(far_field.get("E_theta"), far_field.get("E_phi"), far_field.get("E"))
    dtype = resolve_real_dtype(far_field.get("E_theta"), far_field.get("E_phi"), far_field.get("E"))
    complex_dtype = torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    if "E_theta" in far_field or "E_phi" in far_field:
        e_theta = torch.as_tensor(far_field.get("E_theta", 0.0), device=device, dtype=complex_dtype)
        e_phi = torch.as_tensor(far_field.get("E_phi", 0.0), device=device, dtype=complex_dtype)
        return torch.abs(e_theta).square() + torch.abs(e_phi).square()

    e_field = torch.as_tensor(far_field["E"], device=device, dtype=complex_dtype)
    return torch.sum(torch.abs(e_field).square(), dim=-1)


def compute_bistatic_rcs(
    far_field,
    *,
    incident_amplitude: float | torch.Tensor | None = None,
    source=None,
    scene=None,
    result=None,
    solver=None,
    c: float | None = None,
    eps: float = 1e-30,
) -> dict:
    if incident_amplitude is None:
        incident_amplitude = infer_incident_plane_wave_amplitude(
            source=source,
            scene=scene,
            result=result,
        )

    frequency = _resolve_frequency(far_field, source=source)
    frequency_value = float(frequency) if not isinstance(frequency, torch.Tensor) else float(frequency.item())
    c_value = _resolve_c(solver=solver, c=c)
    wavelength = c_value / frequency_value
    device = _resolve_tensor_device(far_field.get("radius"), incident_amplitude)
    dtype = resolve_real_dtype(far_field.get("radius"), incident_amplitude)
    radius = _to_real_tensor(far_field["radius"], device=device, dtype=dtype)
    incident_amplitude_tensor = _to_real_tensor(incident_amplitude, device=device, dtype=dtype)
    if float(torch.abs(incident_amplitude_tensor).item()) <= 0.0:
        raise ValueError("incident_amplitude must be non-zero.")

    scattered_field_sq = _scattered_field_magnitude_sq(far_field).to(device=device)
    sigma = 4.0 * math.pi * radius.square() * scattered_field_sq / torch.abs(incident_amplitude_tensor).square()
    normalized_rcs = sigma / (wavelength**2)
    rcs_db = 10.0 * torch.log10(torch.clamp_min(normalized_rcs, eps))
    rcs_dbsm = 10.0 * torch.log10(torch.clamp_min(sigma, eps))

    output = dict(far_field)
    output["incident_amplitude"] = incident_amplitude_tensor
    output["wavelength"] = wavelength
    output["rcs"] = sigma
    output["normalized_rcs"] = normalized_rcs
    output["rcs_db"] = rcs_db
    output["rcs_dbsm"] = rcs_dbsm
    return output


def transform_to_bistatic_rcs(
    transformer,
    theta,
    phi,
    *,
    radius: float | torch.Tensor = 1.0,
    incident_amplitude: float | torch.Tensor | None = None,
    source=None,
    scene=None,
    result=None,
    batch_size: int = 1024,
) -> dict:
    far_field = transformer.transform(theta, phi, radius=radius, batch_size=batch_size)
    return compute_bistatic_rcs(
        far_field,
        incident_amplitude=incident_amplitude,
        source=source,
        scene=scene,
        result=result,
        solver=getattr(transformer, "solver", None),
        c=getattr(transformer, "c", None),
    )
