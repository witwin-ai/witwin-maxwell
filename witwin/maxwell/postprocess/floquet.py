"""Fixed-transverse-wavevector spectra on full periodic Cartesian planes.

Phasors use exp(-i omega t), with spatial harmonics exp(+i k.r). The
background is a homogeneous, isotropic, lossless ``Material``. Planes must
span one complete cell and lie in that background. Both tangential E
components are required; no collocation of staggered samples is performed.
"""

from __future__ import annotations

import math

import torch

from ..constants import C_0
from .diffraction import enumerate_diffraction_orders


def _periodic_weights(coords, period):
    gaps = torch.cat((coords[1:] - coords[:-1], (period - coords[-1] + coords[0]).reshape(1)))
    return (gaps + gaps.roll(1)) / (2 * period)


def floquet_orders(result, *, plane_monitor_name, period, bloch_wavevector, frequencies, background,
                   light_line_margin=0.05):
    """Return electric TE/TM amplitudes, with shape ``[frequency, order, 2]``.

    ``period=(La,Lb)`` and ``bloch_wavevector=(ka,kb)`` follow the plane's
    tangential axes in Cartesian order; frequencies are in Hz. ``orders`` is
    an integer ``[order,2]`` tensor, including a surrounding evanescent shell.
    ``mask[frequency,order]`` is True at cutoff, for evanescent orders, and
    below the incident light line times ``1 + light_line_margin`` (default
    5%). Masked amplitudes and kz are NaN. Set the margin to zero for the
    physical propagation cutoff alone.

    TE is ``normal x k_parallel_hat`` for the positive Cartesian normal.
    TM has positive tangential projection along ``k_parallel_hat``, for
    either propagation direction; its amplitude is the full electric-field
    magnitude. At zero transverse wavevector, TM follows the first tangential
    axis and TE follows normal x TM. A plane containing counterpropagating
    waves returns their electric superposition, not a directional separation.
    """
    frequencies = tuple(frequencies)
    payloads = [result.raw_monitor(plane_monitor_name, frequency=f) for f in frequencies]
    axis = payloads[0]["axis"]
    tangents = tuple(a for a in "xyz" if a != axis)
    fields = tuple("E" + a for a in tangents)
    first = payloads[0]["components"][fields[0]]["data"]
    device = first.device
    dtype = first.real.dtype
    index = math.sqrt(background.eps_r * background.mu_r)
    specs, _ = enumerate_diffraction_orders(
        periods=period, k_bloch=bloch_wavevector,
        k0=2 * math.pi * max(frequencies) / C_0, background_index=index,
    )
    orders = torch.tensor([(s["m"], s["n"]) for s in specs], device=device)
    kpar = torch.as_tensor(bloch_wavevector, device=device, dtype=dtype) + orders * (
        2 * math.pi / torch.as_tensor(period, device=device, dtype=dtype)
    )
    transverse = torch.linalg.vector_norm(kpar, dim=-1)
    # Normal incidence has no distinguished plane of incidence.
    tm = kpar / torch.where(transverse == 0, 1, transverse)[:, None]
    tm = torch.where((transverse == 0)[:, None], tm.new_tensor((1., 0.)), tm)
    orientation = -1 if axis == "y" else 1
    te = orientation * torch.stack((-tm[:, 1], tm[:, 0]), dim=-1)
    frequency = torch.tensor(frequencies, device=device, dtype=dtype)
    k = 2 * math.pi * frequency * index / C_0
    kz_squared = k[:, None].square() - transverse[None, :].square()
    incident_squared = sum(v * v for v in bloch_wavevector)
    mask = (kz_squared <= 0) | (k[:, None].square() <= incident_squared * (1 + light_line_margin) ** 2)
    kz = torch.sqrt(kz_squared.masked_fill(mask, 1.))
    coefficients = []
    for payload in payloads:
        projected = []
        for field in fields:
            entry = payload["components"][field]
            a, b = [torch.as_tensor(x, device=device, dtype=dtype) for x in entry["coords"]]
            wa, wb = [_periodic_weights(x, length) for x, length in zip((a, b), period)]
            phase_a = torch.exp(-1j * kpar[:, 0, None] * a) * wa
            phase_b = torch.exp(-1j * kpar[:, 1, None] * b) * wb
            projected.append(torch.einsum("oa,ab,ob->o", phase_a, entry["data"], phase_b))
        coefficients.append(torch.stack(projected, dim=-1))
    electric = torch.stack(coefficients)
    amplitudes = torch.stack((
        (electric * te).sum(-1),
        (electric * tm).sum(-1) * k[:, None] / kz,
    ), dim=-1).masked_fill(mask[..., None], complex(float("nan"), float("nan")))
    return dict(frequencies=frequency, orders=orders, k_parallel=kpar, kz=kz.masked_fill(mask, float("nan")),
                mask=mask, amplitudes=amplitudes, polarizations=("TE", "TM"),
                axis=axis, position=payloads[0]["position"], light_line_margin=light_line_margin)


def _line_reference(result, plane_monitor_name, spectrum, frequencies):
    reference = result.metadata["tfsf_incident_reference"]
    indices = [reference["frequencies"].index(f) for f in frequencies]
    electric = reference["electric"][indices].to(spectrum["amplitudes"])
    omega = 2 * math.pi * spectrum["frequencies"]
    discrete_omega = 2 * torch.sin(omega * reference["dt"] / 2) / (C_0 * reference["dt"])
    transverse_squared = sum(k * k for k in reference["discrete_wavevector"])
    kz = 2 / reference["ds"] * torch.asin(reference["ds"] / 2 *
        torch.sqrt((discrete_omega.square() - transverse_squared).to(electric.dtype)))
    kz = kz * reference["direction_sign"]
    zero = (spectrum["orders"] == 0).all(-1).nonzero().flatten()[0]
    tangents = [i for i, a in enumerate("xyz") if a != spectrum["axis"]]
    fields = []
    for i in tangents:
        propagated = []
        for j, frequency in enumerate(frequencies):
            payload = result.raw_monitor(plane_monitor_name, frequency=frequency)
            field = "E" + "xyz"[i]
            entry = payload["components"][field]
            factor = sum(weight * torch.exp(1j * kz[j] * (position - reference["position"]))
                         for weight, position in zip(entry["plane_weights"], entry["plane_positions"]))
            if "source_normalization_spectra" in payload:
                index = payload["source_normalization_frequencies"].index(frequency)
                denominator = torch.as_tensor(payload["source_normalization_spectra"][field], device=kz.device)
                factor = factor / denominator.reshape(-1)[index]
            propagated.append(factor)
        fields.append(electric[:, i] * torch.stack(propagated))
    fields = torch.stack(fields, -1)
    kpar = spectrum["k_parallel"][zero]
    tm = kpar / torch.linalg.vector_norm(kpar)
    te = (-1 if spectrum["axis"] == "y" else 1) * torch.stack((-tm[1], tm[0]))
    amplitudes = torch.zeros_like(spectrum["amplitudes"])
    amplitudes[:, zero, 0] = (fields * te).sum(-1)
    amplitudes[:, zero, 1] = (fields * tm).sum(-1) * (omega / C_0) / spectrum["kz"][:, zero]
    return dict(spectrum, amplitudes=amplitudes)


def reflection_transmission(result, incident_result=None, *, reflection_monitor_name,
                            transmission_monitor_name, period, bloch_wavevector,
                            frequencies, background, incident_polarization, light_line_margin=0.05):
    """Reference-normalized complex r/t for one TE or TM incident drive.

    Both planes lie INSIDE the total-field region, on opposite sides of the
    scatterer. Subtract the matched empty-cell field at the reflection plane;
    the transmission plane contains only the outgoing transmitted wave.
    Reference incidence is the (0,0) amplitude at the reflection plane.
    Runs must share pulse, k_parallel, grid, monitor positions and DFT window.
    For pulsed fixed-k slab injection, omit incident_result to use the saved
    auxiliary-line incident spectrum and discrete Yee propagation instead.

    r/t have shape [frequency,order,output polarization], including conversion.
    They include sqrt(kz_order/kz_incident), so abs(r)**2 and abs(t)**2 are
    power fractions in the common background. Phases are at the monitor planes;
    no propagation phase is removed. This function does not infer an S matrix
    from a single drive. Evanescent/cutoff entries and frequencies within
    ``light_line_margin`` of the incident light line remain masked NaNs.
    """
    kwargs = dict(period=period, bloch_wavevector=bloch_wavevector,
                  frequencies=tuple(frequencies), background=background, light_line_margin=light_line_margin)
    reflected = floquet_orders(result, plane_monitor_name=reflection_monitor_name, **kwargs)
    transmitted = floquet_orders(result, plane_monitor_name=transmission_monitor_name, **kwargs)
    incident = (_line_reference(result, reflection_monitor_name, reflected, tuple(frequencies))
                if incident_result is None else
                floquet_orders(incident_result, plane_monitor_name=reflection_monitor_name, **kwargs))
    zero = (incident["orders"] == 0).all(-1).nonzero().flatten()[0]
    pol = incident["polarizations"].index(incident_polarization)
    mask = incident["mask"]
    incident_amplitude = incident["amplitudes"].masked_fill(mask[..., None], 0.)
    denominator = incident_amplitude[:, zero, pol].masked_fill(mask[:, zero], 1.)[:, None, None]
    kz = incident["kz"].masked_fill(mask, 1.)
    power_scale = torch.sqrt(kz / kz[:, zero, None])[..., None]
    r = (reflected["amplitudes"].masked_fill(mask[..., None], 0.) - incident_amplitude) / denominator * power_scale
    t = transmitted["amplitudes"].masked_fill(mask[..., None], 0.) / denominator * power_scale
    masked_value = complex(float("nan"), float("nan"))
    return dict(orders=incident["orders"], frequencies=incident["frequencies"],
                polarizations=incident["polarizations"], mask=incident["mask"], light_line_margin=light_line_margin,
                r=r.masked_fill(mask[..., None], masked_value),
                t=t.masked_fill(mask[..., None], masked_value))


def resample_fixed_angle(labels_kpar_omega, theta, *, kpar, omega, mask=None, wave_speed=C_0):
    """Interpolate a fixed-azimuth [k_parallel,omega,...] label grid in Torch.

    kpar is an increasing scalar wavevector axis (rad/m), omega an increasing
    angular-frequency axis (rad/s), and theta is in radians. Return labels on
    k_parallel=omega*sin(theta)/wave_speed at the supplied omega samples.
    Linear interpolation never extrapolates or bridges masked entries. mask
    is True for invalid data, with shape [K,W] or the full label shape; NaNs
    are always masked. Returned labels/mask have shape [W,...].
    """
    labels = labels_kpar_omega
    kpar = torch.as_tensor(kpar, device=labels.device, dtype=labels.real.dtype)
    omega = torch.as_tensor(omega, device=labels.device, dtype=labels.real.dtype)
    target = omega * torch.sin(torch.as_tensor(theta, device=labels.device, dtype=labels.real.dtype)) / wave_speed
    upper = torch.searchsorted(kpar, target).clamp(1, len(kpar) - 1)
    lower = upper - 1
    weight = ((target - kpar[lower]) / (kpar[upper] - kpar[lower])).reshape(
        (-1,) + (1,) * (labels.ndim - 2))
    column = torch.arange(len(omega), device=labels.device)
    invalid = ~torch.isfinite(labels)
    if mask is not None:
        invalid = invalid | mask.reshape(mask.shape + (1,) * (labels.ndim - mask.ndim))
    left = labels[lower, column].masked_fill(invalid[lower, column], 0.)
    right = labels[upper, column].masked_fill(invalid[upper, column], 0.)
    outside = ((target < kpar[0]) | (target > kpar[-1])).reshape(weight.shape)
    masked = outside | (invalid[lower, column] & (weight != 1)) | (invalid[upper, column] & (weight != 0))
    values = torch.where(weight == 1, 0, left * (1 - weight)) + torch.where(weight == 0, 0, right * weight)
    return dict(labels=values.masked_fill(masked, float("nan")), mask=masked,
                kpar=target, omega=omega)
