from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from ..sources import CW, PlaneWave, TFSF, evaluate_source_time
from .stratton_chu import _trapz_weights_1d

_MU0 = 4.0 * torch.pi * 1e-7
_POWER_EPS = 1e-30
_COORD_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def _result_frequencies(result) -> torch.Tensor:
    frequencies = torch.as_tensor(getattr(result, "frequencies", ()), dtype=torch.float64)
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("result must expose one or more frequencies.")
    return frequencies


def _as_1d_tensor(value, *, size: int, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = torch.as_tensor(value, device=device, dtype=dtype)
    if array.ndim == 0:
        array = torch.full((size,), array.item(), device=device, dtype=dtype)
    if tuple(array.shape) != (size,):
        raise ValueError(f"{name} must be broadcastable to shape ({size},), got {tuple(array.shape)}.")
    return array


def _monitor_payload(result, name: str) -> dict:
    try:
        payload = result.monitor(name)
    except KeyError as exc:
        raise KeyError(f"Monitor {name!r} is not available in the supplied result.") from exc
    if "flux" not in payload:
        raise ValueError(
            f"Monitor {name!r} does not provide flux data. "
            "Use FluxMonitor or PlaneMonitor(compute_flux=True)."
        )
    return payload


def _assert_matching_frequencies(actual: torch.Tensor, expected: torch.Tensor, *, name: str):
    if tuple(actual.shape) != tuple(expected.shape) or not torch.allclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"{name} frequencies must match exactly. "
            f"Expected {tuple(expected.tolist())}, got {tuple(actual.tolist())}."
        )


def _monitor_flux(result, name: str, *, expected_frequencies: torch.Tensor) -> torch.Tensor:
    payload = _monitor_payload(result, name)
    monitor_frequencies = torch.as_tensor(
        payload.get("frequencies", (payload.get("frequency"),)),
        dtype=torch.float64,
    )
    _assert_matching_frequencies(monitor_frequencies, expected_frequencies, name=f"Monitor {name!r}")
    return _as_1d_tensor(
        payload["flux"],
        size=expected_frequencies.numel(),
        name=f"Monitor {name!r} flux",
        device=expected_frequencies.device,
        dtype=expected_frequencies.dtype,
    )


def _monitor_area(result, monitor_name: str) -> float:
    payload = _monitor_payload(result, monitor_name)
    axis = str(payload["axis"]).lower()
    coord_names = _COORD_NAMES[axis]
    coords = payload.get("coords")
    if coords is not None and len(coords) == 2:
        coord_a, coord_b = coords
    elif all(name in payload for name in coord_names):
        coord_a, coord_b = payload[coord_names[0]], payload[coord_names[1]]
    else:
        extents = {
            "x": result.scene.domain.bounds[0],
            "y": result.scene.domain.bounds[1],
            "z": result.scene.domain.bounds[2],
        }
        return float(extents[coord_names[0]][1] - extents[coord_names[0]][0]) * float(
            extents[coord_names[1]][1] - extents[coord_names[1]][0]
        )

    coord_a_tensor = torch.as_tensor(coord_a, dtype=torch.float64)
    coord_b_tensor = torch.as_tensor(coord_b, dtype=torch.float64)
    weights = _trapz_weights_1d(coord_a_tensor)[:, None] * _trapz_weights_1d(coord_b_tensor)[None, :]
    return float(torch.sum(weights).item())


def _incident_beam_area(source, result, incident_monitor: str) -> float:
    """Cross-sectional area the incident PlaneWave illuminates at the monitor.

    A TFSF box injects the incident field only inside its aperture, so the flux a
    plane monitor spanning the whole cross-section reports comes from that
    aperture alone; the relevant area is the aperture face transverse to the
    propagation axis, not the full monitor plane. A soft source (or a full-width
    TFSF slab) fills the whole cross-section, so the monitor area is used.
    """

    injection = getattr(source, "injection", "soft")
    if isinstance(injection, TFSF) and getattr(injection, "mode", "box") == "box" and injection.bounds is not None:
        direction = tuple(float(component) for component in getattr(source, "direction", (0.0, 0.0, 1.0)))
        axis_index = max(range(3), key=lambda index: abs(direction[index]))
        transverse = [index for index in range(3) if index != axis_index]
        bounds = injection.bounds
        return float(bounds[transverse[0]][1] - bounds[transverse[0]][0]) * float(
            bounds[transverse[1]][1] - bounds[transverse[1]][0]
        )
    return _monitor_area(result, incident_monitor)


def _plane_wave_cw_amplitude(source_time) -> float | None:
    """Peak amplitude of a steady-state CW PlaneWave, or ``None`` if broadband.

    A missing ``source_time`` defaults to a unit-amplitude CW carrier at the
    monitor frequency, matching the injector's default excitation.
    """

    if source_time is None:
        return 1.0
    if isinstance(source_time, CW):
        return float(source_time.amplitude)
    if isinstance(source_time, Mapping) and str(source_time.get("kind", "")).lower() == "cw":
        return float(source_time["amplitude"])
    return None


def _broadband_plane_wave_incident_power(
    result,
    source_time,
    frequencies: torch.Tensor,
    *,
    eta0: float,
    area: float,
) -> torch.Tensor:
    """Analytic incident power spectrum of a broadband PlaneWave.

    A soft PlaneWave face radiates the incident field forward with unit gain, so
    the incident field at the monitor is the source waveform itself,
    ``E_inc(t) = A * s(t)``, and an empty reference monitor reports the forward
    flux ``|E_inc(f)|^2 / (2*eta0) * area`` per frequency, where ``area`` is the
    illuminated cross-section (the TFSF aperture, or the full monitor plane for a
    soft source). This reduces to ``A^2 / (2*eta0) * area`` at the CW carrier,
    matching the closed-form CW branch.

    The monitor forms ``E_inc(f)`` with a windowed running DFT: for a non-CW
    waveform the solver forces ``window="none"`` and ``start_step=0`` (see
    ``fdtd/runtime/spectral.py``), and the reported phasor is
    ``(2/sum_w) * sum_n w(n) * field(n) * exp(i*w*n*dt)``. Reproducing that exact
    discrete transform of the *known* injected waveform yields the reference
    incident power without a second simulation: the injection-to-monitor
    propagation delay only rotates a global phase, which the ``|.|^2`` magnitude
    discards. The reconstruction is co-located (the leading Yee E/H stagger
    factor ``cos(k~*d/2)`` the monitor applies is dropped), so the absolute power
    matches the reference monitor to the grid-dispersion level of that stagger,
    which vanishes as the grid is refined -- the same regime the soft-PlaneWave
    absolute-power calibration is pinned at.
    """

    solver = getattr(result, "solver", None)
    entries = list(getattr(solver, "_observer_spectral_entries", ()) or ())
    dt = getattr(solver, "dt", None)
    if solver is None or not entries or dt is None:
        raise ValueError(
            "incident_power='auto' for a broadband (non-CW) PlaneWave needs the FDTD "
            "solver's running-DFT schedule (dt and per-frequency spectral entries), which "
            "is only present on a freshly run FDTD Result. Pass reference_result (an empty "
            "run) or an explicit incident_power array instead."
        )

    dt = float(dt)
    window_type = str(getattr(solver, "observer_window_type", "none"))
    entry_frequencies = np.array([float(entry["frequency"]) for entry in entries], dtype=np.float64)

    # Evaluate the injected waveform once over the widest accumulated step range.
    max_count = max(int(entry["sample_count"]) for entry in entries)
    min_start = min(int(entry["start_step"]) for entry in entries)
    if max_count <= 0:
        raise ValueError(
            "incident_power='auto' requires a completed FDTD run: the monitor DFT "
            "accumulated zero samples."
        )
    steps = np.arange(min_start, min_start + max_count, dtype=np.float64)
    waveform = np.array([evaluate_source_time(source_time, float(step) * dt) for step in steps], dtype=np.float64)

    powers = []
    for frequency in frequencies.tolist():
        match_index = int(np.argmin(np.abs(entry_frequencies - float(frequency))))
        if abs(entry_frequencies[match_index] - float(frequency)) > 1e-6 * max(abs(float(frequency)), 1.0):
            raise ValueError(
                f"No solver DFT entry matches monitor frequency {float(frequency)!r}; "
                "the incident-power normalization cannot be reconstructed."
            )
        entry = entries[match_index]
        start = int(entry["start_step"])
        count = int(entry["sample_count"])
        norm = float(entry["window_normalization"])
        if count <= 0 or norm <= _POWER_EPS:
            raise ValueError(
                f"Monitor DFT for frequency {float(frequency)!r} accumulated no weighted "
                "samples; incident-power normalization is undefined."
            )
        local = slice(start - min_start, start - min_start + count)
        step_slice = steps[local]
        signal = waveform[local]
        if window_type == "none":
            weights = np.ones(count, dtype=np.float64)
        else:
            weights = np.array(
                [
                    float(solver._compute_window_weight(int(step), start_step=start, end_step=entry["end_step"], window_type=window_type))
                    for step in step_slice
                ],
                dtype=np.float64,
            )
        phase = np.exp(1j * (2.0 * np.pi * float(frequency)) * step_slice * dt)
        spectral_amplitude = (2.0 / norm) * np.sum(weights * signal * phase)
        powers.append(float(abs(spectral_amplitude) ** 2) / (2.0 * eta0) * area)

    return torch.as_tensor(powers, device=frequencies.device, dtype=frequencies.dtype)


def _plane_wave_incident_power(result, incident_monitor: str, frequencies: torch.Tensor) -> torch.Tensor:
    plane_wave_sources = [source for source in getattr(result.scene, "sources", ()) if isinstance(source, PlaneWave)]
    if len(plane_wave_sources) != 1:
        raise ValueError("incident_power='auto' requires exactly one PlaneWave source on result.scene.")

    source = plane_wave_sources[0]
    source_time = source.source_time

    solver = getattr(result, "solver", None)
    c_value = float(getattr(solver, "c", getattr(solver, "c0", 299792458.0)))
    eta0 = float(_MU0) * c_value
    area = _incident_beam_area(source, result, incident_monitor)

    cw_amplitude = _plane_wave_cw_amplitude(source_time)
    if cw_amplitude is not None:
        power_density = (abs(cw_amplitude) ** 2) / (2.0 * eta0)
        return torch.full(
            (frequencies.numel(),),
            power_density * area,
            device=frequencies.device,
            dtype=frequencies.dtype,
        )

    return _broadband_plane_wave_incident_power(result, source_time, frequencies, eta0=eta0, area=area)


def _resolve_incident_power(
    result,
    incident_monitor: str,
    *,
    frequencies: torch.Tensor,
    reference_result=None,
    incident_power=None,
) -> torch.Tensor:
    if reference_result is not None and incident_power is not None:
        raise ValueError("Pass either reference_result or incident_power, not both.")
    if reference_result is not None:
        reference_frequencies = _result_frequencies(reference_result)
        _assert_matching_frequencies(reference_frequencies, frequencies, name="reference_result")
        return _monitor_flux(reference_result, incident_monitor, expected_frequencies=frequencies).to(
            device=frequencies.device,
            dtype=frequencies.dtype,
        )
    if incident_power is None:
        raise ValueError("incident power normalization is required. Pass reference_result or incident_power.")
    if isinstance(incident_power, str):
        token = incident_power.strip().lower()
        if token != "auto":
            raise ValueError("incident_power string values must be 'auto'.")
        return _plane_wave_incident_power(result, incident_monitor, frequencies)
    return _as_1d_tensor(
        incident_power,
        size=frequencies.numel(),
        name="incident_power",
        device=frequencies.device,
        dtype=frequencies.dtype,
    )


def _to_db(magnitude: torch.Tensor) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp_min(magnitude, _POWER_EPS))


def compute_s_parameters(
    result,
    incident_monitor: str,
    *,
    transmitted_monitor: str | None = None,
    reference_result=None,
    incident_power=None,
) -> dict:
    """
    Compute broadband S-parameter magnitudes from FDTD flux-monitor results.

    The returned complex arrays currently place the magnitude on the real axis.
    Phase recovery requires field-based port decomposition and is left as future work.
    """

    frequencies = _result_frequencies(result)
    incident_flux = _monitor_flux(result, incident_monitor, expected_frequencies=frequencies)
    if isinstance(incident_flux, torch.Tensor):
        frequencies = frequencies.to(device=incident_flux.device, dtype=incident_flux.dtype)
        incident_flux = incident_flux.to(device=frequencies.device, dtype=frequencies.dtype)
    incident_power_array = _resolve_incident_power(
        result,
        incident_monitor,
        frequencies=frequencies,
        reference_result=reference_result,
        incident_power=incident_power,
    ).to(device=frequencies.device, dtype=frequencies.dtype)
    if torch.any(incident_power_array <= _POWER_EPS):
        raise ValueError(
            "incident power must be strictly positive at every frequency. "
            "Check monitor orientation, source spectrum, or the supplied normalization."
        )

    reflected_power = torch.abs(incident_flux - incident_power_array)
    s11_mag = torch.sqrt(reflected_power / incident_power_array)
    complex_dtype = torch.complex64 if s11_mag.dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128

    output = {
        "frequencies": frequencies,
        "S11": s11_mag.to(dtype=complex_dtype),
        "S11_mag": s11_mag,
        "S11_db": _to_db(s11_mag),
        "S21": None,
        "S21_mag": None,
        "S21_db": None,
        "P_incident": incident_power_array,
        "P_reflected": reflected_power,
        "P_transmitted": None,
    }

    if transmitted_monitor is None:
        return output

    transmitted_flux = _monitor_flux(result, transmitted_monitor, expected_frequencies=frequencies).to(
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    transmitted_power = torch.abs(transmitted_flux)
    s21_mag = torch.sqrt(transmitted_power / incident_power_array)

    output["S21"] = s21_mag.to(dtype=complex_dtype)
    output["S21_mag"] = s21_mag
    output["S21_db"] = _to_db(s21_mag)
    output["P_transmitted"] = transmitted_power
    return output
