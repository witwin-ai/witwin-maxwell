"""Host-only diagnostics of declared waveforms; never modify an excitation."""
from __future__ import annotations

import math
import warnings

import numpy as np

from ...sources import evaluate_source_time


def incident_line_band_edge(c: float, dt: float, ds: float, k_t: float = 0.0) -> float:
    """Return the incident line's upper band edge in Hz, clipped to Nyquist.

    An axial vacuum Yee line (``k_t=0``) uses ``asin(min(1, S))/(pi*dt)``,
    where ``S=c*dt/ds``. A fixed-k line with discrete transverse wavenumber
    magnitude ``k_t`` uses ``asin(min(1, S*sqrt(1+(k_t*ds/2)**2)))/(pi*dt)``.
    """
    if k_t == 0:
        return math.asin(min(1.0, c * dt / ds)) / (math.pi * dt)
    s = c * dt / ds
    return math.asin(min(1.0, s * math.sqrt(1 + (k_t * ds / 2)**2))) / (math.pi * dt)


def _spectrum_fractions(source_time, *, f_edge, dt, record_steps, sample_offset=1.0):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    if isinstance(record_steps, bool) or not isinstance(record_steps, (int, np.integer)) or record_steps < 1:
        raise ValueError("record_steps must be a positive integer.")
    if f_edge is not None and (not math.isfinite(f_edge) or not 0 <= f_edge <= 0.5 / dt):
        raise ValueError("f_edge must lie in the sampled Nyquist band, or be None for DC only.")
    # CW is an ongoing analytic carrier, not a rectangularly truncated pulse.
    kind = source_time.get("kind") if isinstance(source_time, dict) else getattr(source_time, "kind", None)
    if kind == "cw":
        return dict(edge_fraction=0.0, dc_fraction=0.0)
    steps = np.arange(record_steps, dtype=np.float64)
    times = (steps + 1) * dt if sample_offset == 1.0 else steps * dt + sample_offset * dt
    samples = np.fromiter((evaluate_source_time(source_time, float(t)) for t in times),
                          dtype=np.float64, count=record_steps)
    # Rescale before summation to make ratios independent of physical amplitude.
    amplitude = float(np.max(np.abs(samples)))
    if amplitude == 0:
        return dict(edge_fraction=0.0, dc_fraction=0.0)
    samples /= amplitude
    dc = abs(float(samples.sum()))
    edge = 0.0 if f_edge is None else float(abs(np.dot(samples, np.exp(-2j * np.pi * f_edge * times))))
    peak = max(float(np.abs(np.fft.rfft(samples, n=8 * record_steps)).max()), edge, dc)
    return dict(edge_fraction=edge / peak, dc_fraction=dc / peak)


def _warn_spectrum(fractions, *, f_edge, line_name, record_steps, pole_period_modulus):
    if not math.isfinite(pole_period_modulus) or pole_period_modulus < 0:
        raise ValueError("pole_period_modulus must be nonnegative and finite.")
    threshold = max(1e-6, np.finfo(np.float32).eps * math.sqrt(record_steps))
    edge = fractions["edge_fraction"]
    # Logarithms avoid overflow for a growing pole and underflow for a damped one.
    if edge > 0 and pole_period_modulus > 0 and (
        math.log(edge) + record_steps * math.log(pole_period_modulus) > math.log(threshold)
    ):
        warnings.warn(
            f"Incident line {line_name!r}: f_edge={f_edge:.9g} Hz, edge_fraction={edge:.9g}. "
            "A slowly decaying numerical mode near the axial band edge of the 1D incident "
            "line may feed a late-time tail; use a narrower-band waveform.",
            RuntimeWarning, stacklevel=3,
        )
    dc = fractions["dc_fraction"]
    if dc > threshold:
        warnings.warn(
            f"Source/incident line {line_name!r}: dc_fraction={dc:.9g}. The waveform has DC "
            "content, which leaves a static charge/field; late-time fields carry a static "
            "offset and time-domain energy shutoff may never trigger. Measured on the "
            "benchmark scenes, this biases source-normalized monitor spectra by about "
            "dc_fraction x (0.6-3) (relative) for current, mode and beam sources and "
            "about dc_fraction x (0.03-0.16) for soft plane-wave sources. Use a zero-DC "
            "waveform (e.g. RickerWavelet, or GaussianPulse with phase=pi/2).",
            RuntimeWarning, stacklevel=3,
        )


def check_incident_line_spectrum(
    source_time, *, f_edge: float | None, dt: float, ds: float, line_name: str,
    record_steps: int, pole_period_modulus: float = 1.0, sample_offset: float = 1.0,
) -> dict[str, float]:
    """Return edge/DC magnitude fractions and warn, without changing the waveform.

    The spectrum is the DTFT of the per-step declared entry-E samples
    ``s[n] = source_time((n+sample_offset)*dt)``, ``n=0..record_steps-1``, where
    ``sample_offset`` is in steps (default 1.0), with a rectangular record
    window (no taper, mean subtraction, filtering or analytic envelope).
    DC and ``f_edge`` are evaluated exactly, including off-bin edges. The
    normalization maximum is taken over the entire occupied sampled band:
    all bins of an eightfold zero-padded real FFT from DC through Nyquist,
    plus the exact edge. Thus the peak is a sampled-frequency estimate of the
    DTFT maximum, with bin spacing ``1/(8*record_steps*dt)``. Zero records give
    zero fractions. CW carriers are exempt from finite-pulse diagnostics.

    The axial vacuum Yee edge is ``asin(S)/(pi*dt)``, ``S=c*dt/ds``. Callers
    supply their own edge: a fixed-k line, for example, uses
    ``asin(S*sqrt(1+(k_t*ds/2)**2))/(pi*dt)`` and
    ``pole_period_modulus=1.0``. ``ds`` records the line spacing; it does not
    determine or override the supplied edge. ``f_edge=None`` requests DC only.

    Each call emits at most one RuntimeWarning per hazard. The edge test is
    ``edge_fraction * pole_period_modulus**record_steps > threshold`` and the
    DC test is ``dc_fraction > threshold``, where
    ``threshold=max(1e-6, fp32_floor)`` and
    ``fp32_floor=eps32*sqrt(record_steps)``. Preparation caches unchanged
    source/timing checks on the Simulation, including across re-prepare;
    a changed waveform, timing, record length or line spacing is checked anew.
    J/M preparation uses the corresponding source reference sampling clock
    (half steps for electric currents, integer steps for shared J/M references).
    All diagnostic arithmetic and waveform evaluation run on the CPU.
    """
    if not math.isfinite(ds) or ds <= 0:
        raise ValueError("ds must be positive and finite.")
    fractions = _spectrum_fractions(source_time, f_edge=f_edge, dt=dt,
                                    record_steps=record_steps, sample_offset=sample_offset)
    _warn_spectrum(fractions, f_edge=f_edge, line_name=line_name,
                   record_steps=record_steps, pole_period_modulus=pole_period_modulus)
    return fractions


def check_prepared_source_spectra(solver, *, record_steps, checked):
    """Check each compiled source once, using actual prepared line geometry."""
    from ..runtime.timeplan import source_spectrum_key, source_time_offset

    state = getattr(solver, "_tfsf_state", None) or {}
    if getattr(solver, "shards", ()):
        state = getattr(solver.shards[0].solver, "_tfsf_state", None) or {}
    aux = state.get("auxiliary_grid")
    for index, source in enumerate(solver._compiled_sources):
        waveform = source["source_time"]
        if waveform["kind"] == "cw":
            continue
        is_line = aux is not None and source.get("injection", {}).get("kind") == "tfsf"
        ds = float(aux.ds) if is_line else None
        dt = float(solver.dt)
        k_t = getattr(aux, "band_edge_transverse_wavenumber", 0.0) if is_line else None
        f_edge = incident_line_band_edge(solver.c, dt, ds, k_t) if is_line else None
        offset = getattr(aux, "drive_sample_offset", 1.0) if is_line else source_time_offset(source)
        key = (index, source.get("name"), source_spectrum_key(waveform), waveform["amplitude"],
               dt, ds, record_steps, k_t, offset)
        if key in checked:
            continue
        label = source.get("name") or f"{source['kind']}[{index}]"
        if is_line:
            check_incident_line_spectrum(waveform, f_edge=f_edge, dt=dt, ds=ds,
                                         line_name=label, record_steps=record_steps, sample_offset=offset)
        else:
            fractions = _spectrum_fractions(waveform, f_edge=None, dt=dt,
                                            record_steps=record_steps, sample_offset=offset)
            _warn_spectrum(fractions, f_edge=None, line_name=label,
                           record_steps=record_steps, pole_period_modulus=1.0)
        checked.add(key)
