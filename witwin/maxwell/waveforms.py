"""Single source of truth for source-time waveform formulas.

This module holds the canonical Gaussian-pulse / Ricker / CW temporal formulas
in both scalar (math/NumPy) and torch form, plus the sampled-table waveform
characterization helpers (spectral-peak estimation and table interpolation).
``sources.py`` dataclasses, the compiled-dict evaluation path, and the FDTD
excitation layer all evaluate through these functions so the formulas are
written exactly once.

The torch entry point mirrors the native device kernel branch-for-branch
(including its reciprocal-width clamping), so it is kept as a separate thin
entry point over the same formulas rather than folded into the scalar path:
its coefficient derivation is intentionally bit-identical to the CUDA kernel,
not to the scalar host path.
"""

from __future__ import annotations

import math

import numpy as np
import torch

SOURCE_TIME_KIND_CW = 0
SOURCE_TIME_KIND_GAUSSIAN_PULSE = 1
SOURCE_TIME_KIND_RICKER_WAVELET = 2
SOURCE_TIME_KIND_CUSTOM = 3


def gaussian_sigma_t(fwidth: float) -> float:
    """Temporal standard deviation of a Gaussian pulse of spectral width ``fwidth``."""
    return 1.0 / (2.0 * math.pi * fwidth)


def cw_signal(t: float, *, frequency: float, amplitude: float, phase: float) -> float:
    """Continuous-wave signal ``A cos(2 pi f t + phi)``."""
    return amplitude * math.cos(2.0 * math.pi * frequency * float(t) + phase)


def gaussian_pulse_signal(
    t: float,
    *,
    frequency: float,
    fwidth: float,
    amplitude: float,
    phase: float,
    delay: float,
) -> float:
    """Gaussian-envelope pulse ``A exp(-tau^2 / (2 sigma^2)) cos(2 pi f tau + phi)``."""
    sigma_t = gaussian_sigma_t(fwidth)
    tau = float(t) - float(delay)
    envelope = math.exp(-0.5 * (tau / sigma_t) ** 2)
    return amplitude * envelope * math.cos(2.0 * math.pi * frequency * tau + phase)


def ricker_wavelet_signal(t: float, *, frequency: float, amplitude: float, delay: float) -> float:
    """Ricker (Mexican-hat) wavelet ``A (1 - 2 a^2) exp(-a^2)`` with ``a = pi f tau``."""
    tau = float(t) - float(delay)
    alpha = math.pi * frequency * tau
    alpha_sq = alpha * alpha
    return amplitude * (1.0 - 2.0 * alpha_sq) * math.exp(-alpha_sq)


def interpolate_waveform_table(t: float, times, amplitudes) -> float:
    """Sampled-table waveform value at ``t`` (linear interpolation, zero outside)."""
    return float(np.interp(float(t), times, amplitudes, left=0.0, right=0.0))


def table_characteristic_frequency(times: np.ndarray, amplitudes: np.ndarray) -> float:
    """Spectral-peak frequency of a sampled waveform table (FFT magnitude peak)."""
    spacing = np.diff(times)
    mean_dt = float(np.mean(spacing))
    if mean_dt <= 0.0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(amplitudes))
    freqs = np.fft.rfftfreq(times.size, d=mean_dt)
    if spectrum.size <= 1:
        return 0.0
    peak_index = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[peak_index])


def evaluate_time_shifted_waveform_torch(source_time, sample_time):
    """Per-cell source-time waveform for a time-shifted patch (torch tensors).

    Matches the ``evaluate_source_time`` device kernel branch-for-branch
    (kind 0 = CW, kind 1 = Gaussian pulse, anything else = Ricker) so a Bloch
    pulse injected through this Torch path is numerically identical to the
    non-Bloch kernel.
    """
    kind_code = int(source_time["kind_code"])
    amplitude = float(source_time["amplitude"])
    frequency = float(source_time["frequency"])
    phase = float(source_time.get("phase", 0.0))
    delay = float(source_time.get("delay", 0.0))
    two_pi = 2.0 * np.pi
    if kind_code == SOURCE_TIME_KIND_CW:
        return amplitude * torch.cos(two_pi * frequency * sample_time + phase)
    if kind_code == SOURCE_TIME_KIND_GAUSSIAN_PULSE:
        inv_sigma = max(two_pi * float(source_time["fwidth"]), 1.0e-30)
        tau = sample_time - delay
        envelope = torch.exp(-0.5 * (tau * inv_sigma) ** 2)
        return amplitude * envelope * torch.cos(two_pi * frequency * tau + phase)
    tau = sample_time - delay
    alpha = np.pi * frequency * tau
    alpha_sq = alpha * alpha
    return amplitude * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)
