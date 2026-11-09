"""Single source of truth for source-time waveform formulas.

This module holds the canonical Gaussian-pulse / Ricker / CW / broadband-pulse
temporal formulas in both scalar (math/NumPy) and torch form, plus the
sampled-table waveform representation (``WaveformTable``: linear or natural
cubic-spline interpolation, zero outside the sampled support) shared by the
scalar host path, the torch paths and the native time-shifted surface kernel.
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
SOURCE_TIME_KIND_BROADBAND_PULSE = 4

# Relative level at which an automatically delayed pulse is considered "off".
_PULSE_OFF_LEVEL = 1.0e-8


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


WAVEFORM_TABLE_INTERPOLATIONS = ("linear", "cubic")
# Relative spacing deviation below which a table is treated as uniformly sampled
# (O(1) segment lookup instead of a binary search on the device).
_UNIFORM_SPACING_RTOL = 1.0e-9


def natural_cubic_curvature(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Second derivatives of the natural cubic spline through ``(times, values)``."""
    from scipy.linalg import solve_banded

    count = int(times.size)
    curvature = np.zeros(count, dtype=np.float64)
    if count < 3:
        return curvature
    h = np.diff(times)
    slopes = np.diff(values) / h
    rhs = 6.0 * (slopes[1:] - slopes[:-1])
    bands = np.zeros((3, count - 2), dtype=np.float64)
    bands[0, 1:] = h[1:-1]
    bands[1, :] = 2.0 * (h[:-1] + h[1:])
    bands[2, :-1] = h[1:-1]
    curvature[1:-1] = solve_banded((1, 1), bands, rhs)
    return curvature


class WaveformTable:
    """Sampled waveform ``y(t)`` with linear or natural-cubic interpolation.

    The value is zero outside ``[times[0], times[-1]]``. Linear interpolation is
    the cubic-segment formula with zero curvature, so every evaluator (NumPy
    host path, torch paths and the native CUDA time-shifted kernel) shares one
    formula. Knots are stored in the normalized coordinate
    ``u = (t - times[0]) * inv_step`` with ``inv_step = (N - 1) / (t_N - t_0)``,
    so a uniformly sampled table has integer knots and an O(1) segment lookup.
    """

    __slots__ = ("interpolation", "start", "stop", "inv_step", "uniform", "knots", "values",
                 "curvature", "_tensor_cache")

    def __init__(self, times, values, *, interpolation: str = "linear"):
        interpolation = str(interpolation).lower()
        if interpolation not in WAVEFORM_TABLE_INTERPOLATIONS:
            raise ValueError("interpolation must be 'linear' or 'cubic'.")
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if times.ndim != 1 or times.shape != values.shape or times.size < 2:
            raise ValueError("A waveform table needs matching one-dimensional times and values (>= 2 samples).")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise ValueError("Waveform table times and values must be finite.")
        spacing = np.diff(times)
        if np.any(spacing <= 0.0):
            raise ValueError("Waveform table times must be strictly increasing (no duplicate samples).")
        span = float(times[-1] - times[0])
        count = int(times.size)
        inv_step = (count - 1) / span
        mean_step = span / (count - 1)
        uniform = bool(np.max(np.abs(spacing - mean_step)) <= _UNIFORM_SPACING_RTOL * mean_step)
        knots = np.arange(count, dtype=np.float64) if uniform else (times - times[0]) * inv_step
        if uniform:
            knots[-1] = float(count - 1)
        curvature_t = (
            natural_cubic_curvature(times, values) if interpolation == "cubic" else np.zeros(count)
        )
        self.interpolation = interpolation
        self.start = float(times[0])
        self.stop = float(times[-1])
        self.inv_step = float(inv_step)
        self.uniform = uniform
        self.knots = knots
        self.values = values.copy()
        # Curvature in normalized-coordinate units: d2y/du2 = d2y/dt2 / inv_step^2.
        self.curvature = curvature_t / (inv_step * inv_step)
        self._tensor_cache = {}

    @property
    def count(self) -> int:
        return int(self.values.size)

    def evaluate(self, t):
        """Interpolated value(s) at ``t`` (float64, zero outside the support)."""
        u = (np.asarray(t, dtype=np.float64) - self.start) * self.inv_step
        index = np.clip(np.searchsorted(self.knots, u, side="right") - 1, 0, self.count - 2)
        u0 = self.knots[index]
        h = self.knots[index + 1] - u0
        b = (u - u0) / h
        a = 1.0 - b
        value = (
            a * self.values[index]
            + b * self.values[index + 1]
            + ((a * a * a - a) * self.curvature[index] + (b * b * b - b) * self.curvature[index + 1]) * (h * h / 6.0)
        )
        inside = (u >= 0.0) & (u <= self.knots[-1])
        value = np.where(inside, value, 0.0)
        return float(value) if np.ndim(value) == 0 else value

    def tensors(self, device, dtype=torch.float32):
        """Cached ``(knots, values, curvature)`` tensors on ``device``."""
        key = (str(torch.device(device)), dtype)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = tuple(
                torch.as_tensor(array, dtype=dtype, device=device).contiguous()
                for array in (self.knots, self.values, self.curvature)
            )
            self._tensor_cache[key] = cached
        return cached

    def evaluate_torch(self, sample_time: torch.Tensor) -> torch.Tensor:
        """Torch twin of the native table kernel (same segment formula)."""
        knots, values, curvature = self.tensors(sample_time.device, sample_time.dtype)
        u = (sample_time - self.start) * self.inv_step
        last = self.count - 2
        if self.uniform:
            index = torch.floor(u).clamp(0, last).to(torch.long)
        else:
            index = (torch.searchsorted(knots, u.contiguous(), right=True) - 1).clamp(0, last)
        u0 = knots[index]
        h = knots[index + 1] - u0
        b = (u - u0) / h
        a = 1.0 - b
        value = (
            a * values[index]
            + b * values[index + 1]
            + ((a * a * a - a) * curvature[index] + (b * b * b - b) * curvature[index + 1]) * (h * h / 6.0)
        )
        inside = (u >= 0.0) & (u <= float(self.knots[-1]))
        return torch.where(inside, value, torch.zeros_like(value))

    def __deepcopy__(self, memo):
        # Immutable host arrays; device tensors are a cache. Sharing is safe.
        return self

    def __getstate__(self):
        return {name: getattr(self, name) for name in self.__slots__ if name != "_tensor_cache"}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self._tensor_cache = {}


CALLABLE_TIME_SHIFT_MESSAGE = (
    "Callable CustomSourceTime(fn) cannot drive a time-shifted surface source (PlaneWave, "
    "GaussianBeam, ModeSource, TFSF beams, CustomFieldSource): those cells sample the waveform "
    "at continuous per-cell delays over a run length that is only fixed after field "
    "initialization. Tabulate it explicitly, e.g. CustomSourceTime(times, [fn(t) for t in times], "
    "interpolation='cubic'), with a sample spacing well below the solver time step."
)


def require_waveform_table(source_time) -> WaveformTable:
    """Return the compiled sampled table of a custom source time or reject a callable."""
    table = source_time.get("table")
    if table is None:
        raise ValueError(CALLABLE_TIME_SHIFT_MESSAGE)
    return table


# ---------------------------------------------------------------------------
# Broadband pulse
# ---------------------------------------------------------------------------
#
# s(t) = A [ exp(-tau^2 / (2 sigma_t^2)) cos(2 pi f_c tau + phi)
#            - p exp(-tau^2 / (2 sigma_d^2)) ],      tau = t - delay
#
# The carrier sits at the band centre f_c. The Gaussian pedestal (relative peak
# p, width sigma_d = 1 / (2 pi nu_d), nu_d = min(sigma_f, f_min / 3)) carries
# exactly the DC content of the modulated Gaussian, so the waveform has zero
# mean and the fields decay, while its spectrum exp(-f^2 / (2 nu_d^2)) is
# below 1.2 % of p at f_min. sigma_f is the narrowest spectral width for which
# the zero-phase real-signal amplitude spectrum stays above
# ``minimum_amplitude`` times its in-band peak over the whole band.

_BROADBAND_BAND_SAMPLES = 1025
_BROADBAND_SCAN = np.logspace(-2.0, 2.0, 161)


def broadband_pedestal(center: float, fwidth: float, minimum_frequency: float, phase: float):
    """DC pedestal ``(relative_peak, pedestal_fwidth)`` of a broadband pulse."""
    pedestal_fwidth = min(float(fwidth), float(minimum_frequency) / 3.0)
    relative_peak = math.cos(phase) * math.exp(-0.5 * (center / fwidth) ** 2) * pedestal_fwidth / fwidth
    return relative_peak, pedestal_fwidth


def broadband_pulse_spectrum(frequencies, *, center, fwidth, amplitude, phase, delay, pedestal, pedestal_fwidth):
    """Exact spectrum ``S(f) = int s(t) exp(+i 2 pi f t) dt`` of the broadband pulse."""
    f = np.asarray(frequencies, dtype=np.float64)
    sigma_t = gaussian_sigma_t(fwidth)
    sigma_d = gaussian_sigma_t(pedestal_fwidth)
    carrier = 0.5 * math.sqrt(2.0 * math.pi) * sigma_t * (
        np.exp(-1j * phase) * np.exp(-0.5 * ((f - center) / fwidth) ** 2)
        + np.exp(1j * phase) * np.exp(-0.5 * ((f + center) / fwidth) ** 2)
    )
    dc = pedestal * math.sqrt(2.0 * math.pi) * sigma_d * np.exp(-0.5 * (f / pedestal_fwidth) ** 2)
    return amplitude * np.exp(2j * math.pi * f * delay) * (carrier - dc)


def _broadband_band_ratio(fwidth, minimum_frequency, maximum_frequency, band):
    center = 0.5 * (minimum_frequency + maximum_frequency)
    pedestal, pedestal_fwidth = broadband_pedestal(center, fwidth, minimum_frequency, 0.0)
    magnitude = np.abs(broadband_pulse_spectrum(
        band, center=center, fwidth=fwidth, amplitude=1.0, phase=0.0, delay=0.0,
        pedestal=pedestal, pedestal_fwidth=pedestal_fwidth,
    ))
    return float(magnitude.min() / magnitude.max())


def solve_broadband_fwidth(minimum_frequency: float, maximum_frequency: float, minimum_amplitude: float) -> float:
    """Narrowest Gaussian spectral width covering the band at ``minimum_amplitude``."""
    bandwidth = maximum_frequency - minimum_frequency
    band = np.linspace(minimum_frequency, maximum_frequency, _BROADBAND_BAND_SAMPLES)
    previous = None
    for width in bandwidth * _BROADBAND_SCAN:
        if _broadband_band_ratio(width, minimum_frequency, maximum_frequency, band) >= minimum_amplitude:
            break
        previous = width
    else:
        raise ValueError("BroadbandPulse could not satisfy minimum_amplitude over freq_range.")
    if previous is None:
        return float(width)
    low, high = float(previous), float(width)
    for _ in range(100):
        middle = math.sqrt(low * high)
        if _broadband_band_ratio(middle, minimum_frequency, maximum_frequency, band) >= minimum_amplitude:
            high = middle
        else:
            low = middle
        if high - low <= 1.0e-13 * high:
            break
    return float(high)


def broadband_auto_delay(fwidth: float, pedestal: float, pedestal_fwidth: float) -> float:
    """Half-duration after which both pulse terms are below ``_PULSE_OFF_LEVEL``."""
    half = 6.0 * gaussian_sigma_t(fwidth)
    magnitude = abs(float(pedestal))
    if magnitude > _PULSE_OFF_LEVEL:
        half = max(half, gaussian_sigma_t(pedestal_fwidth) * math.sqrt(2.0 * math.log(magnitude / _PULSE_OFF_LEVEL)))
    return half


def broadband_pulse_signal(
    t: float,
    *,
    frequency: float,
    fwidth: float,
    amplitude: float,
    phase: float,
    delay: float,
    pedestal: float,
    pedestal_fwidth: float,
) -> float:
    """Zero-mean broadband pulse (modulated Gaussian minus its DC pedestal)."""
    tau = float(t) - float(delay)
    carrier = math.exp(-0.5 * (tau / gaussian_sigma_t(fwidth)) ** 2) * math.cos(2.0 * math.pi * frequency * tau + phase)
    dc = pedestal * math.exp(-0.5 * (tau / gaussian_sigma_t(pedestal_fwidth)) ** 2)
    return amplitude * (carrier - dc)


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
    (kind 0 = CW, kind 1 = Gaussian pulse, 2 = Ricker, 3 = sampled table,
    4 = broadband pulse) so a Bloch pulse injected through this Torch path is
    numerically identical to the non-Bloch kernel.
    """
    kind_code = int(source_time["kind_code"])
    amplitude = float(source_time["amplitude"])
    if kind_code == SOURCE_TIME_KIND_CUSTOM:
        return amplitude * require_waveform_table(source_time).evaluate_torch(sample_time)
    frequency = float(source_time["frequency"])
    phase = float(source_time.get("phase", 0.0))
    delay = float(source_time.get("delay", 0.0))
    two_pi = 2.0 * np.pi
    if kind_code == SOURCE_TIME_KIND_CW:
        return amplitude * torch.cos(two_pi * frequency * sample_time + phase)
    if kind_code in (SOURCE_TIME_KIND_GAUSSIAN_PULSE, SOURCE_TIME_KIND_BROADBAND_PULSE):
        inv_sigma = max(two_pi * float(source_time["fwidth"]), 1.0e-30)
        tau = sample_time - delay
        envelope = torch.exp(-0.5 * (tau * inv_sigma) ** 2)
        if kind_code == SOURCE_TIME_KIND_GAUSSIAN_PULSE:
            # Keep the float32 association (amplitude * envelope) * cos of the
            # original Gaussian-pulse branch; regrouping the product changes the
            # injected samples at round-off and breaks bit identity.
            return amplitude * envelope * torch.cos(two_pi * frequency * tau + phase)
        signal = envelope * torch.cos(two_pi * frequency * tau + phase)
        if kind_code == SOURCE_TIME_KIND_BROADBAND_PULSE:
            pedestal_inv_sigma = two_pi * float(source_time["pedestal_fwidth"])
            signal = signal - float(source_time["pedestal"]) * torch.exp(-0.5 * (tau * pedestal_inv_sigma) ** 2)
        return amplitude * signal
    tau = sample_time - delay
    alpha = np.pi * frequency * tau
    alpha_sq = alpha * alpha
    return amplitude * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)
