from __future__ import annotations

from ._fixed_inputs import require_fixed_input

from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from .waveforms import (
    SOURCE_TIME_KIND_BROADBAND_PULSE,
    SOURCE_TIME_KIND_CW,
    SOURCE_TIME_KIND_GAUSSIAN_PULSE,
    SOURCE_TIME_KIND_RICKER_WAVELET,
    SOURCE_TIME_KIND_CUSTOM,
    WaveformTable,
    broadband_auto_delay,
    broadband_pedestal,
    broadband_pulse_signal,
    broadband_pulse_spectrum,
    cw_signal,
    gaussian_pulse_signal,
    gaussian_sigma_t,
    ricker_wavelet_signal,
    solve_broadband_fwidth,
    table_characteristic_frequency,
)

POINT_DIPOLE_REFERENCE_WIDTH = 0.02


def _require_length3(name, value) -> tuple[float, float, float]:
    require_fixed_input(value, name="Fixed source/observation " + name)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return tuple(float(component) for component in value)


def _normalize_vector(name, value) -> tuple[float, float, float]:
    vector = _require_length3(name, value)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return tuple(component / norm for component in vector)


def _require_beam_waist_pair(value) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("beam_waist must contain exactly two values (w0_u, w0_v).")
    waist = tuple(float(component) for component in value)
    if any(component <= 0.0 for component in waist):
        raise ValueError("beam_waist components must be > 0.")
    return waist


def _normalize_axis(axis) -> str | None:
    if axis is None:
        return None
    normalized = str(axis).lower()
    if normalized not in {"x", "y", "z"}:
        raise ValueError("injection_axis must be 'x', 'y', 'z', or None.")
    return normalized


def _normalize_point_dipole_profile(value) -> str:
    normalized = str(value).lower()
    if normalized not in {"gaussian", "ideal"}:
        raise ValueError("profile must be 'gaussian' or 'ideal'.")
    return normalized


def _normalize_mode_direction(value) -> str:
    normalized = str(value).strip()
    if normalized not in {"+", "-"}:
        raise ValueError("direction must be '+' or '-'.")
    return normalized


def _require_mode_source_size(value) -> tuple[float, float, float]:
    size = _require_length3("size", value)
    if any(component < 0.0 for component in size):
        raise ValueError("size components must be >= 0.")
    return size


def _resolve_mode_source_normal_axis(size) -> str:
    zero_axes = [axis for axis, component in zip("xyz", size) if abs(float(component)) <= 1e-12]
    if len(zero_axes) != 1:
        raise ValueError("ModeSource size must contain exactly one zero component to define the source plane.")
    return zero_axes[0]


def _resolve_mode_source_polarization_axis(normal_axis: str, size, value) -> str:
    tangential_axes = tuple(axis for axis in "xyz" if axis != normal_axis)
    if value is None or (isinstance(value, str) and str(value).lower() == "auto"):
        tangential_sizes = {axis: float(size["xyz".index(axis)]) for axis in tangential_axes}
        return max(tangential_axes, key=lambda axis: (tangential_sizes[axis], axis))

    if isinstance(value, str):
        token = value.lower()
        if token in {"ex", "ey", "ez"}:
            axis = token[-1]
        elif token in {"x", "y", "z"}:
            axis = token
        else:
            raise ValueError("polarization must be 'auto', Ex/Ey/Ez, x/y/z, or an axis-aligned vector.")
    else:
        vector = polarization_vector(value)
        active = [axis for axis, component in zip("xyz", vector) if abs(float(component)) > 1e-9]
        if len(active) != 1:
            raise ValueError("ModeSource polarization vector must be axis-aligned.")
        axis = active[0]

    if axis == normal_axis:
        raise ValueError("ModeSource polarization must be tangential to the source plane.")
    return axis


def _normalize_bend(bend_radius, bend_axis, normal_axis: str) -> tuple[float | None, str | None]:
    """Validate the cylindrical-bend parameters of a mode plane.

    A bent (curved-waveguide) port keeps its cross-section plane axis-aligned but
    solves the guided mode with the Heiblum-Harris conformal transform, which maps
    a bend of signed radius ``bend_radius`` into an equivalent straight guide whose
    index is graded along the in-plane radial direction. ``bend_axis`` is the
    cylinder (rotation) axis: it must be one of the two axes tangential to the mode
    plane, and the remaining tangential axis becomes the radial direction. Returns
    ``(None, None)`` for a straight port and ``(radius, axis)`` for a bent one.
    """
    if bend_radius is None:
        if bend_axis is not None:
            raise ValueError("bend_axis requires bend_radius to be set.")
        return None, None
    radius = float(bend_radius)
    if radius == 0.0 or not math.isfinite(radius):
        raise ValueError("bend_radius must be a finite non-zero length (a straight port omits it).")
    if bend_axis is None:
        raise ValueError("bend_radius requires bend_axis, the cylinder (rotation) axis of the bend.")
    axis = str(bend_axis).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("bend_axis must be 'x', 'y', or 'z'.")
    if axis == normal_axis:
        raise ValueError(
            "bend_axis must be tangential to the mode plane (perpendicular to the propagation axis), "
            f"but it equals the propagation axis {normal_axis!r}."
        )
    return radius, axis


def polarization_vector(value) -> tuple[float, float, float]:
    if isinstance(value, str):
        token = value.lower()
        mapping = {
            "ex": (1.0, 0.0, 0.0),
            "ey": (0.0, 1.0, 0.0),
            "ez": (0.0, 0.0, 1.0),
        }
        if token not in mapping:
            raise ValueError("polarization must be Ex, Ey, Ez, or a length-3 vector.")
        return mapping[token]
    return _normalize_vector("polarization", value)


def _validate_transverse_polarization(
    direction,
    polarization,
    *,
    direction_name="direction",
) -> tuple[float, float, float]:
    direction = _normalize_vector(direction_name, direction)
    polarization = polarization_vector(polarization)
    dot = sum(a * b for a, b in zip(direction, polarization))
    if abs(dot) > 1e-6:
        raise ValueError("polarization must be orthogonal to direction.")
    return direction, polarization


def _validate_frequency(name, value):
    require_fixed_input(value, name=f"source_time.{name}")
    frequency = float(value)
    if frequency <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return frequency


def _validate_amplitude(value):
    require_fixed_input(value, name="source_time.amplitude")
    return float(value)


@dataclass(frozen=True)
class CW:
    frequency: float
    amplitude: float = 1.0
    phase: float = 0.0
    kind: str = "cw"

    def __post_init__(self):
        object.__setattr__(self, "frequency", _validate_frequency("frequency", self.frequency))
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        require_fixed_input(self.phase, name="source_time.phase")
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "kind", "cw")

    def evaluate(self, t: float) -> float:
        return cw_signal(t, frequency=self.frequency, amplitude=self.amplitude, phase=self.phase)

    @property
    def characteristic_frequency(self) -> float:
        return self.frequency

    @property
    def delay(self) -> float:
        return 0.0

    @property
    def settling_time(self) -> float:
        return 0.0


@dataclass(frozen=True)
class GaussianPulse:
    frequency: float
    fwidth: float
    amplitude: float = 1.0
    phase: float = 0.0
    delay: float | None = None
    kind: str = "gaussian_pulse"

    def __post_init__(self):
        frequency = _validate_frequency("frequency", self.frequency)
        fwidth = _validate_frequency("fwidth", self.fwidth)
        sigma_t = gaussian_sigma_t(fwidth)
        require_fixed_input(self.delay, name="source_time.delay")
        delay = 6.0 * sigma_t if self.delay is None else float(self.delay)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "fwidth", fwidth)
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        require_fixed_input(self.phase, name="source_time.phase")
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "kind", "gaussian_pulse")

    @property
    def sigma_t(self) -> float:
        return gaussian_sigma_t(self.fwidth)

    def evaluate(self, t: float) -> float:
        return gaussian_pulse_signal(
            t,
            frequency=self.frequency,
            fwidth=self.fwidth,
            amplitude=self.amplitude,
            phase=self.phase,
            delay=self.delay,
        )

    @property
    def characteristic_frequency(self) -> float:
        return max(self.frequency, self.frequency + 3.0 * self.fwidth)

    @property
    def settling_time(self) -> float:
        return float(self.delay) + 6.0 * self.sigma_t


@dataclass(frozen=True)
class RickerWavelet:
    frequency: float
    amplitude: float = 1.0
    delay: float | None = None
    kind: str = "ricker_wavelet"

    def __post_init__(self):
        frequency = _validate_frequency("frequency", self.frequency)
        require_fixed_input(self.delay, name="source_time.delay")
        delay = 4.0 / (math.pi * frequency) if self.delay is None else float(self.delay)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "kind", "ricker_wavelet")

    def evaluate(self, t: float) -> float:
        return ricker_wavelet_signal(t, frequency=self.frequency, amplitude=self.amplitude, delay=self.delay)

    @property
    def phase(self) -> float:
        return 0.0

    @property
    def fwidth(self) -> float:
        return 0.0

    @property
    def characteristic_frequency(self) -> float:
        return 3.0 * self.frequency

    @property
    def settling_time(self) -> float:
        return float(self.delay) + 4.0 / max(self.frequency, 1e-30)


@dataclass(frozen=True)
class CustomSourceTime:
    """Arbitrary temporal waveform.

    Construct either from a sampled table ``CustomSourceTime(times, amplitudes)``
    or from a callable ``CustomSourceTime(fn)``. A table is interpolated
    linearly (default) or with a natural cubic spline (``interpolation="cubic"``)
    and is zero outside ``[times[0], times[-1]]``; samples may be non-uniform.
    Point and volume current sources evaluate the waveform on the solver time
    grid. Time-shifted surface sources (PlaneWave, GaussianBeam, ModeSource,
    TFSF beams, CustomFieldSource) evaluate the same table natively on the GPU
    at every cell's delayed time, so they require a table; a callable is
    rejected there because its sampling window and resolution are unknown.

    ``frequency`` is the reference (carrier) frequency at which surface sources
    tune their dispersion-matched launch, evaluate beam profiles and solve
    guided modes. ``characteristic_frequency`` bounds the automatic time step
    and mesh; it defaults to the table's spectral peak, and ``frequency``
    defaults to ``characteristic_frequency``.
    """

    times: tuple[float, ...] | None = None
    amplitudes: tuple[float, ...] | None = None
    amplitude: float = 1.0
    interpolation: str = "linear"
    kind: str = "custom"

    def __init__(
        self,
        times=None,
        amplitudes=None,
        *,
        amplitude=1.0,
        characteristic_frequency=None,
        frequency=None,
        interpolation="linear",
    ):
        require_fixed_input(
            (times, amplitudes, characteristic_frequency, frequency), name="CustomSourceTime samples/frequency"
        )
        fn = None
        table_times = None
        table_amplitudes = None
        if callable(times):
            if amplitudes is not None:
                raise ValueError("CustomSourceTime(fn) does not accept a second positional argument.")
            if characteristic_frequency is None:
                raise ValueError("CustomSourceTime(fn) requires characteristic_frequency.")
            fn = times
            char_freq = _validate_frequency("characteristic_frequency", characteristic_frequency)
            settling_time = 0.0
            delay = 0.0
            table = None
        else:
            if times is None or amplitudes is None:
                raise ValueError("CustomSourceTime requires either fn or (times, amplitudes).")
            table_times = np.asarray(times, dtype=np.float64)
            table_amplitudes = np.asarray(amplitudes, dtype=np.float64)
            if table_times.ndim != 1 or table_amplitudes.ndim != 1:
                raise ValueError("times and amplitudes must be one-dimensional.")
            if table_times.shape != table_amplitudes.shape:
                raise ValueError("times and amplitudes must have the same length.")
            if table_times.size < 2:
                raise ValueError("times and amplitudes must contain at least two samples.")
            order = np.argsort(table_times, kind="stable")
            table_times = np.ascontiguousarray(table_times[order])
            table_amplitudes = np.ascontiguousarray(table_amplitudes[order])
            table = WaveformTable(table_times, table_amplitudes, interpolation=interpolation)
            if characteristic_frequency is None:
                char_freq = table_characteristic_frequency(table_times, table_amplitudes)
            else:
                char_freq = float(characteristic_frequency)
            delay = float(table_times[0])
            settling_time = float(table_times[-1])

        object.__setattr__(self, "times", None if table_times is None else tuple(float(v) for v in table_times))
        object.__setattr__(
            self, "amplitudes", None if table_amplitudes is None else tuple(float(v) for v in table_amplitudes)
        )
        object.__setattr__(self, "amplitude", _validate_amplitude(amplitude))
        object.__setattr__(self, "interpolation", "linear" if table is None else table.interpolation)
        object.__setattr__(self, "kind", "custom")
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_characteristic_frequency", char_freq)
        reference_frequency = char_freq if frequency is None else _validate_frequency("frequency", frequency)
        object.__setattr__(self, "_frequency", float(reference_frequency))
        object.__setattr__(self, "_delay", delay)
        object.__setattr__(self, "_settling_time", settling_time)

    @property
    def fn(self):
        return self._fn

    @property
    def table(self) -> WaveformTable | None:
        return self._table

    def evaluate(self, t: float) -> float:
        return float(self.amplitude) * self.evaluate_shape(t)

    def evaluate_shape(self, t: float) -> float:
        """Waveform without the ``amplitude`` factor."""
        if self._fn is not None:
            return float(self._fn(float(t)))
        return float(self._table.evaluate(float(t)))

    @property
    def frequency(self) -> float:
        return float(self._frequency)

    @property
    def characteristic_frequency(self) -> float:
        return float(self._characteristic_frequency)

    @property
    def phase(self) -> float:
        return 0.0

    @property
    def fwidth(self) -> float:
        return 0.0

    @property
    def delay(self) -> float:
        return float(self._delay)

    @property
    def settling_time(self) -> float:
        return float(self._settling_time)


@dataclass(frozen=True)
class BroadbandPulse:
    """Zero-mean pulse with significant energy over a whole frequency range.

    The waveform is a Gaussian-envelope carrier at the band centre ``f_c`` minus
    a slow Gaussian pedestal that removes its DC content exactly::

        s(t) = A [exp(-tau^2 / (2 sigma_t^2)) cos(2 pi f_c tau + phase)
                  - p exp(-tau^2 / (2 sigma_d^2))],      tau = t - delay

    ``fwidth`` (``sigma_f = 1 / (2 pi sigma_t)``) is the narrowest spectral
    width for which the zero-phase amplitude spectrum stays at or above
    ``minimum_amplitude`` times its in-band peak everywhere in ``freq_range``;
    narrower spectra waste less energy out of band and permit coarser grids.
    The pedestal width ``pedestal_fwidth = min(sigma_f, f_min / 3)`` keeps its
    spectrum below 1.2 % of its DC level at ``f_min``. ``offset`` shifts the
    automatic delay in units of ``1 / (2 pi (f_max - f_min))``. The waveform is
    analytic and evaluated natively by every FDTD source path.
    """

    freq_range: tuple[float, float]
    minimum_amplitude: float = 0.3
    amplitude: float = 1.0
    phase: float = 0.0
    offset: float = 0.0
    kind: str = "broadband_pulse"

    def __post_init__(self):
        require_fixed_input(self.freq_range, name="source_time.freq_range")
        if len(self.freq_range) != 2:
            raise ValueError("freq_range must contain exactly two frequencies.")
        minimum = _validate_frequency("freq_range[0]", self.freq_range[0])
        maximum = _validate_frequency("freq_range[1]", self.freq_range[1])
        if maximum <= minimum:
            raise ValueError("freq_range[1] must be greater than freq_range[0].")
        require_fixed_input(self.minimum_amplitude, name="source_time.minimum_amplitude")
        minimum_amplitude = float(self.minimum_amplitude)
        if not 0.05 < minimum_amplitude < 0.5:
            raise ValueError("minimum_amplitude must satisfy 0.05 < minimum_amplitude < 0.5.")
        require_fixed_input(self.phase, name="source_time.phase")
        require_fixed_input(self.offset, name="source_time.offset")
        phase = float(self.phase)
        offset = float(self.offset)
        center = 0.5 * (minimum + maximum)
        fwidth = solve_broadband_fwidth(minimum, maximum, minimum_amplitude)
        # The automatic delay uses the zero-phase pedestal (its largest
        # magnitude) so the delay, like the spectral width, is phase independent.
        auto_pedestal, pedestal_fwidth = broadband_pedestal(center, fwidth, minimum, 0.0)
        half_duration = broadband_auto_delay(fwidth, auto_pedestal, pedestal_fwidth)
        pedestal, _ = broadband_pedestal(center, fwidth, minimum, phase)
        object.__setattr__(self, "freq_range", (minimum, maximum))
        object.__setattr__(self, "minimum_amplitude", minimum_amplitude)
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "kind", "broadband_pulse")
        object.__setattr__(self, "_fwidth", fwidth)
        object.__setattr__(self, "_pedestal", pedestal)
        object.__setattr__(self, "_pedestal_fwidth", pedestal_fwidth)
        object.__setattr__(self, "_half_duration", half_duration)
        object.__setattr__(self, "_delay", half_duration + offset / (2.0 * math.pi * (maximum - minimum)))

    @property
    def frequency(self) -> float:
        """Carrier (band-centre) frequency."""
        return 0.5 * (self.freq_range[0] + self.freq_range[1])

    @property
    def fwidth(self) -> float:
        return float(self._fwidth)

    @property
    def sigma_t(self) -> float:
        return gaussian_sigma_t(self._fwidth)

    @property
    def pedestal(self) -> float:
        """Relative peak of the DC-removing Gaussian pedestal."""
        return float(self._pedestal)

    @property
    def pedestal_fwidth(self) -> float:
        return float(self._pedestal_fwidth)

    @property
    def delay(self) -> float:
        return float(self._delay)

    def evaluate(self, t: float) -> float:
        return broadband_pulse_signal(
            t,
            frequency=self.frequency,
            fwidth=self.fwidth,
            amplitude=self.amplitude,
            phase=self.phase,
            delay=self.delay,
            pedestal=self.pedestal,
            pedestal_fwidth=self.pedestal_fwidth,
        )

    def spectrum(self, frequencies):
        """Exact spectrum ``int s(t) exp(+i 2 pi f t) dt`` (exp(-i omega t) phasors)."""
        return broadband_pulse_spectrum(
            frequencies,
            center=self.frequency,
            fwidth=self.fwidth,
            amplitude=self.amplitude,
            phase=self.phase,
            delay=self.delay,
            pedestal=self.pedestal,
            pedestal_fwidth=self.pedestal_fwidth,
        )

    @property
    def characteristic_frequency(self) -> float:
        return max(self.freq_range[1], self.frequency + 3.0 * self.fwidth)

    @property
    def settling_time(self) -> float:
        return float(self._delay) + float(self._half_duration)


# 10%-to-90% rise time of an error-function edge in units of its Gaussian sigma.
_RISE_TIME_10_90_FACTOR = 2.5631031310892006
_BASEBAND_MAX_SAMPLES = 1_000_000


def _baseband_positive(label, value):
    require_fixed_input(value, name=f"source_time.{label}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be positive and finite.")
    return resolved


def _baseband_table(stop, spacing):
    count = int(math.ceil(stop / spacing)) + 1
    if count > _BASEBAND_MAX_SAMPLES:
        raise ValueError(
            f"Baseband waveform needs {count} table samples; shorten the pulse or lengthen its edges."
        )
    return np.linspace(0.0, stop, count)


class BasebandGaussianPulse(CustomSourceTime):
    """Unmodulated, DC-containing Gaussian pulse for transient (TDR-like) RF drives.

    ``s(t) = amplitude * exp(-(t - offset * twidth)^2 / (2 twidth^2))``, sampled
    on a uniform natural-cubic table (``twidth / 32`` spacing) from ``t = 0`` to
    ``10 twidth`` after the peak. Its spectrum is centred at DC with standard
    deviation ``1 / (2 pi twidth)``; ``characteristic_frequency`` is three of
    those widths. Being a sampled :class:`CustomSourceTime`, it drives every
    FDTD source path that accepts a waveform table, including lumped ports.
    """

    def __init__(self, twidth, *, offset=5.0, amplitude=1.0):
        width = _baseband_positive("twidth", twidth)
        require_fixed_input(offset, name="source_time.offset")
        resolved_offset = float(offset)
        if not math.isfinite(resolved_offset) or resolved_offset < 2.5:
            raise ValueError("BasebandGaussianPulse offset must be >= 2.5 widths.")
        center = resolved_offset * width
        times = _baseband_table(center + 10.0 * width, width / 32.0)
        values = np.exp(-0.5 * ((times - center) / width) ** 2)
        bandwidth = 1.0 / (2.0 * math.pi * width)
        super().__init__(times, values, amplitude=amplitude, characteristic_frequency=3.0 * bandwidth,
                         interpolation="cubic")
        object.__setattr__(self, "twidth", width)
        object.__setattr__(self, "offset", resolved_offset)

    @property
    def fwidth_equiv(self) -> float:
        """Spectral standard deviation ``1 / (2 pi twidth)`` in Hz."""
        return 1.0 / (2.0 * math.pi * self.twidth)

    @property
    def t_center(self) -> float:
        return self.offset * self.twidth


class BasebandRectangularPulse(CustomSourceTime):
    """Smoothed, DC-containing rectangular pulse with error-function edges.

    ``s(t) = amplitude / 2 * [erf((t - t_start) / (sqrt(2) sigma)) - erf((t - t_stop) / (sqrt(2) sigma))]``
    with ``sigma = rise_time / 2.5631`` (10%-to-90% edges), ``t_start =
    offset * rise_time`` and ``t_stop = t_start + twidth``. The waveform is
    tabulated with natural-cubic interpolation at ``sigma / 16`` spacing up to
    ``10 sigma`` after the falling edge.
    """

    def __init__(self, rise_time, twidth, *, offset=5.0, amplitude=1.0):
        from scipy.special import erf

        rise = _baseband_positive("rise_time", rise_time)
        width = _baseband_positive("twidth", twidth)
        require_fixed_input(offset, name="source_time.offset")
        resolved_offset = float(offset)
        if not math.isfinite(resolved_offset) or resolved_offset < 2.5:
            raise ValueError("BasebandRectangularPulse offset must be >= 2.5 rise times.")
        sigma = rise / _RISE_TIME_10_90_FACTOR
        start = resolved_offset * rise
        stop = start + width
        times = _baseband_table(stop + 10.0 * sigma, sigma / 16.0)
        scale = math.sqrt(2.0) * sigma
        values = 0.5 * (erf((times - start) / scale) - erf((times - stop) / scale))
        super().__init__(times, values, amplitude=amplitude,
                         characteristic_frequency=3.0 / (2.0 * math.pi * sigma), interpolation="cubic")
        object.__setattr__(self, "rise_time", rise)
        object.__setattr__(self, "twidth", width)
        object.__setattr__(self, "offset", resolved_offset)

    @property
    def t_start(self) -> float:
        return self.offset * self.rise_time

    @property
    def t_stop(self) -> float:
        return self.t_start + self.twidth

SourceTime = CW | GaussianPulse | RickerWavelet | CustomSourceTime | BroadbandPulse


@dataclass(frozen=True)
class TFSF:
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    mode: str = "box"
    axis: str | None = None
    axis_bounds: tuple[float, float] | None = None
    kind: str = "tfsf"

    def __init__(self, bounds):
        if len(bounds) != 3:
            raise ValueError("bounds must contain x, y, z ranges.")
        normalized = []
        for axis_bounds in bounds:
            if len(axis_bounds) != 2:
                raise ValueError("Each TFSF axis bound must contain exactly two values.")
            start, end = float(axis_bounds[0]), float(axis_bounds[1])
            if end <= start:
                raise ValueError("Each TFSF axis bound must have end > start.")
            normalized.append((start, end))
        object.__setattr__(self, "bounds", tuple(normalized))
        object.__setattr__(self, "mode", "box")
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "axis_bounds", None)
        object.__setattr__(self, "kind", "tfsf")

    @classmethod
    def slab(cls, *, axis, bounds) -> "TFSF":
        axis_name = str(axis).lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError("TFSF slab axis must be 'x', 'y', or 'z'.")
        if len(bounds) != 2:
            raise ValueError("TFSF slab bounds must contain exactly two values.")
        start, end = float(bounds[0]), float(bounds[1])
        if end <= start:
            raise ValueError("TFSF slab bounds must have end > start.")

        source = cls.__new__(cls)
        object.__setattr__(source, "bounds", None)
        object.__setattr__(source, "mode", "slab")
        object.__setattr__(source, "axis", axis_name)
        object.__setattr__(source, "axis_bounds", (start, end))
        object.__setattr__(source, "kind", "tfsf")
        return source


Injection = str | TFSF


def resolve_source_time(source_time: SourceTime | None, *, default_frequency: float) -> SourceTime:
    if source_time is None:
        return CW(frequency=default_frequency)
    if not isinstance(source_time, (CW, GaussianPulse, RickerWavelet, CustomSourceTime, BroadbandPulse)):
        raise TypeError(
            "source_time must be CW, GaussianPulse, RickerWavelet, CustomSourceTime, BroadbandPulse, or None."
        )
    return source_time


def compile_source_time(source_time: SourceTime | None, *, default_frequency: float) -> dict[str, object]:
    resolved = resolve_source_time(source_time, default_frequency=default_frequency)
    if isinstance(resolved, CW):
        return {
            "kind": "cw",
            "kind_code": SOURCE_TIME_KIND_CW,
            "frequency": float(resolved.frequency),
            "fwidth": 0.0,
            "amplitude": float(resolved.amplitude),
            "phase": float(resolved.phase),
            "delay": 0.0,
            "characteristic_frequency": float(resolved.characteristic_frequency),
            "settling_time": 0.0,
        }
    if isinstance(resolved, GaussianPulse):
        return {
            "kind": "gaussian_pulse",
            "kind_code": SOURCE_TIME_KIND_GAUSSIAN_PULSE,
            "frequency": float(resolved.frequency),
            "fwidth": float(resolved.fwidth),
            "amplitude": float(resolved.amplitude),
            "phase": float(resolved.phase),
            "delay": float(resolved.delay),
            "characteristic_frequency": float(resolved.characteristic_frequency),
            "settling_time": float(resolved.settling_time),
        }
    if isinstance(resolved, CustomSourceTime):
        return {
            "kind": "custom",
            "kind_code": SOURCE_TIME_KIND_CUSTOM,
            "frequency": float(resolved.frequency),
            "fwidth": 0.0,
            "amplitude": float(resolved.amplitude),
            "phase": 0.0,
            "delay": float(resolved.delay),
            "characteristic_frequency": float(resolved.characteristic_frequency),
            "settling_time": float(resolved.settling_time),
            "fn": resolved.fn,
            "times": resolved.times,
            "amplitudes": resolved.amplitudes,
            "interpolation": resolved.interpolation,
            "table": resolved.table,
        }
    if isinstance(resolved, BroadbandPulse):
        return {
            "kind": "broadband_pulse",
            "kind_code": SOURCE_TIME_KIND_BROADBAND_PULSE,
            "frequency": float(resolved.frequency),
            "fwidth": float(resolved.fwidth),
            "amplitude": float(resolved.amplitude),
            "phase": float(resolved.phase),
            "delay": float(resolved.delay),
            "pedestal": float(resolved.pedestal),
            "pedestal_fwidth": float(resolved.pedestal_fwidth),
            "freq_range": tuple(resolved.freq_range),
            "minimum_amplitude": float(resolved.minimum_amplitude),
            "characteristic_frequency": float(resolved.characteristic_frequency),
            "settling_time": float(resolved.settling_time),
        }
    return {
        "kind": "ricker_wavelet",
        "kind_code": SOURCE_TIME_KIND_RICKER_WAVELET,
        "frequency": float(resolved.frequency),
        "fwidth": 0.0,
        "amplitude": float(resolved.amplitude),
        "phase": 0.0,
        "delay": float(resolved.delay),
        "characteristic_frequency": float(resolved.characteristic_frequency),
        "settling_time": float(resolved.settling_time),
    }


def evaluate_source_time(source_time: SourceTime | dict[str, float | int | str], t: float) -> float:
    if isinstance(source_time, dict):
        kind = str(source_time["kind"])
        frequency = float(source_time["frequency"])
        amplitude = float(source_time["amplitude"])
        phase = float(source_time.get("phase", 0.0))
        delay = float(source_time.get("delay", 0.0))
        if kind == "custom":
            fn = source_time.get("fn")
            if fn is not None:
                return amplitude * float(fn(float(t)))
            return amplitude * float(source_time["table"].evaluate(float(t)))
        if kind == "broadband_pulse":
            return broadband_pulse_signal(
                t,
                frequency=frequency,
                fwidth=float(source_time["fwidth"]),
                amplitude=amplitude,
                phase=phase,
                delay=delay,
                pedestal=float(source_time["pedestal"]),
                pedestal_fwidth=float(source_time["pedestal_fwidth"]),
            )
        if kind == "cw":
            return cw_signal(t, frequency=frequency, amplitude=amplitude, phase=phase)
        if kind == "gaussian_pulse":
            return gaussian_pulse_signal(
                t,
                frequency=frequency,
                fwidth=float(source_time["fwidth"]),
                amplitude=amplitude,
                phase=phase,
                delay=delay,
            )
        return ricker_wavelet_signal(t, frequency=frequency, amplitude=amplitude, delay=delay)
    return float(source_time.evaluate(float(t)))


def evaluate_source_time_normalization(
    source_time: SourceTime | dict[str, float | int | str], t: float
) -> float:
    """Evaluate only the arbitrary waveform envelope used for spectral normalization.

    User amplitude and phase are physical source controls and must remain in frequency-domain
    results.  This matches Tidy3D's normalization contract while still removing pulse width,
    delay, and other waveform-shape factors.
    """
    if isinstance(source_time, dict):
        normalized = dict(source_time)
        normalized["amplitude"] = 1.0
        normalized["phase"] = 0.0
        if normalized["kind"] == "broadband_pulse":
            normalized["pedestal"] = broadband_pedestal(
                float(normalized["frequency"]),
                float(normalized["fwidth"]),
                float(normalized["freq_range"][0]),
                0.0,
            )[0]
        if isinstance(t, torch.Tensor):
            from .waveforms import evaluate_time_shifted_waveform_torch
            return evaluate_time_shifted_waveform_torch(normalized, t)
        return evaluate_source_time(normalized, t)
    if isinstance(source_time, CustomSourceTime):
        return source_time.evaluate_shape(t)
    if isinstance(source_time, BroadbandPulse):
        # Same compiled-dict path; avoids re-running the broadband width solve.
        return evaluate_source_time_normalization(compile_source_time(source_time, default_frequency=1.0), t)
    updates = {"amplitude": 1.0}
    if hasattr(source_time, "phase"):
        updates["phase"] = 0.0
    normalized = replace(source_time, **updates)
    return float(normalized.evaluate(float(t)))


@dataclass(frozen=True)
class PointDipole:
    """Electric current moment with a spatial profile of unit volume integral.

    Polarization times the source-time amplitude specifies the current moment
    in A m. FDFD and FDTD CW spectra use peak phasors with exp(-i omega t):
    ``CW(amplitude=A, phase=p)`` has current-moment phasor ``A*exp(-i*p)``.
    """

    position: tuple[float, float, float]
    polarization: tuple[float, float, float]
    width: float = POINT_DIPOLE_REFERENCE_WIDTH
    profile: str = "gaussian"
    source_time: SourceTime | None = None
    name: str | None = None
    kind: str = "point_dipole"

    def __init__(
        self,
        position,
        polarization="Ez",
        width=POINT_DIPOLE_REFERENCE_WIDTH,
        profile="gaussian",
        source_time=None,
        name=None,
    ):
        object.__setattr__(self, "position", _require_length3("position", position))
        object.__setattr__(self, "polarization", polarization_vector(polarization))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "profile", _normalize_point_dipole_profile(profile))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "point_dipole")
        if self.width <= 0.0:
            raise ValueError("width must be > 0.")


@dataclass(frozen=True)
class PlaneWave:
    """A transverse plane wave with a declared propagation and polarization.

    For FDTD TFSF CW injection, the spatial phase reference is the injection-box
    centre. A slab uses its normal interval centre and the physical domain's
    transverse centre. The exported peak phasor is proportional to
    ``exp(i*k_discrete dot (r-r_reference) - i*source_time.phase)``; changing only
    PML padding does not change that physical phase reference.

    FDTD E spectra sample after the electric update on the current DFT step
    label; H spectra include the native half-step correction. The CW incident
    carrier compensates that known update offset, so the phase above describes
    the exported field rather than an auxiliary hard seed. An auxiliary-line
    pulse instead imposes the declared waveform on the physical upstream Yee
    plane of the injection box (the entry supporting plane at oblique incidence),
    on the exported integer-step clock. Upstream ghost samples supply the
    discrete staggered face fields; propagation into the box remains causal.
    Bloch analytic slab pulses retain their existing centre reference.
    """

    direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    source_time: SourceTime | None = None
    injection: Injection = "soft"
    injection_axis: str | None = None
    name: str | None = None
    kind: str = "plane_wave"

    def __init__(
        self,
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        source_time=None,
        injection="soft",
        injection_axis=None,
        name=None,
    ):
        direction, polarization = _validate_transverse_polarization(direction, polarization)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "source_time", source_time)
        if isinstance(injection, TFSF):
            normalized_injection = injection
        else:
            normalized_injection = str(injection).lower()
        object.__setattr__(self, "injection_axis", _normalize_axis(injection_axis))
        object.__setattr__(self, "injection", normalized_injection)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "plane_wave")
        if not isinstance(self.injection, TFSF) and self.injection != "soft":
            raise ValueError("PlaneWave injection must be 'soft' or TFSF(...).")


@dataclass(frozen=True, eq=False)
class MaxwellPlaneModes:
    """Physical homogeneous Maxwell modes, sampled at one frequency.

    Rows of q, E and H use 1/m, V/m and A/m respectively, with the
    exp(-i omega t) convention. Declarations retain native tensor gradients;
    the consuming compiler validates the sampled medium and Maxwell equations.
    These are not direction/polarization vectors or dimensionless eigenvalues.
    """
    frequency: float
    wavevectors: torch.Tensor
    electric: torch.Tensor
    magnetic: torch.Tensor
    origin: torch.Tensor | None = None
    name: str | None = None
    kind: str = "maxwell_plane_modes"

    def __post_init__(self):
        if (type(self.frequency) not in (int, float)
                or not math.isfinite(self.frequency) or self.frequency <= 0):
            raise ValueError('Maxwell mode frequency must be finite and positive in Hz')
        object.__setattr__(self, 'frequency', float(self.frequency))
        template = self.wavevectors
        if (not isinstance(template, torch.Tensor) or template.ndim != 2
                or template.shape[1] != 3 or not len(template)
                or template.dtype not in (torch.complex64, torch.complex128)):
            raise ValueError('Maxwell wavevectors must be a nonempty complex [N,3] tensor in 1/m')
        for name in ('wavevectors', 'electric', 'magnetic'):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.shape != template.shape
                    or value.device != template.device or value.dtype != template.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Maxwell q/E/H must be finite matching complex [N,3] tensors')
            object.__setattr__(self, name, value.clone())
        origin = template.real.new_zeros(3) if self.origin is None else self.origin
        if (not isinstance(origin, torch.Tensor) or origin.shape != (3,)
                or origin.device != template.device or origin.dtype != template.real.dtype
                or not bool(torch.isfinite(origin).all())):
            raise ValueError('Maxwell mode origin must be a finite matching real [3] tensor in m')
        object.__setattr__(self, 'origin', origin.clone())
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError('Maxwell mode name must be a string or None')
        if self.kind != 'maxwell_plane_modes':
            raise ValueError('Maxwell mode source kind cannot be changed')


@dataclass(frozen=True)
class GaussianBeam:
    direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    beam_waist: float
    focus: tuple[float, float, float]
    source_time: SourceTime | None = None
    injection: Injection = "soft"
    injection_axis: str | None = None
    name: str | None = None
    kind: str = "gaussian_beam"

    def __init__(
        self,
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        beam_waist=0.5,
        focus=(0.0, 0.0, 0.0),
        source_time=None,
        injection="soft",
        injection_axis=None,
        name=None,
    ):
        direction, polarization = _validate_transverse_polarization(direction, polarization)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "beam_waist", float(beam_waist))
        object.__setattr__(self, "focus", _require_length3("focus", focus))
        object.__setattr__(self, "source_time", source_time)
        if isinstance(injection, TFSF):
            normalized_injection = injection
        else:
            normalized_injection = str(injection).lower()
        object.__setattr__(self, "injection_axis", _normalize_axis(injection_axis))
        object.__setattr__(self, "injection", normalized_injection)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "gaussian_beam")
        if self.beam_waist <= 0.0:
            raise ValueError("beam_waist must be > 0.")
        if not isinstance(self.injection, TFSF) and self.injection != "soft":
            raise ValueError("GaussianBeam injection must be 'soft' or TFSF(...).")


@dataclass(frozen=True)
class AstigmaticGaussianBeam:
    direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    beam_waist: tuple[float, float]
    focus: tuple[float, float, float]
    focus_u: float
    focus_v: float
    source_time: SourceTime | None = None
    injection: Injection = "soft"
    injection_axis: str | None = None
    name: str | None = None
    kind: str = "astigmatic_gaussian_beam"

    def __init__(
        self,
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        beam_waist=(0.5, 0.5),
        focus=(0.0, 0.0, 0.0),
        focus_u=0.0,
        focus_v=0.0,
        source_time=None,
        injection="soft",
        injection_axis=None,
        name=None,
    ):
        direction, polarization = _validate_transverse_polarization(direction, polarization)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "beam_waist", _require_beam_waist_pair(beam_waist))
        object.__setattr__(self, "focus", _require_length3("focus", focus))
        object.__setattr__(self, "focus_u", float(focus_u))
        object.__setattr__(self, "focus_v", float(focus_v))
        object.__setattr__(self, "source_time", source_time)
        if isinstance(injection, TFSF):
            normalized_injection = injection
        else:
            normalized_injection = str(injection).lower()
        object.__setattr__(self, "injection_axis", _normalize_axis(injection_axis))
        object.__setattr__(self, "injection", normalized_injection)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "astigmatic_gaussian_beam")
        if not isinstance(self.injection, TFSF) and self.injection != "soft":
            raise ValueError("AstigmaticGaussianBeam injection must be 'soft' or TFSF(...).")


_FIELD_DATASET_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_CURRENT_DATASET_COMPONENTS = ("Jx", "Jy", "Jz", "Mx", "My", "Mz")


def _normalize_dataset(coords, components, *, allowed, dataset_name):
    if len(coords) != 3:
        raise ValueError(f"{dataset_name} coords must contain three (x, y, z) sample arrays.")
    axes = []
    for axis_name, values in zip("xyz", coords):
        axis = np.asarray(values, dtype=np.float64).reshape(-1)
        if axis.size < 1:
            raise ValueError(f"{dataset_name} {axis_name}-coordinates must be non-empty.")
        if axis.size >= 2 and np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{dataset_name} {axis_name}-coordinates must be strictly increasing.")
        axes.append(axis)
    grid_shape = tuple(int(axis.size) for axis in axes)

    if not isinstance(components, dict) or not components:
        raise ValueError(f"{dataset_name} requires a non-empty component mapping.")
    normalized = {}
    for key, values in components.items():
        if key not in allowed:
            raise ValueError(f"{dataset_name} component {key!r} must be one of {allowed}.")
        data = np.asarray(values, dtype=np.float64)
        if data.shape != grid_shape:
            raise ValueError(
                f"{dataset_name} component {key!r} has shape {data.shape}; expected coords grid {grid_shape}."
            )
        normalized[key] = np.ascontiguousarray(data)
    coord_tuple = tuple(tuple(float(v) for v in axis) for axis in axes)
    return coord_tuple, normalized


@dataclass(frozen=True)
class FieldDataset:
    """Tangential E/H field distribution sampled on a rectilinear region.

    ``coords`` holds three strictly increasing 1D coordinate arrays (x, y, z).
    ``components`` maps field names (``Ex``..``Hz``) to arrays of shape
    ``(len(x), len(y), len(z))``. CustomFieldSource requires exactly one axis to
    be a single sample so the region defines a plane with a well-defined normal.
    """

    coords: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    components: dict

    def __init__(self, coords, components):
        coord_tuple, normalized = _normalize_dataset(
            coords, components, allowed=_FIELD_DATASET_COMPONENTS, dataset_name="FieldDataset"
        )
        object.__setattr__(self, "coords", coord_tuple)
        object.__setattr__(self, "components", normalized)


@dataclass(frozen=True)
class CurrentDataset:
    """Electric/magnetic current distribution sampled on a rectilinear region.

    ``coords`` holds three strictly increasing 1D coordinate arrays (x, y, z).
    ``components`` maps current names (``Jx``..``Mz``) to arrays of shape
    ``(len(x), len(y), len(z))``.
    Values are volume densities in SI units: J in A/m² and M in V/m².
    Each nonsingleton axis has finite support from its first to last sample;
    partial Yee control cells receive fractional coverage with in-span sampling.
    A singleton axis retains its density on the nearest component Yee plane.
    """

    coords: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    components: dict

    def __init__(self, coords, components):
        coord_tuple, normalized = _normalize_dataset(
            coords, components, allowed=_CURRENT_DATASET_COMPONENTS, dataset_name="CurrentDataset"
        )
        object.__setattr__(self, "coords", coord_tuple)
        object.__setattr__(self, "components", normalized)


@dataclass(frozen=True)
class UniformCurrentSource:
    """Uniform additive electric current filling an axis-aligned box region."""

    size: tuple[float, float, float]
    polarization: tuple[float, float, float]
    center: tuple[float, float, float]
    source_time: SourceTime | None
    name: str | None
    kind: str = "uniform_current"

    def __init__(
        self,
        size,
        polarization="Ez",
        source_time=None,
        center=(0.0, 0.0, 0.0),
        name=None,
    ):
        resolved_size = _require_length3("size", size)
        if any(component < 0.0 for component in resolved_size):
            raise ValueError("size components must be >= 0.")
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "polarization", polarization_vector(polarization))
        object.__setattr__(self, "center", _require_length3("center", center))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "uniform_current")


@dataclass(frozen=True)
class CustomCurrentSource:
    """Volume current densities J (A/m²) and M (V/m²) on a sampled region.

    Support is the endpoint-defined volume of ``current_dataset``. Only the
    covered fraction of each Yee control cell is driven, sampling inside the
    dataset span. Singleton axes use the nearest corresponding Yee plane.
    """

    current_dataset: CurrentDataset
    source_time: SourceTime | None
    name: str | None
    kind: str = "custom_current"

    def __init__(self, current_dataset, source_time=None, name=None):
        if not isinstance(current_dataset, CurrentDataset):
            raise TypeError("current_dataset must be a CurrentDataset instance.")
        object.__setattr__(self, "current_dataset", current_dataset)
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "custom_current")


@dataclass(frozen=True)
class CustomFieldSource:
    """Tangential E/H distribution on a plane injected as equivalent currents.

    The provided tangential fields are converted into equivalent surface currents
    ``J = n x H`` and ``M = -n x E`` on the plane (with normal ``n`` along the
    single-sample dataset axis, pointing toward +normal) and injected additively.
    """

    field_dataset: FieldDataset
    source_time: SourceTime | None
    name: str | None
    kind: str = "custom_field"

    def __init__(self, field_dataset, source_time=None, name=None):
        if not isinstance(field_dataset, FieldDataset):
            raise TypeError("field_dataset must be a FieldDataset instance.")
        normal_axes = [axis for axis, values in zip("xyz", field_dataset.coords) if len(values) == 1]
        if len(normal_axes) != 1:
            raise ValueError(
                "CustomFieldSource requires exactly one single-sample dataset axis to define the plane normal."
            )
        object.__setattr__(self, "field_dataset", field_dataset)
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "custom_field")

    @property
    def normal_axis(self) -> str:
        for axis, values in zip("xyz", self.field_dataset.coords):
            if len(values) == 1:
                return axis
        raise ValueError("CustomFieldSource dataset does not define a plane normal.")


@dataclass(frozen=True)
class ModeSource:
    """Launch a guided mode indexed within the requested polarization family."""

    position: tuple[float, float, float]
    size: tuple[float, float, float]
    mode_index: int
    direction: str
    polarization: tuple[float, float, float]
    source_time: SourceTime | None = None
    injection: Injection = "soft"
    name: str | None = None
    normal_axis: str = "z"
    polarization_axis: str = "x"
    bend_radius: float | None = None
    bend_axis: str | None = None
    kind: str = "mode_source"

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        mode_index=0,
        direction="+",
        polarization="auto",
        source_time=None,
        injection="soft",
        name=None,
        bend_radius=None,
        bend_axis=None,
    ):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_mode_source_size(size)
        normal_axis = _resolve_mode_source_normal_axis(resolved_size)
        polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, resolved_size, polarization)
        resolved_bend_radius, resolved_bend_axis = _normalize_bend(bend_radius, bend_axis, normal_axis)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "size", resolved_size)
        object.__setattr__(self, "mode_index", int(mode_index))
        object.__setattr__(self, "direction", _normalize_mode_direction(direction))
        object.__setattr__(self, "polarization", polarization_vector(f"E{polarization_axis}"))
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "injection", str(injection).lower())
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "polarization_axis", polarization_axis)
        object.__setattr__(self, "bend_radius", resolved_bend_radius)
        object.__setattr__(self, "bend_axis", resolved_bend_axis)
        object.__setattr__(self, "kind", "mode_source")
        if self.mode_index < 0:
            raise ValueError("mode_index must be >= 0.")
        if self.injection != "soft":
            raise ValueError("ModeSource injection must currently be 'soft'.")
