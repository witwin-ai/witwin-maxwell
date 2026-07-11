from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SOURCE_TIME_KIND_CW = 0
SOURCE_TIME_KIND_GAUSSIAN_PULSE = 1
SOURCE_TIME_KIND_RICKER_WAVELET = 2
SOURCE_TIME_KIND_CUSTOM = 3
POINT_DIPOLE_REFERENCE_WIDTH = 0.02
POINT_DIPOLE_IDEAL_PROFILE_SCALE = 0.75


def _require_length3(name, value) -> tuple[float, float, float]:
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
    frequency = float(value)
    if frequency <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return frequency


def _validate_amplitude(value):
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
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "kind", "cw")

    def evaluate(self, t: float) -> float:
        return self.amplitude * math.cos(2.0 * math.pi * self.frequency * float(t) + self.phase)

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
        sigma_t = 1.0 / (2.0 * math.pi * fwidth)
        delay = 6.0 * sigma_t if self.delay is None else float(self.delay)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "fwidth", fwidth)
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "kind", "gaussian_pulse")

    @property
    def sigma_t(self) -> float:
        return 1.0 / (2.0 * math.pi * self.fwidth)

    def evaluate(self, t: float) -> float:
        tau = float(t) - float(self.delay)
        envelope = math.exp(-0.5 * (tau / self.sigma_t) ** 2)
        return self.amplitude * envelope * math.cos(2.0 * math.pi * self.frequency * tau + self.phase)

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
        delay = 4.0 / (math.pi * frequency) if self.delay is None else float(self.delay)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "amplitude", _validate_amplitude(self.amplitude))
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "kind", "ricker_wavelet")

    def evaluate(self, t: float) -> float:
        tau = float(t) - float(self.delay)
        alpha = math.pi * self.frequency * tau
        alpha_sq = alpha * alpha
        return self.amplitude * (1.0 - 2.0 * alpha_sq) * math.exp(-alpha_sq)

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


def _table_characteristic_frequency(times: np.ndarray, amplitudes: np.ndarray) -> float:
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


@dataclass(frozen=True)
class CustomSourceTime:
    """Arbitrary temporal waveform.

    Construct either from a sampled table ``CustomSourceTime(times, amplitudes)``
    or from a callable ``CustomSourceTime(fn)``. The signal is evaluated on the
    solver time grid ``n * dt`` through the Python scalar injection path, so the
    native CUDA time-shifted kernel is not involved.
    """

    times: tuple[float, ...] | None = None
    amplitudes: tuple[float, ...] | None = None
    amplitude: float = 1.0
    kind: str = "custom"

    def __init__(self, times=None, amplitudes=None, *, amplitude=1.0, characteristic_frequency=None):
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
            order = np.argsort(table_times)
            table_times = np.ascontiguousarray(table_times[order])
            table_amplitudes = np.ascontiguousarray(table_amplitudes[order])
            if characteristic_frequency is None:
                char_freq = _table_characteristic_frequency(table_times, table_amplitudes)
            else:
                char_freq = float(characteristic_frequency)
            delay = float(table_times[0])
            settling_time = float(table_times[-1])

        object.__setattr__(self, "times", None if table_times is None else tuple(float(v) for v in table_times))
        object.__setattr__(
            self, "amplitudes", None if table_amplitudes is None else tuple(float(v) for v in table_amplitudes)
        )
        object.__setattr__(self, "amplitude", _validate_amplitude(amplitude))
        object.__setattr__(self, "kind", "custom")
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_table_times", table_times)
        object.__setattr__(self, "_table_amplitudes", table_amplitudes)
        object.__setattr__(self, "_characteristic_frequency", char_freq)
        object.__setattr__(self, "_delay", delay)
        object.__setattr__(self, "_settling_time", settling_time)

    @property
    def fn(self):
        return self._fn

    def evaluate(self, t: float) -> float:
        if self._fn is not None:
            return float(self.amplitude) * float(self._fn(float(t)))
        value = np.interp(float(t), self._table_times, self._table_amplitudes, left=0.0, right=0.0)
        return float(self.amplitude) * float(value)

    @property
    def frequency(self) -> float:
        return float(self._characteristic_frequency)

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


SourceTime = CW | GaussianPulse | RickerWavelet | CustomSourceTime


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
    if not isinstance(source_time, (CW, GaussianPulse, RickerWavelet, CustomSourceTime)):
        raise TypeError("source_time must be CW, GaussianPulse, RickerWavelet, CustomSourceTime, or None.")
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
            "frequency": float(resolved.characteristic_frequency),
            "fwidth": 0.0,
            "amplitude": float(resolved.amplitude),
            "phase": 0.0,
            "delay": float(resolved.delay),
            "characteristic_frequency": float(resolved.characteristic_frequency),
            "settling_time": float(resolved.settling_time),
            "fn": resolved.fn,
            "times": None if resolved.times is None else resolved.times,
            "amplitudes": None if resolved.amplitudes is None else resolved.amplitudes,
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
            table_times = source_time["times"]
            table_amplitudes = source_time["amplitudes"]
            value = np.interp(float(t), table_times, table_amplitudes, left=0.0, right=0.0)
            return amplitude * float(value)
        if kind == "cw":
            return amplitude * math.cos(2.0 * math.pi * frequency * float(t) + phase)
        if kind == "gaussian_pulse":
            fwidth = float(source_time["fwidth"])
            sigma_t = 1.0 / (2.0 * math.pi * fwidth)
            tau = float(t) - delay
            envelope = math.exp(-0.5 * (tau / sigma_t) ** 2)
            return amplitude * envelope * math.cos(2.0 * math.pi * frequency * tau + phase)
        alpha = math.pi * frequency * (float(t) - delay)
        alpha_sq = alpha * alpha
        return amplitude * (1.0 - 2.0 * alpha_sq) * math.exp(-alpha_sq)
    return float(source_time.evaluate(float(t)))


@dataclass(frozen=True)
class PointDipole:
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
    """Arbitrary volume electric (J) and magnetic (M) current distribution."""

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
