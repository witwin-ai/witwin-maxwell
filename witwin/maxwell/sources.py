from __future__ import annotations

from dataclasses import dataclass
import math


SOURCE_TIME_KIND_CW = 0
SOURCE_TIME_KIND_GAUSSIAN_PULSE = 1
SOURCE_TIME_KIND_RICKER_WAVELET = 2
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


SourceTime = CW | GaussianPulse | RickerWavelet


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
    if not isinstance(source_time, (CW, GaussianPulse, RickerWavelet)):
        raise TypeError("source_time must be CW, GaussianPulse, RickerWavelet, or None.")
    return source_time


def compile_source_time(source_time: SourceTime | None, *, default_frequency: float) -> dict[str, float | int | str]:
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
    ):
        resolved_position = _require_length3("position", position)
        resolved_size = _require_mode_source_size(size)
        normal_axis = _resolve_mode_source_normal_axis(resolved_size)
        polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, resolved_size, polarization)
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
        object.__setattr__(self, "kind", "mode_source")
        if self.mode_index < 0:
            raise ValueError("mode_index must be >= 0.")
        if self.injection != "soft":
            raise ValueError("ModeSource injection must currently be 'soft'.")
