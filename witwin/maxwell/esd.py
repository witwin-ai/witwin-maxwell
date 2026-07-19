"""Electrostatic-discharge (ESD) excitation waveforms and terminal injection.

Capability level: **stress-only**. These objects reproduce standard ESD current
waveforms and inject them into a terminal port as an ideal (prescribed) current
so the field solver can report local field stress, port V/I, charge, and action
integral. They do NOT model source-impedance networks, discharge-gun geometry,
arc channels, or device failure probability. A standard waveform class name does
not by itself constitute standard certification.

The reference analytic current is the four-parameter two-term Heidler sum

    i(t) = (I1/eta1) * (t/tau1)^n / (1 + (t/tau1)^n) * exp(-t/tau2)
         + (I2/eta2) * (t/tau3)^n / (1 + (t/tau3)^n) * exp(-t/tau4)

with n = 1.8. ``eta_k`` are peak-normalization factors computed numerically so
each term's ``I_k`` is its own peak contribution. The published contact-discharge
parameter set is scaled linearly by the level voltage; the decay/rise time
constants are voltage independent.

Injection reuses the standard ``Scene -> Simulation.fdtd -> Result`` flow: an
:class:`ESDCurrentSource` lowers to a uniform additive current source over the
resolved terminal-port gap (an ideal current injection). Source-impedance
networks and discharge-gun coupling are out of scope for this phase.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import torch

from .sources import CustomSourceTime, UniformCurrentSource, polarization_vector


# Model bookkeeping -----------------------------------------------------------

ESD_MODEL_VERSION = "esd-heidler-1"
ESD_CAPABILITY_LEVEL = "stress-only"
_HEIDLER_EXPONENT = 1.8

# Published two-term Heidler contact-discharge parameter set for the IEC
# 61000-4-2 first-transient current, expressed at a 4 kV reference level. The
# amplitudes ``i`` scale linearly with the requested level voltage; the time
# constants are voltage independent. Values are widely reported in the ESD
# modelling literature for the contact-discharge current waveform.
_SUPPORTED_REVISIONS: dict[str, dict[str, Any]] = {
    "ed2-contact": {
        "reference_voltage": 4000.0,
        "n": _HEIDLER_EXPONENT,
        "terms": (
            {"i": 16.6, "tau_rise": 1.1e-9, "tau_decay": 2.0e-9},
            {"i": 9.3, "tau_rise": 12.0e-9, "tau_decay": 37.0e-9},
        ),
    },
}
_DEFAULT_REVISION = "ed2-contact"
_SUPPORTED_DISCHARGES = ("contact",)


def _heidler_shape(
    t: torch.Tensor, tau_rise: float, tau_decay: float, n: float
) -> torch.Tensor:
    """Un-normalized single Heidler term evaluated on a torch tensor.

    Returns zero for ``t <= 0`` (the fractional power of a negative base is
    undefined and the ESD waveform is defined for ``t >= 0``).
    """

    positive = t > 0.0
    safe_t = torch.where(positive, t, torch.ones_like(t))
    x = safe_t / tau_rise
    xn = torch.pow(x, n)
    shape = xn / (1.0 + xn) * torch.exp(-safe_t / tau_decay)
    return torch.where(positive, shape, torch.zeros_like(t))


def _term_eta(tau_rise: float, tau_decay: float, n: float) -> float:
    """Numerically compute the peak-normalization factor of one Heidler term."""

    # The peak lies within a few rise-time constants; sample densely and refine.
    upper = max(10.0 * tau_rise, 4.0 * tau_decay)
    grid = torch.linspace(0.0, upper, 200_001, dtype=torch.float64)
    shape = _heidler_shape(grid, tau_rise, tau_decay, n)
    peak_index = int(torch.argmax(shape))
    lo = grid[max(peak_index - 2, 0)]
    hi = grid[min(peak_index + 2, grid.numel() - 1)]
    refined = torch.linspace(float(lo), float(hi), 200_001, dtype=torch.float64)
    refined_shape = _heidler_shape(refined, tau_rise, tau_decay, n)
    return float(torch.max(refined_shape))


@dataclass(frozen=True)
class ESDDiagnostics:
    """Scalar summary of an ESD current waveform.

    All values are computed on the analytic (or measured) current with SI units
    (seconds, amperes). ``charge`` is ``integral i dt`` and ``action_integral``
    is ``integral i^2 dt`` over the reported support.
    """

    peak_current: float
    current_at_30ns: float
    current_at_60ns: float
    rise_time_10_90: float
    charge: float
    action_integral: float
    peak_time: float

    def as_dict(self) -> dict[str, float]:
        return {
            "peak_current": self.peak_current,
            "current_at_30ns": self.current_at_30ns,
            "current_at_60ns": self.current_at_60ns,
            "rise_time_10_90": self.rise_time_10_90,
            "charge": self.charge,
            "action_integral": self.action_integral,
            "peak_time": self.peak_time,
        }


@dataclass(frozen=True)
class ESDResampledWaveform:
    """Charge-conserving resampling of a waveform onto an FDTD time grid.

    ``times`` are bin-center sample times and ``currents`` are the per-bin mean
    current ``(1/dt) integral_bin i dt``. Summing ``currents * dt`` recovers the
    total charge over the covered support, so charge is conserved by construction
    regardless of ``dt`` (naive point sampling does not have this property).
    """

    dt: float
    times: torch.Tensor
    currents: torch.Tensor
    charge: float
    action_integral: float
    analytic_charge: float
    analytic_action_integral: float

    @property
    def charge_ratio(self) -> float:
        """Resampled charge divided by the analytic charge (~1 by construction)."""

        return self.charge / self.analytic_charge

    @property
    def action_ratio(self) -> float:
        """Resampled action integral divided by the analytic action integral.

        The gap from unity is an aliasing diagnostic: bin-averaging removes the
        squared curvature that a finer grid retains, so ``action_ratio -> 1`` as
        ``dt -> 0``.
        """

        return self.action_integral / self.analytic_action_integral

    @property
    def aliasing_metric(self) -> float:
        """Fractional action-integral error introduced by the resampling grid."""

        return abs(1.0 - self.action_ratio)


class _WaveformBase(abc.ABC):
    """Shared diagnostics, resampling, and source-time lowering for waveforms."""

    @abc.abstractmethod
    def current(self, t) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def support(self) -> tuple[float, float]:
        ...

    @property
    @abc.abstractmethod
    def provenance(self) -> dict[str, Any]:
        ...

    def _dense_grid(self, samples: int = 400_001) -> torch.Tensor:
        start, stop = self.support
        return torch.linspace(float(start), float(stop), int(samples), dtype=torch.float64)

    def diagnostics(self) -> ESDDiagnostics:
        grid = self._dense_grid()
        current = self.current(grid)
        peak_index = int(torch.argmax(current))
        peak_current = float(current[peak_index])
        peak_time = float(grid[peak_index])
        charge = float(torch.trapezoid(current, grid))
        action = float(torch.trapezoid(current * current, grid))

        def sample_at(time_value: float) -> float:
            point = torch.tensor([time_value], dtype=torch.float64)
            return float(self.current(point)[0])

        rise_time = _rise_time_10_90(grid, current, peak_index)
        return ESDDiagnostics(
            peak_current=peak_current,
            current_at_30ns=sample_at(30.0e-9),
            current_at_60ns=sample_at(60.0e-9),
            rise_time_10_90=rise_time,
            charge=charge,
            action_integral=action,
            peak_time=peak_time,
        )

    def resample_to_grid(
        self,
        dt: float,
        *,
        t_end: float | None = None,
        subsamples: int = 32,
    ) -> ESDResampledWaveform:
        """Charge-conserving binned resampling onto an ``dt``-spaced grid."""

        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be a positive finite time step.")
        start, stop = self.support
        end = float(stop if t_end is None else t_end)
        if end <= float(start):
            raise ValueError("t_end must be greater than the waveform start time.")
        bin_count = int(math.ceil((end - float(start)) / step))
        if bin_count < 1:
            raise ValueError("Resampling requires at least one dt bin.")
        if int(subsamples) < 2:
            raise ValueError("subsamples must be >= 2 for bin integration.")

        edges = float(start) + step * torch.arange(bin_count + 1, dtype=torch.float64)
        # Composite trapezoidal integral of i(t) over each dt bin, divided by dt.
        local = torch.linspace(0.0, 1.0, int(subsamples), dtype=torch.float64)
        sample_times = edges[:-1, None] + step * local[None, :]
        sample_current = self.current(sample_times.reshape(-1)).reshape(bin_count, int(subsamples))
        bin_integral = torch.trapezoid(sample_current, dx=step / (int(subsamples) - 1), dim=1)
        bin_mean = bin_integral / step
        centers = 0.5 * (edges[:-1] + edges[1:])

        charge = float(torch.sum(bin_mean) * step)
        action = float(torch.sum(bin_mean * bin_mean) * step)
        analytic = self.diagnostics()
        return ESDResampledWaveform(
            dt=step,
            times=centers,
            currents=bin_mean,
            charge=charge,
            action_integral=action,
            analytic_charge=analytic.charge,
            analytic_action_integral=analytic.action_integral,
        )

    def to_source_time(self, *, samples: int = 4001) -> CustomSourceTime:
        """Dense analytic table suitable for additive current injection.

        The FDTD scalar-injection path point-samples this table at ``n*dt``.
        Because the analytic ESD current is smooth and the table is dense, the
        injected samples reproduce the target current within tabulation error.
        Charge-conserving grid resampling is provided separately by
        :meth:`resample_to_grid` for reporting and verification.
        """

        grid = self._dense_grid(samples=int(samples))
        current = self.current(grid)
        char_freq = _characteristic_frequency(self)
        return CustomSourceTime(
            grid.tolist(),
            current.tolist(),
            characteristic_frequency=char_freq,
        )


def _rise_time_10_90(grid: torch.Tensor, current: torch.Tensor, peak_index: int) -> float:
    peak = float(current[peak_index])
    if peak <= 0.0:
        return 0.0
    rising_grid = grid[: peak_index + 1]
    rising = current[: peak_index + 1]

    def crossing(level: float) -> float:
        target = level * peak
        above = rising >= target
        first = int(torch.argmax(above.to(torch.int64)))
        if first == 0:
            return float(rising_grid[0])
        t0 = float(rising_grid[first - 1])
        t1 = float(rising_grid[first])
        y0 = float(rising[first - 1])
        y1 = float(rising[first])
        if y1 == y0:
            return t1
        return t0 + (target - y0) * (t1 - t0) / (y1 - y0)

    return crossing(0.9) - crossing(0.1)


def _characteristic_frequency(waveform: "_WaveformBase") -> float:
    diagnostics = waveform.diagnostics()
    rise = diagnostics.rise_time_10_90
    if rise > 0.0:
        # Rise-time bandwidth rule of thumb (0.35 / t_rise).
        return 0.35 / rise
    start, stop = waveform.support
    span = float(stop) - float(start)
    return 1.0 / span if span > 0.0 else 1.0


@dataclass(frozen=True)
class ESDWaveform(_WaveformBase):
    """Versioned standard ESD current waveform (two-term Heidler sum)."""

    level_voltage: float
    discharge: str
    standard_revision: str
    _terms: tuple[dict[str, float], ...] = field(repr=False)
    _n: float = field(repr=False)
    _reference_voltage: float = field(repr=False)
    _support: tuple[float, float] = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "Construct ESDWaveform through a standard factory such as "
            "ESDWaveform.iec_61000_4_2(...)."
        )

    @classmethod
    def iec_61000_4_2(
        cls,
        level_voltage: float,
        *,
        discharge: str = "contact",
        standard_revision: str | None = None,
    ) -> "ESDWaveform":
        """Standard contact-discharge current waveform scaled by ``level_voltage``.

        ``standard_revision`` must name a supported revision; when omitted the
        documented default ``"ed2-contact"`` is used and recorded in provenance.
        """

        voltage = float(level_voltage)
        if not math.isfinite(voltage) or voltage <= 0.0:
            raise ValueError("level_voltage must be a positive finite voltage.")
        resolved_discharge = str(discharge).strip().lower()
        if resolved_discharge not in _SUPPORTED_DISCHARGES:
            raise ValueError(
                f"discharge {discharge!r} is not supported; this phase supports "
                f"{_SUPPORTED_DISCHARGES} only (air discharge is deferred)."
            )
        revision = _DEFAULT_REVISION if standard_revision is None else str(standard_revision).strip()
        if revision not in _SUPPORTED_REVISIONS:
            raise ValueError(
                f"standard_revision {revision!r} is not supported; supported "
                f"revisions are {tuple(_SUPPORTED_REVISIONS)}."
            )
        spec = _SUPPORTED_REVISIONS[revision]
        reference_voltage = float(spec["reference_voltage"])
        n = float(spec["n"])
        scale = voltage / reference_voltage
        terms = tuple(
            {
                "i": float(term["i"]) * scale,
                "tau_rise": float(term["tau_rise"]),
                "tau_decay": float(term["tau_decay"]),
                "eta": _term_eta(float(term["tau_rise"]), float(term["tau_decay"]), n),
            }
            for term in spec["terms"]
        )
        max_decay = max(term["tau_decay"] for term in terms)
        support = (0.0, 20.0 * max_decay)

        instance = object.__new__(cls)
        object.__setattr__(instance, "level_voltage", voltage)
        object.__setattr__(instance, "discharge", resolved_discharge)
        object.__setattr__(instance, "standard_revision", revision)
        object.__setattr__(instance, "_terms", terms)
        object.__setattr__(instance, "_n", n)
        object.__setattr__(instance, "_reference_voltage", reference_voltage)
        object.__setattr__(instance, "_support", support)
        return instance

    @property
    def support(self) -> tuple[float, float]:
        return self._support

    def current(self, t) -> torch.Tensor:
        tensor = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float64)
        if tensor.is_complex():
            raise ValueError("ESD current time argument must be real.")
        working = tensor.to(dtype=torch.float64) if tensor.dtype != torch.float64 else tensor
        total = torch.zeros_like(working)
        for term in self._terms:
            shape = _heidler_shape(working, term["tau_rise"], term["tau_decay"], self._n)
            total = total + (term["i"] / term["eta"]) * shape
        if tensor.dtype != torch.float64 and not torch.is_floating_point(tensor):
            return total
        return total.to(dtype=tensor.dtype) if torch.is_tensor(t) else total

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "kind": "esd_waveform",
            "standard": "IEC 61000-4-2",
            "standard_revision": self.standard_revision,
            "discharge": self.discharge,
            "level_voltage": self.level_voltage,
            "reference_voltage": self._reference_voltage,
            "n": self._n,
            "terms": tuple(dict(term) for term in self._terms),
            "capability_level": ESD_CAPABILITY_LEVEL,
            "model_version": ESD_MODEL_VERSION,
            "units": {"time": "s", "current": "A"},
        }


@dataclass(frozen=True)
class MeasuredWaveform(_WaveformBase):
    """User-supplied measured ESD current samples.

    ``time`` and ``values`` are strictly-increasing time samples (seconds) and
    the corresponding current (amperes). ``bandwidth`` and ``provenance`` record
    measurement metadata that flows into the run result.
    """

    time: torch.Tensor
    values: torch.Tensor
    units: str
    bandwidth: float | None
    _provenance: dict[str, Any] = field(repr=False)

    def __init__(self, time, values, *, units="A", bandwidth=None, provenance=None):
        time_tensor = torch.as_tensor(time, dtype=torch.float64).reshape(-1)
        value_tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if time_tensor.numel() < 2:
            raise ValueError("MeasuredWaveform requires at least two samples.")
        if time_tensor.shape != value_tensor.shape:
            raise ValueError("time and values must have the same length.")
        if bool(torch.any(torch.diff(time_tensor) <= 0.0)):
            raise ValueError("MeasuredWaveform time samples must be strictly increasing.")
        if str(units).strip().upper() != "A":
            raise ValueError("MeasuredWaveform currently supports current samples in amperes ('A').")
        resolved_bandwidth = None if bandwidth is None else float(bandwidth)
        if resolved_bandwidth is not None and (
            not math.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0.0
        ):
            raise ValueError("bandwidth must be a positive finite frequency or None.")
        object.__setattr__(self, "time", time_tensor)
        object.__setattr__(self, "values", value_tensor)
        object.__setattr__(self, "units", "A")
        object.__setattr__(self, "bandwidth", resolved_bandwidth)
        object.__setattr__(self, "_provenance", dict(provenance or {}))

    @property
    def support(self) -> tuple[float, float]:
        return (float(self.time[0]), float(self.time[-1]))

    def current(self, t) -> torch.Tensor:
        tensor = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float64)
        working = tensor.to(dtype=torch.float64)
        flat = working.reshape(-1)
        interpolated = np.interp(
            flat.detach().cpu().numpy(),
            self.time.numpy(),
            self.values.numpy(),
            left=0.0,
            right=0.0,
        )
        result = torch.as_tensor(interpolated, dtype=torch.float64).reshape(working.shape)
        return result.to(dtype=tensor.dtype) if torch.is_tensor(t) and torch.is_floating_point(tensor) else result

    @property
    def provenance(self) -> dict[str, Any]:
        payload = {
            "kind": "measured_waveform",
            "units": self.units,
            "bandwidth": self.bandwidth,
            "sample_count": int(self.time.numel()),
            "capability_level": ESD_CAPABILITY_LEVEL,
            "model_version": ESD_MODEL_VERSION,
        }
        payload.update(self._provenance)
        return payload


Waveform = ESDWaveform | MeasuredWaveform


@dataclass(frozen=True)
class ESDCurrentSource:
    """Bind an ESD waveform to a terminal port as an ideal current injection.

    Capability level: **stress-only**. The bound waveform is injected as a
    prescribed (ideal) additive current across the resolved terminal-port gap;
    there is no source-impedance network (Phase 3). ``direction`` selects the
    sign of the injected current relative to the port's positive terminal.
    """

    name: str
    port_name: str
    waveform: Waveform
    direction: str
    kind: str = "esd_current_source"

    def __init__(self, name, *, port, waveform, direction="+"):
        resolved_name = str(name).strip()
        if not resolved_name:
            raise ValueError("ESDCurrentSource name must not be empty.")
        resolved_port = str(port).strip()
        if not resolved_port:
            raise ValueError("ESDCurrentSource port must name a terminal port.")
        if not isinstance(waveform, (ESDWaveform, MeasuredWaveform)):
            raise TypeError("waveform must be an ESDWaveform or MeasuredWaveform.")
        resolved_direction = str(direction).strip()
        if resolved_direction not in {"+", "-"}:
            raise ValueError("direction must be '+' or '-'.")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "port_name", resolved_port)
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "direction", resolved_direction)
        object.__setattr__(self, "kind", "esd_current_source")

    def _resolve_port(self, scene):
        matches = [
            port
            for port in scene.ports
            if getattr(port, "name", None) == self.port_name
        ]
        if not matches:
            raise ValueError(
                f"ESDCurrentSource {self.name!r} references missing terminal port "
                f"{self.port_name!r}."
            )
        port = matches[0]
        if getattr(port, "kind", None) != "terminal_port":
            raise ValueError(
                f"ESDCurrentSource {self.name!r} requires a TerminalPort; port "
                f"{self.port_name!r} is a {getattr(port, 'kind', type(port).__name__)!r}."
            )
        return port

    def injection_geometry(self, scene) -> dict[str, Any]:
        """Resolve the port-gap injection box, axis, and footprint area."""

        port = self._resolve_port(scene)
        axis = port.voltage_path.axis
        axis_index = "xyz".index(axis)
        positive = port.positive
        negative = port.negative
        surface = port.current_surface
        surface_position = tuple(float(v) for v in torch.as_tensor(surface.position).reshape(-1).tolist())
        surface_size = tuple(float(v) for v in torch.as_tensor(surface.size).reshape(-1).tolist())
        footprint = 1.0
        for index in range(3):
            if index != axis_index:
                footprint *= float(surface_size[index])
        if footprint <= 0.0:
            raise ValueError(
                f"ESDCurrentSource {self.name!r} port {self.port_name!r} has a "
                "degenerate current-contour footprint."
            )
        gap = abs(float(positive[axis_index]) - float(negative[axis_index]))
        center = list(surface_position)
        size = list(surface_size)
        size[axis_index] = gap
        center[axis_index] = 0.5 * (float(positive[axis_index]) + float(negative[axis_index]))
        sign = 1.0 if float(positive[axis_index]) > float(negative[axis_index]) else -1.0
        if self.direction == "-":
            sign = -sign
        return {
            "axis": axis,
            "axis_index": axis_index,
            "center": tuple(center),
            "size": tuple(size),
            "footprint_area": footprint,
            "gap": gap,
            "polarization_sign": sign,
        }

    def resolve(self, scene) -> UniformCurrentSource:
        """Lower to a uniform additive current source over the port gap.

        The waveform current is divided by the current-contour footprint area so
        the uniform current density integrates to the target port current. The
        polarization is aligned with the port voltage-path axis and signed toward
        the positive terminal (an ideal current injection).
        """

        geometry = self.injection_geometry(scene)
        axis = geometry["axis"]
        area = geometry["footprint_area"]
        sign = geometry["polarization_sign"]
        base = self.waveform.to_source_time()
        # Bake the density normalization (target current / footprint area) and the
        # port orientation sign into the tabulated source-time amplitudes.
        density = [sign * float(value) / area for value in base.amplitudes]
        source_time = CustomSourceTime(
            list(base.times),
            density,
            characteristic_frequency=base.characteristic_frequency,
        )
        polarization = polarization_vector(f"E{axis}")
        return UniformCurrentSource(
            size=geometry["size"],
            polarization=polarization,
            source_time=source_time,
            center=geometry["center"],
            name=f"{self.name}::current",
        )

    def provenance(self, scene=None) -> dict[str, Any]:
        payload = {
            "kind": "esd_current_source",
            "name": self.name,
            "port": self.port_name,
            "direction": self.direction,
            "waveform": self.waveform.provenance,
            "capability_level": ESD_CAPABILITY_LEVEL,
            "model_version": ESD_MODEL_VERSION,
            "injection": "ideal_current",
            "source_impedance": "none (Phase 3, out of scope)",
        }
        if scene is not None:
            payload["geometry"] = self.injection_geometry(scene)
        return payload


@dataclass(frozen=True)
class ESDPortRecord:
    """Typed ESD injection summary exposed on a run result.

    ``diagnostics`` describe the analytic (or measured) *target* waveform. The
    ``resampled`` samples are the charge-conserving projection onto the run time
    grid that the injected current reproduces (the injected current on the run
    grid).

    ``measured`` is the *measured* port record recovered from the run, if the
    run recorded terminal-port voltage/current for this port. It carries the RF
    terminal-port :class:`~witwin.maxwell.network.PortData` (frequency-domain
    voltage/current phasors) when present, enabling a target-vs-measured check.
    For the Phase-1 ideal-current injection path it is typically ``None``: the
    ESD source lowers to a volumetric additive current source and does not route
    through the RF terminal-port recorder, so no independent measured port
    current is synthesized. In that case the injected current on the run grid is
    the ``resampled`` record, and a measured *gap voltage* can be obtained by
    attaching a ``FieldTimeMonitor`` across the port gap (whose time derivative
    tracks the injected current for a capacitive gap). This is the documented
    stress-only limitation; a fully calibrated field-integrated H-contour
    measured port current is Phase-3 (source-impedance) scope.
    """

    name: str
    port_name: str
    diagnostics: ESDDiagnostics
    resampled: ESDResampledWaveform | None
    provenance: dict[str, Any]
    measured: Any | None = None

    @property
    def target_times(self) -> torch.Tensor | None:
        return None if self.resampled is None else self.resampled.times

    @property
    def target_currents(self) -> torch.Tensor | None:
        return None if self.resampled is None else self.resampled.currents


def _resolve_esd_sources(scene) -> tuple[ESDCurrentSource, ...]:
    return tuple(source for source in scene.sources if isinstance(source, ESDCurrentSource))


__all__ = [
    "ESD_CAPABILITY_LEVEL",
    "ESD_MODEL_VERSION",
    "ESDCurrentSource",
    "ESDDiagnostics",
    "ESDPortRecord",
    "ESDResampledWaveform",
    "ESDWaveform",
    "MeasuredWaveform",
]
