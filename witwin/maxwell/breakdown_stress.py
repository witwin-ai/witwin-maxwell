"""Non-feedback dielectric-stress and component-rating accumulators.

Capability level: **stress-only**. These objects accumulate auditable field- and
port-stress statistics during a standard FDTD run and compare them against
user-supplied material and component rating envelopes. They perform NO feedback
into the field solve, do NOT switch conductivity, and do NOT model arc channels,
device latch-up, or failure probability. A recorded threshold exceedance is a
*stress indicator*, never a failure prediction.

The two primitives are:

* :class:`BreakdownStressAccumulator` -- a per-cell running reduction over a
  monitor region. Each step it consumes the energy-consistent cell-center
  magnitude ``|E|`` and updates, entirely on device with no host sync:

  - ``max_field``            : running ``max_t |E|``
  - ``exceedance_time``      : ``integral H(|E| - Ecrit) dt`` (H(0) = 1)
  - ``longest_exceedance``   : longest contiguous interval with ``|E| >= Ecrit``
  - ``damage_integral``      : ``integral (|E|/Ecrit)^k dt`` accrued only while
    exceeding (optional; enabled by a finite ``damage_exponent``)

* :class:`ComponentStressData` -- a typed reduction of a bound component's port
  time series ``V(t)``, ``I(t)`` into ``P(t) = V I``, cumulative dissipated
  energy ``integral P dt``, and an exceedance summary versus a
  :class:`ComponentRating` envelope.

The scalar ``|E|`` is colocated onto Yee cell centers with the same
energy-consistent averaging the solver uses for electric energy density (each
component averaged along the two axes on which it is node-staggered); no new
colocation convention is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch


BREAKDOWN_MODEL_VERSION = "breakdown-stress-1"
BREAKDOWN_CAPABILITY_LEVEL = "stress-only"
# Colocation convention recorded in provenance so downstream consumers never
# have to guess how the Yee components were reduced to a scalar cell magnitude.
COLOCATION_CONVENTION = "cell-center energy-consistent (component averaged along its two node-staggered axes)"

# Quantities a BreakdownMonitor may request. Kept as a stable vocabulary so the
# accumulator and result contract never silently accept an unsupported channel.
BREAKDOWN_QUANTITIES = ("electric_field", "exposure", "dissipated_energy", "damage")


# --------------------------------------------------------------------------- #
# Yee colocation                                                              #
# --------------------------------------------------------------------------- #


def colocate_electric_magnitude(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """Energy-consistent cell-center magnitude ``|E|`` from Yee components.

    The three spatial Yee components live on the last three axes ``(X, Y, Z)``:
    ``ex`` is on ``(x_half, y, z)``, ``ey`` on ``(x, y_half, z)`` and ``ez`` on
    ``(x, y, z_half)``. Each component is averaged along the two axes on which it
    sits on integer nodes so all three land on the common cell center
    ``(x_half, y_half, z_half)``; the returned magnitude is
    ``sqrt(Ex_c^2 + Ey_c^2 + Ez_c^2)`` with cell-center spatial shape
    ``(Ncx, Ncy, Ncz)``.

    Any number of leading batch axes (for example a time axis ``(T, X, Y, Z)``)
    is preserved, so the same reduction serves both the per-step observer and a
    recorded ``|E|(t)`` series consumed by the differentiable risk surrogate.

    On a spatially uniform field this reproduces the analytic magnitude exactly,
    matching ``u_E = (1/2) sum_c eps_c E_c^2`` under the same colocation.
    """

    if not (torch.is_tensor(ex) and torch.is_tensor(ey) and torch.is_tensor(ez)):
        raise TypeError("colocate_electric_magnitude requires torch tensors.")
    if not (ex.dim() >= 3 and ey.dim() >= 3 and ez.dim() >= 3):
        raise ValueError("colocate_electric_magnitude expects Yee components on the last three axes.")
    # Slices act on the trailing (X, Y, Z) axes; leading batch/time axes ride
    # through the ellipsis untouched.
    # Ex: average along Y and Z (the two node-staggered axes).
    ex_c = 0.25 * (
        ex[..., :, :-1, :-1] + ex[..., :, 1:, :-1] + ex[..., :, :-1, 1:] + ex[..., :, 1:, 1:]
    )
    # Ey: average along X and Z.
    ey_c = 0.25 * (
        ey[..., :-1, :, :-1] + ey[..., 1:, :, :-1] + ey[..., :-1, :, 1:] + ey[..., 1:, :, 1:]
    )
    # Ez: average along X and Y.
    ez_c = 0.25 * (
        ez[..., :-1, :-1, :] + ez[..., 1:, :-1, :] + ez[..., :-1, 1:, :] + ez[..., 1:, 1:, :]
    )
    if not (ex_c.shape == ey_c.shape == ez_c.shape):
        raise ValueError(
            "Colocated component shapes disagree: "
            f"Ex_c={tuple(ex_c.shape)}, Ey_c={tuple(ey_c.shape)}, Ez_c={tuple(ez_c.shape)}."
        )
    return torch.sqrt(ex_c * ex_c + ey_c * ey_c + ez_c * ez_c)


# --------------------------------------------------------------------------- #
# Field-stress accumulator                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class BreakdownStressAccumulator:
    """Per-cell running dielectric-stress reduction over a monitor region.

    All state tensors live on ``device`` with the region shape. The update rule
    is a pure device reduction; construct with :meth:`allocate` and drive with
    :meth:`update` once per FDTD step. ``occupancy`` (in ``[0, 1]``) is the
    target-material fraction of each cell: cells with zero occupancy never
    contribute to peak or exceedance statistics, and region aggregates weight
    each cell by ``occupancy * cell_volume``.
    """

    critical_field: float
    dt: float
    minimum_duration: float
    damage_exponent: float | None
    cell_volume: torch.Tensor
    occupancy: torch.Tensor
    max_field: torch.Tensor
    exceedance_time: torch.Tensor
    longest_exceedance: torch.Tensor
    _current_run: torch.Tensor
    damage_integral: torch.Tensor | None
    _occupied: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        shape: tuple[int, ...],
        critical_field: float,
        dt: float,
        minimum_duration: float = 0.0,
        damage_exponent: float | None = None,
        cell_volume: torch.Tensor | None = None,
        occupancy: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "BreakdownStressAccumulator":
        ecrit = float(critical_field)
        if not math.isfinite(ecrit) or ecrit <= 0.0:
            raise ValueError("critical_field must be a positive finite field strength.")
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be a positive finite time step.")
        min_duration = float(minimum_duration)
        if not math.isfinite(min_duration) or min_duration < 0.0:
            raise ValueError("minimum_duration must be a non-negative finite time.")
        exponent = None if damage_exponent is None else float(damage_exponent)
        if exponent is not None and (not math.isfinite(exponent) or exponent <= 0.0):
            raise ValueError("damage_exponent must be a positive finite exponent or None.")

        dev = torch.device(device)
        zeros = lambda: torch.zeros(shape, device=dev, dtype=dtype)
        if occupancy is None:
            occ = torch.ones(shape, device=dev, dtype=dtype)
        else:
            occ = occupancy.to(device=dev, dtype=dtype)
            if tuple(occ.shape) != tuple(shape):
                raise ValueError("occupancy must match the region shape.")
            if bool(torch.any(occ < 0.0)) or bool(torch.any(occ > 1.0)):
                raise ValueError("occupancy must lie in [0, 1].")
        if cell_volume is None:
            vol = torch.ones(shape, device=dev, dtype=dtype)
        else:
            vol = cell_volume.to(device=dev, dtype=dtype)
            if tuple(vol.shape) != tuple(shape):
                raise ValueError("cell_volume must match the region shape.")
        occupied = occ > 0.0
        return cls(
            critical_field=ecrit,
            dt=step,
            minimum_duration=min_duration,
            damage_exponent=exponent,
            cell_volume=vol,
            occupancy=occ,
            max_field=zeros(),
            exceedance_time=zeros(),
            longest_exceedance=zeros(),
            _current_run=zeros(),
            damage_integral=(zeros() if exponent is not None else None),
            _occupied=occupied,
        )

    def update(self, e_magnitude: torch.Tensor) -> None:
        """Fold one step's cell-center ``|E|`` into the running statistics.

        Pure device reduction: no ``.item()`` and no host synchronization. The
        exceedance test uses ``|E| >= Ecrit`` so a sample exactly at the
        threshold counts (Heaviside convention ``H(0) = 1``).
        """

        if tuple(e_magnitude.shape) != tuple(self.max_field.shape):
            raise ValueError(
                f"e_magnitude shape {tuple(e_magnitude.shape)} does not match the "
                f"region shape {tuple(self.max_field.shape)}."
            )
        field_mag = e_magnitude
        if field_mag.dtype != self.max_field.dtype:
            field_mag = field_mag.to(dtype=self.max_field.dtype)
        # Only occupied cells contribute to peak/exceedance statistics.
        occupied = self._occupied
        exceed = (field_mag >= self.critical_field) & occupied
        exceed_f = exceed.to(dtype=self.max_field.dtype)

        masked_field = torch.where(occupied, field_mag, torch.zeros_like(field_mag))
        torch.maximum(self.max_field, masked_field, out=self.max_field)
        self.exceedance_time.add_(exceed_f, alpha=self.dt)

        if self.damage_integral is not None:
            ratio = torch.clamp(field_mag / self.critical_field, min=0.0)
            increment = torch.pow(ratio, self.damage_exponent) * exceed_f
            self.damage_integral.add_(increment, alpha=self.dt)

        # Contiguous-run bookkeeping: extend the active run where exceeding,
        # reset it to zero elsewhere, then fold into the longest observed run.
        self._current_run = torch.where(
            exceed,
            self._current_run + self.dt,
            torch.zeros_like(self._current_run),
        )
        torch.maximum(self.longest_exceedance, self._current_run, out=self.longest_exceedance)

    def finalize(self, *, name: str = "", region_bounds=None, provenance_extra=None):
        """Reduce the running state into a typed :class:`BreakdownStressData`."""

        occupied = self._occupied
        weight = self.occupancy * self.cell_volume
        # Region peak restricted to occupied cells.
        if bool(torch.any(occupied)):
            peak_field = float(torch.max(self.max_field))
            flat_index = int(torch.argmax(self.max_field.reshape(-1)))
            peak_index = tuple(
                int(i) for i in torch.unravel_index(
                    torch.tensor(flat_index, device=self.max_field.device),
                    self.max_field.shape,
                )
            )
        else:
            peak_field = 0.0
            peak_index = tuple(0 for _ in self.max_field.shape)
        exceedance_duration = float(torch.max(self.exceedance_time)) if occupied.any() else 0.0
        longest_duration = float(torch.max(self.longest_exceedance)) if occupied.any() else 0.0
        # Cells whose longest contiguous exceedance reaches minimum_duration are
        # the "qualifying" sustained-stress cells.
        qualifying = (self.longest_exceedance >= self.minimum_duration) & (self.exceedance_time > 0.0) & occupied
        qualifying_count = int(torch.sum(qualifying))
        exceedance_volume_time = float(torch.sum(self.exceedance_time * weight))
        damage_volume = (
            float(torch.sum(self.damage_integral * weight))
            if self.damage_integral is not None
            else None
        )
        provenance = {
            "kind": "breakdown_stress",
            "name": str(name),
            "capability_level": BREAKDOWN_CAPABILITY_LEVEL,
            "model_version": BREAKDOWN_MODEL_VERSION,
            "colocation": COLOCATION_CONVENTION,
            "critical_field": self.critical_field,
            "minimum_duration": self.minimum_duration,
            "damage_exponent": self.damage_exponent,
            "exceedance_convention": "|E| >= Ecrit (Heaviside H(0)=1)",
            "occupancy_policy": "only target-material occupancy counts; region aggregates weighted by occupancy*cell_volume",
            "dt": self.dt,
            "units": {"field": "V/m", "time": "s", "volume": "m^3"},
        }
        if provenance_extra:
            provenance.update(dict(provenance_extra))
        return BreakdownStressData(
            name=str(name),
            peak_field=peak_field,
            peak_index=peak_index,
            peak_time=None,
            exceedance_duration=exceedance_duration,
            longest_exceedance_duration=longest_duration,
            qualifying_cell_count=qualifying_count,
            exceedance_volume_time=exceedance_volume_time,
            damage_volume=damage_volume,
            max_field_map=self.max_field,
            exceedance_time_map=self.exceedance_time,
            longest_exceedance_map=self.longest_exceedance,
            damage_map=self.damage_integral,
            qualifying_mask=qualifying,
            occupancy=self.occupancy,
            region_bounds=region_bounds,
            provenance=provenance,
        )


@dataclass(frozen=True)
class BreakdownStressData:
    """Typed non-feedback dielectric-stress result (per-cell maps stay on device).

    Capability level: stress-only. ``peak_field`` is the region maximum of the
    per-cell running ``max|E|``; ``locations``/``qualifying_mask`` identify cells
    whose longest contiguous exceedance reached ``minimum_duration``. Nothing
    here predicts breakdown; it quantifies stress against a declared threshold.
    """

    name: str
    peak_field: float
    peak_index: tuple[int, ...]
    peak_time: float | None
    exceedance_duration: float
    longest_exceedance_duration: float
    qualifying_cell_count: int
    exceedance_volume_time: float
    damage_volume: float | None
    max_field_map: torch.Tensor
    exceedance_time_map: torch.Tensor
    longest_exceedance_map: torch.Tensor
    damage_map: torch.Tensor | None
    qualifying_mask: torch.Tensor
    occupancy: torch.Tensor
    region_bounds: Any
    provenance: Mapping[str, Any]

    @property
    def capability_level(self) -> str:
        return BREAKDOWN_CAPABILITY_LEVEL

    def locations(self) -> dict[str, torch.Tensor]:
        """Indices (and cell count) of the qualifying sustained-stress cells."""

        idx = torch.nonzero(self.qualifying_mask, as_tuple=False)
        return {"indices": idx, "count": int(idx.shape[0])}


# --------------------------------------------------------------------------- #
# Component rating / stress                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComponentRating:
    """Declared absolute-maximum envelope for a rated component.

    Capability level: stress-only. Each populated field is an absolute-maximum
    limit used only for auditable exceedance reporting; ``None`` disables that
    channel. ``model`` names the rating source/version and flows into provenance.
    """

    voltage: float | None = None
    current: float | None = None
    energy: float | None = None
    pulse_width: float | None = None
    model: str | None = None

    def __post_init__(self):
        for name in ("voltage", "current", "energy", "pulse_width"):
            value = getattr(self, name)
            if value is None:
                continue
            resolved = float(value)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"ComponentRating {name} must be positive and finite or None.")
            object.__setattr__(self, name, resolved)
        if not any(
            getattr(self, name) is not None
            for name in ("voltage", "current", "energy", "pulse_width")
        ):
            raise ValueError("ComponentRating must declare at least one rated limit.")
        object.__setattr__(self, "model", None if self.model is None else str(self.model))

    def as_dict(self) -> dict[str, Any]:
        return {
            "voltage": self.voltage,
            "current": self.current,
            "energy": self.energy,
            "pulse_width": self.pulse_width,
            "model": self.model,
        }


def _cumulative_energy(time: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """Cumulative ``integral P dt`` via the trapezoidal rule (device tensor)."""

    if time.numel() < 2:
        return torch.zeros_like(power)
    dt = time[1:] - time[:-1]
    segment = 0.5 * (power[1:] + power[:-1]) * dt
    running = torch.cumsum(segment, dim=0)
    zero = torch.zeros((1,), device=power.device, dtype=running.dtype)
    return torch.cat((zero, running), dim=0)


@dataclass(frozen=True)
class ComponentStressData:
    """Typed component port-stress record and rating exceedance summary.

    Capability level: stress-only. ``power`` is the instantaneous ``V I`` and
    ``cumulative_energy`` is ``integral P dt``. The exceedance summary compares
    absolute peaks and total energy against the :class:`ComponentRating`
    envelope; it flags stress, not device failure.
    """

    name: str
    port_name: str
    time: torch.Tensor
    voltage: torch.Tensor
    current: torch.Tensor
    power: torch.Tensor
    cumulative_energy: torch.Tensor
    peak_voltage: float
    peak_current: float
    peak_power: float
    total_energy: float
    measured_pulse_width: float
    exceedance: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def capability_level(self) -> str:
        return BREAKDOWN_CAPABILITY_LEVEL

    @property
    def any_exceeded(self) -> bool:
        return any(entry["exceeded"] for entry in self.exceedance.values())

    @classmethod
    def from_time_series(
        cls,
        time: torch.Tensor,
        voltage: torch.Tensor,
        current: torch.Tensor,
        rating: ComponentRating,
        *,
        name: str = "",
        port_name: str = "",
        provenance_extra=None,
    ) -> "ComponentStressData":
        if not (torch.is_tensor(time) and torch.is_tensor(voltage) and torch.is_tensor(current)):
            raise TypeError("time, voltage, and current must be torch tensors.")
        if not isinstance(rating, ComponentRating):
            raise TypeError("rating must be a ComponentRating.")
        t = time.reshape(-1)
        v = voltage.reshape(-1)
        i = current.reshape(-1)
        if not (t.shape == v.shape == i.shape):
            raise ValueError("time, voltage, and current must share the same length.")
        if t.numel() < 1:
            raise ValueError("component stress requires at least one sample.")
        if t.numel() >= 2 and bool(torch.any(t[1:] - t[:-1] <= 0.0)):
            raise ValueError("time samples must be strictly increasing.")

        power = v * i
        cumulative = _cumulative_energy(t, power)
        peak_voltage = float(torch.max(torch.abs(v)))
        peak_current = float(torch.max(torch.abs(i)))
        peak_power = float(torch.max(power)) if power.numel() else 0.0
        total_energy = float(cumulative[-1]) if cumulative.numel() else 0.0
        measured_pulse_width = _half_peak_width(t, power)

        exceedance: dict[str, Any] = {}
        _add_exceedance(exceedance, "voltage", peak_voltage, rating.voltage)
        _add_exceedance(exceedance, "current", peak_current, rating.current)
        _add_exceedance(exceedance, "energy", total_energy, rating.energy)
        _add_exceedance(exceedance, "pulse_width", measured_pulse_width, rating.pulse_width)

        provenance = {
            "kind": "component_stress",
            "name": str(name),
            "port": str(port_name),
            "capability_level": BREAKDOWN_CAPABILITY_LEVEL,
            "model_version": BREAKDOWN_MODEL_VERSION,
            "rating": rating.as_dict(),
            "power_definition": "P(t) = V(t) * I(t)",
            "energy_definition": "integral P dt (trapezoidal)",
            "pulse_width_definition": "full width where P >= 0.5 * peak(P)",
            "units": {"voltage": "V", "current": "A", "power": "W", "energy": "J", "time": "s"},
        }
        if provenance_extra:
            provenance.update(dict(provenance_extra))
        return cls(
            name=str(name),
            port_name=str(port_name),
            time=t,
            voltage=v,
            current=i,
            power=power,
            cumulative_energy=cumulative,
            peak_voltage=peak_voltage,
            peak_current=peak_current,
            peak_power=peak_power,
            total_energy=total_energy,
            measured_pulse_width=measured_pulse_width,
            exceedance=exceedance,
            provenance=provenance,
        )


def _add_exceedance(summary: dict[str, Any], key: str, measured: float, limit: float | None) -> None:
    if limit is None:
        summary[key] = {"rated": None, "measured": measured, "exceeded": False, "margin": None}
        return
    margin = measured - limit
    summary[key] = {
        "rated": limit,
        "measured": measured,
        "exceeded": bool(measured > limit),
        "margin": margin,
    }


def _half_peak_width(time: torch.Tensor, power: torch.Tensor) -> float:
    """Total time span where ``P >= 0.5 * peak(P)`` (a coarse pulse-width proxy)."""

    if power.numel() < 2:
        return 0.0
    peak = float(torch.max(power))
    if peak <= 0.0:
        return 0.0
    above = power >= 0.5 * peak
    dt = time[1:] - time[:-1]
    # Weight each interval by the fraction of its endpoints above the half level.
    seg_frac = 0.5 * (above[1:].to(power.dtype) + above[:-1].to(power.dtype))
    return float(torch.sum(seg_frac * dt))


__all__ = [
    "BREAKDOWN_CAPABILITY_LEVEL",
    "BREAKDOWN_MODEL_VERSION",
    "BREAKDOWN_QUANTITIES",
    "COLOCATION_CONVENTION",
    "BreakdownStressAccumulator",
    "BreakdownStressData",
    "ComponentRating",
    "ComponentStressData",
    "colocate_electric_magnitude",
]
