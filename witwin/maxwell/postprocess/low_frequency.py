"""Low-frequency smoothing of port and network spectra from finite FDTD windows."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from ..network import NetworkData, PortData


@dataclass(frozen=True)
class LowFrequencySmoothingSpec:
    """Replace under-sampled low-frequency spectral samples by a trusted polynomial fit.

    A frequency ``f`` is sampled for ``run_time * f`` periods. Samples with at
    least ``min_sampling_time`` and at most ``max_sampling_time`` periods are
    trusted and fitted by a complex least-squares polynomial of ``order`` in
    ``f``. Samples below ``min_sampling_time`` periods are discarded and
    replaced by that polynomial; samples above ``max_sampling_time`` are kept
    unchanged. With ``max_deviation`` set, an extrapolated value is pulled back
    radially so that it differs from the lowest-frequency trusted sample by at
    most ``max_deviation`` times that sample's magnitude.
    """

    min_sampling_time: float = 1.0
    max_sampling_time: float = 5.0
    order: int = 1
    max_deviation: float | None = 0.5
    kind: str = "low_frequency_smoothing"

    def __post_init__(self):
        low = float(self.min_sampling_time)
        high = float(self.max_sampling_time)
        if not (math.isfinite(low) and math.isfinite(high)) or low < 0.0:
            raise ValueError("Sampling times must be finite and non-negative.")
        if low >= high:
            raise ValueError("min_sampling_time must be smaller than max_sampling_time.")
        if isinstance(self.order, bool) or int(self.order) != self.order or not 0 <= int(self.order) <= 3:
            raise ValueError("order must be an integer between 0 and 3.")
        deviation = self.max_deviation
        if deviation is not None:
            deviation = float(deviation)
            if not math.isfinite(deviation) or deviation < 0.0:
                raise ValueError("max_deviation must be None or a finite non-negative fraction.")
        object.__setattr__(self, "min_sampling_time", low)
        object.__setattr__(self, "max_sampling_time", high)
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "max_deviation", deviation)
        object.__setattr__(self, "kind", "low_frequency_smoothing")


def _smooth_tensor(values: torch.Tensor, frequencies: torch.Tensor, spec, run_time: float, dim: int):
    periods = frequencies.detach().to(torch.float64).cpu() * run_time
    trusted = (periods >= spec.min_sampling_time) & (periods <= spec.max_sampling_time)
    replaced = periods < spec.min_sampling_time
    if not bool(replaced.any()):
        return values, 0
    count = int(trusted.sum())
    if count < spec.order + 1:
        raise ValueError(
            f"Low-frequency smoothing of order {spec.order} needs at least {spec.order + 1} "
            f"trusted samples between {spec.min_sampling_time} and {spec.max_sampling_time} periods; "
            f"found {count}."
        )
    moved = torch.movedim(values, dim, 0)
    trusted_index = torch.nonzero(trusted).reshape(-1).to(values.device)
    replaced_index = torch.nonzero(replaced).reshape(-1).to(values.device)
    real_dtype = torch.float64 if moved.dtype in (torch.complex128, torch.float64) else torch.float32
    scale = frequencies.to(device=values.device, dtype=real_dtype).index_select(0, trusted_index).max()
    normalized = frequencies.to(device=values.device, dtype=real_dtype) / scale
    powers = torch.arange(spec.order + 1, device=values.device, dtype=real_dtype)
    basis = normalized[:, None] ** powers[None, :]
    fit_matrix = torch.linalg.pinv(basis.index_select(0, trusted_index))
    samples = moved.index_select(0, trusted_index)
    flat = samples.reshape(samples.shape[0], -1)
    complex_dtype = flat.dtype if flat.is_complex() else (torch.complex128 if real_dtype == torch.float64 else torch.complex64)
    coefficients = fit_matrix.to(complex_dtype) @ flat.to(complex_dtype)
    extrapolated = basis.index_select(0, replaced_index).to(complex_dtype) @ coefficients
    if spec.max_deviation is not None:
        lowest = trusted_index[torch.argmin(frequencies.to(values.device).index_select(0, trusted_index))]
        reference = moved.index_select(0, lowest.reshape(1)).reshape(1, -1).to(complex_dtype)
        deviation = extrapolated - reference
        limit = spec.max_deviation * torch.abs(reference)
        magnitude = torch.abs(deviation)
        factor = torch.where(magnitude > limit, limit / torch.clamp(magnitude, min=torch.finfo(magnitude.dtype).tiny),
                             torch.ones_like(magnitude))
        extrapolated = reference + deviation * factor
    if not values.is_complex():
        extrapolated = extrapolated.real
    replacement = extrapolated.to(moved.dtype).reshape((replaced_index.numel(), *moved.shape[1:]))
    smoothed = moved.index_copy(0, replaced_index, replacement)
    return torch.movedim(smoothed, 0, dim), int(replaced_index.numel())


def smooth_low_frequency(data, spec: LowFrequencySmoothingSpec, *, run_time: float, frequencies=None, dim: int = -1):
    """Apply :class:`LowFrequencySmoothingSpec` to port, network or raw spectral data.

    ``run_time`` is the simulated time window in seconds, for example
    ``stats["dt"] * stats["steps_run"]`` of the producing run. ``NetworkData``
    is smoothed along its frequency axis and returned with a
    ``low_frequency_smoothing`` metadata record; ``PortData`` voltage and
    current are smoothed independently. A raw tensor requires ``frequencies``
    and its frequency ``dim``. The operation is differentiable and stays on
    the data's device.
    """

    if not isinstance(spec, LowFrequencySmoothingSpec):
        raise TypeError("spec must be a LowFrequencySmoothingSpec.")
    run_time = float(run_time)
    if not math.isfinite(run_time) or run_time <= 0.0:
        raise ValueError("run_time must be positive and finite.")
    record = {
        "min_sampling_time": spec.min_sampling_time,
        "max_sampling_time": spec.max_sampling_time,
        "order": spec.order,
        "max_deviation": spec.max_deviation,
        "run_time": run_time,
    }
    if isinstance(data, NetworkData):
        s, replaced = _smooth_tensor(data.s, data.frequencies, spec, run_time, 0)
        metadata = dict(data.metadata)
        metadata["low_frequency_smoothing"] = {**record, "replaced_samples": replaced}
        return replace(data, s=s, metadata=metadata)
    if isinstance(data, PortData):
        voltage, replaced = _smooth_tensor(data.voltage, data.frequencies, spec, run_time, -1)
        current, _ = _smooth_tensor(data.current, data.frequencies, spec, run_time, -1)
        metadata = dict(data.metadata)
        metadata["low_frequency_smoothing"] = {**record, "replaced_samples": replaced}
        return replace(data, voltage=voltage, current=current, metadata=metadata)
    if isinstance(data, torch.Tensor):
        if frequencies is None:
            raise ValueError("A raw spectral tensor requires its frequencies.")
        frequency_tensor = torch.as_tensor(frequencies, dtype=torch.float64)
        if frequency_tensor.ndim != 1 or frequency_tensor.numel() != data.shape[dim]:
            raise ValueError("frequencies must be one-dimensional and match data along dim.")
        return _smooth_tensor(data, frequency_tensor, spec, run_time, dim)[0]
    raise TypeError("smooth_low_frequency expects NetworkData, PortData or a torch.Tensor.")


__all__ = ["LowFrequencySmoothingSpec", "smooth_low_frequency"]
