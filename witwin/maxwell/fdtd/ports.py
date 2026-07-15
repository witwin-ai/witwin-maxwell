from __future__ import annotations

from typing import Literal

import torch


PhasorNormalization = Literal["peak", "rms", "none"]


class PortDFTAccumulator:
    """Accumulate staggered scalar port samples with the ``exp(+iwt)`` convention."""

    def __init__(self, frequencies: torch.Tensor) -> None:
        if not isinstance(frequencies, torch.Tensor):
            raise TypeError("frequencies must be a torch.Tensor.")
        if frequencies.ndim != 1 or frequencies.shape[0] == 0:
            raise ValueError("frequencies must have shape [F] with F > 0.")
        if frequencies.dtype not in (torch.float32, torch.float64):
            raise TypeError("frequencies must use torch.float32 or torch.float64.")

        self.frequencies = frequencies
        self._real_dtype = frequencies.dtype
        zeros = torch.zeros_like(frequencies)
        self._voltage_sum = torch.complex(zeros, zeros)
        self._current_sum = torch.complex(zeros, zeros)
        self._window_weight_sum = torch.zeros(
            (),
            dtype=self._real_dtype,
            device=frequencies.device,
        )
        self._sample_count = 0

    def _scalar(
        self,
        value: torch.Tensor | float | complex,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar tensor.")
            if value.device != self.frequencies.device:
                raise ValueError(f"{name} must be on the same device as frequencies.")
            return value.to(dtype=dtype)
        return torch.as_tensor(
            value,
            dtype=dtype,
            device=self.frequencies.device,
        )

    def _phase(self, sample_time: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * torch.pi * self.frequencies * sample_time
        return torch.complex(torch.cos(angle), torch.sin(angle))

    def accumulate(
        self,
        voltage_sample: torch.Tensor | float | complex,
        current_sample: torch.Tensor | float | complex,
        *,
        electric_sample_time: torch.Tensor | float,
        magnetic_sample_time: torch.Tensor | float,
        window_weight: torch.Tensor | float = 1.0,
    ) -> None:
        """Accumulate one Yee step using each field's physical sample time."""

        voltage = self._scalar(
            voltage_sample,
            name="voltage_sample",
            dtype=self._voltage_sum.dtype,
        )
        current = self._scalar(
            current_sample,
            name="current_sample",
            dtype=self._current_sum.dtype,
        )
        electric_time = self._scalar(
            electric_sample_time,
            name="electric_sample_time",
            dtype=self._real_dtype,
        )
        magnetic_time = self._scalar(
            magnetic_sample_time,
            name="magnetic_sample_time",
            dtype=self._real_dtype,
        )
        weight = self._scalar(
            window_weight,
            name="window_weight",
            dtype=self._real_dtype,
        )

        self._voltage_sum = self._voltage_sum + weight * voltage * self._phase(electric_time)
        self._current_sum = self._current_sum + weight * current * self._phase(magnetic_time)
        self._window_weight_sum = self._window_weight_sum + weight
        self._sample_count += 1

    def phasors(
        self,
        *,
        normalization: PhasorNormalization = "peak",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return voltage and current phasors, each with explicit shape ``[F]``."""

        if normalization == "none":
            return self._voltage_sum, self._current_sum
        if normalization not in ("peak", "rms"):
            raise ValueError("normalization must be 'peak', 'rms', or 'none'.")
        if self._sample_count == 0:
            raise RuntimeError("Cannot normalize an accumulator with no samples.")

        amplitude_scale = torch.as_tensor(
            2.0,
            dtype=self._real_dtype,
            device=self.frequencies.device,
        )
        if normalization == "rms":
            amplitude_scale = torch.sqrt(amplitude_scale)
        scale = amplitude_scale / self._window_weight_sum
        return self._voltage_sum * scale, self._current_sum * scale
