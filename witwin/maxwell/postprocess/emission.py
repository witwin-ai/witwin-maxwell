"""Dipole emission postprocessing: Purcell factor / local density of states.

The power a point dipole delivers to the field, ``P = -(1/2) Re(conj(J) . E)``,
is reported per frequency by a :class:`DipoleEmissionMonitor`. The Purcell factor
is the ratio of that power in the structured environment to the power the same
dipole delivers in vacuum. Vacuum normalization (rather than an analytic
free-space formula) is used because the discrete Yee-grid effective source
volume, which sets the absolute scale of ``power_delivered``, has no reliable
closed form; the ratio cancels it exactly when both runs share the grid, the
source waveform, and the sampling schedule.
"""

from __future__ import annotations

import torch

from ..results import FluxData


def _dipole_power(source, name: str) -> FluxData:
    data = source.monitor(name) if hasattr(source, "monitor") else source
    if not isinstance(data, FluxData) or "current_spectrum" not in data.attrs:
        raise ValueError(f"Monitor {name!r} is not a DipoleEmissionMonitor result.")
    return data


def _as_real_tensor(power) -> torch.Tensor:
    # Keep the device and the autograd graph: the Purcell factor of a trainable
    # run must back-propagate into the structured run's delivered power.
    return torch.as_tensor(power).to(dtype=torch.float64)


def purcell_factor(structured, vacuum, name: str) -> dict:
    """Return the Purcell factor of a dipole from structured and vacuum runs.

    ``structured`` and ``vacuum`` may each be a :class:`Result` or the
    ``FluxData`` its ``DipoleEmissionMonitor`` returns. Both runs must share the grid,
    the dipole waveform, and the frequency sampling so that the discrete source
    normalization cancels in the ratio.

    The factor and both delivered-power spectra are float64 torch tensors on the
    device of the structured run's payload, so gradients flow through them. A
    single-frequency factor is a 0-d tensor.
    """

    structured_data = _dipole_power(structured, name)
    vacuum_data = _dipole_power(vacuum, name)

    structured_power = _as_real_tensor(structured_data.values)
    vacuum_power = _as_real_tensor(vacuum_data.values).to(device=structured_power.device)
    if structured_power.shape != vacuum_power.shape:
        raise ValueError(
            "Structured and vacuum dipole runs must share the frequency sampling; "
            f"got shapes {tuple(structured_power.shape)} and {tuple(vacuum_power.shape)}."
        )

    factor = structured_power / vacuum_power
    frequencies = tuple(structured_data.coords["f"].tolist()) if "f" in structured_data.coords else ()
    scalar = factor.numel() == 1
    return {
        "kind": "dipole_emission",
        "purcell_factor": factor.reshape(()) if scalar else factor,
        "power_delivered": structured_power,
        "power_delivered_vacuum": vacuum_power,
        "frequencies": frequencies,
    }
