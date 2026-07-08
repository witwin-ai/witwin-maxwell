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

import numpy as np
import torch


def _dipole_payload(source, name: str) -> dict:
    if hasattr(source, "monitor"):
        payload = source.monitor(name)
    else:
        payload = source
    if not isinstance(payload, dict) or payload.get("monitor_type") != "dipole_emission":
        raise ValueError(f"Monitor {name!r} is not a DipoleEmissionMonitor result.")
    power = payload.get("power_delivered")
    if power is None:
        raise ValueError(f"DipoleEmissionMonitor {name!r} has no delivered-power samples.")
    return payload


def _as_real_array(power):
    if isinstance(power, torch.Tensor):
        return power.detach().to(dtype=torch.float64).cpu().numpy()
    return np.asarray(power, dtype=np.float64)


def purcell_factor(structured, vacuum, name: str) -> dict:
    """Return the Purcell factor of a dipole from structured and vacuum runs.

    ``structured`` and ``vacuum`` may each be a :class:`Result` or an already
    extracted ``DipoleEmissionMonitor`` payload. Both runs must share the grid,
    the dipole waveform, and the frequency sampling so that the discrete source
    normalization cancels in the ratio.
    """

    structured_payload = _dipole_payload(structured, name)
    vacuum_payload = _dipole_payload(vacuum, name)

    structured_power = _as_real_array(structured_payload["power_delivered"])
    vacuum_power = _as_real_array(vacuum_payload["power_delivered"])
    if structured_power.shape != vacuum_power.shape:
        raise ValueError(
            "Structured and vacuum dipole runs must share the frequency sampling; "
            f"got shapes {structured_power.shape} and {vacuum_power.shape}."
        )

    factor = structured_power / vacuum_power
    frequencies = tuple(float(freq) for freq in structured_payload.get("frequencies", ()))
    scalar = factor.ndim == 0 or factor.size == 1
    return {
        "kind": "dipole_emission",
        "purcell_factor": float(factor.reshape(-1)[0]) if scalar else factor,
        "power_delivered": structured_power,
        "power_delivered_vacuum": vacuum_power,
        "frequencies": frequencies,
    }
