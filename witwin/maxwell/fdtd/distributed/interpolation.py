from __future__ import annotations

import numpy as np


def ideal_axis_weight(coords64: np.ndarray, position: float, index: int) -> float:
    """Match the native point-source compiler's float32 linear-axis weight."""

    coords = np.asarray(coords64, dtype=np.float32)
    probe = np.float32(position)
    upper = int(np.searchsorted(coords, probe, side="left"))
    if upper <= 0:
        return 1.0 if index == 0 else 0.0
    if upper >= coords.size:
        return 1.0 if index == coords.size - 1 else 0.0

    lower = upper - 1
    lower_coord = float(coords[lower])
    upper_coord = float(coords[upper])
    span = upper_coord - lower_coord
    if span <= np.finfo(np.float32).eps:
        return 1.0 if index == lower else 0.0
    fraction = (float(position) - lower_coord) / span
    if fraction <= 1.0e-12:
        return 1.0 if index == lower else 0.0
    if fraction >= 1.0 - 1.0e-12:
        return 1.0 if index == upper else 0.0
    if index == lower:
        return 1.0 - fraction
    if index == upper:
        return fraction
    return 0.0


__all__ = ["ideal_axis_weight"]
