from __future__ import annotations

import math


def solve_numerical_wavenumber(solver, direction, delta_attr_map) -> float:
    omega = float(solver.source_omega)
    if omega <= 0.0:
        return 0.0

    target = math.sin(0.5 * omega * float(solver.dt)) ** 2 / (float(solver.c) ** 2 * float(solver.dt) ** 2)
    if target <= 0.0:
        return omega / float(solver.c)

    axis_limits = []
    for axis, component in zip("xyz", direction):
        abs_component = abs(float(component))
        if abs_component <= 1e-12:
            continue
        delta = float(getattr(solver, delta_attr_map[axis]))
        axis_limits.append(2.0 * math.pi / max(abs_component * delta, 1e-30))
    if not axis_limits:
        return omega / float(solver.c)

    def dispersion_residual(k_mag):
        total = 0.0
        for axis, component in zip("xyz", direction):
            abs_component = abs(float(component))
            if abs_component <= 1e-12:
                continue
            delta = float(getattr(solver, delta_attr_map[axis]))
            total += math.sin(0.5 * k_mag * abs_component * delta) ** 2 / (delta * delta)
        return total - target

    low = 0.0
    high = min(axis_limits)
    for _ in range(80):
        mid = 0.5 * (low + high)
        if dispersion_residual(mid) >= 0.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)
