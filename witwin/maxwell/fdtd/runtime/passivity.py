"""Topology-dependent CPML growth detection in the main single-device step loop.

The envelope fit ports bloch_lumped_acceptance/common.py::assert_passivity.
This is an empirical growth detector, not a discrete passivity certificate.

Step-loop hook contract (the main loop today; the core path in the single-path change):

- At prepare/solve start: ``state = make_passivity_state(solver)``; None means no guard.
- Every ``SAMPLE_INTERVAL`` completed steps: ``sample_passivity(state, step, energy)``,
  where ``energy`` is a float64 device scalar of total E+H storage (``total_energy``
  or an equivalent fused reduction). E-only energy is not acceptable, because a
  lossless resonator exchanges energy between E and H. ``sample_passivity`` is
  device-only except for one ``check_passivity(history.cpu(), state)`` per completed
  window (every WINDOW_SAMPLES * SAMPLE_INTERVAL steps), so a loop may align that
  host copy with its existing shutoff-check synchronization.
- Checkpoints carry the state through ``capture_passivity``/``preflight_passivity``.
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy

import torch

from ..boundary.runtime import closed_transverse_pml, complex_fixed_k

SAMPLE_INTERVAL = 32
WINDOW_SAMPLES = 64
FIT_WINDOWS = 16
SLOPE_LIMIT = 4e-7
ENERGY_FLOOR = 1e-10
# Counted by total_energy, but sampling them every SAMPLE_INTERVAL steps costs
# 39-133% of a step on guarded copper/sheet scenes, and sparser sampling
# false-trips (docs/reference/fdtd-cpml.md). Guarded scenes with these owners
# are reported as skipped until the sampler is graph-captured.
DEFERRED_STORAGE = (("_copper_foils", "CopperFoil"), ("_thin_sheets", "thin sheets"))


class FDTDPassivityError(RuntimeError):
    """The late CPML energy envelope exceeds the accepted growth bound."""


def check_passivity(history, state) -> None:
    """Check complete (step, maximum, peak-to-date) windows on the CPU.

    Non-finite energy raises in every topology. Measured slow growth raises
    for real fields and records one warning in state for complex fixed-k.
    Only the last 16 windows count. The peak includes source excitation. Decay
    and bounded beating need not approach zero. The floor is relative energy,
    not field amplitude. A partial window is never fitted.
    """
    values = torch.as_tensor(history, dtype=torch.float64, device="cpu")[-FIT_WINDOWS:]
    if len(values) < FIT_WINDOWS:
        return
    topology = state["topology"]
    if not torch.isfinite(values).all():
        # A blow-up, not measured slow growth: fail closed in every topology.
        raise FDTDPassivityError(
            f"CPML passivity guard: {topology}; non-finite field energy in the fitted windows. "
            "The run has diverged. See docs/reference/fdtd-cpml.md."
        )
    if topology == "complex fixed-k Bloch" and state.get("warning") is not None:
        return
    x, energy = values[:, 0], values[:, 1]
    peak = values[:, 2].max()
    if energy[-1] <= ENERGY_FLOOR * peak or energy[-1] <= energy[0]:
        return
    y = energy.clamp_min(torch.finfo(torch.float64).tiny).log()
    centered = x - x.mean()
    variance = centered.square().sum()
    slope = (centered * (y - y.mean())).sum() / variance
    residual = y - y.mean() - slope * centered
    stderr = (residual.square().sum() / (len(values) - 2) / variance).sqrt()
    if slope + 3 * stderr <= SLOPE_LIMIT:
        return
    if topology == "complex fixed-k Bloch":
        start, end = int(x[0]), int(x[-1])
        state["warning"] = dict(slope=float(slope), stderr=float(stderr),
                                window=(start, end), step=end)
        warnings.warn(
            f"CPML passivity guard: {topology}; measured log-energy slope={float(slope):.9g} "
            f"per step, stderr={float(stderr):.9g}; window steps [{start}, {end}]. "
            f"Growth criterion exceeded (limit {SLOPE_LIMIT:g} per step); continuing. "
            "Use CW per frequency or short pulsed records for complex fixed-k runs. "
            "See docs/reference/fdtd-cpml.md.",
            RuntimeWarning, stacklevel=2,
        )
        return
    workaround = 'Use absorber="auto" or absorber="absorber" with >= 40 PML layers.'
    raise FDTDPassivityError(
        f"CPML passivity guard: {topology}; measured log-energy slope={float(slope):.9g} "
        f"per step, stderr={float(stderr):.9g}; slope + 3 stderr exceeds {SLOPE_LIMIT:g} "
        f"per step. {workaround} Growth slower than 4e-7 per step "
        "(about 1.5x energy over 1e6 steps) is not detected. "
        "See docs/reference/fdtd-cpml.md."
    )


def uncounted_storage(solver) -> list[str]:
    """Runtime state the guard does not sample: uncounted or DEFERRED_STORAGE.

    In a passive run, energy can move between such state and the Yee fields, so
    a field-energy envelope could show growth that is not CPML instability.
    """
    scene = solver.scene
    labels = [label for attr, label in (
        ("thin_wires", "thin wires"), ("networks", "embedded networks"),
    ) if getattr(scene, attr, ())]
    labels += [label for attr, label in DEFERRED_STORAGE + (
        ("_coaxial_element_runtimes", "coaxial elements"),
        ("surface_impedance_enabled", "SIBC"),
        ("dispersive_enabled", "dispersive/ADE media"),
        ("gyromagnetic_enabled", "gyromagnetic media"),
    ) if getattr(solver, attr, None)]
    return labels


def make_passivity_state(solver):
    solver.passivity_guard_status = "inactive: topology or absorber not guarded"
    solver.passivity_guard_warning = None
    if getattr(solver, "active_absorber_type", "none") not in ("cpml", "stablepml"):
        return None
    fixed_k = complex_fixed_k(solver)
    if not fixed_k and (getattr(solver, "complex_fields_enabled", False)
                        or not closed_transverse_pml(solver.scene)):
        return None
    uncounted = uncounted_storage(solver)
    if uncounted:
        # Not detected: a guarded topology whose energy functional is incomplete.
        reason = "energy storage not counted (" + ", ".join(uncounted) + ")"
        solver.passivity_guard_status = "skipped: " + reason
        warnings.warn(
            f"passivity guard skipped in a topology with known CPML growth: {reason}",
            RuntimeWarning, stacklevel=2)
        return None
    solver.passivity_guard_status = "active"
    plan = getattr(solver, "_time_plan", {})
    # The compiled settling time is delay+6 sigma for Gaussian pulses,
    # delay+4/f for Ricker, and the last tabulated time for custom pulses.
    # Callable support is unknown: its configured transient is the only
    # available start criterion, also used for ongoing/CW sources.
    transient = math.ceil(max(
        plan.get("transient_cycles", 15) / plan.get("reference_frequency", solver.source_frequency),
        10 * max(hi - lo for lo, hi in solver.scene.domain.bounds) / solver.c,
    ) / solver.dt)
    start = (max(transient, plan.get("latest_source_settling_step", 0))
             if not plan.get("shutoff_allowed", True) else
             plan.get("latest_source_settling_step", 0))
    # Keep native widths separate from the sheet functional's half dual cells
    # at domain endpoints. Either owner may be present in the same scene.
    solver._passivity_widths = {
        (coordinate, staggering): torch.as_tensor(
            getattr(solver.scene, f"d{coordinate}_{staggering}64"), device=solver.device,
        ) for coordinate in "xyz" for staggering in ("primal", "dual")
    }
    for runtime in getattr(solver, "_thin_sheets", ()):
        if not hasattr(runtime, "energy"):
            runtime.ensure_energy(solver)
    return dict(
        topology="complex fixed-k Bloch" if fixed_k else "real-field closed-transverse PML",
        start_step=int(start), sample_count=0, window_count=0, warning=None,
        history=torch.zeros((FIT_WINDOWS, 3), dtype=torch.float64, device=solver.device),
        peak=torch.zeros((), dtype=torch.float64, device=solver.device),
        window_max=torch.zeros((), dtype=torch.float64, device=solver.device),
    )


def total_energy(solver):
    """Completed-step field, copper, sheet and lumped/circuit storage in joules.

    After step n the native pairing is E^n, H^(n-1/2), with copper
    modes at n (not their dissipative midpoint). Without sheets we use the
    plain Yee sum, replacing foil edges by both half-cell electric traces;
    with sheets we use thin_sheet_energy's modified Newmark
    functional, including split masses, chord E^n and E^(n-1), recovered
    sheet flux and modal storage at n. Its native term is the Yee sum minus
    dt/2 <H^(n-1/2), curl E^n>; the patch uses the corresponding integer-time
    Hhat and modified electric mass. No previous native field bank is needed.
    With both conductor owners, also replace the foil's native magnetic cross
    term by its two-trace curl: add 1/2 <mu0 V H^(n-1/2), delta_H(E^n)>,
    where delta_H is the foil magnetic step correction evaluated at the
    current left/right traces (not the begin-step old_e snapshot).
    Keep all reductions on device, including the split complex field bank.
    Width views are separable, avoiding persistent dense volume buffers.
    CPML psi has no known positive storage functional and is excluded.
    """
    sheets = getattr(solver, "_thin_sheets", ())
    if sheets:
        from ..thin_sheet import thin_sheet_energy_device

        energy = thin_sheet_energy_device(solver, modified=True)
        widths = solver._thin_sheet_energy_widths
    else:
        energy = _yee_energy(solver)
        widths = solver._passivity_widths
    for entry in getattr(solver, "_copper_foils", ()):
        # Complex runs have separate real and imaginary entries.
        name = entry["electric_name"]
        left = getattr(solver, name)[entry["index"]].double()
        native_mass = getattr(solver, "eps_" + name.removesuffix("_imag"))[entry["index"]].double()
        native_mass = native_mass * _yee_volume_at(name, entry["index"], widths)
        energy = energy - .5*(native_mass*left.square()).sum()
        energy = energy + .5*(entry["area"] * (entry["capacity"][0]*left.square()
                                               + entry["capacity"][1]*entry["right_e"].double().square())).sum()
        energy = energy + .5 * (entry["state"].double().square()
                                * entry["area"] * entry["admittance_scale"]).sum()
        if sheets:
            energy = energy + _foil_modified_cross(solver, entry, left, widths)
    from ..checkpoint import iter_lumped_state_specs

    loads = {id(spec[1]): spec[1] for spec in iter_lumped_state_specs(solver)}
    for load in loads.values():
        for suffix in ("", "_imag") if solver.complex_fields_enabled else ("",):
            energy = energy + .5 * (
                load.inductance * getattr(load, "inductor_current" + suffix).double().square()
                + load.capacitance * getattr(load, "capacitor_voltage" + suffix).double().square()
            ).sum()
    for runtime in getattr(solver, "_circuit_runtimes", ()):
        energy = energy + runtime._stored_energy(runtime.state).double()
    return energy.detach()


def _yee_volume_at(name, indices, widths):
    """Separable volume at selected native E/H locations, entirely on device."""
    volume = 1.
    for axis, coordinate in enumerate("xyz"):
        primal = (coordinate == name[1].lower()) == name.startswith("E")
        volume = volume * widths[coordinate, "primal" if primal else "dual"][indices[axis]]
    return volume


def _foil_modified_cross(solver, entry, left, widths):
    """Replace the native curl by the split-trace curl in the modified energy.

    The existing magnetic stencil supplies delta_H = -dt/mu0 delta_curl(E).
    Contract its sparse support with the *current* staggered H bank; no field
    is mutated, and overlapping stencil destinations contribute linearly.
    """
    difference = entry["right_e"].double() - left
    name, indices = entry["magnetic_name"], entry["high_h"]
    mass = solver.mu0 * _yee_volume_at(name, indices, widths)
    cross = .5 * (mass * getattr(solver, name)[indices].double()
                  * entry["high_h_gain"] * difference).sum()
    name = entry["normal_magnetic_name"]
    field = getattr(solver, name)
    weighted = difference * entry["right_width"] / (entry["left_width"] + entry["right_width"])
    for side in ("low", "high"):
        offsets = entry[side + "_normal_offsets"]
        indices = (offsets // (field.shape[1]*field.shape[2]),
                   (offsets // field.shape[2]) % field.shape[1], offsets % field.shape[2])
        mass = solver.mu0 * _yee_volume_at(name, indices, widths)
        delta_h = entry[side + "_normal_gain"] * weighted[entry[side + "_sources"]]
        cross = cross + .5 * (mass * field.reshape(-1)[offsets].double() * delta_h).sum()
    return cross


def _yee_energy(solver, *, magnetic_mass=None, widths=None):
    """Separable Yee-volume reduction shared with the sheet functional."""
    energy = torch.zeros((), dtype=torch.float64, device=solver.device)
    if widths is None:
        widths = solver._passivity_widths
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        square = getattr(solver, name).double().square()
        if solver.complex_fields_enabled:
            square = square + getattr(solver, name + "_imag").double().square()
        mass = (magnetic_mass if name.startswith("H") and magnetic_mass is not None else
                getattr(solver, ("eps_" if name.startswith("E") else "mu_") + name))
        weighted = square * mass
        for axis, coordinate in enumerate("xyz"):
            primal = (coordinate == name[1].lower()) == name.startswith("E")
            axis_widths = widths[coordinate, "primal" if primal else "dual"]
            shape = [1] * square.ndim
            shape[axis] = axis_widths.numel()
            weighted = weighted * axis_widths.view(shape)
        energy = energy + .5 * weighted.sum()
    return energy


def sample_passivity(state, step, energy):
    """Accumulate device samples; exactly one device-to-host copy per window."""
    state["peak"].copy_(torch.maximum(state["peak"], energy))
    if step < state["start_step"]:
        return
    state["window_max"].copy_(torch.maximum(state["window_max"], energy))
    state["sample_count"] += 1
    if state["sample_count"] != WINDOW_SAMPLES:
        return
    history = state["history"]
    history.copy_(history.roll(-1, dims=0))
    history[-1, 0] = step
    history[-1, 1].copy_(state["window_max"])
    history[-1, 2].copy_(state["peak"])
    state["sample_count"] = 0
    state["window_count"] += 1
    state["window_max"].zero_()
    if state["window_count"] >= FIT_WINDOWS:
        check_passivity(history.cpu(), state)


def capture_passivity(solver):
    state = getattr(solver, "_passivity_state", None)
    return None if state is None else {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in state.items()
    }


def preflight_passivity(solver, payload, copy_ops, host_updates):
    from ..resume import _require_tensor_compatible

    target = getattr(solver, "_passivity_state", None)
    if target is None:
        target = make_passivity_state(solver)
        solver._passivity_state = target
    # Reuse the resume contract checks for missing or changed guard policy.
    expected = {} if target is None else target
    actual = {} if payload is None else payload
    if actual.keys() != expected.keys() or any(
        actual.get(key) != expected.get(key) for key in ("topology", "start_step")
    ):
        raise ValueError("FDTD resume passivity guard policy changed or guard history is missing.")
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            _require_tensor_compatible("passivity." + key, actual[key], value)
            copy_ops.append((value, actual[key]))
        elif key in ("sample_count", "window_count", "warning"):
            host_updates.append((target, key, deepcopy(actual[key])))
    if target is not None:
        warning = deepcopy(actual["warning"])
        host_updates.extend((
            (solver, "passivity_guard_warning", warning),
            (solver, "passivity_guard_status", "warned: complex fixed-k growth" if warning else "active"),
        ))
