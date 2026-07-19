"""Manufactured field-duration/latching state-machine golden tests.

The direct-drive style prescribes the electric field, so the contiguous-exceedance
timer and the trigger step are analytically known:

* holding ``|E| = E0 >= critical`` from step 0, the timer after step ``n`` equals
  ``(n + 1) * dt``; the cell triggers at the first ``n`` with ``(n+1)*dt >=
  minimum_duration``, i.e. ``trigger_step = ceil(minimum_duration/dt) - 1``;
* choosing ``minimum_duration = (K + 0.5) * dt`` makes ``trigger_step = K``
  independent of floating-point rounding (the comparison never lands on an exact
  timer boundary);
* dropping ``|E|`` below ``critical`` for one step resets the contiguous timer,
  so an interrupted exceedance train triggers strictly later.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.breakdown_data import (
    BREAKDOWN_STATE_CONDUCTING,
    BREAKDOWN_STATE_INTACT,
)
from witwin.maxwell.fdtd.runtime.breakdown_data import advance_breakdown_state

from tests.breakdown_data._common import build_breakdown_scene, prepare_solver, set_uniform_ez

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)

_E0 = 100.0  # prescribed field, far above the trigger threshold below


def _drive_constant(solver, value: float, steps: int) -> None:
    set_uniform_ez(solver, value)
    for n in range(steps):
        advance_breakdown_state(solver, n)


def test_trigger_at_golden_step_for_contiguous_exceedance():
    """Constant supra-critical field triggers exactly at ceil(min_dur/dt)-1."""
    # First prepare to read dt (dt is fixed by the grid), then bake
    # minimum_duration = (K + 0.5) * dt so the golden step is K.
    probe = prepare_solver(build_breakdown_scene(critical_field=1.0, minimum_duration=0.0))
    dt = float(probe.dt)

    for golden_k in (0, 2, 5):
        min_dur = (golden_k + 0.5) * dt
        solver = prepare_solver(
            build_breakdown_scene(critical_field=1.0, minimum_duration=min_dur)
        )
        rt = solver.breakdown_runtime
        assert rt is not None and rt["node_count"] > 0
        # Sanity: the compiled minimum_duration matches, and the golden formula.
        expected = math.ceil(min_dur / dt) - 1
        assert expected == golden_k

        _drive_constant(solver, _E0, steps=golden_k + 4)

        trigger_step = rt["trigger_step"]
        state = rt["state"]
        assert int(trigger_step.min().item()) == golden_k
        assert int(trigger_step.max().item()) == golden_k
        assert torch.all(state == BREAKDOWN_STATE_CONDUCTING)


def test_immediate_trigger_when_minimum_duration_zero():
    """minimum_duration == 0 triggers on the very first supra-critical step."""
    solver = prepare_solver(build_breakdown_scene(critical_field=1.0, minimum_duration=0.0))
    rt = solver.breakdown_runtime
    _drive_constant(solver, _E0, steps=1)
    assert int(rt["trigger_step"].max().item()) == 0
    assert torch.all(rt["state"] == BREAKDOWN_STATE_CONDUCTING)


def test_contiguous_reset_delays_trigger():
    """A sub-critical step resets the timer, delaying the latching trigger.

    With minimum_duration = 2.5*dt a cell needs 3 contiguous supra-critical steps.
    Driving [E0, E0, below, E0, E0, E0] triggers only at step 5, not step 2.
    """
    probe = prepare_solver(build_breakdown_scene(critical_field=1.0, minimum_duration=0.0))
    dt = float(probe.dt)
    min_dur = 2.5 * dt  # needs 3 contiguous steps -> uninterrupted trigger at step 2

    solver = prepare_solver(
        build_breakdown_scene(critical_field=1.0, minimum_duration=min_dur)
    )
    rt = solver.breakdown_runtime
    below = 0.5  # < critical (1.0) -> resets the timer

    pattern = (_E0, _E0, below, _E0, _E0, _E0)
    for n, value in enumerate(pattern):
        set_uniform_ez(solver, value)
        advance_breakdown_state(solver, n)

    # Contiguous run of 3 completes only at step 5.
    assert int(rt["trigger_step"].max().item()) == 5
    assert int(rt["trigger_step"].min().item()) == 5
    assert torch.all(rt["state"] == BREAKDOWN_STATE_CONDUCTING)


def test_no_trigger_when_field_below_threshold():
    """A field held below critical never triggers: all cells stay intact."""
    solver = prepare_solver(
        build_breakdown_scene(critical_field=1000.0, minimum_duration=0.0)
    )
    rt = solver.breakdown_runtime
    _drive_constant(solver, _E0, steps=20)  # E0 = 100 < critical = 1000

    assert int(rt["trigger_step"].max().item()) == -1
    assert torch.all(rt["state"] == BREAKDOWN_STATE_INTACT)
    assert float(rt["total_dissipated"].item()) == 0.0


def test_trigger_time_converges_with_dt():
    """Trigger time -> the analytic field-crossing time as dt shrinks.

    A linear physical ramp E(t) = rate * t crosses ``critical`` at t* =
    critical/rate. With minimum_duration = 0 the discrete trigger fires at the
    first sample n with E(n*dt) >= critical, giving trigger_time = ceil(t*/dt)*dt
    and a staircase error in [0, dt). Three grid resolutions give three dt values;
    the error must stay within the reported staircase band and shrink.

    The resolutions are chosen in the grid-CFL-limited regime (fine enough that
    the Courant condition, not the source points-per-period ceiling, sets dt), so
    halving the cell size genuinely halves dt.
    """
    critical = 50.0
    rate = 5.0e12  # V/m/s ; t* = critical/rate = 1e-11 s
    t_star = critical / rate

    records = []
    for h in (0.01, 0.005, 0.0025):
        solver = prepare_solver(
            build_breakdown_scene(
                critical_field=critical,
                minimum_duration=0.0,
                h=h,
                half=0.08,
                box_size=(0.04, 0.04, 0.04),
                with_source=False,
            )
        )
        rt = solver.breakdown_runtime
        dt = float(solver.dt)
        n_steps = int(math.ceil(t_star / dt)) + 5
        for n in range(n_steps):
            set_uniform_ez(solver, rate * n * dt)
            advance_breakdown_state(solver, n)
        trigger_step = int(rt["trigger_step"].max().item())
        assert trigger_step >= 0
        trigger_time = trigger_step * dt
        error = trigger_time - t_star
        records.append((dt, trigger_time, error))
        # Discrete first-crossing: 0 <= error < dt (staircase band).
        assert 0.0 <= error < dt + 1e-18

    # The staircase band ceiling (dt) strictly shrinks with resolution, bounding
    # the trigger-time error tighter at each refinement.
    dts = [r[0] for r in records]
    assert dts[0] > dts[1] > dts[2]
    # Report is captured in the acceptance doc; keep the trace on the assertion.
    assert all(0.0 <= err < dt for (dt, _t, err) in records), records
