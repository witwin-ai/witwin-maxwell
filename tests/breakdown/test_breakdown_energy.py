"""Breakdown dissipation channel closure against an analytic ground truth.

The dissipation channel integrates ``sigma_breakdown * |E|^2 dV dt`` (i.e.
``J . E dV dt`` with ``J = sigma_eff * E``) over the conducting edges. Under a
prescribed uniform field ``Ez = E0`` (``Ex = Ey = 0``) only the Ez edges carry
current, so the total dissipated energy has a closed form:

    W = E0^2 * sigma_post * V_ez * sum_n frac(n) * dt

with ``frac(n) = clamp((n - trigger_step) * dt / ramp_time, 0, 1)`` the linear
conductivity ramp, and ``V_ez`` the conduction-weighted Ez-edge volume

    V_ez = sum_edges 0.5*(mask_left + mask_right) * (dx_dual * dy_dual * dz_primal).

Every quantity on the right is recomputed here from the breakdown *layout* (node
occupancy) and the scene's public dual/primal grid arrays -- independent of the
runtime's energy accumulator -- so agreement is a genuine closure test, not a
tautology. The per-node channel must also partition the same total.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.fdtd.runtime.breakdown_data import (
    advance_breakdown_state,
    finalize_breakdown_data,
)

from tests.breakdown_data._common import build_breakdown_scene, prepare_solver, set_uniform_ez

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)

_E0 = 100.0


def _expected_ez_volume(scene, node_mask: torch.Tensor) -> float:
    """Conduction-weighted Ez-edge volume, recomputed from scene geometry only."""
    dx_dual = torch.as_tensor(scene.dx_dual64, dtype=torch.float64)
    dy_dual = torch.as_tensor(scene.dy_dual64, dtype=torch.float64)
    dz_primal = torch.as_tensor(scene.dz_primal64, dtype=torch.float64)
    mask = node_mask.detach().to(torch.float64).cpu()
    # Ez edge (i, j, k) joins nodes (i, j, k) and (i, j, k+1), k = 0 .. Nz-2.
    weight = 0.5 * (mask[:, :, :-1] + mask[:, :, 1:])
    vol = (
        dx_dual.view(-1, 1, 1)
        * dy_dual.view(1, -1, 1)
        * dz_primal.view(1, 1, -1)
    )
    return float((weight * vol).sum().item())


def test_dissipated_energy_matches_analytic_closed_form():
    """Total and per-node breakdown dissipation match the analytic J.E integral."""
    post_sigma = 4.0
    ramp_steps = 10
    scene = build_breakdown_scene(
        critical_field=1.0,
        minimum_duration=0.0,
        post_sigma=post_sigma,
        default_ramp_steps=ramp_steps,
        sigma_e=0.0,  # base conductivity 0 -> ramp target is exactly post_sigma
    )
    solver = prepare_solver(scene)
    rt = solver.breakdown_runtime
    dt = float(solver.dt)
    ramp_time = ramp_steps * dt

    n_steps = 30  # runs past the ramp so the steady conductivity is exercised
    set_uniform_ez(solver, _E0)
    for n in range(n_steps):
        advance_breakdown_state(solver, n)

    # Analytic recomputation. Trigger is at step 0 (min_dur = 0), so the ramp
    # fraction at step n is clamp(n*dt / ramp_time, 0, 1).
    frac_sum = sum(min(n * dt / ramp_time, 1.0) for n in range(n_steps))
    layout = rt["layout"]
    v_ez = _expected_ez_volume(solver.scene, layout.node_mask)
    expected_total = (_E0**2) * post_sigma * v_ez * frac_sum * dt

    total = float(rt["total_dissipated"].item())
    assert expected_total > 0.0
    assert math.isclose(total, expected_total, rel_tol=2e-4), (
        f"runtime total {total:.6e} vs analytic {expected_total:.6e}"
    )

    # The per-node channel partitions the same total (edge energy split among
    # capable endpoints).
    per_node_sum = float(rt["dissipated_energy"].sum().item())
    assert math.isclose(per_node_sum, total, rel_tol=1e-5)

    # Result assembly exposes the same authoritative scalar.
    data = finalize_breakdown_data(solver)
    assert math.isclose(data.total_dissipated_energy, total, rel_tol=1e-6)
    assert math.isclose(
        float(data.dissipated_energy.sum().item()), total, rel_tol=1e-5
    )


def test_no_dissipation_before_trigger():
    """A never-triggering cell dissipates zero breakdown energy."""
    scene = build_breakdown_scene(
        critical_field=1000.0, minimum_duration=0.0, post_sigma=4.0
    )
    solver = prepare_solver(scene)
    rt = solver.breakdown_runtime
    set_uniform_ez(solver, _E0)  # 100 < critical 1000
    for n in range(15):
        advance_breakdown_state(solver, n)
    assert float(rt["total_dissipated"].item()) == 0.0
    assert float(rt["dissipated_energy"].sum().item()) == 0.0


def test_dissipation_monotonic_nondecreasing_after_trigger():
    """Cumulative dissipation only grows once a cell is conducting."""
    scene = build_breakdown_scene(
        critical_field=1.0, minimum_duration=0.0, post_sigma=4.0, default_ramp_steps=5
    )
    solver = prepare_solver(scene)
    rt = solver.breakdown_runtime
    set_uniform_ez(solver, _E0)
    previous = 0.0
    for n in range(20):
        advance_breakdown_state(solver, n)
        current = float(rt["total_dissipated"].item())
        assert current >= previous - 1e-30
        previous = current
    assert previous > 0.0
