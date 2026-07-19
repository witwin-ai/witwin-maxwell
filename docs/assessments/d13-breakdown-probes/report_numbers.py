"""Regenerate the reproducible metrics quoted in the D13 breakdown acceptance doc.

This is a committed reproduction probe (not a pytest node) for the two number
tables in ``docs/assessments/d13-breakdown-acceptance-2026-07-19.md``:

1. the trigger-time dt-convergence staircase table, and
2. the analytic dissipation-closure scalar.

Both drive the breakdown runtime directly with a prescribed uniform ``Ez`` field
(``Ex = Ey = 0`` so the energy-consistent colocation gives node ``|E| = Ez``),
matching the test harness in ``tests/breakdown/``. It intentionally reuses the
same builders so the printed numbers are the ones the tests assert on.

Run (from the worktree root, with the environment from the acceptance doc):

    export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
    export PATH="$CUDA_HOME/bin:$PATH"
    export PYTHONPATH=$(pwd)
    export CUDA_VISIBLE_DEVICES=1
    conda run -n maxwell --no-capture-output python \
        docs/assessments/d13-breakdown-probes/report_numbers.py
"""

from __future__ import annotations

import math

import torch

from tests.breakdown._common import build_breakdown_scene, prepare_solver, set_uniform_ez
from witwin.maxwell.fdtd.runtime.breakdown import advance_breakdown_state


def dt_convergence_table() -> None:
    """Linear physical ramp E(t) = rate * t; first-crossing staircase band."""
    critical = 50.0
    rate = 5.0e12  # V/m/s ; t* = critical / rate = 1e-11 s
    t_star = critical / rate

    print("dt-convergence (linear ramp E(t)=5e12*t, critical=50, minimum_duration=0)")
    print(f"t* = {t_star:.4e} s")
    header = (
        f"| {'h (m)':^6} | {'dt (s)':^10} | {'trigger_step':^12} | "
        f"{'trigger_time (s)':^16} | {'error (s)':^10} | {'band ceiling = dt (s)':^21} |"
    )
    print(header)
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
        trigger_time = trigger_step * dt
        error = trigger_time - t_star
        print(
            f"| {h:^6.4f} | {dt:^10.4e} | {trigger_step:^12d} | "
            f"{trigger_time:^16.4e} | {error:^10.4e} | {dt:^21.4e} |"
        )


def energy_closure() -> None:
    """Total breakdown dissipation vs the analytic J.E integral."""
    post_sigma = 4.0
    ramp_steps = 10
    e0 = 100.0
    n_steps = 30
    scene = build_breakdown_scene(
        critical_field=1.0,
        minimum_duration=0.0,
        post_sigma=post_sigma,
        default_ramp_steps=ramp_steps,
        sigma_e=0.0,
    )
    solver = prepare_solver(scene)
    rt = solver.breakdown_runtime
    dt = float(solver.dt)
    ramp_time = ramp_steps * dt

    set_uniform_ez(solver, e0)
    for n in range(n_steps):
        advance_breakdown_state(solver, n)

    # Analytic recomputation from the layout node mask + the grid-materialized
    # (prepared) scene's dual/primal arrays.
    prepared_scene = solver.scene
    dx_dual = torch.as_tensor(prepared_scene.dx_dual64, dtype=torch.float64)
    dy_dual = torch.as_tensor(prepared_scene.dy_dual64, dtype=torch.float64)
    dz_primal = torch.as_tensor(prepared_scene.dz_primal64, dtype=torch.float64)
    mask = rt["layout"].node_mask.detach().to(torch.float64).cpu()
    weight = 0.5 * (mask[:, :, :-1] + mask[:, :, 1:])
    vol = dx_dual.view(-1, 1, 1) * dy_dual.view(1, -1, 1) * dz_primal.view(1, 1, -1)
    v_ez = float((weight * vol).sum().item())
    frac_sum = sum(min(n * dt / ramp_time, 1.0) for n in range(n_steps))
    expected = (e0**2) * post_sigma * v_ez * frac_sum * dt

    total = float(rt["total_dissipated"].item())
    rel_err = abs(total - expected) / expected
    print()
    print(
        f"energy closure (post_sigma={post_sigma}, ramp_steps={ramp_steps}, "
        f"E0={e0}, {n_steps} steps)"
    )
    print(f"runtime total  = {total:.6e} J")
    print(f"analytic total = {expected:.6e} J")
    print(f"relative error = {rel_err:.2e}  (test tolerance rel_tol=2e-4)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the breakdown runtime probe.")
    dt_convergence_table()
    energy_closure()
