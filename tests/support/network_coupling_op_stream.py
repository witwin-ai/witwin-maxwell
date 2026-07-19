"""Reproducible per-step op-stream tally for the connected network coupling path.

Gate class: ``perf-opcount`` (see ``docs/reference/gate-classification.md`` §5).
This counts host/device dispatch events (kernel launches, allocations,
device-to-device copies) for the connected 8-port/order-32 embedded-network
feedback block. It never measures wall-clock time, so it is valid on a shared
GPU.

Two profiled schedules run on the *same* prepared solver so they cannot drift:

* ``after``  -- the shipped composite same-step solve
  (``branch_current = (M^-1 C)@state + (M^-1 D)@v``, two dense matvecs);
* ``before`` -- the legacy sequential pivoted-LU triangular substitution
  (``_lu_solve_out``), still present in this module for the delayed
  reference-plane subset and re-driven here as the pre-optimization baseline.

Both apply the identical gather/scatter/finalize; they differ only in the
branch-current solve, so the launch delta isolates the fixed-cost reduction.
The two solves are also numerically compared (composite vs. legacy LU) as the
no-regression oracle.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd().resolve()))

import witwin.maxwell as mw
from tests.support.benchmark_network_embedding import _scene
from tests.support.port_hot_path_tally import tally_hot_path_window
from witwin.maxwell.fdtd import networks as _networks


def build_connected_solver(*, grid_cells: int):
    """Prepare the eager connected 8-port/order-32 solver used by both schedules."""

    scene = _scene(dynamic=True, grid_cells=grid_cells)
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=2.75e9,
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=False,
    )
    return simulation.prepare().solver


def _legacy_solve_currents(runtime) -> torch.Tensor:
    """Pre-optimization delay-free branch-current solve via sequential LU.

    Byte-for-byte the ordinary-Y branch that shipped before the E4b composite
    reduction, expressed against the same prepared buffers so the tally captures
    exactly the removed per-step kernels.
    """

    _networks._matvec_out(runtime.C, runtime.state, runtime.output_buffer)
    _networks._matvec_out(runtime.D, runtime.free_voltage, runtime.direct_drive)
    runtime.output_buffer.add_(runtime.direct_drive)
    _networks._lu_solve_out(
        runtime.loop_lu,
        runtime.loop_permutation,
        runtime.output_buffer,
        runtime.branch_current,
        runtime.solve_workspace,
        runtime.solve_scalar,
    )
    runtime.network_voltage.copy_(runtime.free_voltage)
    runtime.network_voltage.addcmul_(
        runtime.feedback_impedance,
        runtime.branch_current,
        value=-1.0,
    )
    return runtime.network_voltage


def _legacy_apply(solver) -> None:
    for runtime in solver._network_runtimes:
        _networks._gather_network_free_voltage(runtime)
        state_input = _legacy_solve_currents(runtime)
        _networks._scatter_network_branch_current(runtime)
        _networks._finalize_network_step(runtime, state_input)


def measure(solver, *, warmup_steps: int, profiled_steps: int) -> dict:
    """Tally the after (composite) and before (legacy LU) per-step schedules."""

    after = tally_hot_path_window(
        lambda: _networks.apply_network_runtimes(solver),
        warmup_steps=warmup_steps,
        profiled_steps=profiled_steps,
    )
    before = tally_hot_path_window(
        lambda: _legacy_apply(solver),
        warmup_steps=warmup_steps,
        profiled_steps=profiled_steps,
    )
    return {"before": before, "after": after}


def numerical_equivalence(solver, *, steps: int, seed: int = 0) -> float:
    """Worst residual/error-bound ratio, composite vs. legacy LU solve.

    The composite solve ``(M^-1 C)@state + (M^-1 D)@v`` and the sequential
    pivoted-LU solve of ``M x = C@state + D@v`` are mathematically identical, so
    the only difference between them is floating-point rounding: their two
    evaluation orders round the ill-scaled ``C@state`` matvec (||C|| ~ 6e5 for
    the benchmark) differently. The honest no-regression bound is therefore the
    matvec's intrinsic conditioning, ``n * eps * sum|terms| * cond(M)``, not a
    fixed constant -- on physically bounded state the result does not cancel and
    the ratio collapses toward zero, while adversarial random unit state (used
    here to exercise the ``gain_voltage`` path) cancels and rides near the bound.
    A ratio below one means the difference is pure rounding within that bound; a
    real algebra error (e.g. a scaled composite operator) blows far past it.
    """

    generator = torch.Generator(device=solver.device).manual_seed(seed)
    runtime = solver._network_runtimes[0]
    state_count = runtime.state.numel()
    port_count = runtime.free_voltage.numel()
    eps = torch.finfo(runtime.state.dtype).eps
    fan_in = state_count + port_count
    worst_ratio = 0.0
    for _ in range(steps):
        runtime.state.copy_(
            torch.randn(state_count, generator=generator, device=solver.device, dtype=runtime.state.dtype)
        )
        runtime.free_voltage.copy_(
            torch.randn(port_count, generator=generator, device=solver.device, dtype=runtime.free_voltage.dtype)
        )
        legacy = torch.empty_like(runtime.branch_current)
        _networks._lu_solve_out(
            runtime.loop_lu,
            runtime.loop_permutation,
            (runtime.C @ runtime.state + runtime.D @ runtime.free_voltage).clone(),
            legacy,
            runtime.solve_workspace.clone(),
            runtime.solve_scalar.clone(),
        )
        composite = runtime.gain_state @ runtime.state + runtime.gain_voltage @ runtime.free_voltage
        # Per-port absolute roundoff bound of the two matvec evaluation orders,
        # amplified by the direct-loop conditioning.
        bound = (
            2.0 * fan_in * eps * runtime.loop_condition
            * (runtime.C.abs() @ runtime.state.abs() + runtime.D.abs() @ runtime.free_voltage.abs())
        )
        bound = torch.clamp(bound, min=torch.finfo(runtime.state.dtype).tiny)
        ratio = float(((composite - legacy).abs() / bound).max())
        worst_ratio = max(worst_ratio, ratio)
    return worst_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-cells", type=int, default=48)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--profiled-steps", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The network coupling op-stream measurement requires CUDA.")

    solver = build_connected_solver(grid_cells=args.grid_cells)
    residual = numerical_equivalence(
        build_connected_solver(grid_cells=args.grid_cells), steps=64
    )
    tallies = measure(
        solver,
        warmup_steps=args.warmup_steps,
        profiled_steps=args.profiled_steps,
    )
    n = args.profiled_steps
    before_launches = tallies["before"]["launches"] / n
    after_launches = tallies["after"]["launches"] / n
    before_dtod = tallies["before"]["memcpy_dtod"] / n
    after_dtod = tallies["after"]["memcpy_dtod"] / n
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "kind": "network_coupling_op_stream",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "ports": 8,
            "model_order": 32,
            "state_count": int(solver._network_runtimes[0].state.numel()),
            "grid_cells_per_axis": args.grid_cells,
            "profiled_steps": n,
            "warmup_steps": args.warmup_steps,
            "path": "eager delay-free connected network feedback block",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
        },
        "before_sequential_lu": {
            "launches_per_step": before_launches,
            "memcpy_dtod_per_step": before_dtod,
            "allocs_per_step": tallies["before"]["allocs"] / n,
        },
        "after_composite": {
            "launches_per_step": after_launches,
            "memcpy_dtod_per_step": after_dtod,
            "allocs_per_step": tallies["after"]["allocs"] / n,
        },
        "launch_reduction_fraction": (
            0.0 if before_launches == 0 else 1.0 - after_launches / before_launches
        ),
        "numerical_equivalence_residual_over_bound": residual,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
