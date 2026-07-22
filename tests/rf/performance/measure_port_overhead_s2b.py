"""Exclusive-window timed measurement of passive-port step-time overhead (S2b).

Audit ``docs/assessments/next-functional-audit-2026-07-18.md`` step S2.3 defers
the wall-clock ``< 5%`` / ``< 2%`` port overhead gate to an exclusive-GPU window.
This script produces that evidence with the variance-aware gate the S2a harness
built (``tests/support/perf_variance_gate.py``): a regression is accepted only
when the one-sided 95% CI upper bound of the paired per-round overhead clears the
target, never a single-point median.

What is compared (all in one process, one CUDA device, no cross-commit archive):

* ``base``  -- an otherwise identical FDTD scene with **no** RF port.
* ``one``   -- the same scene plus one passive ``LumpedPort`` (matched load) with
  a 181-frequency port observer.
* ``two``   -- the same scene plus two such passive ports.

Per-step time is isolated by the **two-point subtraction** method: each block is
timed at ``--steps-lo`` and ``--steps-hi`` full ``Simulation.run()`` calls, and
``ms_per_step = (t_hi - t_lo) / (steps_hi - steps_lo)``.  This cancels the fixed
one-time preparation / graph-build / result-extraction cost exactly, so the
statistic is the true marginal per-step cost the plan §9.4 target constrains --
not a prepare-polluted average.

Gates (paired per-round ratios fed to ``evaluate_regression_gate``):

* single-port  ``one / base``  vs target 1.05  (``< 5%``).
* per-extra    ``two / one``   vs target 1.02  (``< 2%``).
* A/A          ``base_b / base_a`` calibration: two independent ``base`` blocks
  per round; a two-sided 95% CI that straddles 0% validates the wiring and its
  half-width is the measurement floor.

``cuda_graph`` is intentionally **disabled**: the audit's concern is per-step
port work dispatched *outside* any CUDA graph, so eager stepping is the
apples-to-apples, conservative comparison for both configurations.

Falsification (``--inject-overhead-us``): monkeypatch the port apply hot path to
add a known synthetic per-step cost.  Because only the port configs run that
path, the single-port CI upper bound must cross the target -- proof the gate
detects a real regression rather than rubber-stamping.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SUPPORT_DIR = Path(__file__).resolve().parents[2] / "support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

from perf_variance_gate import (  # noqa: E402
    ci95_upper_bound,
    evaluate_regression_gate,
    median_absolute_deviation,
    student_t_95_one_sided,
)

CENTER_FREQUENCY_HZ = 3.0e9
SWEEP_SPAN_HZ = 2.0e9


def _frequency_grid(count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("frequency count must be positive.")
    if count == 1:
        return (CENTER_FREQUENCY_HZ,)
    low = CENTER_FREQUENCY_HZ - 0.5 * SWEEP_SPAN_HZ
    step = SWEEP_SPAN_HZ / (count - 1)
    return tuple(low + step * index for index in range(count))


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _lumped_port(mw: Any, index: int):
    """A matched passive load: bare reference impedance, no active drive."""

    return mw.LumpedPort(
        name=f"p{index}",
        positive=(0.02 * index, 0.0, 0.005),
        negative=(0.02 * index, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.02 * index, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
        termination=None,
    )


def _domain_bounds(half_x: float, half_yz: float):
    return ((-half_x, half_x), (-half_yz, half_yz), (-half_yz, half_yz))


def _build_scene(mw: Any, *, port_count: int, cell: float, half_x: float, half_yz: float):
    bounds = _domain_bounds(half_x, half_yz)
    ports = tuple(_lumped_port(mw, index) for index in range(port_count))
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(cell),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        ports=ports,
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=1.5 * cell,
            source_time=mw.GaussianPulse(
                frequency=CENTER_FREQUENCY_HZ,
                fwidth=0.5 * SWEEP_SPAN_HZ,
                amplitude=1.0,
            ),
        )
    )
    # The base (no port) needs a valid frequency output -- a bare no-output run
    # is rejected by the solver. A single-frequency point monitor is the cheapest
    # such output: exactly one accumulation kernel per step (a 181-frequency point
    # monitor would instead launch 181 kernels per step, dominating the step and
    # corrupting the reference). Port configs use the port itself as their output
    # and carry the 181-frequency observer under test, so they need no monitor;
    # the base's lone 1-frequency monitor kernel is negligible next to a Yee step,
    # keeping the base a genuine bare-FDTD stepping reference.
    if port_count == 0:
        scene.add_monitor(
            mw.PointMonitor(name="probe", position=(0.05, 0.0, 0.0), fields=("Ez",))
        )
    if len(scene.ports) != port_count:
        raise AssertionError("scene port count mismatch")
    return scene


def _install_synthetic_overhead(microseconds: float) -> None:
    """Add a known synthetic per-step cost to the port apply hot path.

    Only the port configs traverse ``apply_port_runtimes``; ``base`` never does.
    The injected work is a GPU busy loop sized so its device time is roughly the
    requested microseconds, giving the gate a real, port-only regression to
    detect.
    """

    import torch

    from witwin.maxwell.fdtd import runtime as _runtime
    from witwin.maxwell.fdtd.runtime import stepping as _stepping

    original = _stepping.apply_port_runtimes
    device = torch.device("cuda")
    # Size the busy tensor once; a matmul of this size costs ~microseconds.
    burn = torch.empty((512, 512), device=device)

    iterations = max(1, int(microseconds))

    def _slow_apply(solver) -> None:  # pragma: no cover - timing path
        original(solver)
        # apply_port_runtimes is called every step even by the port-less base
        # (it is a no-op there); inject the synthetic cost ONLY when the scene
        # actually has ports, so the overhead is genuinely port-only and the base
        # reference stays clean.
        if not getattr(solver, "_port_runtimes", ()):
            return
        acc = burn
        for _ in range(iterations):
            acc = acc * 1.0000001 + 1.0
        # Keep the work live without a per-step host sync (a .item()/float() sync
        # would serialize the step and distort the very timing under test).
        solver._s2b_falsify_sink = acc

    _stepping.apply_port_runtimes = _slow_apply
    # The stepping module imported the symbol by name; patch that binding too.
    if hasattr(_stepping, "apply_port_runtimes"):
        _stepping.apply_port_runtimes = _slow_apply


def _scene_frequencies(scene, frequencies: tuple[float, ...]) -> tuple[float, ...]:
    """Frequencies to hand a scene so the comparison isolates the port cost.

    A multi-frequency FDTD run *without* a lumped port auto-enables a full-field
    grid DFT (``simulation.py`` ``use_full_field_dft = ... len(freqs) > 1 and not
    has_lumped_ports``), a per-step full-grid accumulation that dwarfs and is
    unrelated to the port hot path.  The plan's reference is "same-grid **base
    FDTD**" -- bare Yee stepping -- so the no-port base is run at a single
    frequency (no full-field DFT), while the port configs carry the full
    181-frequency observer that is the thing under test.  ``has_lumped_ports``
    keeps the port configs off the full-field-DFT path, so their per-step cost is
    exactly stepping + port apply + port 181-bin observer.
    """

    return frequencies if getattr(scene, "ports", ()) else (frequencies[0],)


def _time_run_ms(mw: Any, scene, *, steps: int, frequencies: tuple[float, ...]) -> float:
    import torch

    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=_scene_frequencies(scene, frequencies),
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=False,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = simulation.run()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    del result, simulation
    return elapsed


def _ms_per_step(
    mw: Any,
    scene,
    *,
    steps_lo: int,
    steps_hi: int,
    frequencies: tuple[float, ...],
) -> float:
    """Two-point per-step time: (t_hi - t_lo) / (steps_hi - steps_lo)."""

    t_lo = _time_run_ms(mw, scene, steps=steps_lo, frequencies=frequencies)
    t_hi = _time_run_ms(mw, scene, steps=steps_hi, frequencies=frequencies)
    return (t_hi - t_lo) / (steps_hi - steps_lo)


def _two_sided_ci_pct(ratios: list[float]) -> dict[str, Any]:
    """Two-sided 95% CI of the (ratio-1) regression percentage."""

    pct = [100.0 * (r - 1.0) for r in ratios]
    n = len(pct)
    mean = statistics.fmean(pct)
    se = statistics.stdev(pct) / (n ** 0.5)
    # Two-sided 95% uses the 97.5% one-sided quantile.
    from scipy import stats

    t = float(stats.t.ppf(0.975, n - 1))
    half = t * se
    return {
        "mean_pct": mean,
        "half_width_pct": half,
        "ci_low_pct": mean - half,
        "ci_high_pct": mean + half,
        "straddles_zero": bool((mean - half) < 0.0 < (mean + half)),
        "rounds": n,
    }


def _nvidia_snapshot() -> dict[str, Any]:
    def _q(fields: str) -> list[str]:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    governor = None
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_path.exists():
        try:
            governor = gov_path.read_text().strip()
        except OSError:
            governor = None
    return {
        "clocks": _q("index,name,clocks.sm,clocks.max.sm,clocks.mem,persistence_mode"),
        "utilization": _q("index,utilization.gpu,memory.used,temperature.gpu"),
        "compute_apps": [line.strip() for line in apps.stdout.splitlines() if line.strip()],
        "cpu_governor": governor,
    }


def _assert_exclusive() -> list[str]:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    own = str(os.getpid())
    foreign = [line for line in lines if not line.split(",")[0].strip() == own]
    return foreign


def _run_measure(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)

    import torch

    import witwin.maxwell as mw

    if not torch.cuda.is_available():
        raise RuntimeError("The S2b port-overhead measurement requires one CUDA device.")

    if args.inject_overhead_us > 0.0:
        _install_synthetic_overhead(args.inject_overhead_us)

    frequencies = _frequency_grid(args.frequencies)
    cell = args.cell
    half_x = args.half_x
    half_yz = args.half_yz

    # Build each scene once; scenes are static across rounds.
    scene_base_a = _build_scene(mw, port_count=0, cell=cell, half_x=half_x, half_yz=half_yz)
    scene_base_b = _build_scene(mw, port_count=0, cell=cell, half_x=half_x, half_yz=half_yz)
    scene_one = _build_scene(mw, port_count=1, cell=cell, half_x=half_x, half_yz=half_yz)
    scene_two = _build_scene(mw, port_count=2, cell=cell, half_x=half_x, half_yz=half_yz)

    def per_step(scene) -> float:
        return _ms_per_step(
            mw,
            scene,
            steps_lo=args.steps_lo,
            steps_hi=args.steps_hi,
            frequencies=frequencies,
        )

    # Warmup: burn in the allocator / JIT for every config before timing.
    for _ in range(args.warmup_rounds):
        for scene in (scene_base_a, scene_base_b, scene_one, scene_two):
            _time_run_ms(mw, scene, steps=args.steps_lo, frequencies=frequencies)

    foreign_before = _assert_exclusive()
    clocks_before = _nvidia_snapshot()

    rounds: list[dict[str, float]] = []
    for round_index in range(args.rounds):
        # Palindromic ABBA-style ordering so slow thermal/clock drift is
        # symmetric across the four blocks within (and between) rounds.
        order = ("base_a", "one", "two", "base_b")
        if round_index % 2:
            order = ("base_b", "two", "one", "base_a")
        scenes = {
            "base_a": scene_base_a,
            "base_b": scene_base_b,
            "one": scene_one,
            "two": scene_two,
        }
        measured: dict[str, float] = {}
        for label in order:
            measured[label] = per_step(scenes[label])
        base_round = 0.5 * (measured["base_a"] + measured["base_b"])
        rounds.append(
            {
                "round": round_index,
                "base_a_ms_per_step": measured["base_a"],
                "base_b_ms_per_step": measured["base_b"],
                "base_mean_ms_per_step": base_round,
                "one_ms_per_step": measured["one"],
                "two_ms_per_step": measured["two"],
                "aa_ratio": measured["base_b"] / measured["base_a"],
                "single_port_ratio": measured["one"] / base_round,
                "per_extra_port_ratio": measured["two"] / measured["one"],
            }
        )

    foreign_after = _assert_exclusive()
    clocks_after = _nvidia_snapshot()

    single_ratios = [r["single_port_ratio"] for r in rounds]
    extra_ratios = [r["per_extra_port_ratio"] for r in rounds]
    aa_ratios = [r["aa_ratio"] for r in rounds]

    single_gate = evaluate_regression_gate(single_ratios, target_ratio=1.05)
    extra_gate = evaluate_regression_gate(extra_ratios, target_ratio=1.02)
    aa_ci = _two_sided_ci_pct(aa_ratios)
    aa_upper_one_sided = ci95_upper_bound(aa_ratios)

    def _gate_dict(result, ratios: list[float]) -> dict[str, Any]:
        return {
            "target_ratio": result.target_ratio,
            "target_pct": result.target_regression_pct,
            "rounds": result.rounds,
            "mean_ratio": result.mean_ratio,
            "median_ratio": result.median_ratio,
            "mad_ratio": result.mad_ratio,
            "ci95_upper_ratio": result.ci95_upper_ratio,
            "ci95_upper_pct": result.ci95_upper_regression_pct,
            "mean_pct": 100.0 * (result.mean_ratio - 1.0),
            "passed": result.passed,
            "samples": ratios,
        }

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": 1,
        "kind": "rf_port_overhead_s2b_measurement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "two-point per-step subtraction (t_hi - t_lo)/(steps_hi - steps_lo), "
            "CUDA-event timed full Simulation.run(), eager stepping (cuda_graph=False), "
            "palindromic ABBA per-round ordering, one process on one CUDA device; "
            "variance-aware 95% CI upper-bound gate"
        ),
        "git": {"resolved_root": root_text, "commit": _git_output(root, "rev-parse", "HEAD")},
        "configuration": {
            "device": "cuda",
            "cell_size_m": cell,
            "domain_bounds": _domain_bounds(half_x, half_yz),
            "grid_cells_per_axis": [
                int(round(2 * half_x / cell)),
                int(round(2 * half_yz / cell)),
                int(round(2 * half_yz / cell)),
            ],
            "total_cells": int(round(2 * half_x / cell))
            * int(round(2 * half_yz / cell))
            * int(round(2 * half_yz / cell)),
            "frequency_count": args.frequencies,
            "steps_lo": args.steps_lo,
            "steps_hi": args.steps_hi,
            "rounds": args.rounds,
            "warmup_rounds": args.warmup_rounds,
            "cuda_graph": False,
            "full_field_dft": False,
            "inject_overhead_us": args.inject_overhead_us,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "numactl": os.environ.get("S2B_NUMACTL"),
        },
        "host_state": {
            "before": clocks_before,
            "after": clocks_after,
            "foreign_compute_apps_before": foreign_before,
            "foreign_compute_apps_after": foreign_after,
            "exclusive_window_verified": not foreign_before and not foreign_after,
        },
        "rounds_detail": rounds,
        "gates": {
            "single_port_lt_5pct": {
                "target": "one passive LumpedPort + 181-freq observer < 5% ms/step vs no-port",
                **_gate_dict(single_gate, single_ratios),
            },
            "per_extra_passive_port_lt_2pct": {
                "target": "each additional passive port < 2% ms/step",
                **_gate_dict(extra_gate, extra_ratios),
            },
        },
        "aa_calibration": {
            "criterion": "two-sided 95% CI of base-vs-base regression pct straddles 0; half-width is the measurement floor",
            "two_sided": aa_ci,
            "one_sided_ci95_upper_ratio": aa_upper_one_sided,
            "mad_ratio": median_absolute_deviation(aa_ratios),
            "t95_df": student_t_95_one_sided(len(aa_ratios) - 1),
            "samples": aa_ratios,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--steps-lo", type=int, default=400)
    parser.add_argument("--steps-hi", type=int, default=1600)
    parser.add_argument("--frequencies", type=int, default=181)
    parser.add_argument("--cell", type=float, default=0.005)
    parser.add_argument("--half-x", type=float, default=0.15)
    parser.add_argument("--half-yz", type=float, default=0.06)
    parser.add_argument(
        "--inject-overhead-us",
        type=float,
        default=0.0,
        help="Falsification: add a synthetic per-step cost to the port path.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _run_measure(args)
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
