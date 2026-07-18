"""Reproducible per-step op inventory for the RF port hot path.

This turns the audit's ``14.4x port overhead`` diagnosis (see
``docs/assessments/next-functional-audit-2026-07-18.md`` §1.1 and step S2.1)
into a machine-readable, reproducible artifact.  It profiles the two §9.4
scenarios that drive the port cost model:

* ``single_port_frequency_sweep`` -- one passive :class:`LumpedPort` scored over
  ``--frequencies`` DFT bins (default 181).  This isolates the per-frequency
  accumulation cost.
* ``passive_port_sweep`` -- 1, 2, and 4 passive ports over the same frequency
  grid.  The marginal op count between consecutive counts is the
  *per-additional-passive-port* cost the audit's ``< 2%`` target constrains.

Only host/device **dispatch counts** are recorded (kernel launches, allocations,
device-to-device copies, host-side transfers, scalar syncs) -- never wall-clock
timing.  The timed ``< 5% / < 2%`` assertions run later in an exclusive window
using the variance-aware gate in ``tests/support/perf_variance_gate.py``.

The ``profile`` command inventories the checkout in the current directory.  The
``compare`` command reconstructs the pre-optimization baseline with
``git archive`` (default commit ``eb9258b``), profiles it in an isolated
subprocess whose ``PYTHONPATH`` and CUDA build directory never cross the current
tree, and writes a before/after artifact with the per-scenario deltas.

Examples::

    python tests/rf/performance/profile_port_hot_path.py profile \
        --output docs/assessments/port-hot-path-op-inventory.json
    python tests/rf/performance/profile_port_hot_path.py compare \
        --baseline-commit eb9258b \
        --output docs/assessments/port-hot-path-before-after.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROFILE_KIND = "rf_port_hot_path_profile"
COMPARISON_KIND = "rf_port_hot_path_before_after"
DEFAULT_BASELINE_COMMIT = "eb9258b"
# The audit's diagnosis frequency grid: a 181-bin S-parameter sweep.
DEFAULT_FREQUENCIES = 181
CENTER_FREQUENCY_HZ = 3.0e9
SWEEP_SPAN_HZ = 2.0e9
# Op-inventory steps: warmup absorbs the lazy JIT/allocator setup, then the
# profiled window sees only steady per-step dispatch.
WARMUP_STEPS = 8
PROFILED_STEPS = 16


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


def _repository_root(start: Path) -> Path:
    text = _git_output(start, "rev-parse", "--show-toplevel")
    if text is None:
        raise RuntimeError(f"{start} is not inside a Git checkout.")
    return Path(text).resolve()


def _frequency_grid(count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("frequency count must be positive.")
    if count == 1:
        return (CENTER_FREQUENCY_HZ,)
    low = CENTER_FREQUENCY_HZ - 0.5 * SWEEP_SPAN_HZ
    step = SWEEP_SPAN_HZ / (count - 1)
    return tuple(low + step * index for index in range(count))


def _lumped_port(mw: Any, index: int, *, series_rlc: bool):
    # A single-branch series R+L+C termination exercises the inductor/capacitor
    # state advancement that dominated the audit's 14.4x diagnosis; a bare
    # reference impedance is the matched passive load used for the marginal sweep.
    termination = (
        mw.SeriesRLC(r=60.0, l=1.2e-9, c=0.8e-12) if series_rlc else None
    )
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
        termination=termination,
    )


def _build_solver(
    mw: Any,
    *,
    port_count: int,
    frequencies: tuple[float, ...],
    series_rlc: bool = False,
):
    """Build a multi-port scene and return its prepared FDTD solver.

    The scene is intentionally tiny and boundary-free: it exists only so the
    port runtimes materialize.  The hot-path inventory drives the port apply /
    observe entry points directly, so the field grid size never enters the
    measured counts.
    """

    if port_count < 1:
        raise ValueError("port_count must be positive.")
    ports = tuple(
        _lumped_port(mw, index, series_rlc=series_rlc) for index in range(port_count)
    )
    # A fixed grid-aligned domain wide enough for the largest sweep count keeps
    # the port positions (0.02 * index) on the Yee grid for every port count.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.02, 0.02), (-0.02, 0.02))),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=ports,
        device="cuda",
    )
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=frequencies,
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).prepare()
    return prepared.solver


def _inventory_per_step(mw_ports, solver) -> dict[str, float]:
    """Return per-step dispatch counts for the port apply/observe hot path."""

    import torch

    apply_port_runtimes = mw_ports.apply_port_runtimes
    accumulate_port_observers = mw_ports.accumulate_port_observers
    prepare = mw_ports.prepare_port_spectral_accumulators
    prepare(solver, 64, "none")

    for _ in range(WARMUP_STEPS):
        apply_port_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        acc_events=True,
    ) as prof:
        for _ in range(PROFILED_STEPS):
            apply_port_runtimes(solver)
            accumulate_port_observers(solver)
    torch.cuda.synchronize()

    tally = {
        "launches": 0,
        "allocs": 0,
        "memcpy_dtod": 0,
        "memcpy_hostside": 0,
        "scalar_sync": 0,
        "device_mem_bytes": 0,
    }
    for event in prof.key_averages():
        key = event.key
        if "cudaLaunchKernel" in key:
            tally["launches"] += event.count
        if key in ("aten::empty", "aten::empty_strided", "aten::empty_like"):
            tally["allocs"] += event.count
        if "Memcpy DtoD" in key:
            tally["memcpy_dtod"] += event.count
        if "Memcpy HtoD" in key or "Memcpy DtoH" in key:
            tally["memcpy_hostside"] += event.count
        if key in ("aten::item", "aten::_local_scalar_dense"):
            tally["scalar_sync"] += event.count
        tally["device_mem_bytes"] += max(0, getattr(event, "self_device_memory_usage", 0))
    return {key: value / PROFILED_STEPS for key, value in tally.items()}


def _run_profile(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)

    import torch

    import witwin.maxwell as mw
    from witwin.maxwell.fdtd import ports as mw_ports

    if not torch.cuda.is_available():
        raise RuntimeError("The port hot-path inventory requires one CUDA device.")

    frequencies = _frequency_grid(args.frequencies)

    single_port = _build_solver(mw, port_count=1, frequencies=frequencies)
    single = _inventory_per_step(mw_ports, single_port)

    series_rlc_port = _build_solver(
        mw, port_count=1, frequencies=frequencies, series_rlc=True
    )
    series_rlc = _inventory_per_step(mw_ports, series_rlc_port)

    sweep: dict[str, dict[str, float]] = {}
    for port_count in args.port_counts:
        solver = _build_solver(mw, port_count=port_count, frequencies=frequencies)
        sweep[str(port_count)] = _inventory_per_step(mw_ports, solver)

    marginal: dict[str, dict[str, float]] = {}
    ordered = sorted(int(count) for count in sweep)
    for lower, upper in zip(ordered, ordered[1:]):
        span = upper - lower
        marginal[f"{lower}->{upper}"] = {
            key: (sweep[str(upper)][key] - sweep[str(lower)][key]) / span
            for key in sweep[str(upper)]
        }

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PROFILE_KIND,
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "resolved_root": str(root),
            "commit": _git_output(root, "rev-parse", "HEAD"),
        },
        "configuration": {
            "device": "cuda",
            "frequency_count": args.frequencies,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "sweep_span_hz": SWEEP_SPAN_HZ,
            "port_counts": list(args.port_counts),
            "warmup_steps": WARMUP_STEPS,
            "profiled_steps": PROFILED_STEPS,
            "measured": "port apply + observe per-step dispatch counts",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "scenarios": {
            "single_port_frequency_sweep": single,
            "series_rlc_port_frequency_sweep": series_rlc,
            "passive_port_sweep": sweep,
            "marginal_per_additional_port": marginal,
        },
    }


def _parse_last_json(stdout: str, kind: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("kind") == kind:
            return payload
    raise RuntimeError(f"Profiler did not emit a {kind!r} JSON object. Output:\n{stdout}")


def _profile_baseline_checkout(args: argparse.Namespace, repository_root: Path) -> dict[str, Any]:
    commit = _git_output(repository_root, "rev-parse", f"{args.baseline_commit}^{{commit}}")
    if commit is None:
        raise ValueError(
            f"baseline_commit {args.baseline_commit!r} is not available from {repository_root}."
        )
    runner_relative = Path(__file__).resolve().relative_to(repository_root)
    with tempfile.TemporaryDirectory(prefix="witwin-port-baseline-") as scratch:
        checkout = Path(scratch) / "tree"
        checkout.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repository_root), "archive", commit],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["tar", "-x", "-C", str(checkout)],
            input=archive.stdout,
            check=True,
        )
        # The baseline predates this profiler; inject the current runner so both
        # revisions are measured with identical scene construction and counting.
        baseline_runner = checkout / runner_relative
        baseline_runner.parent.mkdir(parents=True, exist_ok=True)
        baseline_runner.write_bytes(Path(__file__).resolve().read_bytes())
        command = [
            str(args.python),
            str(baseline_runner),
            "profile",
            "--label",
            f"baseline:{commit[:9]}",
            "--frequencies",
            str(args.frequencies),
            "--port-counts",
            *[str(count) for count in args.port_counts],
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(checkout)
        # Isolate the baseline CUDA extension so a source drift never reuses the
        # candidate binary.
        environment["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(Path(scratch) / "cuda-build")
        completed = subprocess.run(
            command,
            cwd=checkout,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Baseline profiler failed for {commit} with exit code "
                f"{completed.returncode}.\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        payload = _parse_last_json(completed.stdout, PROFILE_KIND)
    payload.setdefault("git", {})["commit"] = commit
    return payload


def _scenario_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def _delta_table(base: dict[str, float], cand: dict[str, float]) -> dict[str, Any]:
        keys = sorted(set(base) | set(cand))
        out: dict[str, Any] = {}
        for key in keys:
            b = float(base.get(key, 0.0))
            c = float(cand.get(key, 0.0))
            out[key] = {
                "baseline": b,
                "candidate": c,
                "delta": c - b,
                "ratio": (c / b) if b else None,
            }
        return out

    single = _delta_table(
        baseline["scenarios"]["single_port_frequency_sweep"],
        candidate["scenarios"]["single_port_frequency_sweep"],
    )
    series_rlc = _delta_table(
        baseline["scenarios"]["series_rlc_port_frequency_sweep"],
        candidate["scenarios"]["series_rlc_port_frequency_sweep"],
    )
    marginal: dict[str, Any] = {}
    base_marginal = baseline["scenarios"]["marginal_per_additional_port"]
    cand_marginal = candidate["scenarios"]["marginal_per_additional_port"]
    for span in sorted(set(base_marginal) | set(cand_marginal)):
        marginal[span] = _delta_table(
            base_marginal.get(span, {}), cand_marginal.get(span, {})
        )
    return {
        "single_port_frequency_sweep": single,
        "series_rlc_port_frequency_sweep": series_rlc,
        "marginal_per_additional_port": marginal,
    }


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = _repository_root(Path.cwd())
    candidate = _run_profile(args)
    baseline = _profile_baseline_checkout(args, repository_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPARISON_KIND,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "per-step port apply+observe dispatch counts, candidate profiled "
            "in-place and baseline reconstructed via git archive in an isolated "
            "subprocess; op counts only, no wall-clock timing"
        ),
        "baseline_commit": {
            "requested": args.baseline_commit,
            "resolved": baseline.get("git", {}).get("commit"),
        },
        "configuration": candidate["configuration"],
        "baseline": baseline,
        "candidate": candidate,
        "delta": _scenario_delta(baseline, candidate),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Inventory the checkout in the CWD.")
    profile.add_argument("--label", default="candidate")
    profile.add_argument("--frequencies", type=int, default=DEFAULT_FREQUENCIES)
    profile.add_argument("--port-counts", type=int, nargs="+", default=[1, 2, 4])
    profile.add_argument("--output", type=Path)

    compare = subparsers.add_parser(
        "compare", help="Reconstruct the baseline via git archive and diff it."
    )
    compare.add_argument("--baseline-commit", default=DEFAULT_BASELINE_COMMIT)
    compare.add_argument("--frequencies", type=int, default=DEFAULT_FREQUENCIES)
    compare.add_argument("--port-counts", type=int, nargs="+", default=[1, 2, 4])
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    compare.add_argument("--label", default="candidate")
    compare.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _run_profile(args) if args.command == "profile" else _run_compare(args)
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
