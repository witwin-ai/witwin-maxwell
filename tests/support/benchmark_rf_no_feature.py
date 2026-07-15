"""Reproducible single-device FDTD timing with every RF feature disabled.

The ``sample`` command times one checkout.  The ``compare`` command launches
the sampler against two checkouts in ABBA order so both revisions see similar
thermal and clock conditions.  Timing uses CUDA events after an untimed warmup;
the ordinary pytest suite deliberately does not assert a wall-clock threshold.

Examples::

    python tests/support/benchmark_rf_no_feature.py sample --label candidate
    python tests/support/benchmark_rf_no_feature.py compare \
        --baseline-root E:/tmp/maxwell-baseline \
        --candidate-root E:/Code/witwin-platform/maxwell
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
from typing import Any


SCHEMA_VERSION = 1
FREQUENCY_HZ = 1.0e9


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


def _git_metadata(root: Path) -> dict[str, Any]:
    status = _git_output(root, "status", "--short", "--untracked-files=no")
    return {
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "dirty_tracked_files": bool(status),
    }


def _median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _insert_checkout_on_path(root: Path) -> None:
    root_text = str(root)
    try:
        sys.path.remove(root_text)
    except ValueError:
        pass
    sys.path.insert(0, root_text)


def _build_scene(mw: Any, *, grid_cells: int) -> Any:
    if grid_cells < 12:
        raise ValueError("grid_cells must be at least 12 so CPML leaves an interior region.")
    domain_size = 0.6
    cell_size = domain_size / grid_cells
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-0.5 * domain_size, 0.5 * domain_size),
                (-0.5 * domain_size, 0.5 * domain_size),
                (-0.5 * domain_size, 0.5 * domain_size),
            )
        ),
        grid=mw.GridSpec.uniform(cell_size),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=1.5 * cell_size,
            source_time=mw.GaussianPulse(
                frequency=FREQUENCY_HZ,
                fwidth=0.5 * FREQUENCY_HZ,
                amplitude=1.0,
            ),
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            name="probe",
            position=(0.1, 0.0, 0.0),
            fields=("Ez",),
        )
    )
    if getattr(scene, "ports", ()) or getattr(scene, "lumped_elements", ()):
        raise AssertionError("The no-feature benchmark scene must contain no RF state.")
    if len(scene.monitors) != 1 or not isinstance(scene.monitors[0], mw.PointMonitor):
        raise AssertionError("The benchmark may use only its generic field probe.")
    return scene


def _run_sample(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    _insert_checkout_on_path(root)

    import torch

    import witwin.maxwell as mw

    if not torch.cuda.is_available():
        raise RuntimeError("The RF no-feature benchmark requires one CUDA device.")
    if args.warmup < 1:
        raise ValueError("warmup must be at least 1.")
    if args.repeats < 3:
        raise ValueError("repeats must be at least 3 for a useful median.")
    if args.steps < 1:
        raise ValueError("steps must be positive.")

    scene = _build_scene(mw, grid_cells=args.grid_cells)
    samples_ms: list[float] = []
    total_runs = args.warmup + args.repeats
    for run_index in range(total_runs):
        simulation = mw.Simulation.fdtd(
            scene,
            frequency=FREQUENCY_HZ,
            run_time=mw.TimeConfig(time_steps=args.steps),
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
        elapsed_ms = float(start.elapsed_time(end))
        if run_index >= args.warmup:
            samples_ms.append(elapsed_ms)
        del result, simulation, start, end

    median_ms = float(statistics.median(samples_ms))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rf_no_feature_sample",
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(root),
        "configuration": {
            "device": "cuda",
            "frequency_hz": FREQUENCY_HZ,
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "warmup_runs": args.warmup,
            "timed_runs": args.repeats,
            "full_field_dft": False,
            "cuda_graph": False,
            "ports": 0,
            "lumped_elements": 0,
            "generic_field_monitors": 1,
            "rf_monitors": 0,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "fdtd_backend": os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "cuda"),
            "cuda_extension": os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0"),
        },
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_ms_per_step": median_ms / args.steps,
        "mad_ms": float(_median_absolute_deviation(samples_ms)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("kind") == "rf_no_feature_sample":
            return payload
    raise RuntimeError(f"Sampler did not emit a result JSON object. Output:\n{stdout}")


def _invoke_sampler(
    *,
    python: Path,
    runner: Path,
    root: Path,
    label: str,
    warmup: int,
    repeats: int,
    steps: int,
    grid_cells: int,
) -> dict[str, Any]:
    command = [
        str(python),
        str(runner),
        "sample",
        "--label",
        label,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--steps",
        str(steps),
        "--grid-cells",
        str(grid_cells),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sampler failed for {label!r} in {root} with exit code "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return _parse_last_json_line(completed.stdout)


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    python = args.python.resolve()
    runner = Path(__file__).resolve()
    if baseline_root == candidate_root:
        raise ValueError("baseline_root and candidate_root must be different checkouts.")
    if args.rounds < 1:
        raise ValueError("rounds must be positive.")

    blocks: list[dict[str, Any]] = []
    round_summaries: list[dict[str, float]] = []
    # Every round is ABBA.  Reversing the next round makes the complete sequence
    # symmetric as well and limits bias from temperature or boost-clock drift.
    for round_index in range(args.rounds):
        round_blocks: list[dict[str, Any]] = []
        order = ("baseline", "candidate", "candidate", "baseline")
        if round_index % 2:
            order = ("candidate", "baseline", "baseline", "candidate")
        for label in order:
            root = baseline_root if label == "baseline" else candidate_root
            sample = _invoke_sampler(
                python=python,
                runner=runner,
                root=root,
                label=label,
                warmup=args.warmup,
                repeats=args.repeats,
                steps=args.steps,
                grid_cells=args.grid_cells,
            )
            blocks.append(sample)
            round_blocks.append(sample)
            print(
                f"{label} block median: {sample['median_ms']:.3f} ms "
                f"(MAD {sample['mad_ms']:.3f} ms)",
                file=sys.stderr,
                flush=True,
            )
        baseline_round_ms = float(
            statistics.median(
                block["median_ms"] for block in round_blocks if block["label"] == "baseline"
            )
        )
        candidate_round_ms = float(
            statistics.median(
                block["median_ms"] for block in round_blocks if block["label"] == "candidate"
            )
        )
        round_summaries.append(
            {
                "baseline_median_ms": baseline_round_ms,
                "candidate_median_ms": candidate_round_ms,
                "ratio": candidate_round_ms / baseline_round_ms,
            }
        )

    pooled: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for block in blocks:
        pooled[str(block["label"])].extend(float(value) for value in block["samples_ms"])
    baseline_median = float(statistics.median(pooled["baseline"]))
    candidate_median = float(statistics.median(pooled["candidate"]))
    paired_ratio = float(statistics.median(round_["ratio"] for round_ in round_summaries))
    regression_pct = 100.0 * (paired_ratio - 1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rf_no_feature_comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "CUDA event timing, untimed warmup, alternating ABBA/BAAB blocks, "
            "median paired-round ratio"
        ),
        "configuration": {
            "rounds": args.rounds,
            "warmup_runs_per_block": args.warmup,
            "timed_runs_per_block": args.repeats,
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "maximum_regression_pct": args.max_regression_pct,
        },
        "baseline": {
            "git": next(block["git"] for block in blocks if block["label"] == "baseline"),
            "environment": next(
                block["environment"] for block in blocks if block["label"] == "baseline"
            ),
            "samples_ms": pooled["baseline"],
            "median_ms": baseline_median,
            "mad_ms": float(_median_absolute_deviation(pooled["baseline"])),
        },
        "candidate": {
            "git": next(block["git"] for block in blocks if block["label"] == "candidate"),
            "environment": next(
                block["environment"] for block in blocks if block["label"] == "candidate"
            ),
            "samples_ms": pooled["candidate"],
            "median_ms": candidate_median,
            "mad_ms": float(_median_absolute_deviation(pooled["candidate"])),
        },
        "regression_pct": regression_pct,
        "passed": regression_pct < args.max_regression_pct,
        "rounds": round_summaries,
        "blocks": blocks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Time the checkout in the current directory.")
    sample.add_argument("--label", default="candidate")
    sample.add_argument("--warmup", type=int, default=1)
    sample.add_argument("--repeats", type=int, default=7)
    sample.add_argument("--steps", type=int, default=2000)
    sample.add_argument("--grid-cells", type=int, default=24)
    sample.add_argument("--output", type=Path)

    compare = subparsers.add_parser("compare", help="Compare two checkout roots in ABBA order.")
    compare.add_argument("--baseline-root", type=Path, required=True)
    compare.add_argument("--candidate-root", type=Path, required=True)
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    compare.add_argument("--rounds", type=int, default=2)
    compare.add_argument("--warmup", type=int, default=1)
    compare.add_argument("--repeats", type=int, default=5)
    compare.add_argument("--steps", type=int, default=2000)
    compare.add_argument("--grid-cells", type=int, default=24)
    compare.add_argument("--max-regression-pct", type=float, default=2.0)
    compare.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _run_sample(args) if args.command == "sample" else _run_compare(args)
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
