"""Op-stream equivalence comparator for the circuit-free FDTD path.

Gate (d) of the SPICE/MNA Phase 4 exit requires that adding the circuit feature
does not slow circuit-free simulations.  A single wall-clock ABBA run cannot
resolve the ``< 1%`` threshold on this host class (see the archived A/A
calibration), so the controlling evidence is *host-code equivalence*: the
per-step host dispatch stream for the no-feature scene must be identical between
the immutable Phase 3 baseline and the Phase 4 candidate.

The ``profile`` command times and profiles one checkout.  The ``compare`` command
launches the profiler against two checkouts, **one subprocess per root** so the
two revisions' ``import witwin.maxwell`` never cross, and writes a JSON artifact
holding both roots' op tables, their diff (which must be empty for equivalence),
the prepare-dominated ``run()`` timings, and provenance/environment metadata.

Conventions mirror ``benchmark_rf_no_feature.py``: the no-feature scene is built
by importing ``_build_scene`` from *that root's* benchmark module; the immutable
baseline is verified against ``--baseline-archive`` and ``--baseline-commit``;
and the candidate worktree content manifest is captured before and after.

Examples::

    python tests/support/compare_no_feature_op_stream.py profile --label candidate
    python tests/support/compare_no_feature_op_stream.py compare \
        --baseline-root build/baselines/phase3_0a69fc8_exact_raw \
        --candidate-root . \
        --baseline-commit 0a69fc877f83d96e5cb75d6b8564375d488c4d63 \
        --baseline-archive build/baselines/phase3_0a69fc8.zip \
        --output docs/assessments/spice-mna-phase-4-no-feature-op-stream.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


SCHEMA_VERSION = 1
FREQUENCY_HZ = 1.0e9
PROFILE_KIND = "rf_no_feature_op_stream_profile"
COMPARISON_KIND = "rf_no_feature_op_stream_comparison"


def _load_module(path: Path, name: str) -> ModuleType:
    """Load a Python file as an isolated module by explicit path.

    The benchmark module imports only the standard library at import time, so
    loading a foreign checkout's copy this way never pulls in that checkout's
    ``witwin`` package.  ``_build_scene`` still receives ``mw`` as an argument.
    """

    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Cannot load module; file does not exist: {resolved}")
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not build an import spec for {resolved}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The benchmark module that ships beside this comparator supplies the provenance
# and small git/stat helpers.  It is always the candidate's copy (this file), so
# the provenance format matches the existing gate (d) artifacts exactly.
_BENCH = _load_module(
    Path(__file__).with_name("benchmark_rf_no_feature.py"),
    "_op_stream_bench_helpers",
)


def _root_build_scene(root: Path) -> Callable[..., Any]:
    """Return ``_build_scene`` from *root*'s own benchmark module."""

    module = _load_module(
        root / "tests" / "support" / "benchmark_rf_no_feature.py",
        "_op_stream_bench_scene",
    )
    return module._build_scene


def _make_simulation(mw: Any, scene: Any, *, time_steps: int) -> Any:
    return mw.Simulation.fdtd(
        scene,
        frequency=FREQUENCY_HZ,
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=False,
    )


def _run_profile(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    _BENCH._insert_checkout_on_path(root)

    import torch

    import witwin.maxwell as mw

    if not torch.cuda.is_available():
        raise RuntimeError("The op-stream comparator requires one CUDA device.")
    if args.steps < 1:
        raise ValueError("steps must be positive.")
    if args.warmup < 1:
        raise ValueError("warmup must be at least 1 so JIT builds do not pollute the op stream.")
    if args.prepare_repeats < 15:
        raise ValueError("prepare-repeats must be at least 15 for a useful median.")

    from torch.profiler import ProfilerActivity, profile

    build_scene = _root_build_scene(root)
    scene = build_scene(mw, grid_cells=args.grid_cells)

    # Untimed, unprofiled warmup: force the CUDA extension build and any lazy
    # allocator/autograd setup so the profiled run captures only steady dispatch.
    for _ in range(args.warmup):
        _make_simulation(mw, scene, time_steps=args.steps).run()
    torch.cuda.synchronize()

    simulation = _make_simulation(mw, scene, time_steps=args.steps)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        simulation.run()
    torch.cuda.synchronize()

    op_table: dict[str, int] = {}
    for event in prof.key_averages():
        op_table[event.key] = op_table.get(event.key, 0) + int(event.count)
    aten_table = {key: count for key, count in op_table.items() if key.startswith("aten::")}
    # A per-step op executes a whole number of times per step; one-time prepare
    # ops do not.  This isolates the hot-loop kernel/dispatch set from prepare.
    per_step_kernel_op_counts = {
        key: count // args.steps
        for key, count in sorted(op_table.items())
        if count >= args.steps and count % args.steps == 0
    }

    prepare_samples_ms: list[float] = []
    for _ in range(args.prepare_repeats):
        prepare_sim = _make_simulation(mw, scene, time_steps=1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        prepare_sim.run()
        torch.cuda.synchronize()
        prepare_samples_ms.append((time.perf_counter() - start) * 1.0e3)
        del prepare_sim

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PROFILE_KIND,
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _BENCH._git_metadata(root),
        "configuration": {
            "device": "cuda",
            "frequency_hz": FREQUENCY_HZ,
            "grid_cells_per_axis": args.grid_cells,
            "profiled_time_steps": args.steps,
            "profiler_activities": ["cpu"],
            "warmup_runs": args.warmup,
            "prepare_time_steps": 1,
            "prepare_repeats": args.prepare_repeats,
            "full_field_dft": False,
            "cuda_graph": False,
            "ports": 0,
            "lumped_elements": 0,
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
        "op_stream": {
            "aten_call_total": sum(aten_table.values()),
            "distinct_op_keys": len(op_table),
            "distinct_aten_op_keys": len(aten_table),
            "op_table": dict(sorted(op_table.items())),
            "aten_op_table": dict(sorted(aten_table.items())),
            "per_step_kernel_op_counts": per_step_kernel_op_counts,
        },
        "prepare": {
            "time_steps": 1,
            "repeats": args.prepare_repeats,
            "samples_ms": prepare_samples_ms,
            "median_ms": float(statistics.median(prepare_samples_ms)),
            "mad_ms": float(_BENCH._median_absolute_deviation(prepare_samples_ms)),
            "min_ms": min(prepare_samples_ms),
            "max_ms": max(prepare_samples_ms),
        },
    }


def _parse_last_profile_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("kind") == PROFILE_KIND:
            return payload
    raise RuntimeError(f"Profiler did not emit a result JSON object. Output:\n{stdout}")


def _invoke_profiler(
    *,
    python: Path,
    runner: Path,
    root: Path,
    label: str,
    steps: int,
    prepare_repeats: int,
    grid_cells: int,
    warmup: int,
) -> dict[str, Any]:
    command = [
        str(python),
        str(runner),
        "profile",
        "--label",
        label,
        "--steps",
        str(steps),
        "--prepare-repeats",
        str(prepare_repeats),
        "--grid-cells",
        str(grid_cells),
        "--warmup",
        str(warmup),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Profiler failed for {label!r} in {root} with exit code "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    profile_payload = _parse_last_profile_json(completed.stdout)
    profile_root = Path(profile_payload["git"]["resolved_root"])
    if profile_root != root:
        raise RuntimeError(
            f"Profiler root provenance mismatch: expected {root}, reported {profile_root}."
        )
    profile_payload["execution"] = {"resolved_root": str(root), "command": command}
    return profile_payload


def _diff_int_table(
    baseline: dict[str, int], candidate: dict[str, int]
) -> dict[str, dict[str, int]]:
    diff: dict[str, dict[str, int]] = {}
    for key in sorted(set(baseline) | set(candidate)):
        base_value = int(baseline.get(key, 0))
        candidate_value = int(candidate.get(key, 0))
        if base_value != candidate_value:
            diff[key] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "delta": candidate_value - base_value,
            }
    return diff


def _op_stream_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    op_table_diff = _diff_int_table(baseline["op_table"], candidate["op_table"])
    per_step_diff = _diff_int_table(
        baseline["per_step_kernel_op_counts"], candidate["per_step_kernel_op_counts"]
    )
    aten_total_delta = candidate["aten_call_total"] - baseline["aten_call_total"]
    distinct_delta = candidate["distinct_op_keys"] - baseline["distinct_op_keys"]
    equivalent = (
        not op_table_diff
        and not per_step_diff
        and aten_total_delta == 0
        and distinct_delta == 0
    )
    return {
        "op_table_diff": op_table_diff,
        "per_step_kernel_op_counts_diff": per_step_diff,
        "aten_call_total_delta": aten_total_delta,
        "distinct_op_keys_delta": distinct_delta,
        "equivalent": equivalent,
    }


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    python = args.python.resolve()
    runner = Path(__file__).resolve()
    if baseline_root == candidate_root:
        raise ValueError("baseline_root and candidate_root must be different checkouts.")
    if not baseline_root.is_dir() or not candidate_root.is_dir():
        raise ValueError("baseline_root and candidate_root must both be existing directories.")

    baseline_commit = _BENCH._git_output(
        candidate_root, "rev-parse", f"{args.baseline_commit}^{{commit}}"
    )
    if baseline_commit is None:
        raise ValueError(
            f"baseline_commit {args.baseline_commit!r} is not available from candidate_root."
        )
    baseline_git_tree = _BENCH._git_output(
        candidate_root, "rev-parse", f"{baseline_commit}^{{tree}}"
    )
    if baseline_git_tree is None:
        raise ValueError(f"Could not resolve the Git tree for baseline commit {baseline_commit}.")

    candidate_content_before = _BENCH._worktree_content_metadata(candidate_root)
    baseline_binding_before = _BENCH._verify_baseline_snapshot(
        candidate_root, baseline_root, baseline_commit, args.baseline_archive
    )
    baseline_tree_before = _BENCH._tree_sha256(baseline_root)

    baseline_profile = _invoke_profiler(
        python=python,
        runner=runner,
        root=baseline_root,
        label="baseline",
        steps=args.steps,
        prepare_repeats=args.prepare_repeats,
        grid_cells=args.grid_cells,
        warmup=args.warmup,
    )
    candidate_profile = _invoke_profiler(
        python=python,
        runner=runner,
        root=candidate_root,
        label="candidate",
        steps=args.steps,
        prepare_repeats=args.prepare_repeats,
        grid_cells=args.grid_cells,
        warmup=args.warmup,
    )

    baseline_tree_after = _BENCH._tree_sha256(baseline_root)
    if baseline_tree_after != baseline_tree_before:
        raise RuntimeError("The immutable baseline tree changed during the comparison.")
    baseline_binding_after = _BENCH._verify_baseline_snapshot(
        candidate_root, baseline_root, baseline_commit, args.baseline_archive
    )
    if baseline_binding_after != baseline_binding_before:
        raise RuntimeError("The baseline Git-tree binding changed during the comparison.")
    candidate_content_after = _BENCH._worktree_content_metadata(candidate_root)
    if candidate_content_after != candidate_content_before:
        raise RuntimeError("candidate_root HEAD or tracked dirty state changed during comparison.")

    op_stream_diff = _op_stream_diff(
        baseline_profile["op_stream"], candidate_profile["op_stream"]
    )
    prepare_delta_ms = (
        candidate_profile["prepare"]["median_ms"] - baseline_profile["prepare"]["median_ms"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPARISON_KIND,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "torch.profiler CPU-activity op-stream capture over N steps in each "
            "checkout (one subprocess per root), plus prepare-dominated run() "
            "timing at time_steps=1; equivalence is an empty op-table diff"
        ),
        "provenance": {
            "orchestrator_command": [
                str(Path(sys.executable).resolve()),
                str(runner),
                *sys.argv[1:],
            ],
            "baseline": {
                "resolved_root": str(baseline_root),
                "expected_commit": args.baseline_commit,
                "resolved_commit": baseline_commit,
                "git_tree": baseline_git_tree,
                "git_tree_binding": baseline_binding_before,
                "tree_digest": baseline_tree_before,
            },
            "candidate": candidate_content_before,
            "python": str(python),
            "runner": str(runner),
        },
        "configuration": {
            "profiled_time_steps": args.steps,
            "prepare_repeats": args.prepare_repeats,
            "grid_cells_per_axis": args.grid_cells,
            "warmup_runs": args.warmup,
        },
        "baseline": baseline_profile,
        "candidate": candidate_profile,
        "op_stream_diff": op_stream_diff,
        "prepare_median_delta_ms": prepare_delta_ms,
        "equivalent": bool(op_stream_diff["equivalent"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser(
        "profile", help="Profile the no-feature op stream in the current directory."
    )
    profile.add_argument("--label", default="candidate")
    profile.add_argument("--steps", type=int, default=512)
    profile.add_argument("--prepare-repeats", type=int, default=15)
    profile.add_argument("--grid-cells", type=int, default=24)
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--output", type=Path)

    compare = subparsers.add_parser(
        "compare", help="Compare the no-feature op stream between two checkout roots."
    )
    compare.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Exact non-Git extraction of --baseline-archive; extra files are rejected.",
    )
    compare.add_argument("--candidate-root", type=Path, required=True)
    compare.add_argument("--baseline-commit", required=True)
    compare.add_argument(
        "--baseline-archive",
        type=Path,
        required=True,
        help="Git archive whose paths, modes, and raw blobs must match --baseline-commit.",
    )
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    compare.add_argument("--steps", type=int, default=512)
    compare.add_argument("--prepare-repeats", type=int, default=15)
    compare.add_argument("--grid-cells", type=int, default=24)
    compare.add_argument("--warmup", type=int, default=1)
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
