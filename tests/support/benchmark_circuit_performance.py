"""Single-device circuit/FDTD performance gate and batched-MNA microbenchmark.

The sampler reports 8/32/128-unknown RC ladders, a same-port native 50-ohm
termination baseline, eager-vs-CUDA-Graph timing for 32 unknowns, cached factor
counts, and batched dense solve throughput. Timing uses CUDA events after
untimed warmup.

Example::

    python tests/support/benchmark_circuit_performance.py --output build/perf/circuit.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell.compiler import compile_batched_mna_factors  # noqa: E402


FREQUENCY_HZ = 3.0e9
SCHEMA_VERSION = 2


def _paired_order(round_index: int) -> tuple[str, str, str, str]:
    if round_index < 0:
        raise ValueError("round_index must be non-negative.")
    return (
        ("baseline", "circuit", "circuit", "baseline")
        if round_index % 2 == 0
        else ("circuit", "baseline", "baseline", "circuit")
    )


def _port(dx: float, *, termination=None) -> mw.LumpedPort:
    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 2.0 * dx),
        negative=(0.0, 0.0, -2.0 * dx),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.5 * dx),
            size=(5.0 * dx, 5.0 * dx, 0.0),
        ),
        reference_impedance=50.0,
        termination=termination,
    )


def _circuit(unknown_count: int) -> mw.Circuit:
    circuit = mw.Circuit(f"rc_ladder_{unknown_count}")
    nodes = tuple(circuit.node(f"n{index}") for index in range(unknown_count))
    for index, node in enumerate(nodes):
        negative = nodes[index + 1] if index + 1 < unknown_count else circuit.ground
        circuit.add(mw.Resistor(f"R{index}", node, negative, 50.0 + index % 7))
        circuit.add(mw.Capacitor(f"C{index}", node, circuit.ground, (0.5 + 0.02 * index) * 1.0e-12))
    circuit.bind_port("feed", positive=nodes[0], negative=circuit.ground)
    return circuit


def _scene(
    *,
    grid_cells: int,
    unknown_count: int | None,
    pure_no_rf: bool = False,
) -> mw.Scene:
    domain_size = 0.16
    dx = domain_size / grid_cells
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5 * domain_size, 0.5 * domain_size),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.none(),
        ports=(
            ()
            if pure_no_rf
            else (
                _port(
                    dx,
                    termination=(
                        mw.SeriesRLC(r=50.0) if unknown_count is None else None
                    ),
                ),
            )
        ),
        circuits=() if unknown_count is None else (_circuit(unknown_count),),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2 * domain_size, 0.0, 0.0),
            polarization="Ez",
            width=1.5 * dx,
            source_time=mw.GaussianPulse(
                frequency=FREQUENCY_HZ,
                fwidth=1.5e9,
            ),
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            name="probe",
            position=(0.2 * domain_size, 0.0, 0.0),
            fields=("Ez",),
        )
    )
    return scene


def _run_once(
    *,
    grid_cells: int,
    steps: int,
    unknown_count: int | None,
    cuda_graph: bool,
    pure_no_rf: bool = False,
):
    simulation = mw.Simulation.fdtd(
        _scene(
            grid_cells=grid_cells,
            unknown_count=unknown_count,
            pure_no_rf=pure_no_rf,
        ),
        frequency=FREQUENCY_HZ,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=cuda_graph,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = simulation.run()
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    stats = result.stats()
    steady_steps = stats["steady_steps"]
    if stats["steady_ms_per_step"] is None:
        # The pure no-RF diagnostic intentionally avoids production loop
        # instrumentation. Its outer CUDA events therefore report total runtime.
        steady_steps = steps
        steady_ms_per_step = elapsed_ms / steps
        steady_elapsed_ms = elapsed_ms
    else:
        steady_ms_per_step = float(stats["steady_ms_per_step"])
        steady_elapsed_ms = steady_ms_per_step * int(steady_steps)
    factorization_count = None
    checkpoint_tensor_bytes = 0
    if unknown_count is not None:
        factorization_count = result.circuit(f"rc_ladder_{unknown_count}").diagnostics[
            "factorization_count"
        ]
        checkpoint_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in result.solver._circuit_runtimes[0].checkpoint_tensors().values()
        )
    return {
        "elapsed_ms": elapsed_ms,
        "setup_and_finalize_ms": elapsed_ms - steady_elapsed_ms,
        "ms_per_step": steady_ms_per_step,
        "field_graph": stats["cuda_graph_active"],
        "circuit_graph": stats["circuit_cuda_graph_active"],
        "factorization_count": factorization_count,
        "checkpoint_tensor_bytes": checkpoint_tensor_bytes,
        "communication_bytes_per_step": 0,
    }


def _summarize_case(
    configuration: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [sample["ms_per_step"] for sample in samples]
    setup_values = [sample["setup_and_finalize_ms"] for sample in samples]
    return {
        "configuration": configuration,
        "samples_ms_per_step": values,
        "median_ms_per_step": float(statistics.median(values)),
        "min_ms_per_step": min(values),
        "max_ms_per_step": max(values),
        "samples_setup_and_finalize_ms": setup_values,
        "median_setup_and_finalize_ms": float(statistics.median(setup_values)),
        "field_graph": samples[-1]["field_graph"],
        "circuit_graph": samples[-1]["circuit_graph"],
        "factorization_count": samples[-1]["factorization_count"],
        "checkpoint_tensor_bytes": samples[-1]["checkpoint_tensor_bytes"],
        "communication_bytes_per_step": samples[-1]["communication_bytes_per_step"],
    }


def _sample_case(*, warmup: int, repeats: int, **configuration) -> dict[str, Any]:
    for _ in range(warmup):
        _run_once(**configuration)
    samples = [_run_once(**configuration) for _ in range(repeats)]
    return _summarize_case(configuration, samples)


def _sample_paired_case(
    *,
    warmup: int,
    repeats: int,
    baseline_configuration: dict[str, Any],
    circuit_configuration: dict[str, Any],
) -> dict[str, Any]:
    for _ in range(warmup):
        _run_once(**baseline_configuration)
        _run_once(**circuit_configuration)
    baseline_samples = []
    circuit_samples = []
    paired_ratios = []
    for round_index in range(repeats):
        labels = _paired_order(round_index)
        round_samples = {"baseline": [], "circuit": []}
        for label in labels:
            configuration = (
                baseline_configuration if label == "baseline" else circuit_configuration
            )
            sample = _run_once(**configuration)
            round_samples[label].append(sample)
            (baseline_samples if label == "baseline" else circuit_samples).append(sample)
        baseline_round = statistics.median(
            sample["ms_per_step"] for sample in round_samples["baseline"]
        )
        circuit_round = statistics.median(
            sample["ms_per_step"] for sample in round_samples["circuit"]
        )
        paired_ratios.append(circuit_round / baseline_round)
    result = _summarize_case(circuit_configuration, circuit_samples)
    result["paired_baseline"] = _summarize_case(
        baseline_configuration,
        baseline_samples,
    )
    result["paired_round_ratios"] = paired_ratios
    result["overhead_pct"] = 100.0 * (statistics.median(paired_ratios) - 1.0)
    return result


def _batched_solve_sample(unknown_count: int, *, batch_size: int, iterations: int):
    generator = torch.Generator(device="cuda").manual_seed(20260715 + unknown_count)
    raw = torch.randn(
        (batch_size, unknown_count, unknown_count),
        generator=generator,
        device="cuda",
    )
    matrices = raw @ raw.transpose(-1, -2)
    matrices.add_(
        unknown_count * torch.eye(unknown_count, device="cuda").unsqueeze(0)
    )
    rhs = torch.randn(
        (batch_size, unknown_count),
        generator=generator,
        device="cuda",
    )
    output = torch.empty_like(rhs)
    factor_start = torch.cuda.Event(enable_timing=True)
    factor_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    factor_start.record()
    factors = compile_batched_mna_factors(matrices)
    factor_end.record()
    factor_end.synchronize()
    factorization_ms = float(factor_start.elapsed_time(factor_end))
    for _ in range(10):
        factors.solve(rhs, out=output)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        factors.solve(rhs, out=output)
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    return {
        "batch_size": batch_size,
        "unknown_count": unknown_count,
        "iterations": iterations,
        "elapsed_ms": elapsed_ms,
        "factorization_ms": factorization_ms,
        "microseconds_per_batched_solve": 1.0e3 * elapsed_ms / iterations,
        "microseconds_per_system": 1.0e3 * elapsed_ms / (iterations * batch_size),
        "max_condition": float(factors.condition.max()),
    }


def _summarize_nsys_sqlite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Nsight Systems SQLite export is not a file: {resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {
            "StringIds",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        }
        missing = sorted(required - tables)
        if missing:
            raise ValueError(f"Nsight Systems SQLite export is missing tables: {missing}.")

        def totals(table: str) -> dict[str, int]:
            count, duration = connection.execute(
                f"SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM {table}"
            ).fetchone()
            return {"count": int(count), "total_duration_ns": int(duration)}

        api = totals("CUPTI_ACTIVITY_KIND_RUNTIME")
        kernels = totals("CUPTI_ACTIVITY_KIND_KERNEL")
        graph_trace = totals("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        synchronization = totals("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
        graph_launch_count, graph_launch_duration = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(runtime.end - runtime.start), 0)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            WHERE strings.value LIKE 'cudaGraphLaunch%'
            """
        ).fetchone()
        sync_api_count, sync_api_duration = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(runtime.end - runtime.start), 0)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            WHERE strings.value LIKE '%Synchronize%'
            """
        ).fetchone()

        def memcpy_totals(copy_kinds: tuple[int, ...] | None = None) -> dict[str, int]:
            where = ""
            parameters: tuple[int, ...] = ()
            if copy_kinds is not None:
                placeholders = ",".join("?" for _ in copy_kinds)
                where = f" WHERE copyKind IN ({placeholders})"
                parameters = copy_kinds
            count, operations, byte_count, duration = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(COALESCE(copyCount, 1)), 0), "
                "COALESCE(SUM(bytes), 0), COALESCE(SUM(end - start), 0) "
                f"FROM CUPTI_ACTIVITY_KIND_MEMCPY{where}",
                parameters,
            ).fetchone()
            return {
                "records": int(count),
                "operations": int(operations),
                "bytes": int(byte_count),
                "total_duration_ns": int(duration),
            }

        graph_exec_ids = connection.execute(
            "SELECT COUNT(DISTINCT graphExecId) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        ).fetchone()[0]
        return {
            "source": {
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
            },
            "cuda_api": api,
            "kernels": kernels,
            "cuda_graph": {
                "launch_api_calls": int(graph_launch_count),
                "launch_api_duration_ns": int(graph_launch_duration),
                "trace_records": graph_trace["count"],
                "trace_duration_ns": graph_trace["total_duration_ns"],
                "distinct_graph_exec_ids": int(graph_exec_ids),
            },
            "synchronization": {
                "api_calls": int(sync_api_count),
                "api_duration_ns": int(sync_api_duration),
                "activity_records": synchronization["count"],
                "activity_duration_ns": synchronization["total_duration_ns"],
            },
            "memcpy": {
                "all": memcpy_totals(),
                "host_to_device": memcpy_totals((1, 11)),
                "device_to_host": memcpy_totals((2, 12)),
            },
        }
    finally:
        connection.close()


def _summarize_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_payload is None:
        raise ValueError("--profile-payload is required with --summarize-nsys-sqlite.")
    payload_path = args.profile_payload.resolve()
    profile_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if profile_payload.get("kind") != "circuit_fdtd_profile_case":
        raise ValueError("Profile payload must have kind 'circuit_fdtd_profile_case'.")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "circuit_fdtd_profile_summary",
        "profile_payload": profile_payload,
        "profile_payload_path": str(payload_path),
        "nsight_systems": _summarize_nsys_sqlite(args.summarize_nsys_sqlite),
        "summary_command": [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Circuit performance benchmark requires CUDA.")
    if args.profile_only:
        unknown_count = args.unknowns[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "circuit_fdtd_profile_case",
            "configuration": {
                "grid_cells": args.grid_cells,
                "steps": args.steps,
                "unknown_count": unknown_count,
                "cuda_graph": True,
            },
            "result": _run_once(
                grid_cells=args.grid_cells,
                steps=args.steps,
                unknown_count=unknown_count,
                cuda_graph=True,
            ),
        }
    if args.repeats < 3 or args.warmup < 1:
        raise ValueError("Use at least one warmup and three timed repeats.")
    baseline_configuration = {
        "grid_cells": args.grid_cells,
        "steps": args.steps,
        "unknown_count": None,
        "cuda_graph": True,
    }
    circuits = {}
    for unknown_count in args.unknowns:
        circuits[str(unknown_count)] = _sample_paired_case(
            warmup=args.warmup,
            repeats=args.repeats,
            baseline_configuration=baseline_configuration,
            circuit_configuration={
                "grid_cells": args.grid_cells,
                "steps": args.steps,
                "unknown_count": unknown_count,
                "cuda_graph": True,
            },
        )
    baseline_key = "32" if "32" in circuits else str(args.unknowns[0])
    baseline = circuits[baseline_key]["paired_baseline"]
    pure_pair = _sample_paired_case(
        warmup=args.warmup,
        repeats=args.repeats,
        baseline_configuration={
            "grid_cells": args.grid_cells,
            "steps": args.steps,
            "unknown_count": None,
            "cuda_graph": True,
            "pure_no_rf": True,
        },
        circuit_configuration={
            "grid_cells": args.grid_cells,
            "steps": args.steps,
            "unknown_count": 32,
            "cuda_graph": True,
            "pure_no_rf": False,
        },
    )
    eager_32 = _sample_case(
        warmup=args.warmup,
        repeats=args.repeats,
        grid_cells=args.grid_cells,
        steps=args.steps,
        unknown_count=32,
        cuda_graph=False,
    )
    circuits["32"]["graph_speedup_vs_eager"] = (
        eager_32["median_ms_per_step"] / circuits["32"]["median_ms_per_step"]
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "circuit_fdtd_performance",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "method": (
            "CUDA events around the prepared steady time loop after graph capture; "
            "alternating paired baseline/circuit blocks; setup/factor/capture/finalize "
            "reported separately; median paired-round ratio after untimed warmup"
        ),
        "field_grid_rationale": (
            "64^3 default Yee grid over a 1.6-wavelength vacuum domain at 3 GHz "
            "(about 40 cells per wavelength), fixed before gate measurement as a "
            "moderate production-representative single-device RF workload"
        ),
        "baseline": baseline,
        "pure_no_rf": pure_pair["paired_baseline"],
        "circuit_32_vs_pure_no_rf": {
            "circuit_median_ms_per_step": pure_pair["median_ms_per_step"],
            "paired_round_ratios": pure_pair["paired_round_ratios"],
            "overhead_pct": pure_pair["overhead_pct"],
        },
        "circuits": circuits,
        "eager_32": eager_32,
        "batched_solves": [
            _batched_solve_sample(
                count,
                batch_size=args.batch_size,
                iterations=args.batch_iterations,
            )
            for count in args.unknowns
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unknowns", nargs="+", type=int, default=(8, 32, 128))
    parser.add_argument("--grid-cells", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-iterations", type=int, default=1000)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--summarize-nsys-sqlite", type=Path)
    parser.add_argument("--profile-payload", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = (
        _summarize_profile(args)
        if args.summarize_nsys_sqlite is not None
        else run(args)
    )
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
