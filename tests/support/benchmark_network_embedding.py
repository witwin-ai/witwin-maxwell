"""Measure connected 8-port/order-32 FDTD overhead with CUDA event timing."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd().resolve()))
import witwin.maxwell as mw


FREQUENCY_HZ = 2.75e9
PORT_COUNT = 8
MODEL_ORDER = 32


def _mad(values: list[float]) -> float:
    median = statistics.median(values)
    return float(statistics.median(abs(value - median) for value in values))


def _ports(*, domain_size: float, grid_cells: int) -> tuple[mw.LumpedPort, ...]:
    cell = domain_size / grid_cells
    offsets = tuple(-14 + 4 * index for index in range(PORT_COUNT))
    positions = tuple(offset * cell for offset in offsets)
    return tuple(
        mw.LumpedPort(
            name=f"field_{index}",
            positive=torch.tensor((x, 0.0, cell), dtype=torch.float64),
            negative=torch.tensor((x, 0.0, -cell), dtype=torch.float64),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(
                position=torch.tensor(
                    (x, 0.0, -0.5 * cell), dtype=torch.float64
                ),
                size=torch.tensor(
                    (3.0 * cell, 3.0 * cell, 0.0), dtype=torch.float64
                ),
            ),
            reference_impedance=50.0,
        )
        for index, x in enumerate(positions)
    )


def _network(device: torch.device) -> mw.NetworkBlock:
    dtype = torch.float64
    complex_dtype = torch.complex128
    port_names = tuple(f"network_{index}" for index in range(PORT_COUNT))
    direct = 1.0e-2 * torch.eye(PORT_COUNT, device=device, dtype=dtype)
    poles = -torch.linspace(1.0e9, 32.0e9, MODEL_ORDER, device=device, dtype=dtype)
    residues = torch.zeros(
        (PORT_COUNT, PORT_COUNT, MODEL_ORDER),
        device=device,
        dtype=complex_dtype,
    )
    diagonal = torch.arange(PORT_COUNT, device=device)
    residues[diagonal, diagonal, :] = 2.0e7 / MODEL_ORDER
    model = mw.RationalModel(
        poles=poles.to(dtype=complex_dtype),
        residues=residues,
        direct=direct.to(dtype=complex_dtype),
        representation="Y",
    )
    frequencies = torch.linspace(0.25e9, 6.0e9, 33, device=device, dtype=dtype)
    data = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies),
        z0=50.0,
        port_names=port_names,
    )
    return mw.NetworkBlock(
        name="benchmark_load",
        network=data,
        connections={
            f"network_{index}": f"field_{index}" for index in range(PORT_COUNT)
        },
        fit=False,
        model=model,
    )


def _scene(*, dynamic: bool, grid_cells: int) -> mw.Scene:
    # Binary-exact bounds keep the public float32 geometry objects on the same
    # Yee planes as the float64 compiler grid at large representative meshes.
    domain_size = 0.0625
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=tuple((-0.5 * domain_size, 0.5 * domain_size) for _ in range(3))
        ),
        grid=mw.GridSpec.uniform(domain_size / grid_cells),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        ports=(
            _ports(domain_size=domain_size, grid_cells=grid_cells)
            if dynamic
            else ()
        ),
        networks=(
            (_network(torch.device("cuda")),)
            if dynamic
            else ()
        ),
        device="cuda",
    )
    return scene


def _sample(
    scene: mw.Scene, *, steps: int
) -> tuple[float, int, int, tuple[bool, bool, bool]]:
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=FREQUENCY_HZ,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=True,
    )
    solver = simulation.prepare().solver
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start.record()
    solver.solve(
        time_steps=steps,
        dft_frequency=FREQUENCY_HZ,
        dft_window="none",
        full_field_dft=False,
        use_cuda_graph=True,
    )
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    state_count = sum(
        runtime.state.numel()
        for runtime in getattr(solver, "_network_runtimes", ())
    )
    graph_state = (
        bool(getattr(solver, "_cuda_graph_active", False)),
        bool(getattr(solver, "_network_cuda_graph_active", False)),
        bool(getattr(solver, "_port_observer_graph_active", False)),
    )
    return elapsed_ms, state_count, int(torch.cuda.max_memory_allocated()), graph_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grid-cells", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-overhead-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The network embedding benchmark requires CUDA.")
    if args.steps < 1 or args.grid_cells < 16 or args.rounds < 2:
        raise ValueError("steps must be positive, grid-cells >= 16, and rounds >= 2.")

    scenes = {
        "baseline": _scene(dynamic=False, grid_cells=args.grid_cells),
        "connected": _scene(dynamic=True, grid_cells=args.grid_cells),
    }
    _sample(scenes["baseline"], steps=max(8, args.steps // 10))
    _sample(scenes["connected"], steps=max(8, args.steps // 10))

    samples = {"baseline": [], "connected": []}
    state_count = 0
    peak_memory_bytes = 0
    connected_graph_state = (False, False, False)
    for round_index in range(args.rounds):
        order = (
            ("baseline", "connected", "connected", "baseline")
            if round_index % 2 == 0
            else ("connected", "baseline", "baseline", "connected")
        )
        for label in order:
            elapsed_ms, measured_state_count, measured_peak_memory, graph_state = _sample(
                scenes[label],
                steps=args.steps,
            )
            samples[label].append(elapsed_ms)
            if label == "connected":
                state_count = measured_state_count
                peak_memory_bytes = max(peak_memory_bytes, measured_peak_memory)
                connected_graph_state = graph_state

    if connected_graph_state != (True, True, True):
        raise RuntimeError(
            "Connected benchmark did not activate the full field/network/observer CUDA graph."
        )

    baseline_ms = float(statistics.median(samples["baseline"]))
    connected_ms = float(statistics.median(samples["connected"]))
    overhead_pct = 100.0 * (connected_ms / baseline_ms - 1.0)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "kind": "network_embedding_performance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "rounds": args.rounds,
            "ports": PORT_COUNT,
            "model_order": MODEL_ORDER,
            "baseline": "same FDTD grid with no ports or embedded network",
            "network_state_count": state_count,
            "expected_linear_state_count": PORT_COUNT * MODEL_ORDER,
            "fit_time_ms": 0.0,
            "implicit_solve_size": PORT_COUNT,
            "spatial_communication_bytes_per_step": 0,
            "peak_memory_bytes": peak_memory_bytes,
            "cuda_graph_requested": True,
            "cuda_graph_active": {
                "field": connected_graph_state[0],
                "network": connected_graph_state[1],
                "port_observer": connected_graph_state[2],
            },
            "maximum_overhead_pct": args.max_overhead_pct,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "baseline": {
            "samples_ms": samples["baseline"],
            "median_ms": baseline_ms,
            "mad_ms": _mad(samples["baseline"]),
        },
        "connected": {
            "samples_ms": samples["connected"],
            "median_ms": connected_ms,
            "mad_ms": _mad(samples["connected"]),
        },
        "overhead_pct": overhead_pct,
        "passed": state_count == PORT_COUNT * MODEL_ORDER and overhead_pct < args.max_overhead_pct,
    }
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
