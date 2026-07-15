"""Hardware-qualified performance and memory harness for joint-solve FDTD.

The harness intentionally stays outside pytest: wall-clock assertions are noisy on a
shared GPU host.  It reports synchronized solve timing, per-device allocator peaks,
strong/optional weak scaling, and a direct CUDA P2P bandwidth health check.  Example::

    python scripts/dev/fdtd/multi_gpu/bench_joint.py \
        --workload cpml_dielectric \
        --nodes-x 513 --nodes-y 257 --nodes-z 257 --steps 500 --repeats 5 \
        --weak-scaling --json /tmp/fdtd-multi-gpu-a6000.json

Pass ``--assert-gates`` only on an otherwise idle, thermally stable machine.  The
memory gate accepts an explicitly measured fixed per-device overhead instead of
hiding it inside the 15% local-volume allowance from the implementation plan.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import witwin.maxwell as mw


_FREQUENCY = 1.0e9
_MIB = 1024**2
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_MAX_FIELD_ABS = 2.0e-6
_MAX_FIELD_REL = 2.0e-5


@dataclass(frozen=True)
class _WorkloadSpec:
    name: str
    frequencies: tuple[float, ...]
    full_field_dft: bool
    cpml_mode: str
    dielectric: bool = False
    broadband_source: bool = False
    graph_mode: str = "disabled"


_WORKLOADS = {
    "vacuum": _WorkloadSpec(
        name="vacuum",
        frequencies=(_FREQUENCY,),
        full_field_dft=False,
        cpml_mode="disabled",
    ),
    "cpml_dielectric": _WorkloadSpec(
        name="cpml_dielectric",
        frequencies=(_FREQUENCY,),
        full_field_dft=False,
        cpml_mode="slab",
        dielectric=True,
    ),
    "multifrequency_dft": _WorkloadSpec(
        name="multifrequency_dft",
        frequencies=(0.8e9, 1.2e9),
        full_field_dft=True,
        cpml_mode="disabled",
        broadband_source=True,
    ),
}


def _workload_spec(name: str) -> _WorkloadSpec:
    try:
        return _WORKLOADS[str(name)]
    except KeyError as exc:
        choices = ", ".join(_WORKLOADS)
        raise ValueError(f"Unknown workload {name!r}; expected one of: {choices}.") from exc


def _require_devices(values: tuple[str, ...]) -> tuple[torch.device, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("This benchmark requires at least two visible CUDA devices.")
    if len(values) != 2:
        raise SystemExit("The current x-slab benchmark requires exactly two devices.")
    devices = tuple(torch.device(value) for value in values)
    for device in devices:
        if device.type != "cuda" or device.index is None:
            raise SystemExit(f"Expected indexed CUDA devices, got {device}.")
        if device.index >= torch.cuda.device_count():
            raise SystemExit(f"{device} is not visible to this process.")
    if not torch.cuda.can_device_access_peer(devices[0].index, devices[1].index):
        raise SystemExit(f"No direct peer access from {devices[0]} to {devices[1]}.")
    if not torch.cuda.can_device_access_peer(devices[1].index, devices[0].index):
        raise SystemExit(f"No direct peer access from {devices[1]} to {devices[0]}.")
    properties = tuple(torch.cuda.get_device_properties(device) for device in devices)
    if len({(item.name, item.major, item.minor) for item in properties}) != 1:
        raise SystemExit("Joint-solve acceptance requires homogeneous CUDA devices.")
    return devices


def _axis_nodes(count: int, spacing: float) -> np.ndarray:
    if count < 4:
        raise ValueError("Every benchmark node count must be >= 4.")
    centered = np.arange(count, dtype=np.float64) - 0.5 * (count - 1)
    return centered * float(spacing)


def _vacuum_scene(
    shape: tuple[int, int, int],
    *,
    spacing: float,
    device: torch.device,
) -> mw.Scene:
    x, y, z = (_axis_nodes(count, spacing) for count in shape)
    source_position = (float(x[len(x) // 2]), 0.0, 0.0)
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x[0]), float(x[-1])),
                (float(y[0]), float(y[-1])),
                (float(z[0]), float(z[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device=str(device),
    )
    scene.add_source(
        mw.PointDipole(
            position=source_position,
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(
                frequency=_FREQUENCY,
                amplitude=1.0,
            ),
            name="source",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", source_position, fields=("Ez",)))
    return scene


def _workload_scene(
    workload: str,
    shape: tuple[int, int, int],
    *,
    spacing: float,
    device: torch.device,
) -> mw.Scene:
    spec = _workload_spec(workload)
    if spec.name == "vacuum":
        return _vacuum_scene(shape, spacing=spacing, device=device)

    x, y, z = (_axis_nodes(count, spacing) for count in shape)
    source_x_index = max(1, len(x) // 3)
    probe_x_index = min(len(x) - 2, 2 * len(x) // 3)
    source_position = (float(x[source_x_index]), 0.0, 0.0)
    probe_position = (float(x[probe_x_index]), 0.0, 0.0)
    boundary = (
        mw.BoundarySpec.pml(num_layers=4, strength=1.0e6)
        if spec.cpml_mode != "disabled"
        else mw.BoundarySpec.none()
    )
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x[0]), float(x[-1])),
                (float(y[0]), float(y[-1])),
                (float(z[0]), float(z[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=boundary,
        device=str(device),
    )
    if spec.dielectric:
        spans = tuple(float(axis[-1] - axis[0]) for axis in (x, y, z))
        scene.add_structure(
            mw.Structure(
                name="dielectric_scatterer",
                geometry=mw.Box(
                    position=(0.1 * spans[0], 0.0, 0.0),
                    size=tuple(0.25 * span for span in spans),
                ),
                material=mw.Material(eps_r=4.0),
            )
        )
    source_time = (
        mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.5e9, amplitude=1.0)
        if spec.broadband_source
        else mw.CW(frequency=_FREQUENCY, amplitude=1.0)
    )
    scene.add_source(
        mw.PointDipole(
            position=source_position,
            polarization="Ez",
            profile="ideal",
            source_time=source_time,
            name="source",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", probe_position, fields=("Ez",)))
    return scene


def _simulation(
    workload: str,
    shape: tuple[int, int, int],
    *,
    spacing: float,
    steps: int,
    device: torch.device,
    parallel: mw.FDTDParallelConfig | None = None,
):
    spec = _workload_spec(workload)
    cpml_options = (
        {"cpml_config": {"memory_mode": spec.cpml_mode}}
        if spec.cpml_mode != "disabled"
        else {}
    )
    return mw.Simulation.fdtd(
        _workload_scene(workload, shape, spacing=spacing, device=device),
        frequencies=spec.frequencies,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=spec.full_field_dft,
        cuda_graph=False,
        parallel=parallel,
        **cpml_options,
    )


def _release_cuda(devices: tuple[torch.device, ...]) -> None:
    gc.collect()
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _baseline_allocated(devices: tuple[torch.device, ...]) -> dict[str, int]:
    baseline = {}
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline[str(device)] = int(torch.cuda.memory_allocated(device))
    return baseline


def _single_sample(
    shape: tuple[int, int, int],
    *,
    workload: str = "vacuum",
    spacing: float,
    steps: int,
    device: torch.device,
) -> dict[str, object]:
    _release_cuda((device,))
    baseline = _baseline_allocated((device,))[str(device)]
    result = _simulation(
        workload,
        shape,
        spacing=spacing,
        steps=steps,
        device=device,
    ).run()
    torch.cuda.synchronize(device)
    stats = result.stats()
    sample = {
        "elapsed_s": float(stats["elapsed_s"]),
        "ms_per_step": float(stats["elapsed_s"]) * 1.0e3 / steps,
        "peak_memory_bytes": max(
            int(torch.cuda.max_memory_allocated(device)) - baseline,
            0,
        ),
        "local_node_shapes": {str(device): shape},
    }
    for key in ("cpml_requested_memory_mode", "cpml_memory_mode"):
        if key in stats:
            sample[key] = stats[key]
    del result
    _release_cuda((device,))
    return sample


def _multi_sample(
    shape: tuple[int, int, int],
    *,
    workload: str = "vacuum",
    spacing: float,
    steps: int,
    devices: tuple[torch.device, torch.device],
    overlap: bool,
    gather_fields: bool,
) -> dict[str, object]:
    _release_cuda(devices)
    baseline = _baseline_allocated(devices)
    parallel = mw.FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=overlap,
        gather_fields=gather_fields,
        result_device=devices[0],
    )
    result = _simulation(
        workload,
        shape,
        spacing=spacing,
        steps=steps,
        device=devices[0],
        parallel=parallel,
    ).run()
    for device in devices:
        torch.cuda.synchronize(device)
    stats = result.stats()
    parallel_stats = stats["parallel_stats"]
    peaks = {
        name: max(int(value) - baseline[name], 0)
        for name, value in parallel_stats["peak_memory_bytes"].items()
    }
    sample = {
        "elapsed_s": float(stats["elapsed_s"]),
        "ms_per_step": float(stats["elapsed_s"]) * 1.0e3 / steps,
        "peak_memory_bytes": peaks,
        "halo_bytes_per_step": int(parallel_stats["halo_bytes_per_step"]),
        "halo_bytes_total": int(parallel_stats["halo_bytes_total"]),
        "overlap_active": bool(parallel_stats["overlap_active"]),
        "transport": parallel_stats["transport"],
        "partitions": parallel_stats["partitions"],
        "topology": parallel_stats["topology"],
        "local_node_shapes": {
            str(shard.device): (
                int(shard.solver.Nx),
                int(shard.solver.Ny),
                int(shard.solver.Nz),
            )
            for shard in result.solver.shards
        },
    }
    for key in ("cpml_requested_memory_mode", "cpml_memory_mode"):
        if key in stats:
            sample[key] = stats[key]
    del result
    _release_cuda(devices)
    return sample


def _field_error(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    actual = actual.detach().to(device="cpu")
    reference = reference.detach().to(device="cpu")
    absolute = torch.abs(actual - reference)
    max_abs = float(absolute.max().item())
    reference_scale = float(torch.abs(reference).max().item())
    significant = torch.abs(reference) >= max(1.0e-6, 1.0e-4 * reference_scale)
    max_rel = 0.0
    if bool(significant.any()):
        relative = absolute[significant] / torch.abs(reference[significant])
        max_rel = float(relative.max().item())
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "actual_peak": float(torch.abs(actual).max().item()),
        "reference_peak": reference_scale,
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
    }


def _to_cpu_data(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, Mapping):
        return {key: _to_cpu_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_data(item) for item in value)
    return value


def _numeric_leaves(value, path: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return {}
        return {".".join(path): value}
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.number):
            return {}
        return {".".join(path): torch.as_tensor(value)}
    if isinstance(value, Mapping):
        leaves = {}
        for key, item in value.items():
            leaves.update(_numeric_leaves(item, (*path, str(key))))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_numeric_leaves(item, (*path, str(index))))
        return leaves
    return {}


def _monitor_error(
    actual: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object]:
    actual_names = tuple(actual)
    reference_names = tuple(reference)
    missing_monitors = tuple(name for name in reference_names if name not in actual)
    unexpected_monitors = tuple(name for name in actual_names if name not in reference)
    per_monitor = {}
    all_errors = []
    finite = not missing_monitors and not unexpected_monitors
    for name in reference_names:
        if name not in actual:
            continue
        actual_leaves = _numeric_leaves(actual[name])
        reference_leaves = _numeric_leaves(reference[name])
        missing_paths = tuple(path for path in reference_leaves if path not in actual_leaves)
        unexpected_paths = tuple(path for path in actual_leaves if path not in reference_leaves)
        common_paths = tuple(path for path in reference_leaves if path in actual_leaves)
        path_errors = {}
        shapes_match = True
        for path in common_paths:
            actual_value = actual_leaves[path]
            reference_value = reference_leaves[path]
            if actual_value.shape != reference_value.shape:
                shapes_match = False
                path_errors[path] = {
                    "actual_shape": tuple(actual_value.shape),
                    "reference_shape": tuple(reference_value.shape),
                    "shape_match": False,
                }
                continue
            row = _field_error(actual_value, reference_value)
            row["shape_match"] = True
            path_errors[path] = row
            all_errors.append(row)
        monitor_finite = (
            bool(common_paths)
            and shapes_match
            and not missing_paths
            and not unexpected_paths
            and all(bool(row.get("finite", False)) for row in path_errors.values())
        )
        finite = finite and monitor_finite
        per_monitor[name] = {
            "paths": path_errors,
            "missing_numeric_paths": missing_paths,
            "unexpected_numeric_paths": unexpected_paths,
            "finite": monitor_finite,
        }

    max_abs = max((float(row["max_abs"]) for row in all_errors), default=0.0)
    max_rel = max((float(row["max_rel"]) for row in all_errors), default=0.0)
    return {
        "per_monitor": per_monitor,
        "missing_monitors": missing_monitors,
        "unexpected_monitors": unexpected_monitors,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "finite": finite,
        "abs_tolerance": _MAX_FIELD_ABS,
        "rel_tolerance": _MAX_FIELD_REL,
        "passed": finite and max_abs <= _MAX_FIELD_ABS and max_rel <= _MAX_FIELD_REL,
    }


def _parity_diagnostics(
    shape: tuple[int, int, int],
    *,
    workload: str = "vacuum",
    spacing: float,
    steps: int,
    devices: tuple[torch.device, torch.device],
) -> dict[str, object]:
    """Run a cheap public-API parity check and report real six-field errors."""

    _release_cuda(devices)
    single = _simulation(
        workload,
        shape,
        spacing=spacing,
        steps=steps,
        device=devices[0],
    ).run()
    single_raw = single.raw_output if isinstance(single.raw_output, dict) else {}
    references = {
        name: single_raw.get(name, getattr(single.solver, name)).detach().to(device="cpu")
        for name in _FIELD_NAMES
    }
    reference_monitors = _to_cpu_data(single.monitors)
    del single
    _release_cuda(devices)

    parallel = mw.FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=True,
        gather_fields=True,
        result_device=devices[0],
    )
    distributed = _simulation(
        workload,
        shape,
        spacing=spacing,
        steps=steps,
        device=devices[0],
        parallel=parallel,
    ).run()
    solver = distributed.solver
    per_component = {}
    for name in _FIELD_NAMES:
        if name in distributed.raw_output:
            actual = distributed.raw_output[name]
        else:
            local_values = tuple(getattr(shard.solver, name) for shard in solver.shards)
            actual = solver._gather_component(name, local_values)
        per_component[name] = _field_error(actual, references[name])

    max_abs = max(float(row["max_abs"]) for row in per_component.values())
    max_rel = max(float(row["max_rel"]) for row in per_component.values())
    finite = all(bool(row["finite"]) for row in per_component.values())
    reference_peak = max(float(row["reference_peak"]) for row in per_component.values())
    monitor_error = _monitor_error(distributed.monitors, reference_monitors)
    fields_passed = (
        finite
        and reference_peak > 0.0
        and max_abs <= _MAX_FIELD_ABS
        and max_rel <= _MAX_FIELD_REL
    )
    report = {
        "workload": workload,
        "node_shape": shape,
        "steps": steps,
        "per_component": per_component,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "field_max_abs": max_abs,
        "field_max_rel": max_rel,
        "monitor_max_abs": float(monitor_error["max_abs"]),
        "monitor_max_rel": float(monitor_error["max_rel"]),
        "monitor_error": monitor_error,
        "reference_peak": reference_peak,
        "finite": finite,
        "abs_tolerance": _MAX_FIELD_ABS,
        "rel_tolerance": _MAX_FIELD_REL,
        "fields_passed": fields_passed,
        "monitors_passed": bool(monitor_error["passed"]),
        "passed": fields_passed and bool(monitor_error["passed"]),
    }
    del distributed, solver, references, reference_monitors
    _release_cuda(devices)
    return report


def _run_samples(
    function: Callable[[], dict[str, object]],
    *,
    warmups: int,
    repeats: int,
) -> list[dict[str, object]]:
    for _ in range(warmups):
        function()
    return [function() for _ in range(repeats)]


def _median_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    elapsed = [float(sample["elapsed_s"]) for sample in samples]
    ms_per_step = [float(sample["ms_per_step"]) for sample in samples]
    summary = {
        "elapsed_s_median": statistics.median(elapsed),
        "elapsed_s_runs": elapsed,
        "ms_per_step_median": statistics.median(ms_per_step),
        "ms_per_step_runs": ms_per_step,
    }
    peak = samples[0]["peak_memory_bytes"]
    if isinstance(peak, dict):
        summary["peak_memory_bytes_median"] = {
            device: int(
                statistics.median(
                    [int(sample["peak_memory_bytes"][device]) for sample in samples]
                )
            )
            for device in peak
        }
    else:
        summary["peak_memory_bytes_median"] = int(
            statistics.median([int(sample["peak_memory_bytes"]) for sample in samples])
        )
    for key in (
        "halo_bytes_per_step",
        "halo_bytes_total",
        "overlap_active",
        "transport",
        "partitions",
        "topology",
        "local_node_shapes",
        "cpml_requested_memory_mode",
        "cpml_memory_mode",
    ):
        if key in samples[-1]:
            summary[key] = samples[-1][key]
    return summary


def _p2p_bandwidth(
    devices: tuple[torch.device, torch.device],
    *,
    size_bytes: int,
    trials: int = 5,
) -> dict[str, object]:
    count = max(1, int(math.ceil(size_bytes / 4)))
    actual_bytes = count * 4
    repeats = max(4, int(math.ceil((1024**3) / actual_bytes)))
    rows = {}
    for source_device, destination_device in (devices, tuple(reversed(devices))):
        with torch.cuda.device(source_device):
            source = torch.empty(count, device=source_device, dtype=torch.float32).normal_()
        with torch.cuda.device(destination_device):
            destination = torch.empty(count, device=destination_device, dtype=torch.float32)
            stream = torch.cuda.Stream(device=destination_device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                for _ in range(4):
                    destination.copy_(source, non_blocking=True)
            stream.synchronize()
            bandwidths = []
            latency_us = []
            for _ in range(trials):
                with torch.cuda.stream(stream):
                    start.record(stream)
                    for _ in range(repeats):
                        destination.copy_(source, non_blocking=True)
                    end.record(stream)
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end)) / repeats
                latency_us.append(elapsed_ms * 1.0e3)
                bandwidths.append(actual_bytes / (elapsed_ms / 1.0e3) / 1.0e9)
        label = f"{source_device}->{destination_device}"
        rows[label] = {
            "size_bytes": actual_bytes,
            "repeats": repeats,
            "bandwidth_gbps_median": statistics.median(bandwidths),
            "bandwidth_gbps_runs": bandwidths,
            "latency_us_median": statistics.median(latency_us),
        }
        del source, destination
        _release_cuda(devices)
    return rows


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _hardware(devices: tuple[torch.device, torch.device]) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "nvidia_smi_query": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,pstate,"
                "clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_topology": _command_output(["nvidia-smi", "topo", "-m"]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "devices": tuple(
            {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "compute_capability": torch.cuda.get_device_capability(device),
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            for device in devices
        ),
        "peer_access": {
            f"{devices[0]}->{devices[1]}": torch.cuda.can_device_access_peer(
                devices[0].index, devices[1].index
            ),
            f"{devices[1]}->{devices[0]}": torch.cuda.can_device_access_peer(
                devices[1].index, devices[0].index
            ),
        },
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=tuple(_WORKLOADS), default="vacuum")
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"))
    parser.add_argument("--nodes-x", type=int, default=257)
    parser.add_argument("--nodes-y", type=int, default=129)
    parser.add_argument("--nodes-z", type=int, default=129)
    parser.add_argument("--spacing", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--gather-fields", action="store_true")
    parser.add_argument("--weak-scaling", action="store_true")
    parser.add_argument("--p2p-size-mib", type=float, default=256.0)
    parser.add_argument("--fixed-overhead-mib", type=float, default=0.0)
    parser.add_argument("--diagnostic-nodes-x", type=int, default=33)
    parser.add_argument("--diagnostic-nodes-y", type=int, default=17)
    parser.add_argument("--diagnostic-nodes-z", type=int, default=17)
    parser.add_argument("--diagnostic-steps", type=int, default=24)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--min-p2p-gbps", type=float, default=40.0)
    parser.add_argument("--min-strong-speedup", type=float, default=1.0)
    parser.add_argument("--min-weak-efficiency", type=float, default=0.70)
    parser.add_argument("--max-memory-factor", type=float, default=1.15)
    parser.add_argument("--assert-gates", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)

    if args.steps <= 0 or args.diagnostic_steps <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("steps/repeats must be positive and warmups must be nonnegative.")
    devices = _require_devices(tuple(args.devices))
    shape = (args.nodes_x, args.nodes_y, args.nodes_z)
    workload = _workload_spec(args.workload)
    p2p = _p2p_bandwidth(
        devices,
        size_bytes=max(4, int(args.p2p_size_mib * _MIB)),
    )
    diagnostics = None
    if not args.skip_diagnostics:
        diagnostics = _parity_diagnostics(
            (
                args.diagnostic_nodes_x,
                args.diagnostic_nodes_y,
                args.diagnostic_nodes_z,
            ),
            workload=workload.name,
            spacing=args.spacing,
            steps=args.diagnostic_steps,
            devices=devices,
        )

    single_samples = _run_samples(
        lambda: _single_sample(
            shape,
            workload=workload.name,
            spacing=args.spacing,
            steps=args.steps,
            device=devices[0],
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    multi_samples = _run_samples(
        lambda: _multi_sample(
            shape,
            workload=workload.name,
            spacing=args.spacing,
            steps=args.steps,
            devices=devices,
            overlap=not args.no_overlap,
            gather_fields=args.gather_fields,
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    single = _median_summary(single_samples)
    multi = _median_summary(multi_samples)
    strong_speedup = float(single["ms_per_step_median"]) / float(multi["ms_per_step_median"])

    single_peak = int(single["peak_memory_bytes_median"])
    multi_peaks = {key: int(value) for key, value in multi["peak_memory_bytes_median"].items()}
    fixed_overhead = int(args.fixed_overhead_mib * _MIB)
    memory_limit = args.max_memory_factor * single_peak / 2.0 + fixed_overhead
    memory_factor_to_ideal = max(multi_peaks.values()) / max(single_peak / 2.0, 1.0)

    weak = None
    if args.weak_scaling:
        local_x_cells = int(math.ceil((args.nodes_x - 1) / 2))
        weak_shape = (local_x_cells + 1, args.nodes_y, args.nodes_z)
        weak_samples = _run_samples(
            lambda: _single_sample(
                weak_shape,
                workload=workload.name,
                spacing=args.spacing,
                steps=args.steps,
                device=devices[0],
            ),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        weak_single = _median_summary(weak_samples)
        weak = {
            "single_local_shape": weak_shape,
            "single": weak_single,
            "efficiency": float(weak_single["ms_per_step_median"])
            / float(multi["ms_per_step_median"]),
        }

    local_shapes = multi.get("local_node_shapes", {})
    cpml_actual = (
        "disabled"
        if workload.cpml_mode == "disabled"
        else multi.get("cpml_memory_mode", workload.cpml_mode)
    )
    numerical_error = {
        "diagnostics_run": diagnostics is not None,
        "field_max_abs": (
            float(diagnostics["field_max_abs"]) if diagnostics is not None else None
        ),
        "field_max_rel": (
            float(diagnostics["field_max_rel"]) if diagnostics is not None else None
        ),
        "monitor_max_abs": (
            float(diagnostics["monitor_max_abs"]) if diagnostics is not None else None
        ),
        "monitor_max_rel": (
            float(diagnostics["monitor_max_rel"]) if diagnostics is not None else None
        ),
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": workload.name,
        "hardware": _hardware(devices),
        "shapes": {
            "global_node_shape": shape,
            "local_node_shapes": local_shapes,
        },
        "execution": {
            "steps": args.steps,
            "frequencies": workload.frequencies,
            "full_field_dft": workload.full_field_dft,
            "graph_mode": workload.graph_mode,
            "cpml_mode_requested": workload.cpml_mode,
            "cpml_mode_actual": cpml_actual,
            "gather_fields": args.gather_fields,
            "overlap_requested": not args.no_overlap,
        },
        "communication": {
            "transport": multi.get("transport"),
            "halo_bytes_per_step": multi.get("halo_bytes_per_step"),
            "halo_bytes_total": multi.get("halo_bytes_total"),
            "topology": multi.get("topology"),
        },
        "numerical_error": numerical_error,
        "protocol": {
            "workload": workload.name,
            "global_node_shape": shape,
            "local_node_shapes": local_shapes,
            "steps": args.steps,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "spacing": args.spacing,
            "overlap_requested": not args.no_overlap,
            "gather_fields": args.gather_fields,
            "frequencies": workload.frequencies,
            "full_field_dft": workload.full_field_dft,
            "graph_mode": workload.graph_mode,
            "cpml_mode": cpml_actual,
        },
        "p2p": p2p,
        "numerical_parity": diagnostics,
        "single_gpu": single,
        "two_gpu": multi,
        "strong_speedup": strong_speedup,
        "memory": {
            "single_peak_bytes": single_peak,
            "two_gpu_peak_bytes": multi_peaks,
            "fixed_overhead_bytes": fixed_overhead,
            "allowed_peak_per_gpu_bytes": memory_limit,
            "factor_to_ideal_half_single": memory_factor_to_ideal,
        },
        "weak_scaling": weak,
    }

    failures = []
    if diagnostics is not None and not bool(diagnostics["passed"]):
        failures.append(
            "field/monitor parity "
            f"field_max_abs={float(diagnostics['field_max_abs']):.3e}, "
            f"field_max_rel={float(diagnostics['field_max_rel']):.3e}, "
            f"monitor_max_abs={float(diagnostics['monitor_max_abs']):.3e}, "
            f"monitor_max_rel={float(diagnostics['monitor_max_rel']):.3e}"
        )
    if min(float(row["bandwidth_gbps_median"]) for row in p2p.values()) < args.min_p2p_gbps:
        failures.append(f"P2P bandwidth < {args.min_p2p_gbps:.1f} GB/s")
    if strong_speedup < args.min_strong_speedup:
        failures.append(f"strong speedup {strong_speedup:.3f} < {args.min_strong_speedup:.3f}")
    if max(multi_peaks.values()) > memory_limit:
        failures.append(
            f"per-GPU peak {max(multi_peaks.values())} > memory allowance {memory_limit:.0f}"
        )
    if weak is not None and float(weak["efficiency"]) < args.min_weak_efficiency:
        failures.append(
            f"weak efficiency {float(weak['efficiency']):.3f} < {args.min_weak_efficiency:.3f}"
        )
    report["gates"] = {"passed": not failures, "failures": failures}

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if args.assert_gates and failures:
        raise SystemExit("; ".join(failures))
    return report


if __name__ == "__main__":
    main()
