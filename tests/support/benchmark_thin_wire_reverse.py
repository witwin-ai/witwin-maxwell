"""Measure the sparse thin-wire reverse per 100k segment-steps on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.wire import (
    _component_plan,
    _group_indices,
    reverse_wire_step,
)


def _fixture(segment_count: int):
    device = torch.device("cuda")
    dtype = torch.float32
    segments = torch.arange(segment_count, device=device, dtype=torch.int64)
    nodes = segment_count + 1
    node_segments = torch.empty(2 * segment_count, device=device, dtype=torch.int64)
    node_signs = torch.empty(2 * segment_count, device=device, dtype=torch.int32)
    node_offsets = torch.empty(nodes + 1, device=device, dtype=torch.int64)
    cursor = 0
    node_offsets[0] = 0
    for node in range(nodes):
        if node > 0:
            node_segments[cursor] = node - 1
            node_signs[cursor] = -1
            cursor += 1
        if node < segment_count:
            node_segments[cursor] = node
            node_signs[cursor] = 1
            cursor += 1
        node_offsets[node + 1] = cursor

    weights = torch.full((segment_count,), 2.5e-3, device=device, dtype=dtype)
    contribution_scales = torch.full(
        (segment_count,), 1.0e-4, device=device, dtype=dtype
    )
    group_offsets = torch.arange(segment_count + 1, device=device, dtype=torch.int64)
    components = torch.zeros(segment_count, device=device, dtype=torch.int32)
    grounded = torch.zeros(nodes, device=device, dtype=torch.bool)
    coefficients = {
        "segment_offsets": group_offsets,
        "edge_components": components,
        "edge_offsets": segments,
        "weights": weights,
        "tail": segments,
        "head": segments + 1,
        "inductance": torch.full((segment_count,), 2.0, device=device, dtype=dtype),
        "node_capacitance": torch.full((nodes,), 3.0, device=device, dtype=dtype),
        "grounded": grounded,
        "node_offsets": node_offsets,
        "node_segments": node_segments,
        "node_signs": node_signs,
        "edge_group_offsets": group_offsets,
        "target_components": components,
        "target_offsets": segments,
        "contribution_segments": segments,
        "contribution_weights": weights,
        "contribution_scales": contribution_scales,
        "sample_masses": torch.full((segment_count,), 5.0e-2, device=device, dtype=dtype),
        "sample_deposition_scales": contribution_scales,
    }
    runtime = SimpleNamespace(
        coefficients=coefficients,
        # reverse_wire_step consumes the topology plans that are resolved once in
        # initialize_wire_runtime; the harness rebuilds them from the same
        # component/offset tensors so the duck-typed runtime matches production.
        sample_plan=_component_plan(components, segments),
        target_plan=_component_plan(components, segments),
        sample_segments=_group_indices(group_offsets),
    )
    solver = SimpleNamespace(device="cuda", dt=1.0e-3, _wire_runtime=runtime)
    generator = torch.Generator(device=device).manual_seed(702)
    forward_state = {
        "Ex": torch.randn(segment_count, device=device, dtype=dtype, generator=generator),
        "Ey": torch.zeros(1, device=device, dtype=dtype),
        "Ez": torch.zeros(1, device=device, dtype=dtype),
        "wire_current": torch.randn(
            segment_count, device=device, dtype=dtype, generator=generator
        ),
        "wire_charge": torch.randn(nodes, device=device, dtype=dtype, generator=generator),
    }
    post_adjoint = {
        name: torch.randn(value.shape, device=device, dtype=dtype, generator=generator)
        for name, value in forward_state.items()
    }
    eps = {
        "Ex": torch.ones(segment_count, device=device, dtype=dtype),
        "Ey": torch.ones(1, device=device, dtype=dtype),
        "Ez": torch.ones(1, device=device, dtype=dtype),
    }
    return solver, forward_state, post_adjoint, eps


def _measure(segment_count: int, samples: int) -> dict[str, object]:
    solver, forward_state, post_adjoint, eps = _fixture(segment_count)
    loops = max(1, 100_000 // segment_count)
    with torch.no_grad():
        for _ in range(10):
            reverse_wire_step(
                solver, forward_state, post_adjoint, eps_by_field=eps
            )
        torch.cuda.synchronize()
        timings = []
        peaks = []
        for _ in range(samples):
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = None
            for _ in range(loops):
                result = reverse_wire_step(
                    solver, forward_state, post_adjoint, eps_by_field=eps
                )
            end.record()
            end.synchronize()
            if result is None or not result.pre_current.is_cuda:
                raise RuntimeError("Thin-wire reverse benchmark did not produce CUDA output.")
            elapsed_ms = start.elapsed_time(end)
            timings.append(elapsed_ms * 100_000.0 / (loops * segment_count))
            peaks.append(torch.cuda.max_memory_allocated() - baseline)
    return {
        "segments": segment_count,
        "loops": loops,
        "samples_ms_per_100k_segment_steps": timings,
        "median_ms_per_100k_segment_steps": statistics.median(timings),
        "peak_incremental_bytes": max(peaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, nargs="+", default=(100, 1000))
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The thin-wire reverse benchmark requires CUDA.")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "dtype": "float32",
                "results": [_measure(value, args.samples) for value in args.segments],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
