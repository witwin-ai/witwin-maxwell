"""Layered network stage timings for ~2k/5k/8k RWGs and twenty RFIC frequencies.

Use the assigned GPU1 timing runner, one target per <=30 minute lock window.
The script does not select a device or acquire a workstation lock. V and P
share the same quadrature/gather pass, so their assembly time is reported
jointly. Chunk GPU times are divided by their frequency count.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["OMP_NUM_THREADS"] = "8"

import torch

import witwin.maxwell as mw
from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.assembly import assemble_potentials
from witwin.maxwell.mom.layered.lowfreq import build_loop_tree, scale_potentials
from witwin.maxwell.mom.layered.network import (
    available_work_memory, frequency_chunk_capacity, prepare_layered_network,
    recover_scaled_chunk, solve_scaled_chunk)


def timed(operation):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    begin.record()
    result = operation()
    end.record()
    end.synchronize()
    return {"gpu_seconds": begin.elapsed_time(end) / 1000,
            "wall_seconds": time.perf_counter() - wall}, result



def device_peaks():
    """CUDA-event peaks on the assigned timing device, before each case."""
    from witwin.maxwell.mom.layered.build import build_extension
    build_extension()
    rates = {}
    threads, iterations = 131072, 4096
    for dtype in (torch.float32, torch.float64):
        anchor = torch.empty(0, device="cuda", dtype=dtype)
        torch.ops.witwin_maxwell_mom_layered.fma_peak(anchor, threads, iterations)
        samples = [timed(lambda: torch.ops.witwin_maxwell_mom_layered.fma_peak(
            anchor, threads, iterations))[0]["gpu_seconds"] for _ in range(5)]
        rates[str(dtype).removeprefix("torch.")] = {
            "seconds": min(samples), "FLOP_per_second": threads * iterations * 16 / min(samples)}
    source = torch.empty(32 * 1024**2, device="cuda", dtype=torch.float64)
    target = torch.empty_like(source)
    target.copy_(source)
    samples = [timed(lambda: target.copy_(source))[0]["gpu_seconds"] for _ in range(5)]
    rates["copy"] = {"seconds": min(samples), "bytes_per_second": source.numel() * 16 / min(samples)}
    return rates


def kernel_model(prepared, chunk_size):
    """Logical traffic and real FLOP; cache hits and SFU instructions are separate."""
    import numpy as np
    mesh, lt = prepared.layout.mesh, prepared.loop_tree
    centres = mesh.centroids.cpu().numpy()
    triangles = mesh.triangles.cpu().numpy()
    size = ((triangles - np.roll(triangles, 1, axis=1))**2).sum(-1).max(-1)
    distances = prepared.table.image_distances.cpu().numpy()
    coefficients = prepared.table.image_coefficients[:chunk_size].ne(0).any(0).cpu().numpy()
    # This fixture has one metal plane. Count every ordered face pair and
    # every extracted image, with the exact runtime distance predicates.
    assert len(prepared.table.planes) == 1
    thresholds = np.array([0., 0., 3., 4., 8., 24., 1024., 1e8])**2
    counts = np.array([64, 25, 13, 12, 7, 6, 3, 1])
    samples = {"remainder_fp64": 0, "remainder_fp32": 0,
               "static_fp64": 0, "static_fp32": 0, "analytic_nodes": 0, "analytic_pairs": 0,
               "regular_image_pairs": 0, "vector_static": 0, "vector_analytic_nodes": 0,
               "vector_regular_images": 0, "vector_near_images": 0,
               "scalar_regular_images": 0, "scalar_near_images": 0}
    def nodes(r2):
        return counts[np.searchsorted(thresholds, r2, side="right") - 1]
    faces = len(centres)
    for begin in range(0, faces, 64):
        xy = centres[:, None, :2] - centres[None, begin:begin + 64, :2]
        lateral = (xy**2).sum(-1)
        scale = np.maximum(size[:, None], size[None, begin:begin + 64])
        near = lateral < 4 * scale
        q2 = nodes(lateral / scale)**2
        samples["remainder_fp64"] += int(q2[near].sum())
        samples["remainder_fp32"] += int(q2[~near].sum())
        for image, d in enumerate(distances):
            if not np.any(coefficients[:, image]):
                continue
            r2 = (lateral + d*d) / scale
            analytic = r2 < 4
            outer = np.select([r2 < .5, r2 < 1, r2 < 2], [1024, 576, 256], default=64)
            samples["analytic_nodes"] += int(outer[analytic].sum())
            samples["analytic_pairs"] += int(analytic.sum())
            samples["regular_image_pairs"] += int((~analytic).sum())
            q2 = nodes(r2)**2
            samples["static_fp64"] += int(q2[~analytic & near].sum())
            samples["static_fp32"] += int(q2[~analytic & ~near].sum())
            if coefficients[0,image]:
                samples["vector_static"] += int(q2[~analytic].sum())
                samples["vector_analytic_nodes"] += int(outer[analytic].sum())
                samples["vector_regular_images"] += int((~analytic).sum())
                samples["vector_near_images"] += int(analytic.sum())
            if coefficients[1,image]:
                samples["scalar_regular_images"] += int((~analytic).sum())
                samples["scalar_near_images"] += int(analytic.sum())
    regular_samples = sum(v for k, v in samples.items() if k.startswith(("remainder_", "static_")))
    remainder = samples["remainder_fp64"] + samples["remainder_fp32"]
    static = samples["static_fp64"] + samples["static_fp32"]
    n, trees = prepared.layout.basis.n_edges, lt.trees
    nnz, charge_nnz = lt.sparse_q[1].numel(), lt.sparse_charge[1].numel()
    sparse_terms = 2 * n * nnz + (faces + trees) * charge_nnz
    return {"counts": samples, "nnz_transform": nnz, "nnz_charge": charge_nnz,
            "regular_face_pairs": {
                "logical_bytes": 128*(samples["remainder_fp64"]*((chunk_size+1)//2)/chunk_size + samples["remainder_fp32"]*((chunk_size+3)//4)/chunk_size)
                                 + 128*remainder + 320*faces**2,
                "real_FLOP": (270+40*((chunk_size+1)//2)/chunk_size)*samples["remainder_fp64"] + (300+40*((chunk_size+3)//4)/chunk_size)*samples["remainder_fp32"],
                "SFU": "rho/interval once per at most four far frequencies; fp64 scalar and selected-precision vector moments"},
            "horizontal_static_regular_geometry": {
                "logical_bytes": (96*static+160*faces**2)/chunk_size+288*samples["vector_regular_images"]+32*samples["scalar_regular_images"],
                "real_FLOP": (50*static+90*samples["vector_static"])/chunk_size+36*samples["vector_regular_images"]+4*samples["scalar_regular_images"]},
            "horizontal_static_near_geometry": {
                "logical_bytes": 32*samples["analytic_nodes"]/chunk_size+288*samples["vector_near_images"]+32*samples["scalar_near_images"],
                "real_FLOP": (420*samples["analytic_nodes"]+80*samples["vector_analytic_nodes"])/chunk_size+36*samples["vector_near_images"]+4*samples["scalar_near_images"]},
            "scalar_points_kernel": {"logical_bytes": (72*faces+48*faces*8*65)/chunk_size,
                                     "real_FLOP": 8*faces*8*65/chunk_size},
            "radial_metadata_kernel": {"logical_bytes": 40/chunk_size, "real_FLOP": 2/chunk_size},
            "gather_potentials": {"logical_bytes": 256*n*n + 32*faces*faces, "real_FLOP": 32*n*n},
            "sparse_product": {"logical_bytes": 16*sparse_terms + 32*(n*n + trees*trees),
                               "real_FLOP": 4*sparse_terms},
            "scaled_kernel": {"logical_bytes": 128*n*n, "real_FLOP": 70*n*n},
            "LU_complex64": {"real_FLOP": 8*n**3/3, "logical_bytes": 8*n*n}}


def profile_kernels(prepared, output, frequency_count=1):
    """Profile a real chunk and report amortized seconds per frequency."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        vector, scalar = assemble_potentials(prepared.layout, prepared.table, frequency_index=-1,
                                            frequency_stop=frequency_count, far_dtype=torch.float32,
                                            _network_accuracy=True)
        matrix, rhs, recovery = scale_potentials(prepared, vector, scalar)
        solution, _ = solve_scaled_chunk(matrix, rhs)
        recover_scaled_chunk(prepared, solution, rhs, recovery)
        torch.cuda.synchronize()
    trace = output.with_name(output.stem + "-kernels.json")
    profiler.export_chrome_trace(str(trace))
    groups = {}
    for event in json.loads(trace.read_text())["traceEvents"]:
        if event.get("cat") == "kernel":
            name = event["name"]
            groups[name] = groups.get(name, 0.) + event["dur"] * 1e-6 / frequency_count
    return groups


def fixture(unknowns):
    stack = LayerStack(
        (Layer("silicon", 300e-6, 11.9, 10.), Layer("oxide1", 2e-6, 4.1),
         Layer("oxide2", 3e-6, 4.1), Layer("nitride", .1e-6, 7.5),
         Layer("oxide3", 4e-6, 4.1)), HalfSpace(11.9, 10.), HalfSpace())
    # Binary spacing gives exactly N=46*M+28 for two eight-cell-wide strips
    # and two two-cell-wide port bridges. The dual graph has one component.
    h = 2.**-20
    cells = round((unknowns - 28) / 46)
    length, width, gap, bridge = cells * h, 8 * h, 4 * h, 2 * h
    z = stack.interfaces()[5]
    rectangles = [(0., length, gap / 2, gap / 2 + width, z),
                  (0., length, -gap / 2 - width, -gap / 2, z)]
    ports = [mw.LumpedPort(
        name, positive=(x, gap / 2, z), negative=(x, -gap / 2, z),
        voltage_path=mw.AxisPath("y"),
        current_surface=mw.Box(position=(x, 0., z), size=(bridge, gap, bridge)),
        reference_impedance=50.) for name, x in (("p1", h), ("p2", length - h))]
    frequencies = tuple(10.**(6 + 3 * i / 19) for i in range(20))
    return stack, rectangles, ports, frequencies, h, bridge, 46 * cells + 28


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unknowns", type=int, choices=(2000, 5000, 8000), required=True)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--frequency-start", type=int, default=0)
    parser.add_argument("--frequency-stop", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.frequency_start < args.frequency_stop <= 20:
        raise ValueError("Require 0 <= frequency-start < frequency-stop <= 20")
    measured = args.frequency_stop - args.frequency_start
    os.nice(10)
    torch.set_num_threads(8)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Use the GPU1 timing runner"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stack, rectangles, ports, frequencies, h, bridge, expected = fixture(args.unknowns)
    print(f"Preparing target N~{args.unknowns}, expected N={expected}, 20 frequencies", flush=True)
    peaks = device_peaks()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prepared = prepare_layered_network(
        stack, rectangles, ports, frequencies, bridge_width=bridge, max_edge=h)
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - started
    prepare_peak = torch.cuda.max_memory_allocated()
    assert prepared.layout.basis.n_edges == expected
    print(f"Prepared N={expected}, faces={len(prepared.layout.mesh.faces)} in {prepare_seconds:.3f}s", flush=True)

    # Rebuild once to isolate setup from Green-table construction, releasing
    # the first basis so the measured setup memory contains only one basis.
    del prepared.loop_tree
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    setup_time, prepared.loop_tree = timed(lambda: build_loop_tree(prepared.layout, length_scale=h))
    setup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    available = available_work_memory(prepared.excitation.device)
    capacity = frequency_chunk_capacity(expected, len(prepared.layout.mesh.faces), 20, available)
    if args.chunk_size is not None:
        capacity = min(capacity, args.chunk_size)
    capacity = min(capacity, measured)
    data = {
        "worktree": str(Path(__file__).resolve().parents[3]),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[3])):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().parents[3] /
                         "witwin/maxwell/mom/layered").rglob("*")
            if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".cuh")},
        "gpu": torch.cuda.get_device_name(), "physical_gpu": 1,
        "device_peaks": peaks,
        "target_unknowns": args.unknowns, "unknowns": expected,
        "frequency_start": args.frequency_start, "frequency_stop": args.frequency_stop,
        "faces": len(prepared.layout.mesh.faces), "ports": 2,
        "frequencies_Hz": frequencies, "prepare_seconds": prepare_seconds,
        "loop_tree_setup": setup_time, "prepare_peak_allocated_bytes": prepare_peak,
        "loop_tree_peak_allocated_bytes": setup_peak,
        "available_work_bytes": available, "frequency_chunk_size": capacity,
        "assembly_note": "V and P emitted together by the same face-pair pass",
        "chunks": [], "factor_dtype": [], "refinement_steps": [], "backward_error": []}
    torch.cuda.reset_peak_memory_stats()
    for start in range(args.frequency_start, args.frequency_stop, capacity):
        stop = min(start + capacity, args.frequency_stop)
        assembly_time, (vector, scalar) = timed(lambda: assemble_potentials(
            prepared.layout, prepared.table, frequency_index=-1,
            frequency_start=start, frequency_stop=stop, far_dtype=torch.float32,
            _network_accuracy=True))
        scale_time, (matrix, rhs, recovery) = timed(lambda: scale_potentials(
            prepared, vector, scalar, frequency_start=start))
        del vector, scalar
        solve_time, (solution, info) = timed(lambda: solve_scaled_chunk(matrix, rhs))
        recovery_time, (y, x) = timed(lambda: recover_scaled_chunk(
            prepared, solution, rhs, recovery, frequency_start=start))
        count = stop - start
        record = {"frequency_start": start, "frequency_stop": stop,
                  "assembly_VP": assembly_time, "scale": scale_time,
                  "LU_refinement": solve_time, "recovery": recovery_time,
                  "assembly_seconds_per_frequency": assembly_time["gpu_seconds"] / count,
                  "scale_seconds_per_frequency": scale_time["gpu_seconds"] / count,
                  "LU_refinement_seconds_per_frequency": solve_time["gpu_seconds"] / count}
        data["chunks"].append(record)
        data["factor_dtype"].extend(info[0])
        data["refinement_steps"].extend(info[1])
        data["backward_error"].extend(info[2])
        data["solve_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        data["solve_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        args.output.write_text(json.dumps(data, indent=2))
        print(f"N={expected} frequencies {start}:{stop}, "
              f"assembly={record['assembly_seconds_per_frequency']:.6f}s/f, "
              f"scale={record['scale_seconds_per_frequency']:.6f}s/f, "
              f"LU/refinement={record['LU_refinement_seconds_per_frequency']:.6f}s/f, "
              f"steps={info[1]}, dtype={info[0]}, "
              f"peak={data['solve_peak_allocated_bytes']/2**30:.3f}GiB", flush=True)
        del matrix, rhs, recovery, solution, y, x
    data["seconds_per_frequency"] = {
        name: sum(c[field]["gpu_seconds"] for c in data["chunks"]) / measured
        for name, field in (("assembly_VP", "assembly_VP"), ("scale", "scale"),
                            ("LU_refinement", "LU_refinement"), ("recovery", "recovery"))}
    data["peak_allocated_bytes"] = max(
        prepare_peak, setup_peak, data["solve_peak_allocated_bytes"])
    data["kernel_model"] = kernel_model(prepared, capacity)
    if args.profile_kernels:
        data["profile_frequency_count"] = capacity
        data["kernel_seconds"] = profile_kernels(prepared, args.output, capacity)
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps({"unknowns": expected, "seconds_per_frequency": data["seconds_per_frequency"],
                      "peak_GiB": data["peak_allocated_bytes"] / 2**30}), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
