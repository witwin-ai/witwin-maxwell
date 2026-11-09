"""R1 kernel evidence: run only in the workstation's assigned timing window.

Use the timing GPU under its coordinator lock. This script intentionally does
not select a device, acquire a workstation lock, or alter any running process.
FLOP counts exclude transcendental operations; report them as an arithmetic
model, not an instruction count or a measured hardware-counter roofline.
"""

from collections import Counter
from time import perf_counter
import os
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from witwin.maxwell.layers import Layer, HalfSpace, LayerStack
from witwin.maxwell.mom.layered import LayeredGreenTable
from witwin.maxwell.mom.layered.build import build_extension
from witwin.maxwell.mom.layered.green import _GAUSS_NODES, _GAUSS_WEIGHTS


def bessel_special_counts(q, weights, kinds, rho, begin, end):
    # Exact counts of explicit complex exp/root calls, excluding libdevice J0
    # and the compensated small-argument series. Integer reductions are exact.
    counts = torch.zeros(2, dtype=torch.int64, device=q.device)
    for first in range(begin, end, 128):
        z = q[:, None, :] * rho[None, first:min(first + 128, end), None]
        kind = kinds[:, None, :]
        active = weights[:, None, :] != 0
        large = active & (kind == 0) & (z.abs() > 32)
        tail = active & (kind != 0) & (kind * z.imag <= 745)
        nl, nt = large.sum(), tail.sum()
        counts[0] += 2 * nl + nt
        counts[1] += nl + nt
    return counts


def timed(operation, repeats=5):
    operation()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    result = None
    begin.record()
    for _ in range(repeats):
        # Keep only one Bessel chunk alive across timed samples.
        del result
        result = operation()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1e-3 / repeats, result


def main():
    os.nice(10)
    torch.set_num_threads(8)
    ext = build_extension()
    anchor = torch.empty((), dtype=torch.float64, device="cuda")
    iterations = 8192
    fp64_time, _ = timed(lambda: ext.peak(anchor, iterations, True))
    fp32_time, _ = timed(lambda: ext.peak(anchor, iterations, False))
    work = 65536 * iterations * 16
    peak64, peak32 = work / fp64_time, work / fp32_time
    special_iterations = 256
    exp_time, _ = timed(lambda: ext.special_peak(anchor, special_iterations, 0))
    root_time, _ = timed(lambda: ext.special_peak(anchor, special_iterations, 1))
    divide_time, _ = timed(lambda: ext.special_peak(anchor, special_iterations, 2))
    special_work = 65536 * special_iterations * 4
    exp_peak, root_peak = special_work / exp_time, special_work / root_time
    divide_peak = special_work / divide_time
    print('Measured complex exp/sqrt/divide ceilings (calls/s):', exp_peak, root_peak, divide_peak, flush=True)
    source = torch.empty(32 * 1024**2, dtype=torch.float64, device="cuda")
    target = torch.empty_like(source)
    copy_time, copied = timed(lambda: target.copy_(source))
    bandwidth = 2 * source.numel() * source.element_size() / copy_time
    print(f"Measured peaks: fp64={peak64 / 1e9:.3f} GFLOP/s, "
          f"fp32={peak32 / 1e9:.3f} GFLOP/s, D2D={bandwidth / 1e9:.3f} GB/s")
    del source, target, copied

    stack = LayerStack((Layer("silicon", 300e-6, 11.9, 10), Layer("oxide1", 2e-6, 4.1),
                        Layer("oxide2", 3e-6, 4.1), Layer("nitride", .1e-6, 7.5),
                        Layer("oxide3", 4e-6, 4.1)), HalfSpace(11.9, 10), HalfSpace())
    interfaces = stack.interfaces()
    planes = (100e-6, 200e-6, interfaces[1], interfaces[2], 303e-6,
              interfaces[3], interfaces[4], interfaces[5])
    frequencies = np.geomspace(.1e9, 200e9, 50)
    torch.cuda.synchronize()
    start = perf_counter()
    table = LayeredGreenTable.build(stack, planes, frequencies, 2e-3, device="cuda", extract_distance=20e-6)
    torch.cuda.synchronize()
    build_time = perf_counter() - start
    print(f"Full table build (host metadata and CUDA, excluding JIT): {build_time:.3f} s; "
          f"50 frequencies, 36 pairs, {table.physical_nodes} rho nodes + 4 guards")
    image_counts = table.image_offsets.diff().cpu().tolist()
    histogram = Counter(image_counts)
    print('Image histogram (images per pair: pairs):', dict(sorted(histogram.items())), flush=True)
    print('Grouped images:', table.image_distances.numel(), '; extract_distance=20 um; T=', table.truncation,
          '; coarse admission nodes=', (table.physical_nodes + 1) // 2, flush=True)
    gl = torch.tensor(_GAUSS_NODES, dtype=torch.float64, device="cuda")
    gw = torch.tensor(_GAUSS_WEIGHTS, dtype=torch.float64, device="cuda")
    F, P, L = 50, len(planes) * (len(planes) + 1) // 2, len(stack.layers) + 2
    images_per_pair = table.image_distances.numel() / P
    from bisect import bisect_right
    sorted_planes = sorted(planes)
    segments = pair_exps = 0
    for source_index, zs in enumerate(sorted_planes):
        a = bisect_right(interfaces, zs)
        for zo in sorted_planes[source_index:]:
            b = bisect_right(interfaces, zo)
            pair_exps += int(a < L - 1) + int(a > 0)
            z = zs
            for layer in range(a, b + 1):
                target_z = zo if layer == b else interfaces[layer]
                if target_z != z:
                    segments += 1
                    pair_exps += 3 if layer < L - 1 else 1
                z = target_z
    image_count = table.image_distances.numel()
    # Pair is the fastest thread index. A warp runs the longest image list;
    # masked lanes still consume SIMD issue slots. Count one exact LCM period.
    period = math.lcm(P, 32)
    image_exp_slots = sum(32 * max(image_counts[(first + lane) % P] for lane in range(32))
                          for first in range(0, period, 32)) / (period // P)
    print('K1 image exp warp slots per (f,q):', image_exp_slots,
          '; active-lane fraction:', image_count / image_exp_slots, flush=True)
    spectral_exps = 2 * (L - 2) + pair_exps + image_count
    spectral_roots = L + P
    spectral_divides = 3 * L + 12 * (L - 1) + 6 * P + 3 * segments
    spectral_flops = 480 * (L - 1) + 40 * L + 250 * P + 220 * segments + 38 * image_count
    print('K1 per (f,q): complex exp=', spectral_exps, ', complex sqrt=', spectral_roots,
          '; complex divides=', spectral_divides, '; modeled explicit FLOP=', spectral_flops, flush=True)

    i, j = sorted(planes).index(interfaces[3]), sorted(planes).index(interfaces[4])
    pair = i * len(planes) - i * (i - 1) // 2 + j - i
    offsets = table.image_offsets[pair:pair + 2].cpu().tolist()
    query_images = offsets[1] - offsets[0]
    # Cacheable logical bytes and explicit arithmetic exclude libdevice work.
    print("Per-bin timings cover the single transform, including the coarse admission midpoints.")
    for index, plan in enumerate(table.quadrature_bins):
        k1_time = k2_time = gemm_time = accumulation_time = 0
        N, Q = plan.end - plan.begin, plan.nodes
        k2_counts = torch.zeros(2, dtype=torch.int64, device=table.rho.device)
        chunks = (Q + plan.chunk_nodes - 1) // plan.chunk_nodes
        total = torch.zeros((F, N, P * 2), dtype=torch.complex128, device="cuda")
        correction = torch.zeros_like(total)
        for first in range(0, Q, plan.chunk_nodes):
            count = min(plan.chunk_nodes, Q - first)
            q, weights, kinds = ext.nodes(gl, gw, table.frequencies, table.media, table.sigma,
                                          table.truncation, plan.rho_min, plan.rho_max,
                                          plan.linear_panels, first, count)
            seconds, spectrum = timed(lambda: table.spectral_samples(q, subtract=True))
            k1_time += seconds
            seconds, bessel = timed(lambda: ext.bessel(q, weights, kinds, table.rho,
                                                      plan.begin, plan.end, False))
            k2_time += seconds
            seconds, transformed = timed(lambda: torch.matmul(bessel, spectrum))
            gemm_time += seconds
            seconds, _ = timed(lambda: ext.accumulate(total, correction, transformed))
            accumulation_time += seconds
            del bessel, transformed
            k2_counts += bessel_special_counts(q, weights, kinds, table.rho, plan.begin, plan.end)
            del q, weights, kinds, spectrum
        del total, correction
        query = table.rho[plan.begin:min(plan.end, table.physical_nodes)]
        k3_time, _ = timed(lambda: table.evaluate(interfaces[3], interfaces[4], query))
        R = query.numel()
        special_floor = max(F * Q * spectral_exps / exp_peak,
                            F * Q * spectral_roots / root_peak,
                            F * Q * spectral_divides / divide_peak,
                            F * Q * image_exp_slots / exp_peak)
        ceiling = min(1., F * Q * spectral_flops / (special_floor * peak64))
        print('  K1 special-function floor:', special_floor * 1e3,
              'ms; explicit-FLOP/FMA-peak ceiling:', ceiling, flush=True)

        exps, roots = k2_counts.tolist()
        k2_floor = max(exps / exp_peak, roots / root_peak, 2 * roots / divide_peak)
        k2_ceiling = F * N * Q * 1200 / (max(k2_floor, F * N * Q * 1200 / peak64) * peak64)
        print('  K2 explicit complex exp/root calls:', exps, roots,
              '; special-function floor:', k2_floor * 1e3,
              'ms; model-FLOP/FMA-peak ceiling:', k2_ceiling, flush=True)

        print(f"Bin {index}: rho=[{plan.rho_min:.6g},{plan.rho_max:.6g}], "
              f"{Q} nodes in {chunks} chunks; B <= {plan.bessel_bytes / 1024**3:.6f} GiB")
        models = (
            ("K1 spectral", k1_time, F * Q * P * (32 + 24 * L + 56 * images_per_pair),
             F * Q * spectral_flops, "compute"),
            ("K2 Bessel/Hankel", k2_time, F * N * Q * 60, F * N * Q * 1200, "compute"),
            ("ZGEMM", gemm_time, 16 * F * (N * Q + Q * P * 2 + chunks * N * P * 2),
             8 * F * N * Q * P * 2, "compute"),
            ("Chunk accumulation", accumulation_time, 80 * chunks * F * N * P * 2,
             8 * chunks * F * N * P * 2, "bandwidth"),
            ("K3 interpolation", k3_time, F * R * 2 * (64 + 24 * query_images),
             F * R * 2 * (60 + 18 * query_images), "bandwidth"),
        )
        for name, seconds, byte_count, flops, resource in models:
            rate = flops / seconds / peak64 if resource == "compute" else byte_count / seconds / bandwidth
            print(f"  {name}: {seconds * 1e3:.3f} ms; model {byte_count:.0f} B, {flops:.0f} FLOP; "
                  f"{byte_count / seconds / 1e9:.3f} GB/s, {flops / seconds / 1e9:.3f} GFLOP/s; "
                  f"model/measured {resource} peak={rate:.3%}")
    print(f"Interpolation admission={table.interpolation_error.item():.6g}; "
          f"tail estimate max={table.tail_error_estimate.cpu().numpy().max():.6g}; "
          f"grouped images={table.image_distances.numel()}")


def benchmark_via():
    """RFIC envelope: report the incremental eight-z-node table cost."""
    from witwin.maxwell.mom.layered.vertical import VerticalGreenTable
    torch.set_num_threads(8)
    build_extension()
    stack = LayerStack((Layer("silicon", 300e-6, 11.9, 10), Layer("oxide1", 2e-6, 4.1),
                        Layer("oxide2", 3e-6, 4.1), Layer("nitride", .1e-6, 7.5),
                        Layer("oxide3", 4e-6, 4.1)), HalfSpace(11.9, 10), HalfSpace())
    edges = stack.interfaces()
    planes = (100e-6, 200e-6, edges[1], edges[2], 303e-6, edges[3], edges[4], edges[5])
    frequencies = np.geomspace(.1e9, 200e9, 50)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    table = LayeredGreenTable.build(stack, planes, frequencies, 2e-3, device="cuda",
                                   extract_distance=20e-6)
    torch.cuda.synchronize()
    baseline_seconds = perf_counter() - start
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_peak = torch.cuda.max_memory_allocated()
    baseline_reserved_peak = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    table.vertical = VerticalGreenTable.build(table, stack, ((edges[2], edges[3]),),
                                               z_nodes=8, extract_distance=20e-6)
    torch.cuda.synchronize()
    added_seconds = perf_counter() - start
    added_allocated = torch.cuda.memory_allocated() - baseline_allocated
    print(dict(frequencies=50, via_spans=((edges[2], edges[3]),), z_nodes=8,
               baseline_seconds=baseline_seconds, added_build_seconds=added_seconds,
               baseline_peak_bytes=baseline_peak, baseline_peak_reserved_bytes=baseline_reserved_peak,
               added_resident_bytes=added_allocated,
               combined_peak_bytes=torch.cuda.max_memory_allocated(),
               combined_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               z_admission=table.vertical.z_interpolation_error.item(),
               rho_admission=table.vertical.rho_interpolation_error.item()), flush=True)


if __name__ == "__main__":
    benchmark_via() if "--via" in sys.argv else main()
