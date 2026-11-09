"""R2a assembly evidence for exactly N=2000,5000,10000 RWG unknowns.

Run only in an explicitly assigned timing window on the selected device. This
script selects no device and acquires no workstation lock. Logical bytes/FLOP
models exclude cache reuse and transcendental instructions, as in R1. No
performance qualification is implied by the model; retain the profiler trace.
"""

import tempfile
import argparse
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered import LayeredGreenTable, assemble_layered, mesh_rectangles
from witwin.maxwell.mom.layered.build import build_extension


def timed(operation):
    warmup = operation()
    torch.cuda.synchronize()
    del warmup
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    result = operation()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 1000, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "witwin-rfic-r2a")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ext = build_extension()
    anchor = torch.empty((), device="cuda", dtype=torch.float64)
    iterations = 8192
    peak64_time, _ = timed(lambda: ext.peak(anchor, iterations, True))
    peak32_time, _ = timed(lambda: ext.peak(anchor, iterations, False))
    source = torch.empty(32 * 1024**2, device="cuda", dtype=torch.float64)
    target = torch.empty_like(source)
    copy_time, _ = timed(lambda: target.copy_(source))
    print(f"Measured fp64/fp32 peaks: {65536 * iterations * 16 / peak64_time / 1e9:.3f}/"
          f"{65536 * iterations * 16 / peak32_time / 1e9:.3f} GFLOP/s; "
          f"D2D={2 * source.numel() * source.element_size() / copy_time / 1e9:.3f} GB/s")
    del source, target
    stack = LayerStack((Layer("silicon", 300e-6, 11.9, 10.), Layer("oxide1", 2e-6, 4.1),
                        Layer("oxide2", 3e-6, 4.1), Layer("nitride", .1e-6, 7.5),
                        Layer("oxide3", 4e-6, 4.1)), HalfSpace(11.9, 10.), HalfSpace())
    planes = (stack.interfaces()[1], stack.interfaces()[5])
    table = LayeredGreenTable.build(stack, planes, (10e9,), 100e-6, device="cuda")
    # For an nx*ny sheet N=3*nx*ny-nx-ny. Binary spacing makes the requested
    # cell counts exact, with no ceil-rounding ambiguity in mesh setup.
    shapes = ((2000, (13, 16), (21, 23)), (5000, (10, 10), (40, 40)),
              (10000, (30, 33), (47, 51)))
    h, source_tile, far_order, near_order = 2**-20, 64, 8, 32
    for unknowns, first, second in shapes:
        layout = mesh_rectangles(((0, first[0] * h, 0, first[1] * h, planes[0]),
                                  (0, second[0] * h, 0, second[1] * h, planes[1])), table.planes, h)
        assert layout.basis.n_edges == unknowns
        faces = len(layout.mesh.faces)
        torch.cuda.synchronize()
        start = time.perf_counter()
        z = assemble_layered(layout, table, source_tile=source_tile,
                              far_order=far_order, near_order=near_order)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        del z
        # Profiling is separate from wall time and retains every kernel event.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            z = assemble_layered(layout, table, source_tile=source_tile,
                                  far_order=far_order, near_order=near_order)
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(args.output / f"assembly-{unknowns}.json"))
        kernel_ms = {}
        for event in trace.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernel_ms[event.name] = kernel_ms.get(event.name, 0.) + event.device_time_total / 1000
        tiles = (faces + source_tile - 1) // source_tile
        pairs, products = faces**2, far_order**4
        print(f"N={unknowns}, faces={faces}, tiles={tiles}: assembly={elapsed:.6f} s; "
              f"Z={z.numel() * z.element_size()} B, tile<={faces * source_tile * 160} B")
        print(f"Model parameters: P={pairs} face pairs, q={products} products/pair, "
              f"Qn={near_order**2}; H=log2({table.rho.numel()}); I_p=images for pair p; "
              "A=sum of admitted nonzero near-image pairs (geometry-dependent).")
        models = {
            "regular_face_pairs": (
                "sum_p[256+q*(64+2*(80+8H)+8*I_p+32*I_far,p)+160] B; "
                "sum_p q*(310+18*I_far,p) FLOP + sqrt/exp/search"),
            "near_image_pairs": (
                f"P*(256+320)+sum_p 40*I_p+32*A*{near_order**2} B; "
                f"590*A*{near_order**2}+300*P FLOP + log/asinh/atan2/sqrt"),
            "gather_mp_ie": (
                f"<={576 * unknowns**2} B; "
                f"<={256 * unknowns**2} FLOP (at most two visits/entry across all tiles)"),
        }
        for kernel, model in models.items():
            milliseconds = sum(ms for name, ms in kernel_ms.items() if kernel in name)
            assert milliseconds > 0, f"Missing CUDA profiler events for {kernel}"
            print(f"  {kernel}: {milliseconds:.3f} ms; logical model: {model}")
        del z


if __name__ == "__main__":
    main()
