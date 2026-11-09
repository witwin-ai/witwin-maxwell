"""Count captured thin-sheet kernels after gather/scatter fusion; no timings.

Only scene construction is imported from the overhead driver. Its measurement
functions are never called. Profiler durations are neither read nor exported.
Run one scenario per process through the prescribed GPU-0 Python launcher.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev/fdtd"))
from thin_sheet_overhead import SCENARIOS, _scene

import witwin.maxwell as mw
from witwin.maxwell.fdtd.cuda.build import STABLE_ABI_VERSION
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("copper_patch", "copper_board_60k"), required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    scratch = "/tmp/pcb-d3b-build"
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = scratch
    os.environ["TORCH_EXTENSIONS_DIR"] = scratch
    assert os.environ.get("WITWIN_MAXWELL_THIN_SHEET_KERNEL", "fused") == "fused"
    torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    prepared = mw.Simulation.fdtd(_scene(SCENARIOS[args.scenario], True), frequencies=[2.4e9],
                                  run_time=mw.TimeConfig(time_steps=1)).prepare()
    solver = prepared.solver
    sheet = solver._thin_sheets[0]
    replay = _make_field_update_runner(solver, True)
    assert solver._cuda_graph_active
    for _ in range(20):
        replay(0.)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("step1 captured kernel count"):
            replay(0.)
        torch.cuda.synchronize()
    events = prof.events()
    kernels = [event.name for event in events
               if event.device_type == torch.autograd.DeviceType.CUDA and not event.is_user_annotation]
    counts = Counter(kernels)
    gathers = sum(count for name, count in counts.items() if "thin_sheet_gather_kernel" in name)
    scatters = sum(count for name, count in counts.items() if "thin_sheet_scatter_kernel" in name)
    assert gathers == 4 and scatters == 3
    assert sum(event.name == "cudaGraphLaunch" for event in events) == 1
    result = dict(scenario=args.scenario, abi=STABLE_ABI_VERSION, selector="fused", device=0,
                  captured=True, profiled_steps=1, total_kernels=len(kernels),
                  phase0_total_kernels=263, gather_kernels=gathers, scatter_kernels=scatters,
                  gather_scatter_kernels=gathers + scatters, phase0_gather_scatter_kernels=77,
                  per_plan_calls=dict(patch_h=3, patch_e=2, h_cols=1, e_rows=1),
                  chebyshev_degree=sheet.copper["chebyshev"][2], chords=sheet.copper["count"],
                  kernel_counts=dict(counts), peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
