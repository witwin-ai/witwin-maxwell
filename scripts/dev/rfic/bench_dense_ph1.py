"""PH1 dense layered stage timing, kernel profile and roofline ratios on GPU 1.

Invoke through the GPU 1 timing runner (<= 30 min per lock hold). Each
case prepares once, then times one frequency per chunk through the public
dense pipeline: in-place system assembly (V and packed P), loss fold,
congruence, equilibration, complex64 factor (cast + in-place LU), complex128
refinement and recovery. A profiled repeat attributes CUDA kernels; ratios
compare each kernel's logical model with the measured device peaks.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["OMP_NUM_THREADS"] = "8"
import torch

from scripts.dev.rfic.bench_layered_network import device_peaks, timed
from tests.support.rfic_fixtures import fixture_fa
import witwin.maxwell as mw
from witwin.maxwell.mom.layered import lowfreq
from witwin.maxwell.mom.layered.network import recover_scaled_chunk, _wall_nodes
from witwin.maxwell.mom.layered.operator import dense_chunk_bytes, dense_frequency_bytes


def simulation(case, frequencies, fb_fixture):
    if case == "fb_fine":
        sys.path.insert(0, str(Path(fb_fixture).parent))
        from fb_fixture import fixture_fb
        return fixture_fb(mesh="full").simulation(frequencies)
    fixture = fixture_fa(substrate=case != "fa_closed_fine", mesh="test" if case == "fa_test" else "full")
    if case == "fa_closed_fine":
        layered = dataclasses.replace(fixture.config.layered, conductor_model="closed")
        return mw.Simulation.mom(fixture.scene, frequencies=frequencies,
                                 config=dataclasses.replace(fixture.config, layered=layered))
    return fixture.simulation(frequencies)


def stages(prepared, start):
    op, lt = torch.ops.witwin_maxwell_mom_layered, prepared.loop_tree
    out = {}
    out["assembly"], system = timed(lambda: lowfreq.assemble_chunk(
        prepared, frequency_start=start, frequency_stop=start + 1))
    if prepared.loss is not None:
        sheet = prepared.loss
        out["loss_fold"], _ = timed(lambda: op.loss_fold(
            system.matrix, sheet.blocks.contiguous(), sheet.columns.contiguous(), sheet.supports.contiguous(),
            sheet.impedance[start:start + 1].contiguous(), lt.order, system.frequencies))
    out["congruence"], _ = timed(lambda: op.congruence(system.matrix, lt.sparse_q, lt.order, lt.loops))
    out["drive"], drive = timed(lambda: lowfreq.port_drive(prepared))
    out["equilibration"], (system.rhs, system.recovery) = timed(lambda: op.scale_system(
        system.matrix, system.workspace, lt.sparse_charge, drive, system.frequencies, lt.loops, lt.omega0))
    out["factor"], _ = timed(system.factor)
    out["refinement"], (solution, info) = timed(system.solve)
    out["recovery"], _ = timed(lambda: recover_scaled_chunk(prepared, solution, system.rhs, system.recovery,
                                                            frequency_start=start))
    out["total_gpu_seconds"] = sum(v["gpu_seconds"] for v in out.values())
    out["refinement_steps"], out["factor_dtype"] = info[1][0], info[0][0]
    return out


def kernel_profile(prepared, start):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        system = lowfreq.scaled_system(prepared, frequency_start=start, frequency_stop=start + 1)
        system.factor()
        solution, _ = system.solve()
        recover_scaled_chunk(prepared, solution, system.rhs, system.recovery, frequency_start=start)
        torch.cuda.synchronize()
    groups = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            groups[event.name] = groups.get(event.name, 0.) + event.device_time_total * 1e-6
    return groups


def model(prepared):
    """Logical bytes / real FLOP of the new or changed kernels for one frequency."""
    lt = prepared.loop_tree
    n, faces, trees, ports = len(lt.norms), len(prepared.layout.mesh.faces), lt.trees, prepared.excitation.shape[1]
    loop_terms = int(lt.sparse_q[0][lt.loops].item())
    tree_pairs_charge = 4 * trees * trees
    return {
        "gather_system": dict(bytes=256 * n * n + 16 * faces * faces, flop=32 * n * n, resource="bytes"),
        "loss_fold_kernel": dict(bytes=6 * 40 * n + 6 * 32 * n, flop=6 * 14 * n, resource="bytes"),
        "congruence_rows_kernel": dict(bytes=16 * n * (loop_terms + lt.loops), flop=4 * n * loop_terms,
                                       resource="bytes"),
        "congruence_columns_kernel": dict(bytes=16 * n * (loop_terms + lt.loops), flop=4 * n * loop_terms,
                                          resource="bytes"),
        "system_diagonal_kernel": dict(bytes=n * (48 + 32 * ports) + 64 * trees, flop=40 * n, resource="bytes"),
        "system_entries_kernel": dict(bytes=32 * n * n + 16 * tree_pairs_charge,
                                      flop=40 * n * n + 4 * tree_pairs_charge, resource="bytes"),
        "factor_cast_kernel": dict(bytes=24 * n * n, flop=0, resource="bytes"),
        "LU_complex64": dict(bytes=8 * n * n, flop=8 * n**3 / 3, resource="fp32"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("fa_test", "fa_full", "fa_closed_fine", "fb_fine"), required=True)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[1e9])
    parser.add_argument("--fb-fixture", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Use the GPU-1 timing runner"
    os.nice(10)
    torch.set_num_threads(8)
    peaks = device_peaks()
    torch.cuda.synchronize()
    started = time.perf_counter()
    prepared = simulation(args.case, tuple(args.frequencies), args.fb_fixture).prepare().solver.prepared
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - started
    lt = prepared.loop_tree
    n, faces, ports = len(lt.norms), len(prepared.layout.mesh.faces), prepared.excitation.shape[1]
    data = dict(gpu=torch.cuda.get_device_name(), physical_gpu=1, case=args.case, device_peaks=peaks,
                unknowns=n, faces=faces, ports=ports, loops=lt.loops, prepare_seconds=prepare_seconds,
                prepare_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                estimate_frequency_bytes=dense_frequency_bytes(n, faces, ports),
                estimate_chunk_bytes=dense_chunk_bytes(faces, _wall_nodes(prepared.table)),
                frequencies_Hz=args.frequencies, stages=[])
    for start in range(len(args.frequencies)):
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        record = stages(prepared, start)
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() - base
        data["stages"].append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(data, indent=2))
    kernels = kernel_profile(prepared, 0)
    data["kernel_seconds"] = kernels
    roofline = {}
    for name, entry in model(prepared).items():
        if name == "LU_complex64":
            # Vendor LU: the factor stage minus the profiled cast kernel.
            seconds = data["stages"][0]["factor"]["gpu_seconds"] - sum(
                v for k, v in kernels.items() if "factor_cast_kernel" in k)
        else:
            seconds = sum(v for k, v in kernels.items() if name in k)
        if seconds <= 0:
            continue
        rate = entry["bytes"] / seconds if entry["resource"] == "bytes" else entry["flop"] / seconds
        peak = (peaks["copy"]["bytes_per_second"] if entry["resource"] == "bytes"
                else peaks["float32"]["FLOP_per_second"])
        roofline[name] = dict(seconds=seconds, logical_bytes=entry["bytes"], real_FLOP=entry["flop"],
                              dominant=entry["resource"], achieved=rate, peak=peak, ratio=rate / peak)
    data["roofline"] = roofline
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps(roofline, indent=1), flush=True)


if __name__ == "__main__":
    main()
