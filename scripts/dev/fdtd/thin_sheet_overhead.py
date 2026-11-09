"""Per-step cost of the conformal thin-sheet patch on the native CUDA backend.

For each scenario, prepare the same grid and dt with and without the sheet.
Time their captured field-update graphs back to back with CUDA events, in
alternating order across three pairs. Preparation, capture, sources and monitor
sampling are outside the timing. Record GPU load with each pair. Run on GPU 0:

    python scripts/dev/fdtd/thin_sheet_overhead.py [--scenario copper_patch pec_board] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner

SCENARIOS = {
    # 2.4 GHz-class tilted copper patch (35 um) on a 0.5 mm grid (lambda/250).
    "copper_patch": dict(half=(0.012, 0.010), cell=5e-4, extent=0.032, material="copper"),
    # Tilted PEC board 30 x 25 mm on the same 0.5 mm grid.
    "pec_board": dict(half=(0.015, 0.0125), cell=5e-4, extent=0.032, material="pec"),
    # Trace-count scaling: a slightly tilted 130 x 110 mm, 35 um copper board
    # (about 6e4 traces) in a flat domain, so the cut-complex box stays thin.
    "copper_board": dict(half=(0.065, 0.055), cell=5e-4, extent=(0.075, 0.065, 0.02), material="copper",
                         normal=(0.04, 0.025, 1.0)),
    # The same board at 90 x 80 mm (about 6e4 traces).
    "copper_board_60k": dict(half=(0.045, 0.04), cell=5e-4, extent=(0.055, 0.05, 0.02), material="copper",
                             normal=(0.04, 0.025, 1.0)),
}
NORMAL = np.array((1.0, 0.6, 0.35)) / np.linalg.norm((1.0, 0.6, 0.35))


def _normal(spec):
    normal = np.asarray(spec.get("normal", NORMAL), dtype=np.float64)
    return normal / np.linalg.norm(normal)


def _mesh(half, normal=NORMAL, **kwargs):
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p = np.array((1.3e-4, -0.7e-4, 0.4e-4))
    corners = np.stack([p - half[0] * u - half[1] * v, p + half[0] * u - half[1] * v,
                        p + half[0] * u + half[1] * v, p - half[0] * u + half[1] * v])
    return mw.SurfaceMesh(corners, [(0, 1, 2), (0, 2, 3)], **kwargs)


def _scene(spec, with_sheet, *, device="cuda"):
    structures = []
    if with_sheet:
        if spec["material"] == "pec":
            structures.append(mw.Structure(geometry=_mesh(spec["half"], _normal(spec), sheet_model="conformal_two_sided"),
                                           material=mw.Material.pec() if hasattr(mw.Material, "pec") else mw.PEC))
        else:
            structures.append(mw.Structure(geometry=_mesh(spec["half"], _normal(spec)),
                                           material=mw.CopperFoil(thickness=35e-6, frequency_range=(1e9, 4e9))))
    extent = np.broadcast_to(np.asarray(spec["extent"], dtype=np.float64), (3,))
    offset = (0.8 if extent.min() == extent.max() else 0.6) * extent.min() * _normal(spec)
    return mw.Scene(
        domain=mw.Domain(bounds=tuple((-float(e), float(e)) for e in extent)), grid=mw.GridSpec.uniform(spec["cell"]),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=10), device=device,
        sources=[mw.PointDipole(position=tuple(-offset), polarization=(0.3, 0.5, 0.8), profile="ideal",
                                source_time=mw.GaussianPulse(frequency=2.4e9, fwidth=1.5e9))],
        structures=structures,
        monitors=[mw.PointMonitor("p", tuple(offset), fields=("Ez",))])


def _gpu_load():
    return {
        "gpu": subprocess.check_output([
            "nvidia-smi", "-i", "0", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"], text=True).strip(),
        "processes": subprocess.check_output([
            "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_gpu_memory,process_name",
            "--format=csv,noheader,nounits"], text=True).strip(),
    }


def _operator_bytes(cu):
    # Chord matrices, CSR indices, 2x2 preconditioner and diagonal coefficients.
    sparse_bytes = sum(t.numel() * t.element_size()
                       for key in ("Acc", "Acx", "Q", "S", "h_chord", "R")
                       for t in (cu[key].crow_indices(), cu[key].col_indices(), cu[key].values()))
    return sparse_bytes + sum(cu[key].numel() * cu[key].element_size()
                              for key in ("mass", "conductance", "area", "held", "effective"))


def _event_ms(run, steps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        run(0.0)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / steps


def measure(name, steps=200, warmup=20):
    spec = SCENARIOS[name]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = mw.Simulation.fdtd(_scene(spec, True), frequencies=[2.4e9],
                                  run_time=mw.TimeConfig(time_steps=1)).prepare()
    torch.cuda.synchronize()
    prepare_s = time.perf_counter() - start
    solver = prepared.solver
    sheet = solver._thin_sheets[0]
    dt = float(solver.dt)
    record = dict(scenario=name, grid=(solver.Nx, solver.Ny, solver.Nz), dt=dt, prepare_s=prepare_s,
                  traces=int(sheet.complex.statistics["traces"]),
                  chords=len(sheet.complex.chord_face), patch_e=len(sheet.patch_e), patch_h=len(sheet.patch_h),
                  limiting_factor=solver._time_step_metadata.get("limiting_factor"),
                  prepare_peak_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
    phase = np.pi * 2.4e9 * dt
    record["newmark_minus_leapfrog_rad_per_step"] = float(2 * (np.arctan(phase) - np.arcsin(phase)))
    if sheet.copper is not None:
        cu = sheet.copper
        operator_bytes = _operator_bytes(cu)
        record.update(chord_operator_bytes=operator_bytes,
                      chord_operator_bytes_per_trace=operator_bytes / record["traces"],
                      modal_state_bytes=cu["state"].numel() * cu["state"].element_size(),
                      chebyshev_degree=cu["chebyshev"][2], spectrum=cu["spectrum"])
    # One public step initializes the runtime before capturing the timed blocks.
    results = {"sheet": prepared.run()}
    results["plain"] = mw.Simulation.fdtd(_scene(spec, False), frequencies=[2.4e9],
                                         run_time=mw.TimeConfig(time_steps=1, dt=dt)).run()
    runs = {key: _make_field_update_runner(result.solver, True) for key, result in results.items()}
    for key, result in results.items():
        assert result.solver._cuda_graph_active
        record[f"{key}_graph"] = True
        for _ in range(warmup):
            runs[key](0.0)
    torch.cuda.synchronize()
    pairs = []
    for pair in range(3):
        sample = dict(before=_gpu_load())
        order = ("plain", "sheet") if pair % 2 == 0 else ("sheet", "plain")
        for key in order:
            sample[f"{key}_ms"] = _event_ms(runs[key], steps)
        sample.update(ratio=sample["sheet_ms"] / sample["plain_ms"], after=_gpu_load())
        pairs.append(sample)
    record.update(pairs=pairs, steps=steps, warmup=warmup,
                  plain_ms_per_step=float(np.median([p["plain_ms"] for p in pairs])),
                  sheet_ms_per_step=float(np.median([p["sheet_ms"] for p in pairs])),
                  step_ratio=float(np.median([p["ratio"] for p in pairs])),
                  peak_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--json", default=None)
    parser.add_argument("--steps", type=int, default=200, help="graph replays per timing sample")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--memory-gib", type=float, default=None, help="cap the CUDA allocator of this process")
    args = parser.parse_args()
    if args.memory_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gib / total))
    records = []
    for name in args.scenario:
        record = measure(name, args.steps, args.warmup)
        print(json.dumps(record), flush=True)
        records.append(record)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()
