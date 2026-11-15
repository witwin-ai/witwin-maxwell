"""Serial GPU ladder driver; --dry-run imports no solver or CUDA runtime.

Execute from the repository root with the maxwell environment. Each configuration
runs in a fresh child process, and each horizon has an independent preparation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.dev.fdtd.thin_wire_ladder.common import (
    CHECK_FREQUENCIES, F0, FREQUENCIES, GIB, build_scene, complex_pairs, resonance, runs,
)


class ProcessMemorySampler:
    """Sample whole-process device memory in addition to allocator high-water marks."""

    def __init__(self):
        self.peak = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5)
            for line in query.stdout.splitlines():
                pid, memory = (item.strip() for item in line.split(",", 1))
                if int(pid) == os.getpid():
                    size = int(memory)*1024**2
                    self.peak = max(self.peak or 0, size)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = f"{type(error).__name__}: {error}"

    def _loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(.25)

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.sample()


def _numpy(value):
    return value.detach().cpu().numpy()


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def static_diagnostics(solver, run):
    """Read coefficients from the actual prepared network; no state changes."""
    import numpy as np
    from scipy.constants import mu_0, epsilon_0
    network = solver._wire_runtime.network
    arrays = {name: _numpy(getattr(network, name)) for name in (
        "tail", "head", "length", "radius", "inductance", "capacitance_per_length",
        "coupling_distance", "radius_to_spacing", "node_capacitance", "node_positions",
        "port_negative_node_ids", "port_positive_node_ids", "segment_wire_ids", "segment_source_ids", "node_wire_ids")}
    node_ids = [int(arrays["port_negative_node_ids"][0]), int(arrays["port_positive_node_ids"][0])]
    caps = arrays["node_capacitance"][node_ids]
    near = np.flatnonzero(np.isin(arrays["tail"], node_ids) | np.isin(arrays["head"], node_ids))
    lp = arrays["inductance"]/arrays["length"]
    cp = arrays["capacitance_per_length"]
    def segment(j):
        tail, head = int(arrays["tail"][j]), int(arrays["head"][j])
        return dict(segment_id=int(j), wire=network.wire_names[int(arrays["segment_wire_ids"][j])],
                    source_segment=int(arrays["segment_source_ids"][j]), tail=tail, head=head,
                    tail_position_m=arrays["node_positions"][tail].tolist(),
                    head_position_m=arrays["node_positions"][head].tolist(),
                    delta_m=float(arrays["radius"][j]/arrays["radius_to_spacing"][j]),
                    nominal_grid_delta_m=run.delta, length_m=float(arrays["length"][j]),
                    a_m=float(arrays["radius"][j]), r0_m=float(arrays["coupling_distance"][j]),
                    log_r0_a=float(np.log(arrays["coupling_distance"][j]/arrays["radius"][j])),
                    radius_to_spacing=float(arrays["radius_to_spacing"][j]),
                    l_prime_H_m=float(lp[j]), c_prime_F_m=float(cp[j]),
                    tail_C_node_F=float(arrays["node_capacitance"][tail]),
                    head_C_node_F=float(arrays["node_capacitance"][head]),
                    lc_over_mu0_eps0=float(lp[j]*cp[j]/(mu_0*epsilon_0)))
    return dict(feed_node_ids=node_ids, feed_C_node_F=caps.tolist(),
                C_eff_F=float(1/np.sum(1/caps)),
                feed_node_positions_m=arrays["node_positions"][node_ids].tolist(),
                feed_nodes=[dict(node_id=node, C_node_F=float(arrays["node_capacitance"][node]),
                    dual_length_m=float(.5*np.sum(arrays["length"][(arrays["tail"] == node) | (arrays["head"] == node)])),
                    adjacent_C_prime_F_m=cp[(arrays["tail"] == node) | (arrays["head"] == node)].tolist())
                    for node in node_ids],
                feed_segments=[segment(j) for j in near],
                median=dict(l_prime_H_m=float(np.median(lp)), c_prime_F_m=float(np.median(cp)),
                            C_node_F=float(np.median(arrays["node_capacitance"])),
                            r0_m=float(np.median(arrays["coupling_distance"])),
                            log_r0_a=float(np.median(np.log(arrays["coupling_distance"]/arrays["radius"]))),
                            lc_over_mu0_eps0=float(np.median(lp*cp/(mu_0*epsilon_0)))),
                median_by_wire={name: dict(
                    l_prime_H_m=float(np.median(lp[arrays["segment_wire_ids"] == i])),
                    c_prime_F_m=float(np.median(cp[arrays["segment_wire_ids"] == i])),
                    C_node_F=float(np.median(arrays["node_capacitance"][arrays["node_wire_ids"] == i])))
                    for i, name in enumerate(network.wire_names)},
                max_abs_lc_relative_error=float(np.max(np.abs(lp*cp/(mu_0*epsilon_0)-1))),
                gap_segment=dict(length_m=float(np.linalg.norm(
                    arrays["node_positions"][node_ids[1]]-arrays["node_positions"][node_ids[0]])),
                    current_source="PortData.current; removed source segment has no wire runtime state")
                    if run.binding == "gap" else None)


def wire_readouts(result, solver, diagnostic):
    """Extract only feed-adjacent spectral quantities, avoiding large JSONs."""
    import numpy as np
    network = solver._wire_runtime.network
    spectra = {}
    for name in network.wire_names:
        data = result.monitor(f"wire_{name}")
        for column, node in enumerate(_numpy(data.metadata["node_ids"]).tolist()):
            if node in diagnostic["feed_node_ids"]:
                cap = float(_numpy(network.node_capacitance)[node])
                q = _numpy(data.charge[:, column])
                spectra[f"node_{node}"] = dict(q_C=complex_pairs(q), V_q_over_C=complex_pairs(q/cap), C_node_F=cap)
        selected = {s["segment_id"] for s in diagnostic["feed_segments"]}
        for column, segment in enumerate(_numpy(data.metadata["segment_ids"]).tolist()):
            if segment in selected:
                spectra[f"segment_{segment}"] = dict(current_A=complex_pairs(_numpy(data.current[:, column])))
    return dict(frequencies_Hz=CHECK_FREQUENCIES.tolist(), raw_phasor="exp(-i omega t)", quantities=spectra)


def execute(run, args):
    import numpy as np
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.runtime.stepping import _electric_field_energy
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires CUDA; use --dry-run on CPU")
    torch.cuda.set_device(args.device)
    free, _ = torch.cuda.mem_get_info()
    if free < run.estimated_bytes + 10*GIB:
        raise RuntimeError(f"Need estimate + 10 GiB free; free={free/GIB:.2f} GiB, estimate={run.estimated_bytes/GIB:.2f}")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    process_memory = ProcessMemorySampler()
    process_memory.start()
    scene = build_scene(run)
    frequencies = np.unique(np.concatenate((FREQUENCIES, CHECK_FREQUENCIES)))
    simulation = mw.Simulation.fdtd(
        scene, frequencies=tuple(frequencies),
        excitations=mw.rf.PortExcitation("feed", source_impedance="matched",
            source_time=mw.GaussianPulse(frequency=F0, fwidth=1.6e9)),
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=1),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none", normalize_source=False),
            full_field_dft=False, shutoff=1e-7 if args.horizon == "shutoff" else 0.,
            shutoff_check_interval=args.energy_interval, device=f"cuda:{args.device}",
        ),
    )
    prepared = simulation.prepare()
    solver = prepared.solver
    simulation.config = dataclasses.replace(
        simulation.config, run_time=mw.TimeConfig(time_steps=math.ceil(args.horizon_ns*1e-9/float(solver.dt))))
    actual_cells = tuple(int(len(getattr(solver.scene, f"{axis}_nodes64"))-1) for axis in "xyz")
    if actual_cells != run.cells:
        raise RuntimeError(f"Prepared grid {actual_cells} differs from dry-run {run.cells}")
    diagnostics = static_diagnostics(solver, run)
    result = prepared.run()
    torch.cuda.synchronize()
    port = result.port("feed")
    raw_z = _numpy(port.z_in)
    indices = np.searchsorted(frequencies, FREQUENCIES)
    check_indices = np.searchsorted(frequencies, CHECK_FREQUENCIES)
    z = raw_z[indices].conj()  # engineering e^(+jwt), as used by the oracle
    s11 = _numpy(port.reflection_coefficient)[indices]
    f_res, z_res = resonance(FREQUENCIES, z)
    final_energy = float(_electric_field_energy(solver).item())
    sampled_peak = float(solver._shutoff_peak.item()) if args.horizon == "shutoff" else None
    energy_ratio = final_energy/sampled_peak if sampled_peak and sampled_peak > 0 else None
    stats = result.stats()
    process_memory.finish()
    payload = dict(schema_version=1, run=run.manifest(), horizon=args.horizon, actual_cells=actual_cells,
        horizon_ns=args.horizon_ns, planned_steps=simulation.config.run_time.time_steps,
        dt_s=float(solver.dt), steps_run=stats["steps_run"], shutoff_triggered=stats["shutoff_triggered"],
        frequencies_Hz=FREQUENCIES.tolist(), voltage_raw_V=complex_pairs(_numpy(port.voltage)[indices]),
        current_raw_A=complex_pairs(_numpy(port.current)[indices]),
        z_raw_ohm=complex_pairs(raw_z[indices]), z_engineering_ohm=complex_pairs(z),
        s11_raw=complex_pairs(s11), raw_phasor=port.phasor_convention,
        engineering_phasor="exp(+i omega t)",
        check_frequencies_Hz=CHECK_FREQUENCIES.tolist(),
        z_check_engineering_ohm=complex_pairs(raw_z[check_indices].conj()),
        gap_current_raw_A=complex_pairs(_numpy(port.current)[indices]) if run.binding == "gap" else None,
        z_f0_ohm=complex_pairs(z[16]), f_res_Hz=f_res,
        z_res_ohm=complex_pairs(z_res) if z_res is not None else None,
        wire_static=diagnostics, wire_spectra=wire_readouts(result, solver, diagnostics),
        energy_qc=dict(definition="sum eps_r E^2, same unscaled electric metric as runtime shutoff",
            final=final_energy, sampled_peak=sampled_peak, final_over_peak=energy_ratio,
            sampling_interval_steps=args.energy_interval,
            limitation="Fixed-horizon peak is not retained by runtime" if sampled_peak is None else None),
        passivity=dict(max_abs_s11=float(np.max(np.abs(s11))),
            all_finite=bool(np.isfinite(raw_z).all() and np.isfinite(s11).all()),
            within_one=bool(np.isfinite(s11).all() and (np.abs(s11) <= 1+1e-6).all()), tolerance=1e-6),
        wall_seconds=time.perf_counter()-started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        memory_limit_bytes=4*GIB,
        sampled_process_peak_bytes=process_memory.peak,
        process_memory_sampling_interval_s=.25, process_memory_sampling_error=process_memory.error,
        memory_limit_pass=bool(process_memory.peak is not None
                               and max(process_memory.peak, torch.cuda.max_memory_reserved()) <= 4*GIB),
        memory_scope="Fresh process; exact PyTorch peaks plus sampled whole-process device memory; sampling may miss short non-PyTorch peaks",
        aliases=([f"{run.antenna}_G-phys_{run.binding}_r1_primary"] if run.level == 1 and run.gap_rule == "G1" and run.control == "primary" else []),
        limitations=["MoM independent table qualification pending", "F4 resonance estimator unavailable in checkout",
                    "Removed gap has no wire-current or line-constant state; port current is recorded",
                    "Fixed-horizon electric-energy peak unavailable read-only"])
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="*", help="comma-separated case-sensitive glob patterns over run names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/thin-wire-ladder"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--horizons", choices=("both", "shutoff", "fixed"), default="both")
    parser.add_argument("--horizon-ns", type=float, default=80.)
    parser.add_argument("--energy-interval", type=int, default=100)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--horizon", choices=("shutoff", "fixed"), default="shutoff", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.horizon_ns <= 0 or args.energy_interval < 1:
        parser.error("horizon and energy interval must be positive")
    selected = runs(args.only)
    if args.run_id:
        run = next(r for r in runs() if r.name == args.run_id)
        path = args.output/f"{run.name}_{args.horizon}.json"
        try:
            payload = execute(run, args)
            _dump(path, payload)
        except Exception as error:
            _dump(path, dict(schema_version=1, run=run.manifest(), horizon=args.horizon,
                             status="error", error=f"{type(error).__name__}: {error}"))
            raise
        return
    if not selected:
        parser.error("--only matched no runs")
    horizons = ("shutoff", "fixed") if args.horizons == "both" else (args.horizons,)
    probe = next(r for r in runs() if r.antenna == "D1" and r.level == 3)
    if any(r.level == 3 for r in selected):
        if probe not in selected:
            print(f"Adding required r3 memory probe: {probe.name}")
            selected.insert(0, probe)
        else:
            selected.remove(probe)
            first_r3 = next((i for i, r in enumerate(selected) if r.level == 3), len(selected))
            selected.insert(first_r3, probe)
    print("run | cells (Nx x Ny x Nz) | total cells | estimated GiB per process")
    for run in selected:
        print(f"{run.name} | {' x '.join(map(str, run.cells))} | {math.prod(run.cells):,} | {run.estimated_bytes/GIB:.3f}")
    print(f"{len(selected)} configurations; {len(selected)*len(horizons)} horizon executions; "
          "estimate = 256 bytes/cell + 256 MiB (sizing estimate, not a measured peak)")
    if args.dry_run:
        return
    # Gate every r3 configuration on the fresh-process D1 probe.
    r3_allowed = None
    for run in selected:
        if run.level == 3 and r3_allowed is False:
            print(f"SKIP {run.name}: r3 probe exceeded memory limit or failed")
            continue
        for horizon in horizons:
            command = [sys.executable, str(Path(__file__).resolve()), "--run-id", run.name,
                "--horizon", horizon, "--output", str(args.output), "--device", str(args.device),
                "--horizon-ns", str(args.horizon_ns), "--energy-interval", str(args.energy_interval)]
            completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
            path = args.output/f"{run.name}_{horizon}.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            memory_ok = data.get("memory_limit_pass", False)
            if run == probe:
                r3_allowed = memory_ok if r3_allowed is None else r3_allowed and memory_ok
            if completed.returncode or not memory_ok:
                if run.level == 3:
                    r3_allowed = False
                    print(f"SKIP remaining r3 runs: probe/run failed or memory gate unavailable/exceeded: {path}")
                    break
                raise SystemExit(f"Run failed or exceeded process memory limit: {path}; remaining runs stopped")


if __name__ == "__main__":
    main()
