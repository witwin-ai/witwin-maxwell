"""Bounded, observable guard-on/off cost experiment for reviewer-owned CUDA runs.

Run from the repository root. Timings require an exclusively assigned device;
the deterministic storage/capture gate is test_guard_storage_bytes.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import faulthandler
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import passivity

SCENES = ("copper_board_60k", "copper_patch", "copper_foil_planar")
REPEATS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--device-index", type=int, default=0,
                        help="logical device after CUDA_VISIBLE_DEVICES filtering")
    parser.add_argument("--phase-timeout", type=float, default=600.)
    parser.add_argument("--phase-timeout-prepare", type=float, default=1200.)
    parser.add_argument("--memory-gib", type=float, default=12.)
    parser.add_argument("--json", type=Path, default=Path("guard_cost_ab.json"))
    parser.add_argument("--scene", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args(argv)
    if args.steps < 8*passivity.SAMPLE_INTERVAL:
        parser.error(f"--steps must allow at least eight guard samples ({8*passivity.SAMPLE_INTERVAL})")
    if (args.device_index < 0 or args.phase_timeout <= 0
            or args.phase_timeout_prepare <= 0 or args.memory_gib <= 0):
        parser.error("device index must be nonnegative; timeout and memory cap must be positive")
    return args


def verdict(fraction):
    return "PASS" if fraction <= .02 else "CHECK" if fraction <= .05 else "FAIL"


def _stats(values):
    median = statistics.median(values)
    return dict(median=median, min=min(values), max=max(values),
                spread=max(values)-min(values),
                relative_spread=(max(values)-min(values))/abs(median) if median else None, values=values)


@contextmanager
def _phase(label, timeout):
    print(f"{label} start", flush=True)
    started = time.monotonic()
    faulthandler.dump_traceback_later(timeout, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        print(f"{label} end wall_s={time.monotonic()-started:.3f}", flush=True)


def _build(name, steps):
    if name == "copper_foil_planar":
        from tests.materials.surface_impedance.test_copper_fused_kernel import _simulation
        simulation = _simulation("planar", steps=steps, graph=True)
        # The fixture's closed-transverse plane wave is a magnetic surface source,
        # which keeps the field-update block eager; an electric point source keeps
        # the same foil, grid and boundaries with graph capture in both arms.
        source = simulation.scene.sources[0]
        simulation.scene.sources = [mw.PointDipole(position=(-.1, 0., 0.), polarization="Ez",
                                                   source_time=source.source_time, name="pw")]
        return simulation
    from scripts.dev.fdtd.thin_sheet_overhead import SCENARIOS, _scene
    scene = _scene(SCENARIOS[name], True)
    scene = scene.clone(boundary=mw.BoundarySpec.faces(
        default="pml", num_layers=10, z=("pec", "pec")))
    return mw.Simulation.fdtd(
        scene, frequencies=[2.4e9], absorber="cpml", cuda_graph=True, shutoff=0.,
        run_time=mw.TimeConfig(time_steps=steps, transient_cycles=0))


def _run_arm(name, enabled, args, label, *, sample=False):
    original = passivity.make_passivity_state
    original_energy = passivity.total_energy
    sample_count = 0

    def counted_energy(solver):
        nonlocal sample_count
        sample_count += 1
        return original_energy(solver)

    passivity.total_energy = counted_energy
    if not enabled:
        passivity.make_passivity_state = lambda solver: None
    try:
        with _phase(f"{label} prepare", args.phase_timeout_prepare):
            simulation = _build(name, args.steps)
            assert simulation.config.cuda_graph is True, "cuda_graph must be requested in both arms"
            prepared = simulation.prepare()
            # Exclude one-time sampling-cache construction from run timing.
            state = original(prepared.solver) if enabled else None
            assert not enabled or state is not None, f"{label}: guard inactive"
            passivity.make_passivity_state = lambda solver, state=state: state
            torch.cuda.synchronize()
        with _phase(f"{label} run", args.phase_timeout):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = prepared.run()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            solver = prepared.solver
            assert solver._cuda_graph_active is True, f"{label}: CUDA graph capture inactive"
            if enabled:
                assert solver._passivity_state is not None, f"{label}: guard inactive"
                # Sampling also updates peak before source settling; those
                # energy reductions cost the same as envelope-window samples.
                assert sample_count >= 8, f"{label}: fewer than eight guard samples: {sample_count}"
            else:
                assert solver._passivity_state is None
                assert sample_count == 0
        sample_us = None
        if sample:
            # One sample per repeat, not 200 large sparse solves in a silent loop.
            with _phase(f"{label} energy sample", args.phase_timeout):
                start.record()
                energy = passivity.total_energy(solver)
                end.record()
                torch.cuda.synchronize()
                sample_us = 1000*start.elapsed_time(end)
                del energy
        print(f"{label} run_ms={elapsed:.6g} sample_us={sample_us} graph=True "
              f"run_samples={sample_count-int(sample)}", flush=True)
        del result, prepared, solver, simulation
        gc.collect()
        return elapsed, sample_us
    finally:
        passivity.make_passivity_state = original
        passivity.total_energy = original_energy


def main(argv=None):
    args = parse_args(argv)
    # Measure the counted copper/sheet functional that v0.6.0 skips by default.
    passivity.DEFERRED_STORAGE = ()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires reviewer-owned CUDA; CPU smoke uses parsing only.")
    torch.cuda.set_device(args.device_index)
    total = torch.cuda.get_device_properties(args.device_index).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., args.memory_gib*1024**3/total), args.device_index)
    report = dict(steps=args.steps, sample_interval=passivity.SAMPLE_INTERVAL,
                  repeats_per_arm=REPEATS, warmup_excluded=True, device_index=args.device_index,
                  memory_gib=args.memory_gib, phase_timeout=args.phase_timeout,
                  phase_timeout_prepare=args.phase_timeout_prepare, results={})
    for name in args.scene:
        print(f"scene={name} warm-up (excluded)", flush=True)
        for enabled in (True, False):
            _run_arm(name, enabled, args, f"{name} warmup guard={enabled}")
        on, off, samples = [], [], []
        for repeat in range(REPEATS):
            print(f"scene={name} repeat={repeat+1}/{REPEATS} AB", flush=True)
            elapsed, sample = _run_arm(name, True, args, f"{name} repeat={repeat+1} A", sample=True)
            on.append(elapsed)
            samples.append(sample)
            elapsed, _ = _run_arm(name, False, args, f"{name} repeat={repeat+1} B")
            off.append(elapsed)
        plain = [1000*x/args.steps for x in off]
        amortised = [x/passivity.SAMPLE_INTERVAL for x in samples]
        fractions = [a/p for a, p in zip(amortised, plain)]
        overhead = [(a-b)/b for a, b in zip(on, off)]
        fraction = statistics.median(amortised)/statistics.median(plain)
        record = dict(sample_us=_stats(samples), amortised_us_per_step=_stats(amortised),
                      plain_step_us=_stats(plain), amortised_fraction=fraction,
                      amortised_fraction_stats=_stats(fractions),
                      run_ms_guard_on=_stats(on), run_ms_guard_off=_stats(off),
                      measured_run_overhead=(statistics.median(on)-statistics.median(off))/statistics.median(off),
                      measured_run_overhead_stats=_stats(overhead), verdict=verdict(fraction),
                      cuda_graph_active_guard_on=True, cuda_graph_active_guard_off=True)
        report["results"][name] = record
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix+".tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.json)
        print(f"scene={name} complete {json.dumps(record)} json={args.json}", flush=True)


if __name__ == "__main__":
    main()
