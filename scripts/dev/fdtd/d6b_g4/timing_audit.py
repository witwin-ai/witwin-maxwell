"""Audit independent FDTD clocks and reanalyse the saved D6b G4 traces.

Prepares public simulations without advancing fields. The stored probe traces
obey total[index] = scattered[index] + incident[index], even though the two
terms were sampled at different physical times. Recover total, assign its
actual clock, interpolate incident onto that clock, then fit the physical tail.
No solver or compiler modifications are made.
"""
from __future__ import annotations

import os
import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.d6b_g4 import isotropy as probe
from scripts.dev.fdtd.d6b_g4.root_cause import pole


def clock(scene):
    simulation = mw.Simulation.fdtd(scene, frequencies=[probe.F0],
                                    run_time=mw.TimeConfig(time_steps=1), backend="native")
    solver = simulation.prepare().solver
    return float(solver.dt)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out")
    parser.add_argument("--artifacts", default=str(Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "d6b-g4" / "v3"))
    parser.add_argument("--z", default="0.05,0.3,0.5,0.75")
    parser.add_argument("--dl", type=float, default=.00375)
    parser.add_argument("--floor", action="store_true", help="also reanalyse the saved no-floor trace")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(dl=args.dl, cases=[])
    for zoff in map(float, filter(None, args.z.split(","))):
        dt = {}
        for name, present in (("empty", False), ("loaded", True)):
            scene, _ = probe.scene_for(0, probe.pec_material(), args.dl,
                                       with_sheet=present, offset=(.37, .21), zoff=zoff)
            dt[name] = clock(scene)
        artifact = Path(args.artifacts)/f"zoff{zoff:g}-pec-0.npz"
        raw = np.load(artifact)
        ratio = dt["loaded"]/dt["empty"]
        t = raw["t"]*ratio
        total = raw["scattered"]+raw["incident"]
        incident = np.interp(t, raw["t"], raw["incident"])
        scattered = total-incident
        late = float(t[np.argmax(abs(incident))])+6/probe.F0
        result = dict(z=zoff, dt=dt, dt_ratio=ratio,
                      saved_dt=float(raw["t"][1]-raw["t"][0]),
                      original=probe.pencil_pole(raw["t"], raw["scattered"],
                          float(raw["t"][np.argmax(abs(raw["incident"]))])+6/probe.F0)[0],
                      corrected=pole(t, scattered, late),
                      total_only=pole(t, total, late), artifact=str(artifact))
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    if args.floor:
        # Recover the original short-cut experiment's actual clock. This is a
        # preparation-only, process-local override, matching the supplied run.
        import witwin.maxwell.fdtd.thin_sheet as runtime
        original_compiler = runtime.compile_thin_sheet
        runtime.compile_thin_sheet = functools.partial(original_compiler,
                                                        stiffness_ratio=1e6, minimum_area=1e-9)
        try:
            scene, _ = probe.scene_for(0, probe.pec_material(), args.dl, with_sheet=True,
                                       offset=(.37, .21), zoff=.05)
            loaded_dt = clock(scene)
        finally:
            runtime.compile_thin_sheet = original_compiler
        raw = np.load(Path(args.artifacts)/"floor-1e6-pec-0.npz")
        empty_dt = float(raw["t"][1]-raw["t"][0])
        ratio = loaded_dt/empty_dt
        t = raw["t"]*ratio
        total = raw["scattered"]+raw["incident"]
        incident = np.interp(t, raw["t"], raw["incident"])
        late = float(t[np.argmax(abs(incident))])+6/probe.F0
        report["floor"] = dict(dt_empty=empty_dt, dt_loaded=loaded_dt, dt_ratio=ratio,
                                legacy_widened=pole(raw["t"], raw["scattered"],
                                    float(raw["t"][np.argmax(abs(raw["incident"]))])+6/probe.F0),
                                corrected=pole(t, total-incident, late),
                                total_only=pole(t, total, late))
        print(json.dumps(report["floor"]), flush=True)
        output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
