"""Record the phase-1a endpoint-owned rim sweeps with the shared-clock probe.

Run through the prescribed GPU-0 interpreter/environment. JSON evidence goes
in the requested output directory; trace arrays remain in a scratch directory.
Every propagation pair requests the same 39 ns physical duration. The default
grid wavelength is the assessment's nominal 0.3 m, keeping its grid phase.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.dev.fdtd.d6b_g4 import guided_mode, isotropy
from witwin.maxwell.scene import prepare_scene


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--traces", type=Path, default=Path("/tmp/sheet-rim-phase1a"))
    parser.add_argument("--grid-wavelength", type=float, default=.3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.traces.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    duration = 39e-9
    for resolution in (40, 60, 80, 120):
        dl = args.grid_wavelength / resolution
        cases = [("native", .5, "0", True), ("sheet", .5, "0", False)]
        if resolution in (80, 120):
            cases = [("native", .5, "0", True),
                     ("sheet", .5, "0,15,30,45", False),
                     ("planar", .5, "0,15,30,45", "planar")]
            cases += [("sheet", z, "0", False) for z in (.05, .3, .75)]
        for kind, z, angles, native in cases:
            name = f"{kind}-n{resolution}-z{z:g}"
            empty, _ = isotropy.scene_for(0., isotropy.pec_material(), dl, with_sheet=False, zoff=z)
            grid = prepare_scene(empty)
            nodes = [getattr(grid, axis + "_nodes64") for axis in "xyz"]
            steps = math.ceil(duration / isotropy.native_dt(empty))
            trace_file = args.traces / f"{name}.json"
            sys.argv = ["isotropy.py", str(trace_file), "--dl", str(dl), "--materials", "pec",
                        "--angles", angles, "--offset", "0.37,0.21", "--zoff", str(z),
                        "--steps", str(steps)]
            if native:
                sys.argv += ["--native"] + (["planar"] if native == "planar" else [])
            print(f"BEGIN {name}", flush=True)
            isotropy.main()
            record = json.loads(trace_file.read_text())
            record.update(resolution=resolution, kind=kind, zoff=z, offset=[.37, .21],
                          grid_wavelength=args.grid_wavelength, requested_duration=duration,
                          grid_shape=[len(n) for n in nodes],
                          actual_spacing=[float(np.diff(n).mean()) for n in nodes],
                          nearest_zero_node=[float(np.min(abs(n))) for n in nodes],
                          trace_file=str(trace_file))
            (args.out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"END {name}", flush=True)
    sys.argv = ["guided_mode.py", str(args.out / "guided.json"), "--z", "0,0.05,0.3,0.5,0.75"]
    guided_mode.main()


if __name__ == "__main__":
    main()
