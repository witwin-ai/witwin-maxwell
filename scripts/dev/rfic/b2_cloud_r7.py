"""R7-1 cloud driver: via-array de-embedding probes (mesh sequence) and the ExtrudedPolygon FDTD G1-style probe.

Writes one JSON per case and the logs under $RFIC_OUT (default runs/r7-1).
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("RFIC_OUT", ROOT / "runs" / "r7-1"))
OUT.mkdir(parents=True, exist_ok=True)
CASES = [("array", "4", "40", "80"), ("array", "2", "40", "80"), ("array", "4", "60", "120"), ("bar", "4", "40", "80")]
rcs = {}
for case in CASES:
    name = "via_array_" + "_".join(case)
    with open(OUT / f"{name}.log", "w") as log:
        rcs[name] = subprocess.call([sys.executable, "-u", str(ROOT / "scripts/dev/rfic/via_array_probe_r7.py"), *case,
                                     str(OUT / f"{name}.json")], stdout=log, stderr=subprocess.STDOUT)
    print(name, "rc", rcs[name], flush=True)
with open(OUT / "fdtd_g1.log", "w") as log:
    rcs["fdtd_g1"] = subprocess.call([sys.executable, "-u", str(ROOT / "scripts/dev/rfic/extruded_polygon_fdtd_g1_r7.py"),
                                      str(OUT / "fdtd_g1.json")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
print("fdtd_g1 rc", rcs["fdtd_g1"], flush=True)
sys.exit(max(rcs.values()))
