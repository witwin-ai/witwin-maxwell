"""F-C scale series: fixture_fc.py cases in separate processes (one failure does not stop the series).

Usage: fixture_fc_series.py [--expandable] OUTDIR|- CASE [CASE ...]   (- = $RFIC_OUT)
    CASE = ROWSxCOLS[:backend][:MAX/EDGE], e.g. 1x2:matrix_free 1x2 2x2 3x3 4x4 (defaults hmatrix, 40/10).
Each case writes OUTDIR/fc-<case>.json, .json.y.pt and .log; OUTDIR/series.json lists rc and wall time per case.
--expandable runs every case with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (otherwise the
environment is passed through unchanged).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

argv = sys.argv[1:]
env = dict(os.environ)
if argv[0] == "--expandable":
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    argv = argv[1:]
out = Path(os.environ["RFIC_OUT"] if argv[0] == "-" else argv[0])  # "-": the cloud task's $RFIC_OUT
out.mkdir(parents=True, exist_ok=True)
script = Path(__file__).resolve().parent / "fixture_fc.py"
series = []
for case in argv[1:]:
    parts = case.split(":")
    rows, cols = (int(v) for v in parts[0].split("x"))
    backend = next((p for p in parts[1:] if p in ("hmatrix", "matrix_free")), "hmatrix")
    mesh = next((p for p in parts[1:] if "/" in p), "40/10")
    maximum, edge = mesh.split("/")
    tag = case.replace(":", "-").replace("/", "_")
    started = time.time()
    with open(out / f"fc-{tag}.log", "w") as log:
        rc = subprocess.run([sys.executable, "-u", str(script), str(out / f"fc-{tag}.json"), "--rows", str(rows),
                             "--cols", str(cols), "--backend", backend, "--maximum", maximum, "--edge", edge],
                            stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    series.append(dict(case=case, rc=rc, wall_seconds=time.time() - started))
    print(json.dumps(series[-1]), flush=True)
    (out / "series.json").write_text(json.dumps(series, indent=1))
