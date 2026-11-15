"""R7-1 via-array mesh sequence (cloud): array 2 um at 60/120, bar 2 um at 40/80, then array 1.5 um at 60/120 when
the device has >= 60 GB. Each case in its own subprocess; JSON and logs under $RFIC_OUT (default runs/r7-1)."""
import os, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("RFIC_OUT", ROOT / "runs" / "r7-1"))
OUT.mkdir(parents=True, exist_ok=True)
total = torch.cuda.get_device_properties(0).total_memory
CASES = [("a2", ("array", "2", "60", "120")), ("b2", ("bar", "2", "40", "80")), ("a15", ("array", "1.5", "60", "120"))]
rcs = {}
for name, case in CASES:
    if name == "a15" and total < 60e9:
        print(f"a15 skipped: device memory {total / 2**30:.1f} GiB < 60 GB", flush=True)
        continue
    with open(OUT / f"{name}.log", "w") as log:
        rcs[name] = subprocess.call([sys.executable, "-u", str(ROOT / "scripts/dev/rfic/via_array_probe_r7.py"), *case,
                                     str(OUT / f"{name}.json")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    print(name, "rc", rcs[name], flush=True)
sys.exit(max(rcs.values()))
