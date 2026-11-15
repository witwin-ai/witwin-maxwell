"""A4 (R7-2) MoM mesh campaign: every F-D / F-E / F-F producer record of the acceptance mesh sequences.

python a4_campaign_r7.py OUT_DIR [--only TAG ...] [--cap-gib 75] [--repeat] [--set a4|r2]
Writes OUT_DIR/<tag>.json (scripts/dev/rfic/a4_fixtures_r7.py records plus the producer arguments) and
OUT_DIR/campaign.json (device, torch, commit, per-tag seconds). Each record is written as soon as it finishes.
--repeat then runs the locked levels (0-2) again into OUT_DIR/reacquisition in the D-P1 acquisition layout of
scripts/dev/rfic/evidence_lock.py (a4/<part>/<tag>.json + acquisition.json), the re-acquisition that the first
renewal of the new locked records is judged on.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

import witwin.maxwell as mw

UM = 1e-6
# (fixture, edge_m, max_edge_m or None); mesh sequences halve the edge cell (F-F: ratio sqrt 2).
# Fixture by fixture (F-F, the largest, last); levels 0-2 are the locked acceptance sequences, level 3 a one-off
# check of the extrapolation; with --repeat each fixture's locked levels are re-acquired right after it.
LEVELS = {"fe-microstrip": [254 * UM / n for n in (16, 32, 64, 128)], "fe-cpw": [e * UM for e in (2.5, 1.25, .625, .3125)],
          "fd": [e * UM for e in (.5, .25, .125, .0625)], "ff": [e * UM for e in (2., 2 ** .5, 1., 2 ** -.5)]}
JOBS = [(fixture, edges[level], None, level) for fixture, edges in LEVELS.items() for level in range(4)]
# Second campaign (--set r2): F-F with the maximum edge refined with the edge cell (max = 2 edge; the edge-only sequence
# is not in the asymptotic range), levels 0-2 locked (fit a 15 GiB lane); F-E microstrip attribution of the L' offset: maximum
# (longitudinal) edge at W/32, and the air-filled line (exact eps_eff = 1).
R2 = [("fe-microstrip", 254 / 32 * UM, m * UM, 9) for m in (63.5, 31.75, 15.875)]
R2 += [("fe-microstrip-air", 254 / n * UM, None, 9) for n in (16, 32, 64)]
R2 += [("ff", e * UM, 2 * e * UM, level) for level, e in enumerate((2., 2 ** .5, 1., 2 ** -.5))]
LOCKED = {"a4": (0, 1, 2), "r2": (0, 1, 2)}


def tag(fixture, edge, maximum):
    return f"{fixture}-e{edge * 1e6:.6g}um" + (f"-m{maximum * 1e6:.6g}um" if maximum else "")


def producer(fixture, edge, maximum):
    from scripts.dev.rfic.evidence_lock import a4_record
    return a4_record(dict(fixture=fixture, edge=edge, max=maximum))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--cap-gib", type=float, default=75.)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--set", choices=("a4", "r2"), default="a4")
    args = parser.parse_args()
    print("witwin.maxwell", mw.__file__, flush=True)
    assert mw.__file__.startswith(str(Path(__file__).resolve().parents[3])), mw.__file__
    torch.cuda.set_per_process_memory_fraction(min(1., args.cap_gib * 2**30 /
                                                   torch.cuda.get_device_properties(0).total_memory))
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    log = dict(commit=commit, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
               seconds={}, peak_allocated_bytes={})
    def run(directory, layout, selected):
        for fixture, edge, maximum, level in jobs:
            name = tag(fixture, edge, maximum)
            if fixture != selected or (args.only and name not in args.only) or (layout and level not in locked):
                continue
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            record = producer(fixture, edge, maximum)
            record["producer"] = dict(fixture=fixture, edge=edge, max=maximum)
            seconds = time.time() - start
            out = directory / "a4" / fixture.split("-")[0] / (name + ".json") if layout else directory / (name + ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=1) + "\n")
            if not layout:
                log["seconds"][name] = seconds
                log["peak_allocated_bytes"][name] = torch.cuda.max_memory_allocated()
                (directory / "campaign.json").write_text(json.dumps(log, indent=1) + "\n")
            print(name, f"{seconds:.0f} s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    again = args.out / "reacquisition"
    jobs, locked = (JOBS, LOCKED["a4"]) if args.set == "a4" else (R2, LOCKED["r2"])
    locked_jobs = [job for job in jobs if job[3] in locked and (not args.only or tag(*job[:3]) in args.only)]
    finished = set()
    selected = [job for job in jobs if not args.only or tag(*job[:3]) in args.only]
    for fixture in dict.fromkeys(job[0] for job in selected):
        run(args.out, False, fixture)
        if args.repeat:
            run(again, True, fixture)
            finished.add(fixture)
            # Rewritten after every fixture: the parts list names only parts whose locked jobs are all re-acquired.
            parts = sorted({job[0].split("-")[0] for job in locked_jobs} -
                           {job[0].split("-")[0] for job in locked_jobs if job[0] not in finished})
            again.mkdir(parents=True, exist_ok=True)
            (again / "acquisition.json").write_text(json.dumps(dict(
                commit=commit, parts=parts,
                driver="scripts/dev/rfic/a4_campaign_r7.py --repeat", torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0)), indent=2) + "\n")
