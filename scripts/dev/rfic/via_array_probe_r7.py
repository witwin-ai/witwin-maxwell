"""Probe: coax via-array R/L de-embedding vs the 2D PEC-limit BEM. Args: kind edge h1 h2 [out.json]."""
import json, math, os, sys, time
WT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, WT)
import torch
import witwin.maxwell as mw, witwin.core
assert str(__import__("pathlib").Path(mw.__file__).resolve()).startswith(WT) and hasattr(witwin.core, "ExtrudedPolygon"), (mw.__file__, witwin.core.__file__)
print(mw.__file__, witwin.core.__file__, flush=True)
# Local shared GPU 0: the 16 GiB runner lane (both RTX A6000s report 47.99 GiB). Rented GPUs: no cap.
if os.environ.get("RFIC_OUT") is None:
    torch.cuda.set_per_process_memory_fraction(min(1.0, 15.0 / 47.99))
from tests.support.via_array_fixtures import measure_coax, via_outlines
from tests.support.via_array_reference import converged, MU0
from tests.support.rfic_planar_fixtures import UM
kind, edge, h1, h2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
frequencies = (1e6, 1e9)
sigma = 5.8e7
rows = {}
for h in (h1, h2):
    t = time.time()
    torch.cuda.reset_peak_memory_stats()
    m = measure_coax(h, kind, edge, frequencies, sigma)
    z = m["z"][1].item()
    walls = m["walls"]
    rows[h] = dict(R=z.real, L=-z.imag / (2 * math.pi * frequencies[1]), unknowns=m["unknowns"], faces=m["faces"],
                   kcl_1MHz=max(float((w - w[:1]).abs().max()) for w in walls[0]),
                   cluster_sum_1MHz=float(abs(abs(sum(w[0] for w, (_, s) in zip(walls[0], via_outlines(kind)) if s > 0)) - 1)),
                   split_mid_1GHz=[float(w[len(w) // 2].real) for w in walls[1]],
                   seconds=time.time() - t, peak=torch.cuda.max_memory_allocated())
    print(h, json.dumps(rows[h]), flush=True)
    del m
    torch.cuda.empty_cache()
outlines = [[(x * UM, y * UM) for x, y in o] for o, _ in via_outlines(kind)]
nets = [s for _, s in via_outlines(kind)]
rs = math.sqrt(math.pi * frequencies[1] * MU0 / sigma)
ref = converged(outlines, nets, surface_resistance=rs, omega=2 * math.pi * frequencies[1])
dh = (h2 - h1) * UM
dL, dR = rows[h2]["L"] - rows[h1]["L"], rows[h2]["R"] - rows[h1]["R"]
ref_L = dh * ref["L_per_m"]
ref_R = dh * ref["R_per_m"]
out = dict(kind=kind, edge=edge, h=(h1, h2), rows=rows, dL=dL, dR=dR, ref_dL=ref_L, ref_dR=ref_R,
           L_err=dL / ref_L - 1, R_err=dR / ref_R - 1, ref_L_change=ref["L_change"], ref_R_change=ref["R_change"],
           ref_split=ref["via_currents"].tolist())
print(json.dumps(out, default=str), flush=True)
if len(sys.argv) > 5:
    json.dump(out, open(sys.argv[5], "w"), default=str, indent=1)
