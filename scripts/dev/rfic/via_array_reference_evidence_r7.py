"""R7-1 evidence for the via-array 2D BEM reference (tests/support/via_array_reference.py). Writes JSON to argv[1].

(1) Closed forms: coaxial line with the Leontovich impedance (R' = Rs (1/a + 1/b)/2pi, L' = mu0/2pi ln(b/a) + R'/omega)
    and the PEC-limit two-wire line (proximity factor), regular N-gons for N = 48/96/192: errors fall as N^-2.
(2) The gate geometries (2x2 array in an 8-via fence; L-shaped bar in the fence) at 1 GHz, copper: PEC-limit and
    impedance reference values side by side, with Rs/(omega L').
(3) Zs -> 0: the impedance solution approaches the PEC-limit L' and R' (relative gaps shrink by about 4-6x per decade of Rs).
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.support.via_array_reference import MU0, closed_form_checks, converged, solve  # noqa: E402
from tests.support.via_array_fixtures import via_outlines  # noqa: E402

UM, F, SIGMA = 1e-6, 1e9, 5.8e7
OMEGA, RS = 2 * math.pi * F, math.sqrt(math.pi * F * MU0 / SIGMA)
out = {"closed_forms": {}, "gate_geometries": {}, "zs_to_zero": {}}
for sides in (48, 96, 192):
    out["closed_forms"][sides] = dict(pec=closed_form_checks(sides),
                                      impedance=closed_form_checks(sides, omega=OMEGA, surface_resistance=RS))
for kind in ("array", "bar"):
    outlines = [[(x * UM, y * UM) for x, y in o] for o, _ in via_outlines(kind)]
    nets = [s for _, s in via_outlines(kind)]
    pec = converged(outlines, nets, surface_resistance=RS)
    imp = converged(outlines, nets, surface_resistance=RS, omega=OMEGA)
    out["gate_geometries"][kind] = dict(
        pec_limit=dict(R_per_m=pec["R_per_m"], L_ext_per_m=pec["L_per_m"], L_total_per_m=pec["L_per_m"] + pec["R_per_m"] / OMEGA,
                       R_change=pec["R_change"], L_change=pec["L_change"]),
        impedance=dict(R_per_m=imp["R_per_m"], L_per_m=imp["L_per_m"], R_change=imp["R_change"], L_change=imp["L_change"]),
        rs_over_omega_l=pec["R_per_m"] / (OMEGA * pec["L_per_m"]),
        reason="R' = Rs int K^2 with the PEC-limit K assumes omega L' >> R'. At Rs/(omega L') ~ 0.1 the wall "
               "resistance enters the current distribution (the Leontovich condition A_z - Zs K/(i omega) = const per "
               "net), relaxing proximity crowding; the MoM via walls use exactly this Leontovich model.")
    rows = []
    for scale in (1e-1, 1e-2, 1e-3, 1e-4):
        z = solve(outlines, nets, panels=24, surface_resistance=RS * scale, omega=OMEGA)
        p = solve(outlines, nets, panels=24, surface_resistance=RS * scale)
        rows.append(dict(rs_scale=scale, R_ratio=z["R_per_m"] / p["R_per_m"] - 1,
                         L_ratio=z["L_per_m"] / (p["L_per_m"] + p["R_per_m"] / OMEGA) - 1))
    out["zs_to_zero"][kind] = rows
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
