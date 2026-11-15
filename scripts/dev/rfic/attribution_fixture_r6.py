"""Quantified attribution of the R6 no-shield L / Q-peak gaps (R6-1, CPU only).

Exact pi split of a reciprocal two-port: series Ys = -(Y12+Y21)/2, shunt
Y1 = Y11 + Y21. The R6 quantities depend on them only through
L = -Im(1/Ys)/omega and Q11 = -Im(1/Y11)/Re(1/Y11), Y11 = Ys + Y1, so swapping a
branch (or the R / L part of Zs = 1/Ys, or the G / B part of Y1) between the
native network and the open-source FDTD reference attributes the Q-peak gap
term by term; every swap is evaluated on the native sample frequencies with
the reference branches interpolated linearly from its 49.9-MHz grid.

The reference's own grid bias is estimated from its PEC controls (same driver,
5 -> 2.5 um lateral pitch, R6b): the branch shifts Y(PEC 2.5) - Y(PEC 5) are
added to the 5-um copper branches ("reference + 2.5-um PEC grid shift"). That
is a conditional estimate (PEC shifts applied to copper; one refinement step),
not a converged reference.

  python scripts/dev/rfic/attribution_fixture_r6.py OUT.json --native A.json [B.json ...] [--label sheet]

--transfer-series SA.json --transfer-base SHEET.json [SHEET2.json ...] adds the case
"native_series_plus_transfer": the native series impedance plus the unshielded
surface_admittance-minus-sheet series impedance change, interpolated to the native
frequencies (estimate of the shielded variant under surface_admittance, which
refuses the poly-shield/copper-via contacts; additive in Zs, shield eddy-current
change of Zs assumed model-independent).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

EVIDENCE = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106"
REFERENCE = {"cu5": "codex/r6/reference-none-5/network.json", "cu10": "codex/r6/reference-none-10/network.json",
             "pec5": "codex/r6/controls/reference-none-5-pec/network.json",
             "pec2p5": "codex/r6b/fdtd-pec-2p5/network.json", "pec10": "codex/r6b/fdtd-pec-10/network.json"}
PEAK_BANDS = {"none": (2.5e9, 4.5e9), "pgs": (1.5e9, 3.5e9), "solid": (1.5e9, 3.5e9)}


def load(paths):
    f, y = [], []
    for path in paths:
        data = json.loads(Path(path).read_text())
        f.extend(data["frequency_Hz"])
        y.extend(np.array(data["y_real"]) + 1j*np.array(data["y_imag"]))
    f, y = np.array(f), np.array(y)
    order = np.argsort(f)
    f, y = f[order], y[order]
    keep = np.concatenate(([True], np.diff(f) > 0))
    return f[keep], y[keep]


def branches(y):
    return -(y[:, 0, 1]+y[:, 1, 0])/2, y[:, 0, 0]+y[:, 1, 0]


def interpolate(f_from, values, f_to):
    return np.interp(f_to, f_from, values.real) + 1j*np.interp(f_to, f_from, values.imag)


def quantities(f, ys, y1, band_Hz):
    omega = 2*np.pi*f
    zs, zin = 1/ys, 1/(ys+y1)
    q = -zin.imag/zin.real
    band = (f >= band_Hz[0]) & (f <= band_Hz[1]) & (zin.imag < 0)
    i = np.flatnonzero(band)[np.argmax(q[band])]
    at = lambda v, x: float(np.interp(x, f, v))
    return dict(L_1GHz_H=at(-zs.imag/omega, 1e9), Q_peak=float(q[i]), peak_frequency_Hz=float(f[i]),
                Rs_1GHz_ohm=at(zs.real, 1e9), Rs_peak_ohm=float(zs.real[i]), Ls_peak_H=float(-zs.imag[i]/omega[i]),
                G1_peak_S=float(y1.real[i]), B1_peak_S=float(y1.imag[i]))


def zsplit(ys_a, ys_b, part):
    """Series admittance with the R (real) or L (imaginary) part of Zs from b."""
    za, zb = 1/ys_a, 1/ys_b
    return 1/(zb.real+1j*za.imag) if part == "R" else 1/(za.real+1j*zb.imag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--native", nargs="+", required=True)
    parser.add_argument("--label", default="native")
    parser.add_argument("--variant", default="none", choices=("none", "pgs", "solid"))
    parser.add_argument("--transfer-series", nargs="+")
    parser.add_argument("--transfer-base", nargs="+")
    args = parser.parse_args()
    # Copper references of the variant; the PEC grid-shift controls exist for the unshielded F-A only.
    REFERENCE.update(cu5=f"codex/r6/reference-{args.variant}-5/network.json",
                     cu10=f"codex/r6/reference-{args.variant}-10/network.json")
    f, y = load(args.native)
    if args.transfer_series:
        # Keep the native samples inside the transferred band (no extrapolation).
        (fa, ya), (fb, yb) = load(args.transfer_series), load(args.transfer_base)
        inside = (f >= fa[0]) & (f <= fa[-1])
        f, y = f[inside], y[inside]
        delta = interpolate(fa, 1/branches(ya)[0] - interpolate(fb, 1/branches(yb)[0], fa), f)
    ns, n1 = branches(y)
    ref = {k: load([EVIDENCE/v]) for k, v in REFERENCE.items()}
    rb = {k: tuple(interpolate(fr, b, f) for b in branches(yr)) for k, (fr, yr) in ref.items()}
    rs, r1 = rb["cu5"]
    shift_s, shift_1 = rb["pec2p5"][0]-rb["pec5"][0], rb["pec2p5"][1]-rb["pec5"][1]
    # The series shift is applied in impedance (the grid moves L), the shunt shift in admittance.
    rs_corrected = 1/(1/rs + (1/rb["pec2p5"][0] - 1/rb["pec5"][0]))
    cases = {
        "native": (ns, n1),
        "reference_cu5": (rs, r1),
        "native_series_R_from_reference": (zsplit(ns, rs, "R"), n1),
        "native_series_L_from_reference": (zsplit(ns, rs, "L"), n1),
        "native_series_from_reference": (rs, n1),
        "native_shunt_G_from_reference": (ns, r1.real+1j*n1.imag),
        "native_shunt_from_reference": (ns, r1),
        "reference_cu5_plus_pec_grid_shift_shunt": (rs, r1+shift_1),
        "reference_cu5_plus_pec_grid_shift_series_and_shunt": (rs_corrected, r1+shift_1),
        "native_shunt_from_reference_plus_pec_grid_shift": (ns, r1+shift_1),
    }
    if args.transfer_series:
        cases["native_series_plus_transfer"] = (1/(1/ns + delta), n1)
        cases["native_series_plus_transfer_shunt_from_reference"] = (1/(1/ns + delta), r1)
    table = {name: quantities(f, *b, PEAK_BANDS[args.variant]) for name, b in cases.items()}
    native, reference = table["native"], table["reference_cu5"]
    for row in table.values():
        row["Q_peak_vs_reference_cu5"] = row["Q_peak"]/reference["Q_peak"]-1
        row["L_1GHz_vs_reference_cu5"] = row["L_1GHz_H"]/reference["L_1GHz_H"]-1
    gap = native["Q_peak"]-reference["Q_peak"]
    # Two swap orders; each term is the Q-peak change (absolute) of one swap.
    order_series_first = dict(series=native["Q_peak"]-table["native_series_from_reference"]["Q_peak"],
                              shunt=table["native_series_from_reference"]["Q_peak"]-reference["Q_peak"])
    order_shunt_first = dict(shunt=native["Q_peak"]-table["native_shunt_from_reference"]["Q_peak"],
                             series=table["native_shunt_from_reference"]["Q_peak"]-reference["Q_peak"])
    report = dict(label=args.label, variant=args.variant, native_sources=[str(p) for p in args.native], reference=REFERENCE,
                  sample_frequencies_Hz=f.tolist(), peak_band_Hz=PEAK_BANDS[args.variant], cases=table,
                  q_gap_absolute=gap, q_gap_relative=gap/reference["Q_peak"],
                  q_gap_terms_series_first=order_series_first, q_gap_terms_shunt_first=order_shunt_first,
                  pec_grid_shift_at_peak=dict(
                      G1_S=float(np.interp(native["peak_frequency_Hz"], f, shift_1.real)),
                      Ls_H=float(np.interp(native["peak_frequency_Hz"], f,
                                           -(1/rb["pec2p5"][0]-1/rb["pec5"][0]).imag/(2*np.pi*f)))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1)+"\n")
    for name, row in table.items():
        print(f"{name:55s} L1 {row['L_1GHz_H']*1e9:.5f} nH ({row['L_1GHz_vs_reference_cu5']:+.2%})  "
              f"Q {row['Q_peak']:.4f} ({row['Q_peak_vs_reference_cu5']:+.2%}) at {row['peak_frequency_Hz']/1e9:.3f} GHz  "
              f"Rs {row['Rs_peak_ohm']:.4f}  G1 {row['G1_peak_S']*1e3:.4f} mS")
    print("gap", gap, "series-first", order_series_first, "shunt-first", order_shunt_first)


if __name__ == "__main__":
    main()
