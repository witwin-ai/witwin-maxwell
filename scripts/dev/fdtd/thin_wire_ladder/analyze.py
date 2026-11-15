"""CPU analysis of ladder JSONs, writing one CSV summary table.

Each MoM entry carries its oracle's qualification flag: straight dipoles are qualified
against published tables (tests/support/test_thin_wire_mom.py); loops are not, so loop
comparisons stay diagnostic.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from scripts.dev.fdtd.thin_wire_ladder.common import F0, CHECK_FREQUENCIES, Run, from_pairs, resonance, wire_geometry
from tests.support.thin_wire_mom import dipole, loop_path


def observed_order(levels, values):
    """Variable-projection fit of X=X0-k*r**p on actual, unequal refinements.

    Positive p indicates growth on refinement under the spec's sign convention;
    negative p indicates decay. A flat or nonmonotone series has no reported p.
    The bounded minimum and normalized residual are reported, not just p.
    """
    r, x = np.asarray(levels), np.asarray(values, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or np.ptp(x) < 1e-10:
        return None
    difference = np.diff(x)
    if not (np.all(difference > 0) or np.all(difference < 0)):
        return None
    def projected(p):
        # expm1/p is continuous through p=0, giving a logarithmic limit.
        basis = np.log(r) if abs(p) < 1e-7 else np.expm1(p*np.log(r))/p
        matrix = np.column_stack((np.ones_like(r), basis))
        coefficients = np.linalg.lstsq(matrix, x, rcond=None)[0]
        residual = np.linalg.norm(matrix@coefficients-x)/np.linalg.norm(x-x.mean())
        return residual, coefficients
    # Split intervals ensure both positive/negative p and a boundary minimum
    # are examined; a pure bounded search over both can miss a shallow basin.
    choices = [minimize_scalar(lambda p: projected(p)[0], bounds=bound, method="bounded")
               for bound in ((-8., -1e-6), (1e-6, 8.))]
    result = min(choices, key=lambda opt: opt.fun)
    p = float(result.x)
    residual, coefficients = projected(p)
    return dict(p=p, residual_fraction=float(residual), at_bound=abs(p)>7.99,
                logarithmic_limit=abs(p)<1e-4, sign_convention="X0-k*r^p")


def group_diagnostics(data):
    ordered = sorted(data, key=lambda d: d["run"]["level"])
    levels = [d["run"]["level"] for d in ordered]
    fixed = np.array([from_pairs(d["z_f0_ohm"]) for d in ordered])
    checks = np.array([from_pairs(d["z_check_engineering_ohm"]) for d in ordered])
    triples = []
    for i in range(len(ordered)-2):
        denominator = abs(fixed[i+1]-fixed[i+2])
        triples.append(dict(levels=levels[i:i+3], ratio=float(abs(fixed[i]-fixed[i+1])/denominator)
                            if denominator > 0 else None))
    capacitors = []
    for i in range(len(ordered)-1):
        ci, cj = (ordered[j]["wire_static"]["C_eff_F"] for j in (i, i+1))
        dx = (checks[i+1]-checks[i]).imag
        predicted = -(1/cj-1/ci)/(2*np.pi*CHECK_FREQUENCIES)
        residual = dx-predicted
        fractions = [float(a/b) if abs(b)>1e-12 else None for a, b in zip(residual, dx)]
        capacitors.append(dict(levels=levels[i:i+2], frequencies_Hz=CHECK_FREQUENCIES.tolist(),
            dX_ohm=dx.tolist(), minus_d_inverse_wC_ohm=predicted.tolist(),
            residual_ohm=residual.tolist(), residual_fraction_signed=fractions,
            residual_fraction_abs=[abs(v) if v is not None else None for v in fractions]))
    # Track one crossing continuously from coarse to fine. A missing crossing
    # stays absent and never becomes an endpoint minimum of |X|.
    tracked = {}
    target = F0
    for d in ordered:
        f, z = resonance(d["frequencies_Hz"], from_pairs(d["z_engineering_ohm"]), target)
        tracked[d["run"]["name"]] = (f, z)
        if f is not None:
            target = f
    return dict(richardson=triples,
        order_X_f0=observed_order(levels, fixed.imag),
        order_R_f0=observed_order(levels, fixed.real),
        order_X_check=[observed_order(levels, checks[:, i].imag) for i in range(3)],
        capacitor=capacitors, tracked=tracked)


@lru_cache(maxsize=16384)
def reference_result(antenna, length, radius, frequency, gap, segments, contour=None, feed_segment=None):
    if antenna == "L1":
        return loop_path(points=contour, feed_segment=feed_segment, radius=radius,
                         frequency=frequency, finite_gap=gap > 0, segments=segments)
    return dipole(length=length, radius=radius, frequency=frequency, gap=gap, segments=segments)


def reference(*arguments):
    return reference_result(*arguments).impedance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/thin-wire-ladder-summary.csv"))
    parser.add_argument("--mom-segments", type=int, default=320)
    args = parser.parse_args()
    data = [json.loads(p.read_text()) for p in sorted(args.input.glob("*.json"))]
    data = [d for d in data if "z_engineering_ohm" in d]
    if not data:
        parser.error("No completed ladder runs in input directory")
    groups = {}
    # Restore the intentional r1 G1/G-phys aliases for each crossed family.
    expanded = list(data)
    for d in data:
        if d["run"]["level"] == 1 and d["run"]["gap_rule"] == "G1" and d["run"]["control"] == "primary":
            alias = dict(d, run=dict(d["run"], gap_rule="G-phys"))
            expanded.append(alias)
    for d in expanded:
        r = d["run"]
        key = (r["antenna"], r["gap_rule"], r["binding"], r["control"], d["horizon"])
        groups.setdefault(key, []).append(d)
    diagnostics = {key: group_diagnostics(values) for key, values in groups.items()}
    rows = []
    for d in data:
        r = d["run"]
        key = (r["antenna"], r["gap_rule"], r["binding"], r["control"], d["horizon"])
        group = diagnostics[key]
        z = complex(from_pairs(d["z_f0_ohm"]))
        arguments = (r["antenna"], r["length_m"], r["radius_m"])
        contour, feed_segment = None, None
        if r["antenna"] == "L1":
            run = Run(r["antenna"], r["level"], r["gap_rule"], r["binding"], r["control"])
            geometry, negative, positive = wire_geometry(run)
            points = list(geometry["wire"])
            if points[-1] != points[0]:
                points.append(points[0])
            contour = tuple(points)
            feed_segment = negative[1]
        def mom_reference(frequency, gap, segments):
            return reference(*arguments, frequency, gap, segments, contour, feed_segment)
        frequencies = d["frequencies_Hz"]
        finite = np.array([mom_reference(f, r["gap_m"], args.mom_segments) for f in frequencies])
        delta = np.array([mom_reference(f, 0., args.mom_segments) for f in frequencies])
        mom = complex(np.interp(F0, frequencies, finite))
        doubled = mom_reference(F0, r["gap_m"], 2*args.mom_segments)
        refinement = abs(doubled-mom)/abs(doubled)
        residual_x = (z-mom).imag
        # A positive residual X cannot be interpreted as a positive series C.
        c_gap = -1/(2*np.pi*F0*residual_x) if residual_x < 0 else None
        f_res, z_res = group["tracked"][r["name"]]
        mom_res = mom_reference(f_res, r["gap_m"], args.mom_segments) if f_res else None
        full_z = from_pairs(d["z_engineering_ohm"])
        reference_metadata = reference_result(*arguments, F0, r["gap_m"], args.mom_segments, contour, feed_segment)
        doubled_metadata = reference_result(*arguments, F0, r["gap_m"], 2*args.mom_segments, contour, feed_segment)
        row = dict(run=r["name"], horizon=d["horizon"], antenna=r["antenna"], gap_rule=r["gap_rule"],
            binding=r["binding"], control=r["control"], level=r["level"], delta_m=r["delta_m"],
            gap_m=r["gap_m"], gap_cells=r["gap_cells"], radius_m=r["radius_m"],
            total_cells=r["total_cells"], R_f0=z.real, X_f0=z.imag,
            tracked_f_res_Hz=f_res, R_res=z_res.real if z_res is not None else None,
            X_res=z_res.imag if z_res is not None else None,
            mom_qualified=reference_metadata.qualified, mom_requested_segments=args.mom_segments,
            mom_actual_segments=reference_metadata.segments, mom_actual_doubled_segments=doubled_metadata.segments,
            mom_formulation=reference_metadata.formulation, mom_feed_model=reference_metadata.feed_model,
            mom_finite_R_f0=mom.real, mom_finite_X_f0=mom.imag,
            mom_delta_R_f0=delta[16].real, mom_delta_X_f0=delta[16].imag,
            mom_finite_R_res=mom_res.real if mom_res is not None else None,
            mom_finite_X_res=mom_res.imag if mom_res is not None else None,
            mom_double_segments_relative_change=refinement,
            mom_segment_check_pass=(refinement <= .005 and doubled_metadata.segments >= 1.9*reference_metadata.segments),
            diagnostic_relative_error_f0=abs(z-mom)/abs(mom),
            implied_series_C_gap_F=c_gap, residual_X_f0=residual_x,
            implied_series_C_status=("diagnostic" if reference_metadata.qualified else "diagnostic; unqualified reference") if c_gap else "no positive series-capacitor interpretation",
            C_eff_F=d["wire_static"]["C_eff_F"],
            C_node_negative_F=d["wire_static"]["feed_C_node_F"][0],
            C_node_positive_F=d["wire_static"]["feed_C_node_F"][1],
            median_L_prime=d["wire_static"]["median"]["l_prime_H_m"],
            median_C_prime=d["wire_static"]["median"]["c_prime_F_m"],
            lc_max_relative_error=d["wire_static"]["max_abs_lc_relative_error"],
            max_abs_s11=d["passivity"]["max_abs_s11"], passivity_pass=d["passivity"]["within_one"],
            energy_final_over_peak=d["energy_qc"]["final_over_peak"],
            energy_qc_pass=(d["energy_qc"]["final_over_peak"] <= 1e-7 if d["energy_qc"]["final_over_peak"] is not None else None),
            wall_seconds=d["wall_seconds"], torch_peak_allocated_bytes=d["torch_peak_allocated_bytes"],
            order_X_f0=json.dumps(group["order_X_f0"]), order_R_f0=json.dumps(group["order_R_f0"]),
            order_X_check=json.dumps(group["order_X_check"]),
            richardson=json.dumps(group["richardson"]), capacitor_test=json.dumps(group["capacitor"]),
            mom_spectra=json.dumps(dict(frequencies_Hz=frequencies, finite_R=finite.real.tolist(),
                finite_X=finite.imag.tolist(), delta_R=delta.real.tolist(), delta_X=delta.imag.tolist(),
                fdtd_minus_finite_R=(full_z-finite).real.tolist(), fdtd_minus_finite_X=(full_z-finite).imag.tolist())),
            limitations="; ".join(d["limitations"]))
        # Independent fixed-horizon comparison with the same geometry/level.
        peer = next((p for p in data if p["run"]["name"] == r["name"] and p["horizon"] != d["horizon"]), None)
        row["horizon_relative_Z_difference_f0"] = (abs(z-from_pairs(peer["z_f0_ohm"]))/abs(z) if peer and abs(z)>0 else None)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
