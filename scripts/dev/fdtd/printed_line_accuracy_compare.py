"""FDTD-vs-independent-reference comparison for the printed CPW / coupled pair.

Inputs: reference.json and prediction.json (printed_line_accuracy.py), the
recorded FDTD two-length reductions (public bridge, 12 columns) and the V4
stripline material-separated reduction. Tolerances are the ones declared in the
accuracy README before this comparison was run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.dev.fdtd.printed_line_accuracy import BRIDGE, DOCS, FREQUENCIES, SIGMA, TOLERANCE, V4, decode, encode

NAMES = {"cpw": ("line",), "differential_pair": ("odd", "even")}
FILES = {"cpw": "cpw-two-length.json", "differential_pair": "pair-two-length.json"}


def load(path):
    return decode(json.loads(Path(path).read_text()))


def fdtd_line(kind, fdtd_dir=BRIDGE):
    data = load(Path(fdtd_dir) / FILES[kind])
    out = {n: dict(beta=[], alpha=[]) for n in NAMES[kind]}
    out["Z"] = []
    for record in data["frequency_records"]:
        t = record["independent_two_length"]
        modes = sorted(t["forward_modes"], key=lambda m: m["beta_rad_per_m"])
        for name, mode in zip(NAMES[kind], modes):
            out[name]["beta"].append(mode["beta_rad_per_m"])
            out[name]["alpha"].append(mode["alpha_np_per_m"])
        out["Z"].append(np.asarray(t["forward_characteristic_impedance_matrix_ohm"], dtype=complex))
    return out


def rel(a, b):
    return float((a - b) / b)


def gate(error, tolerance):
    return dict(relative_error=float(error), tolerance=tolerance, passed=bool(abs(error) <= tolerance))


def box_correction(ref, name, key):
    """Open-boundary correction from the level-1 box study at 1 GHz (relative)."""
    box = ref["box_level1_1GHz"]
    v = lambda label: box[label][name][key]
    open4 = 0.5 * (v("pec_x4") + v("pmc_x4"))
    open2 = 0.5 * (v("pec_x2") + v("pmc_x2"))
    delta = open4 / v("pec_x1") - 1
    unc = max(abs(v("pec_x4") - v("pmc_x4")) / 2, abs(open4 - open2)) / abs(open4)
    return delta, unc


def compare(output, fdtd_dir=BRIDGE):
    ref = load(DOCS / "reference.json")
    pred = load(DOCS / "prediction.json") if (DOCS / "prediction.json").exists() else None
    result = dict(schema="printed-line-accuracy-comparison/1", tolerances=TOLERANCE, frequencies_hz=FREQUENCIES)
    tables = []
    for kind in ("cpw", "differential_pair"):
        rk, fd = ref[kind], fdtd_line(kind, fdtd_dir)
        out = result[kind] = {}
        for name in NAMES[kind]:
            rows = []
            dbeta, ubeta_box = box_correction(rk, name, "beta")
            dalpha, ualpha_box = box_correction(rk, name, "alpha")
            for i, f in enumerate(FREQUENCIES):
                b0 = rk["propagation_t0"][name]["beta"][i]
                a0 = rk["propagation_t0"][name]["alpha_d"][i]
                b35 = rk["full_wave_t35um"][name]["beta"][i]
                a35 = rk["full_wave_t35um"][name]["alpha_d"][i]
                ac = rk["full_wave_t35um"][name]["alpha_c"][i]
                if kind == "cpw":
                    # Homogeneous TEM: geometry-independent exact values.
                    beta_d, alpha_d = rk["exact_homogeneous"]["beta"][i], rk["exact_homogeneous"]["alpha_d"][i]
                    u_beta, u_alpha = abs(b0["value"] / beta_d - 1), abs(a0["value"] / alpha_d - 1)
                    beta_d35, alpha_d35 = beta_d, alpha_d
                else:
                    beta_d = b0["value"] * (1 + dbeta)
                    alpha_d = a0["value"] * (1 + dalpha)
                    u_beta = math.hypot(b0["relative_uncertainty"], ubeta_box)
                    u_alpha = math.hypot(a0["relative_uncertainty"], ualpha_box)
                    beta_d35 = b35["value"] * (1 + dbeta)
                    alpha_d35 = a35["value"] * (1 + dalpha)
                alpha_c, u_ac = ac["value"], ac["relative_uncertainty"]
                # Thick Cu: Im Zs = Re Zs, so the internal inductance adds alpha_c to beta.
                beta_model = beta_d + alpha_c
                beta_phys = beta_d35 + alpha_c
                alpha_model = alpha_d + alpha_c
                alpha_phys = alpha_d35 + alpha_c
                bf, af = fd[name]["beta"][i], fd[name]["alpha"][i]
                analog_b = rk["fdtd_grid_analog"][name]["beta_fdtd_equivalent"][i]
                analog_a = rk["fdtd_grid_analog"][name]["alpha_d"][i]
                implied_ac = af - alpha_d
                rows.append(dict(
                    frequency_hz=f,
                    reference=dict(beta_model_t0=beta_model, beta_physical_t35um=beta_phys,
                                   beta_relative_uncertainty=u_beta, alpha_dielectric_model_t0=alpha_d,
                                   alpha_dielectric_physical_t35um=alpha_d35, alpha_dielectric_relative_uncertainty=u_alpha,
                                   alpha_conductor_t35um=alpha_c, alpha_conductor_relative_uncertainty=u_ac,
                                   alpha_total_model=alpha_model, alpha_total_physical=alpha_phys),
                    fdtd=dict(beta=bf, alpha_total=af),
                    fdtd_grid_analog=dict(beta=analog_b, alpha_dielectric=analog_a,
                                          fdtd_beta_minus_analog_plus_own_cu_relative=rel(bf, analog_b + (af - analog_a))),
                    gates=dict(beta_model=gate(rel(bf, beta_model), TOLERANCE["beta"]),
                               beta_physical=gate(rel(bf, beta_phys), TOLERANCE["beta"]),
                               alpha_total_model=gate(rel(af, alpha_model), TOLERANCE["alpha_total"]),
                               alpha_total_physical=gate(rel(af, alpha_phys), TOLERANCE["alpha_total"]),
                               alpha_conductor_implied=gate(rel(implied_ac, alpha_c), TOLERANCE["alpha_conductor"])),
                    implied_fdtd_alpha_conductor=dict(minus_reference_dielectric=implied_ac,
                                                      minus_same_grid_analog_dielectric=af - analog_a)))
            out[name] = rows
        # Impedance at every frequency
        zrows = []
        for i, f in enumerate(FREQUENCIES):
            Zf = fd["Z"][i]
            levels = rk["impedance_functional_levels"][i]
            fw_levels = rk["impedance_fw_levels"][i]
            if kind == "cpw":
                omega = 2 * math.pi * f
                factor = np.sqrt(2.25 / (2.25 - 1j * SIGMA / (omega * 8.8541878128e-12)))
                z_open = np.conj(rk["conformal_open_Z_ohm"] * factor)
                z_fun = levels[-1]["Z"]
                u_fun = abs(levels[-1]["Z"] - levels[-2]["Z"]) / abs(z_fun)
                z_fdtd = Zf[0, 0]
                row = dict(frequency_hz=f, fdtd=z_fdtd, continuum_VI_open_zero_thickness=z_open,
                           continuum_VI_box=fw_levels[-1]["line"]["VI"],
                           functional_on_continuum_field=z_fun, functional_level_change=u_fun,
                           gates=dict(definition_VI=gate(rel(abs(z_fdtd), abs(z_open)), TOLERANCE["z_definition"]),
                                      same_functional=gate(rel(abs(z_fdtd), abs(z_fun)), TOLERANCE["z_same_functional"])))
            else:
                z_fun = np.asarray(levels[-1]["mixed_matrix"])
                z_vi = np.asarray(fw_levels[-1]["mixed_VI_matrix"]).copy()
                box = rk["box_level1_1GHz"]
                for k in (0, 1):
                    entry = lambda label: np.asarray(box[label]["impedance"]["mixed_VI_matrix"])[k, k]
                    z_vi[k, k] *= 0.5 * (entry("pec_x4") + entry("pmc_x4")) / entry("pec_x1")
                row = dict(frequency_hz=f, fdtd=Zf, continuum_VI_open=z_vi,
                           continuum_PV=dict(common=fw_levels[-1]["even"]["mixed_PV"], differential=fw_levels[-1]["odd"]["mixed_PV"]),
                           continuum_PI=dict(common=fw_levels[-1]["even"]["mixed_PI"], differential=fw_levels[-1]["odd"]["mixed_PI"]),
                           functional_on_continuum_field=z_fun,
                           functional_level_change=float(np.max(np.abs(z_fun - np.asarray(levels[-2]["mixed_matrix"]))) / abs(z_fun[1, 1])),
                           gates={})
                for label, k in (("common", 0), ("differential", 1)):
                    row["gates"][f"{label}_definition_VI"] = gate(rel(abs(Zf[k, k]), abs(z_vi[k, k])), TOLERANCE["z_definition"])
                    row["gates"][f"{label}_same_functional"] = gate(rel(abs(Zf[k, k]), abs(z_fun[k, k])), TOLERANCE["z_same_functional"])
            zrows.append(row)
        out["impedance"] = zrows
        out["impedance_analog_1GHz"] = dict(continuum_on_fdtd_grid=rk["fdtd_grid_analog"]["impedance_1GHz"],
                                            functional_on_fdtd_grid=rk["fdtd_grid_analog"]["functional_1GHz"],
                                            qs_per_line_on_fdtd_grid=rk["fdtd_grid_analog"]["qs_Z_per_line"])
        out["alpha_conductor_diagnostics_1GHz"] = dict(
            yee_sheet_on_fdtd_grid=rk["fdtd_grid_analog"]["yee_sheet_alpha_c_1GHz"],
            uniform_zero_thickness_sheet=rk["uniform_zero_thickness_sheet_alpha_c_1GHz"],
            quasi_static_t35um=rk["quasi_static_t35um_1GHz"])
    # Stripline V4 material separation
    v4 = load(V4)
    st = ref["stripline"]
    mats = v4["materials"]
    ex0, ex1 = st["exact_homogeneous"]["0.0"], st["exact_homogeneous"]["0.01"]
    ac1 = st["alpha_c_1GHz_t35um"]["value"]
    srows = []
    for i, f in enumerate(FREQUENCIES):
        alpha_c_ref = ac1 * math.sqrt(f / 1e9)
        cu = v4["separation"]["raw_Cu_only_minus_PEC0"][i]
        diel = mats["pec_sigma001"]["signed_alpha_np_per_m"][i] - mats["pec_sigma0"]["signed_alpha_np_per_m"][i]
        srows.append(dict(frequency_hz=f, fdtd_alpha_conductor=cu, reference_alpha_conductor=alpha_c_ref,
                          fdtd_alpha_dielectric=diel, exact_alpha_dielectric=ex1[i][0],
                          fdtd_beta_pec=mats["pec_sigma0"]["beta_rad_per_m"][i], exact_beta=ex0[i][1],
                          fdtd_cu_minus_pec_beta=mats["copper_sigma0"]["beta_rad_per_m"][i] - mats["pec_sigma0"]["beta_rad_per_m"][i],
                          gates=dict(alpha_conductor=gate(rel(cu, alpha_c_ref), TOLERANCE["alpha_conductor"]),
                                     alpha_dielectric=gate(rel(diel, ex1[i][0]), TOLERANCE["alpha_dielectric"]),
                                     beta=gate(rel(mats["pec_sigma0"]["beta_rad_per_m"][i], ex0[i][1]), TOLERANCE["beta"]))))
    z_v4 = abs(np.asarray(mats["pec_sigma0"]["two_length_modes"][2]["forward_characteristic_impedance_matrix_ohm"])[0, 0])
    result["stripline_v4"] = dict(rows=srows, fdtd_Z_pec_1GHz=z_v4, cohn_Z=st["cohn_Z_t0_ohm"],
                                  v4_grid_analog=st["v4_grid_analog"], Z_t35um=st["Z_t35um"],
                                  Z_gate=gate(rel(z_v4, st["cohn_Z_t0_ohm"]), TOLERANCE["z_definition"]))
    if pred is not None:
        result["grid_predictions"] = pred["grids"]
    # Reference self-qualification
    qual = {}
    for kind in ("cpw", "differential_pair"):
        for name in NAMES[kind]:
            rows = result[kind][name]
            qual[f"{kind}/{name}"] = dict(
                beta_max_relative_uncertainty=max(r["reference"]["beta_relative_uncertainty"] for r in rows),
                alpha_dielectric_max_relative_uncertainty=max(r["reference"]["alpha_dielectric_relative_uncertainty"] for r in rows),
                alpha_conductor_max_relative_uncertainty=max(r["reference"]["alpha_conductor_relative_uncertainty"] for r in rows))
    qual["stripline/alpha_c"] = st["alpha_c_1GHz_t35um"]["relative_uncertainty"]
    result["reference_uncertainty"] = qual
    result["reference_targets_met"] = all(
        v["beta_max_relative_uncertainty"] < 3e-3 and v["alpha_dielectric_max_relative_uncertainty"] < 2e-2
        and v["alpha_conductor_max_relative_uncertainty"] < 2e-2 for k, v in qual.items() if isinstance(v, dict))
    Path(output).write_text(json.dumps(encode(result), indent=1) + "\n")
    summarize(result)
    return result


def summarize(result):
    for kind in ("cpw", "differential_pair"):
        for name in NAMES[kind]:
            print(f"\n{kind} {name}")
            print("f GHz | beta FDTD | beta ref(model) | err | beta analog | alpha FDTD | alpha ref | err | alpha_c ref | alpha_c implied | err")
            for r in result[kind][name]:
                g, ref = r["gates"], r["reference"]
                print(f"{r['frequency_hz']/1e9:.1f} | {r['fdtd']['beta']:.4f} | {ref['beta_model_t0']:.4f} | {100*g['beta_model']['relative_error']:+.3f}% | "
                      f"{r['fdtd_grid_analog']['beta']:.4f} | {r['fdtd']['alpha_total']:.5f} | {ref['alpha_total_model']:.5f} | {100*g['alpha_total_model']['relative_error']:+.2f}% | "
                      f"{ref['alpha_conductor_t35um']:.5f} | {r['implied_fdtd_alpha_conductor']['minus_reference_dielectric']:.5f} | {100*g['alpha_conductor_implied']['relative_error']:+.1f}%")
        for z in result[kind]["impedance"]:
            print("Z", z["frequency_hz"] / 1e9, {k: (v["relative_error"], v["passed"]) for k, v in z["gates"].items()})
    for r in result["stripline_v4"]["rows"]:
        print("stripline", r["frequency_hz"] / 1e9, {k: round(100 * v["relative_error"], 3) for k, v in r["gates"].items()})
    print("reference uncertainty", json.dumps(result["reference_uncertainty"], indent=1), "targets met", result["reference_targets_met"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fdtd-dir", type=Path, default=BRIDGE,
                        help="Directory containing cpw-two-length.json and pair-two-length.json; "
                             "Yee-analog diagnostics remain those of the original T1 grid.")
    args = parser.parse_args()
    compare(args.output, fdtd_dir=args.fdtd_dir)
