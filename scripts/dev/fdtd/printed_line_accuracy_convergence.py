"""CPU analysis of the completed printed-line mesh, padding and Cu-only queue.

Uses the original comparison's reference definitions and gates. No solver runs.
The section output is a table fragment for section 6 of the accuracy assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.dev.fdtd.printed_line_accuracy import BRIDGE, DOCS, FREQUENCIES, TOLERANCE, encode
from scripts.dev.fdtd.printed_line_accuracy_compare import FILES, NAMES, fdtd_line, gate, load, rel

MESH_LABELS = ("T0_dx2.5", "T1_dx1.25_current", "T2_dx0.625")
CU_LABELS = ("T0_dx2.5_Cu_only", "T1_dx1.25_Cu_only", "T2_dx0.625_Cu_only")
MODE_LABELS = {"line": "CPW", "odd": "pair odd (Z_dm)", "even": "pair even (Z_cm)"}
GATES = {"beta": "beta", "alpha_total": "alpha_total", "alpha_cu": "alpha_conductor",
         "z_vi": "z_definition", "z_functional": "z_same_functional"}


def refinement(values, reference):
    """Three-level scalar estimate; oscillation/divergence cannot qualify a GCI."""
    coarse, medium, fine = values
    d1, d2 = medium - coarse, fine - medium
    q = d1 / d2
    p = math.log2(abs(q))
    status = "monotonic" if q > 1 else "oscillatory" if q < 0 else "divergent"
    # For q > 1 this is Richardson. Otherwise it is only a formal geometric fit.
    formal = fine + d2 / (q - 1)
    valid = status == "monotonic"
    gci = 1.25 * abs(formal - fine) / abs(fine) if valid else None
    return dict(values=values, reference=reference, relative_errors=[rel(v, reference) for v in values],
                signed_difference_ratio=q, log2_absolute_difference_ratio=p, behavior=status,
                observed_order=p if valid else None,
                richardson_value=formal if valid else None, formal_limit=formal,
                extrapolated_relative_error=rel(formal, reference),
                gci_fine_relative=gci, gci_fine_absolute=gci * abs(fine) if valid else None)


def convergence(artifacts_dir, output, section_output):
    directories = {label: artifacts_dir / label for label in (*MESH_LABELS, *CU_LABELS, "P2_padding0.16")}
    directories[MESH_LABELS[1]] = BRIDGE
    baseline = load(DOCS / "comparison.json")
    inputs = [DOCS / name for name in ("comparison.json", "reference.json", "prediction.json")]
    inputs += [artifacts_dir / name for name in ("ENV_NOTES.txt", "SOURCE_HEAD")]
    provenance = {}
    for label, directory in directories.items():
        provenance[label] = {}
        for kind, filename in FILES.items():
            inputs.append(directory / filename)
            provenance[label][kind] = []
            for length in (0, 1):
                basis = directory / filename.replace("two-length", f"L{length}-basis")
                inputs.append(basis)
                record = load(basis)
                provenance[label][kind].append({k: record[k] for k in
                                               ("case", "executed_identity", "grid_sha256", "clock", "executed_operators")})
    result = dict(
        schema="printed-line-accuracy-convergence/1", date="2026-10-31", cpu_only=True,
        queue_source_head=(artifacts_dir / "SOURCE_HEAD").read_text().strip(),
        queue_branch_tree_range="0af1d3cd..9a56ae05", frozen_core_version="0.4.3",
        environment_notes=(artifacts_dir / "ENV_NOTES.txt").read_text(),
        tolerances=TOLERANCE, frequencies_hz=FREQUENCIES,
        mesh_labels=MESH_LABELS, transverse_dx_m=[.0025, .00125, .000625],
        fixed_longitudinal_dl_m=.0025, cu_only_labels=CU_LABELS,
        directories={k: str(v) for k, v in directories.items()}, provenance=provenance,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        method=dict(
            ratio=2, safety_factor=1.25,
            difference_ratio="q=(T1-T0)/(T2-T1)", order="p=log2(q), only for q>1",
            extrapolation="T2+(T2-T1)/(q-1); formal geometric fit only when q<=1",
            gci_fine="1.25*abs(extrapolation-T2)/abs(T2), only for q>1",
            gci_scope="Provisional transverse-grid estimate; three grids do not independently establish the asymptotic range.",
            provenance_scope="Sigma-on triples and P2-T1 mix the original T1 execution identity with the frozen-core queue; not an isolated mesh/padding experiment.",
            impedance="Analyze diagonal |Z|, as in the declared comparison gates; preserve complex Z separately. Odd=dm index 1, even=cm index 0.",
            reference="Same model reference and continuum-functional targets as comparison.json; no gates retuned.",
            cu_only="Sigma=0 two-length alpha, without a matched PEC0 background subtraction; existing t35um Cu reference is retained.",
            extrapolated_gates="Numerical tolerance checks of the formal limits. Qualified=false means no accepted extrapolated verdict."),
        modes={}, verdicts=[])
    for kind, names in NAMES.items():
        reductions = {label: fdtd_line(kind, directory) for label, directory in directories.items()}
        for name in names:
            mode = result["modes"][f"{kind}/{name}"] = []
            k = 1 if name == "odd" else 0
            for i, frequency in enumerate(FREQUENCIES):
                ref = baseline[kind][name][i]["reference"]
                zrow = baseline[kind]["impedance"][i]
                z_vi = (zrow["continuum_VI_open_zero_thickness"] if kind == "cpw"
                        else zrow["continuum_VI_open"][k][k])
                z_fun = (zrow["functional_on_continuum_field"] if kind == "cpw"
                         else zrow["functional_on_continuum_field"][k][k])
                zs = [complex(reductions[label]["Z"][i][k, k]) for label in MESH_LABELS]
                metrics = dict(
                    beta=refinement([reductions[label][name]["beta"][i] for label in MESH_LABELS], ref["beta_model_t0"]),
                    alpha_total=refinement([reductions[label][name]["alpha"][i] for label in MESH_LABELS], ref["alpha_total_model"]),
                    Z=refinement([abs(z) for z in zs], abs(z_vi)),
                    alpha_cu=refinement([reductions[label][name]["alpha"][i] for label in CU_LABELS], ref["alpha_conductor_t35um"]))
                metrics["Z"].update(complex_values_ohm=zs, continuum_VI_complex_ohm=z_vi,
                                     functional_complex_ohm=z_fun, functional_magnitude_ohm=abs(z_fun),
                                     functional_relative_errors=[rel(abs(z), abs(z_fun)) for z in zs],
                                     extrapolated_functional_relative_error=rel(metrics["Z"]["formal_limit"], abs(z_fun)))
                padding = {}
                for quantity, source_key in (("beta", "beta"), ("alpha_total", "alpha"), ("Z", None)):
                    value = (abs(reductions["P2_padding0.16"]["Z"][i][k, k]) if quantity == "Z"
                             else reductions["P2_padding0.16"][name][source_key][i])
                    medium = metrics[quantity]["values"][1]
                    padding[quantity] = dict(T1=medium, P2=value, difference=value - medium,
                                             relative_to_T1=rel(value, medium),
                                             relative_to_reference=(value - medium) / metrics[quantity]["reference"])
                padding["Z"]["complex_difference_ohm"] = complex(reductions["P2_padding0.16"]["Z"][i][k, k]) - zs[1]
                gates = {}
                for label, tolerance_key in GATES.items():
                    metric = metrics["Z" if label.startswith("z_") else label]
                    target = abs(z_fun) if label == "z_functional" else metric["reference"]
                    gates[label] = dict(
                        finest=gate(rel(metric["values"][-1], target), TOLERANCE[tolerance_key]),
                        extrapolated=gate(rel(metric["formal_limit"], target), TOLERANCE[tolerance_key]),
                        extrapolation_qualified=metric["behavior"] == "monotonic")
                mode.append(dict(frequency_hz=frequency, metrics=metrics, padding=padding, gates=gates))
            for label in GATES:
                gs = [row["gates"][label] for row in mode]
                fine_pass = sum(g["finest"]["passed"] for g in gs)
                extrap_pass = sum(g["extrapolated"]["passed"] for g in gs)
                qualified = sum(g["extrapolation_qualified"] for g in gs)
                qualified_failure = any(g["extrapolation_qualified"] and not g["extrapolated"]["passed"] for g in gs)
                result["verdicts"].append(dict(
                    mode=MODE_LABELS[name], gate=label, tolerance=TOLERANCE[GATES[label]],
                    finest_pass_count=fine_pass, finest_verdict="PASS" if fine_pass == 5 else "FAIL",
                    finest_error_range=[min(g["finest"]["relative_error"] for g in gs), max(g["finest"]["relative_error"] for g in gs)],
                    extrapolated_numerical_pass_count=extrap_pass,
                    extrapolated_numerical_verdict="PASS" if extrap_pass == 5 else "FAIL",
                    extrapolated_qualified_count=qualified,
                    extrapolated_verdict="FAIL" if qualified_failure else "UNQUALIFIED" if qualified < 5 else "PASS",
                    extrapolated_error_range=[min(g["extrapolated"]["relative_error"] for g in gs), max(g["extrapolated"]["relative_error"] for g in gs)]))
    Path(output).write_text(json.dumps(encode(result), indent=1, allow_nan=False) + "\n")
    Path(section_output).write_text(tables(result))
    return result


def tables(result):
    lines = []
    for key, title, scale in (("beta", "beta (rad/m)", 1), ("alpha_total", "total alpha (Np/m)", 1),
                               ("Z", "|Z| (Ohm; pair odd Z_dm, even Z_cm)", 1),
                               ("alpha_cu", "Cu-only alpha (mNp/m)", 1000)):
        lines += [f"#### {title}", "",
                  "| Mode | f GHz | Reference | T0 (err %) | T1 (err %) | T2 (err %) | p | Extrap. (err %) | GCI fine % |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for mode, records in result["modes"].items():
            name = MODE_LABELS[mode.split("/")[1]].split(" (")[0]
            for row in records:
                m = row["metrics"][key]
                digits = 4 if key in ("beta", "Z", "alpha_cu") else 6
                value_error = lambda v, e: f"{v * scale:.{digits}f} ({100 * e:+.3f})"
                samples = [value_error(v, e) for v, e in zip(m["values"], m["relative_errors"])]
                p = f"{m['observed_order']:.3f}" if m["observed_order"] is not None else "O" if m["behavior"] == "oscillatory" else "D"
                extra = value_error(m["formal_limit"], m["extrapolated_relative_error"])
                gci = f"{100 * m['gci_fine_relative']:.4f}" if m["gci_fine_relative"] is not None else "—"
                if m["observed_order"] is None:
                    extra += "†"
                lines.append(f"| {name} | {row['frequency_hz']/1e9:.1f} | {m['reference']*scale:.{digits}f} | "
                             + " | ".join(samples) + f" | {p} | {extra} | {gci} |")
        lines.append("")
        if key == "Z":
            lines += ["Same-functional Z gate (2%): signed magnitude errors against the original continuum-field functional.", "",
                      "| Mode | f GHz | Reference magnitude Ohm | T0 err % | T1 err % | T2 err % | Formal extrap. err % |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for mode, records in result["modes"].items():
                name = MODE_LABELS[mode.split("/")[1]].split(" (")[0]
                for row in records:
                    m = row["metrics"]["Z"]
                    errors = " | ".join(f"{100*e:+.3f}" for e in m["functional_relative_errors"])
                    lines.append(f"| {name} | {row['frequency_hz']/1e9:.1f} | {m['functional_magnitude_ohm']:.4f} | "
                                 + errors + f" | {100*m['extrapolated_functional_relative_error']:+.3f}† |")
            lines.append("")
    lines += ["#### Padding sensitivity: P2 minus original T1", "",
              "| Mode | f GHz | Delta beta rad/m (% ref) | Delta alpha mNp/m (% ref) | Delta magnitude Z Ohm (% ref) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for mode, records in result["modes"].items():
        name = MODE_LABELS[mode.split("/")[1]].split(" (")[0]
        for row in records:
            cells = [f"{row['padding'][q]['difference'] * scale:+.6f} ({100*row['padding'][q]['relative_to_reference']:+.4f})"
                     for q, scale in (("beta", 1), ("alpha_total", 1000), ("Z", 1))]
            lines.append(f"| {name} | {row['frequency_hz']/1e9:.1f} | " + " | ".join(cells) + " |")
    lines += ["", "#### Gate verdicts over 0.8–1.2 GHz", "",
              "| Mode | Gate | T2 error range %; verdict (passed/5) | Extrap. error range %; numerical verdict (passed/5) | Monotonic triples / 5; extrap. verdict |",
              "| --- | --- | --- | --- | ---: |"]
    for v in result["verdicts"]:
        a, b = [100 * e for e in v["finest_error_range"]]
        c, d = [100 * e for e in v["extrapolated_error_range"]]
        lines.append(f"| {v['mode']} | {v['gate']} ({100*v['tolerance']:g}%) | {a:+.3f} to {b:+.3f}; "
                     f"{v['finest_verdict']} ({v['finest_pass_count']}/5) | {c:+.3f} to {d:+.3f}; "
                     f"{v['extrapolated_numerical_verdict']} ({v['extrapolated_numerical_pass_count']}/5) | "
                     f"{v['extrapolated_qualified_count']}/5; {v['extrapolated_verdict']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section-output", type=Path, required=True)
    args = parser.parse_args()
    convergence(args.artifacts_dir, args.output, args.section_output)
