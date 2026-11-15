"""Numerical lock of the layered MoM evidence (R6, R6b, R4b): re-acquire, judge by tier, renew.

Rule D-P1 (user decision 2026-11-09): the committed evidence values in
tests/mom/layered/data are locked by value with the tiers of the numerical-
equivalence standard. Source hashes are not gates. Fast tests check the committed
values against fixtures and physics targets; this script re-runs the evidence
producers and judges every committed numeric field against its committed value.

Modes:

* ``acquire DIR`` (GPU 0 through gpu0.py, with ``--cap-gib``): re-run every producer
  into DIR and record the commit in DIR/acquisition.json. The solver tree
  (witwin/) must be clean.
* ``judge DIR`` (CPU): compare DIR with the committed evidence and write
  DIR/tier_table.json, DIR/tier_table.md and DIR/fields.json. Exit code 1 on any
  locked field outside T0/T1 that is not an accepted T2.
* ``renew DIR [--accept-t2 FILE]`` (CPU): judge, then write every moved locked value
  (T1 and accepted T2) into the committed evidence and append a renewal record to
  each file's ``numerical_lock``. Refuses on an unaccepted T2, a discrete mover or
  an accept entry that does not name an actual T2.

Tiers of a locked field (committed value vs re-acquisition):

* T0: every float64 bit equal (discrete fields: equal);
* T1: max|new-ref| <= 1e-10*max|ref| per field, the fp64 run-level bound of the
  standard (network S/Y/Z and the quantities derived from them);
* accepted T2: outside the bound and named in the accept file with its approval;
* T2: anything else, including every discrete change.

Producer artifacts outside the repository (R6 MPIE sweeps, R4b closed cases) are
judged with the same tiers in a report-only scope: they are acquisition records
whose pi-fit diagnostics are ill-conditioned nonlinear fits, not committed values.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

DATA = Path("tests/mom/layered/data")
FILES = {"r6": "fixture_r6_native_evidence.json", "r6b": "fixture_r6b_shunt_evidence.json",
         "r4b": "fixture_r4b_closed_comparison.json"}
# A4 certification fixtures (R7-2): each file's "mom" records are re-produced by scripts/dev/rfic/a4_fixtures_r7.py
# with the producer arguments stored in the record and locked field by field.
A4 = {"fd": "fixture_fd_mim_evidence.json", "fe": "fixture_fe_lines_evidence.json",
      "ff": "fixture_ff_octagon_evidence.json"}
FILES.update(A4)
LEGACY = ("r6", "r6b", "r4b")
# Committed evidence names producer artifacts under the scrubbed "<evidence-root>/" prefix.
EVIDENCE = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence"))
PREFIX = "<evidence-root>/"
R4B = PREFIX + "rfic-mom-20261106/codex/r4b"
OBSERVATIONS = {"seconds", "peak_bytes", "peak_allocated_bytes", "source_sha256", "prepare_seconds",
                "solve_seconds", "artifact_sha256", "sources"}
DIAGNOSTICS = {"backward_error", "frequency_chunk_size", "operator_backend"}
BOUND = 1e-10
# Acquisition options echoed from the frozen R4b record (inputs, not results).
ECHO = ("case", "model", "substrate", "pec", "maximum_um", "edge_um", "thickness_um", "thin_all", "exact_repeats")
# Normalized rounding residuals: compared against their physics gate, not a relative bound.
RESIDUALS = {"reciprocity": 1e-10}
# Report-scope nonlinear least-squares pi fits of sampled windows.
FITS = ("pi_fit",)
RULE = ("D-P1 (user decision 2026-11-09): committed numeric evidence is locked by value with the tiers of "
        "numerical_equivalence_standard.md (T0 bitwise, T1 within the fp64 run-level bound, T2 only with a "
        "per-field user approval); source hashes are not gates. Harness: scripts/dev/rfic/evidence_lock.py.")


def resolve(name):
    """Resolve a committed "<evidence-root>/..." path against WITWIN_EVIDENCE_ROOT."""
    return EVIDENCE / name[len(PREFIX):] if name.startswith(PREFIX) else Path(name)


def scrub(path):
    path = str(Path(path).resolve())
    root = str(EVIDENCE.resolve()) + "/"
    return PREFIX + path[len(root):] if path.startswith(root) else path


def same_bits(a, b):
    return struct.pack(">d", a) == struct.pack(">d", b)


def floats(value):
    if isinstance(value, float):
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, (list, float, int)) and not isinstance(v, bool)
                                                 for v in value):
        out = []
        for item in value:
            if isinstance(item, int):
                return None
            sub = floats(item)
            if sub is None:
                return None
            out.extend(sub)
        return out
    return None


def compare(old, new, where, rows, scope):
    """Append one row per leaf of ``old``; ``where`` is (file, key path)."""
    name, path = where
    field = name + ":" + "/".join(map(str, path))
    if isinstance(old, dict):
        for key, value in old.items():
            if key in OBSERVATIONS or key == "numerical_lock":
                continue
            if key in DIAGNOSTICS:
                rows.append(dict(scope=scope, field=f"{field}/{key}", tier="diagnostic", old=value, new=new.get(key)))
                continue
            if key not in new:
                rows.append(dict(scope=scope, field=f"{field}/{key}", file=name, path=path + [key], tier="T2",
                                 kind="missing"))
                continue
            compare(value, new[key], (name, path + [key]), rows, scope)
        return
    leaf = str(path[-1])
    if scope == "report" and leaf in RESIDUALS and isinstance(old, float):
        rows.append(dict(scope=scope, field=field, tier="residual", old=old, new=new, gate=RESIDUALS[leaf],
                         passes_gate=bool(new < RESIDUALS[leaf])))
        return
    values = floats(old)
    if values is not None:
        other = floats(new)
        if other is None or len(other) != len(values):
            rows.append(dict(scope=scope, field=field, file=name, path=path, tier="T2", kind="shape"))
            return
        a, b = np.asarray(values), np.asarray(other)
        bits = all(same_bits(x, y) for x, y in zip(values, other))
        scale = float(np.abs(a).max())
        delta = float(np.abs(a - b).max())
        ratio = 0. if delta == 0 else (np.inf if scale == 0 else delta / (BOUND * scale))
        tier = "T0" if bits else ("T1" if ratio <= 1 else "T2")
        if scope == "report" and tier == "T2" and any(f in field for f in FITS):
            tier = "fit"
        rows.append(dict(scope=scope, field=field, file=name, path=path, tier=tier,
                         kind="tensor" if isinstance(old, list) else "scalar", size=int(a.size),
                         max_abs_delta=delta, max_abs_ref=scale,
                         max_rel_diff=0. if delta == 0 else (np.inf if scale == 0 else delta / scale),
                         delta_over_bound=float(ratio), new=new))
        return
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            rows.append(dict(scope=scope, field=field, file=name, path=path, tier="T2", kind="length"))
            return
        for i, (x, y) in enumerate(zip(old, new)):
            compare(x, y, (name, path + [i]), rows, scope)
        return
    tier = "T0" if old == new else ("fit" if scope == "report" and any(f in field for f in FITS) else "T2")
    rows.append(dict(scope=scope, field=field, file=name, path=path, tier=tier, kind="discrete", old=old, new=new))


def committed():
    return {key: json.loads((DATA / name).read_text()) for key, name in FILES.items()}


def last_acquisition(evidence):
    renewals = evidence.get("numerical_lock", {}).get("renewals", [])
    return resolve(renewals[-1]["acquisition"]) if renewals else None


def r6_producers(native):
    """Reference producer artifacts of the R6 native evidence (latest renewal, else the original sweep)."""
    original = sorted({resolve(p) for level in ("coarse", "finest") for r in native[level].values()
                       for p in r["sources"]})
    root = original[0].parent
    original += sorted(set(root.glob("none-shield_fine-*.json")) - set(original))
    previous = last_acquisition(native)
    return [previous / "mpie" / p.name if previous else p for p in original]


def r4b_producer(comparison, name):
    previous = last_acquisition(comparison)
    return previous / name if previous else resolve(R4B) / name


def pi_conductance(collected):
    from tests.support.rfic_r6_references import fit_pi, pi_admittance, pi_terms, unpack_network
    groups = {v: collected[v + "/shield_fine"] for v in ("none", "pgs", "solid")}
    common = sorted(set.intersection(*(set(d["frequency_Hz"]) for d in groups.values())))
    conductance = {}
    for variant, data in groups.items():
        values = dict(zip(data["frequency_Hz"], unpack_network(data)))
        fitted = fit_pi(common, np.stack([values[f] for f in common]))
        conductance[variant] = float(pi_terms(pi_admittance(np.array([3.4e9]),
                                                            list(fitted["parameters"].values())))[0, 1:].real.sum())
    return conductance


def r4b_samples(comparison, case, new):
    """The committed comparison samples of one closed case, derived from its re-acquisition."""
    target = comparison["target_relative"]
    samples = []
    for j, (sample, reference) in enumerate(zip(case["samples"], comparison["quasi_static_reference"])):
        assert new["frequency_Hz"][j] == sample["frequency_Hz"] == reference["frequency_Hz"]
        l_rel = new["L_H"][j] / reference["series_L_H"] - 1
        r_rel = new["R_ohm"][j] / reference["series_R_ohm"] - 1
        samples.append(dict(frequency_Hz=sample["frequency_Hz"], L_H=new["L_H"][j], R_ohm=new["R_ohm"][j],
                            L_relative=l_rel, R_relative=r_rel, L_pass=abs(l_rel) <= target,
                            R_pass=abs(r_rel) <= target))
    return samples


def judged_parts(directory):
    """Parts of an acquisition (records written before the A4 parts existed hold the three legacy parts)."""
    return acquisition_record(directory).get("parts", list(LEGACY))


def a4_record(producer):
    """Re-run one A4 MoM producer record (scripts/dev/rfic/a4_fixtures_r7.py) on the current device."""
    from scripts.dev.rfic import a4_fixtures_r7 as a4
    if producer["fixture"] == "ff":
        from scripts.dev.rfic.octagon_reference_r7 import mom_record
        return mom_record(producer["edge"], producer["max"])
    if producer["fixture"] == "fd":
        return a4.fd(producer["edge"], producer["max"])
    return a4.fe(producer["fixture"][3:], producer["edge"], producer["max"])


def judge_a4(directory, evidence, parts, rows):
    for key in parts:
        if key not in A4:
            continue
        for label, record in evidence[key]["mom"].items():
            new = json.loads((directory / "a4" / key / (label + ".json")).read_text())
            compare(record, new, (A4[key], ["mom", label]), rows, "locked")


def judge(directory, accepted):
    from scripts.dev.rfic.summarize_fixture_r6 import collect_mpie
    evidence = committed()
    parts = judged_parts(directory)
    native, shunt, comparison = evidence["r6"], evidence["r6b"], evidence["r4b"]
    rows = []
    judge_a4(directory, evidence, parts, rows)
    if "r6" not in parts:
        return accept(rows, accepted)
    for old in r6_producers(native):
        compare(json.loads(old.read_text()), json.loads((directory / "mpie" / old.name).read_text()),
                ("producer/r6/" + old.name, []), rows, "report")
    collected = collect_mpie(directory / "mpie")
    for level in ("coarse", "finest"):
        for variant, frozen in native[level].items():
            actual = collected[variant + "/" + frozen["mesh"]]
            for key in ("metrics", "unknowns"):
                compare(frozen[key], actual[key], (FILES["r6"], [level, variant, key]), rows, "locked")
    compare(native["pi_effective_shunt_conductance_3p4GHz_S"], pi_conductance(collected),
            (FILES["r6"], ["pi_effective_shunt_conductance_3p4GHz_S"]), rows, "locked")
    for key in ("native_coarse", "native_previous", "native_fine"):
        compare(shunt[key], json.loads((directory / (key + ".json")).read_text()), (FILES["r6b"], [key]),
                rows, "locked")
    for i, case in enumerate(comparison["cases"]):
        old = r4b_producer(comparison, case["name"])
        new = json.loads((directory / case["name"]).read_text())
        frozen = json.loads(old.read_text())
        compare(frozen, dict(new, **{key: frozen[key] for key in ECHO if key in frozen}),
                ("producer/r4b/" + case["name"], []),
                rows, "report")
        compare(case["unknowns"], new["unknowns"], (FILES["r4b"], ["cases", i, "unknowns"]), rows, "locked")
        compare(case["samples"], r4b_samples(comparison, case, new), (FILES["r4b"], ["cases", i, "samples"]),
                rows, "locked")
    return accept(rows, accepted)


def accept(rows, accepted):
    keys = {entry["field"] for entry in accepted}
    for row in rows:
        if row["scope"] == "locked" and row["tier"] == "T2" and row["field"] in keys:
            if row.get("kind") == "discrete":
                raise SystemExit(f"discrete mover cannot be an accepted T2: {row['field']}")
            row["tier"] = "accepted_T2"
    used = {row["field"] for row in rows if row["tier"] == "accepted_T2"}
    if keys - used:
        raise SystemExit(f"accept entries that name no T2 field: {sorted(keys - used)}")
    return rows


def acquisition_record(directory):
    path = directory / "acquisition.json"
    return json.loads(path.read_text()) if path.exists() else dict(commit=None)


def tier_tables(directory, rows, accepted):
    tiers = ("T0", "T1", "accepted_T2", "T2")
    locked = [r for r in rows if r["scope"] == "locked"]
    report = [r for r in rows if r["scope"] == "report"]
    per_file = {FILES[key]: {t: sum(r.get("file") == FILES[key] and r["tier"] == t for r in locked) for t in tiers}
                for key in judged_parts(directory)}
    movers = [dict(field=r["field"], tier=r["tier"], max_rel_diff=r["max_rel_diff"],
                   delta_over_bound=r["delta_over_bound"]) for r in locked if r["tier"] in ("T1", "accepted_T2")]
    report_summary = {t: sum(r["tier"] == t for r in report) for t in tiers + ("fit", "residual", "diagnostic")}
    failing = [r["field"] for r in locked if r["tier"] == "T2"]
    acquisition = acquisition_record(directory)
    table = dict(rule=RULE, bound=f"fp64 run level: max|new-ref| <= {BOUND:g}*max|ref| per field",
                 acquisition=scrub(directory), acquisition_commit=acquisition["commit"],
                 locked=per_file, locked_movers=movers, accepted_t2=accepted, unaccepted_t2=failing,
                 report_only_producers=report_summary,
                 report_only_residual_gates_pass=all(r["passes_gate"] for r in report if r["tier"] == "residual"))
    (directory / "tier_table.json").write_text(json.dumps(table, indent=1) + "\n")
    (directory / "fields.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "new" or
                                                        r.get("kind") == "discrete"} for r in rows], indent=1) + "\n")
    lines = ["# Layered MoM evidence tier table (D-P1)", "",
             f"Acquisition `{table['acquisition']}` at commit `{acquisition['commit']}`; bound: {table['bound']}.", "",
             "| committed file | T0 | T1 | accepted T2 | T2 |", "|---|---|---|---|---|"]
    lines += [f"| {name} | " + " | ".join(str(count[t]) for t in tiers) + " |" for name, count in per_file.items()]
    lines += ["", "## Moved locked fields", "", "| field | tier | max rel diff | delta / bound |", "|---|---|---|---|"]
    lines += [f"| {m['field']} | {m['tier']} | {m['max_rel_diff']:.3g} | {m['delta_over_bound']:.3g} |"
              for m in sorted(movers, key=lambda m: -m["delta_over_bound"])]
    lines += ["", "## Accepted T2", ""] + [f"- {a['field']}: {a['approval']} ({a['analysis']})" for a in accepted]
    lines += ["", "## Unaccepted T2 (renewal refused while non-empty)", ""] + [f"- {f}" for f in failing]
    lines += ["", "## Report-only producer artifacts", "",
              ", ".join(f"{t} {n}" for t, n in report_summary.items()),
              f"; residual gates pass: {table['report_only_residual_gates_pass']}", "",
              "## Merge-declaration text", "",
              merge_text(table)]
    (directory / "tier_table.md").write_text("\n".join(lines) + "\n")
    return table


def merge_text(table):
    totals = {t: sum(c[t] for c in table["locked"].values()) for t in ("T0", "T1", "accepted_T2", "T2")}
    worst = max((m for m in table["locked_movers"] if m["tier"] == "T1"), key=lambda m: m["max_rel_diff"],
                default=None)
    text = (f"Layered evidence lock (D-P1), acquisition {table['acquisition_commit']}: locked fields "
            f"T0 {totals['T0']}, T1 {totals['T1']}, accepted T2 {totals['accepted_T2']}, T2 {totals['T2']}")
    if worst:
        text += f"; worst T1 {worst['field']} max_rel_diff {worst['max_rel_diff']:.3g}"
    for entry in table["accepted_t2"]:
        text += f"; accepted T2 {entry['field']} ({entry['approval']})"
    return text + f". Tier table {table['acquisition']}/tier_table.json."


def set_path(data, path, value):
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


def renew(directory, rows, table, accepted):
    if table["unaccepted_t2"]:
        raise SystemExit(f"renewal refused, unaccepted T2: {table['unaccepted_t2']}")
    if acquisition_record(directory)["commit"] is None:
        raise SystemExit("renewal needs DIR/acquisition.json with the acquisition commit")
    digest = hashlib.sha256((directory / "tier_table.json").read_bytes()).hexdigest()
    evidence = committed()
    for key in judged_parts(directory):
        name, data = FILES[key], evidence[key]
        moved = [r for r in rows if r["scope"] == "locked" and r.get("file") == name
                 and r["tier"] in ("T1", "accepted_T2")]
        for row in moved:
            set_path(data, row["path"], row["new"])
        lock = data.setdefault("numerical_lock", dict(rule=RULE, renewals=[]))
        lock["renewals"].append(dict(
            acquisition=table["acquisition"], acquisition_commit=table["acquisition_commit"],
            tier_table=table["acquisition"] + "/tier_table.json", tier_table_sha256=digest,
            summary=table["locked"][name], renewed_fields=[r["field"] for r in moved],
            accepted_t2=[a for a in accepted if a["field"].startswith(name + ":")]))
        (DATA / name).write_text(json.dumps(data, indent=2) + "\n")
        print(f"renewed {name}: {len(moved)} fields", flush=True)


def acquire(directory, parts):
    import torch
    import witwin.maxwell as mw
    from scripts.dev.rfic.analyze_fixture_r6 import acquire as acquire_r6
    from scripts.dev.rfic.native_fixture_r6b import record
    from tests.support.rfic_closed_references import network_metrics
    from tests.support.rfic_fixtures import fixture_fa
    from tests.support.rfic_r6b_references import FREQUENCIES, plate_fixture
    status = subprocess.run(["git", "status", "--porcelain", "witwin"], capture_output=True, text=True, check=True)
    assert not status.stdout, "acquisition needs a clean witwin/ tree:\n" + status.stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    directory.mkdir(parents=True, exist_ok=True)
    evidence = committed()
    if "r6" in parts:
        for old in r6_producers(evidence["r6"]):
            frozen = json.loads(old.read_text())
            label = old.stem.split("-sigma10", 1)[1].lstrip("-")
            acquire_r6(directory / "mpie", frozen["variant"], frozen["mesh"], frozen["grounded"],
                       frozen["substrate_sigma_S_m"], tuple(frozen["frequency_Hz"]), label, True)
    if "r6b" in parts:
        for key in ("native_coarse", "native_previous", "native_fine"):
            old = evidence["r6b"][key]
            scene, config = plate_fixture(round(old["metadata"]["max_edge"] * 1e6, 8),
                                          round(old["metadata"]["edge_cell"] * 1e6, 8))
            record(directory, key, mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=config),
                   old["exact_repeats"])
            gc.collect()
            torch.cuda.empty_cache()
    if "r4b" in parts:
        for case in evidence["r4b"]["cases"]:
            frozen = json.loads(r4b_producer(evidence["r4b"], case["name"]).read_text())
            fixture = fixture_fa(substrate=frozen["substrate"], lossy=not frozen["pec"], mesh="test",
                                 top_thickness_um=frozen["thickness_um"])
            config = replace(fixture.config, layered=replace(
                fixture.config.layered, conductor_model=frozen["model"],
                max_edge_length_m=frozen["maximum_um"] * 1e-6, edge_cell_m=frozen["edge_um"] * 1e-6))
            frequencies = tuple(frozen["frequency_Hz"])
            result = mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=config).run()
            y = result.network.to_y().detach().cpu()
            data = network_metrics(result)
            data.update(unknowns=result.network.metadata["unknowns"], y_real=y.real.tolist(),
                        y_imag=y.imag.tolist(), metadata=dict(result.network.metadata),
                        reciprocity=((y - y.transpose(-1, -2)).abs().max() / y.abs().max()).item(),
                        max_singular_S=torch.linalg.matrix_norm(result.network.s, ord=2).max().item())
            zin = 1 / y[:, 0, 0]
            srf = None
            for i in range(len(frequencies) - 1):
                if zin.imag[i] < 0 <= zin.imag[i + 1]:
                    fraction = (-zin.imag[i] / (zin.imag[i + 1] - zin.imag[i])).item()
                    srf = frequencies[i] + fraction * (frequencies[i + 1] - frequencies[i])
                    break
            peak = max(range(len(data["Q"])), key=lambda i: data["Q"][i])
            data.update(SRF_Hz=srf, Q_peak=data["Q"][peak], Q_peak_frequency_Hz=frequencies[peak],
                        **{key: frozen[key] for key in ECHO})
            (directory / case["name"]).write_text(json.dumps(data, indent=2) + "\n")
            del result
            gc.collect()
            torch.cuda.empty_cache()
    for key in parts:
        if key not in A4:
            continue
        for label, record in evidence[key]["mom"].items():
            out = directory / "a4" / key / (label + ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(a4_record(record["producer"]) | dict(producer=record["producer"]),
                                      indent=1) + "\n")
            gc.collect()
            torch.cuda.empty_cache()
    status = subprocess.run(["git", "status", "--porcelain", "witwin"], capture_output=True, text=True, check=True)
    assert not status.stdout, "witwin/ changed during acquisition"
    (directory / "acquisition.json").write_text(json.dumps(dict(
        commit=commit, parts=sorted(parts), driver="scripts/dev/rfic/evidence_lock.py acquire",
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0)), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("acquire", "judge", "renew"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parts", nargs="+", default=["r6b", "r4b", "r6"],
                        help="r6 r6b r4b (legacy, acquired together) and/or the A4 parts fd fe ff")
    parser.add_argument("--cap-gib", type=float, help="per-process allocator cap for acquire")
    parser.add_argument("--accept-t2", type=Path,
                        help='JSON {"accepted_t2": [{"field": "<file>:<path>", "approval": .., "analysis": ..}]}')
    args = parser.parse_args()
    if args.mode == "acquire":
        import torch
        torch.cuda.set_per_process_memory_fraction(
            args.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        acquire(args.directory, args.parts)
    else:
        accepted = json.loads(args.accept_t2.read_text())["accepted_t2"] if args.accept_t2 else []
        rows = judge(args.directory, accepted)
        table = tier_tables(args.directory, rows, accepted)
        print(merge_text(table), flush=True)
        if args.mode == "renew":
            renew(args.directory, rows, table, accepted)
        elif table["unaccepted_t2"]:
            sys.exit(1)
