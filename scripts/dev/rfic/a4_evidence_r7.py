"""Assemble the A4 (R7-2) committed evidence files from a MoM campaign and the reference generator outputs.

python a4_evidence_r7.py SDA_JSON FH_DIR --campaigns DIR... [--proportional-ff] [--attribution DIR...]
  DIR:          scripts/dev/rfic/a4_campaign_r7.py outputs (<tag>.json, campaign.json), in priority order
  --proportional-ff: F-F proportional sequence (set r2, max = 2 edge); default the edge-only sequence of set a4
  --attribution: directories holding the microstrip attribution runs (report-only)
  SDA_JSON:     python scripts/dev/rfic/quasistatic_sda_r7.py references > SDA_JSON
  FH_DIR:       FastHenry runs of scripts/dev/rfic/octagon_reference_r7.py (<name>/model.inp, gen.log, result.json)
Writes tests/mom/layered/data/fixture_fd_mim_evidence.json, fixture_fe_lines_evidence.json,
fixture_ff_octagon_evidence.json with a fresh D-P1 numerical_lock (renewals are appended by
scripts/dev/rfic/evidence_lock.py renew on the campaign's re-acquisition).
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.dev.rfic.evidence_lock import RULE, scrub  # noqa: E402
from tests.support import rfic_a4_fixtures as fx  # noqa: E402
from tests.support.rfic_a4_fixtures import richardson  # noqa: E402

DATA = ROOT / "tests/mom/layered/data"
C0 = 299792458.0
# Locked acceptance sequences (edge cell, maximum edge in um; None = fixture default) and one-off finer checks.
SEQUENCES = {"mim": ("fd", [(e, None) for e in (.5, .25, .125)]),
             "microstrip": ("fe-microstrip", [(254 / n, None) for n in (16, 32, 64)]),
             "cpw": ("fe-cpw", [(e, None) for e in (2.5, 1.25, .625)]),
             "inductor": ("ff", [(e, 2 * e) for e in (2., 2 ** .5, 1.)])}
CHECKS = {"mim": ("fd", [(.0625, None)]), "microstrip": ("fe-microstrip", [(254 / 128, None)]),
          "cpw": ("fe-cpw", [(.3125, None)]), "inductor": ("ff", [(2 ** -.5, 2 ** .5)])}
EDGE_ONLY = {"inductor": ("ff", [(e, None) for e in (2., 2 ** .5, 1.)])}, {"inductor": ("ff", [(2 ** -.5, None)])}
ATTRIBUTION = [("fe-microstrip", 254 / 32, m) for m in (63.5, 31.75, 15.875)] + \
              [("fe-microstrip-air", 254 / n, None) for n in (16, 32, 64)]


# Rows outside their derived tolerance, with the quantified attribution (T2: open for the user's decision).
MICROSTRIP_OFFSET = dict(
    attribution="PEC-ground microstrip (h = w = 254 um, eps 9.8): the de-embedded L' = eps_eff/(c^2 C') converges "
    "first order in the transverse edge cell (W/16..W/128: +0.970/+0.674/+0.526/+0.453%, steps halving) to a nonzero "
    "limit, while C' converges to the exact value (limit +0.02%). eps_eff carries the L' offset, Z0 half of it. Not the "
    "transverse mesh and not the de-embedding (two-line gamma, length-differenced C'). The finite-ground CPW on a "
    "half-space with the same edge-cell sequence and cell aspect up to 64 has no offset (L' limit +0.005%), which "
    "points at the PEC-ground path. The longitudinal cell is excluded: halving the maximum edge (127 -> 63.5 um, "
    "W/32) moves L' by +0.003% (fe-microstrip-e7.9375um-m63.5um). The dielectric is excluded: the air-filled line "
    "(fe-microstrip-air-e7.9375um, exact eps_eff = 1, pure TEM) has C' exact to +0.002% and the same L' error "
    "(+0.674% vs +0.6735% with eps 9.8). Remaining term: the vector-potential (horizontal current) part over the PEC "
    "ground, i.e. its PEC image / the image-extracted Green table on that path; the scalar-potential image is exact "
    "(C'). To be localized in the solver (layered green/assembly owners); no solver change here.",
    quantified=dict(L_per_length_limit_rel=3.79e-3, C_per_length_limit_rel=2e-4, eps_eff_limit_rel=4.0e-3,
                    Z0_limit_rel=1.8e-3))
INDUCTOR_OFFSET = dict(
    attribution="Edge-only mesh sequence (edge 2/1.414/1 um, maximum 4 um) is outside the asymptotic range "
    "(steps -0.06%, -0.46%), so no extrapolation is made and the derived tolerance is the larger step (0.47%) plus the "
    "reference bound (0.18%) = 0.64%, against +0.656% at 1 um. The 0.707 um check gives +0.558%. The sheet-edge "
    "current converges first order from above (straight trace pair, same mesher: +5.3/+2.8% at 2/1 um against the exact "
    "2D value), so the residual is the edge discretization still present at 1 um, inside the 1% plan target.",
    quantified=dict(finest_rel=6.56e-3, check_0p707um_rel=5.58e-3, pair_per_length_rel_2um=5.26e-2,
                    pair_per_length_rel_1um=2.80e-2))


def tag(fixture, edge_um, max_um=None):
    return f"{fixture}-e{edge_um:.6g}um" + (f"-m{max_um:.6g}um" if max_um else "")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


DEVICES = {}


def find(campaigns, label):
    """First campaign directory (priority order) holding the record ``label``."""
    return next(d for d in campaigns if (d / (label + ".json")).exists())


def records(campaigns, prefixes):
    """Locked sequences and one-off checks; the locked records of one file must come from one device type."""
    mom, sequences, checks, devices = {}, {}, {}, set()
    for prefix in prefixes:
        fixture, levels = SEQUENCES[prefix]
        sequences[prefix] = [tag(fixture, *level) for level in levels]
        for label in sequences[prefix]:
            source = find(campaigns, label)
            mom[label] = json.loads((source / (label + ".json")).read_text())
            devices.add(json.loads((source / "campaign.json").read_text())["device"])
        fixture, levels = CHECKS[prefix]
        for level in levels:
            label = tag(fixture, *level)
            checks[label] = json.loads((find(campaigns, label) / (label + ".json")).read_text())
    assert len(devices) == 1, devices
    DEVICES[prefixes] = devices.pop()
    return mom, sequences, checks


def lock():
    return dict(rule=RULE, renewals=[])


def sda_provenance(sda_path):
    return dict(generator="scripts/dev/rfic/quasistatic_sda_r7.py", generator_sha256=sha(ROOT / "scripts/dev/rfic/quasistatic_sda_r7.py"),
                version=f"witwin-maxwell {git_commit()} (numpy/scipy, CPU fp64)", inputs=dict(orders=32,
                command="python scripts/dev/rfic/quasistatic_sda_r7.py references"),
                outputs=scrub(sda_path), outputs_sha256=sha(sda_path),
                self_checks="CPW vs exact finite-ground conformal map 6e-12; microstrip grounded vs image pair 3e-12; "
                            "Hammerstad-Jensen -0.07% eps_eff (stated 0.2%)")


def build_fd(campaign, sda, sda_path):
    mom, sequences, checks = records(campaign, ("mim",))
    device = DEVICES[("mim",)]
    beta = [2 * math.pi * f * math.sqrt(fx.MIM_EPS) / C0 for f in mom[sequences["mim"][0]]["frequency_Hz"]]
    l1, l2 = 60e-6, 120e-6
    # Distributed plate line: C_in = C tan(bl)/(bl) ~ C (1 + (bl)^2/3); the length difference keeps
    # b^2 (l1^2 + l1 l2 + l2^2)/3, bounded here with a factor 2.
    bound = [2 * b * b * (l1 * l1 + l1 * l2 + l2 * l2) / 3 for b in beta]
    return dict(
        fixture="F-D", description="MIM capacitor: equal PEC plates on the faces of a 1 um nitride film (eps 7) in oxide "
        "(tests/support/rfic_a4_fixtures.mim); vertical delta-gap port at the +x edge. C' = dC/dL from 60/120 um plates "
        "at equal width cancels port, ends and corners; squares checked against eps A/d + 4 s f(s).",
        reference=dict(values=dict(fd=sda["fd"]), quasi_static_bound=bound, provenance=sda_provenance(sda_path)),
        sequences=sequences, fast_record=sequences["mim"][0], mom=mom, checks=checks, acquisition_device=device,
        numerical_lock=lock())


def build_fe(campaign, attribution_dirs, sda, sda_path):
    mom, sequences, checks = records(campaign, ("microstrip", "cpw"))
    device = DEVICES[("microstrip", "cpw")]
    # Attribution runs (ATTRIBUTION list) found in the given campaign directories; report-only, not locked.
    attribution = {tag(*job): json.loads((d / (tag(*job) + ".json")).read_text())
                   for job in ATTRIBUTION for d in attribution_dirs if (d / (tag(*job) + ".json")).exists()}
    bounds = {}
    for kind, (l1, l2), eps in (("microstrip", fx.MS_LENGTHS, sda["fe_microstrip"]["eps_eff"]),
                                ("cpw", fx.CPW_LENGTHS, sda["fe_cpw"]["eps_eff"])):
        b = 2 * math.pi * 1e7 * math.sqrt(eps) / C0
        bounds[kind] = 2 * b * b * (l1 * l1 + l1 * l2 + l2 * l2) / 3
    return dict(
        fixture="F-E", description="Microstrip on a PEC ground (h = w = 254 um, eps 9.8) and finite-ground CPW on a "
        "silicon half-space (10/10/40 um), full-width edge ports (tests/support/rfic_a4_fixtures). De-embedded line "
        "quantities: two-line gamma (deembed.line_parameters, error-box free) -> eps_eff(f), quasi-static limit by the "
        "even f^2 dispersion term; C' = dC_open/dL at 10 MHz (ports and ends cancel); Z0 = sqrt(eps_eff)/(c C').",
        reference=dict(values=dict(fe_microstrip=sda["fe_microstrip"], fe_cpw=sda["fe_cpw"]),
                       dispersion_pairs=dict(microstrip=[[3e8, 1e9], [1e9, 3e9]], cpw=[[2e9, 5e9], [5e9, 1e10]]),
                       low_frequency_bound=bounds, provenance=sda_provenance(sda_path)),
        sequences=sequences, fast_record=sequences["cpw"][0], mom=mom, checks=checks, attribution=attribution,
        known_offsets={"microstrip L' [H/m]": MICROSTRIP_OFFSET, "microstrip eps_eff (f->0)": MICROSTRIP_OFFSET},
        acquisition_device=device, numerical_lock=lock())


def build_ff(campaign, fh, sda, sda_path):
    mom, sequences, checks = records(campaign, ("inductor",))
    device = DEVICES[("inductor",)]
    runs = {}
    for directory in sorted(p for p in fh.iterdir() if (p / "result.json").exists()):
        generated = json.loads((directory / "gen.log").read_text())
        result = json.loads((directory / "result.json").read_text())
        runs[directory.name] = dict(name=directory.name, n=generated["n"], step=generated["step"], h=generated["h"],
                                    ratio=generated["ratio"], segments=generated["segments"],
                                    input_sha256=sha(directory / "model.inp"), L_H=result["L_H"])
    pair_runs = {}
    for directory in sorted(p for p in (fh.parent / "fhpair").iterdir() if (p / "result.json").exists()):
        generated = json.loads((directory / "gen.log").read_text())
        pair_runs[directory.name] = dict(name=directory.name, n=generated["n"], h=generated["h"],
                                         length=generated["length"], ratio=generated["ratio"],
                                         segments=generated["segments"], input_sha256=sha(directory / "model.inp"),
                                         L_H=json.loads((directory / "result.json").read_text())["L_H"])
    value = ff_reference(runs, pair_runs, sda["ff_pair"]["L_per_length_H_per_m"])
    return dict(
        fixture="F-F", description="Two-turn symmetric octagonal differential inductor (apothems 70/58 um, w 8, s 4, "
        "top-plane diagonal and underpass crossover, 4x4 um vias), PEC sheets in lossless oxide, one floating port "
        "(tests/support/rfic_a4_fixtures.octagonal_inductor); polygon (triangle) mesher.",
        reference=dict(values=value, provenance=dict(
            generator="scripts/dev/rfic/octagon_reference_r7.py",
            generator_sha256=sha(ROOT / "scripts/dev/rfic/octagon_reference_r7.py"),
            version="FastHenry 3.0wr (tools/fasthenry-build, -m direct -t 1e-10), "
                    f"generator at witwin-maxwell {git_commit()}",
            inputs=dict(frequency_Hz=1e8, units="mm with um numbers, f/1000"), runs=list(runs.values()),
            pair_runs=list(pair_runs.values()), outputs=scrub(fh), pair_exact=sda_provenance(sda_path))),
        sequences=sequences, fast_record=sequences["inductor"][0], mom=mom, checks=checks,
        known_offsets={"differential L at 1e8 Hz [H]": INDUCTOR_OFFSET} if SEQUENCES["inductor"][1][0][1] is None else {},
        acquisition_device=device, numerical_lock=lock())


def ff_reference(runs, pair_runs, pair_exact):
    """L of the octagonal inductor from the sub-strip lattice, calibrated on the trace pair, and its relative uncertainty.

    Per sub-strip count n: zero-thickness limit L0(n) by Richardson in h (0.2/0.1/0.05 um, ratio 2). The same procedure
    on the straight trace pair (octagon_reference_r7.pair) has the error e(n) = L0'(n)/L'_exact - 1 against the exact
    2D SDA value; its convergence in n is not first order at n <= 16 (an n = 8/12/16 extrapolation overshoots by 0.6%),
    so the inductor is not extrapolated in n. Instead its sensitivity to n relative to the pair,
    k = [L0(8) - L0(n*)]/L0(n*) / [e(8) - e(n*)], scales the pair error at the finest inductor n*:
    L_ref = L0(n*) (1 - k e(n*)), u = k e(n*) (the whole correction) + |e(n_max)| of the pair (residual of the procedure)
    + the station-spacing effect (step 2 vs 4 um at n = 8).
    """
    def L(name):
        return runs[name]["L_H"]

    thin = {}
    for n in sorted(int(name[1:]) for name in runs if name[1:].isdigit()):
        if f"n{n}-h0.2" in runs and f"n{n}-h0.05" in runs:
            thin[n] = richardson([L(f"n{n}-h0.2"), L(f"n{n}"), L(f"n{n}-h0.05")], 2.)
    pair = {}
    for n in sorted({int(name.split("-")[0][1:]) for name in pair_runs}):
        per_length = [(pair_runs[f"n{n}-h{h}-X800"]["L_H"] - pair_runs[f"n{n}-h{h}-X400"]["L_H"]) / 400e-6
                      for h in ("0.2", "0.1", "0.05")]
        pair[n] = richardson(per_length, 2.)["value"] / pair_exact - 1
    finest = max(thin)
    k = (thin[8]["value"] - thin[finest]["value"]) / thin[finest]["value"] / (pair[8] - pair[finest])
    correction = k * pair[finest]
    step = abs(L("n8-s2") / L("n8") - 1)
    u = abs(correction) + abs(pair[max(pair)]) + step
    return dict(L_H=thin[finest]["value"] * (1 - correction), u_rel=u, finest_substrip_count=finest,
                zero_thickness={str(n): t for n, t in thin.items()}, pair_error={str(n): e for n, e in pair.items()},
                sensitivity_ratio=k, correction_rel=correction, station_step_effect_rel=step,
                pair_exact_L_per_length_H_per_m=pair_exact)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sda", type=Path)
    parser.add_argument("fh", type=Path)
    parser.add_argument("--campaigns", type=Path, nargs="+", required=True,
                        help="campaign directories in priority order (locked records: first match; one device per file)")
    parser.add_argument("--proportional-ff", action="store_true",
                        help="F-F proportional sequence (set r2) instead of the edge-only sequence of set a4")
    parser.add_argument("--attribution", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    sda = json.loads(args.sda.read_text())
    if not args.proportional_ff:
        SEQUENCES.update(EDGE_ONLY[0])
        CHECKS.update(EDGE_ONLY[1])
    # The D-P1 T0 lock holds on one device type (fp32 far-field arithmetic differs ~1e-8 across architectures).
    for name, data in (("fixture_fd_mim_evidence.json", build_fd(args.campaigns, sda, args.sda)),
                       ("fixture_fe_lines_evidence.json", build_fe(args.campaigns, args.attribution, sda, args.sda)),
                       ("fixture_ff_octagon_evidence.json", build_ff(args.campaigns, args.fh, sda, args.sda))):
        (DATA / name).write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", name, data["acquisition_device"])
