"""A4 certification fixtures with internal independent references (R7-2): F-D MIM, F-E lines, F-F octagonal inductor.

References (independent of the layered MoM, provenance recorded in each evidence file):
- F-D / F-E: scripts/dev/rfic/quasistatic_sda_r7.py, a 2D quasi-static spectral-domain Galerkin solver (exact
  finite-ground CPW conformal map to 1e-9 here; Hammerstad-Jensen within its stated accuracy), plus the
  Kirschning-Jansen dispersion of the microstrip with a stated 30% bound on the dispersion term;
- F-F: FastHenry on the true-geometry sub-strip lattice (scripts/dev/rfic/octagon_reference_r7.py), extrapolated in
  sub-strip count and to zero thickness.
Each acceptance compares the Richardson-extrapolated MoM mesh sequence against the reference with the tolerance
u_ref + u_mesh (reference convergence/validity bound + 1.25 |q_ext - q_finest|), and the finest mesh against the
plan-level target. Evidence: tests/mom/layered/data/fixture_f{d,e,f}_*.json, locked by value (D-P1,
scripts/dev/rfic/evidence_lock.py parts fd/fe/ff).
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from tests.support import rfic_a4_fixtures as fx

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).with_name("data")
FILES = {"fd": "fixture_fd_mim_evidence.json", "fe": "fixture_fe_lines_evidence.json",
         "ff": "fixture_ff_octagon_evidence.json"}
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


def evidence(key):
    return json.loads((DATA / FILES[key]).read_text())


def sda():
    sys.path.insert(0, str(ROOT / "scripts/dev/rfic"))
    import quasistatic_sda_r7
    return quasistatic_sda_r7


def test_sda_reference_against_closed_forms():
    q = sda()
    um = 1e-6
    cpw = q.line(q.Stack([(100 * um, 11.9)], 11.9, 1.), q.cpw_strips(10 * um, 10 * um, 40 * um, 100 * um), "sig",
                 "gnd", orders=16)
    eps, z0 = q.ghione_naldi(10 * um, 10 * um, 40 * um, 11.9)
    # Zero-thickness CPW on the interface of two half-spaces: the conformal map is exact.
    assert abs(cpw["eps_eff"] / eps - 1) < 1e-9 and abs(cpw["Z0_ohm"] / z0 - 1) < 1e-9
    h = 254 * um
    ms = q.line(q.Stack([(h, 9.8)], "pec", 1.), [("sig", 0., h / 2, h)], "sig", None, orders=16)
    eps, z0 = q.hammerstad_jensen(1., 9.8)
    # Hammerstad-Jensen: eps_eff within 0.2%, Z0 within 0.2% (stated accuracy incl. the eps part).
    assert abs(ms["eps_eff"] / eps - 1) < 2e-3 and abs(ms["Z0_ohm"] / z0 - 1) < 2e-3
    # The grounded path equals the ungrounded image pair (two independent code paths).
    pair = q.capacitance(q.Stack([(2 * h, 1.)], 1., 1.), [("a", 0., h / 2, 0.), ("b", 0., h / 2, 2 * h)], ["a", "b"],
                         orders=16)
    assert abs(2 * pair[0, 0] / ms["C_air_F_per_m"] - 1) < 1e-9


@pytest.mark.parametrize("key", ("fd", "fe"))
def test_sda_reference_reproduces_the_committed_values(key):
    committed = evidence(key)["reference"]
    values = sda().references(orders=committed["provenance"]["inputs"]["orders"])
    names = ("fd",) if key == "fd" else ("fe_microstrip", "fe_cpw")
    for name in names:
        fresh, frozen = values[name], committed["values"][name]
        for field, value in frozen.items():
            if isinstance(value, float):
                assert abs(fresh[field] - value) <= 1e-10 * abs(value), (name, field)
    assert committed["provenance"]["generator"] == "scripts/dev/rfic/quasistatic_sda_r7.py"


def test_ff_reference_inputs_are_generated_by_the_recorded_generator():
    from scripts.dev.rfic.octagon_reference_r7 import generate
    provenance = evidence("ff")["reference"]["provenance"]
    for run in provenance["runs"]:
        text, segments = generate(run["n"], run["step"], run["h"], run["ratio"])
        assert hashlib.sha256(text.encode()).hexdigest() == run["input_sha256"], run["name"]
        assert segments == run["segments"]


def acceptance(key):
    from tests.support.rfic_a4_acceptance import ACCEPTANCE
    return ACCEPTANCE[key](evidence(key))


def known_offsets(key):
    """Rows outside their derived tolerance with a quantified attribution in the evidence (T2, user decision)."""
    return evidence(key).get("known_offsets", {})


@pytest.mark.parametrize("key", ("fd", "fe", "ff"))
def test_a4_acceptance(key, record_property):
    rows = acceptance(key)
    record_property("acceptance", rows)
    offsets = known_offsets(key)
    failing = [row for row in rows if not row["pass"] and row["quantity"] not in offsets]
    assert not failing, failing
    for quantity, record in offsets.items():
        assert record["attribution"].strip() and record["quantified"], quantity


@pytest.mark.parametrize("key", [key for key in FILES if json.loads((DATA / FILES[key]).read_text()).get("known_offsets")])
@pytest.mark.xfail(strict=True, reason="rows outside the derived tolerance with a quantified attribution (T2, open): "
                   "see known_offsets in the evidence file")
def test_a4_known_offsets_closed(key):
    rows = [row for row in acceptance(key) if row["quantity"] in known_offsets(key)]
    assert rows and all(row["pass"] for row in rows)


PRIMARY = {"fd": ("C_per_length_F_per_m",), "fe": ("eps_eff", "C_per_length_F_per_m"), "ff": ("L_H",)}


def _leaves(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in _leaves(v)]
    return list(value) if isinstance(value, list) else [value]


def unknowns(record):
    return record["unknowns"] if "unknowns" in record else [case["unknowns"] for case in record["cases"]]


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("key", ("fd", "fe", "ff"))
def test_a4_coarsest_record_reproduces(key):
    """Re-run the coarsest locked record of each fixture on the local device.

    The locked values were acquired on one device type (D-P1 T0 on re-acquisition there); across GPU architectures the
    fp32 far-field arithmetic moves the primary quantities by ~1e-8 relative (L40S vs A6000), so the primary quantities
    are checked at 1e-6 (inside the 1e-4 run-level bound of the numerical-equivalence standard), discrete fields exactly.
    """
    from scripts.dev.rfic.evidence_lock import a4_record
    data = evidence(key)
    record = data["mom"][data["fast_record"]]
    fresh = a4_record(record["producer"])
    assert fresh["exact_repeat"] is True
    assert unknowns(fresh) == unknowns(record)
    for field in PRIMARY[key]:
        old, new = _leaves(record[field]), _leaves(fresh[field])
        assert len(old) == len(new)
        assert max(abs(a / b - 1) for a, b in zip(new, old)) <= 1e-6, field
