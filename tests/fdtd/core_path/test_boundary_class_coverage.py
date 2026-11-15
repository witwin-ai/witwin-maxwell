"""Every boundary class runs on the core path, and each has an explicit G1 gate scene."""
import pytest

import witwin.maxwell as mw
from benchmark.core_path.scenes import GATE_SCENES, W7R5_SCENES, build_core_case
from benchmark.scenes import CPML_PINNED_SCENARIOS
from witwin.maxwell.fdtd.boundary.runtime import closed_transverse_pml, resolve_absorber

_PML = dict(num_layers=2, strength=1e6)
# Class -> (boundary, requested absorber).
CLASSES = {
    "cpml": (mw.BoundarySpec.pml(**_PML), "cpml"),
    "stablepml": (mw.BoundarySpec.pml(**_PML), "stablepml"),
    "pml": (mw.BoundarySpec.pml(**_PML), "pml"),
    "absorber": (mw.BoundarySpec.pml(**_PML), "absorber"),
    "none": (mw.BoundarySpec.none(), "cpml"),
    "pec": (mw.BoundarySpec.pec(), "cpml"),
    "pmc": (mw.BoundarySpec.pmc(), "cpml"),
    "mur": (mw.BoundarySpec.mur(), "cpml"),
    "mode_abc": (mw.BoundarySpec.faces(default="pml", x_high="mode_abc", **_PML), "cpml"),
    # Real periodic + PML resolves to the graded absorber under absorber="auto"; explicit CPML in this
    # closed-transverse topology runs under the passivity guard (gate scenes: CPML_PINNED_SCENARIOS).
    "periodic": (mw.BoundarySpec.faces(default="periodic", z="pml", **_PML), "absorber"),
    "periodic_cpml": (mw.BoundarySpec.faces(default="periodic", z="pml", **_PML), "cpml"),
    "bloch": (mw.BoundarySpec.faces(default="bloch", bloch_wavevector=(1.0, 0.0, 0.0), z="pml", **_PML), "cpml"),
}


def _classes(boundary, absorber):
    found = {absorber if boundary.uses_kind("pml") else "none"}
    found |= {kind for kind in ("pec", "pmc", "mur", "mode_abc", "periodic", "bloch") if boundary.uses_kind(kind)}
    return found


def _admitted():
    # The core path has no admission blocker (W7 row 16c): every class runs on it.
    return set(CLASSES)


def test_admitted_boundary_classes():
    # A new class joins together with a G1 gate scene for it.
    assert _admitted() == {"cpml", "stablepml", "pml", "absorber", "none", "pec", "pmc", "mur", "mode_abc",
                           "periodic", "periodic_cpml", "bloch"}


def test_each_admitted_boundary_class_has_a_gate_scene():
    covered = set()
    for name in GATE_SCENES:
        case = build_core_case(name)
        active, _ = resolve_absorber(
            case.scene, case.absorber or "auto", complex_fields=case.scene.boundary.uses_kind("bloch"),
        )
        found = _classes(case.scene.boundary, active)
        if active == "cpml" and found & {"periodic", "pec"} and closed_transverse_pml(case.scene):
            found.add("periodic_cpml")
        covered |= found
    assert _admitted() <= covered, _admitted() - covered


@pytest.mark.parametrize("name", sorted(CPML_PINNED_SCENARIOS))
def test_cpml_pinned_benchmark_fixtures_are_gate_scenes(name):
    # Every CPML-pinned closed-transverse benchmark cell is a G1 gate scene on the core path.
    case = build_core_case(name)
    assert case.absorber == "cpml"
    assert closed_transverse_pml(case.scene)
    assert name in W7R5_SCENES and name in GATE_SCENES
