"""Every boundary class the opt-in core path admits has an explicit G1 gate scene."""
from types import SimpleNamespace

import pytest

import witwin.maxwell as mw
from benchmark.core_path.scenes import CORE_REFUSED_BENCHMARK_SCENES, GATE_SCENES, build_core_case
from witwin.maxwell.fdtd.boundary.runtime import resolve_absorber
from witwin.maxwell.fdtd.runtime.core_path import core_path_blockers

_PML = dict(num_layers=2, strength=1e6)
# Class -> (boundary, requested absorber).
CLASSES = {
    "cpml": (mw.BoundarySpec.pml(**_PML), "cpml"),
    "stablepml": (mw.BoundarySpec.pml(**_PML), "stablepml"),
    "pml": (mw.BoundarySpec.pml(**_PML), "pml"),
    "absorber": (mw.BoundarySpec.pml(**_PML), "absorber"),
    "none": (mw.BoundarySpec.none(), "cpml"),
    "pec": (mw.BoundarySpec.pec(), "cpml"),
    # Real periodic + PML resolves to the graded absorber under absorber="auto"; explicit CPML in this
    # topology is refused (see test_periodic_cpml_is_refused).
    "periodic": (mw.BoundarySpec.faces(default="periodic", z="pml", **_PML), "absorber"),
    "bloch": (mw.BoundarySpec.faces(default="bloch", bloch_wavevector=(1.0, 0.0, 0.0), z="pml", **_PML), "cpml"),
}


def _solver(boundary, absorber):
    """The boundary attributes core_path_blockers reads, resolved as boundary/runtime.py does."""
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.05),
                     boundary=boundary, device="cpu")
    active = absorber if boundary.uses_kind("pml") else "none"
    return SimpleNamespace(scene=scene, active_absorber_type=active, uses_cpml=active in ("cpml", "stablepml"),
                           complex_fields_enabled=boundary.uses_kind("bloch"))


def _classes(boundary, absorber):
    found = {absorber if boundary.uses_kind("pml") else "none"}
    found |= {kind for kind in ("pec", "periodic", "bloch") if boundary.uses_kind(kind)}
    return found


def _admitted():
    return {name for name, (boundary, absorber) in CLASSES.items()
            if set(core_path_blockers(_solver(boundary, absorber))) <= {"non-CUDA execution"}}


def test_admitted_boundary_classes():
    # Widen this set only together with a G1 gate scene for the new class.
    assert _admitted() == {"cpml", "stablepml", "pml", "absorber", "none", "pec", "periodic"}


def test_each_admitted_boundary_class_has_a_gate_scene():
    covered = set()
    for name in GATE_SCENES:
        case = build_core_case(name)
        active, _ = resolve_absorber(
            case.scene, case.absorber or "auto", complex_fields=case.scene.boundary.uses_kind("bloch"),
        )
        covered |= _classes(case.scene.boundary, active)
        solver = SimpleNamespace(scene=case.scene, active_absorber_type=active)
        assert "closed-transverse CPML (passivity guard)" not in core_path_blockers(solver), name
    assert _admitted() <= covered, _admitted() - covered


def test_periodic_cpml_is_refused():
    boundary = mw.BoundarySpec.faces(default="periodic", z="pml", **_PML)
    assert "closed-transverse CPML (passivity guard)" in core_path_blockers(_solver(boundary, "cpml"))


@pytest.mark.parametrize("name", sorted(CORE_REFUSED_BENCHMARK_SCENES | {"planewave_periodic_vacuum"}))
def test_benchmark_fixtures_keep_the_closed_transverse_cpml_refusal(name):
    case = build_core_case(name)
    assert case.absorber == "cpml"
    assert name not in GATE_SCENES
    solver = SimpleNamespace(scene=case.scene, active_absorber_type=case.absorber)
    assert "closed-transverse CPML (passivity guard)" in core_path_blockers(solver)
