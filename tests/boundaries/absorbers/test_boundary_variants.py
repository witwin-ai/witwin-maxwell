import types

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.common import BOUNDARY_NONE, BOUNDARY_KIND_TO_CODE
from witwin.maxwell.fdtd.boundary.runtime import _configure_face_boundary_codes
from witwin.maxwell.simulation import AbsorberKind, _normalize_absorber_kind


# --------------------------------------------------------------------------- #
# Tier A: construction / plumbing (CPU, no solver)                            #
# --------------------------------------------------------------------------- #


def test_mur_boundary_resolves_all_faces_to_mur():
    boundary = mw.BoundarySpec.mur()
    assert boundary.kind == "mur"
    for axis in ("x", "y", "z"):
        for side in ("low", "high"):
            assert boundary.face_kind(axis, side) == "mur"
    assert boundary.uses_kind("mur")
    # Mur is not a PML variant and must not request PML layers.
    assert boundary.num_layers == 0


def test_mur_faces_map_to_inert_boundary_code():
    assert BOUNDARY_KIND_TO_CODE["mur"] == BOUNDARY_NONE

    solver = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            boundary=mw.BoundarySpec.mur(),
            symmetry=(None, None, None),
        )
    )
    _configure_face_boundary_codes(solver)

    for axis in ("x", "y", "z"):
        assert getattr(solver, f"boundary_{axis}_low_code") == BOUNDARY_NONE
        assert getattr(solver, f"boundary_{axis}_high_code") == BOUNDARY_NONE
    assert solver.has_mur_faces is True
    assert set(solver.mur_faces) == {
        (axis, side) for axis in ("x", "y", "z") for side in ("low", "high")
    }
    # Inert faces must not be mistaken for PEC / PML.
    assert solver.has_pec_faces is False
    assert solver.has_pml_faces is False


def test_absorber_kind_normalizes_new_variants():
    assert _normalize_absorber_kind("absorber") is AbsorberKind.ABSORBER
    assert _normalize_absorber_kind("stablepml") is AbsorberKind.STABLE_PML
    assert _normalize_absorber_kind("STABLEPML") is AbsorberKind.STABLE_PML
    assert _normalize_absorber_kind(AbsorberKind.ABSORBER) is AbsorberKind.ABSORBER
    with pytest.raises(ValueError):
        _normalize_absorber_kind("not-a-real-absorber")


# --------------------------------------------------------------------------- #
# Tier B: real physics (CUDA)                                                 #
# --------------------------------------------------------------------------- #

_FREQ = 3e8
_FWIDTH = 1.5e8


def _build_scene(boundary):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.9, 0.9), (-0.9, 0.9), (-0.9, 0.9))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.1,
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=_FWIDTH, amplitude=1.0),
            name="pulse",
        )
    )
    # A monitor keeps the solve output non-empty; residual energy is read from
    # the raw solver field tensors, not from this monitor.
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ez",)))
    return scene


def _run_solver(boundary, absorber, steps):
    scene = _build_scene(boundary)
    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQ],
        absorber=absorber,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    prepared = sim.prepare()
    prepared.run()
    return prepared.solver


def _residual_energy(solver):
    # Total EM energy proxy: eps0*|E|^2 + mu0*|H|^2. Including both field kinds
    # avoids the E<->H energy sloshing that a single E snapshot would show.
    e_energy = sum(
        float((getattr(solver, name).double() ** 2).sum().item())
        for name in ("Ex", "Ey", "Ez")
    )
    h_energy = sum(
        float((getattr(solver, name).double() ** 2).sum().item())
        for name in ("Hx", "Hy", "Hz")
    )
    return solver.eps0 * e_energy + solver.mu0 * h_energy


def _assert_finite(solver):
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.isfinite(getattr(solver, name)).all(), f"{name} diverged"


_STEADY_STEPS = 350
_LONG_STEPS = 600


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize(
    "boundary, absorber, max_ratio",
    [
        # Adiabatic absorber and StablePML absorb comparably to standard PML.
        (mw.BoundarySpec.pml(num_layers=5, strength=1.0), "absorber", 0.2),
        (mw.BoundarySpec.pml(num_layers=5, strength=1.0), "stablepml", 0.25),
        # First-order Mur reflects more than a PML, so it gets a looser bound.
        (mw.BoundarySpec.mur(), "cpml", 0.5),
    ],
)
def test_absorbing_boundary_dissipates_and_stays_stable(boundary, absorber, max_ratio):
    # Reference: a reflecting PEC box traps the pulse energy.
    pec_solver = _run_solver(mw.BoundarySpec.pec(), "cpml", _STEADY_STEPS)
    absorbing_solver = _run_solver(boundary, absorber, _STEADY_STEPS)
    long_solver = _run_solver(boundary, absorber, _LONG_STEPS)

    _assert_finite(pec_solver)
    _assert_finite(absorbing_solver)
    _assert_finite(long_solver)

    pec_energy = _residual_energy(pec_solver)
    absorbing_energy = _residual_energy(absorbing_solver)
    long_energy = _residual_energy(long_solver)

    assert pec_energy > 0.0
    assert absorbing_energy > 0.0
    # The absorbing boundary must retain far less energy than the PEC box.
    assert absorbing_energy < max_ratio * pec_energy
    # A longer run must not exhibit late-time growth (no blow-up / instability).
    assert long_energy < 1.5 * absorbing_energy
