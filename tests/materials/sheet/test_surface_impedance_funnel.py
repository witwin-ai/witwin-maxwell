"""Phase 0 re-authored fail-closed funnel for the surface-impedance boundary (S0.3).

The incumbent narrowband good-conductor slab must still compile unchanged, while the
generalized rational surface and overlapping surface ownership fail closed through the
single re-authored capability funnel. Every rejection must state a physical reason and
name the phase that lifts it, never a bare deferral phrase (the P5.5 phrase gate).
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.scene import prepare_scene

_BANNED_DEFERRALS = ("not implemented yet", "not supported yet", "in v1")
_FLUSH_SLAB = dict(position=(0.3, 0.0, 0.0), size=(0.4, 0.4, 0.4))


def _scene(structures):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
        device="cpu",
        structures=structures,
    )


def _good_conductor_medium():
    frequencies = torch.logspace(9.0, math.log10(40.0e9), 96, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(5.8e7, frequencies)).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(frequencies, admittance, order=10, band=(1.0e9, 40.0e9))
    return SurfaceImpedanceMedium(impedance=model, name="coating")


def test_incumbent_good_conductor_slab_still_compiles():
    scene = prepare_scene(
        _scene([mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7))])
    )
    descriptor = scene.compile_materials().get("sibc")
    assert descriptor is not None
    assert descriptor["axis"] == 0
    assert descriptor["metal_side"] == "high"
    assert descriptor["conductivity"] == pytest.approx(5.8e7)


def test_generic_surface_impedance_medium_fails_closed():
    scene = prepare_scene(
        _scene([mw.Structure(geometry=Box(**_FLUSH_SLAB), material=_good_conductor_medium())])
    )
    with pytest.raises(NotImplementedError) as info:
        scene.compile_materials()
    message = str(info.value)
    assert "SurfaceImpedanceMedium" in message
    assert "Phase" in message
    assert not any(phrase in message.lower() for phrase in _BANNED_DEFERRALS)


def test_overlapping_pec_and_surface_ownership_fails_closed():
    scene = prepare_scene(
        _scene([
            mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7)),
            mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.Material.pec()),
        ])
    )
    with pytest.raises(NotImplementedError, match="two contradictory owners"):
        scene.compile_materials()


def test_non_overlapping_pec_does_not_trip_the_owner_guard():
    """Falsification: the overlap guard must not fire on a disjoint PEC structure."""
    scene = prepare_scene(
        _scene([
            mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7)),
            mw.Structure(geometry=Box(position=(-0.3, 0.0, 0.0), size=(0.1, 0.1, 0.1)), material=mw.Material.pec()),
        ])
    )
    assert scene.compile_materials().get("sibc") is not None


def test_every_surface_rejection_names_a_phase_and_avoids_deferral_phrases():
    cases = [
        # laterally finite block
        [mw.Structure(geometry=Box(position=(0.3, 0.0, 0.0), size=(0.4, 0.2, 0.2)),
                      material=mw.LossyMetalMedium(conductivity=5.8e7))],
        # mid-domain double-sided plate
        [mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.4, 0.4)),
                      material=mw.LossyMetalMedium(conductivity=5.8e7))],
    ]
    for structures in cases:
        scene = prepare_scene(
            mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
                grid=mw.GridSpec.uniform(0.05),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                device="cpu",
                structures=structures,
            )
        )
        with pytest.raises(NotImplementedError) as info:
            scene.compile_materials()
        message = str(info.value)
        assert "Phase" in message
        assert not any(phrase in message.lower() for phrase in _BANNED_DEFERRALS)
