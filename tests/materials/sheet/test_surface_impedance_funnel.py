"""Compiled surface-impedance layout and the remaining fail-closed funnel (S1.1).

The generalized layout (``compile_surface_impedance_layout``) replaces the single-plane
v1 SIBC descriptor: it enumerates every axis-aligned exposed face of every
surface-impedance metal (finite blocks, mid-domain double-sided plates, multiple metals,
multiple orientations), with a deterministic owner order for shared corner edges and
overlap rejection for contradictory owners. The incumbent narrowband good-conductor slab
must still compile as an order-0 (pure resistance) surface. The oblique/curved and Bloch
cases still fail closed through the single re-authored capability funnel, always naming a
physical reason and a phase, never a bare deferral phrase (the P5.5 phrase gate).
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.materials import compile_surface_impedance_layout
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.scene import prepare_scene

_BANNED_DEFERRALS = ("not implemented yet", "not supported yet", "in v1")
_FLUSH_SLAB = dict(position=(0.3, 0.0, 0.0), size=(0.4, 0.4, 0.4))


def _scene(structures, boundary=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=boundary
        or mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
        device="cpu",
        structures=structures,
    )


def _good_conductor_medium(name="coating"):
    frequencies = torch.logspace(9.0, math.log10(40.0e9), 96, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(5.8e7, frequencies)).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(frequencies, admittance, order=8, band=(1.0e9, 40.0e9))
    return SurfaceImpedanceMedium(impedance=model, name=name)


# --- incumbent narrowband slab (order-0) -------------------------------------


def test_incumbent_good_conductor_slab_still_compiles_as_order0():
    scene = prepare_scene(
        _scene([mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7))])
    )
    layout = scene.compile_materials().get("surface_impedance")
    assert layout is not None and bool(layout)
    # Full transverse span, flush against +x: exactly one illuminated face.
    assert len(layout.faces) == 1
    face = layout.faces[0]
    assert face.axis == 0
    assert face.metal_side == "high"
    assert face.full_plane is True
    # A narrowband LossyMetalMedium is realized as an order-0 pure-resistance surface.
    assert layout.metals[face.metal_index].conductivity == pytest.approx(5.8e7)


# --- generic rational surface now compiles -----------------------------------


def test_generic_surface_impedance_medium_now_compiles():
    scene = prepare_scene(
        _scene([mw.Structure(geometry=Box(**_FLUSH_SLAB), material=_good_conductor_medium())])
    )
    layout = scene.compile_materials().get("surface_impedance")
    assert layout is not None and bool(layout)
    assert len(layout.faces) == 1
    # A generic rational surface has no narrowband conductivity (uses the ADE path).
    assert layout.metals[layout.faces[0].metal_index].conductivity is None


# --- finite block: all six exposed faces + area sums -------------------------


def test_finite_block_exposes_six_faces_with_area_sums():
    block = Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
    scene = prepare_scene(_scene([mw.Structure(geometry=block, material=mw.LossyMetalMedium(conductivity=5.8e7))]))
    layout = compile_surface_impedance_layout(scene)
    assert len(layout.faces) == 6
    # Every face is a 0.2 x 0.2 square; none spans the full cross-section.
    for face in layout.faces:
        assert face.area == pytest.approx(0.04)
        assert face.full_plane is False
    # The two faces normal to each axis are present.
    axes = sorted(face.axis for face in layout.faces)
    assert axes == [0, 0, 1, 1, 2, 2]
    assert layout.total_area == pytest.approx(sum(f.area for f in layout.faces))
    assert layout.total_area == pytest.approx(0.24)


def test_mid_domain_double_sided_plate_exposes_two_faces():
    plate = Box(position=(0.0, 0.0, 0.0), size=(0.2, 1.0, 1.0))  # bounded along x only
    scene = prepare_scene(_scene([mw.Structure(geometry=plate, material=mw.LossyMetalMedium(conductivity=5.8e7))]))
    layout = compile_surface_impedance_layout(scene)
    assert len(layout.faces) == 2
    sides = sorted(face.metal_side for face in layout.faces)
    assert sides == ["high", "low"]
    for face in layout.faces:
        assert face.axis == 0
        assert face.full_plane is True


# --- multiple metals and multiple orientations -------------------------------


def test_multiple_metals_and_orientations_enumerate_all_faces():
    x_plate = Box(position=(0.0, 0.0, 0.0), size=(0.2, 1.0, 1.0))  # normal x
    y_plate = Box(position=(0.0, 0.0, 0.0), size=(1.0, 0.1, 1.0))  # normal y (disjoint interface)
    scene = prepare_scene(
        _scene(
            [
                mw.Structure(geometry=x_plate, material=mw.LossyMetalMedium(conductivity=5.8e7)),
                mw.Structure(geometry=y_plate, material=mw.LossyMetalMedium(conductivity=1.0e6)),
            ]
        )
    )
    layout = compile_surface_impedance_layout(scene)
    assert len(layout.metals) == 2
    axes = {face.axis for face in layout.faces}
    assert 0 in axes and 1 in axes  # both orientations represented


# --- deterministic owner ordering --------------------------------------------


def test_face_owner_order_is_deterministic_and_sorted():
    block = Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
    scene = prepare_scene(_scene([mw.Structure(geometry=block, material=mw.LossyMetalMedium(conductivity=5.8e7))]))
    first = compile_surface_impedance_layout(scene)
    second = compile_surface_impedance_layout(scene)
    ranks_first = [face.owner_rank for face in first.faces]
    ranks_second = [face.owner_rank for face in second.faces]
    # Deterministic across recompiles and monotone non-increasing (min-rank owner last).
    assert ranks_first == ranks_second
    assert ranks_first == sorted(ranks_first, reverse=True)


# --- overlap rejection (contradictory owners) --------------------------------


def test_overlapping_pec_and_surface_ownership_fails_closed():
    scene = prepare_scene(
        _scene(
            [
                mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7)),
                mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.Material.pec()),
            ]
        )
    )
    with pytest.raises(NotImplementedError, match="two contradictory owners"):
        scene.compile_materials()


def test_non_overlapping_pec_does_not_trip_the_owner_guard():
    """Falsification: the overlap guard must not fire on a disjoint PEC structure."""
    scene = prepare_scene(
        _scene(
            [
                mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7)),
                mw.Structure(
                    geometry=Box(position=(-0.3, 0.0, 0.0), size=(0.1, 0.1, 0.1)), material=mw.Material.pec()
                ),
            ]
        )
    )
    assert bool(scene.compile_materials().get("surface_impedance"))


def test_two_different_surface_materials_on_one_interface_fail_closed():
    scene = prepare_scene(
        _scene(
            [
                mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7)),
                mw.Structure(geometry=Box(**_FLUSH_SLAB), material=_good_conductor_medium(name="other")),
            ]
        )
    )
    with pytest.raises(NotImplementedError, match="contradictory owners"):
        scene.compile_materials()


# --- remaining fail-closed cases name a phase, avoid deferral phrases ---------


def test_rotated_surface_still_fails_closed_with_a_phase():
    rotated = Box(position=(0.3, 0.0, 0.0), size=(0.4, 0.4, 0.4), rotation=(0.9238795, 0.0, 0.0, 0.3826834))
    scene = prepare_scene(_scene([mw.Structure(geometry=rotated, material=mw.LossyMetalMedium(conductivity=5.8e7))]))
    with pytest.raises(NotImplementedError) as info:
        scene.compile_materials()
    message = str(info.value)
    assert "Phase" in message
    assert not any(phrase in message.lower() for phrase in _BANNED_DEFERRALS)


def test_bloch_surface_still_fails_closed_with_a_phase():
    scene = prepare_scene(
        _scene(
            [mw.Structure(geometry=Box(**_FLUSH_SLAB), material=mw.LossyMetalMedium(conductivity=5.8e7))],
            boundary=mw.BoundarySpec.faces(
                default="periodic", num_layers=4, strength=1.0, x=("pml", "pml"),
                y=("bloch", "bloch"), bloch_wavevector=(0.0, 10.0, 0.0),
            ),
        )
    )
    with pytest.raises(NotImplementedError) as info:
        scene.compile_materials()
    message = str(info.value)
    assert "Phase" in message
    assert not any(phrase in message.lower() for phrase in _BANNED_DEFERRALS)
