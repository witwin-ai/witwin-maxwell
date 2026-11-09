"""Export of one-way absorbing boundaries to the Tidy3D reference adapter."""
from __future__ import annotations

import math

import pytest

import witwin.maxwell as mw

td = pytest.importorskip("tidy3d")

_PLANE = (0.0, math.inf, math.inf)


def _export(boundary, absorbers=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=boundary,
        device="cpu",
        internal_absorbers=absorbers,
    )
    return scene.to_tidy3d(frequencies=[2e9])


def test_mur_faces_export_as_abc_with_explicit_permittivity():
    simulation = _export(mw.BoundarySpec.mur())
    for axis in "xyz":
        for side in ("minus", "plus"):
            face = getattr(getattr(simulation.boundary_spec, axis), side)
            assert isinstance(face, td.ABCBoundary)
            # maxwell's default one-way speed is vacuum; the export pins it.
            assert face.permittivity == 1.0
    mixed = _export(mw.BoundarySpec.faces(default="pml", num_layers=8, z_high="mur", mur_abc=mw.MurABC(permittivity=2.5)))
    assert isinstance(mixed.boundary_spec.z.plus, td.ABCBoundary)
    assert mixed.boundary_spec.z.plus.permittivity == 2.5
    assert isinstance(mixed.boundary_spec.z.minus, td.PML)


def test_second_order_mur_export_is_rejected():
    with pytest.raises(ValueError, match="Second-order Mur"):
        _export(mw.BoundarySpec.mur(order=2))
    with pytest.raises(ValueError, match="Second-order Mur"):
        _export(mw.BoundarySpec.pml(num_layers=8),
                absorbers=(mw.InternalAbsorber((0.0, 0, 0), _PLANE, boundary=mw.MurABC(order=2)),))


def test_mode_abc_face_exports_mode_plane_and_frequency():
    simulation = _export(
        mw.BoundarySpec.faces(default="pml", num_layers=8, x_high="mode_abc",
                              mode_abc=mw.ModeABC(mode_index=1, polarization="Ez", frequency=2e9))
    )
    face = simulation.boundary_spec.x.plus
    assert isinstance(face, td.ModeABCBoundary)
    assert face.mode_index == 1 and face.freq_spec == 2e9
    assert face.plane.center == pytest.approx((0.3e6, 0.0, 0.0))
    assert face.plane.size == pytest.approx((0.0, 0.4e6, 0.4e6))
    broadband = _export(
        mw.BoundarySpec.faces(default="pml", num_layers=8, y_low="mode_abc",
                              mode_abc=mw.ModeABC(frequency_range=(1e9, 3e9), order=2))
    ).boundary_spec.y.minus
    assert isinstance(broadband.freq_spec, td.BroadbandModeABCSpec)
    assert broadband.freq_spec.frequency_range == (1e9, 3e9)


def test_internal_absorbers_export():
    simulation = _export(
        mw.BoundarySpec.pml(num_layers=8),
        absorbers=(
            mw.InternalAbsorber((0.1, 0.0, 0.0), _PLANE, "+", grid_shift=1),
            mw.InternalAbsorber((-0.1, 0.0, 0.0), _PLANE, "-", boundary=mw.ModeABC(frequency=2e9)),
        ),
    )
    first, second = simulation.internal_absorbers
    assert first.direction == "+" and first.grid_shift == 1
    assert first.center == pytest.approx((0.1e6, 0.0, 0.0))
    assert isinstance(first.boundary_spec, td.ABCBoundary) and first.boundary_spec.permittivity == 1.0
    assert second.direction == "-" and isinstance(second.boundary_spec, td.ModeABCBoundary)
