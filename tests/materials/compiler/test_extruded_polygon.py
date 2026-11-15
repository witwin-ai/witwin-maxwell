"""ExtrudedPolygon (witwin.core, published as mw.geometry.ExtrudedPolygon) on the generic FDTD SDF/occupancy route.

A regular-pentagon ExtrudedPolygon compiles to the Prism's material tensors bit for bit except inside Prism's tolerance band; a non-convex L
compiles within float32 rounding of the same L as a PolySlab; a short CUDA FDTD run of the L is bitwise
the PolySlab run and repeatable.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import core_path
from witwin.maxwell.scene import prepare_scene

L_SHAPE = [(-0.3, -0.3), (0.3, -0.3), (0.3, 0.0), (0.0, 0.0), (0.0, 0.3), (-0.3, 0.3)]


def _scene(geometry):
    return mw.Scene(domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))), grid=mw.GridSpec.uniform(0.1),
                    structures=(mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0)),))


def _pentagon():
    angles = torch.linspace(0, 2 * math.pi, 6)[:-1]
    return torch.stack([0.4 * torch.cos(angles), 0.4 * torch.sin(angles)], 1)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_regular_extruded_polygon_compiles_like_prism(axis):
    """Bitwise equal except where Prism's absolute 1e-7 convex inside tolerance flips the sign of a near-zero
    distance (one sample on the x-axis frame, 3.8e-6 in eps); ExtrudedPolygon uses the exact even-odd sign."""
    frame = dict(position=(0.05, -0.1, 0.02), rotation=(0.2, -0.1, 0.3))
    prism = prepare_scene(_scene(mw.geometry.Prism(radius=0.4, height=0.6, num_sides=5, axis=axis, **frame)), device="cpu")
    extruded = prepare_scene(_scene(mw.geometry.ExtrudedPolygon(_pentagon(), height=0.6, axis=axis, **frame)), device="cpu")
    difference = (prism.permittivity - extruded.permittivity).abs()
    assert difference.max() <= 1e-5 and int((difference > 0).sum()) <= 1
    if axis != "x":
        assert torch.equal(prism.permittivity, extruded.permittivity)


def test_nonconvex_extruded_polygon_matches_polyslab():
    extruded = prepare_scene(_scene(mw.geometry.ExtrudedPolygon(L_SHAPE, height=0.4, position=(0.0, 0.0, 0.1))), device="cpu")
    slab = prepare_scene(_scene(mw.PolySlab(L_SHAPE, bounds=(-0.1, 0.3), axis="z")), device="cpu")
    assert torch.allclose(extruded.permittivity, slab.permittivity, rtol=0, atol=1e-5)
    assert extruded.permittivity.max() > 3.9


def _fdtd_simulation(geometry):
    """CUDA-configured FDTD request on CPU tensors: admission reads the scene and config only (no CUDA init)."""
    return mw.Simulation.fdtd(_scene(geometry), frequencies=1e9,
                              config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=8), device="cuda"))


def test_fdtd_core_path_admits_extruded_polygon_like_polyslab(monkeypatch):
    """Admission: a detached ExtrudedPolygon is a plain SDF/occupancy structure, so the automatic CUDA core-path
    decision admits it exactly as it admits the same L as a PolySlab (no geometry-specific blocker)."""
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("admission must not initialize CUDA"))
    extruded = _fdtd_simulation(mw.geometry.ExtrudedPolygon(L_SHAPE, height=0.4, position=(0.0, 0.0, 0.1)))
    slab = _fdtd_simulation(mw.PolySlab(L_SHAPE, bounds=(-0.1, 0.3), axis="z"))
    # There is no path decision or blocker left (W7 rows 16c/16d): both prepare on the core path.
    assert extruded.scene.structures and slab.scene.structures
    assert not hasattr(core_path, "core_path_decision")


def test_fdtd_core_path_admits_trainable_extruded_polygon(monkeypatch):
    """Admission: trainable ExtrudedPolygon vertices keep the core path, as trainable PolySlab vertices do;
    differentiable runs take the core forward (W7 row 14)."""
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("admission must not initialize CUDA"))
    vertices = torch.tensor(L_SHAPE, dtype=torch.float64, requires_grad=True)
    extruded = _fdtd_simulation(mw.geometry.ExtrudedPolygon(vertices, height=0.4, position=(0.0, 0.0, 0.1)))
    slab = _fdtd_simulation(mw.PolySlab(torch.tensor(L_SHAPE, dtype=torch.float64, requires_grad=True),
                                        bounds=(-0.1, 0.3), axis="z"))
    # There is no path decision or blocker left (W7 rows 16c/16d): both prepare on the core path.
    assert extruded.scene.structures and slab.scene.structures
    assert not hasattr(core_path, "core_path_decision")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_fdtd_run_is_bitwise_polyslab_and_repeatable():
    """A non-convex L ExtrudedPolygon runs bitwise like the same L as a PolySlab and repeats bitwise.

    (Against Prism a regular N-gon is Rule-2 T1, not bitwise: Prism's absolute 1e-7 inside tolerance flips
    near-zero samples; probe scripts/dev/rfic/extruded_polygon_fdtd_g1_r7.py.)"""
    def run(geometry):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.4, 0.4),) * 3), grid=mw.GridSpec.uniform(0.025),
                         boundary=mw.BoundarySpec.pml(num_layers=8),
                         structures=(mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0)),),
                         sources=(
                             mw.PointDipole(position=(0.0, 0.0, -0.25), polarization="Ez",
                                            source_time=mw.GaussianPulse(frequency=2e9, fwidth=0.5e9)),
                         ))
        result = mw.Simulation.fdtd(scene, frequencies=(2e9,),
                                    config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=200),
                                                         full_field_dft=True)).run()
        # Admission on the real run: both geometries take the CUDA core path.
        assert result.solver._core_runtime is not None
        return {name: field.values for name, field in result.fields.items()}
    shape = [(x * 0.5, y * 0.5) for x, y in L_SHAPE]
    slab = run(mw.PolySlab(shape, bounds=(-0.1, 0.1), axis="z", device="cuda"))
    extruded = run(mw.geometry.ExtrudedPolygon(shape, height=0.2, device="cuda"))
    again = run(mw.geometry.ExtrudedPolygon(shape, height=0.2, device="cuda"))
    assert slab.keys() == extruded.keys() and slab
    for key in slab:
        assert torch.equal(slab[key], extruded[key]), key
        assert torch.equal(extruded[key], again[key]), key
