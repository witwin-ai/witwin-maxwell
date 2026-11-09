"""Tidy3D export of projection, Gaussian-overlap and time-trace monitors."""

from __future__ import annotations

import math

import pytest

import witwin.maxwell as mw

td = pytest.importorskip("tidy3d")

from witwin.maxwell.adapters.tidy3d import scene_to_tidy3d  # noqa: E402

_S = 1.0e6  # m -> um


def _scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cpu",
    )
    scene.add_source(mw.PointDipole(
        position=(0, 0, 0), polarization="Ez", width=0.05,
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9), name="src"))
    return scene


def _by_name(simulation):
    return {monitor.name: monitor for monitor in simulation.monitors}


def test_projection_monitors_export_as_single_reference_projection_monitors():
    scene = _scene()
    scene.add_monitor(mw.AngleProjectionMonitor(
        "angle", size=(0.6, 0.6, 0.6), theta=(0.1, 0.5), phi=(0.0, 1.0), radius=5.0,
        exclude_surfaces=("z_neg",), origin=(0.0, 0.0, -0.1)))
    scene.add_monitor(mw.CartesianProjectionMonitor(
        "cart", position=(0, 0, 0.2), size=(0.6, 0.6, 0.0), x=(-0.5, 0.5), y=(0.0,), axis="y",
        distance=2.0, window_size=(0.2, 0.1)))
    scene.add_monitor(mw.KSpaceProjectionMonitor("ks", size=(0.6, 0.6, 0.6), ux=(0.0, 0.5), uy=(0.0,), axis="x"))
    scene.add_monitor(mw.ClosedSurfaceMonitor.box("huygens", size=(0.3, 0.3, 0.3)))
    monitors = _by_name(scene_to_tidy3d(scene, frequencies=[1e9]))

    # Projection faces export as their parent, ClosedSurfaceMonitor faces stay faces.
    assert not any(name.startswith(("angle::", "cart::", "ks::")) for name in monitors)
    assert "huygens::x_neg" in monitors

    angle = monitors["angle"]
    assert isinstance(angle, td.FieldProjectionAngleMonitor)
    assert tuple(angle.size) == pytest.approx((0.6 * _S,) * 3)
    assert tuple(angle.theta) == (0.1, 0.5) and tuple(angle.phi) == (0.0, 1.0)
    assert angle.proj_distance == pytest.approx(5.0 * _S)
    assert angle.far_field_approx is True
    assert tuple(angle.exclude_surfaces) == ("z-",)
    assert tuple(angle.custom_origin) == pytest.approx((0.0, 0.0, -0.1 * _S))

    cart = monitors["cart"]
    assert isinstance(cart, td.FieldProjectionCartesianMonitor)
    assert cart.proj_axis == 1 and cart.proj_distance == pytest.approx(2.0 * _S)
    assert tuple(cart.x) == pytest.approx((-0.5 * _S, 0.5 * _S))
    assert cart.normal_dir == "+" and tuple(cart.window_size) == (0.2, 0.1)
    assert cart.far_field_approx is False

    ks = monitors["ks"]
    assert isinstance(ks, td.FieldProjectionKSpaceMonitor)
    assert ks.proj_axis == 0 and tuple(ks.ux) == (0.0, 0.5)


def test_kspace_export_rejects_a_hemisphere_without_reference_equivalent():
    scene = _scene()
    scene.add_monitor(mw.KSpaceProjectionMonitor(
        "ks", size=(0.6, 0.6, 0.6), ux=(0.0,), uy=(0.0,), axis="z", hemisphere="-"))
    with pytest.raises(NotImplementedError, match="hemisphere"):
        scene_to_tidy3d(scene, frequencies=[1e9])


def test_gaussian_overlap_export_matches_the_exported_gaussian_beam_parameters():
    direction = (math.sin(0.2), 0.0, math.cos(0.2))
    polarization = (math.cos(0.2), 0.0, -math.sin(0.2))
    scene = _scene()
    scene.sources.clear()
    scene.add_source(mw.GaussianBeam(
        direction=direction, polarization=polarization, beam_waist=0.2, focus=(0.0, 0.0, 0.1),
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9), name="beam"))
    scene.add_monitor(mw.GaussianOverlapMonitor(
        "overlap", position=(0.0, 0.0, 0.3), size=(1.0, 1.0, 0.0), beam_waist=0.2,
        direction=direction, polarization=polarization, focus=(0.0, 0.0, 0.1)))
    scene.add_monitor(mw.AstigmaticGaussianOverlapMonitor(
        "astigmatic", position=(0.0, 0.0, 0.3), size=(1.0, 1.0, 0.0), beam_waist=(0.2, 0.3),
        focus=(0.0, 0.0, 0.1), focus_u=0.05, focus_v=-0.05))
    simulation = scene_to_tidy3d(scene, frequencies=[1e9])
    monitors = _by_name(simulation)
    beam = next(source for source in simulation.sources if isinstance(source, td.GaussianBeam))

    overlap = monitors["overlap"]
    assert isinstance(overlap, td.GaussianOverlapMonitor)
    assert [monitor.name for monitor in simulation.monitors].count("overlap") == 1
    assert overlap.angle_theta == pytest.approx(beam.angle_theta)
    assert overlap.angle_phi == pytest.approx(beam.angle_phi)
    assert overlap.pol_angle == pytest.approx(beam.pol_angle)
    assert overlap.waist_radius == pytest.approx(beam.waist_radius)
    assert overlap.waist_distance == pytest.approx((0.3 - 0.1) / math.cos(0.2) * _S)

    astigmatic = monitors["astigmatic"]
    assert isinstance(astigmatic, td.AstigmaticGaussianOverlapMonitor)
    assert tuple(astigmatic.waist_sizes) == pytest.approx((0.2 * _S, 0.3 * _S))
    assert tuple(astigmatic.waist_distances) == pytest.approx((0.15 * _S, 0.25 * _S))


def test_gaussian_overlap_export_rejects_backward_normal_beams():
    scene = _scene()
    scene.add_monitor(mw.GaussianOverlapMonitor(
        "overlap", position=(0.0, 0.0, 0.3), size=(1.0, 1.0, 0.0), beam_waist=0.2, direction=(0, 0, -1)))
    with pytest.raises(NotImplementedError, match=r"backward \('-'\) amplitude, so export a \+normal beam"):
        scene_to_tidy3d(scene, frequencies=[1e9])


def test_time_trace_monitor_export():
    scene = _scene()
    scene.add_monitor(mw.SurfaceFieldTimeMonitor(
        "surface", size=(0.4, 0.4, 0.4), components=("Ez", "Hx"), start=10, stop=90, interval=5,
        exclude_surfaces=("z_neg", "z_pos")))
    simulation = scene_to_tidy3d(scene, frequencies=[1e9])
    monitors = _by_name(simulation)
    faces = {name: monitor for name, monitor in monitors.items() if name.startswith("surface_")}
    assert set(faces) == {"surface_x_neg", "surface_x_pos", "surface_y_neg", "surface_y_pos"}
    face = faces["surface_x_pos"]
    assert isinstance(face, td.FieldTimeMonitor)
    assert tuple(face.center) == pytest.approx((0.2 * _S, 0.0, 0.0))
    assert tuple(face.size) == pytest.approx((0.0, 0.4 * _S, 0.4 * _S))
    assert tuple(face.fields) == ("Ez", "Hx") and face.interval == 5
    assert face.stop / face.start == pytest.approx(9.0)

    scene.add_monitor(mw.AuxFieldTimeMonitor("aux"))
    with pytest.raises(NotImplementedError, match="no equivalent"):
        scene_to_tidy3d(scene, frequencies=[1e9])
