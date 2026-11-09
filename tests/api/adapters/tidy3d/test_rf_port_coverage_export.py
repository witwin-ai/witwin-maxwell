"""Reference-adapter export of coaxial ports/resistors, PEC frames, baseband drives and smoothing."""
from __future__ import annotations

import pytest

td = pytest.importorskip("tidy3d")

import witwin.maxwell as mw
from benchmark.scenes.rf.coax_thru import coax_thru_scene
from witwin.maxwell.adapters.tidy3d import (
    _convert_source_time,
    coaxial_lumped_port_to_tidy3d,
    low_frequency_smoothing_to_tidy3d,
    scene_to_tidy3d,
)

A, B = 1.0e-3, 2.3e-3


def _coax_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-3e-3, 3e-3), (-3e-3, 3e-3), (0.0, 18e-3))),
        grid=mw.GridSpec.uniform(0.25e-3),
        boundary=mw.BoundarySpec.pec(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(mw.Cylinder(position=(0, 0, 8.5e-3), radius=A, height=15e-3, axis="z"),
                                     mw.Material.pec(), name="inner"))
    scene.add_port(mw.CoaxialLumpedPort("feed", center=(0, 0, 1e-3), inner_diameter=2 * A, outer_diameter=2 * B,
                                        normal_axis="z", direction="+", reference_impedance=50.0))
    scene.add_lumped_element(mw.CoaxialLumpedResistor("load", center=(0, 0, 16e-3), inner_diameter=2 * A,
                                                      outer_diameter=2 * B, normal_axis="z", resistance=75.0))
    return scene


def test_coaxial_port_maps_to_the_reference_terminal_port():
    port = mw.CoaxialLumpedPort("feed", center=(0, 0, 1e-3), inner_diameter=2 * A, outer_diameter=2 * B,
                                normal_axis="z", direction="-", reference_impedance=60.0)
    reference = coaxial_lumped_port_to_tidy3d(port)
    assert isinstance(reference, td.plugins.smatrix.CoaxialLumpedPort)
    assert reference.center == pytest.approx((0.0, 0.0, 1000.0))
    assert reference.inner_diameter == pytest.approx(2000.0)
    assert reference.outer_diameter == pytest.approx(4600.0)
    assert reference.normal_axis == 2 and reference.direction == "-"
    assert reference.impedance == 60.0


def test_coaxial_scene_exports_drive_monitors_and_annular_resistors():
    sim = scene_to_tidy3d(_coax_scene(), frequencies=(1e9, 2e9))
    assert [type(source).__name__ for source in sim.sources] == ["CustomCurrentSource"]
    assert sim.sources[0].name == "feed"
    assert {monitor.name for monitor in sim.monitors} >= {"feed_voltage", "feed_current"}
    elements = {element.name: element for element in sim.lumped_elements}
    assert set(elements) == {"feed_resistor", "load"}
    assert isinstance(elements["load"], td.CoaxialLumpedResistor)
    assert elements["load"].resistance == 75.0
    assert elements["load"].center == pytest.approx((0.0, 0.0, 16000.0))
    assert elements["feed_resistor"].resistance == 50.0


def test_baseband_drives_map_to_reference_baseband_source_times():
    gaussian = _convert_source_time(mw.BasebandGaussianPulse(40e-12, offset=6.0, amplitude=2.0), td)
    assert isinstance(gaussian, td.BasebandGaussianPulse)
    assert (gaussian.twidth, gaussian.offset, gaussian.amplitude) == (40e-12, 6.0, 2.0)
    rectangle = _convert_source_time(mw.BasebandRectangularPulse(10e-12, 200e-12), td)
    assert isinstance(rectangle, td.BasebandRectangularPulse)
    assert (rectangle.rise_time, rectangle.twidth, rectangle.offset) == (10e-12, 200e-12, 5.0)


def test_pec_frame_exports_as_the_driven_mode_source_frame_and_receiver_walls():
    scene = coax_thru_scene(dx=0.01, device="cpu")
    for port in scene.ports:
        for structure in mw.PECFrame(3).structures(port, scene):
            scene.add_structure(structure)
    sim = scene_to_tidy3d(scene, frequencies=(1e9,))
    drive = next(source for source in sim.sources if isinstance(source, td.ModeSource))
    assert drive.name == "left::drive"
    assert isinstance(drive.frame, td.PECFrame) and drive.frame.length == 3
    walls = [structure for structure in sim.structures
             if isinstance(structure.medium, td.PECMedium) and isinstance(structure.geometry, td.Box)]
    # The receiving port keeps its four zero-thickness PEC walls.
    assert len(walls) == 4
    assert all(sorted(wall.geometry.size)[0] == 0.0 for wall in walls)


def test_low_frequency_smoothing_maps_to_the_reference_spec():
    spec = mw.LowFrequencySmoothingSpec(min_sampling_time=2.0, max_sampling_time=7.0, order=2, max_deviation=0.3)
    reference = low_frequency_smoothing_to_tidy3d(spec, ["port_mode"])
    assert isinstance(reference, td.LowFrequencySmoothingSpec)
    assert reference.monitors == ("port_mode",)
    assert (reference.min_sampling_time, reference.max_sampling_time, reference.order,
            reference.max_deviation) == (2.0, 7.0, 2, 0.3)
