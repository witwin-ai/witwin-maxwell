"""Coordinate helpers prepare a declarative Scene on the device its geometry lives on.

Regression for the p7a rewrite that prepared these scenes on the host: a full prepare
runs the PML-contact census (geometry.signed_distance on the grid), so CUDA-resident
geometry touching the PML failed with a device mismatch. Each helper now takes the
device explicitly; on CUDA geometry it must reproduce the CPU-geometry baseline exactly.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import terminal_waveport, waveport_sweep
from witwin.maxwell.compiler.terminal_waveports import compile_terminal_waveport
from witwin.maxwell.compiler.waveports import compile_waveport_cross_section
from witwin.maxwell.fdtd.coords import centered_cell_coords, component_coords
from witwin.maxwell.fdtd.excitation.spatial import (
    physical_interior_indices,
    soft_plane_wave_coordinate,
    soft_plane_wave_index,
    soft_plane_wave_region_spacing,
    source_plane_index,
)
from witwin.maxwell.scene import place_scene, prepare_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA-resident geometry")

CUDA = "cuda:0"
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _box(position, size):
    return mw.Box(position=torch.tensor(position, dtype=torch.float64), size=torch.tensor(size, dtype=torch.float64))


def _scene():
    """Microstrip-like guide whose conductors and substrate end on the PML interfaces (extruded by the census)."""
    wave = mw.rf.WavePort("wave", position=(-0.04, 0.0, 0.0), size=(0.0, 0.12, 0.12), direction="+",
                          reference_plane=-0.04, modes=(mw.rf.WaveModeSpec("te", polarization="Ey"),))
    terminal = mw.rf.TerminalWavePort("terminal", position=(0.04, 0.0, 0.0), size=(0.0, 0.16, 0.16),
                                      direction="-", reference_plane=0.04,
                                      terminals=(mw.rf.TerminalRef("strip"),), reference=mw.rf.TerminalRef("ground"))
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1),) * 3),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        structures=(
            mw.Structure(_box((0.0, 0.0, -0.065), (0.2, 0.2, 0.01)), mw.Material.pec(), name="ground"),
            mw.Structure(_box((0.0, 0.0, -0.03), (0.2, 0.2, 0.06)), mw.Material(eps_r=2.2), name="substrate"),
            mw.Structure(_box((0.0, 0.0, 0.005), (0.2, 0.04, 0.01)), mw.Material.pec(), name="strip"),
        ),
        ports=(wave, terminal),
    )


@pytest.fixture(scope="module")
def scenes():
    host = _scene()
    device = place_scene(host, CUDA)
    assert all(s.geometry.position.device.type == "cuda" for s in device.structures)
    # The PML census reads the CUDA geometry: every structure crosses the x PML.
    assert prepare_scene(device, device=CUDA).pml_extrusions
    return {"cpu": host, CUDA: device}


def _equal_arrays(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_fdtd_coords_on_cuda_geometry_equal_the_host_baseline(scenes):
    for component in COMPONENTS:
        _equal_arrays(component_coords(scenes[CUDA], component, device=CUDA),
                      component_coords(scenes["cpu"], component, device="cpu"))
    _equal_arrays(centered_cell_coords(scenes[CUDA], device=CUDA), centered_cell_coords(scenes["cpu"], device="cpu"))


def test_spatial_excitation_helpers_on_cuda_geometry_equal_the_host_baseline(scenes):
    def evaluate(device):
        scene = scenes[device]
        values = {}
        for axis in "xyz":
            for sign in (1.0, -1.0):
                values["plane", axis, sign] = source_plane_index(scene, axis, sign, device=device)
                index = soft_plane_wave_index(scene, axis, sign, device=device)
                values["soft", axis, sign] = index
                values["coordinate", axis, sign] = soft_plane_wave_coordinate(scene, axis, sign, device=device)
                values["spacing", axis, sign] = soft_plane_wave_region_spacing(
                    scene, injection_axis=axis, plane_index=index, direction_sign=int(sign), device=device)
            values["interior", axis] = physical_interior_indices(scene, axis, device=device)
        return values

    assert evaluate(CUDA) == evaluate("cpu")


def test_pec_frame_on_cuda_geometry_equals_the_host_baseline(scenes):
    def frame(device):
        scene = scenes[device]
        return mw.rf.PECFrame(2).structures(scene.ports[0], scene, device=device)

    device_walls, host_walls = frame(CUDA), frame("cpu")
    assert [s.name for s in device_walls] == [s.name for s in host_walls] and len(host_walls) == 4
    for a, b in zip(device_walls, host_walls):
        assert a.geometry.normal_axis == b.geometry.normal_axis
        assert a.geometry.coordinate == b.geometry.coordinate
        assert torch.equal(a.geometry.vertices.cpu(), b.geometry.vertices.cpu())


def test_waveport_source_plane_on_cuda_geometry_equals_the_host_baseline(scenes):
    def column(device):
        scene = scenes[device]
        port = scene.ports[0]
        compiled = compile_waveport_cross_section(scene, port, device=device)
        mode = SimpleNamespace(mode_index=0, polarization_axis="y", family="te", conductor_model=None)
        prepared = SimpleNamespace(port=port, compiled=SimpleNamespace(modes=(mode,),
                                                                       electric_plane_index=compiled.electric_plane_index),
                                   tracking=SimpleNamespace(orientation=torch.ones((1, 1, 1), dtype=torch.complex128)))
        manifest = SimpleNamespace(frequencies=(1e9,), prepared_ports=(prepared,))
        plane = waveport_sweep._source_plane(scene, prepared, device=device)
        built = waveport_sweep._column_scene(scene, manifest, active_port_index=0, active_mode_index=0,
                                             frequency_index=0, amplitude=1.0, device=device)
        return compiled.electric_plane_index, plane, [source.position for source in built.sources]

    device_column, host_column = column(CUDA), column("cpu")
    assert device_column == host_column and len(host_column[2]) == 1


def test_terminal_waveport_column_on_cuda_geometry_equals_the_host_baseline(scenes):
    def column(device):
        scene = scenes[device]
        port = scene.ports[1]
        compiled = compile_terminal_waveport(scene, port, device=device)
        prepared = SimpleNamespace(port=port, compiled=compiled, polarization_axis="z",
                                   modal_potentials=torch.ones((1, 1), dtype=torch.float64),
                                   drive=torch.ones((1, 1, 1), dtype=torch.complex128))
        manifest = SimpleNamespace(frequencies=(1e9,), prepared_ports=(prepared,))
        built = terminal_waveport._column_scene(scene, manifest, active_port=0, active_channel=0,
                                                frequency_index=0, amplitude=1.0, device=device)
        return compiled.electric_plane_index, [source.position for source in built.sources]

    device_column, host_column = column(CUDA), column("cpu")
    assert device_column == host_column and len(host_column[1]) == 1
