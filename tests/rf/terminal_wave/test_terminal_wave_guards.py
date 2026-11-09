"""Explicit rejections of the TerminalWavePort capability boundary."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.terminal_waveport import resolve_terminal_waveport_run_manifest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="terminal mode solves run on CUDA")

DX = 0.005


def _scene(substrate=None, *, extra_ports=()):
    x = np.arange(-4, 5) * DX
    y = np.arange(-12, 13) * DX
    z = np.arange(-1, 14) * DX
    scene = mw.Scene(domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
                     grid=mw.GridSpec.custom(x, y, z), boundary=mw.BoundarySpec.pec(), device="cuda")
    scene.add_structure(mw.Structure(mw.PlanarPEC(((-1, -1), (1, -1), (1, 1), (-1, 1)), normal_axis="z", coordinate=0.0),
                                     mw.Material.pec(), name="ground"))
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, 0.01), size=(1, 1, 0.02)),
                                     substrate or mw.Material(eps_r=3.0), name="substrate"))
    scene.add_structure(mw.Structure(mw.PlanarPEC(((-1, -0.01), (1, -0.01), (1, 0.01), (-1, 0.01)), normal_axis="z",
                                                  coordinate=0.02), mw.Material.pec(), name="strip"))
    scene.add_port(mw.TerminalWavePort("P", position=(0, 0, 0.03), size=(0, 0.11, 0.06), direction="+",
                                       reference_plane=0.0, terminals=(mw.TerminalRef("strip"),),
                                       reference=mw.TerminalRef("ground")))
    for port in extra_ports:
        scene.add_port(port)
    return scene


def _manifest(scene):
    return resolve_terminal_waveport_run_manifest(prepare_scene(scene), mw.PortSweep(), (1.0e9,))


def test_lossy_cross_section_is_rejected():
    with pytest.raises(NotImplementedError, match="require a lossless cross-section"):
        _manifest(_scene(mw.Material(eps_r=3.0, sigma_e=0.05)))


def test_magnetic_cross_section_is_rejected():
    with pytest.raises(NotImplementedError, match="require a nonmagnetic cross-section"):
        _manifest(_scene(mw.Material(eps_r=3.0, mu_r=2.0)))


def test_mixing_with_single_mode_waveports_is_rejected():
    mode = mw.WaveModeSpec("te", polarization="Ey")
    wave = mw.WavePort("W", position=(0.01, 0, 0.03), size=(0, 0.11, 0.06), direction="-", reference_plane=0.01,
                       modes=(mode,))
    with pytest.raises(NotImplementedError, match="cannot yet be combined with single-mode WavePorts"):
        _manifest(_scene(extra_ports=(wave,)))


def test_direct_port_excitation_is_rejected():
    simulation = mw.Simulation.fdtd(_scene(), frequencies=(1.0e9,), excitations=mw.PortExcitation("P"))
    with pytest.raises(NotImplementedError, match="driven through PortSweep columns only"):
        simulation.run()


def test_trainable_scene_is_rejected_by_the_adjoint_gate():
    eps = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    simulation = mw.Simulation.fdtd(_scene(mw.Material(eps_r=eps)), frequencies=(1.0e9,), excitations=mw.PortSweep())
    with pytest.raises(NotImplementedError, match="TerminalWavePort sweeps are forward-only"):
        simulation.run()
