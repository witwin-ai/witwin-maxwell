"""Small CUDA FDTD checks of the ModalPortSweep modal S-matrix.

Units are scaled so the free-space wavelength at 300 MHz is ~1 m; the n=2 core
is snapped to the 0.05 m grid (grid-aligned edges keep the mode-solver
propagation constant consistent with the time-stepped guide). Each scene has
about 0.2 M cells and runs in a few seconds per column.
"""

from __future__ import annotations

import cmath

import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_F0 = 3.0e8
_FREQUENCIES = (0.95 * _F0, _F0, 1.05 * _F0)
_PORT_SPACING = 1.2


def _base_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-1.4, 1.4), (-0.9, 0.9), (-0.9, 0.9))),
        grid=mw.GridSpec.uniform(0.05),
        # Keep the grid-aligned arithmetic interface used to qualify this coarse modal phase gate.
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="arithmetic"),
        boundary=mw.BoundarySpec.pml(num_layers=10),
        device="cuda",
    )


def _add_ports(scene):
    half = 0.5 * _PORT_SPACING
    scene.add_port(
        mw.ModePort("left", position=(-half, 0.0, 0.0), size=(0.0, 1.2, 1.0), direction="+",
                    polarization="Ey", monitor_offset=0.1)
    )
    scene.add_port(
        mw.ModePort("right", position=(half, 0.0, 0.0), size=(0.0, 1.2, 1.0), direction="-",
                    polarization="Ey", monitor_offset=-0.1)
    )
    return scene


def _simulation(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=_FREQUENCIES,
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=30),
        full_field_dft=False,
    )


def _straight_guide():
    scene = _base_scene()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4.0, 0.5, 0.35)),
            material=mw.Material(eps_r=4.0),
            name="core",
        )
    )
    return _add_ports(scene)


def test_straight_waveguide_two_port_transmits_with_beta_length_phase():
    result = mw.run_modal_port_sweep(_simulation(_straight_guide()), mw.ModalPortSweep())
    network = result.network

    assert network.port_names == ("left::Ey0", "right::Ey0")
    assert network.is_complete
    s = network.s.detach().cpu()
    beta = network.metadata["propagation_constants"].detach().cpu()
    assert torch.all(torch.abs(s[:, 0, 0]) < 0.05)
    assert torch.all(torch.abs(s[:, 1, 1]) < 0.05)
    assert torch.all(torch.abs(torch.abs(s[:, 1, 0]) - 1.0) < 0.04)
    assert torch.all(torch.abs(torch.abs(s[:, 0, 1]) - 1.0) < 0.04)
    for index in range(len(_FREQUENCIES)):
        # Reference planes are the ModePort positions, so the monitor offsets are
        # de-embedded and the through phase is beta times the port spacing.
        expected = cmath.exp(1j * float(beta[index, 0].real) * _PORT_SPACING)
        assert abs(cmath.phase(complex(s[index, 1, 0]) / expected)) < 0.15
        assert abs(cmath.phase(complex(s[index, 0, 1]) / expected)) < 0.15
    assert float(network.metadata["passive_incident_ratio"].max()) < 0.05
    assert set(result.ports) == {"left", "right"}
    assert result.port("left").mode_names == ("Ey0",)


def test_straight_waveguide_polarization_channels_do_not_cross_couple():
    sweep = mw.ModalPortSweep(modes={"left": (0, ("Ez", 0)), "right": (0, ("Ez", 0))})
    network = mw.run_modal_port_sweep(_simulation(_straight_guide()), sweep).network

    assert network.port_names == ("left::Ey0", "left::Ez0", "right::Ey0", "right::Ez0")
    magnitude = torch.abs(network.s.detach().cpu())
    through = {(2, 0), (3, 1), (0, 2), (1, 3)}
    for out in range(4):
        for column in range(4):
            if (out, column) in through:
                assert torch.all(torch.abs(magnitude[:, out, column] - 1.0) < 0.06)
            else:
                assert torch.all(magnitude[:, out, column] < 0.06)


def test_asymmetric_junction_is_reciprocal():
    scene = _base_scene()
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(-1.0, 0.0, 0.0), size=(2.0, 0.5, 0.35)),
                     material=mw.Material(eps_r=4.0), name="narrow")
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(1.0, 0.1, 0.0), size=(2.0, 0.7, 0.35)),
                     material=mw.Material(eps_r=4.0), name="wide")
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(0.15, -0.15, 0.0), size=(0.2, 0.2, 0.35)),
                     material=mw.Material(eps_r=6.0), name="bump")
    )
    network = mw.run_modal_port_sweep(_simulation(_add_ports(scene)), mw.ModalPortSweep()).network
    s = network.s.detach().cpu()

    s21 = s[:, 1, 0]
    s12 = s[:, 0, 1]
    assert torch.all(torch.abs(s21) > 0.5)
    assert torch.all(torch.abs(s21 - s12) / torch.abs(s21) < 2.0e-2)
    # The device is genuinely asymmetric: its two reflections differ.
    assert float(torch.max(torch.abs(torch.abs(s[:, 0, 0]) - torch.abs(s[:, 1, 1])))) > 0.01
    # Passive: no column carries more guided power out than in.
    assert torch.all(torch.sum(torch.abs(s) ** 2, dim=1) < 1.02)
