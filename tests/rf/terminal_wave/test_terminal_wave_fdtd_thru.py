"""Coupled-microstrip matched thru through two 2-terminal TerminalWavePorts (GPU FDTD).

Both ports use ``reference_impedance="characteristic"``: the full (coupled) 2x2
terminal characteristic matrix from the power-voltage definition, so the network S
carries a non-diagonal reference matrix. Converting to mixed mode with
``Zd = 2 Zo`` and ``Zc = Ze / 2`` must give a matched thru: ``|Sdd21| ~ 1``, low
reflection and no differential/common conversion on the symmetric pair.

Budget-reduced case (one frequency, dx = h/4, 0.06 m between ports). At 0.7 GHz
this electrically thick substrate (h = 0.02 m) shows quasi-TEM dispersion in the
common mode (|Scc21| 0.930 here, 0.983 at 0.35 GHz on the same aperture), so the
gate on ``|Scc21|`` is looser than on ``|Sdd21|``. Aperture truncation is covered
separately by ``test_terminal_wave_aperture.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD thru runs on CUDA")

DX = 0.005
HEIGHT, WIDTH, GAP, EPS = 0.02, 0.025, 0.015, 4.4


def _thru_scene():
    nx, ny = round(0.09 / DX), round(0.2 / DX)
    x = np.arange(-nx, nx + 1) * DX
    y = np.arange(-ny, ny + 1) * DX
    z = np.arange(round(-0.03 / DX), round(0.18 / DX) + 1) * DX
    scene = mw.Scene(domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
                     grid=mw.GridSpec.custom(x, y, z), boundary=mw.BoundarySpec.pml(num_layers=6), device="cuda")
    length = 2 * x[-1] + 20 * DX

    def sheet(name, low, high, coordinate):
        vertices = ((-length / 2, low), (length / 2, low), (length / 2, high), (-length / 2, high))
        scene.add_structure(mw.Structure(mw.PlanarPEC(vertices, normal_axis="z", coordinate=coordinate),
                                         mw.Material.pec(), name=name))

    sheet("ground", -0.3, 0.3, 0.0)
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, HEIGHT / 2), size=(length, 0.6, HEIGHT)),
                                     mw.Material(eps_r=EPS), name="substrate"))
    offset = 0.5 * (WIDTH + GAP)
    sheet("p", offset - WIDTH / 2, offset + WIDTH / 2, HEIGHT)
    sheet("n", -offset - WIDTH / 2, -offset + WIDTH / 2, HEIGHT)
    for name, x_port, direction in (("P1", -0.03, "+"), ("P2", 0.03, "-")):
        scene.add_port(mw.TerminalWavePort(
            name, position=(x_port, 0.0, 0.07), size=(0.0, 0.32, 0.14), direction=direction, reference_plane=x_port,
            terminals=(mw.TerminalRef("p"), mw.TerminalRef("n")), reference=mw.TerminalRef("ground"),
            voltage_integrals=tuple(mw.CustomVoltageIntegral(mw.AxisSegment((x_port, c, HEIGHT), (x_port, c, 0.0)))
                                    for c in (offset, -offset)),
            reference_impedance="characteristic"))
    return scene


def test_coupled_microstrip_thru_is_matched_in_mixed_mode():
    result = mw.Simulation.fdtd(
        _thru_scene(), frequencies=(0.7e9,), excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=16),
        spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False,
    ).run()
    network = result.network
    assert network.port_names == ("P1::T1", "P1::T2", "P2::T1", "P2::T2")
    reference = network.reference_matrix[0].real
    assert float(reference[0, 1]) > 0.05 * float(reference[0, 0])          # coupled, non-diagonal
    s = network.s[0]
    assert float(network.metadata["extraction_condition_number"].max()) < 1.5
    assert float(torch.linalg.svdvals(s).max()) < 1.02
    assert float((s - s.transpose(0, 1)).abs().max()) < 0.02               # reciprocity

    zc = result.ports["P1"].metadata["characteristic_impedance_matrix"][0].real
    z_even, z_odd = float(zc[0, 0] + zc[0, 1]), float(zc[0, 0] - zc[0, 1])
    mixed = network.to_mixed_mode((("P1::T1", "P1::T2"), ("P2::T1", "P2::T2")),
                                  z0=(2 * z_odd, z_even / 2, 2 * z_odd, z_even / 2)).s[0].abs()
    # mixed order: d1, c1, d2, c2
    assert float(mixed[2, 0]) > 0.97                                         # |Sdd21|
    assert float(mixed[3, 1]) > 0.88                                         # |Scc21|
    assert float(mixed[0, 0]) < 0.1 and float(mixed[1, 1]) < 0.12           # |Sdd11|, |Scc11|
    conversion = (mixed[1, 0], mixed[0, 1], mixed[3, 0], mixed[2, 1], mixed[1, 2], mixed[0, 3])
    assert max(float(value) for value in conversion) < 0.02                  # |Sdc|, |Scd|

    port = result.ports["P1"]
    assert port.voltage.shape == (4, 2, 1) and port.metadata["impedance_definition"] == "power_voltage"
    modal = network.to_basis("modal")
    assert modal.port_names == ("P1::M1", "P1::M2", "P2::M1", "P2::M2")
    assert float(modal.s[0].abs()[2, 0]) > 0.88 and float(modal.s[0].abs()[3, 1]) > 0.97
