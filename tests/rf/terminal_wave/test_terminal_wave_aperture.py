"""Aperture truncation of the TerminalWavePort common mode (GPU FDTD).

The quasi-static terminal basis grounds the whole aperture boundary. For a coupled
microstrip the wide even (common) mode is truncated by those artificial walls; the
port reports the share of terminal charge that terminates on them and warns above
``APERTURE_WALL_CHARGE_LIMIT`` (5 %). Measured on this pair (h = 0.02, W = 1.25 h,
s = 0.75 h, eps 4.4, dx = h/4, 0.35 GHz, matched coupled characteristic reference):

| aperture (half width x height) | even-mode wall charge | abs(Scc21) | abs(Scc11) | abs(Sdd21) |
|---|---|---|---|---|
| 0.10 x 0.07 (3.4 h margin, 3.5 h top) | 10.7 % (warns) | 0.952 | 0.183 | 0.995 |
| 0.16 x 0.14 (6.4 h margin, 7 h top)   | 4.6 %          | 0.983 | 0.055 | 0.999 |
| 0.24 x 0.21 (10.4 h margin, 10.5 h top) | 2.9 %        | 0.987 | 0.045 | 0.999 |

The remaining common-mode deficit at the largest aperture grows with frequency
(0.987 / 0.941 / 0.876 at 0.35 / 0.7 / 1.0 GHz) and is the quasi-TEM dispersion of
this electrically thick substrate, not truncation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD thru runs on CUDA")

DX = 0.005
HEIGHT, WIDTH, GAP, EPS = 0.02, 0.025, 0.015, 4.4


def _scene(half_y, top):
    nx, ny = round(0.09 / DX), round((half_y + 0.04) / DX)
    x = np.arange(-nx, nx + 1) * DX
    y = np.arange(-ny, ny + 1) * DX
    z = np.arange(round(-0.03 / DX), round((top + 0.04) / DX) + 1) * DX
    scene = mw.Scene(domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
                     grid=mw.GridSpec.custom(x, y, z), boundary=mw.BoundarySpec.pml(num_layers=6), device="cuda")
    length, wide = 2 * x[-1] + 20 * DX, 2 * y[-1]

    def sheet(name, low, high, coordinate):
        vertices = ((-length / 2, low), (length / 2, low), (length / 2, high), (-length / 2, high))
        scene.add_structure(mw.Structure(mw.PlanarPEC(vertices, normal_axis="z", coordinate=coordinate),
                                         mw.Material.pec(), name=name))

    sheet("ground", -wide, wide, 0.0)
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, HEIGHT / 2), size=(length, 2 * wide, HEIGHT)),
                                     mw.Material(eps_r=EPS), name="substrate"))
    offset = 0.5 * (WIDTH + GAP)
    sheet("p", offset - WIDTH / 2, offset + WIDTH / 2, HEIGHT)
    sheet("n", -offset - WIDTH / 2, -offset + WIDTH / 2, HEIGHT)
    for name, x_port, direction in (("P1", -0.03, "+"), ("P2", 0.03, "-")):
        scene.add_port(mw.TerminalWavePort(
            name, position=(x_port, 0.0, top / 2), size=(0.0, 2 * half_y, top), direction=direction,
            reference_plane=x_port, terminals=(mw.TerminalRef("p"), mw.TerminalRef("n")),
            reference=mw.TerminalRef("ground"),
            voltage_integrals=tuple(mw.CustomVoltageIntegral(mw.AxisSegment((x_port, c, HEIGHT), (x_port, c, 0.0)))
                                    for c in (offset, -offset)),
            reference_impedance="characteristic"))
    return scene


def _run(half_y, top):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mw.Simulation.fdtd(
            _scene(half_y, top), frequencies=(0.35e9,), excitations=mw.PortSweep(),
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=16),
            spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False,
        ).run()
    warned = any(issubclass(item.category, mw.TerminalWavePortApertureWarning) for item in caught)
    zc = result.ports["P1"].metadata["characteristic_impedance_matrix"][0].real
    z_even, z_odd = float(zc[0, 0] + zc[0, 1]), float(zc[0, 0] - zc[0, 1])
    mixed = result.network.to_mixed_mode((("P1::T1", "P1::T2"), ("P2::T1", "P2::T2")),
                                         z0=(2 * z_odd, z_even / 2, 2 * z_odd, z_even / 2)).s[0].abs()
    wall = result.ports["P1"].metadata["aperture_wall_charge_fraction"]
    return dict(warned=warned, wall=float(wall[0]), scc21=float(mixed[3, 1]), sdd21=float(mixed[2, 0]),
                scc11=float(mixed[1, 1]))


def test_common_mode_transmission_rises_as_the_aperture_grows_and_truncation_warns():
    rows = [_run(*size) for size in ((0.10, 0.07), (0.16, 0.14), (0.24, 0.21))]
    print("aperture rows", rows)
    small, medium, large = rows
    assert small["warned"] and not medium["warned"] and not large["warned"]
    assert small["wall"] > mw.terminal_waveport.APERTURE_WALL_CHARGE_LIMIT > medium["wall"] > large["wall"]
    assert small["scc21"] < medium["scc21"] < large["scc21"]
    assert small["scc21"] < 0.97 and large["scc21"] > 0.98
    assert small["scc11"] > medium["scc11"]
    assert all(row["sdd21"] > 0.99 for row in rows)
