"""Reference-adapter export of TerminalWavePort and custom / composite V-I integrals."""
from __future__ import annotations

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")

import witwin.maxwell as mw
from witwin.maxwell.adapters.tidy3d import (
    current_integral_to_tidy3d,
    terminal_wave_port_to_tidy3d,
    voltage_integral_to_tidy3d,
)

UM = 1.0e6


def _pair_port(**kwargs):
    voltages = tuple(mw.CustomVoltageIntegral(mw.PolylinePath(((0, c, 2e-4), (0, c + 1e-4, 1e-4), (0, c + 1e-4, 0.0))))
                     for c in (3e-4, -3e-4))
    currents = (
        mw.CustomCurrentIntegral(mw.RectangleLoop(center=(0, 3e-4, 2e-4), size=(0, 3.1e-4, 2.1e-4))),
        mw.CompositeCurrentIntegral(((1.0, mw.CustomCurrentIntegral(mw.CircleLoop((0, -3e-4, 2e-4), 1.6e-4, "x", 16))),
                                     (-1.0, mw.CustomCurrentIntegral(mw.PolygonLoop(
                                         ((0, -5e-4, 0.5e-4), (0, -1e-4, 0.5e-4), (0, -3e-4, 4e-4))))))),
    )
    return mw.TerminalWavePort("P1", position=(0, 0, 4e-4), size=(0, 2e-3, 8e-4), direction="+", reference_plane=0.0,
                               terminals=(mw.TerminalRef("a"), mw.TerminalRef("b")), reference=mw.TerminalRef("g"),
                               voltage_integrals=voltages, current_integrals=currents, **kwargs)


def test_terminal_wave_port_maps_to_the_reference_terminal_wave_port():
    reference = terminal_wave_port_to_tidy3d(_pair_port(reference_impedance=50.0))
    assert isinstance(reference, td.plugins.smatrix.TerminalWavePort)
    assert reference.center == pytest.approx((0.0, 0.0, 400.0)) and reference.size == pytest.approx((0.0, 2000.0, 800.0))
    assert reference.direction == "+" and reference.reference_impedance == 50.0
    first, second = reference.terminal_specs
    # Reference V = phi(last) - phi(first): the witwin terminal-to-reference path is reversed.
    np.testing.assert_allclose(np.asarray(first.voltage_spec.vertices), [[400, 0], [400, 100], [300, 200]], atol=1e-9)
    assert first.voltage_spec.axis == 0 and first.voltage_spec.position == 0.0
    loop = np.asarray(first.current_spec.vertices)
    assert np.allclose(loop[0], loop[-1])
    area = 0.5 * np.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1])
    assert area > 0      # right-handed about +x for a '+' port
    assert isinstance(second.current_spec, td.CompositeCurrentIntegralSpec)
    circle, polygon = second.current_spec.path_specs
    signed = [0.5 * np.sum(np.asarray(spec.vertices)[:-1, 0] * np.asarray(spec.vertices)[1:, 1]
                           - np.asarray(spec.vertices)[1:, 0] * np.asarray(spec.vertices)[:-1, 1])
              for spec in (circle, polygon)]
    assert signed[0] > 0 > signed[1]
    assert terminal_wave_port_to_tidy3d(_pair_port(reference_impedance="characteristic")).reference_impedance == "Z0"


def test_reference_adapter_rejects_constructs_without_an_equivalent():
    with pytest.raises(ValueError, match="coupled reference-impedance matrix"):
        terminal_wave_port_to_tidy3d(_pair_port(reference_impedance=((60.0, 10.0), (10.0, 60.0))))
    with pytest.raises(ValueError, match="Per-terminal reference impedances"):
        terminal_wave_port_to_tidy3d(_pair_port(reference_impedance=(40.0, 60.0)))
    with pytest.raises(ValueError, match="modal-basis"):
        terminal_wave_port_to_tidy3d(_pair_port(basis="modal"))
    segment = mw.CustomVoltageIntegral(mw.AxisSegment((0, 0, 1e-4), (0, 0, 0)))
    with pytest.raises(ValueError, match="Only CustomVoltageIntegral"):
        voltage_integral_to_tidy3d(mw.CompositeVoltageIntegral(((0.5, segment), (0.5, segment))), axis="x")
    loop = mw.CustomCurrentIntegral(mw.RectangleLoop(center=(0, 0, 0), size=(0, 1e-4, 1e-4)))
    with pytest.raises(ValueError, match="flat composites"):
        current_integral_to_tidy3d(mw.CompositeCurrentIntegral(((0.5, loop),)), axis="x", direction="+")
    reversed_loop = np.asarray(current_integral_to_tidy3d(loop, axis="x", direction="-").vertices)
    area = 0.5 * np.sum(reversed_loop[:-1, 0] * reversed_loop[1:, 1] - reversed_loop[1:, 0] * reversed_loop[:-1, 1])
    assert area < 0      # right-handed about -x for a '-' port
