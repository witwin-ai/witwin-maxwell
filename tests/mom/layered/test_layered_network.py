"""R2b network gates: delta-gap ports, dense solve and N-port extraction on CUDA."""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.network import solve_layered_network
from tests.support.layered_network_references import (
    coplanar_strips_quasi_static, small_loop_radiation_resistance, square_loop_inductance)


pytestmark = pytest.mark.cuda
C0 = 299792458.0


def gap_port(name, positive, negative, axis, size):
    return mw.LumpedPort(name, positive=positive, negative=negative, voltage_path=mw.AxisPath(axis),
                         current_surface=mw.Box(position=tuple((p + n) / 2 for p, n in zip(positive, negative)),
                                                size=(size, size, size)),
                         reference_impedance=50.0)


def square_loop(side, width, gap, z):
    edge = side / 2 + width / 2
    inner = side / 2 - width / 2
    return [(-edge, -gap / 2, -edge, -inner, z), (gap / 2, edge, -edge, -inner, z),
            (-edge, edge, inner, edge, z), (-edge, -inner, -edge, edge, z), (inner, edge, -edge, edge, z)]


def test_square_loop_inductance_and_radiation():
    side, width, frequency, z = 1e-3, 20e-6, 1e9, .5e-3
    stack = LayerStack((Layer("air", 1e-3, 1.0),), HalfSpace(), HalfSpace())
    port = gap_port("loop", (width / 2, -side / 2, z), (-width / 2, -side / 2, z), "x", width)
    network = solve_layered_network(stack, square_loop(side, width, width, z), [port], (frequency,),
                                    bridge_width=width, max_edge=20e-6, edge_cell=5e-6)
    impedance = network.to_z()[0, 0, 0].item()
    omega = 2 * math.pi * frequency
    # exp(-i*omega*t): an inductor is Z = -i*omega*L.
    inductance = -impedance.imag / omega
    assert abs(inductance / square_loop_inductance(side, width) - 1) <= .03
    resistance = small_loop_radiation_resistance(side * side, C0 / frequency)
    assert abs(impedance.real / resistance - 1) <= .03


def coplanar_strips(length, width, gap, bridge, z):
    strips = [(0, length, gap / 2, gap / 2 + width, z), (0, length, -gap / 2 - width, -gap / 2, z)]
    ports = [gap_port(name, (x, gap / 2, z), (x, -gap / 2, z), "y", bridge)
             for name, x in (("p1", bridge / 2), ("p2", length - bridge / 2))]
    return strips, ports


def abcd(z):
    z11, z12, z21, z22 = z[..., 0, 0], z[..., 0, 1], z[..., 1, 0], z[..., 1, 1]
    return torch.stack((torch.stack((z11, z11 * z22 - z12 * z21), -1),
                        torch.stack((torch.ones_like(z11), z22), -1)), -2) / z21[..., None, None]


def test_coplanar_strips_on_dielectric_half_space():
    width = gap = 10e-6
    eps_r, frequency, z, bridge = 11.9, 10e9, 100e-6, 4e-6
    stack = LayerStack((Layer("substrate", z, eps_r),), HalfSpace(eps_r), HalfSpace())
    matrices = []
    for length in (1e-3, 2e-3):
        strips, ports = coplanar_strips(length, width, gap, bridge, z)
        # Z0 converges first order in the strip-edge cell (+3.0% at 2.5 um, +1.5% at 1.25 um).
        network = solve_layered_network(stack, strips, ports, (frequency,), bridge_width=bridge,
                                        max_edge=20e-6, edge_cell=1.25e-6)
        matrices.append(abcd(network.to_z()[0]))
    # Identical port error boxes cancel in a similarity transform of the line difference.
    difference = matrices[1] @ torch.linalg.inv(matrices[0])
    gamma_length = torch.arccosh(torch.trace(difference) / 2).item()
    beta = abs(gamma_length.imag) / 1e-3
    eps_eff = (beta * C0 / (2 * math.pi * frequency)) ** 2
    impedance = torch.sqrt(difference[0, 1] / difference[1, 0]).item()
    expected_eps, expected_impedance = coplanar_strips_quasi_static(width, gap, eps_r)
    assert abs(eps_eff / expected_eps - 1) <= .01
    assert abs(abs(impedance) / expected_impedance - 1) <= .03


def rfic_stack():
    return LayerStack((Layer("silicon", 300e-6, 11.9, 10.0),
                       Layer("oxide1", 2e-6, 4.1), Layer("oxide2", 3e-6, 4.1),
                       Layer("nitride", .1e-6, 7.5), Layer("oxide3", 4e-6, 4.1)),
                      HalfSpace(11.9, 10.0), HalfSpace())


@pytest.fixture(scope="module")
def rfic_network():
    z = rfic_stack().interfaces()[5]
    strips, ports = coplanar_strips(100e-6, 5e-6, 5e-6, 2e-6, z)
    run = lambda: solve_layered_network(rfic_stack(), strips, ports, (1e9, 50e9), bridge_width=2e-6,
                                        max_edge=5e-6, edge_cell=1.25e-6)
    return run(), run


def test_rfic_reciprocity_and_passivity(rfic_network):
    network, _ = rfic_network
    y = network.to_y()
    assert (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) / torch.linalg.matrix_norm(y)).max() <= 1e-10
    assert torch.linalg.matrix_norm(network.s, ord=2).max() <= 1 + 1e-9
    # Lossy silicon must dissipate: the port conductance matrix is positive definite.
    assert torch.linalg.eigvalsh((y + y.conj().transpose(-1, -2)) / 2).min() > 0


def test_rfic_network_is_deterministic(rfic_network):
    network, run = rfic_network
    again = run()
    assert torch.equal(network.s, again.s)
    assert torch.equal(network.to_y(), again.to_y())


def test_port_validation():
    stack = LayerStack((Layer("air", 1e-3, 1.0),), HalfSpace(), HalfSpace())
    strips, _ = coplanar_strips(100e-6, 10e-6, 10e-6, 4e-6, 0.)
    with pytest.raises(ValueError, match="axis-aligned"):
        gap_port("plane", (2e-6, 5e-6, 0.), (2e-6, -5e-6, 1e-6), "y", 4e-6)
    cases = ((gap_port("off", (2e-6, 30e-6, 0.), (2e-6, -5e-6, 0.), "y", 4e-6), "on metal"),
             (gap_port("short", (2e-6, 12e-6, 0.), (2e-6, 6e-6, 0.), "y", 4e-6), "bridge a gap"))
    for port, message in cases:
        with pytest.raises(ValueError, match=message):
            solve_layered_network(stack, strips, [port], (1e9,), bridge_width=4e-6, max_edge=5e-6)
    port = gap_port("vertical", (2e-6, 5e-6, 0.), (2e-6, 5e-6, 1e-6), "z", 4e-6)
    with pytest.raises(ValueError, match="horizontal"):
        solve_layered_network(stack, strips, [port], (1e9,), bridge_width=4e-6, max_edge=5e-6)
