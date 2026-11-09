"""FDFD stretched-coordinate PML profile: shared FDTD semantics and grading."""
import math

import numpy as np
import pytest
import torch

from witwin.maxwell.fdfd.pml import stretch_profile_1d
from witwin.maxwell.fdtd.boundary.cpml import DEFAULT_CPML_CONFIG, _cpml_profile_1d

MU0 = 4e-7 * math.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = MU0 * C0
OMEGA = 2 * math.pi * 1.0e9
DT = 1.0e-12


def _graded_nodes(layers):
    # Nonuniform interior plus appended uniform PML cells on both sides.
    interior = np.cumsum(np.r_[0.0, np.linspace(0.01, 0.02, 17)])
    interior -= interior.mean()
    low = interior[0] - 0.01 * np.arange(layers, 0, -1)
    high = interior[-1] + 0.02 * np.arange(1, layers + 1)
    return np.concatenate((low, interior, high))


def _fdtd_rate(positions, nodes, layers, strength, vacuum_constant, *, low=True, high=True):
    """sigma / vacuum_constant of the FDTD CPML recovered from its decay factor."""
    scale = 1.0 if strength <= 0 or strength > 100 else strength
    if vacuum_constant == MU0:
        scale *= MU0 / EPS0
    _kappa, decay, _c = _cpml_profile_1d(
        positions, nodes, layers, DT, vacuum_constant,
        torch.tensor(scale, dtype=torch.float64), ETA0, DEFAULT_CPML_CONFIG,
        apply_low=low, apply_high=high,
    )
    return -np.log(decay.double().numpy()) / DT


@pytest.mark.parametrize("strength", [1e6, 1.0, 4.0, 0.0])
@pytest.mark.parametrize("sides", [(True, True), (True, False), (False, True)])
def test_fdfd_stretch_matches_fdtd_cpml_conductivity(strength, sides):
    layers = 8
    nodes = _graded_nodes(layers)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    low_layers, high_layers = (layers if side else 0 for side in sides)
    for positions, constant in ((nodes, EPS0), (half, MU0)):
        stretch = stretch_profile_1d(positions, nodes, low_layers, high_layers,
                                     omega=OMEGA, eps0=EPS0, eta0=ETA0, strength=strength)
        np.testing.assert_allclose(stretch.real, 1.0, rtol=0, atol=1e-15)
        expected = _fdtd_rate(positions, nodes, layers, strength, constant, low=sides[0], high=sides[1])
        # s = 1 + i sigma/(omega eps0) under exp(-i omega t)
        np.testing.assert_allclose(stretch.imag * OMEGA, expected, rtol=1e-9, atol=1e-9 * expected.max())


def test_stretch_is_identity_outside_layer_and_graded_inside():
    layers = 12
    nodes = np.arange(-(10 + layers), 11 + layers) * 0.015
    stretch = stretch_profile_1d(nodes, nodes, layers, layers,
                                 omega=OMEGA, eps0=EPS0, eta0=ETA0, strength=1e6)
    assert np.all(stretch[layers:-layers] == 1.0)
    low = stretch.imag[:layers + 1]
    assert np.all(np.diff(low) < 0) and low[-1] == 0.0
    np.testing.assert_allclose(stretch.imag, stretch.imag[::-1], rtol=1e-12)
    # Nominal cubic layer: sigma_max = -(m + 1) ln R / (2 eta0 d)
    sigma_max = -4.0 * math.log(1e-6) / (2.0 * ETA0 * layers * 0.015)
    assert stretch.imag[0] == pytest.approx(sigma_max / (OMEGA * EPS0), rel=1e-12)
    # The absorber is weak at its entrance, unlike a reflecting wall.
    assert stretch.imag[layers - 1] < 1e-2
