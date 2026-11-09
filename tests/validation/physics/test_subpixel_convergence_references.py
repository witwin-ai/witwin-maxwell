"""CPU checks of the analytic references used by the subpixel convergence study.

The study (scripts/dev/fdtd/subpixel_convergence.py) measures FDTD errors of
order 1e-3..1e-5, so the references must be correct well below that.
"""
import math

import numpy as np
import pytest

from scripts.dev.fdtd import subpixel_analytic as A


def test_sphere_mie_matches_bohren_huffman_example():
    # Bohren & Huffman appendix example: m = 1.55, radius 0.525, wavelength 0.6328.
    x = 2 * math.pi * 0.525 / 0.6328
    q = A.mie_sphere_efficiencies(x, 1.55)
    assert q["q_sca"] == pytest.approx(3.10543, abs=1e-5)
    assert q["q_back"] == pytest.approx(2.92534, abs=1e-5)
    assert q["q_ext"] == pytest.approx(q["q_sca"], rel=1e-12)  # lossless optical theorem


def test_sphere_mie_limits_and_angular_integral():
    assert A.mie_sphere_efficiencies(1e-2, 2.0)["q_sca"] == pytest.approx(A.rayleigh_sphere_q_sca(1e-2, 2.0), rel=1e-3)
    # PEC sphere: monostatic peak sigma/(pi a^2) ~ 3.64 near ka = 1, and the
    # optical theorem holds for the perfectly conducting limit.
    pec = A.mie_sphere_efficiencies(1.0, pec=True)
    assert pec["q_back"] == pytest.approx(3.6376, abs=1e-3)
    assert pec["q_ext"] == pytest.approx(pec["q_sca"], rel=1e-12)
    # C_sca equals the angular integral of the bistatic cross section / (4 pi).
    radius, wavelength = 0.2387, 1.0
    mu, w = np.polynomial.legendre.leggauss(64)
    theta = np.arccos(mu)
    phi = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    sigma = A.mie_sphere_bistatic_rcs(radius, wavelength, th, ph, 2.0)
    c_sca = np.sum(w[:, None] * sigma) * (2 * math.pi / phi.size) / (4 * math.pi)
    x = 2 * math.pi * radius / wavelength
    assert c_sca == pytest.approx(A.mie_sphere_efficiencies(x, 2.0)["q_sca"] * math.pi * radius**2, rel=1e-10)


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_cylinder_series_limits(polarization):
    q = A.mie_cylinder_efficiencies(1.3, 2.0, polarization=polarization)
    assert q["q_ext"] == pytest.approx(q["q_sca"], rel=1e-12)
    small = A.mie_cylinder_efficiencies(1e-2, 2.0, polarization=polarization)["q_sca"]
    assert small == pytest.approx(A.rayleigh_cylinder_q_sca(1e-2, 2.0, polarization), rel=5e-3)
    pec = A.mie_cylinder_efficiencies(1.0, polarization=polarization, pec=True)
    assert pec["q_ext"] == pytest.approx(pec["q_sca"], rel=1e-12)


def test_parallel_plate_tm0_quasistatic_limit_and_semi_discrete_map():
    h, d, eps = 0.01, 0.004137, 2.25
    kx = 1e-2  # deep quasi-static: eps_eff is the series-capacitor value
    omega = A.parallel_plate_tm0_omega(kx, eps, d, h)
    eps_eff = (A.C0 * kx / omega) ** 2
    assert eps_eff == pytest.approx(h / (d / eps + (h - d)), rel=1e-6)
    # The Yee x/t map reduces to the exact discrete TEM dispersion in vacuum-like media.
    dx, dt = 1e-3, 1e-12
    k = 300.0
    expected = 2 / dt * math.asin(dt / 2 * A.C0 * 2 / dx * math.sin(k * dx / 2))
    assert A.semi_discrete_frequency(lambda kt: A.C0 * kt, k, dx, dt) == pytest.approx(expected, rel=1e-14)
