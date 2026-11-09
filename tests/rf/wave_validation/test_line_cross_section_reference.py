"""CPU tests for the independent 2D printed-line cross-section reference.

Each analytic case is exact (or converges to an exact value), so a sign, metric,
face-separation or boundary error makes these tests fail.
"""
import math

import numpy as np
import pytest
from scipy.optimize import brentq, newton

import benchmark.line_cross_section_reference as R
from scripts.dev.fdtd.printed_line_accuracy import richardson


def parallel_plate(width, gap, *, sheet=False, slabs=(), lossy=True):
    """PMC side walls; bottom (net 0) and top plates. With ``sheet`` a
    zero-thickness signal sheet sits at a quarter of the gap between two grounds."""
    if sheet:
        rects = (R.Rect(0.0, width, 0.0, 0.0, 0, lossy), R.Rect(0.0, width, gap, gap, 0, lossy),
                 R.Rect(0.0, width, gap / 4, gap / 4, 1, lossy))
    else:
        rects = (R.Rect(0.0, width, 0.0, 0.0, 0, lossy), R.Rect(0.0, width, gap, gap, 1, lossy))
    return R.CrossSection(box=(0.0, width, 0.0, gap), rects=rects, slabs=tuple(slabs),
                          walls=("pmc",) * 4)


def test_fixture_dimensions_match_scene():
    import benchmark.scenes.rf.printed_lines as scene
    for name in ("WIDTH", "HEIGHT", "GAP", "PAIR_WIDTH", "PAIR_OFFSET"):
        assert getattr(R, name) == getattr(scene, name)
    material = scene.mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9))
    assert material.thickness == R.COPPER_THICKNESS and material.conductivity == R.COPPER_CONDUCTIVITY
    # Same diffusion two-port per face (scene uses exp(-i w t): conjugate).
    z = material.surface_impedance([1e9]).numpy()[0]
    foil = R.foil_surface_impedance(1e9)
    np.testing.assert_allclose(np.conj(z[0, 0]), foil["self_impedance_ohm"], rtol=1e-10)
    np.testing.assert_allclose(np.conj(z[0, 1]), foil["mutual_impedance_ohm"], rtol=1e-6, atol=1e-12)


def test_parallel_plate_impedance_and_conductor_loss_exact():
    width, gap = 0.02, 0.004
    qs = R.QuasiStatic(R.Mesh(parallel_plate(width, gap), R.MeshSpec(5e-4, 5e-4, 0.0)))
    rs = R.foil_surface_impedance(1e9)["surface_resistance_ohm"]
    (mode,) = qs.modes(1e9, surface_resistance=rs)
    z = mode["voltage"][0] / mode["current"][0]
    assert abs(z / (R.ETA0 * gap / width) - 1) < 1e-12
    assert abs(mode["alpha_c_np_per_m"] / (rs / (R.ETA0 * gap)) - 1) < 1e-12


def test_sheet_faces_are_separate_surfaces():
    # Sheet at d/4: its lower face carries 3I/4 and its upper face I/4, as do
    # the facing grounds. Separate faces give alpha_c = (10/3) Rs/(eta d);
    # merging the two sheet faces into one surface would give (8/3) Rs/(eta d).
    width, gap = 0.02, 0.004
    qs = R.QuasiStatic(R.Mesh(parallel_plate(width, gap, sheet=True), R.MeshSpec(2.5e-4, 2.5e-4, 0.0)))
    rs = R.foil_surface_impedance(1e9)["surface_resistance_ohm"]
    (mode,) = qs.modes(1e9, surface_resistance=rs)
    assert abs(mode["voltage"][0] / mode["current"][0] / (3 * R.ETA0 * gap / (16 * width)) - 1) < 1e-12
    assert abs(mode["alpha_c_np_per_m"] / (10 / 3 * rs / (R.ETA0 * gap)) - 1) < 1e-12


def test_full_wave_conductor_loss_matches_parallel_plate():
    width, gap, f = 0.02, 0.004, 1e9
    fw = R.FullWave(R.Mesh(parallel_plate(width, gap), R.MeshSpec(5e-4, 5e-4, 0.0)), f)
    mode = fw.solve(2 * math.pi * f / R.C0, count=1)[0]
    rs = R.foil_surface_impedance(f)["surface_resistance_ohm"]
    assert abs(mode["beta_rad_per_m"] / (2 * math.pi * f / R.C0) - 1) < 1e-10
    assert abs(fw.conductor_alpha(mode, rs) / (rs / (R.ETA0 * gap)) - 1) < 1e-8


def _two_layer_tm0(f, d1, d2, eps1, eps2=1.0):
    k0 = 2 * math.pi * f / R.C0

    def F(beta):
        kz1 = np.sqrt(eps1 * k0 ** 2 - beta ** 2 + 0j)
        kap = np.sqrt(beta ** 2 - eps2 * k0 ** 2 + 0j)
        return kz1 / eps1 * np.tan(kz1 * d1) - kap / eps2 * np.tanh(kap * d2)
    real = brentq(lambda b: F(b).real, k0 * math.sqrt(eps2) * (1 + 1e-9), k0 * math.sqrt(eps1.real) * (1 - 1e-9))
    return complex(newton(F, real + 0j, tol=1e-14, maxiter=100)) if np.iscomplexobj(eps1) else real, k0


@pytest.mark.parametrize("sigma", [0.0, 0.05])
def test_full_wave_two_layer_parallel_plate_dispersion(sigma):
    # TM0 of a two-layer parallel plate (PMC sides) at strong dispersion.
    f, d1, d2, eps = 6e9, 0.004, 0.006, 4.0
    omega = 2 * math.pi * f
    eps_c = eps - 1j * sigma / (omega * R.EPS0) if sigma else eps
    exact, k0 = _two_layer_tm0(f, d1, d2, eps_c)
    errors = []
    for h in (5e-4, 2.5e-4):
        section = parallel_plate(0.004, d1 + d2, slabs=(R.Slab(0.0, d1, eps, sigma),))
        fw = R.FullWave(R.Mesh(section, R.MeshSpec(h, h, 0.0)), f)
        mode = fw.solve(k0 * 1.6, count=1)[0]
        errors.append(abs(mode["kx"] / exact - 1))
    assert errors[-1] < 1e-3
    assert errors[-1] < errors[0] / 3  # second order: 4x per halving
    static = (d1 + d2) / (d1 / eps + d2)
    assert abs(exact) ** 2 / k0 ** 2 > static * 1.05  # the case is genuinely dispersive


def test_stripline_zero_thickness_converges_to_cohn():
    values = []
    for level in (2, 3, 4):
        qs = R.QuasiStatic(R.Mesh(R.printed_line_section("stripline"), R.MeshSpec(2e-4, 4e-3, 0.3).refined(level)))
        (mode,) = qs.modes(1e9)
        values.append(abs(mode["voltage"][0] / mode["current"][0]))
    exact = R.stripline_cohn()
    assert abs(values[-1] / exact - 1) < 1e-3
    extrapolated = richardson(values)
    assert abs(extrapolated["value"] / exact - 1) < 1e-4
    assert 0.8 < extrapolated["order"] < 1.6  # first order from the sheet-edge singularity


def test_homogeneous_lossy_line_is_exact_tem():
    f = 1e9
    fw = R.FullWave(R.Mesh(R.printed_line_section("cpw", sigma=0.01), R.MeshSpec(4e-4, 8e-3, 0.6)), f)
    mode = fw.solve(2 * math.pi * f / R.C0 * 1.5, count=1)[0]
    alpha, beta = R.homogeneous_gamma(f, 2.25, 0.01)
    assert abs(mode["beta_rad_per_m"] / beta - 1) < 1e-8
    assert abs(mode["alpha_np_per_m"] / alpha - 1) < 1e-7


def test_full_wave_reduces_to_quasi_static_at_low_frequency():
    section = R.printed_line_section("differential_pair")
    mesh = R.Mesh(section, R.MeshSpec(4e-4, 8e-3, 0.6))
    qs = R.QuasiStatic(mesh).modes(2e7)
    fw = R.FullWave(mesh, 2e7)
    modes = fw.solve(2 * math.pi * 2e7 / R.C0 * 1.35, count=2)
    for q, w in zip(sorted(qs, key=lambda m: m["eps_eff"]), sorted(modes, key=lambda m: m["eps_eff"])):
        assert abs(w["eps_eff"] / q["eps_eff"] - 1) < 2e-3


def test_explicit_nodes_must_contain_geometry():
    section = R.printed_line_section("cpw")
    y = np.linspace(-0.14, 0.14, 113)  # 2.5 mm: contains +-5 and +-10 mm
    z = np.linspace(-0.13, 0.13, 105)
    R.Mesh(section, nodes=(y, z))
    with pytest.raises(ValueError):
        R.Mesh(section, nodes=(np.linspace(-0.14, 0.14, 100), z))


def test_richardson_recovers_power_law():
    h = 0.1 / 2 ** np.arange(4)
    values = 3.0 + 0.7 * h ** 1.5
    out = richardson(values)
    assert abs(out["value"] - 3.0) < 1e-12 and abs(out["order"] - 1.5) < 1e-9
