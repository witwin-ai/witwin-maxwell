import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0

pp = mw.postprocess

FREQUENCY = 1.0e9
WAVELENGTH = C_0 / FREQUENCY


def _linear_cut(count, weights=None, *, spacing=0.5, samples=18001, steering=None):
    """Signed-theta cut in the x-z plane of an x-axis linear array."""

    positions = pp.rectangular_lattice(count, 1, spacing * WAVELENGTH)
    angles = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, samples, dtype=torch.float64)
    if weights is not None:
        weights = weights.to(torch.complex128)
    af = pp.array_factor(
        positions, weights, angles, torch.zeros_like(angles), frequency=FREQUENCY, steering=steering
    )
    return angles, af


def test_uniform_broadside_beamwidth_nulls_and_sidelobe():
    count = 64
    angles, af = _linear_cut(count)
    metrics = pp.lobe_metrics(angles, af)
    assert float(metrics.peak_angle) == pytest.approx(0.0, abs=1e-9)
    assert float(metrics.half_power_beamwidth) == pytest.approx(0.886 * 2.0 / count, rel=5e-3)
    assert float(metrics.first_null_beamwidth) == pytest.approx(2.0 * math.asin(2.0 / count), rel=1e-5)
    # Large-N limit of the uniform first sidelobe is -13.26 dB.
    assert float(metrics.first_sidelobe_db) == pytest.approx(-13.26, abs=0.03)
    assert float(metrics.max_sidelobe_db) == pytest.approx(float(metrics.first_sidelobe_db))
    torch.testing.assert_close(metrics.half_power_angles[0], -metrics.half_power_angles[1])


def test_null_directions_match_uniform_array_zeros_on_coarse_sampling():
    count = 8
    angles, af = _linear_cut(count, samples=361)
    metrics = pp.lobe_metrics(angles, af)
    expected = torch.tensor(
        [math.asin(m * 2.0 / count) for m in (-3, -2, -1, 1, 2, 3)], dtype=torch.float64
    )
    torch.testing.assert_close(metrics.null_angles, expected, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(metrics.first_null_angles, expected[2:4], rtol=0.0, atol=2e-5)
    assert bool(torch.all(metrics.null_levels_db < -20.0))


def test_beamwidth_is_robust_to_sampling_density():
    fine = pp.lobe_metrics(*_linear_cut(16, samples=20001))
    coarse = pp.lobe_metrics(*_linear_cut(16, samples=181))
    assert float(coarse.half_power_beamwidth) == pytest.approx(float(fine.half_power_beamwidth), rel=2e-3)
    assert float(coarse.first_sidelobe_db) == pytest.approx(float(fine.first_sidelobe_db), abs=0.05)
    assert float(coarse.first_null_beamwidth) == pytest.approx(float(fine.first_null_beamwidth), rel=1e-3)


@pytest.mark.parametrize("count", [10, 16, 17])
@pytest.mark.parametrize("sidelobe_db", [20.0, 30.0, 40.0])
def test_dolph_chebyshev_sidelobes_equal_design_level(count, sidelobe_db):
    weights = pp.taper_weights(count, "chebyshev", sidelobe_db=sidelobe_db)
    metrics = pp.lobe_metrics(*_linear_cut(count, weights, samples=6001))
    assert float(metrics.max_sidelobe_db) == pytest.approx(-sidelobe_db, abs=0.1)
    assert float(metrics.first_sidelobe_db) == pytest.approx(-sidelobe_db, abs=0.1)
    assert bool(torch.all(torch.abs(metrics.sidelobe_levels_db + sidelobe_db) < 0.1))


@pytest.mark.parametrize(("count", "nbar"), [(20, 4), (32, 5), (64, 6)])
def test_taylor_peak_sidelobe_near_design_level(count, nbar):
    weights = pp.taper_weights(count, "taylor", sidelobe_db=30.0, nbar=nbar)
    metrics = pp.lobe_metrics(*_linear_cut(count, weights))
    assert float(metrics.max_sidelobe_db) == pytest.approx(-30.0, abs=0.5)
    uniform = pp.lobe_metrics(*_linear_cut(count))
    assert float(metrics.half_power_beamwidth) > float(uniform.half_power_beamwidth)


def test_steering_moves_main_lobe():
    target = math.radians(30.0)
    metrics = pp.lobe_metrics(*_linear_cut(16, samples=1801, steering=(target, 0.0)))
    assert float(metrics.peak_angle) == pytest.approx(target, abs=1e-5)
    # Off-broadside scan broadens the beam by roughly 1 / cos(theta0).
    broadside = pp.lobe_metrics(*_linear_cut(16, samples=1801))
    ratio = float(metrics.half_power_beamwidth / broadside.half_power_beamwidth)
    assert ratio == pytest.approx(1.0 / math.cos(target), rel=0.03)
    # The back direction lies outside the sampled half-plane.
    assert math.isnan(float(metrics.front_to_back_db))


def test_power_field_and_db_scales_agree():
    angles, af = _linear_cut(12, samples=2001)
    power = pp.lobe_metrics(angles, af.abs().square())
    complex_metrics = pp.lobe_metrics(angles, af)
    field = pp.lobe_metrics(angles, af.abs(), scale="field")
    db = pp.lobe_metrics(angles, 10.0 * torch.log10(af.abs().square().clamp_min(1e-30)), scale="db")
    for other in (complex_metrics, field, db):
        torch.testing.assert_close(other.half_power_beamwidth, power.half_power_beamwidth)
        torch.testing.assert_close(other.max_sidelobe_db, power.max_sidelobe_db, rtol=0.0, atol=1e-9)


def test_periodic_cut_wraps_peak_and_front_to_back():
    angles = torch.linspace(0.0, 2.0 * math.pi, 721, dtype=torch.float64)
    field = (1.0 + 0.8 * torch.cos(angles - 0.1)) / 1.8
    with_endpoint = pp.lobe_metrics(angles, field, scale="field", periodic=True)
    open_grid = pp.lobe_metrics(angles[:-1], field[:-1], scale="field", periodic=True)
    assert float(with_endpoint.peak_angle) == pytest.approx(0.1, abs=1e-4)
    assert float(with_endpoint.front_to_back_db) == pytest.approx(20.0 * math.log10(1.8 / 0.2), abs=1e-3)
    torch.testing.assert_close(with_endpoint.half_power_beamwidth, open_grid.half_power_beamwidth)
    # Cardioid: the half-power crossings straddle the 0/2*pi seam.
    left, right = (float(value) for value in with_endpoint.half_power_angles)
    assert left > math.pi and right < math.pi
    assert math.isnan(float(with_endpoint.max_sidelobe_db))
    assert with_endpoint.null_angles.numel() == 1
    assert float(with_endpoint.null_angles[0]) == pytest.approx(math.pi + 0.1, abs=1e-3)


def test_degenerate_patterns():
    angles = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    constant = pp.lobe_metrics(angles, torch.ones_like(angles))
    assert math.isnan(float(constant.half_power_beamwidth))
    assert math.isnan(float(constant.first_null_beamwidth))
    assert math.isnan(float(constant.max_sidelobe_db))
    assert constant.null_angles.numel() == 0

    edge_peak = pp.lobe_metrics(angles, torch.exp(-((angles + 1.0) / 0.3) ** 2))
    assert float(edge_peak.peak_angle) == pytest.approx(-1.0)
    assert math.isnan(float(edge_peak.half_power_angles[0]))
    assert not math.isnan(float(edge_peak.half_power_angles[1]))

    with pytest.raises(ValueError):
        pp.lobe_metrics(angles, torch.zeros_like(angles))
    with pytest.raises(ValueError):
        pp.lobe_metrics(angles, -torch.ones_like(angles))
    with pytest.raises(ValueError):
        pp.lobe_metrics(angles.flip(0), torch.ones_like(angles))


def test_sidelobe_level_is_differentiable_through_taper_amplitudes():
    count = 8
    positions = pp.rectangular_lattice(count, 1, 0.5 * WAVELENGTH)
    angles = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, 1201, dtype=torch.float64)
    base = pp.taper_weights(count, "hamming")

    def max_sidelobe(amplitudes):
        af = pp.array_factor(
            positions, amplitudes.to(torch.complex128), angles, torch.zeros_like(angles), frequency=FREQUENCY
        )
        metrics = pp.lobe_metrics(angles, af)
        return metrics.max_sidelobe_db, metrics.half_power_beamwidth

    amplitudes = base.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(max_sidelobe, (amplitudes,), eps=1e-6, atol=1e-5)


def _sphere(theta_count=91, phi_count=181):
    theta = torch.linspace(0.0, math.pi, theta_count, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, phi_count, dtype=torch.float64)
    return theta, phi


def test_sphere_metrics_for_steered_planar_array_with_hemispherical_element():
    theta, phi = _sphere()
    theta_grid, _ = torch.meshgrid(theta, phi, indexing="ij")
    element = torch.clamp_min(torch.cos(theta_grid), 0.0)
    positions = pp.rectangular_lattice(8, 8, 0.5 * WAVELENGTH)
    target = (math.radians(24.0), math.radians(40.0))
    data = pp.compute_array_pattern(
        positions, element=element, theta=theta, phi=phi, frequencies=FREQUENCY, steering=target
    )
    metrics = data.lobe_metrics()
    assert float(metrics.peak_theta) == pytest.approx(target[0], abs=1e-12)
    assert float(metrics.peak_phi) == pytest.approx(target[1], abs=1e-12)
    assert float(metrics.peak_value) == pytest.approx(float(data.directivity_max[0]))
    assert math.isinf(float(metrics.front_to_back_db))
    assert -16.0 < float(metrics.max_sidelobe_db) < -10.0
    # The cos(theta) element pulls the total-pattern peak slightly toward broadside.
    dense_theta = torch.linspace(0.3, 0.5, 20001, dtype=torch.float64)
    dense = pp.array_factor(
        positions, None, dense_theta, torch.full_like(dense_theta, target[1]),
        frequency=FREQUENCY, steering=target,
    ).abs().square() * torch.cos(dense_theta).square()
    true_peak = float(dense_theta[torch.argmax(dense)])
    assert true_peak < target[0]
    assert float(metrics.elevation.peak_angle) == pytest.approx(true_peak, abs=1e-3)
    assert float(metrics.azimuth.peak_angle) == pytest.approx(target[1], abs=2e-3)
    assert metrics.main_lobe_mask.shape == data.directivity.shape[1:]
    assert bool(metrics.main_lobe_mask[:, 0].eq(metrics.main_lobe_mask[:, -1]).all())


def test_sphere_metrics_broadside_matches_principal_cut():
    theta, phi = _sphere(181, 361)
    theta_grid, _ = torch.meshgrid(theta, phi, indexing="ij")
    element = torch.clamp_min(torch.cos(theta_grid), 0.0)
    data = pp.compute_array_pattern(
        pp.rectangular_lattice(10, 1, 0.5 * WAVELENGTH),
        element=element, theta=theta, phi=phi, frequencies=FREQUENCY,
    )
    metrics = data.lobe_metrics()
    assert float(metrics.peak_theta) == 0.0
    # The peak sits on the pole; the azimuth cut is constant there.
    assert math.isnan(float(metrics.azimuth.half_power_beamwidth))
    assert float(metrics.max_sidelobe_db) == pytest.approx(float(metrics.elevation.max_sidelobe_db), abs=0.1)
    assert float(metrics.elevation.half_power_beamwidth) > 0.0


def test_isotropic_planar_array_reports_mirror_beam_as_sidelobe():
    theta, phi = _sphere()
    data = pp.compute_array_pattern(
        pp.rectangular_lattice(4, 4, 0.5 * WAVELENGTH), theta=theta, phi=phi, frequencies=FREQUENCY
    )
    metrics = data.lobe_metrics()
    assert float(metrics.max_sidelobe_db) == pytest.approx(0.0, abs=1e-9)
    assert float(metrics.front_to_back_db) == pytest.approx(0.0, abs=1e-9)
