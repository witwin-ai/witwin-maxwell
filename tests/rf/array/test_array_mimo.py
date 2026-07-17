import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.array import ARRAY_ACCEPTANCE_BUDGET

_ETA = 376.730313668


def _sphere(points_theta=181, points_phi=361, dtype=torch.float64):
    theta_vector = torch.linspace(0.0, math.pi, points_theta, dtype=dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, points_phi, dtype=dtype)
    return torch.meshgrid(theta_vector, phi_vector, indexing="ij")


def _isotropic_dipole_basis(spacing, *, frequency=1.0e9):
    """Two co-polarized isotropic sensors separated by ``spacing`` along z.

    Each column carries unit realized gain (magnitude chosen so 4*pi*U = 1) and a
    spatial phase offset. This is the independent analytic reference: in a 3-D
    isotropic (uniform) multipath field the complex correlation between two such
    sensors is the closed-form ``sin(k d) / (k d)`` (Clarke's spatial-correlation
    result), so the envelope correlation is ``(sin(k d) / (k d))^2``.
    """

    dtype = torch.float64
    cdtype = torch.complex128
    frequencies = torch.tensor([frequency], dtype=dtype)
    wave_number = 2.0 * math.pi * frequency / 299792458.0
    theta, phi = _sphere(dtype=dtype)
    magnitude = math.sqrt(2.0 * _ETA / (4.0 * math.pi))
    positions = (-0.5 * spacing, 0.5 * spacing)
    columns = [
        magnitude * torch.exp(-1j * wave_number * z * torch.cos(theta)).to(cdtype)
        for z in positions
    ]
    e_theta = torch.stack(columns, dim=0)[None]
    e_phi = torch.zeros_like(e_theta)
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=e_theta, e_phi=e_phi,
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    network = mw.NetworkData(
        frequencies=frequencies, s=torch.zeros((1, 2, 2), dtype=cdtype), z0=50.0,
        port_names=("a", "b"),
    )
    return mw.ArrayBasisData(network=network, embedded_patterns=patterns, fingerprint=f"iso-{spacing}"), wave_number


@pytest.mark.parametrize("spacing", (0.05, 0.15, 0.35, 0.6))
def test_ecc_matches_closed_form_two_isotropic_sensors(spacing):
    """Independent analytic gate: ECC == (sin(kd)/(kd))^2 in an isotropic field.

    Reference: Clarke's spatial correlation for isotropic 3-D scattering
    (R. H. Clarke, 1968; Vaughan & Andersen, "Antenna Diversity in Mobile
    Communications"). This reference is not produced by the code under test.
    """

    basis, wave_number = _isotropic_dipole_basis(spacing)
    environment = mw.MultipathEnvironment(
        theta=basis.eep.theta, phi=basis.eep.phi, cross_polar_ratio_db=6.0
    )

    mimo = basis.mimo(environment)

    kd = wave_number * spacing
    analytic_rho = math.sin(kd) / kd
    analytic_ecc = analytic_rho * analytic_rho
    correlation = mimo.correlation[0]
    normalized = correlation[0, 1] / torch.sqrt(correlation[0, 0].real * correlation[1, 1].real)
    assert abs(float(normalized.imag)) < 1.0e-9
    assert abs(float(normalized.real) - analytic_rho) <= ARRAY_ACCEPTANCE_BUDGET.reference_ecc_error
    assert abs(float(mimo.ecc[0, 0, 1]) - analytic_ecc) <= ARRAY_ACCEPTANCE_BUDGET.reference_ecc_error
    assert mimo.source == "dual_polarized_far_field_integral"


def test_mean_effective_gain_of_ideal_isotropic_antenna_is_half():
    """Analytic gate: an ideal lossless isotropic antenna has MEG = 0.5 at 0 dB XPR.

    Taga's MEG identity: MEG = XPR/(1+XPR) for a single-polarization isotropic
    antenna, which is 0.5 at a balanced (0 dB) cross-polar ratio.
    """

    basis, _ = _isotropic_dipole_basis(0.2)
    balanced = mw.MultipathEnvironment(theta=basis.eep.theta, phi=basis.eep.phi, cross_polar_ratio_db=0.0)
    meg = basis.mimo(balanced).mean_effective_gain
    torch.testing.assert_close(meg, torch.full_like(meg, 0.5), rtol=0.0, atol=1.0e-6)

    # 6 dB XPR biases MEG toward the (theta-only) polarization: XPR/(1+XPR).
    biased = mw.MultipathEnvironment(theta=basis.eep.theta, phi=basis.eep.phi, cross_polar_ratio_db=6.0)
    xpr = 10.0 ** 0.6
    expected = xpr / (1.0 + xpr)
    meg_biased = basis.mimo(biased).mean_effective_gain
    torch.testing.assert_close(meg_biased, torch.full_like(meg_biased, expected), rtol=0.0, atol=1.0e-6)


def test_orthogonal_and_identical_patterns_reach_ecc_limits():
    dtype = torch.float64
    cdtype = torch.complex128
    frequencies = torch.tensor([1.0e9], dtype=dtype)
    theta, phi = _sphere(91, 181, dtype=dtype)
    # Port a is theta-polarized, port b is phi-polarized -> orthogonal -> ECC 0.
    base = torch.sin(theta).to(cdtype)
    e_theta = torch.stack((base, torch.zeros_like(base)), dim=0)[None]
    e_phi = torch.stack((torch.zeros_like(base), base), dim=0)[None]
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=e_theta, e_phi=e_phi,
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(
            frequencies=frequencies, s=torch.zeros((1, 2, 2), dtype=cdtype), z0=50.0, port_names=("a", "b")
        ),
        embedded_patterns=patterns, fingerprint="ortho",
    )
    environment = mw.MultipathEnvironment(theta=theta, phi=phi, cross_polar_ratio_db=0.0)
    mimo = basis.mimo(environment)
    assert float(mimo.ecc[0, 0, 1]) <= ARRAY_ACCEPTANCE_BUDGET.reference_ecc_error

    # Identical co-polarized patterns -> ECC 1 and diagonal ECC exactly 1.
    identical = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=torch.stack((base, base), dim=0)[None],
        e_phi=torch.zeros((1, 2, *base.shape), dtype=cdtype),
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    same_basis = mw.ArrayBasisData(
        network=basis.network, embedded_patterns=identical, fingerprint="identical"
    )
    same = same_basis.mimo(environment)
    torch.testing.assert_close(same.ecc[0, 0, 1], torch.ones((), dtype=dtype), rtol=0.0, atol=1.0e-9)
    torch.testing.assert_close(torch.diagonal(same.ecc[0]), torch.ones(2, dtype=dtype))


def test_correlation_matrix_is_hermitian_positive_semidefinite():
    basis, _ = _isotropic_dipole_basis(0.25)
    environment = mw.MultipathEnvironment(theta=basis.eep.theta, phi=basis.eep.phi, cross_polar_ratio_db=3.0)
    correlation = basis.mimo(environment).correlation
    tolerance = 512.0 * torch.finfo(torch.float64).eps
    torch.testing.assert_close(correlation, correlation.mH, rtol=tolerance, atol=tolerance)
    eigenvalues = torch.linalg.eigvalsh(correlation)
    largest = float(torch.amax(eigenvalues))
    assert float(torch.amin(eigenvalues)) >= -1.0e-9 * largest


def test_diversity_gain_tracks_envelope_correlation():
    basis, _ = _isotropic_dipole_basis(0.15)
    environment = mw.MultipathEnvironment(theta=basis.eep.theta, phi=basis.eep.phi)
    mimo = basis.mimo(environment)
    expected = 10.0 * torch.sqrt(torch.clamp(1.0 - mimo.ecc, min=0.0))
    torch.testing.assert_close(mimo.diversity_gain, expected)


def test_scattering_ecc_approximation_is_a_distinct_named_method():
    dtype = torch.complex128
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    # Lossless-ish two-port with coupling; Blanch closed form for reference.
    s = torch.tensor([[[0.1 + 0.0j, 0.3 + 0.1j], [0.3 + 0.1j, -0.2 + 0.0j]]], dtype=dtype)
    theta, phi = _sphere(9, 17)
    base = torch.sin(theta).to(dtype)
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=torch.stack((base, base), dim=0)[None],
        e_phi=torch.zeros((1, 2, *base.shape), dtype=dtype),
        phase_center=torch.zeros(3, dtype=torch.float64), frame=torch.eye(3, dtype=torch.float64),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(frequencies=frequencies, s=s, z0=50.0, port_names=("a", "b")),
        embedded_patterns=patterns, fingerprint="s-ecc",
    )
    ecc = basis.ecc_from_scattering()
    s0 = s[0]
    numerator = abs(torch.conj(s0[0, 0]) * s0[0, 1] + torch.conj(s0[1, 0]) * s0[1, 1]) ** 2
    denominator = (
        (1.0 - abs(s0[0, 0]) ** 2 - abs(s0[1, 0]) ** 2)
        * (1.0 - abs(s0[0, 1]) ** 2 - abs(s0[1, 1]) ** 2)
    )
    torch.testing.assert_close(ecc[0, 0, 1], (numerator / denominator).to(torch.float64))


def test_scattering_ecc_rejects_non_passive_or_reflective_columns():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    # Column power sum_n |S_ni|^2 = 0.81 + 0.25 = 1.06 > 1: a non-passive column.
    # The guard rejects non-passive / fully-reflective columns; it cannot observe
    # genuine ohmic loss (invisible in S alone).
    s = torch.tensor([[[0.9 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.9 + 0.0j]]], dtype=torch.complex128)
    theta, phi = _sphere(9, 17)
    base = torch.sin(theta).to(torch.complex128)
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=torch.stack((base, base), dim=0)[None],
        e_phi=torch.zeros((1, 2, *base.shape), dtype=torch.complex128),
        phase_center=torch.zeros(3, dtype=torch.float64), frame=torch.eye(3, dtype=torch.float64),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(frequencies=frequencies, s=s, z0=50.0, port_names=("a", "b")),
        embedded_patterns=patterns, fingerprint="lossy-s",
    )
    with pytest.raises(ValueError, match="non-passive or fully-reflective columns"):
        basis.ecc_from_scattering()


def test_environment_grid_and_power_density_are_validated():
    basis, _ = _isotropic_dipole_basis(0.2)
    coarse_theta, coarse_phi = _sphere(5, 7)
    with pytest.raises(ValueError, match="must match the embedded-pattern grid"):
        basis.mimo(mw.MultipathEnvironment(theta=coarse_theta, phi=coarse_phi))
    with pytest.raises(ValueError, match="non-negative"):
        basis.mimo(
            mw.MultipathEnvironment(
                theta=basis.eep.theta, phi=basis.eep.phi,
                power_density=-torch.ones_like(basis.eep.theta),
            )
        )
    with pytest.raises(ValueError, match="magnitude must be <= 1"):
        mw.MultipathEnvironment(
            theta=basis.eep.theta, phi=basis.eep.phi, polarization_correlation=1.5
        )


def test_mimo_metrics_stay_in_the_torch_autograd_graph():
    """MIMO metrics are torch-native: a differentiable EEP flows to ECC/MEG.

    The basis stores measured (detached) patterns in production, but the metric
    kernel itself must not detach through NumPy, so a trainable pattern proves the
    autograd path is intact.
    """

    dtype = torch.float64
    cdtype = torch.complex128
    frequencies = torch.tensor([1.0e9], dtype=dtype)
    theta, phi = _sphere(9, 17, dtype=dtype)
    base = torch.sin(theta).to(cdtype)
    raw = torch.stack((base, 0.9 * base), dim=0)[None].clone().requires_grad_(True)
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=raw, e_phi=torch.zeros((1, 2, *base.shape), dtype=cdtype),
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(frequencies=frequencies, s=torch.zeros((1, 2, 2), dtype=cdtype), z0=50.0, port_names=("a", "b")),
        embedded_patterns=patterns, fingerprint="grad",
    )
    environment = mw.MultipathEnvironment(theta=theta, phi=phi, cross_polar_ratio_db=2.0)
    mimo = basis.mimo(environment)
    (mimo.ecc[0, 0, 1] + mimo.mean_effective_gain.sum()).backward()
    assert raw.grad is not None
    assert torch.all(torch.isfinite(raw.grad.real)) and torch.all(torch.isfinite(raw.grad.imag))


def test_polarization_correlation_cross_term_matches_brute_force_integral():
    """Exercise the rho != 0 dual-polarized cross term in mimo().

    Every other MIMO test uses the default rho = 0, so the two cross-polar
    coupling terms in the correlation integral (array.py) are otherwise never
    checked; a conj/sign regression there would survive Hermitian symmetrization
    and pass every existing gate. This validates the full complex-rho correlation
    matrix against an independent brute-force trapezoidal integral.
    """

    dtype = torch.float64
    cdtype = torch.complex128
    frequencies = torch.tensor([1.0e9], dtype=dtype)
    theta, phi = _sphere(61, 121, dtype=dtype)
    e_theta = torch.stack(
        [
            (torch.sin(theta) * torch.exp(1j * 0.7 * torch.cos(theta))).to(cdtype),
            (torch.sin(theta) * torch.exp(-1j * 1.3 * torch.cos(theta) + 1j * 0.4 * torch.cos(phi))).to(cdtype),
        ],
        dim=0,
    )[None]
    e_phi = torch.stack(
        [
            (0.5 * torch.sin(theta) * torch.exp(1j * 0.2 * torch.sin(phi))).to(cdtype),
            (0.8 * torch.sin(theta) * torch.exp(1j * 1.1 * torch.cos(theta))).to(cdtype),
        ],
        dim=0,
    )[None]
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=e_theta, e_phi=e_phi,
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(
            frequencies=frequencies, s=torch.zeros((1, 2, 2), dtype=cdtype), z0=50.0, port_names=("a", "b")
        ),
        embedded_patterns=patterns, fingerprint="rho-cross",
    )
    rho = 0.3 + 0.4j
    xpr_db = 5.0
    environment = mw.MultipathEnvironment(
        theta=theta, phi=phi, cross_polar_ratio_db=xpr_db, polarization_correlation=rho
    )
    correlation = basis.mimo(environment).correlation[0]

    def _trapz(values):
        deltas = values[1:] - values[:-1]
        weights = torch.empty_like(values)
        weights[0] = 0.5 * deltas[0]
        weights[-1] = 0.5 * deltas[-1]
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
        return weights

    measure = torch.sin(theta) * _trapz(theta[:, 0])[:, None] * _trapz(phi[0, :])[None, :]
    power = torch.ones_like(theta)
    power = power / torch.sum(power * measure)  # uniform spectrum, unit-normalized
    xpr = 10.0 ** (xpr_db / 10.0)
    reference = torch.zeros((2, 2), dtype=cdtype)
    for i in range(2):
        for j in range(2):
            ei_t, ej_t = e_theta[0, i], e_theta[0, j]
            ei_p, ej_p = e_phi[0, i], e_phi[0, j]
            term = (
                xpr * ei_t * ej_t.conj()
                + ei_p * ej_p.conj()
                + rho * math.sqrt(xpr) * ei_t * ej_p.conj()
                + complex(rho).conjugate() * math.sqrt(xpr) * ei_p * ej_t.conj()
            )
            reference[i, j] = torch.sum(term * power.to(cdtype) * measure.to(cdtype))

    torch.testing.assert_close(correlation, reference, atol=1e-9, rtol=1e-9)


def test_mimo_fails_closed_on_a_dark_zero_power_port():
    """A zero-power (dark) EEP column must fail closed, not return silent NaN ECC."""

    dtype = torch.float64
    cdtype = torch.complex128
    frequencies = torch.tensor([1.0e9], dtype=dtype)
    theta, phi = _sphere(9, 17, dtype=dtype)
    base = torch.sin(theta).to(cdtype)
    e_theta = torch.stack((base, torch.zeros_like(base)), dim=0)[None]
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=("a", "b"), theta=theta, phi=phi,
        e_theta=e_theta, e_phi=torch.zeros_like(e_theta),
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    basis = mw.ArrayBasisData(
        network=mw.NetworkData(
            frequencies=frequencies, s=torch.zeros((1, 2, 2), dtype=cdtype), z0=50.0, port_names=("a", "b")
        ),
        embedded_patterns=patterns, fingerprint="dark-mimo",
    )
    with pytest.raises(ValueError, match="non-positive"):
        basis.mimo(mw.MultipathEnvironment(theta=theta, phi=phi))
