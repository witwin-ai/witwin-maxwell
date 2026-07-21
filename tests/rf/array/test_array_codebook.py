import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.solver import FDTD


def _basis(*, device="cpu", dtype=torch.complex128, port_count=2):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    frequencies = torch.tensor([1.0e9, 1.5e9], device=device, dtype=real_dtype)
    scattering = torch.zeros((2, port_count, port_count), device=device, dtype=dtype)
    for index in range(port_count):
        scattering[:, index, index] = 0.08 - 0.03j
    z0 = torch.full((port_count,), 50.0, device=device, dtype=dtype)
    network = mw.NetworkData(
        frequencies=frequencies, s=scattering, z0=z0,
        port_names=tuple(f"p{index + 1}" for index in range(port_count)),
    )
    theta_vector = torch.linspace(0.0, math.pi, 9, device=device, dtype=real_dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, 17, device=device, dtype=real_dtype)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    base = torch.sin(theta).to(dtype)
    e_theta = torch.stack(
        [base * torch.exp(torch.tensor(0.1j * index, device=device, dtype=dtype)) for index in range(port_count)],
        dim=0,
    )[None].expand(2, -1, -1, -1).clone()
    e_phi = 0.2j * e_theta
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies, port_names=network.port_names, theta=theta, phi=phi,
        e_theta=e_theta, e_phi=e_phi,
        phase_center=torch.zeros(3, device=device, dtype=real_dtype),
        frame=torch.eye(3, device=device, dtype=real_dtype),
    )
    return mw.ArrayBasisData(network=network, embedded_patterns=patterns, fingerprint="codebook-basis")


def test_codebook_combine_matches_per_beam_combine():
    basis = _basis()
    weights = torch.stack(
        [
            torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j], [0.5 + 0.5j, -0.5 + 0.5j]], dtype=torch.complex128),
            torch.tensor([[0.3 - 0.2j, 0.7 + 0.1j], [-0.4 + 0.6j, 0.2 + 0.2j]], dtype=torch.complex128),
        ]
    )
    codebook = mw.BeamCodebook(weights=weights, names=("main", "steered"))

    beams = basis.combine(codebook)

    assert beams.names == ("main", "steered")
    assert beams.metadata["codebook"] is True and beams.metadata["beam_count"] == 2
    for index in range(2):
        single = basis.combine(weights[index])
        torch.testing.assert_close(beams.network.b[index], single.network.b)
        torch.testing.assert_close(beams.far_field.e_theta[index], single.far_field.e_theta)
        torch.testing.assert_close(beams.antenna.realized_gain[index], single.antenna.realized_gain)


def test_frequency_flat_codebook_broadcasts_over_frequency():
    basis = _basis()
    flat = torch.tensor([[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.complex128)
    codebook = mw.BeamCodebook(weights=flat, names=("b0", "b1"))

    beams = basis.combine(codebook)

    assert beams.network.a.shape == (2, 2, 2)
    # frequency-flat weights repeat across the frequency axis.
    torch.testing.assert_close(beams.network.a[:, 0, :], beams.network.a[:, 1, :])


def test_codebook_rejects_bad_shape_and_duplicate_names():
    basis = _basis()
    with pytest.raises(ValueError, match="one non-empty name per beam"):
        mw.BeamCodebook(weights=torch.ones((3, 2), dtype=torch.complex128), names=("a", "b"))
    with pytest.raises(ValueError, match="unique"):
        mw.BeamCodebook(weights=torch.ones((2, 2), dtype=torch.complex128), names=("a", "a"))
    with pytest.raises(TypeError, match="complex"):
        mw.BeamCodebook(weights=torch.ones((2, 2), dtype=torch.float64), names=("a", "b"))
    wrong_ports = mw.BeamCodebook(weights=torch.ones((2, 3), dtype=torch.complex128), names=("a", "b"))
    with pytest.raises(ValueError, match=r"\[B, N\] with N=2"):
        basis.combine(wrong_ports)


@pytest.mark.parametrize("beam_count", (64, 256, 1024))
def test_combining_any_number_of_beams_executes_zero_fdtd_steps(beam_count, monkeypatch):
    """Hard contract: codebook combination runs no field-solver time steps.

    Instrument the FDTD time-stepping entry with a step-summing fingerprint and
    assert combining 64/256/1024 beams (and taking a max-hold envelope) adds zero
    steps. Falsification (recorded 2026-07-17): temporarily calling ``FDTD.solve``
    inside ``combine`` drives the delta to a positive step count and reddens this
    test; the guard restores it.
    """

    basis = _basis()
    executed = {"steps": 0}
    original_solve = FDTD.solve

    def counting_solve(self, time_steps, *args, **kwargs):
        executed["steps"] += int(time_steps)
        return original_solve(self, time_steps, *args, **kwargs)

    monkeypatch.setattr(FDTD, "solve", counting_solve)
    fingerprint_before = executed["steps"]

    weights = torch.randn(beam_count, 2, 2, dtype=torch.complex128)
    weights = weights / torch.clamp(torch.abs(weights), min=1e-3)
    codebook = mw.BeamCodebook(weights=weights, names=tuple(f"b{i}" for i in range(beam_count)))
    beams = basis.combine(codebook)
    envelope = beams.max_hold("realized_gain")

    assert executed["steps"] - fingerprint_before == 0
    assert beams.weights.shape == (beam_count, 2, 2)
    assert envelope.envelope.shape == (2, 9, 17)
    assert envelope.winning_beam.shape == (2, 9, 17)


def test_max_hold_envelope_and_argmax_are_consistent():
    basis = _basis()
    weights = torch.stack(
        [
            torch.tensor([[2.0 + 0.0j, 0.0 + 0.0j]] * 2, dtype=torch.complex128),
            torch.tensor([[0.0 + 0.0j, 2.0 + 0.0j]] * 2, dtype=torch.complex128),
        ]
    )
    beams = basis.combine(mw.BeamCodebook(weights=weights, names=("port1", "port2")))

    envelope = beams.max_hold("realized_gain")

    per_beam = beams.antenna.realized_gain
    torch.testing.assert_close(envelope.envelope, torch.amax(per_beam, dim=0))
    torch.testing.assert_close(envelope.winning_beam, torch.argmax(per_beam, dim=0))
    # eirp envelope reduces the scalar-per-beam metric.
    eirp_envelope = beams.max_hold("eirp")
    assert eirp_envelope.envelope.shape == (2,)


def test_max_hold_requires_batch_and_rejects_masked_metric():
    basis = _basis()
    with pytest.raises(ValueError, match="batched beam result"):
        basis.combine(torch.ones(2, dtype=torch.complex128)).max_hold("realized_gain")
    # A dark, fully reflected beam yields a NaN gain; max-hold must refuse it.
    reflective = mw.NetworkData(
        frequencies=basis.frequencies,
        s=torch.eye(2, dtype=torch.complex128)[None].expand(2, -1, -1).clone(),
        z0=basis.network.z0,
        port_names=basis.port_names,
    )
    dark = mw.ArrayBasisData(
        network=reflective, embedded_patterns=basis.eep, fingerprint="dark"
    )
    codebook = mw.BeamCodebook(
        weights=torch.ones((2, 2), dtype=torch.complex128), names=("a", "b")
    )
    with pytest.raises(ValueError, match="masked"):
        dark.combine(codebook).max_hold("gain")


def test_from_scan_angles_builds_progressive_phase_and_steers_broadside():
    device = torch.device("cpu")
    frequency = torch.tensor([1.0e9, 1.5e9], dtype=torch.float64, device=device)
    positions = torch.tensor([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=torch.float64)
    theta = torch.tensor([math.pi / 2, math.pi / 3], dtype=torch.float64)
    phi = torch.tensor([0.0, math.pi / 4], dtype=torch.float64)

    codebook = mw.BeamCodebook.from_scan_angles(
        element_positions=positions, frequencies=frequency, theta=theta, phi=phi
    )

    assert codebook.weights.shape == (2, 2, 2)
    assert codebook.target_angles.shape == (2, 2)
    # Broadside scan (theta=pi/2, phi=0 -> direction +x) at spacing 0.1 m.
    wave_number = 2.0 * math.pi * frequency / 299792458.0
    expected_phase = wave_number[:, None] * torch.tensor([-0.05, 0.05], dtype=torch.float64)[None, :]
    torch.testing.assert_close(torch.angle(codebook.weights[0]), expected_phase)


def test_from_scan_angles_rejects_bad_positions_and_mismatched_angles():
    frequency = torch.tensor([1.0e9], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        mw.BeamCodebook.from_scan_angles(
            element_positions=torch.zeros((2, 2)),
            frequencies=frequency,
            theta=torch.tensor([0.0]),
            phi=torch.tensor([0.0]),
        )
    with pytest.raises(ValueError, match="equal length"):
        mw.BeamCodebook.from_scan_angles(
            element_positions=torch.zeros((2, 3), dtype=torch.float64),
            frequencies=frequency,
            theta=torch.tensor([0.0, 1.0], dtype=torch.float64),
            phi=torch.tensor([0.0], dtype=torch.float64),
        )


def test_cache_key_is_weight_invariant():
    basis = _basis()
    key = basis.cache_key
    # Any weight vector reuses the same basis: combine never changes the key.
    for weights in (
        torch.ones(2, dtype=torch.complex128),
        torch.tensor([0.3 + 0.9j, -0.7 + 0.1j], dtype=torch.complex128),
    ):
        assert basis.combine(weights).metadata["basis_fingerprint"] == key
    # Content sensitivity of the extraction-time fingerprint (physical geometry /
    # material / port / frequency / surface changes invalidating the key) is
    # covered against real extracted bases in test_array_fullwave.py; asserting it
    # here on a manually supplied fingerprint string would only compare literals.


def test_scene_gradient_through_basis_aggregates_live_columns():
    """The former fail-closed guard is now the aggregated per-column VJP.

    ``scene_gradient_vjp`` requires *live* embedded-pattern columns (the retained
    basis stores them detached) and returns a finite scene gradient. Full
    equivalence to autograd, central-difference agreement, and the FDTD end-to-end
    gate live in ``test_array_scene_gradient.py``; here we only pin that the
    method is wired and still fails closed on detached columns.
    """

    basis = _basis()
    frequency_count, port_count, _ = basis.network.s.shape
    theta_shape = basis.embedded_patterns.theta.shape
    weights = torch.ones(port_count, dtype=torch.complex128)

    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = [
        (
            basis.embedded_patterns.e_theta[:, index] * (1.0 + 0.2 * param.to(torch.complex128)),
            basis.embedded_patterns.e_phi[:, index] * torch.exp(1j * param.to(torch.complex128)),
        )
        for index in range(port_count)
    ]
    gradient = basis.scene_gradient_vjp(
        columns=columns,
        weights=weights,
        parameters=param,
        objective=lambda e_theta, e_phi: (e_theta * e_theta.conj()).real.sum()
        + (e_phi * e_phi.conj()).real.sum(),
    )
    assert gradient.shape == ()
    assert torch.isfinite(gradient)

    detached = [(e_theta.detach(), e_phi.detach()) for e_theta, e_phi in columns]
    with pytest.raises(ValueError, match="stores detached patterns"):
        basis.scene_gradient_vjp(
            columns=detached,
            weights=weights,
            parameters=param,
            objective=lambda e_theta, e_phi: e_theta.real.sum(),
        )


def _z_axis_isotropic_basis(*, spacing_wavelengths=0.25, port_count=4, points_theta=37, points_phi=5):
    """Isotropic sensors on the z-axis whose EEP columns carry the receive phase.

    Column ``n`` is ``exp(-j k z_n cos(theta))`` (isotropic magnitude), so the
    physical steering behaviour of :meth:`BeamCodebook.from_scan_angles` can be
    checked against the combined array factor rather than by recomputing the
    weight formula.
    """

    dtype, cdtype = torch.float64, torch.complex128
    frequency = torch.tensor([1.0e9], dtype=dtype)
    wave_number = float(2.0 * math.pi * frequency.item() / 299792458.0)
    wavelength = 299792458.0 / float(frequency.item())
    spacing = spacing_wavelengths * wavelength
    offsets = (torch.arange(port_count, dtype=dtype) - 0.5 * (port_count - 1)) * spacing
    positions = torch.stack((torch.zeros_like(offsets), torch.zeros_like(offsets), offsets), dim=-1)
    theta_vector = torch.linspace(0.0, math.pi, points_theta, dtype=dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, points_phi, dtype=dtype)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    columns = [torch.exp(-1j * wave_number * z * torch.cos(theta)).to(cdtype) for z in offsets]
    e_theta = torch.stack(columns, dim=0)[None]
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequency, port_names=tuple(f"e{i}" for i in range(port_count)),
        theta=theta, phi=phi, e_theta=e_theta, e_phi=torch.zeros_like(e_theta),
        phase_center=torch.zeros(3, dtype=dtype), frame=torch.eye(3, dtype=dtype),
    )
    network = mw.NetworkData(
        frequencies=frequency, s=torch.zeros((1, port_count, port_count), dtype=cdtype),
        z0=50.0, port_names=patterns.port_names,
    )
    basis = mw.ArrayBasisData(network=network, embedded_patterns=patterns, fingerprint="scan-peak")
    return basis, positions, theta_vector


def test_from_scan_angles_beam_peaks_at_the_commanded_direction():
    """Non-circular steering check: the combined pattern peaks at the target angle.

    Builds progressive-phase weights with ``from_scan_angles`` and combines them
    against an independent z-axis isotropic basis; the realized-gain maximum must
    fall on the commanded scan angle, proving the sign/phase convention steers the
    beam (not merely that the weight formula reproduces itself).
    """

    basis, positions, theta_vector = _z_axis_isotropic_basis()
    target_theta_index = 12  # off broadside and away from the poles (~60 deg)
    target_theta = theta_vector[target_theta_index][None]
    target_phi = torch.zeros(1, dtype=torch.float64)

    codebook = mw.BeamCodebook.from_scan_angles(
        element_positions=positions,
        frequencies=basis.frequencies,
        theta=target_theta,
        phi=target_phi,
    )
    realized_gain = basis.combine(codebook).antenna.realized_gain[0, 0]  # [theta, phi]
    # A z-axis array pattern is rotationally symmetric in phi; reduce over phi.
    peak_theta_index = int(torch.argmax(realized_gain.amax(dim=1)))
    assert peak_theta_index == target_theta_index


def test_codebook_weight_gradient_backprops_through_max_hold():
    """Rank-3 [B, F, N] codebook weights are differentiable through the envelope.

    The batched codebook path feeds ``max_hold('realized_gain').envelope.sum()``;
    the incident power-wave weights must receive finite, nonzero real and
    imaginary gradients with zero solver reruns. Falsification (recorded
    2026-07-17): detaching the normalized weights inside ``combine`` severs the
    graph so ``backward`` raises 'does not require grad and does not have a
    grad_fn'; restoring the path returns finite grads.
    """

    basis = _basis()
    torch.manual_seed(0)
    weights = torch.randn(3, 2, 2, dtype=torch.complex128)
    weights = weights / torch.clamp(torch.abs(weights), min=0.5)
    weights = weights.requires_grad_(True)
    codebook = mw.BeamCodebook(weights=weights, names=("b0", "b1", "b2"))

    loss = basis.combine(codebook).max_hold("realized_gain").envelope.sum()
    loss.backward()

    assert weights.grad is not None
    assert torch.all(torch.isfinite(weights.grad.real))
    assert torch.all(torch.isfinite(weights.grad.imag))
    assert weights.grad.real.abs().sum() > 0.0
    assert weights.grad.imag.abs().sum() > 0.0


def test_codebook_weight_gradient_matches_high_precision_gradcheck():
    """High-precision gate for the batched codebook -> max_hold envelope path.

    Mirrors test_array_contracts.py conventions (complex128, eps 1e-6, atol 1e-5,
    rtol 0.02): autograd must agree with a finite-difference reference on the
    subgradient-carrying ``realized_gain`` envelope of a rank-3 codebook.
    """

    basis = _basis()
    torch.manual_seed(1)
    weights = (torch.randn(3, 2, 2, dtype=torch.complex128) + 2.0).requires_grad_(True)
    names = ("b0", "b1", "b2")

    def envelope_sum(value):
        codebook = mw.BeamCodebook(weights=value, names=names)
        return basis.combine(codebook).max_hold("realized_gain").envelope.sum()

    assert torch.autograd.gradcheck(
        envelope_sum,
        (weights,),
        eps=1e-6,
        atol=1e-5,
        rtol=0.02,
    )
