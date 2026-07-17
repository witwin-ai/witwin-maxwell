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


def test_cache_key_is_weight_invariant_but_content_sensitive():
    basis = _basis()
    key = basis.cache_key
    # Any weight vector reuses the same basis: combine never changes the key.
    for weights in (
        torch.ones(2, dtype=torch.complex128),
        torch.tensor([0.3 + 0.9j, -0.7 + 0.1j], dtype=torch.complex128),
    ):
        assert basis.combine(weights).metadata["basis_fingerprint"] == key
    # A different embedded pattern (physical content change) must invalidate.
    shifted = mw.EmbeddedElementPatternData(
        frequencies=basis.frequencies, port_names=basis.port_names,
        theta=basis.eep.theta, phi=basis.eep.phi,
        e_theta=basis.eep.e_theta * 2.0, e_phi=basis.eep.e_phi,
        phase_center=basis.eep.phase_center, frame=basis.eep.frame,
    )
    other = mw.ArrayBasisData(
        network=basis.network, embedded_patterns=shifted, fingerprint="codebook-basis-shifted"
    )
    assert other.cache_key != key


def test_scene_gradient_through_basis_fails_closed():
    basis = _basis()
    with pytest.raises(NotImplementedError, match="aggregated per-column adjoint envelope"):
        basis.scene_gradient_vjp()
