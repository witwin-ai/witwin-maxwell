import pytest
import torch

from witwin.maxwell.network import NetworkData


def _frequencies(count=2):
    return torch.linspace(1.0e9, float(count) * 1.0e9, count, dtype=torch.float64)


def test_lossless_thru_power_balance_is_below_two_percent():
    # Gate class (S0.3): tautology. Power balance is asserted on a hand-written
    # unitary S matrix (zero solve content); it validates the algebra only and is
    # NOT a wave-level exit gate.
    phase = torch.tensor((0.2, 0.7), dtype=torch.float64)
    transmission = torch.exp(-1j * phase)
    scattering = torch.zeros((2, 2, 2), dtype=torch.complex128)
    scattering[:, 0, 1] = transmission
    scattering[:, 1, 0] = transmission
    network = NetworkData(
        frequencies=_frequencies(),
        s=scattering,
        z0=50.0,
        port_names=("p1", "p2"),
    )

    output_power = network.s.abs().square().sum(dim=1)
    imbalance = torch.abs(1.0 - output_power)

    assert torch.max(imbalance) < 0.02


def test_shift_reference_planes_applies_two_one_way_factors_and_preserves_autograd():
    s = torch.tensor(
        [
            [[0.1 + 0.2j, 0.7 - 0.1j], [0.6 + 0.05j, -0.2 + 0.1j]],
            [[0.2 - 0.1j, 0.5 + 0.2j], [0.4 - 0.1j, 0.05 + 0.15j]],
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    network = NetworkData(
        frequencies=_frequencies(),
        s=s,
        z0=50.0,
        port_names=("p1", "p2"),
    )
    distances = torch.tensor([0.01, -0.02], dtype=torch.float64)
    propagation = torch.tensor(
        [[20.0 + 0.2j, 30.0 + 0.3j], [40.0 + 0.4j, 50.0 + 0.5j]],
        dtype=torch.complex128,
    )

    shifted = network.shift_reference_planes(
        distances,
        propagation_constants=propagation,
    )

    one_way = torch.exp(1j * propagation * distances)
    expected = one_way.unsqueeze(-1) * s * one_way.unsqueeze(-2)
    torch.testing.assert_close(shifted.s, expected)
    torch.testing.assert_close(shifted.z0, network.z0)
    assert shifted.port_names == network.port_names
    assert shifted.metadata["network_transform_history"][-1]["operation"] == (
        "shift_reference_planes"
    )

    shifted.s.abs().square().sum().backward()
    assert s.grad is not None
    assert torch.all(torch.isfinite(s.grad))


def test_mixed_mode_uses_power_preserving_voltage_basis_and_default_z0():
    z = torch.tensor(
        [
            [
                [60.0 + 4.0j, 8.0 - 1.0j, 2.0 + 0.5j],
                [7.0 + 2.0j, 55.0 - 3.0j, 3.0 - 0.5j],
                [1.5 + 0.2j, 2.5 - 0.3j, 70.0 + 1.0j],
            ],
            [
                [62.0 + 3.0j, 9.0 - 2.0j, 1.0 + 0.3j],
                [8.0 + 1.0j, 57.0 - 2.0j, 2.0 - 0.2j],
                [1.2 + 0.1j, 2.2 - 0.1j, 72.0 + 2.0j],
            ],
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    network = NetworkData.from_z(
        frequencies=_frequencies(),
        z=z,
        z0=torch.tensor([50.0, 50.0, 75.0], dtype=torch.complex128),
        port_names=("p", "n", "aux"),
    )

    mixed = network.to_mixed_mode([("p", "n")])

    basis = torch.tensor(
        [[1.0, -1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.complex128,
    )
    expected_z = basis @ z @ basis.transpose(-2, -1)
    torch.testing.assert_close(mixed.to_z(), expected_z, rtol=1.0e-11, atol=1.0e-11)
    torch.testing.assert_close(
        mixed.z0,
        torch.tensor([100.0, 25.0, 75.0], dtype=torch.complex128)
        .unsqueeze(0)
        .expand(2, 3),
    )
    assert mixed.port_names == ("d(p,n)", "c(p,n)", "aux")

    mixed.s.abs().square().sum().backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


def test_mixed_mode_requires_equal_pair_z0_unless_explicit_reference_is_given():
    network = NetworkData(
        frequencies=_frequencies(),
        s=torch.zeros((2, 2, 2), dtype=torch.complex128),
        z0=torch.tensor([50.0, 75.0], dtype=torch.complex128),
        port_names=("p", "n"),
    )

    with pytest.raises(ValueError, match="must be equal"):
        network.to_mixed_mode([("p", "n")])

    mixed = network.to_mixed_mode([("p", "n")], z0=(100.0, 30.0))
    torch.testing.assert_close(
        mixed.z0,
        torch.tensor([[100.0, 30.0], [100.0, 30.0]], dtype=torch.complex128),
    )


def test_full_matrix_transforms_reject_incomplete_excitation_columns():
    network = NetworkData(
        frequencies=_frequencies(),
        s=torch.zeros((2, 2, 2), dtype=torch.complex128),
        z0=50.0,
        port_names=("p", "n"),
        valid_columns=torch.tensor([True, False]),
    )

    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.shift_reference_planes([0.0, 0.0], propagation_constants=1.0)
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.to_mixed_mode([("p", "n")])


def test_renormalize_records_source_and_target_reference_impedances():
    network = NetworkData(
        frequencies=_frequencies(),
        s=torch.zeros((2, 2, 2), dtype=torch.complex128),
        z0=50.0,
        port_names=("p1", "p2"),
        metadata={"case": "fixture"},
    )

    renormalized = network.renormalize((75.0, 100.0))

    assert renormalized.metadata["case"] == "fixture"
    record = renormalized.metadata["network_transform_history"][-1]
    assert record["operation"] == "renormalize"
    torch.testing.assert_close(record["source_z0"], network.z0)
    torch.testing.assert_close(record["target_z0"], renormalized.z0)
    assert not record["source_z0"].requires_grad
    assert not record["target_z0"].requires_grad


def test_network_algebra_reports_nonfinite_and_ill_conditioned_inputs():
    frequencies = torch.tensor([1.0], dtype=torch.float64)
    nonfinite = torch.zeros((1, 2, 2), dtype=torch.complex128)
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        NetworkData(
            frequencies=frequencies,
            s=nonfinite,
            z0=50.0,
            port_names=("p1", "p2"),
        )

    singular_s = NetworkData(
        frequencies=frequencies,
        s=torch.eye(2, dtype=torch.complex128).unsqueeze(0),
        z0=50.0,
        port_names=("p1", "p2"),
    )
    with pytest.raises(RuntimeError, match=r"ill-conditioned.*frequency indices \[0\]"):
        singular_s.to_z()

    singular_y = -torch.eye(2, dtype=torch.complex128).unsqueeze(0) / 50.0
    with pytest.raises(RuntimeError, match="Y/S conversion.*ill-conditioned"):
        NetworkData.from_y(
            frequencies=frequencies,
            y=singular_y,
            z0=50.0,
            port_names=("p1", "p2"),
        )


def test_exact_open_converts_directly_between_s_and_y():
    frequencies = _frequencies(count=3)
    z0 = torch.tensor(
        [
            [40.0 + 5.0j, 70.0 - 3.0j],
            [45.0 - 2.0j, 75.0 + 4.0j],
            [50.0 + 1.0j, 80.0 - 6.0j],
        ],
        dtype=torch.complex128,
    )
    identity = torch.eye(2, dtype=torch.complex128).expand(3, 2, 2)
    network = NetworkData(
        frequencies=frequencies,
        s=identity,
        z0=z0,
        port_names=("p1", "p2"),
    )

    admittance = network.to_y()
    restored = NetworkData.from_y(
        frequencies=frequencies,
        y=admittance,
        z0=z0,
        port_names=network.port_names,
    )

    torch.testing.assert_close(admittance, torch.zeros_like(admittance), rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored.s, identity, rtol=0.0, atol=0.0)


def test_direct_y_round_trip_with_complex_z0_preserves_autograd():
    generator = torch.Generator().manual_seed(20260715)
    real = 0.002 * torch.randn((3, 3, 3), generator=generator, dtype=torch.float64)
    imag = 0.002 * torch.randn((3, 3, 3), generator=generator, dtype=torch.float64)
    admittance = torch.complex(real, imag).requires_grad_()
    z0 = torch.tensor(
        [
            [35.0 + 4.0j, 50.0 - 3.0j, 80.0 + 2.0j],
            [38.0 - 2.0j, 55.0 + 5.0j, 75.0 - 4.0j],
            [42.0 + 1.0j, 60.0 - 6.0j, 70.0 + 3.0j],
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    network = NetworkData.from_y(
        frequencies=_frequencies(count=3),
        y=admittance,
        z0=z0,
        port_names=("p1", "p2", "p3"),
    )

    torch.testing.assert_close(network.to_y(), admittance, rtol=1.0e-11, atol=1.0e-12)

    network.s.abs().square().sum().backward()
    assert admittance.grad is not None
    assert z0.grad is not None
    assert torch.all(torch.isfinite(admittance.grad))
    assert torch.all(torch.isfinite(z0.grad))
    assert torch.count_nonzero(admittance.grad) > 0
    assert torch.count_nonzero(z0.grad) > 0
