import pytest
import torch

from witwin.maxwell.network import NetworkData


def _impedance_fixture(*, requires_grad=False):
    return torch.tensor(
        [
            [[62.0 + 6.0j, 5.0 - 1.0j], [4.0 + 2.0j, 58.0 - 3.0j]],
            [[68.0 + 4.0j, 6.0 + 0.5j], [5.5 - 1.5j, 61.0 + 2.0j]],
            [[72.0 - 2.0j, 7.0 + 1.0j], [6.0 + 0.25j, 66.0 + 5.0j]],
        ],
        dtype=torch.complex128,
        requires_grad=requires_grad,
    )


def _network_from_z(*, requires_grad=False):
    frequencies = torch.tensor([1.0e9, 2.0e9, 3.0e9], dtype=torch.float64)
    z0 = torch.tensor(
        [
            [50.0 + 5.0j, 75.0 - 2.0j],
            [52.0 + 4.0j, 73.0 + 1.0j],
            [55.0 - 3.0j, 70.0 + 2.0j],
        ],
        dtype=torch.complex128,
    )
    z = _impedance_fixture(requires_grad=requires_grad)
    network = NetworkData.from_z(
        frequencies=frequencies,
        z=z,
        z0=z0,
        port_names=("p1", "p2"),
    )
    return network, z


def test_network_data_freezes_frequency_port_and_matrix_shapes():
    network, _ = _network_from_z()

    assert network.frequencies.shape == (3,)
    assert network.s.shape == (3, 2, 2)
    assert network.z0.shape == (3, 2)
    assert network.port_names == ("p1", "p2")
    assert network.valid_columns.dtype == torch.bool
    assert network.valid_columns.shape == (2,)
    assert bool(torch.all(network.valid_columns))
    assert network.is_complete


def test_network_data_s_z_y_round_trips_with_complex_per_port_z0():
    network, expected_z = _network_from_z()

    actual_z = network.to_z()
    actual_y = network.to_y()
    from_y = NetworkData.from_y(
        frequencies=network.frequencies,
        y=actual_y,
        z0=network.z0,
        port_names=network.port_names,
    )

    torch.testing.assert_close(actual_z, expected_z, rtol=1.0e-11, atol=1.0e-11)
    torch.testing.assert_close(
        torch.linalg.solve(actual_y, torch.eye(2, dtype=actual_y.dtype).expand(3, 2, 2)),
        expected_z,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(from_y.s, network.s, rtol=1.0e-11, atol=1.0e-11)


def test_network_renormalization_preserves_physical_impedance_and_autograd():
    network, source_z = _network_from_z(requires_grad=True)
    new_z0 = torch.tensor(
        [[45.0 + 2.0j, 80.0 - 4.0j]],
        dtype=torch.complex128,
    ).expand(3, 2)

    renormalized = network.renormalize(new_z0)

    torch.testing.assert_close(renormalized.z0, new_z0)
    torch.testing.assert_close(renormalized.to_z(), source_z, rtol=1.0e-11, atol=1.0e-11)
    assert renormalized is not network

    loss = renormalized.s.abs().square().sum() + renormalized.to_y().abs().sum()
    loss.backward()
    assert source_z.grad is not None
    assert torch.all(torch.isfinite(source_z.grad))


def test_network_return_loss_and_vswr_use_diagonal_reflections():
    frequencies = torch.tensor([1.0, 2.0], dtype=torch.float64)
    s = torch.zeros((2, 2, 2), dtype=torch.complex128)
    s[:, 0, 0] = torch.tensor([0.1 + 0.0j, 0.2 + 0.0j], dtype=torch.complex128)
    s[:, 1, 1] = torch.tensor([0.25 + 0.0j, 1.0 + 0.0j], dtype=torch.complex128)
    network = NetworkData(
        frequencies=frequencies,
        s=s,
        z0=50.0,
        port_names=("p1", "p2"),
    )

    reflection = torch.diagonal(s, dim1=-2, dim2=-1)
    torch.testing.assert_close(network.reflection_coefficient, reflection)
    torch.testing.assert_close(network.return_loss_db, -20.0 * torch.log10(reflection.abs()))
    assert network.vswr[0, 0] == pytest.approx(1.1 / 0.9)
    assert network.vswr[0, 1] == pytest.approx(1.25 / 0.75)
    assert torch.isinf(network.vswr[1, 1])


def test_network_incomplete_columns_cannot_be_used_as_a_complete_matrix():
    frequencies = torch.tensor([1.0, 2.0], dtype=torch.float64)
    network = NetworkData(
        frequencies=frequencies,
        s=torch.zeros((2, 2, 2), dtype=torch.complex128),
        z0=50.0,
        port_names=("p1", "p2"),
        valid_columns=torch.tensor([True, False]),
    )

    assert not network.is_complete
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.to_z()
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.to_y()
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.renormalize(75.0)


def test_network_data_validates_shapes_names_dtypes_and_reference_impedance():
    frequencies = torch.tensor([1.0, 2.0], dtype=torch.float64)
    s = torch.zeros((2, 2, 2), dtype=torch.complex128)

    with pytest.raises(ValueError, match=r"\[F, N, N\]"):
        NetworkData(
            frequencies=frequencies,
            s=torch.zeros((2, 2), dtype=torch.complex128),
            z0=50.0,
            port_names=("p1", "p2"),
        )
    with pytest.raises(ValueError, match="unique"):
        NetworkData(frequencies=frequencies, s=s, z0=50.0, port_names=("p1", "p1"))
    with pytest.raises(TypeError, match="complex"):
        NetworkData(frequencies=frequencies, s=torch.zeros((2, 2, 2)), z0=50.0, port_names=("p1", "p2"))
    with pytest.raises(ValueError, match=r"Re\(z0\) must be strictly positive"):
        NetworkData(frequencies=frequencies, s=s, z0=-50.0 + 1.0j, port_names=("p1", "p2"))
    with pytest.raises(ValueError, match=r"shape \[F, N\]"):
        NetworkData(
            frequencies=frequencies,
            s=s,
            z0=torch.ones((2, 2, 1), dtype=torch.complex128),
            port_names=("p1", "p2"),
        )


def test_network_scalar_reference_impedance_is_materialized_as_f_by_n():
    frequencies = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    network = NetworkData(
        frequencies=frequencies,
        s=torch.zeros((3, 2, 2), dtype=torch.complex64),
        z0=50.0,
        port_names=("p1", "p2"),
    )

    assert network.z0.shape == (3, 2)
    assert network.z0.dtype == torch.complex64
    torch.testing.assert_close(network.z0, torch.full((3, 2), 50.0 + 0.0j, dtype=torch.complex64))
