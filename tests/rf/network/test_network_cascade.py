"""Analytic identity gates for the public cascade/termination helpers.

These are pure S-parameter algebra checks (no field solve). They verify the
first-principles connection algebra against closed-form references:

- cascading a two-port through an ideal matched thru returns the original;
- cascading two attenuators adds their attenuation in dB;
- terminating a two-port matches the closed-form input reflection
  ``S11 + S12 * gamma * S21 / (1 - S22 * gamma)``.
"""

import pytest
import torch

from witwin.maxwell.network import NetworkData


def _frequencies(count: int = 3) -> torch.Tensor:
    return torch.linspace(1.0e9, float(count) * 1.0e9, count, dtype=torch.float64)


def _thru(frequencies: torch.Tensor, port_names=("t1", "t2")) -> NetworkData:
    scattering = torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128)
    scattering[:, 0, 1] = 1.0
    scattering[:, 1, 0] = 1.0
    return NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=port_names,
    )


def _attenuator(frequencies: torch.Tensor, t: complex, port_names) -> NetworkData:
    scattering = torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128)
    scattering[:, 0, 1] = t
    scattering[:, 1, 0] = t
    return NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=port_names,
    )


def _random_two_port(frequencies: torch.Tensor, *, requires_grad: bool = False) -> NetworkData:
    generator = torch.Generator().manual_seed(4711)
    real = 0.3 * torch.randn((frequencies.numel(), 2, 2), generator=generator, dtype=torch.float64)
    imag = 0.3 * torch.randn((frequencies.numel(), 2, 2), generator=generator, dtype=torch.float64)
    scattering = torch.complex(real, imag)
    if requires_grad:
        scattering = scattering.requires_grad_()
    return NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("a", "b"),
    )


def test_cascade_through_matched_thru_returns_identity():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    thru = _thru(frequencies)

    cascaded = device.cascade(thru, port_map={"b": "t1"})

    assert cascaded.port_names == ("a", "t2")
    torch.testing.assert_close(cascaded.s, device.s, rtol=1.0e-12, atol=1.0e-13)
    assert cascaded.metadata["network_transform_history"][-1]["operation"] == "cascade"


def test_cascade_of_two_attenuators_adds_attenuation_in_db():
    frequencies = _frequencies()
    t1 = 0.5 + 0.0j
    t2 = 0.25 + 0.0j
    first = _attenuator(frequencies, t1, ("in", "mid"))
    second = _attenuator(frequencies, t2, ("mid2", "out"))

    cascaded = first.cascade(second, port_map={"mid": "mid2"})

    assert cascaded.port_names == ("in", "out")
    transmission = cascaded.s[:, 1, 0]
    torch.testing.assert_close(
        transmission,
        torch.full_like(transmission, t1 * t2),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    db_total = -20.0 * torch.log10(torch.abs(transmission))
    db_first = -20.0 * torch.log10(torch.tensor(abs(t1), dtype=torch.float64))
    db_second = -20.0 * torch.log10(torch.tensor(abs(t2), dtype=torch.float64))
    torch.testing.assert_close(
        db_total,
        torch.full_like(db_total, float(db_first + db_second)),
        rtol=1.0e-12,
        atol=1.0e-11,
    )
    # Reflection at both open external ports stays zero for ideal attenuators.
    torch.testing.assert_close(
        torch.diagonal(cascaded.s, dim1=-2, dim2=-1),
        torch.zeros((frequencies.numel(), 2), dtype=torch.complex128),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_terminate_two_port_matches_closed_form_input_reflection():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    gamma = torch.tensor(0.3 - 0.4j, dtype=torch.complex128)

    terminated = device.terminate("b", gamma=gamma)

    s = device.s
    expected = s[:, 0, 0] + s[:, 0, 1] * gamma * s[:, 1, 0] / (1.0 - s[:, 1, 1] * gamma)
    assert terminated.port_names == ("a",)
    torch.testing.assert_close(terminated.s[:, 0, 0], expected, rtol=1.0e-12, atol=1.0e-13)


def test_terminate_by_impedance_uses_reference_plane_reflection():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    load = torch.tensor(75.0, dtype=torch.complex128)
    gamma = (load - 50.0) / (load + 50.0)

    by_impedance = device.terminate("b", impedance=load)
    by_gamma = device.terminate("b", gamma=gamma)

    torch.testing.assert_close(by_impedance.s, by_gamma.s, rtol=1.0e-12, atol=1.0e-13)


def test_terminate_matched_load_gives_s11_of_two_port():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)

    matched = device.terminate("b", gamma=0.0)

    torch.testing.assert_close(
        matched.s[:, 0, 0], device.s[:, 0, 0], rtol=1.0e-12, atol=1.0e-13
    )


def test_cascade_three_port_double_connection_matches_terminated_reference():
    # Connecting a two-port network across ports 1 and 2 of a three-port device
    # (double connection) must equal: cascade the two-port onto port 1, then feed
    # its output into port 2 of the device and close the loop. Cross-check the
    # general double-connection against a two-step single-connection composition.
    frequencies = _frequencies(count=4)
    generator = torch.Generator().manual_seed(2027)
    real = 0.25 * torch.randn((frequencies.numel(), 3, 3), generator=generator, dtype=torch.float64)
    imag = 0.25 * torch.randn((frequencies.numel(), 3, 3), generator=generator, dtype=torch.float64)
    device = NetworkData(
        frequencies=frequencies,
        s=torch.complex(real, imag),
        z0=50.0,
        port_names=("p0", "p1", "p2"),
    )
    net = _attenuator(frequencies, 0.6 + 0.1j, ("n0", "n1"))

    # Single-shot double connection: device.p1<->n0 and device.p2<->n1.
    joined = device.cascade(net, port_map={"p1": "n0", "p2": "n1"})
    assert joined.port_names == ("p0",)

    # Two-step reference: first connect n1 to device.p2 (leaves n0 and p0, p1),
    # then connect the remaining n0 to p1. Independent path through the algebra.
    step1 = device.cascade(net, port_map={"p2": "n1"})  # ports: p0, p1, n0
    step2 = step1.cascade(_thru(frequencies, ("j0", "j1")), port_map={"n0": "j0"})
    # step2 ports: p0, p1, j1 ; now short p1<->j1 via a thru is identity, so
    # connect p1 to j1 directly using a self-consistency terminate on a thru.
    bridge = _thru(frequencies, ("b0", "b1"))
    closed = step2.cascade(bridge, port_map={"p1": "b0", "j1": "b1"})
    assert closed.port_names == ("p0",)

    torch.testing.assert_close(joined.s, closed.s, rtol=1.0e-9, atol=1.0e-10)


def test_cascade_preserves_autograd_through_connection_algebra():
    frequencies = _frequencies()
    device = _random_two_port(frequencies, requires_grad=True)
    load = _attenuator(frequencies, 0.4 + 0.2j, ("l0", "l1"))

    cascaded = device.cascade(load, port_map={"b": "l0"})
    cascaded.s.abs().square().sum().backward()

    assert device.s.grad is not None
    assert torch.all(torch.isfinite(device.s.grad))
    assert torch.count_nonzero(device.s.grad) > 0


def test_terminate_preserves_autograd():
    frequencies = _frequencies()
    device = _random_two_port(frequencies, requires_grad=True)

    terminated = device.terminate("b", gamma=0.5 + 0.1j)
    terminated.s.abs().square().sum().backward()

    assert device.s.grad is not None
    assert torch.all(torch.isfinite(device.s.grad))
    assert torch.count_nonzero(device.s.grad) > 0


def test_cascade_rejects_frequency_grid_mismatch():
    device = _random_two_port(_frequencies(count=3))
    other = _attenuator(_frequencies(count=4), 0.5 + 0j, ("n0", "n1"))
    with pytest.raises(ValueError, match="identical frequency grid"):
        device.cascade(other, port_map={"b": "n0"})


def test_cascade_rejects_mismatched_connected_reference_impedance():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    other = NetworkData(
        frequencies=frequencies,
        s=torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128),
        z0=torch.tensor([75.0, 50.0], dtype=torch.complex128),
        port_names=("n0", "n1"),
    )
    with pytest.raises(ValueError, match="same reference impedance"):
        device.cascade(other, port_map={"b": "n0"})


def test_cascade_rejects_complex_connected_reference_impedance():
    frequencies = _frequencies()
    device = NetworkData(
        frequencies=frequencies,
        s=torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128),
        z0=torch.tensor([50.0 + 5.0j, 50.0], dtype=torch.complex128),
        port_names=("a", "b"),
    )
    other = _attenuator(frequencies, 0.5 + 0j, ("n0", "n1"))
    with pytest.raises(ValueError, match="real reference impedance"):
        device.cascade(other, port_map={"a": "n0"})


def test_cascade_rejects_duplicate_connection_and_empty_map():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    other = _attenuator(frequencies, 0.5 + 0j, ("n0", "n1"))
    with pytest.raises(ValueError, match="connected more than once"):
        device.cascade(other, port_map={"a": "n0", "b": "n0"})
    with pytest.raises(ValueError, match="at least one"):
        device.cascade(other, port_map={})


def test_cascade_rejects_fully_connected_result_without_external_ports():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    other = _attenuator(frequencies, 0.5 + 0j, ("n0", "n1"))
    with pytest.raises(ValueError, match="at least one external port"):
        device.cascade(other, port_map={"a": "n0", "b": "n1"})


def test_terminate_requires_exactly_one_of_gamma_or_impedance():
    frequencies = _frequencies()
    device = _random_two_port(frequencies)
    with pytest.raises(ValueError, match="exactly one"):
        device.terminate("b")
    with pytest.raises(ValueError, match="exactly one"):
        device.terminate("b", gamma=0.1, impedance=75.0)


def test_terminate_rejects_incomplete_columns():
    frequencies = _frequencies()
    device = NetworkData(
        frequencies=frequencies,
        s=torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128),
        z0=50.0,
        port_names=("a", "b"),
        valid_columns=torch.tensor([True, False]),
    )
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        device.terminate("b", gamma=0.1)
