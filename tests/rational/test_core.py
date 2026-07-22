import numpy as np
import pytest
import torch
from scipy.signal import cont2discrete

from witwin.maxwell.rational import RationalModel, StateSpaceNetwork


def test_real_realization_has_n_times_order_states_and_matches_response() -> None:
    poles = torch.tensor(
        [-3.0 + 0.0j, -2.0 + 5.0j, -2.0 - 5.0j], dtype=torch.complex128
    )
    residues = torch.tensor(
        [
            [
                [[0.8, 0.3 + 0.2j, 0.3 - 0.2j]],
                [[0.1, 0.05 - 0.03j, 0.05 + 0.03j]],
            ],
            [
                [[0.1, 0.05 - 0.03j, 0.05 + 0.03j]],
                [[0.6, 0.2 + 0.1j, 0.2 - 0.1j]],
            ],
        ],
        dtype=torch.complex128,
    ).squeeze(2)
    model = RationalModel(poles, residues, torch.tensor([[0.4, 0.02], [0.02, 0.5]]))
    frequencies = torch.logspace(-2, 1.5, 80, dtype=torch.float64)

    state_space = model.to_state_space(port_order=("left", "right"))

    assert state_space.state_count == model.input_count * model.order
    torch.testing.assert_close(
        state_space.evaluate(frequencies), model.evaluate(frequencies), rtol=2e-12, atol=2e-12
    )


def test_bilinear_matches_scipy_and_strict_pole_mapping() -> None:
    A = torch.tensor([[-2.0, -3.0], [3.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    C = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    D = torch.tensor([[0.1]], dtype=torch.float64)
    system = StateSpaceNetwork(A, B, C, D)
    discrete = system.discretize(0.04)
    oracle = cont2discrete(tuple(x.numpy() for x in (A, B, C, D)), 0.04, method="bilinear")

    for actual, expected in zip((discrete.A, discrete.B, discrete.C, discrete.D), oracle[:4]):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-13, atol=2e-13)
    assert discrete.pole_radius < 1.0 - 1e-7
    continuous_poles = torch.linalg.eigvals(A)
    expected_poles = (1.0 + 0.5 * 0.04 * continuous_poles) / (
        1.0 - 0.5 * 0.04 * continuous_poles
    )
    actual_poles = torch.linalg.eigvals(discrete.A)
    torch.testing.assert_close(
        torch.sort(actual_poles.imag).values,
        torch.sort(expected_poles.imag).values,
        rtol=2e-13,
        atol=2e-13,
    )


def test_passivity_detection_enforcement_and_trainable_coefficients() -> None:
    frequencies = torch.logspace(-3, 2, 100, dtype=torch.float64)
    passive = RationalModel(
        torch.tensor([-1.0 + 0.0j], dtype=torch.complex128),
        torch.tensor([1.0 + 0.0j], dtype=torch.complex128),
        0.05,
    )
    active = RationalModel(passive.poles, -passive.residues, -0.05)
    assert passive.check_passivity(frequencies).passive
    assert not active.check_passivity(frequencies).passive
    enforced, report = active.enforce_passivity(frequencies)
    assert report.passive and report.enforcement_change > 0.0
    assert enforced.check_passivity(frequencies).passive

    free_residue = torch.tensor(0.5 + 0.2j, dtype=torch.complex128, requires_grad=True)
    direct = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    residues = torch.stack((free_residue, free_residue.conj())).reshape(1, 1, 2)
    model = RationalModel(
        torch.tensor([-2.0 + 3.0j, -2.0 - 3.0j], dtype=torch.complex128),
        residues,
        direct,
    )
    torch.sum(torch.abs(model.evaluate(torch.tensor([0.2, 0.8]))) ** 2).backward()
    assert free_residue.grad is not None and direct.grad is not None

    proportional = RationalModel(passive.poles, passive.residues, proportional=0.2)
    with pytest.raises(ValueError, match="proportional"):
        proportional.to_state_space()


def test_real_realization_rejects_coefficients_that_would_be_discarded() -> None:
    with pytest.raises(ValueError, match="direct must be real"):
        RationalModel(
            torch.tensor([-1.0 + 0.0j], dtype=torch.complex128),
            torch.tensor([1e12 + 0.0j], dtype=torch.complex128),
            1.0j,
        )


def test_adaptive_passivity_check_finds_between_sample_resonance() -> None:
    resonance = 1.0003
    poles = torch.tensor(
        [
            -1e-8 + 2j * torch.pi * resonance,
            -1e-8 - 2j * torch.pi * resonance,
        ],
        dtype=torch.complex128,
    )
    model = RationalModel(
        poles,
        torch.tensor([-1e-6 + 0.0j, -1e-6 + 0.0j], dtype=torch.complex128),
        1.0,
    )

    report = model.check_passivity(torch.tensor([0.0, 2.0], dtype=torch.float64))

    assert not report.passive
    assert report.margin < -90.0
    assert report.certified

    enforced, enforced_report = model.enforce_passivity(
        torch.tensor([0.0, 2.0], dtype=torch.float64)
    )
    assert enforced_report.passive and enforced_report.certified
    assert enforced.check_passivity(
        torch.tensor([0.0, 2.0], dtype=torch.float64)
    ).certified


def test_enforcement_rejects_unresolved_lossless_boundary() -> None:
    allpass = RationalModel(
        torch.tensor([-1.0 + 0.0j], dtype=torch.complex128),
        torch.tensor([-2.0 + 0.0j], dtype=torch.complex128),
        1.0,
        representation="S",
    )
    frequencies = torch.tensor([0.0, 10.0], dtype=torch.float64)

    report = allpass.check_passivity(frequencies)

    assert report.passive and not report.certified
    with pytest.raises(RuntimeError, match="could not be certified"):
        allpass.enforce_passivity(frequencies)


def test_rational_model_persistence_round_trip(tmp_path) -> None:
    model = RationalModel(
        torch.tensor([-2.0 + 3.0j, -2.0 - 3.0j], dtype=torch.complex128),
        torch.tensor([0.5 + 0.2j, 0.5 - 0.2j], dtype=torch.complex128),
        0.1,
    )
    path = tmp_path / "network_model.pt"

    model.save(path)
    loaded = RationalModel.load(path)

    torch.testing.assert_close(loaded.poles, model.poles)
    torch.testing.assert_close(loaded.residues, model.residues)
    torch.testing.assert_close(loaded.direct, model.direct)
    frequencies = torch.logspace(-2, 1, 20, dtype=torch.float64)
    torch.testing.assert_close(loaded.evaluate(frequencies), model.evaluate(frequencies))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefitted_model_gradient_and_discretization_stay_on_cuda() -> None:
    device = torch.device("cuda")
    residue = torch.tensor([1.0 + 0.0j], device=device, requires_grad=True)
    direct = torch.tensor(0.1, dtype=torch.float32, device=device, requires_grad=True)
    model = RationalModel(
        torch.tensor([-2.0 + 0.0j], device=device),
        residue,
        direct,
    )

    loss = torch.sum(torch.abs(model.evaluate(torch.tensor([0.2, 0.8], device=device))) ** 2)
    loss.backward()
    discrete = model.to_state_space().discretize(0.01)

    assert residue.grad is not None and residue.grad.device.type == "cuda"
    assert direct.grad is not None and direct.grad.device.type == "cuda"
    assert all(
        tensor.device.type == "cuda"
        for tensor in (discrete.A, discrete.B, discrete.C, discrete.D)
    )


def test_passive_discrete_network_obeys_driven_supply_inequality() -> None:
    time = torch.arange(400, dtype=torch.float64) * 0.01
    voltage = torch.sin(2.0 * torch.pi * 0.7 * time) + 0.3 * torch.sin(
        2.0 * torch.pi * 1.3 * time
    )

    def accumulated_work(sign: float) -> torch.Tensor:
        model = RationalModel(
            torch.tensor([-2.0 + 0.0j], dtype=torch.complex128),
            torch.tensor([sign + 0.0j], dtype=torch.complex128),
            0.1 * sign,
        )
        discrete = model.to_state_space().discretize(0.01)
        state = torch.zeros(1, dtype=torch.float64)
        work = torch.zeros((), dtype=torch.float64)
        history = []
        for sample in voltage:
            state, current = discrete.step(state, sample.reshape(1))
            work = work + sample * current[0]
            history.append(work)
        return torch.stack(history)

    passive_work = accumulated_work(1.0)
    active_work = accumulated_work(-1.0)
    assert torch.all(passive_work >= -1e-12)
    assert active_work[-1] < -1.0
