import math

import pytest
import torch

from witwin.maxwell.antenna import AntennaData
from witwin.maxwell.network import NetworkData, PortData
from witwin.maxwell.postprocess.antenna import compute_antenna_data


_FINITE_DIFFERENCE_STEPS = (1.0e-2, 3.0e-3, 1.0e-3)


@pytest.fixture(
    params=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ]
)
def device(request):
    """Run each check on exactly one device, never across devices."""

    return request.param


def _central_difference_errors(function, *, value: float, device: torch.device):
    parameter = torch.tensor(value, device=device, dtype=torch.float64, requires_grad=True)
    objective = function(parameter)
    assert objective.ndim == 0
    assert objective.requires_grad
    assert objective.grad_fn is not None

    (gradient,) = torch.autograd.grad(objective, parameter)
    assert torch.isfinite(gradient)
    assert abs(float(gradient)) > 1.0e-8

    relative_errors = []
    for step in _FINITE_DIFFERENCE_STEPS:
        upper = torch.tensor(value + step, device=device, dtype=torch.float64)
        lower = torch.tensor(value - step, device=device, dtype=torch.float64)
        finite_difference = (function(upper) - function(lower)) / (2.0 * step)
        scale = torch.clamp_min(torch.abs(gradient), 1.0e-12)
        relative_errors.append(float(torch.abs(finite_difference - gradient) / scale))
    return gradient, relative_errors


def _network_objective(parameter: torch.Tensor) -> torch.Tensor:
    device = parameter.device
    frequencies = torch.tensor([1.0e9, 1.3e9], device=device, dtype=torch.float64)
    frequency_scale = torch.tensor([0.9, 1.1], device=device, dtype=torch.float64)
    reflection = (
        0.13 + 0.04 * parameter * frequency_scale
        + 1j * (0.05 - 0.015 * parameter * frequency_scale)
    )
    reverse_reflection = (
        -0.08 + 0.02 * parameter * frequency_scale
        + 1j * (0.04 + 0.01 * parameter * frequency_scale)
    )
    phase = 0.25 * frequency_scale + 0.3 * parameter
    transmission = (0.68 + 0.03 * parameter) * torch.exp(-1j * phase)
    reverse_transmission = (0.61 - 0.02 * parameter) * torch.exp(-0.8j * phase)
    scattering = torch.stack(
        (
            torch.stack((reflection, transmission), dim=-1),
            torch.stack((reverse_transmission, reverse_reflection), dim=-1),
        ),
        dim=-2,
    ).to(torch.complex128)
    network = NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=(50.0, 60.0),
        port_names=("input", "output"),
    )

    impedance = network.to_z()
    renormalized = network.renormalize((75.0, 45.0))
    return (
        0.20 * network.return_loss_db.mean()
        + 1.0e-4 * impedance.abs().square().mean()
        + renormalized.s.abs().square().mean()
    )


def test_network_derived_objective_matches_three_step_centered_difference(device):
    _, errors = _central_difference_errors(
        _network_objective,
        value=0.35,
        device=device,
    )

    # Require two step sizes to agree, so a single accidental crossing is not enough.
    assert sorted(errors)[:2][-1] < 0.02, errors


def _antenna_grid(device: torch.device):
    theta = torch.linspace(0.0, math.pi, 25, device=device, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 33, device=device, dtype=torch.float64)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    intensity = (
        0.7
        + 0.25 * torch.sin(theta_grid).square()
        + 0.05 * torch.cos(2.0 * phi_grid).square()
    )[None]
    return theta, phi, intensity


def _antenna_data(parameter: torch.Tensor, *, target: str) -> AntennaData:
    device = parameter.device
    frequencies = torch.tensor([1.0e9], device=device, dtype=torch.float64)
    theta, phi, intensity = _antenna_grid(device)
    complex_dtype = torch.complex128
    if target == "gain":
        incident = torch.full((1,), 1.5 + 0.0j, device=device, dtype=complex_dtype)
        reflected = (parameter * torch.exp(torch.tensor(0.2j, device=device))).reshape(1)
    elif target == "realized_gain":
        incident = parameter.to(complex_dtype).reshape(1)
        reflected = torch.full(
            (1,), 0.3 - 0.05j, device=device, dtype=complex_dtype
        )
    else:
        raise ValueError(f"Unknown antenna target {target!r}.")

    port = PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=incident,
        b=reflected,
        z0=50.0,
    )
    return compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        radiation_intensity=intensity,
        driven_port=port,
    )


def _gain_objective(parameter: torch.Tensor) -> torch.Tensor:
    data = _antenna_data(parameter, target="gain")
    return data.gain.square().mean() + 0.1 * data.gain_max.sum()


def _realized_gain_objective(parameter: torch.Tensor) -> torch.Tensor:
    data = _antenna_data(parameter, target="realized_gain")
    return data.realized_gain.square().mean() + 0.1 * data.realized_gain_max.sum()


@pytest.mark.parametrize(
    "target,function,value,power_name",
    [
        ("gain", _gain_objective, 0.42, "p_accepted"),
        ("realized_gain", _realized_gain_objective, 1.45, "p_incident"),
    ],
)
def test_antenna_gain_denominator_matches_three_step_centered_difference(
    device,
    target,
    function,
    value,
    power_name,
):
    parameter = torch.tensor(value, device=device, dtype=torch.float64, requires_grad=True)
    data = _antenna_data(parameter, target=target)
    denominator = getattr(data, power_name)
    (denominator_gradient,) = torch.autograd.grad(
        denominator.sum(), parameter, retain_graph=True
    )
    assert denominator.requires_grad
    assert denominator.grad_fn is not None
    assert torch.isfinite(denominator_gradient)
    assert abs(float(denominator_gradient)) > 1.0e-8

    _, errors = _central_difference_errors(function, value=value, device=device)
    assert sorted(errors)[:2][-1] < 0.02, errors
