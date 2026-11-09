import math

import pytest
import torch

from witwin.maxwell import invdes


def test_tanh_projection_fixes_endpoints_and_is_monotone():
    x = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    for beta in (0.5, 4.0, 64.0):
        for eta in (0.3, 0.5, 0.7):
            y = invdes.tanh_projection(x, beta, eta)
            assert float(y[0]) == pytest.approx(0.0, abs=1e-14)
            assert float(y[-1]) == pytest.approx(1.0, abs=1e-14)
            assert (y[1:] - y[:-1] >= 0).all()


def test_tanh_projection_approaches_step_as_beta_grows():
    x = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    away = (x - 0.4).abs() > 0.02
    step = invdes.tanh_projection(x, math.inf, 0.4)
    assert set(step.unique().tolist()) <= {0.0, 0.5, 1.0}
    torch.testing.assert_close(step[away], (x[away] > 0.4).to(torch.float64))
    errors = [float((invdes.tanh_projection(x, beta, 0.4) - step)[away].abs().max()) for beta in (8, 32, 128, 1024)]
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-12


def test_tanh_projection_accepts_tensor_eta_and_beta():
    x = torch.rand(6, 5, dtype=torch.float64)
    eta = torch.full_like(x, 0.45)
    torch.testing.assert_close(
        invdes.tanh_projection(x, torch.tensor(8.0, dtype=torch.float64), eta),
        invdes.tanh_projection(x, 8.0, 0.45),
    )
    with pytest.raises(ValueError):
        invdes.tanh_projection(x, 0.0)


def test_robust_projections_are_ordered():
    x = torch.rand(10, 10, dtype=torch.float64)
    eroded, nominal, dilated = invdes.robust_projections(x, 16.0, eta=0.5, delta=0.15)
    assert (eroded <= nominal + 1e-15).all() and (nominal <= dilated + 1e-15).all()
    with pytest.raises(ValueError):
        invdes.robust_projections(x, 16.0, eta=0.5, delta=0.6)


def _ramp(n=24, h=0.01):
    x = torch.linspace(0.05, 0.95, n, dtype=torch.float64)[:, None].expand(n, 6).contiguous()
    return x, (h, h)


def test_smoothed_projection_matches_tanh_away_from_the_interface():
    flat = torch.full((8, 8), 0.8, dtype=torch.float64)
    torch.testing.assert_close(
        invdes.smoothed_projection(flat, 16.0, spacing=0.01), invdes.tanh_projection(flat, 16.0)
    )
    x, spacing = _ramp()
    out = invdes.smoothed_projection(x, 16.0, spacing=spacing)
    ref = invdes.tanh_projection(x, 16.0)
    far = (x - 0.5).abs() > 0.1
    torch.testing.assert_close(out[far], ref[far])


def test_smoothed_projection_is_differentiable_at_infinite_beta():
    x, spacing = _ramp()
    x = x.clone().requires_grad_(True)
    out = invdes.smoothed_projection(x, math.inf, spacing=spacing)
    assert ((out >= 0) & (out <= 1)).all()
    interior = out.detach()[(out.detach() > 0) & (out.detach() < 1)]
    assert interior.numel() > 0  # interface cells carry fractional fill
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # Large finite beta converges to the infinite-beta fill fraction.
    big = invdes.smoothed_projection(x.detach(), 1.0e5, spacing=spacing)
    torch.testing.assert_close(big, out.detach(), atol=1e-3, rtol=0)


def test_beta_schedule_is_deterministic():
    schedule = invdes.BetaSchedule(initial=1.0, factor=2.0, interval=10, maximum=16.0, warmup=5)
    values = schedule.values(70)
    assert values == invdes.BetaSchedule(initial=1.0, factor=2.0, interval=10, maximum=16.0, warmup=5).values(70)
    assert values[:15] == [1.0] * 15
    assert values[15] == 2.0 and values[25] == 4.0 and values[35] == 8.0 and values[45] == 16.0
    assert values[-1] == 16.0
    assert schedule.milestones(70) == [15, 25, 35, 45]
    assert schedule(10**9) == 16.0
    assert invdes.BetaSchedule(maximum=math.inf)(200) == 2.0**8
    for bad in (dict(initial=0.0), dict(factor=0.5), dict(interval=0), dict(maximum=0.5), dict(warmup=-1)):
        with pytest.raises(ValueError):
            invdes.BetaSchedule(**bad)
