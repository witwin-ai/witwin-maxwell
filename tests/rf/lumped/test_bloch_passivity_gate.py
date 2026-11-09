"""Lossless acceptance uses total stored energy and a one-sided growth bound."""
import pytest
import torch

from scripts.dev.fdtd.bloch_lumped_acceptance.common import assert_decay, assert_passivity


@pytest.mark.parametrize("energy_rate", [0., -1e-5, 1e-8])
def test_constant_decaying_and_roundoff_scale_energy_pass(energy_rate):
    steps = torch.arange(0, 120000, 32, dtype=torch.float64)
    steps = torch.cat((steps, steps.new_tensor([119999.])))
    history = torch.zeros((steps.numel(), 6), dtype=torch.float64)
    history[:, 2] = torch.exp(energy_rate * steps)
    history[:, 3] = .75 * history[:, 2]
    assert_passivity(history)


def test_growing_circuit_energy_fails_even_when_fields_decay():
    steps = torch.arange(0, 120000, 32, dtype=torch.float64)
    steps = torch.cat((steps, steps.new_tensor([119999.])))
    history = torch.zeros((steps.numel(), 6), dtype=torch.float64)
    history[:, 0] = torch.exp(-1e-4 * steps)
    history[:, 3] = torch.exp(1e-5 * steps)
    history[:, 2] = history[:, 0] + history[:, 3]
    with pytest.raises(AssertionError):
        assert_passivity(history)


@pytest.mark.parametrize("channel", [0, 1])
def test_lossy_field_energy_and_current_squared_share_the_decay_threshold(channel):
    history = torch.ones((128, 6), dtype=torch.float64)
    history[-16:, :2] = 1e-8
    history[-16:, channel] = 4.2395913808978435e-9
    assert_decay(history, loaded=True)
    history[-16:, channel] = 5e-7
    assert_decay(history, loaded=True)
    history[-16:, channel] = 1e-6
    with pytest.raises(AssertionError):
        assert_decay(history, loaded=True)
