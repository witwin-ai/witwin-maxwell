"""Milestone tier: one million passive full-field copper steps stay bounded.

Launch-bound (about 157 tiny kernels per step on a 4x4x4-cell domain), so its wall
time is dominated by GPU contention. The batch tier runs the same assertions over
1e5 steps in test_copper_foil_execution.py.
"""
import pytest
import torch

from tests.support.copper_long_horizon import run_passive_full_field_steps

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("thickness", [2e-8, 60e-6])
def test_one_million_passive_full_field_steps_remain_bounded(thickness):
    run_passive_full_field_steps(thickness, replays=10_000)
