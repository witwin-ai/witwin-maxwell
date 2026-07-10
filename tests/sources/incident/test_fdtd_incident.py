import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.spatial import AuxiliaryGrid1D


def test_auxiliary_grid_interpolates_grid_and_half_grid_samples():
    grid = AuxiliaryGrid1D(
        s_min=0.0,
        s_max=1.0,
        ds=0.1,
        dt=0.05,
        wave_speed=1.0,
        impedance=2.0,
        source_time=mw.CW(frequency=1.0),
        device="cpu",
        dtype=torch.float64,
    )

    grid.electric[:] = torch.arange(grid.electric.numel(), dtype=grid.electric.dtype)
    grid.magnetic[:] = torch.arange(grid.magnetic.numel(), dtype=grid.magnetic.dtype) + 0.5

    e_positions = torch.tensor([0.0, 0.2, 0.9], dtype=torch.float64)
    h_positions = torch.tensor([0.05, 0.25, 0.85], dtype=torch.float64)

    e_values = grid.sample_e(e_positions)
    h_values = grid.sample_h(h_positions)

    assert torch.allclose(e_values, torch.tensor([0.0, 2.0, 9.0], dtype=torch.float64))
    assert torch.allclose(h_values, torch.tensor([0.5, 2.5, 8.5], dtype=torch.float64))


def test_auxiliary_grid_pulse_propagates_forward_causally():
    grid = AuxiliaryGrid1D(
        s_min=0.0,
        s_max=4.0,
        ds=0.02,
        dt=0.01,
        wave_speed=1.0,
        impedance=1.0,
        source_time=mw.GaussianPulse(frequency=5.0, fwidth=2.0, amplitude=1.0, delay=0.4),
        device="cpu",
        dtype=torch.float64,
        absorber_cells=20,
        source_buffer_cells=8,
    )

    near_position = torch.tensor(1.0, dtype=torch.float64)
    far_position = torch.tensor(1.6, dtype=torch.float64)
    near_samples = []
    far_samples = []
    for _ in range(500):
        grid.step()
        near_samples.append(float(torch.abs(grid.sample_e(near_position)).item()))
        far_samples.append(float(torch.abs(grid.sample_e(far_position)).item()))

    near_peak_step = int(np.argmax(near_samples))
    far_peak_step = int(np.argmax(far_samples))
    expected_delay_steps = int(round((float(far_position - near_position) / grid.wave_speed) / grid.dt))

    assert far_peak_step > near_peak_step
    assert abs((far_peak_step - near_peak_step) - expected_delay_steps) <= 8
    assert max(far_samples[: near_peak_step]) < max(far_samples) * 0.35


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_auxiliary_grid_cuda_step_matches_cpu_reference():
    kwargs = dict(
        s_min=0.0,
        s_max=4.0,
        ds=0.02,
        dt=0.01,
        wave_speed=1.0,
        impedance=1.0,
        source_time=mw.GaussianPulse(frequency=5.0, fwidth=2.0, amplitude=1.0, delay=0.4),
        dtype=torch.float32,
        absorber_cells=12,
        source_buffer_cells=8,
    )
    cpu_grid = AuxiliaryGrid1D(device="cpu", **kwargs)
    cuda_grid = AuxiliaryGrid1D(device="cuda", **kwargs)

    initial_electric = torch.linspace(-0.75, 0.9, cpu_grid.electric.numel(), dtype=torch.float32)
    initial_magnetic = torch.linspace(0.35, -0.45, cpu_grid.magnetic.numel(), dtype=torch.float32)
    for grid in (cpu_grid, cuda_grid):
        grid.electric.copy_(initial_electric.to(device=grid.electric.device))
        grid.magnetic.copy_(initial_magnetic.to(device=grid.magnetic.device))
        grid.time_step = 7

    for _ in range(32):
        cpu_grid.step()
        cuda_grid.step()
        assert torch.allclose(cuda_grid.electric.cpu(), cpu_grid.electric, rtol=1e-6, atol=1e-6)
        assert torch.allclose(cuda_grid.magnetic.cpu(), cpu_grid.magnetic, rtol=1e-6, atol=1e-6)
