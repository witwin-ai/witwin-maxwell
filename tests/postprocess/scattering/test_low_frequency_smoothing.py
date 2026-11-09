"""Low-frequency smoothing of short-window port and network spectra."""
from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.network import NetworkData, PortData

RUN_TIME = 1.0e-9  # 1 ns window: f periods = f / 1 GHz


def _frequencies():
    return torch.linspace(0.2e9, 8.0e9, 40, dtype=torch.float64)


def _truth(frequencies):
    x = frequencies / 1.0e9
    return (0.1 + 0.05 * x) + 1j * (0.02 * x)


def test_spec_validates_its_window_and_order():
    with pytest.raises(ValueError):
        mw.LowFrequencySmoothingSpec(min_sampling_time=5.0, max_sampling_time=5.0)
    with pytest.raises(ValueError):
        mw.LowFrequencySmoothingSpec(order=4)
    with pytest.raises(ValueError):
        mw.LowFrequencySmoothingSpec(max_deviation=-0.1)
    spec = mw.LowFrequencySmoothingSpec(min_sampling_time=2, max_sampling_time=6, order=1, max_deviation=None)
    assert (spec.min_sampling_time, spec.max_sampling_time, spec.order) == (2.0, 6.0, 1)


def test_undersampled_network_samples_are_replaced_by_the_trusted_polynomial():
    frequencies = _frequencies()
    truth = _truth(frequencies)
    corrupted = truth.clone()
    low = frequencies * RUN_TIME < 2.0
    corrupted[low] += 0.3 * torch.exp(1j * frequencies[low] / 1e8)
    s = torch.zeros((40, 2, 2), dtype=torch.complex128)
    s[:, 0, 0] = corrupted
    s[:, 1, 0] = corrupted
    network = NetworkData(frequencies=frequencies, s=s, z0=50.0, port_names=("a", "b"))
    spec = mw.LowFrequencySmoothingSpec(min_sampling_time=2.0, max_sampling_time=6.0, order=1, max_deviation=None)
    smoothed = mw.smooth_low_frequency(network, spec, run_time=RUN_TIME)
    assert torch.allclose(smoothed.s[:, 0, 0], truth, atol=1e-12)
    assert torch.allclose(smoothed.s[:, 1, 0], truth, atol=1e-12)
    # Samples above max_sampling_time and the untouched columns are bitwise unchanged.
    assert torch.equal(smoothed.s[~low], s[~low])
    record = smoothed.metadata["low_frequency_smoothing"]
    assert record["replaced_samples"] == int(low.sum()) and record["run_time"] == RUN_TIME
    assert smoothed.port_names == network.port_names


def test_max_deviation_pulls_extrapolation_back_toward_the_lowest_trusted_sample():
    frequencies = _frequencies()
    values = 1.0 + 0.0j + 0.5 * frequencies.to(torch.complex128) / 1e9
    spec = mw.LowFrequencySmoothingSpec(min_sampling_time=4.0, max_sampling_time=8.0, order=1, max_deviation=0.1)
    smoothed = mw.smooth_low_frequency(values, spec, run_time=RUN_TIME, frequencies=frequencies)
    trusted = (frequencies * RUN_TIME >= 4.0) & (frequencies * RUN_TIME <= 8.0)
    reference = values[trusted][0]
    replaced = frequencies * RUN_TIME < 4.0
    deviation = torch.abs(smoothed[replaced] - reference)
    assert torch.all(deviation <= 0.1 * torch.abs(reference) + 1e-12)
    assert torch.any(deviation > 0.099 * torch.abs(reference))


def test_port_data_smoothing_is_differentiable_and_keeps_the_device():
    frequencies = _frequencies()
    voltage = _truth(frequencies).clone().requires_grad_(True)
    current = (voltage / 50.0).detach()
    data = PortData(port_name="p", frequencies=frequencies, voltage=voltage, current=current, z0=50.0)
    smoothed = mw.smooth_low_frequency(data, mw.LowFrequencySmoothingSpec(min_sampling_time=2.0,
                                                                           max_sampling_time=6.0), run_time=RUN_TIME)
    smoothed.voltage.real.sum().backward()
    assert voltage.grad is not None and torch.all(torch.isfinite(voltage.grad))
    assert smoothed.voltage.device == voltage.device


def test_smoothing_requires_enough_trusted_samples():
    frequencies = torch.tensor([0.5e9, 1.0e9, 3.0e9], dtype=torch.float64)
    with pytest.raises(ValueError, match="trusted samples"):
        mw.smooth_low_frequency(torch.ones(3, dtype=torch.complex128),
                                mw.LowFrequencySmoothingSpec(order=2), run_time=RUN_TIME, frequencies=frequencies)
    with pytest.raises(TypeError):
        mw.smooth_low_frequency([1.0], mw.LowFrequencySmoothingSpec(), run_time=RUN_TIME)
