import pytest
import torch

from witwin.maxwell.fdtd.ports import PortDFTAccumulator


def _accumulate_two_tones(device: torch.device) -> PortDFTAccumulator:
    dtype = torch.float64
    frequencies = torch.tensor([5.0, 13.0], dtype=dtype, device=device)
    voltage_peak = torch.tensor([2.5, 0.75], dtype=dtype, device=device)
    current_peak = torch.tensor([0.04, 0.015], dtype=dtype, device=device)
    voltage_phase = torch.tensor([0.3, -0.45], dtype=dtype, device=device)
    current_phase = torch.tensor([-0.2, 0.65], dtype=dtype, device=device)
    sample_count = 512
    dt = torch.tensor(1.0 / sample_count, dtype=dtype, device=device)

    accumulator = PortDFTAccumulator(frequencies)
    for step in range(sample_count):
        electric_time = step * dt
        magnetic_time = (step + 0.5) * dt
        voltage = torch.sum(
            voltage_peak
            * torch.cos(2.0 * torch.pi * frequencies * electric_time - voltage_phase)
        )
        current = torch.sum(
            current_peak
            * torch.cos(2.0 * torch.pi * frequencies * magnetic_time - current_phase)
        )
        window_weight = 0.5 - 0.5 * torch.cos(
            2.0 * torch.pi * torch.as_tensor(step, dtype=dtype, device=device) / sample_count
        )
        accumulator.accumulate(
            voltage,
            current,
            electric_sample_time=electric_time,
            magnetic_sample_time=magnetic_time,
            window_weight=window_weight,
        )

    return accumulator


def _expected_phasors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    voltage_peak = torch.tensor([2.5, 0.75], dtype=dtype, device=device)
    current_peak = torch.tensor([0.04, 0.015], dtype=dtype, device=device)
    voltage_phase = torch.tensor([0.3, -0.45], dtype=dtype, device=device)
    current_phase = torch.tensor([-0.2, 0.65], dtype=dtype, device=device)
    voltage = torch.polar(voltage_peak, voltage_phase)
    current = torch.polar(current_peak, current_phase)
    return voltage, current


def test_port_dft_uses_staggered_sample_times_and_window_normalization():
    device = torch.device("cpu")
    accumulator = _accumulate_two_tones(device)

    voltage, current = accumulator.phasors(normalization="peak")
    expected_voltage, expected_current = _expected_phasors(device)

    assert voltage.shape == (2,)
    assert current.shape == (2,)
    assert voltage.is_complex()
    assert current.is_complex()
    torch.testing.assert_close(voltage, expected_voltage, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(current, expected_current, rtol=1.0e-12, atol=1.0e-12)


def test_port_dft_peak_and_rms_phasors_have_sqrt_two_ratio():
    device = torch.device("cpu")
    accumulator = _accumulate_two_tones(device)
    peak_voltage, peak_current = accumulator.phasors(normalization="peak")
    rms_voltage, rms_current = accumulator.phasors(normalization="rms")
    expected_voltage, expected_current = _expected_phasors(device)
    sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))

    torch.testing.assert_close(rms_voltage, expected_voltage / sqrt_two, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(rms_current, expected_current / sqrt_two, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(peak_voltage, sqrt_two * rms_voltage, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(peak_current, sqrt_two * rms_current, rtol=1.0e-12, atol=1.0e-12)


def test_reusing_electric_phase_for_magnetic_sample_exposes_half_step_error():
    dtype = torch.float64
    frequency = torch.tensor([17.0], dtype=dtype)
    phase = torch.tensor([0.4], dtype=dtype)
    peak = torch.tensor(3.0, dtype=dtype)
    sample_count = 256
    dt = torch.tensor(1.0 / sample_count, dtype=dtype)
    correct = PortDFTAccumulator(frequency)
    wrong_current_sum = torch.zeros(1, dtype=torch.complex128)

    for step in range(sample_count):
        electric_time = step * dt
        magnetic_time = (step + 0.5) * dt
        current = peak * torch.cos(2.0 * torch.pi * frequency[0] * magnetic_time - phase[0])
        correct.accumulate(
            torch.zeros((), dtype=dtype),
            current,
            electric_sample_time=electric_time,
            magnetic_sample_time=magnetic_time,
        )
        wrong_angle = 2.0 * torch.pi * frequency * electric_time
        wrong_current_sum = wrong_current_sum + current * torch.complex(
            torch.cos(wrong_angle),
            torch.sin(wrong_angle),
        )

    _, correct_current = correct.phasors(normalization="peak")
    wrong_current = (2.0 / sample_count) * wrong_current_sum
    expected = torch.polar(peak.expand_as(phase), phase)
    expected_wrong = expected * torch.exp(-1j * torch.pi * frequency * dt)

    torch.testing.assert_close(correct_current, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(wrong_current, expected_wrong, rtol=1.0e-12, atol=1.0e-12)
    assert not torch.allclose(wrong_current, expected, rtol=1.0e-3, atol=1.0e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_port_dft_cpu_cuda_parity():
    cpu_accumulator = _accumulate_two_tones(torch.device("cpu"))
    cuda_accumulator = _accumulate_two_tones(torch.device("cuda"))

    cpu_voltage, cpu_current = cpu_accumulator.phasors(normalization="peak")
    cuda_voltage, cuda_current = cuda_accumulator.phasors(normalization="peak")

    torch.testing.assert_close(cuda_voltage, cpu_voltage.to("cuda"), rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(cuda_current, cpu_current.to("cuda"), rtol=1.0e-12, atol=1.0e-12)
