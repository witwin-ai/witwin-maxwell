from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.ports import _evaluate_drive


def test_complex_cw_drive_recovers_peak_phasor_under_exp_minus_iwt_convention():
    dtype = torch.float64
    frequency = 7.0
    sample_count = 512
    dt = 1.0 / sample_count
    expected = torch.tensor(1.25 + 0.4j, dtype=torch.complex128)
    runtime = SimpleNamespace(
        source_kind="cw",
        source_frequency=frequency,
        source_fwidth=0.0,
        source_phase=0.0,
        source_delay=0.0,
        source_amplitude=expected,
        drive_buffer=torch.zeros((), dtype=dtype),
        magnetic_time=torch.tensor(0.5 * dt, dtype=dtype),
    )

    # Independent DFT oracle for the live drive waveform. Production accumulates
    # the same exp(+i*2*pi*f*t) kernel through ``accumulate_precomputed``; here
    # we sum it directly so the assertion isolates ``_evaluate_drive``.
    frequencies = torch.tensor([frequency], dtype=dtype)
    voltage_sum = torch.zeros(1, dtype=torch.complex128)
    for _ in range(sample_count):
        drive = _evaluate_drive(runtime)
        angle = 2.0 * torch.pi * frequencies * runtime.magnetic_time
        phase = torch.complex(torch.cos(angle), torch.sin(angle))
        voltage_sum = voltage_sum + drive.to(torch.complex128) * phase
        runtime.magnetic_time.add_(dt)

    phasor = (2.0 / sample_count) * voltage_sum

    torch.testing.assert_close(phasor, expected.unsqueeze(0), rtol=1.0e-12, atol=1.0e-12)
