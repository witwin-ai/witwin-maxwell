from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.ports import PortDFTAccumulator, _evaluate_drive


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
    accumulator = PortDFTAccumulator(torch.tensor([frequency], dtype=dtype))

    for _ in range(sample_count):
        drive = _evaluate_drive(runtime)
        accumulator.accumulate(
            drive,
            torch.zeros_like(drive),
            electric_sample_time=runtime.magnetic_time,
            magnetic_sample_time=runtime.magnetic_time,
        )
        runtime.magnetic_time.add_(dt)

    phasor, _ = accumulator.phasors(normalization="peak")

    torch.testing.assert_close(phasor, expected.unsqueeze(0), rtol=1.0e-12, atol=1.0e-12)
