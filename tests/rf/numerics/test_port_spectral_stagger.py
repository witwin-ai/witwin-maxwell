"""Stagger convention test for the LIVE port spectral accumulators.

``prepare_port_spectral_accumulators`` builds the phase-weight tables that
production actually uses (``accumulate_precomputed`` /
``_accumulate_embedded_port_observers_gpu``). The older
``tests/rf/numerics/test_port_dft_stagger.py`` only exercises
``PortDFTAccumulator.accumulate`` / ``._scalar``, which have no production caller
anymore, and it even encodes a different sample-time convention
(``electric_time = step * dt``) than the live path
(``electric_times = (step + 1) * dt``). This test pins the live E/H stagger and
the lumped-port voltage-at-magnetic-half-step choice against an independent
DFT-kernel oracle, so a sign flip or half-step error in ``_weighted_phase_table``
can no longer pass the whole suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators


_DT = 1.0e-12
_TIME_STEPS = 6


def _runtime(field_name: str, *, lumped: object | None) -> SimpleNamespace:
    frequencies = torch.tensor([1.0e9, 2.5e9], dtype=torch.float64)
    return SimpleNamespace(
        # A pulse source forces window="none" with start_at_zero, giving unit
        # window weights so the assertion isolates the phase-time stagger.
        source_kind="gaussian_pulse",
        frequencies=frequencies,
        excitation=None,
        lumped=lumped,
        field_name=field_name,
    )


def _kernel(frequencies: torch.Tensor, sample_times: torch.Tensor) -> torch.Tensor:
    # Independent oracle: unit-weight DFT kernel exp(+i * 2*pi * f * t).
    angle = 2.0 * torch.pi * sample_times[:, None] * frequencies[None, :]
    return torch.complex(torch.cos(angle), torch.sin(angle))


def _solver(runtimes: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        _port_runtimes=runtimes,
        dt=_DT,
        device=torch.device("cpu"),
        Ez=torch.zeros(1, dtype=torch.float32),
        _compute_spectral_start_step=lambda frequency, window_type: 0,
    )


def test_live_phase_tables_encode_the_e_h_stagger_and_lumped_voltage_half_step():
    wave = _runtime("Ez", lumped=None)
    lumped = _runtime("Ez", lumped=object())
    solver = _solver((wave, lumped))

    prepare_port_spectral_accumulators(solver, _TIME_STEPS, window_type="none")

    step = torch.arange(_TIME_STEPS, dtype=torch.float64)
    electric_times = (step + 1.0) * _DT
    magnetic_times = (step + 0.5) * _DT

    # Window weights are unit for a pulse source, so the phase tables must equal
    # the bare kernel at the staggered sample times.
    expected_current = _kernel(wave.frequencies, magnetic_times)
    expected_wave_voltage = _kernel(wave.frequencies, electric_times)

    # Current is always sampled at the magnetic half-step.
    torch.testing.assert_close(wave.current_phase_weights, expected_current)
    torch.testing.assert_close(lumped.current_phase_weights, expected_current)

    # A non-lumped port samples voltage at the electric integer step -- a
    # genuinely different half-step, so the two tables must NOT coincide.
    torch.testing.assert_close(wave.voltage_phase_weights, expected_wave_voltage)
    assert not torch.allclose(
        wave.voltage_phase_weights, wave.current_phase_weights
    )

    # A lumped port samples voltage at the SAME magnetic half-step as current
    # (the physics choice the old oracle never asserted); the live path reuses
    # the identical tensor object.
    assert lumped.voltage_phase_weights is lumped.current_phase_weights
    torch.testing.assert_close(lumped.voltage_phase_weights, expected_current)
