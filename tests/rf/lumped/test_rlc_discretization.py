import math

import torch

from witwin.maxwell.lumped import (
    Capacitor,
    Inductor,
    ParallelRLC,
    RLCBranchState,
    Resistor,
    SeriesRLC,
    branch_energy,
    trapezoidal_branch_step,
    trapezoidal_impedance,
)


def _scalar(value):
    return torch.tensor(value, dtype=torch.float64)


def test_trapezoidal_reference_handles_resistor_capacitor_and_inductor_steps():
    resistor_state = RLCBranchState(
        current=_scalar(1.0),
        capacitor_voltage=_scalar(0.0),
        previous_voltage=_scalar(2.0),
    )
    resistor_next, resistor_mid = trapezoidal_branch_step(
        Resistor("r", (0, 0, 1), (0, 0, 0), 2.0),
        resistor_state,
        voltage=_scalar(2.0),
        dt=_scalar(0.1),
    )
    torch.testing.assert_close(resistor_mid, _scalar(1.0))
    torch.testing.assert_close(resistor_next.current, _scalar(1.0))

    capacitor_state = RLCBranchState.zeros(dtype=torch.float64)
    capacitor_next, capacitor_mid = trapezoidal_branch_step(
        Capacitor("c", (0, 0, 1), (0, 0, 0), 2.0),
        capacitor_state,
        voltage=_scalar(1.0),
        dt=_scalar(0.1),
    )
    torch.testing.assert_close(capacitor_mid, _scalar(20.0))
    torch.testing.assert_close(capacitor_next.capacitor_voltage, _scalar(1.0))

    inductor_state = RLCBranchState(
        current=_scalar(0.0),
        capacitor_voltage=_scalar(0.0),
        previous_voltage=_scalar(1.0),
    )
    inductor_next, inductor_mid = trapezoidal_branch_step(
        Inductor("l", (0, 0, 1), (0, 0, 0), 2.0),
        inductor_state,
        voltage=_scalar(1.0),
        dt=_scalar(0.1),
    )
    torch.testing.assert_close(inductor_next.current, _scalar(0.05))
    torch.testing.assert_close(inductor_mid, _scalar(0.025))


def test_passive_series_rlc_energy_does_not_increase_without_drive():
    model = SeriesRLC(r=0.5, l=1.0, c=1.0)
    state = RLCBranchState(
        current=_scalar(1.0),
        capacitor_voltage=_scalar(0.5),
        previous_voltage=_scalar(0.0),
    )
    previous_energy = branch_energy(model, state)

    for _ in range(500):
        state, _ = trapezoidal_branch_step(
            model,
            state,
            voltage=_scalar(0.0),
            dt=_scalar(0.01),
        )
        energy = branch_energy(model, state)
        assert energy <= previous_energy + 1.0e-12
        previous_energy = energy


def test_trapezoidal_series_and_parallel_resonance_error_is_below_two_percent():
    inductance = 1.0e-3
    capacitance = 1.0e-6
    expected = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))
    frequencies = torch.linspace(0.8 * expected, 1.2 * expected, 4001, dtype=torch.float64)
    omega = 2.0 * math.pi * frequencies
    dt = _scalar(1.0 / (100.0 * expected))

    series = SeriesRLC(r=0.1, l=inductance, c=capacitance)
    parallel = ParallelRLC(r=1000.0, l=inductance, c=capacitance)
    series_peak = frequencies[torch.argmin(torch.abs(trapezoidal_impedance(series, omega, dt)))]
    parallel_peak = frequencies[torch.argmax(torch.abs(trapezoidal_impedance(parallel, omega, dt)))]

    assert torch.abs(series_peak - expected) / expected < 0.02
    assert torch.abs(parallel_peak - expected) / expected < 0.02


def test_trapezoidal_reference_and_impedance_keep_autograd():
    resistance = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    model = SeriesRLC(r=resistance, l=1.0, c=1.0)
    state = RLCBranchState(
        current=_scalar(1.0),
        capacitor_voltage=_scalar(0.5),
        previous_voltage=_scalar(0.0),
    )
    next_state, midpoint_current = trapezoidal_branch_step(
        model,
        state,
        voltage=_scalar(0.0),
        dt=_scalar(0.01),
    )
    impedance = model.impedance(_scalar(2.0))
    loss = branch_energy(model, next_state) + midpoint_current.square() + torch.abs(impedance).square()
    loss.backward()

    assert resistance.grad is not None
    assert torch.isfinite(resistance.grad)
    assert resistance.grad != 0.0
