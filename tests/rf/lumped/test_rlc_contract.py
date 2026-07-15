import math

import pytest
import torch

from witwin.core import Box
from witwin.maxwell.lumped import (
    Capacitor,
    Inductor,
    ParallelRLC,
    PortExcitation,
    Resistor,
    SeriesRLC,
)
from witwin.maxwell.ports import AxisPath, LumpedPort
from witwin.maxwell.sources import CW


def test_two_terminal_elements_keep_tensor_values_and_terminal_order():
    resistance = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    capacitance = torch.tensor(2.0e-12, dtype=torch.float64)
    inductance = torch.tensor(3.0e-9, dtype=torch.float64)

    resistor = Resistor("r1", (0, 0, 1), (0, 0, 0), resistance)
    capacitor = Capacitor("c1", (0, 0, 1), (0, 0, 0), capacitance)
    inductor = Inductor("l1", (0, 0, 1), (0, 0, 0), inductance)

    assert resistor.value is resistance
    assert capacitor.value is capacitance
    assert inductor.value is inductance
    assert resistor.positive == capacitor.positive == inductor.positive == (0.0, 0.0, 1.0)
    assert resistor.negative == capacitor.negative == inductor.negative == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("element_type", "value"),
    [(Resistor, -1.0), (Capacitor, 0.0), (Inductor, -1.0)],
)
def test_two_terminal_elements_reject_nonpassive_or_degenerate_values(element_type, value):
    with pytest.raises(ValueError, match="positive"):
        element_type("bad", (0, 0, 1), (0, 0, 0), value)


def test_rlc_descriptors_require_one_effective_passive_component():
    with pytest.raises(ValueError, match="at least one"):
        SeriesRLC(r=0.0, l=0.0, c=None)
    with pytest.raises(ValueError, match="non-negative"):
        ParallelRLC(r=-1.0)

    series = SeriesRLC(r=50.0, l=0.0, c=2.0e-12)
    parallel = ParallelRLC(r=100.0, l=1.0e-9, c=0.0)
    assert series.r is not None and series.c is not None
    assert parallel.r is not None and parallel.l is not None


def test_series_and_parallel_rlc_match_analytic_impedance():
    omega = torch.tensor(2.0 * math.pi * 1.0e9, dtype=torch.float64)
    series = SeriesRLC(r=20.0, l=3.0e-9, c=2.0e-12)
    parallel = ParallelRLC(r=80.0, l=4.0e-9, c=1.5e-12)

    expected_series = 20.0 + 1j * omega * 3.0e-9 + 1.0 / (1j * omega * 2.0e-12)
    expected_parallel = 1.0 / (
        1.0 / 80.0 + 1.0 / (1j * omega * 4.0e-9) + 1j * omega * 1.5e-12
    )

    torch.testing.assert_close(series.impedance(omega), expected_series)
    torch.testing.assert_close(parallel.impedance(omega), expected_parallel)


def test_port_excitation_preserves_tensor_amplitude_and_validates_source_impedance():
    amplitude = torch.tensor(1.5 + 0.25j, dtype=torch.complex128, requires_grad=True)
    source_time = CW(frequency=1.0e9)
    matched = PortExcitation("p1", amplitude=amplitude, source_impedance="matched", source_time=source_time)
    explicit = PortExcitation("p1", amplitude=2.0, source_impedance=75.0)

    assert matched.amplitude is amplitude
    assert matched.source_impedance == "matched"
    assert matched.source_time is source_time
    assert explicit.source_impedance == 75.0
    assert torch.is_tensor(explicit.amplitude)

    with pytest.raises(ValueError, match="real part"):
        PortExcitation("p1", source_impedance=-50.0)
    with pytest.raises(ValueError, match="matched"):
        PortExcitation("p1", source_impedance="auto")


def test_lumped_port_accepts_an_optional_rlc_termination_without_changing_defaults():
    geometry = dict(
        name="p1",
        positive=(0.0, 0.0, 0.8),
        negative=(0.0, 0.0, 0.2),
        voltage_path=AxisPath("z"),
        current_surface=Box(position=(0.0, 0.0, 0.45), size=(0.5, 0.5, 0.0)),
    )
    default_port = LumpedPort(**geometry)
    termination = SeriesRLC(r=50.0)
    terminated_port = LumpedPort(**geometry, termination=termination)

    assert default_port.termination is None
    assert terminated_port.termination is termination
