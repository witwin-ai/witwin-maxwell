"""Shared rectifier circuit builders for the standalone nonlinear transient gates.

Both the frozen-golden generator (``_generate_goldens``) and the pytest gates
import these builders, so the golden and the graded run use one topology, one
parameter set, and one Norton source model. Keeping the definition here (not
duplicated in the test) is what makes the golden a genuine regression anchor
rather than a restatement of the test's own arithmetic.

Topologies (node-voltage MNA, ground eliminated):

* Half-wave: an ideal ``Vs(t)`` with series resistance ``Rs`` is folded into a
  floating Norton (conductance ``1/Rs`` between the source node and ground,
  injection ``Vs/Rs``); a diode rectifies into an ``RL || C`` smoothing load.
* Full-wave (center-tapped): two identical diodes fed by two ground-referenced
  Norton sources in anti-phase (the center tap is ground); the DC output node
  carries the same ``RL || C`` load. Every node keeps a DC path to ground, so the
  matrix is non-singular in all conduction states -- unlike a floating bridge,
  whose all-diodes-off common mode is unreferenced.

The smoothing capacitor is a :class:`VoltageDependentCapacitor` with a *linear*
charge ``q(v) = C v`` (``q_coefficients = [0, C]``). This deliberately drives the
nonlinear charge-companion integrator with a case whose exact companion is the
native linear capacitor, so the transient charge path is exercised end to end
while the reference stays analytic in the reactive element.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.nonlinear_devices import (
    NonlinearMNASystem,
    compile_nonlinear_devices,
)

DTYPE = torch.float64

# Frozen parameter set (SI units). Chosen so RC ~ one source period: the output
# rides a visible ripple, which is the discriminating waveform feature.
FREQUENCY = 1.0e3
AMPLITUDE = 5.0
SOURCE_RESISTANCE = 50.0
LOAD_RESISTANCE = 1.0e3
SMOOTHING_CAPACITANCE = 1.0e-6
SATURATION_CURRENT = 1.0e-12
IDEALITY = 1.0
PERIODS = 3
STEPS_PER_PERIOD = 300
REFERENCE_SUBDIVISION = 25  # fine reference dt = coarse dt / 25


def coarse_times() -> torch.Tensor:
    period = 1.0 / FREQUENCY
    total = PERIODS * period
    count = PERIODS * STEPS_PER_PERIOD
    return torch.linspace(0.0, total, count + 1, dtype=DTYPE)


def _linear_cap():
    # q(v) = 0 + C * v  ->  a linear capacitor expressed as a single-valued Q(V).
    return torch.tensor([0.0, SMOOTHING_CAPACITANCE], dtype=DTYPE)


def build_half_wave_system() -> tuple[NonlinearMNASystem, callable, int]:
    """Return ``(system, source_injection, output_index)`` for the half-wave cell."""

    circuit = mw.Circuit("half_wave")
    src = circuit.node("src")
    out = circuit.node("out")
    node_index = {"src": 0, "out": 1}
    diode = mw.Diode(
        "d1", src, out, saturation_current=SATURATION_CURRENT, ideality=IDEALITY
    )
    cap = mw.VoltageDependentCapacitor("cs", out, circuit.ground, q_coefficients=_linear_cap())
    compiled = compile_nonlinear_devices([diode, cap], node_index, dtype=DTYPE, device="cpu")

    gs = 1.0 / SOURCE_RESISTANCE
    gl = 1.0 / LOAD_RESISTANCE
    conductance = torch.tensor([[gs, 0.0], [0.0, gl]], dtype=DTYPE)
    injection = torch.zeros(2, dtype=DTYPE)
    system = NonlinearMNASystem(2, conductance, injection, compiled)

    def source_injection(t: float) -> torch.Tensor:
        vs = AMPLITUDE * math.sin(2.0 * math.pi * FREQUENCY * t)
        return torch.tensor([vs * gs, 0.0], dtype=DTYPE)

    return system, source_injection, 1


def build_full_wave_system() -> tuple[NonlinearMNASystem, callable, int]:
    """Return ``(system, source_injection, output_index)`` for the center-tapped cell."""

    circuit = mw.Circuit("full_wave")
    a = circuit.node("a")
    b = circuit.node("b")
    top = circuit.node("top")
    node_index = {"a": 0, "b": 1, "top": 2}
    gnd = circuit.ground
    diodes = [
        mw.Diode("d1", a, top, saturation_current=SATURATION_CURRENT, ideality=IDEALITY),
        mw.Diode("d2", b, top, saturation_current=SATURATION_CURRENT, ideality=IDEALITY),
    ]
    cap = mw.VoltageDependentCapacitor("cs", top, gnd, q_coefficients=_linear_cap())
    compiled = compile_nonlinear_devices([*diodes, cap], node_index, dtype=DTYPE, device="cpu")

    gs = 1.0 / SOURCE_RESISTANCE
    gl = 1.0 / LOAD_RESISTANCE
    # Each secondary end is a ground-referenced Norton source; the center tap is
    # ground, so a and b each keep a DC path to ground in every conduction state.
    conductance = torch.tensor(
        [[gs, 0.0, 0.0], [0.0, gs, 0.0], [0.0, 0.0, gl]], dtype=DTYPE
    )
    injection = torch.zeros(3, dtype=DTYPE)
    system = NonlinearMNASystem(3, conductance, injection, compiled)

    def source_injection(t: float) -> torch.Tensor:
        vs = AMPLITUDE * math.sin(2.0 * math.pi * FREQUENCY * t)
        return torch.tensor([vs * gs, -vs * gs, 0.0], dtype=DTYPE)

    return system, source_injection, 2


def reference_times() -> torch.Tensor:
    period = 1.0 / FREQUENCY
    total = PERIODS * period
    count = PERIODS * STEPS_PER_PERIOD * REFERENCE_SUBDIVISION
    return torch.linspace(0.0, total, count + 1, dtype=DTYPE)
