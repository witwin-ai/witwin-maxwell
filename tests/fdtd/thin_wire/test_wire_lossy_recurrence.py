"""Passive lossy-wire recurrence gates (B2): analytic AC, energy, DC, stability.

These exercise the exact per-segment companion that the FDTD runtime steps
(``fdtd/wire_lossy.py``), driven directly in torch float64 so the gates are
deterministic oracles independent of the 3D solver. The end-to-end integration
(PEC bitwise parity, nonzero ohmic_loss from a real run) lives in
``test_thin_wire_forward.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.compiler.wire_impedance import (
    dc_resistance,
    internal_impedance,
)
from witwin.maxwell.fdtd.wire_lossy import (
    MU_0,
    build_lossy_segment_model,
    resolve_lossy_band,
)


# Copper reference conductor, sub-grid radius, B1 robust fitting band.
_SIGMA = 5.8e7
_RADIUS = 5.0e-4
_LENGTH = 0.02
_INDUCTANCE = 1.0e-8
_BAND = (4.0e8, 3.0e9)
_DT = 1.0e-12
_ORDER = 13
_SAMPLES = 240


def _single_segment_model(
    *,
    radius=_RADIUS,
    sigma=_SIGMA,
    mu=MU_0,
    length=_LENGTH,
    inductance=_INDUCTANCE,
    kind="finite",
    band=_BAND,
    dt=_DT,
    order=_ORDER,
):
    metadata = {
        "conductor": {
            "wire_names": ("w",),
            "kinds": (kind,),
            "conductivity": (None if kind == "pec" else sigma,),
            "permeability": (None if kind == "pec" else mu,),
        }
    }
    return build_lossy_segment_model(
        inductance=torch.tensor([inductance], dtype=torch.float64),
        radius=torch.tensor([radius], dtype=torch.float64),
        length=torch.tensor([length], dtype=torch.float64),
        segment_wire_ids=torch.tensor([0], dtype=torch.int64),
        metadata=metadata,
        band=band,
        dt=dt,
        order=order,
        samples=_SAMPLES,
    )


def _drive_current_sinusoid(model, frequency, *, periods=40, steps_per_period=400):
    """Impose I(t)=cos(wt) and record the internal series voltage v_loss.

    Returns steady-state phasors ``(voltage, current)`` extracted by a single-tone
    DFT over the final half of the run (transient decayed).
    """

    dt = model.dt
    omega = 2.0 * math.pi * frequency
    total = periods * steps_per_period
    state = model.initial_state()
    companion = model.companion_conductance
    ade_output = model.ade_output

    def current_at(step_index):
        return torch.tensor(
            [math.cos(omega * dt * step_index)], dtype=torch.float64
        )

    times = []
    voltages = []
    currents = []
    current_prev = current_at(-0.5)
    for n in range(total):
        current_half = current_at(n + 0.5)
        current_bar = 0.5 * (current_half + current_prev)
        history = (ade_output * state).sum(dim=-1)
        v_loss = companion * current_bar + history
        # Advance ADE state with the trapezoidal integer-step current.
        state = torch.einsum(
            "skj,sj->sk", model.ade_transition, state
        ) + model.ade_input * current_bar.unsqueeze(-1)
        times.append((n) * dt)
        voltages.append(float(v_loss.item()))
        currents.append(float(current_bar.item()))
        current_prev = current_half

    times_t = torch.tensor(times, dtype=torch.float64)
    voltages_t = torch.tensor(voltages, dtype=torch.float64)
    currents_t = torch.tensor(currents, dtype=torch.float64)
    tail = slice(total // 2, total)
    phase = torch.exp(-1j * omega * times_t[tail])
    voltage_phasor = torch.sum(voltages_t[tail] * phase)
    current_phasor = torch.sum(currents_t[tail] * phase)
    return voltage_phasor, current_phasor


@pytest.fixture(scope="module")
def lossy_model():
    return _single_segment_model()


def test_pec_conductor_builds_no_model():
    # A PEC-only network never enters the lossy path.
    assert _single_segment_model(kind="pec") is None


def test_companion_is_passive_and_positive():
    model = _single_segment_model()
    # R0 (the exact DC series resistance) is non-negative. The trapezoidal (Tustin)
    # companion instantaneous term ``Dd`` is NOT sign-definite: skin-effect Z ~ sqrt(s)
    # is improper, so the shared rational fit carries a large direct term balanced by
    # the state history ``Cs x``. Passivity holds for the coupled (I, x) system, not
    # term by term, and is exercised by the stability/energy gates below. The discrete
    # ADE transition is stable (spectral radius < 1).
    assert bool(torch.all(model.resistance_dc >= 0.0))
    assert bool(torch.all(model.is_lossy))
    eigenvalues = torch.linalg.eigvals(model.ade_transition[0])
    assert float(eigenvalues.abs().max().item()) < 1.0
    # Build-time stability certificate: the full [I; x] companion is non-growing.
    assert model.spectral_radius < 1.0
    assert all(order >= 6 for order in model.model_orders)


def test_dc_resistance_exact():
    # Gate (c): the DC series resistance the recurrence carries is the exact
    # analytic R_dc * length, to double precision.
    model = _single_segment_model()
    expected = float(dc_resistance(_RADIUS, _SIGMA).item()) * _LENGTH
    assert model.resistance_dc.item() == pytest.approx(expected, rel=1e-12)


def test_analytic_ac_resistance_sweep(lossy_model):
    # Gate (a): the internal series resistance the recurrence realizes (extracted by
    # time-stepping the ADE, not just reading the fit) tracks the analytic
    # scaled-Bessel curve across the band.
    #
    # Pre-registered tolerance: 8% (compile-layer relative-max gate class). The
    # shared complex vector fit recovers Re(Z') only to ~1-6% and *nondeterministically*
    # (blocker B1, recorded in the finite-conductivity memory: identical config
    # flipped 1.4%/2.48%); the stability certificate here further caps the usable
    # fit order (13 here), so a strict 2% or 5% gate flips run-to-run. 8% is the
    # robust envelope and matches the compile layer's relative_max<=10% fit class.
    # bilinear warping at dt=1e-12 is negligible (< 1e-4). The separate
    # realized-vs-fitted test carries the tight recurrence-correctness proof.
    frequencies = [5.0e8, 1.0e9, 1.5e9, 2.0e9, 2.5e9]
    errors = []
    for frequency in frequencies:
        voltage, current = _drive_current_sinusoid(lossy_model, frequency)
        realized = (voltage / current).real / _LENGTH
        analytic = float(
            internal_impedance(_RADIUS, _SIGMA, MU_0, [frequency]).real.item()
        )
        errors.append(float(abs(realized - analytic) / analytic))
    assert max(errors) < 0.08, dict(zip(frequencies, errors))


def test_realized_matches_fitted_model(lossy_model):
    # The time-stepped extraction reproduces the fitted AC resistance (i.e. the
    # recurrence really implements the discrete model) far tighter than the fit's
    # own analytic error, isolating recurrence correctness from fit accuracy.
    frequency = 1.5e9
    voltage, current = _drive_current_sinusoid(lossy_model, frequency)
    realized = (voltage / current).real / _LENGTH
    fitted = float(
        lossy_model.models[0].ac_resistance([frequency]).item()
    )
    assert realized == pytest.approx(fitted, rel=2e-3)


def test_energy_closure_single_tone(lossy_model):
    # Gate (b): drive the full companion (L + R0 + ADE) with a voltage tone. Over
    # whole cycles at steady state the input work equals the recurrence's own
    # dissipation (inductor and ADE storage return to their cycle-start values),
    # so the scheme is energy-consistent and creates no numerical energy:
    #   * tight, fit-independent: input power == the companion's actual dissipation;
    #   * the reported ohmic_loss channel 0.5*Re(Z')*length*|I|^2 matches that
    #     actual dissipation within the fit tolerance (B1), so the monitored
    #     dissipation faithfully accounts for the removed energy.
    model = lossy_model
    frequency = 1.2e9
    omega = 2.0 * math.pi * frequency
    dt = model.dt
    steps_per_period = 500
    periods = 120
    total = periods * steps_per_period

    current = torch.zeros(1, dtype=torch.float64)
    state = model.initial_state()
    times = []
    input_powers = []
    dissipations = []
    current_bars = []
    for n in range(total):
        drive = torch.tensor([math.cos(omega * dt * n)], dtype=torch.float64)
        current_next, state_next = model.advance_current(current, state, drive)
        current_bar = 0.5 * (current_next + current)
        times.append(n * dt)
        input_powers.append(float((drive * current_bar).item()))
        dissipations.append(
            float(
                model.instantaneous_dissipation(current, current_next, state).item()
            )
        )
        current_bars.append(float(current_bar.item()))
        current = current_next
        state = state_next

    tail = slice(total // 2, total)
    mean_input_power = float(torch.tensor(input_powers[tail]).mean().item())
    mean_dissipation = float(torch.tensor(dissipations[tail]).mean().item())
    assert mean_input_power > 0.0
    assert mean_input_power == pytest.approx(mean_dissipation, rel=3e-3)

    times_t = torch.tensor(times[tail], dtype=torch.float64)
    phase = torch.exp(-1j * omega * times_t)
    current_phasor = torch.sum(
        torch.tensor(current_bars[tail], dtype=torch.float64) * phase
    )
    current_amplitude = 2.0 * abs(current_phasor) / times_t.numel()
    ac_resistance = float(model.ac_resistance_per_length([frequency])[0, 0].item())
    reported = 0.5 * ac_resistance * _LENGTH * float(current_amplitude) ** 2
    assert reported == pytest.approx(mean_dissipation, rel=0.08)


def test_unforced_non_growing_from_physical_state(lossy_model):
    # Gate (d): the build certifies the combined [I; x] companion is non-growing
    # (spectral radius < 1). From a physically reached state (built by a drive
    # burst) the released response never exceeds the burst envelope (no numerical
    # growth). The least-damped internal skin-effect mode is high-Q, so meaningful
    # decay takes far longer than the run; strict non-growth is the passivity gate.
    model = lossy_model
    assert model.spectral_radius < 1.0
    dt = model.dt
    omega = 2.0 * math.pi * 1.5e9
    current = torch.zeros(1, dtype=torch.float64)
    state = model.initial_state()
    burst_envelope = 0.0
    for n in range(4000):
        drive = torch.tensor([math.cos(omega * dt * n)], dtype=torch.float64)
        current, state = model.advance_current(current, state, drive)
        if n >= 3000:
            burst_envelope = max(burst_envelope, float(current.abs().item()))
    released_max = 0.0
    for _ in range(60000):
        current, state = model.advance_current(
            current, state, torch.zeros(1, dtype=torch.float64)
        )
        released_max = max(released_max, float(current.abs().item()))
    # Bounded by the driven envelope: no growth over the long unforced run.
    assert released_max < 1.5 * burst_envelope


def test_long_run_stability_no_growth(lossy_model):
    # Gate (d): a bounded random passive drive produces a bounded response over a
    # long run (no numerical growth / passivity of the companion).
    model = lossy_model
    generator = torch.Generator().manual_seed(1)
    current = torch.zeros(1, dtype=torch.float64)
    state = model.initial_state()
    peak = 0.0
    for _ in range(50000):
        drive = torch.randn(1, generator=generator, dtype=torch.float64)
        current, state = model.advance_current(current, state, drive)
        peak = max(peak, float(current.abs().item()))
    assert math.isfinite(peak)
    assert peak < 1.0e6


def test_resolve_band_requires_frequencies():
    with pytest.raises(ValueError):
        resolve_lossy_band(())
    band = resolve_lossy_band((1.0e9, 2.0e9))
    assert band[0] < 1.0e9 < 2.0e9 < band[1]
