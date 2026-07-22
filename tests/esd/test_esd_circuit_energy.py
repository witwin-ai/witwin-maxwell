"""Circuit-driven ESD energy accounting (gate b, reuses the F1a pattern).

The standard 330 ohm / 150 pF ESD generator network (:class:`ESDVoltageSource`)
drives its bound port through the strong FDTD+MNA coupling inside a closed,
non-absorbing (PEC) vacuum box, so the global energy balance closes with no
boundary outflow and no material loss beyond the circuit resistors:

    S_source(t) = dU_field(t) + dU_circuit(t) + D_circuit(t)

Every balance term is measured from an *independent* record (the MNA branch
voltage/current record, the MNA companion state, and the raw Yee E/H arrays).
Honest gate classes are identical to the F1a coupled-conservation suite: the
circuit-internal channels are consistency-class (KCL/KVL forced), while the
genuine two-sided content is ``dU_field == -W_port`` (the field-link gate) and
the whole balance closing simultaneously (the conservation gate).

Falsification (gate b): scaling either throughput channel (source-delivered or
resistor-dissipated energy) by a few percent drives the balance residual well
above the pre-registered tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.circuits import prepare_circuit_time_series
from witwin.maxwell.fdtd.ports import _edge_control_volume, apply_port_runtimes
from witwin.maxwell.fdtd.runtime.stepping import (
    enforce_pec_boundaries,
    update_electric_fields,
    update_magnetic_fields,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit-driven ESD energy conservation requires CUDA.",
)

# Pre-registered tolerances (frozen before measurement; see acceptance doc).
_STEPS = 2400
_CONSERVATION_TOL = 1.5e-2  # global balance residual / energy throughput
_LINK_TOL = 2.0e-2          # field-link residual / peak field energy
_FALSIFY_IMBALANCE = 1.03   # throughput-channel corruption for the falsification

_RD = 330.0
_CS = 150.0e-12
_RLOAD = 470.0
# A moderated ESD level keeps the closed-box field well inside float32 dynamic
# range while still exercising the full generator network; the balance is scale
# invariant so the level only sets the throughput magnitude.
_LEVEL_VOLTAGE = 500.0


def _port() -> mw.LumpedPort:
    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=50.0,
    )


def _closed_box_solver(circuit):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),),
        circuits=(circuit,),
        device="cuda",
    )
    return mw.Simulation.fdtd(scene, frequency=3.0e9).prepare().solver


def _esd_circuit(*, t_end):
    generator = mw.ESDVoltageSource(
        "gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(_LEVEL_VOLTAGE),
        discharge_resistance=_RD, storage_capacitance=_CS,
    )
    circuit = generator.build_circuit(t_end=t_end)
    circuit.add(mw.Resistor("Rload", circuit.node("tip"), circuit.ground, _RLOAD))
    return circuit


def _magnetic_control_volumes(solver):
    scene = solver.scene
    dtype = solver.Ex.dtype
    device = solver.device

    def volume(ax, ay, az):
        x = torch.as_tensor(ax, device=device, dtype=dtype)
        y = torch.as_tensor(ay, device=device, dtype=dtype)
        z = torch.as_tensor(az, device=device, dtype=dtype)
        return x[:, None, None] * y[None, :, None] * z[None, None, :]

    return {
        "Hx": volume(scene.dx_dual64, scene.dy_primal64, scene.dz_primal64),
        "Hy": volume(scene.dx_primal64, scene.dy_dual64, scene.dz_primal64),
        "Hz": volume(scene.dx_primal64, scene.dy_primal64, scene.dz_dual64),
    }


def _electric_energy(solver, cvE):
    return 0.5 * sum(
        (getattr(solver, f"eps_{name}") * cvE[name] * getattr(solver, name).square()).sum()
        for name in ("Ex", "Ey", "Ez")
    )


def _magnetic_energy(solver, cvH, half_previous):
    return 0.5 * sum(
        (getattr(solver, f"mu_{name}") * cvH[name] * half_previous[name] * getattr(solver, name)).sum()
        for name in ("Hx", "Hy", "Hz")
    )


@dataclass(frozen=True)
class _BalanceRecord:
    u_field: np.ndarray
    u_circuit: np.ndarray
    d_circuit: np.ndarray
    s_source: np.ndarray
    cum_field_change: np.ndarray

    def conservation_residual(self, *, dissipation_scale=1.0, source_scale=1.0):
        d_field = self.u_field - self.u_field[0]
        d_circuit = self.u_circuit - self.u_circuit[0]
        return d_field + d_circuit + dissipation_scale * self.d_circuit - source_scale * self.s_source

    @property
    def throughput(self):
        return float(max(np.abs(self.s_source).max(), self.d_circuit.max()))

    @property
    def peak_field(self):
        return float(self.u_field.max())


def _run_balance(circuit, *, resistor_names, source_names, steps=_STEPS):
    solver = _closed_box_solver(circuit)
    dt = float(solver.dt)
    cvH = _magnetic_control_volumes(solver)
    cvE = {name: _edge_control_volume(solver, name) for name in ("Ex", "Ey", "Ez")}

    prepare_circuit_time_series(solver, steps)
    runtime = solver._circuit_runtimes[0]
    device_power = runtime.device_power_samples

    u_field = np.empty(steps)
    u_circuit = np.empty(steps)
    d_circuit = np.empty(steps)
    s_source = np.empty(steps)
    dissipation = 0.0
    source_energy = 0.0
    for step in range(steps):
        half_previous = {name: getattr(solver, name).clone() for name in ("Hx", "Hy", "Hz")}
        update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
        u_field[step] = float(_electric_energy(solver, cvE) + _magnetic_energy(solver, cvH, half_previous))
        u_circuit[step] = float(runtime._stored_energy(runtime.state))
        update_electric_fields(
            solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz, time_value=step * dt
        )
        apply_port_runtimes(solver)
        enforce_pec_boundaries(solver)
        sample = step + 1
        dissipation += dt * sum(float(device_power[name][sample]) for name in resistor_names)
        source_energy += -dt * sum(float(device_power[name][sample]) for name in source_names)
        d_circuit[step] = dissipation
        s_source[step] = source_energy

    field_change = np.asarray(
        runtime.field_energy_change_samples[1 : steps + 1, 0].detach().cpu().numpy(), dtype=np.float64
    )
    cum_field_change = np.concatenate([[0.0], np.cumsum(field_change)[:-1]])
    return _BalanceRecord(
        u_field=u_field,
        u_circuit=u_circuit,
        d_circuit=d_circuit,
        s_source=s_source,
        cum_field_change=cum_field_change,
    )


@pytest.fixture(scope="module")
def record():
    t_end = _STEPS * float(_closed_box_solver(_esd_circuit(t_end=1.0e-9)).dt)
    return _run_balance(
        _esd_circuit(t_end=t_end), resistor_names=("Rd_gun", "Rload"), source_names=("Vgen_gun",)
    )


def test_circuit_driven_esd_conserves_coupled_energy(record):
    throughput = record.throughput
    assert throughput > 0.0
    assert np.all(np.isfinite(record.u_field))
    assert np.all(np.isfinite(record.u_circuit))
    assert record.d_circuit[-1] > 0.0
    assert bool(np.all(np.diff(record.d_circuit) >= -1e-30))
    # The 150 pF storage cap genuinely holds energy in this network.
    assert record.u_circuit.max() > 0.0
    residual = np.abs(record.conservation_residual()).max()
    assert residual <= _CONSERVATION_TOL * throughput


def test_circuit_driven_esd_field_link(record):
    peak_field = record.peak_field
    assert peak_field > 0.0
    assert np.abs(record.cum_field_change).max() > 0.0
    assert (record.u_field.max() - record.u_field.min()) > 0.0
    d_field = record.u_field - record.u_field[0]
    link_residual = np.abs(d_field - record.cum_field_change).max()
    assert link_residual <= _LINK_TOL * peak_field


def test_conservation_gate_rejects_channel_imbalance(record):
    """Falsification (b): a corrupted throughput channel is rejected."""

    baseline = np.abs(record.conservation_residual()).max()
    assert baseline <= _CONSERVATION_TOL * record.throughput
    for channel in ("dissipation_scale", "source_scale"):
        perturbed = np.abs(record.conservation_residual(**{channel: _FALSIFY_IMBALANCE})).max()
        assert perturbed > _CONSERVATION_TOL * record.throughput
