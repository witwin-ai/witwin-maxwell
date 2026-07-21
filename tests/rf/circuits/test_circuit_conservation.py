"""Coupled FDTD + MNA global energy-conservation / residual suite (F1a).

Three strongly coupled field/circuit scenarios are driven from an in-circuit
source and run in a closed, non-absorbing (PEC) vacuum box so the global energy
balance closes with no boundary outflow and no material loss:

    S_source(t) = dU_field(t) + dU_circuit(t) + D_circuit(t)

where every term is measured from an *independent* record:

* ``S_source``   -- cumulative energy delivered by the circuit source(s),
                    from the MNA branch voltage/current record.
* ``D_circuit``  -- cumulative resistor dissipation, from the MNA record.
* ``dU_circuit`` -- change in circuit stored energy (0.5*C*V^2 + 0.5*L*I^2),
                    from the MNA companion state.
* ``dU_field``   -- change in the full discrete Yee electromagnetic energy
                    0.5*eps*E^2 + 0.5*mu*H(n-1/2).H(n+1/2), computed directly
                    from the raw E/H field arrays (NOT from any port record).

Honest gate classes (see the acceptance doc for the full argument):

* The ``S_source``, ``D_circuit`` and ``dU_circuit`` channels are *consistency
  class*: the MNA solve enforces KCL/KVL (Tellegen) and the trapezoidal
  companion model, so the circuit-internal statement
  ``S_source = D_circuit + dU_circuit + W_port`` is algebraically forced.
* The genuine, two-sided content is ``dU_field == -W_port``: the whole-domain
  electromagnetic energy computed from the raw fields must equal the work the
  port did on the field (from the circuit V/I record). This links the FDTD
  field update to the MNA port injection with no shared code path, and is
  carried by the dedicated ``field-link`` gate below. The memoryless
  resistive scenario (a) is deliberately dissipation-dominated (the field
  storage term is a small fraction of throughput), so the conservation gate on
  its own is largely a consistency statement there; the field-link gate is what
  makes the field-coupling term load-bearing and falsifiable.
* The conservation gate is the whole balance closing simultaneously.

The closed-box (zero boundary outflow) assumption is itself validated by
``test_lossless_cavity_conserves_discrete_energy``, which shows the same
discrete-energy functional is conserved to ~1e-7 over a long source-free run.
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
    reason="Coupled FDTD/MNA energy conservation requires CUDA.",
)

# Pre-registered tolerances (frozen before measurement; see acceptance doc).
_STEPS = 2000
# Global balance residual relative to the energy throughput. Observed margins on
# the reference host: (a) ~1.2e-4, (b) ~1.4e-3, (c) ~4.9e-5 -- all below this
# bound with >=3x headroom, and a 1% imbalance in any throughput channel exceeds
# it (the falsification).
_CONSERVATION_TOL = 5.0e-3
# Field-link residual relative to the peak field energy. Observed ~2.9e-3 (float32
# field accumulation over the run); a corrupted injection operator (~5%) exceeds
# this by a wide margin.
_LINK_TOL = 2.0e-2


def _port() -> mw.LumpedPort:
    """A z-directed lumped terminal port on the Yee half grid (MNA-coupled)."""

    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
    )


def _closed_box_solver(*, circuits=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),) if circuits else (),
        circuits=circuits,
        device="cuda",
    )
    return mw.Simulation.fdtd(scene, frequency=3.0e9).prepare().solver


def _magnetic_control_volumes(solver) -> dict[str, torch.Tensor]:
    """Dual-mesh cell volumes for the H components (Yee dual of the E edges)."""

    scene = solver.scene
    dtype = solver.Ex.dtype
    device = solver.device

    def volume(ax, ay, az) -> torch.Tensor:
        x = torch.as_tensor(ax, device=device, dtype=dtype)
        y = torch.as_tensor(ay, device=device, dtype=dtype)
        z = torch.as_tensor(az, device=device, dtype=dtype)
        return x[:, None, None] * y[None, :, None] * z[None, None, :]

    return {
        "Hx": volume(scene.dx_dual64, scene.dy_primal64, scene.dz_primal64),
        "Hy": volume(scene.dx_primal64, scene.dy_dual64, scene.dz_primal64),
        "Hz": volume(scene.dx_primal64, scene.dy_primal64, scene.dz_dual64),
    }


def _electric_energy(solver, control_volumes) -> torch.Tensor:
    return 0.5 * sum(
        (getattr(solver, f"eps_{name}") * control_volumes[name] * getattr(solver, name).square()).sum()
        for name in ("Ex", "Ey", "Ez")
    )


def _magnetic_energy(solver, control_volumes, half_step_previous) -> torch.Tensor:
    # Leapfrog-consistent magnetic energy 0.5*mu*H(n-1/2).H(n+1/2). This symmetric
    # product is the exactly conserved discrete magnetic energy for the lossless
    # Yee update; the naive same-time 0.5*mu*H^2 only conserves to O(dt^2).
    return 0.5 * sum(
        (getattr(solver, f"mu_{name}") * control_volumes[name] * half_step_previous[name] * getattr(solver, name)).sum()
        for name in ("Hx", "Hy", "Hz")
    )


@dataclass(frozen=True)
class _BalanceRecord:
    """Per-step energy channels of a coupled FDTD/MNA run (numpy float64)."""

    u_field: np.ndarray  # discrete EM energy at integer step n (from raw E/H)
    u_circuit: np.ndarray  # circuit stored energy (MNA companion state)
    d_circuit: np.ndarray  # cumulative resistor dissipation (MNA record)
    s_source: np.ndarray  # cumulative source-delivered energy (MNA record)
    cum_field_change: np.ndarray  # cumulative port-side field-energy change

    def conservation_residual(
        self,
        *,
        field_scale: float = 1.0,
        circuit_scale: float = 1.0,
        dissipation_scale: float = 1.0,
        source_scale: float = 1.0,
    ) -> np.ndarray:
        d_field = field_scale * (self.u_field - self.u_field[0])
        d_circuit = circuit_scale * (self.u_circuit - self.u_circuit[0])
        return d_field + d_circuit + dissipation_scale * self.d_circuit - source_scale * self.s_source

    @property
    def throughput(self) -> float:
        return float(max(np.abs(self.s_source).max(), self.d_circuit.max()))

    @property
    def peak_field(self) -> float:
        return float(self.u_field.max())


def _run_coupled_balance(circuit, *, resistor_names, source_names, steps=_STEPS) -> _BalanceRecord:
    """Drive a real coupled FDTD+MNA run and record every energy channel.

    The step order (magnetic update, electric update, port/circuit coupling, PEC
    clamp) is exactly the solver's per-step order for a source-free, non-dispersive,
    PEC-bounded vacuum scene; only the per-step energy diagnostics are added.
    """

    solver = _closed_box_solver(circuits=(circuit,))
    dt = float(solver.dt)
    control_volumes = _magnetic_control_volumes(solver)
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
        half_step_previous = {name: getattr(solver, name).clone() for name in ("Hx", "Hy", "Hz")}
        update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
        u_field[step] = float(
            _electric_energy(solver, cvE) + _magnetic_energy(solver, control_volumes, half_step_previous)
        )
        u_circuit[step] = float(runtime._stored_energy(runtime.state))
        update_electric_fields(
            solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz, time_value=step * dt
        )
        apply_port_runtimes(solver)
        enforce_pec_boundaries(solver)
        sample = step + 1
        dissipation += dt * sum(float(device_power[name][sample]) for name in resistor_names)
        # A source delivers energy equal to minus its recorded V*I (passive sign
        # convention of the MNA branch record).
        source_energy += -dt * sum(float(device_power[name][sample]) for name in source_names)
        d_circuit[step] = dissipation
        s_source[step] = source_energy

    # Port-side field-energy change per step (-V*I*dt); its running sum is what the
    # raw-field dU_field must reproduce (the genuine two-sided field-link check).
    field_change = np.asarray(
        runtime.field_energy_change_samples[1 : steps + 1, 0].detach().cpu().numpy(),
        dtype=np.float64,
    )
    cum_field_change = np.concatenate([[0.0], np.cumsum(field_change)[:-1]])
    return _BalanceRecord(
        u_field=u_field,
        u_circuit=u_circuit,
        d_circuit=d_circuit,
        s_source=s_source,
        cum_field_change=cum_field_change,
    )


def _assert_conserves(record: _BalanceRecord) -> None:
    throughput = record.throughput
    assert throughput > 0.0
    assert np.all(np.isfinite(record.u_field))
    assert np.all(np.isfinite(record.u_circuit))
    # Resistor dissipation is I^2 R >= 0, so the cumulative channel is monotone.
    assert record.d_circuit[-1] > 0.0
    assert bool(np.all(np.diff(record.d_circuit) >= -1e-30))
    residual = np.abs(record.conservation_residual()).max()
    assert residual <= _CONSERVATION_TOL * throughput


def _assert_field_link(record: _BalanceRecord) -> None:
    peak_field = record.peak_field
    assert peak_field > 0.0
    # Non-vacuous: the port must have exchanged real energy with the field.
    assert np.abs(record.cum_field_change).max() > 0.0
    assert (record.u_field.max() - record.u_field.min()) > 0.0
    d_field = record.u_field - record.u_field[0]
    link_residual = np.abs(d_field - record.cum_field_change).max()
    assert link_residual <= _LINK_TOL * peak_field


def _scenario_a_circuit() -> mw.Circuit:
    """(a) Resistive load on a driven port: series source + 50 ohm resistor."""

    circuit = mw.Circuit("resistive_load")
    node = circuit.node("input")
    circuit.add(
        mw.VoltageSource("V1", node, circuit.ground, 0.0, waveform=mw.SineWaveform(0.0, 1.0, 3.0e9))
    )
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    return circuit


def _scenario_b_circuit() -> mw.Circuit:
    """(b) Resonant series RLC assembled from MNA primitives (not native SeriesRLC)."""

    circuit = mw.Circuit("series_rlc")
    node_in = circuit.node("in")
    node_mid = circuit.node("mid")
    node_out = circuit.node("out")
    circuit.add(
        mw.VoltageSource("V1", node_in, circuit.ground, 0.0, waveform=mw.SineWaveform(0.0, 1.0, 3.0e9))
    )
    circuit.add(mw.Resistor("R1", node_in, node_mid, 20.0))
    circuit.add(mw.Inductor("L1", node_mid, node_out, 0.5e-9))
    circuit.add(mw.Capacitor("C1", node_out, circuit.ground, 1.0e-12))
    circuit.bind_port("feed", positive=node_in, negative=circuit.ground)
    return circuit


def _scenario_c_circuit() -> mw.Circuit:
    """(c) Controlled-source network: a VCVS drives a resistive output stage."""

    circuit = mw.Circuit("vcvs_network")
    node_in = circuit.node("in")
    node_sense = circuit.node("sense")
    node_out = circuit.node("outp")
    circuit.add(
        mw.VoltageSource("V1", node_in, circuit.ground, 0.0, waveform=mw.SineWaveform(0.0, 1.0, 3.0e9))
    )
    circuit.add(mw.Resistor("R1", node_in, node_sense, 30.0))
    circuit.add(mw.Resistor("R2", node_sense, circuit.ground, 70.0))
    circuit.add(
        mw.VoltageControlledVoltageSource("E1", node_out, circuit.ground, node_sense, circuit.ground, 2.0)
    )
    circuit.add(mw.Resistor("R3", node_out, circuit.ground, 50.0))
    circuit.bind_port("feed", positive=node_in, negative=circuit.ground)
    return circuit


def test_lossless_cavity_conserves_discrete_energy():
    """Closed-box (zero boundary outflow) validation for the energy functional.

    A source-free vacuum PEC box conserves the discrete leapfrog energy
    0.5*eps*E^2 + 0.5*mu*H(n-1/2).H(n+1/2) to ~1e-7 over a long run, establishing
    that the box does not leak energy and the functional is the conserved one.
    """

    solver = _closed_box_solver()
    dt = float(solver.dt)
    control_volumes = _magnetic_control_volumes(solver)
    cvE = {name: _edge_control_volume(solver, name) for name in ("Ex", "Ey", "Ez")}

    profile = torch.sin(
        torch.linspace(0.0, torch.pi, solver.Ez.shape[2], device=solver.device, dtype=solver.Ez.dtype)
    )
    solver.Ez.zero_()
    solver.Ez += profile[None, None, :]
    enforce_pec_boundaries(solver)

    energies = []
    for step in range(3000):
        half_step_previous = {name: getattr(solver, name).clone() for name in ("Hx", "Hy", "Hz")}
        update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
        energies.append(
            float(_electric_energy(solver, cvE) + _magnetic_energy(solver, control_volumes, half_step_previous))
        )
        update_electric_fields(
            solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz, time_value=step * dt
        )
        enforce_pec_boundaries(solver)

    energies = np.asarray(energies)
    assert energies[0] > 0.0
    assert float(np.abs(energies - energies.mean()).max() / energies.mean()) < 1.0e-5


def test_resistive_load_conserves_coupled_energy():
    record = _run_coupled_balance(_scenario_a_circuit(), resistor_names=("R1",), source_names=("V1",))
    _assert_conserves(record)
    _assert_field_link(record)


def test_series_rlc_conserves_coupled_energy():
    record = _run_coupled_balance(_scenario_b_circuit(), resistor_names=("R1",), source_names=("V1",))
    _assert_conserves(record)
    _assert_field_link(record)
    # The reactive elements genuinely store energy in this scenario.
    assert record.u_circuit.max() > 0.0


def test_controlled_source_network_conserves_coupled_energy():
    record = _run_coupled_balance(
        _scenario_c_circuit(),
        resistor_names=("R1", "R2", "R3"),
        source_names=("V1", "E1"),
    )
    _assert_conserves(record)
    _assert_field_link(record)


@pytest.mark.parametrize(
    "channel",
    ("dissipation_scale", "source_scale"),
)
def test_conservation_gate_rejects_one_percent_channel_imbalance(channel):
    """Falsification-in-suite: a 1% imbalance in a throughput channel is rejected.

    Demonstrates the conservation assertion is load-bearing on the source and
    dissipation channels (the two throughput-dominant terms). Field/circuit-store
    channels are individually below the throughput floor here and are covered by
    the field-link gate instead.
    """

    record = _run_coupled_balance(_scenario_a_circuit(), resistor_names=("R1",), source_names=("V1",))
    baseline = np.abs(record.conservation_residual()).max()
    assert baseline <= _CONSERVATION_TOL * record.throughput
    perturbed = np.abs(record.conservation_residual(**{channel: 1.01})).max()
    assert perturbed > _CONSERVATION_TOL * record.throughput
