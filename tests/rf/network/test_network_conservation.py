"""Multi-scenario passivity / conservation gates for embedded networks.

For three embedded scenarios (a lossy two-port, a reactive two-port, and a
four-port) this suite gates each real FDTD run on:

(a) Port power balance. The power flowing into the network computed from the
    field-solve port voltages and the network-solve branch currents,
    ``P1 = sum 0.5*Re(V * conj(I))``, must equal the power the network model
    would dissipate for the same field-imposed voltages,
    ``P2 = sum 0.5*Re(V^H * Y(w) * V)``. These are two independent computations
    (P1 uses the solved current, P2 uses only the field voltage and the model
    admittance), so agreement is a genuine terminal conservation check, not an
    algebraic identity. The incident wave power must also exceed the reflected +
    transmitted wave power (the network is a passive sink).

(b) Passivity. The passive network never generates net energy: the cumulative
    generated energy stays negligible relative to the absorbed energy, and the
    running cumulative net energy into the network stays non-negative when
    sampled at several run lengths. Both energies are accumulated in the time
    domain from the per-step port V/I.

(c) Time-domain stability. Between a run of length T and 2T there is no late
    time growth: the accumulated net energy converges (does not diverge), the
    dynamic state norm does not grow (it rings down under the PML), and every
    diagnostic stays finite.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.network import voltage_current_to_power_waves

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"
FULL_GRID = torch.linspace(0.0, 6.0e9, 121, dtype=torch.float64, device=DEVICE)
MEASURE_FREQUENCIES = (2.0e9, 2.5e9, 3.0e9)
SOURCE_TIME = mw.GaussianPulse(frequency=2.5e9, fwidth=1.0e9)
BASE_STEPS = 400


def _lumped_port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0.0, -0.0025), size=(0.005, 0.005, 0.0)),
        reference_impedance=50.0,
    )


def _scene(port_layout, block):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.03, 0.03), (-0.02, 0.02), (-0.02, 0.02))),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        sources=(),
        ports=tuple(_lumped_port(name, x) for name, x in port_layout),
        networks=(block,),
        device=DEVICE,
    )


def _lossy_scenario():
    conductance = torch.tensor([[0.03, -0.02], [-0.02, 0.03]], dtype=torch.float64, device=DEVICE)
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64, device=DEVICE),
        B=torch.zeros((0, 2), dtype=torch.float64, device=DEVICE),
        C=torch.zeros((2, 0), dtype=torch.float64, device=DEVICE),
        D=conductance,
        representation="Y",
        port_order=("n0", "n1"),
    )
    data = mw.NetworkData.from_y(
        frequencies=FULL_GRID,
        y=conductance.to(torch.complex128).expand(FULL_GRID.numel(), 2, 2),
        z0=50.0,
        port_names=("n0", "n1"),
    )
    block = mw.NetworkBlock(
        name="lossy",
        network=data,
        connections={"n0": "d1", "n1": "d2"},
        fit=False,
        model=model,
    )
    layout = (("d0", -0.005), ("d1", 0.0), ("d2", 0.005))
    return layout, block, model, False


def _reactive_scenario():
    resistance = 50.0
    capacitance = 5.0e-12
    shunt = 2.0e-3
    incidence = torch.tensor((1.0, -1.0), dtype=torch.float64, device=DEVICE)
    model = mw.StateSpaceNetwork(
        A=torch.tensor(((-1.0 / (resistance * capacitance),),), dtype=torch.float64, device=DEVICE),
        B=(incidence / (resistance * capacitance)).reshape(1, 2),
        C=(-incidence / resistance).reshape(2, 1),
        D=torch.outer(incidence, incidence) / resistance
        + shunt * torch.eye(2, dtype=torch.float64, device=DEVICE),
        representation="Y",
        port_order=("n0", "n1"),
    )
    data = mw.NetworkData.from_y(
        frequencies=FULL_GRID,
        y=model.evaluate(FULL_GRID),
        z0=50.0,
        port_names=("n0", "n1"),
    )
    block = mw.NetworkBlock(
        name="reactive",
        network=data,
        connections={"n0": "d1", "n1": "d2"},
        fit=False,
        model=model,
    )
    layout = (("d0", -0.005), ("d1", 0.0), ("d2", 0.005))
    return layout, block, model, True


def _four_port_scenario():
    conductance = (
        0.015 * torch.eye(4, dtype=torch.float64, device=DEVICE)
        + 0.001 * torch.ones((4, 4), dtype=torch.float64, device=DEVICE)
    )
    names = tuple(f"n{i}" for i in range(4))
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64, device=DEVICE),
        B=torch.zeros((0, 4), dtype=torch.float64, device=DEVICE),
        C=torch.zeros((4, 0), dtype=torch.float64, device=DEVICE),
        D=conductance,
        representation="Y",
        port_order=names,
    )
    data = mw.NetworkData.from_y(
        frequencies=FULL_GRID,
        y=conductance.to(torch.complex128).expand(FULL_GRID.numel(), 4, 4),
        z0=50.0,
        port_names=names,
    )
    block = mw.NetworkBlock(
        name="fourport",
        network=data,
        connections={f"n{i}": f"d{i + 1}" for i in range(4)},
        fit=False,
        model=model,
    )
    layout = tuple((f"d{i}", -0.01 + i * 0.005) for i in range(5))
    return layout, block, model, False


SCENARIOS = {
    "lossy": _lossy_scenario,
    "reactive": _reactive_scenario,
    "fourport": _four_port_scenario,
}


def _run(layout, block, steps):
    result = mw.Simulation.fdtd(
        _scene(layout, block),
        frequencies=MEASURE_FREQUENCIES,
        excitations=mw.PortExcitation("d0", source_time=SOURCE_TIME),
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    return result.embedded_network(block.name)


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
def test_embedded_network_power_balance_and_passivity(scenario):
    layout, block, model, _has_state = SCENARIOS[scenario]()
    diagnostics = _run(layout, block, BASE_STEPS)

    voltage = diagnostics.voltage
    current = diagnostics.current
    assert torch.all(torch.isfinite(torch.abs(voltage)))
    assert torch.all(torch.isfinite(torch.abs(current)))

    # (a) Power balance: field-side V + solved I vs field-side V + model Y.
    admittance = model.evaluate(torch.tensor(MEASURE_FREQUENCIES, dtype=torch.float64, device=DEVICE))
    solved_power = torch.sum(0.5 * torch.real(voltage * torch.conj(current)), dim=0)
    model_current = torch.einsum("fij,jf->if", admittance, voltage)
    model_power = torch.sum(0.5 * torch.real(torch.conj(voltage) * model_current), dim=0)
    balance_error = torch.max(torch.abs(solved_power - model_power) / torch.abs(solved_power)).item()
    assert balance_error < 1.0e-3, f"{scenario}: power balance error {balance_error:.3e}"

    # Incident wave power must dominate reflected + transmitted wave power: the
    # network is a passive sink for the field-imposed operating point.
    incident, reflected = voltage_current_to_power_waves(voltage, current, 50.0)
    incident_power = torch.sum(torch.abs(incident).square(), dim=0)
    emerging_power = torch.sum(torch.abs(reflected).square(), dim=0)
    assert torch.all(incident_power >= emerging_power - 1.0e-12 * incident_power)

    # (b) Passivity: negligible generated energy, non-negative net energy.
    absorbed = diagnostics.metadata["absorbed_energy"]
    generated = diagnostics.metadata["generated_energy"]
    assert absorbed > 0.0
    assert generated <= 1.0e-6 * absorbed, f"{scenario}: generated={generated:.3e} absorbed={absorbed:.3e}"
    assert (absorbed - generated) >= -1.0e-9 * absorbed


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
def test_embedded_network_time_domain_stability(scenario):
    # (c) No late-time growth: run T and 2T. The accumulated net energy must
    # converge (a divergent run would keep accumulating), the dynamic state norm
    # must not grow (it rings down under the PML), and diagnostics stay finite.
    layout, block, _model, has_state = SCENARIOS[scenario]()
    short = _run(layout, block, BASE_STEPS)
    long = _run(layout, block, 2 * BASE_STEPS)

    net_short = short.metadata["absorbed_energy"] - short.metadata["generated_energy"]
    net_long = long.metadata["absorbed_energy"] - long.metadata["generated_energy"]
    assert net_short > 0.0 and net_long > 0.0
    # Energy converges rather than diverging: the network interaction has
    # completed by T, so 2T adds a negligible increment.
    assert abs(net_long - net_short) <= 1.0e-3 * net_short + 1.0e-18

    state_short = short.state_norm.item()
    state_long = long.state_norm.item()
    assert torch.isfinite(short.state_norm) and torch.isfinite(long.state_norm)
    if has_state:
        # A stable dynamic network rings down: the later state norm is smaller.
        assert state_long <= state_short * (1.0 + 1.0e-3)
        assert state_short > 0.0
    else:
        # Memoryless networks carry no dynamic state.
        assert state_short == 0.0 and state_long == 0.0


def test_reactive_cumulative_energy_stays_nonnegative_at_all_sampled_times():
    # (b) "at all times": sample the running cumulative net energy at increasing
    # run lengths spanning the pulse rise, peak, and ring-down. A passive network
    # never lets the running integral of P_net go negative.
    layout, block, _model, _has_state = _reactive_scenario()
    previous = -1.0e-18
    for steps in (100, 200, 400, 800):
        diagnostics = _run(layout, block, steps)
        absorbed = diagnostics.metadata["absorbed_energy"]
        generated = diagnostics.metadata["generated_energy"]
        cumulative = absorbed - generated
        assert cumulative >= -1.0e-15, f"steps={steps}: cumulative net energy {cumulative:.3e} < 0"
        # The running integral is non-decreasing up to the same negligible
        # generated-energy floor (energy accumulates as the pulse arrives).
        assert cumulative >= previous - 1.0e-6 * abs(previous) - 1.0e-18
        previous = cumulative
