"""Graph-captured port step phase of single FDTD runs.

Single runs replay the port terminations, the post-source tail and the port V/I
DFT as one CUDA graph per step, driven by a precomputed drive table. These
gates pin the drive table to the eager per-step drive, the graph path to the
eager path bit for bit, and the eligibility guards that keep circuits and
embedded networks on the eager path.
"""
from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import _advance_port_sample_time, _evaluate_drive, prepare_port_drive_tables
from tests.support import frozen_oracles

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="port step graphs require CUDA")

PULSE = mw.GaussianPulse(frequency=0.5e9, fwidth=0.3e9)


def _scene(*, circuit=False):
    structures = [
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.75), size=(0.5, 0.5, 0.1)), mw.Material.pec(), name="signal"),
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.25), size=(0.5, 0.5, 0.1)), mw.Material.pec(), name="ground"),
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.5), size=(0.5, 0.5, 0.4)),
                     mw.Material(eps_r=2.2, sigma_e=0.02), name="lossy_fill"),
    ]
    ports = (
        mw.LumpedPort("p1", positive=(0.5, 0.5, 0.7), negative=(0.5, 0.5, 0.3), voltage_path=mw.AxisPath("z"),
                      current_surface=mw.Box(position=(0.5, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
                      reference_impedance=50.0),
        mw.LumpedPort("p2", positive=(1.2, 0.5, 0.6), negative=(1.2, 0.5, 0.4), voltage_path=mw.AxisPath("z"),
                      current_surface=mw.Box(position=(1.2, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
                      reference_impedance=75.0),
    )
    circuits = ()
    if circuit:
        net = mw.circuits.Circuit("load")
        node = net.node("input")
        net.add(mw.rf.Resistor("R1", node, net.ground, 50.0))
        net.bind_port("p2", positive=node, negative=net.ground)
        circuits = (net,)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        structures=structures,
        ports=ports,
        lumped_elements=(mw.rf.Resistor("shunt", positive=(1.4, 0.5, 0.6), negative=(1.4, 0.5, 0.4), resistance=100.0),),
        circuits=circuits,
    )
    return scene


def _simulation(*, steps=203, cuda_graph=True, circuit=False, source_time=PULSE, **options):
    simulation = mw.Simulation.fdtd(
        _scene(circuit=circuit),
        frequencies=(0.4e9, 0.6e9),
        excitations=mw.rf.PortExcitation("p1", source_time=source_time),
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=steps),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
            cuda_graph=cuda_graph,
            **options,
        ),
    )
    return simulation


def _payload(result):
    """Every quantity the bitwise port checks compare, as CPU tensors."""
    payload = {name: getattr(result.solver, name).detach().cpu().clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    for name in ("p1", "p2"):
        port = result.port(name)
        payload[f"{name}/voltage"] = port.voltage.detach().cpu().clone()
        payload[f"{name}/current"] = port.current.detach().cpu().clone()
        payload[f"{name}/dft_samples"] = torch.tensor(port.metadata["dft_samples"])
    for index, runtime in enumerate(result.solver._port_runtimes):
        payload[f"runtime{index}/sample_index"] = torch.tensor(runtime.sample_index)
        for key, value in (("drive_buffer", runtime.drive_buffer), ("magnetic_time", runtime.magnetic_time),
                           ("voltage_sum", runtime.accumulator._voltage_sum),
                           ("current_sum", runtime.accumulator._current_sum)):
            payload[f"runtime{index}/{key}"] = value.detach().cpu().clone()
    return payload


def _assert_bitwise(actual, expected):
    actual, expected = _payload(actual), expected if isinstance(expected, dict) else _payload(expected)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype and torch.equal(actual[name], expected[name]), name


def _existing():
    """The frozen uninterrupted existing-path run of ``_simulation()``."""
    return frozen_oracles.load("tests/rf/lumped/test_port_step_graph.py::existing_203")


@pytest.mark.parametrize("source_time", [
    PULSE,
    mw.CW(frequency=0.5e9),
    mw.sources.RickerWavelet(frequency=0.5e9),
    mw.sources.BroadbandPulse(freq_range=(0.3e9, 0.7e9)),
    mw.sources.CustomSourceTime(torch.linspace(0.0, 12e-9, 241, dtype=torch.float64),
                        torch.sin(2 * math.pi * 0.5e9 * torch.linspace(0.0, 12e-9, 241, dtype=torch.float64)),
                        interpolation="cubic"),
], ids=["gaussian", "cw", "ricker", "broadband", "custom"])
def test_drive_table_rows_equal_the_eager_per_step_drive(source_time):
    prepared = _simulation(source_time=source_time).prepare()
    solver = prepared.solver
    runtime = solver._port_runtimes[0]
    steps = 1500
    prepare_port_drive_tables(solver, steps)
    eager = []
    for _ in range(steps):
        eager.append(_evaluate_drive(runtime).clone())
        _advance_port_sample_time(solver, runtime)
    assert torch.equal(runtime._drive_table, torch.stack(eager))
    # A resumed run starts from the restored clock and sample index.
    runtime.sample_index = 1000
    runtime.magnetic_time.copy_(torch.as_tensor(0.0, dtype=torch.float64))
    for _ in range(1000):
        _advance_port_sample_time(solver, runtime)
    prepare_port_drive_tables(solver, steps)
    tail = []
    for _ in range(steps - 1000):
        tail.append(_evaluate_drive(runtime).clone())
        _advance_port_sample_time(solver, runtime)
    assert torch.equal(runtime._drive_table[1000:], torch.stack(tail))
    assert not torch.any(runtime._drive_table[:1000])


def test_automatic_default_runs_core_and_matches_the_existing_port_graph():
    existing = _existing()
    automatic = _simulation().run()
    assert automatic.solver._core_runtime is not None
    actual = _payload(automatic)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(actual[name], existing[name]), name
    for name in ("p1", "p2"):
        assert torch.equal(actual[f"{name}/voltage"], existing[f"{name}/voltage"]), name
        assert torch.equal(actual[f"{name}/current"], existing[f"{name}/current"]), name


@pytest.mark.parametrize("stop", [1, 2, 64, 101])
def test_resume_across_the_port_graph_matches_the_uninterrupted_run(stop, tmp_path):
    # The core path resumes inside its whole-step graph.
    expected = _existing()
    checkpoint = _simulation().prepare().run_until(stop)
    path = tmp_path / "checkpoint.pt"
    checkpoint.save(path)
    loaded = mw.fdtd.FDTDResumeCheckpoint.load(path, map_location="cpu")
    resumed = _simulation().prepare().run(resume_from=loaded)
    assert resumed.solver._core_runtime is not None
    assert resumed.solver._core_step_graph_active
    assert sum(resumed.solver._core_step_graph_chunks) == 203 - stop
    _assert_bitwise(resumed, expected)


def test_circuit_bound_ports_keep_the_eager_port_phase():
    result = _simulation(circuit=True).run()
    assert result.stats()["cuda_graph_active"]
    assert not result.stats()["port_cuda_graph_active"]
