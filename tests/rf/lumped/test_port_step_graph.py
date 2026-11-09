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
        net = mw.Circuit("load")
        node = net.node("input")
        net.add(mw.Resistor("R1", node, net.ground, 50.0))
        net.bind_port("p2", positive=node, negative=net.ground)
        circuits = (net,)
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        structures=structures,
        ports=ports,
        lumped_elements=(mw.Resistor("shunt", positive=(1.4, 0.5, 0.6), negative=(1.4, 0.5, 0.4), resistance=100.0),),
        circuits=circuits,
        device="cuda",
    )


def _simulation(*, steps=203, cuda_graph=True, circuit=False, source_time=PULSE, core_path=False):
    # These checks cover the existing runner's port graph; under the automatic
    # default the eligible lumped-port scene would take the core whole-step graph.
    return mw.Simulation.fdtd(
        _scene(circuit=circuit),
        frequencies=(0.4e9, 0.6e9),
        excitations=mw.PortExcitation("p1", source_time=source_time),
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=cuda_graph,
        fdtd_core_path=core_path,
    )


def _assert_bitwise(actual, expected):
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(actual.solver, name), getattr(expected.solver, name)), name
    for name in ("p1", "p2"):
        a, b = actual.port(name), expected.port(name)
        assert torch.equal(a.voltage, b.voltage) and torch.equal(a.current, b.current), name
        assert a.metadata["dft_samples"] == b.metadata["dft_samples"]
    for a, b in zip(actual.solver._port_runtimes, expected.solver._port_runtimes):
        assert a.sample_index == b.sample_index
        assert torch.equal(a.drive_buffer, b.drive_buffer)
        assert torch.equal(a.magnetic_time, b.magnetic_time)
        assert torch.equal(a.accumulator._voltage_sum, b.accumulator._voltage_sum)
        assert torch.equal(a.accumulator._current_sum, b.accumulator._current_sum)


@pytest.mark.parametrize("source_time", [
    PULSE,
    mw.CW(frequency=0.5e9),
    mw.RickerWavelet(frequency=0.5e9),
    mw.BroadbandPulse(freq_range=(0.3e9, 0.7e9)),
    mw.CustomSourceTime(torch.linspace(0.0, 12e-9, 241, dtype=torch.float64),
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


def test_graph_port_step_matches_the_eager_path_bitwise():
    graph = _simulation().run()
    eager = _simulation(cuda_graph=False).run()
    assert graph.solver_stats["port_cuda_graph_active"]
    assert not eager.solver_stats["port_cuda_graph_active"]
    _assert_bitwise(graph, eager)


def test_automatic_default_runs_core_and_matches_the_existing_port_graph():
    automatic = _simulation(core_path="auto").run()
    existing = _simulation().run()
    assert automatic.stats()["execution_path"] == "core"
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(automatic.solver, name), getattr(existing.solver, name)), name
    for name in ("p1", "p2"):
        a, b = automatic.port(name), existing.port(name)
        assert torch.equal(a.voltage, b.voltage) and torch.equal(a.current, b.current), name


@pytest.mark.parametrize("stop", [1, 2, 64, 101])
def test_resume_across_the_port_graph_matches_the_uninterrupted_run(stop, tmp_path):
    expected = _simulation().run()
    checkpoint = _simulation().prepare().run_until(stop)
    path = tmp_path / "checkpoint.pt"
    checkpoint.save(path)
    loaded = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    resumed = _simulation().prepare().run(resume_from=loaded)
    assert resumed.solver_stats["port_cuda_graph_active"]
    _assert_bitwise(resumed, expected)


def test_circuit_bound_ports_keep_the_eager_port_phase():
    result = _simulation(circuit=True).run()
    assert result.solver_stats["cuda_graph_active"]
    assert not result.solver_stats["port_cuda_graph_active"]
