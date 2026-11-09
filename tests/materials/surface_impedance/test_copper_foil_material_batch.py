"""Local and remote bulk ADE retain native batch lane state and RF timing."""
import pytest
import torch
import witwin.maxwell as mw
from tests.materials.surface_impedance.test_copper_foil_batch import _simulation
from tests.materials.surface_impedance.test_copper_foil_material_ade import _local_material
from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot
from witwin.maxwell.fdtd.runtime.batch_materials import material_state_tensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _material_simulation(index, kind, *, rf=False):
    base = _simulation(index, steps=64)
    scene = base.scene.clone(structures=(*base.scene.structures, mw.Structure(name="substrate",
        geometry=mw.Box(position=(.01, 0., 0.), size=(.02, .04, .04)), material=_local_material(kind))))
    kwargs = {}
    if rf:
        port = mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)),
            termination=None)
        load = mw.LumpedPort(name="load", positive=(.005, 0., .005), negative=(.005, 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(.005, 0., -.0025), size=(.015, .015, 0.)),
            termination=mw.SeriesRLC(r=75., l=2e-9, c=1e-12))
        scene = scene.clone(sources=(), ports=(port, load))
        kwargs["excitations"] = mw.PortExcitation("feed", amplitude=1., source_time=mw.CW(frequency=1e9))
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=64, dt=5e-12),
        spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=True, cuda_graph=False, **kwargs)


@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz", "magnetic"))
@pytest.mark.parametrize("rf", (False, True))
def test_material_foil_native_graph_batch_matches_independent_and_repeated(kind, rf):
    simulations = [_material_simulation(index, kind, rf=rf) for index in range(2)]
    expected = [simulation.run() for simulation in simulations]
    batch = mw.Simulation.batch(simulations, variation=mw.BatchVariation(sources=() if rf else ("drive",)), outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run().results()
        second = prepared.run().results()
        assert prepared.report.execution_tier == "native_pipeline_graph"
    for lane, reference in enumerate(expected):
        for actual in (first[lane], second[lane]):
            torch.testing.assert_close(actual.tensor("Ez"), reference.tensor("Ez"), rtol=2e-4, atol=1e-6)
            assert actual.conductor_losses["foil"]["dissipated_energy"] == pytest.approx(
                reference.conductor_losses["foil"]["dissipated_energy"], rel=2e-4)
            if rf:
                for name in ("voltage", "current"):
                    torch.testing.assert_close(getattr(actual.port("feed"), name), getattr(reference.port("feed"), name), rtol=2e-4, atol=1e-8)


@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz", "magnetic"))
def test_inactive_material_batch_lane_freezes_nonzero_histories(kind):
    solvers = [_material_simulation(index, kind).prepare().solver for index in range(2)]
    slot = NativeFieldSlot(2, "cuda:0")
    slot.bind(solvers)
    states = material_state_tensors(solvers[0]) + material_state_tensors(solvers[0], magnetic=True)
    for tensor in states:
        tensor.fill_(.1)
    before = tuple(tensor.clone() for tensor in states)
    fields = tuple(getattr(solvers[0], name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    slot.active[0] = 0
    for _ in range(3):
        slot.advance()
    for tensor, initial in zip(states, before):
        torch.testing.assert_close(tensor, initial, rtol=0, atol=0)
    for name, initial in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), fields):
        torch.testing.assert_close(getattr(solvers[0], name), initial, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
@pytest.mark.parametrize("graph", ("off", "required"))
def test_material_mna_native_batch_matches_independent(kind, graph):
    from tests.rf.lumped.test_dispersive_field_port_coupling import _scene
    simulations = []
    for index in range(2):
        scene = _scene(kind, .2, False, True)
        scene.sources = (mw.PointDipole(name="drive", position=(-.01, 0., 0.), polarization="Ez", profile="ideal",
                                       source_time=mw.CW(3.e9, amplitude=index+1.)),)
        simulations.append(mw.Simulation.fdtd(scene, frequency=3.e9,
            run_time=mw.TimeConfig(time_steps=64), full_field_dft=True,
            spectral_sampler=mw.SpectralSampler(window="none")))
    expected = [simulation.run() for simulation in simulations]
    batch = mw.Simulation.batch(simulations, variation=mw.BatchVariation(sources=("drive",)),
        outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph=graph, graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run().results()
        second = prepared.run().results()
    for lane, reference in enumerate(expected):
        for actual in (first[lane], second[lane]):
            torch.testing.assert_close(actual.tensor("Ez"), reference.tensor("Ez"), rtol=2e-4, atol=1e-6)
            for name in ("voltage", "current"):
                torch.testing.assert_close(getattr(actual.port("feed"), name), getattr(reference.port("feed"), name), rtol=2e-4, atol=1e-8)
            for name in ("node_voltages", "branch_currents", "energy_balance"):
                torch.testing.assert_close(getattr(actual.circuits["load"], name), getattr(reference.circuits["load"], name), rtol=2e-4, atol=1e-8)


@pytest.mark.parametrize("graph", ("off", "required"))
def test_material_mna_pulse_switch_factor_classes_match_independent(graph):
    from tests.rf.lumped.test_dispersive_field_port_coupling import _scene
    scene = _scene("debye", .2, False, True)
    circuit = mw.Circuit("events")
    source, node = circuit.node("source"), circuit.node("node")
    circuit.add(mw.VoltageSource("voltage", source, circuit.ground,
        waveform=mw.PulseWaveform(initial=0., pulsed=1., delay=4e-12, rise=10e-12,
                                 fall=10e-12, width=8e-12, period=40e-12)))
    circuit.add(mw.Resistor("r", source, node, 50.))
    circuit.add(mw.Capacitor("c", node, circuit.ground, 1e-12))
    circuit.add(mw.TimedSwitch("switch", node, circuit.ground, transition_times=(7.5e-12, 15.5e-12),
                               on_resistance=25., off_resistance=1e6))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    scene.circuits = (circuit,)
    simulation = mw.Simulation.fdtd(scene, frequency=3.e9,
        run_time=mw.TimeConfig(time_steps=96), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none"))
    reference = simulation.run()
    batch = mw.Simulation.batch([simulation, simulation], outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph=graph, graph_steps=7))
    with batch.prepare() as prepared:
        result = prepared.run().results()
        repeated = prepared.run().results()
        switch_keys = {key for runtime in prepared._slot.solvers[0]._circuit_runtimes for key in runtime.switch_keys}
        assert len(switch_keys) > 1
        keys = {key for runtime in prepared._slot.solvers[0]._circuit_runtimes for key in runtime.integration_keys}
        assert len(keys) > 1
    for actual in (*result, *repeated):
        torch.testing.assert_close(actual.tensor("Ez"), reference.tensor("Ez"), rtol=2e-4, atol=1e-8)
        for name in ("node_voltages", "branch_currents", "energy_balance"):
            torch.testing.assert_close(getattr(actual.circuits["events"], name), getattr(reference.circuits["events"], name), rtol=2e-4, atol=1e-8)


@pytest.mark.parametrize("kind", ("debye", "lorentz"))
def test_material_state_space_network_batch_matches_independent(kind):
    from tests.rf.lumped.test_dispersive_field_port_coupling import _scene
    from tests.rf.network.test_network_block_contract import _port
    model = mw.StateSpaceNetwork(A=torch.tensor([[-2e9]], dtype=torch.float64),
        B=torch.tensor([[2e9]], dtype=torch.float64), C=torch.tensor([[-.01]], dtype=torch.float64),
        D=torch.tensor([[.02]], dtype=torch.float64), representation="Y", port_order=("load",))
    frequencies = torch.tensor([0., 3e9, 4e9], dtype=torch.float64)
    data = mw.NetworkData.from_y(frequencies=frequencies, y=model.evaluate(frequencies), z0=50., port_names=("load",))
    network = mw.NetworkBlock("matching", network=data, model=model, fit=False, connections={"load": "feed"})
    scene = _scene(kind, .2, False, False)
    scene.ports, scene.networks = (_port(),), (network,)
    scene.sources = (mw.PointDipole(name="drive", position=(-.01, 0., 0.), polarization="Ez", profile="ideal", source_time=mw.CW(3.e9)),)
    simulation = mw.Simulation.fdtd(scene, frequency=3.e9, run_time=mw.TimeConfig(time_steps=64),
        full_field_dft=True, spectral_sampler=mw.SpectralSampler(window="none"))
    reference = simulation.run()
    batch = mw.Simulation.batch([simulation, simulation], outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        results = prepared.run().results()
        repeated = prepared.run().results()
    for actual in (*results, *repeated):
        torch.testing.assert_close(actual.tensor("Ez"), reference.tensor("Ez"), rtol=2e-4, atol=1e-6)
        for name in ("voltage", "current"):
            torch.testing.assert_close(getattr(actual.port("feed"), name), getattr(reference.port("feed"), name), rtol=2e-4, atol=1e-8)


def test_material_mna_native_batch_reuses_graphs_and_table_addresses():
    from tests.rf.lumped.test_dispersive_field_port_coupling import _scene
    scene = _scene("debye", .2, False, True)
    scene.sources = (mw.PointDipole(name="drive", position=(-.01, 0., 0.), polarization="Ez", profile="ideal", source_time=mw.CW(3.e9)),)
    simulation = mw.Simulation.fdtd(scene, frequency=3.e9, run_time=mw.TimeConfig(time_steps=64),
        full_field_dft=True, spectral_sampler=mw.SpectralSampler(window="none"))
    batch = mw.Simulation.batch([simulation, simulation], outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        prepared.run().results()
        graphs = tuple(prepared._graphs.values())
        runtime = prepared._slot.solvers[0]._circuit_runtimes[0]
        pointers = tuple(tensor.data_ptr() for tensor in (runtime.node_samples, runtime.branch_samples, runtime.sample_times,
            *(value for factors in runtime.factor_cache.values() for value in factors)))
        prepared.run().results()
        assert graphs == tuple(prepared._graphs.values())
        assert pointers == tuple(tensor.data_ptr() for tensor in (runtime.node_samples, runtime.branch_samples, runtime.sample_times,
            *(value for factors in runtime.factor_cache.values() for value in factors)))


def test_material_mna_native_batch_gradient_matches_independent():
    import math
    from tests.rf.lumped.test_dispersive_field_port_coupling import _scene
    class Design(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.log_eps = torch.nn.Parameter(torch.tensor(math.log(2.), dtype=torch.float64, device="cuda"))
            self.log_sigma = torch.nn.Parameter(torch.tensor(math.log(.2), dtype=torch.float64, device="cuda"))

        def to_scene(self):
            scene = _scene("debye", .2, False, True)
            old = scene.structures[0]
            scene.structures = (mw.Structure(geometry=old.geometry, material=mw.Material(
                eps_r=self.log_eps.exp(), sigma_e=self.log_sigma.exp(), debye_poles=old.material.debye_poles)),)
            scene.sources = (mw.PointDipole(name="drive", position=(-.01, 0., 0.), polarization="Ez", profile="ideal", source_time=mw.CW(3.e9)),)
            return scene

        def simulation(self):
            result = mw.Simulation.fdtd(self, frequency=3.e9, run_time=mw.TimeConfig(time_steps=32),
                full_field_dft=True, spectral_sampler=mw.SpectralSampler(window="none"))
            result.config.adjoint_checkpoint_stride = 7
            return result
    model = Design()
    independent = model.simulation().run().tensor("Ez").abs().square().sum()
    independent.backward()
    expected = torch.stack((model.log_eps.grad, model.log_sigma.grad)).detach().clone()
    model.zero_grad(set_to_none=True)
    batch = mw.Simulation.batch([model.simulation()], outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        output = prepared.run().results()[0]
        objective = output.tensor("Ez").abs().square().sum()
        objective.backward()
    torch.testing.assert_close(objective, independent, rtol=2e-4, atol=1e-8)
    torch.testing.assert_close(torch.stack((model.log_eps.grad, model.log_sigma.grad)), expected, rtol=.001, atol=1e-8)
