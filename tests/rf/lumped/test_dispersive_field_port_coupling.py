"""Electric ADE history enters the same-step field/circuit free response."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.lumped import apply_lumped_runtime, prepare_lumped_runtime


def test_history_voltage_matches_independent_final_field_block_and_energy():
    dtype = torch.float64
    eps = torch.tensor([1.3, 2.7], dtype=dtype)
    sigma = torch.tensor([.8, 3.1], dtype=dtype)
    volume = torch.tensor([.2, .7], dtype=dtype)
    weights = torch.tensor([.3, -.5], dtype=dtype)
    dt = .1
    coefficient = dt / (eps + sigma * dt / 2)
    currents = (torch.tensor([.4, -.1], dtype=dtype).reshape(2, 1, 1),
                torch.tensor([.2, .3], dtype=dtype).reshape(2, 1, 1))
    geometry = SimpleNamespace(port_name="history", voltage_weights=weights,
                               voltage_indices=torch.tensor([[0, 0, 0], [1, 0, 0]]))
    runtime = prepare_lumped_runtime(geometry, dt=dt, eps_edge=eps.reshape(2, 1, 1),
                                    yee_control_volume=volume.reshape(2, 1, 1),
                                    electric_update_coefficient=coefficient.reshape(2, 1, 1),
                                    electric_history_currents=currents, resistance=.7, diagnostics=True)
    eold = torch.tensor([.8, -.3], dtype=dtype)
    total_j = sum(current.flatten() for current in currents)
    free = (eps - sigma * dt / 2) / (eps + sigma * dt / 2) * eold
    runtime.last_voltage_after.copy_(weights.dot(eold))
    field = free.reshape(2, 1, 1).clone()
    branch = apply_lumped_runtime(runtime, field)
    field.flatten().sub_(coefficient * total_j)
    mass = eps * volume
    matrix = torch.diag(mass / dt + sigma * volume / 2) + torch.outer(weights, weights) / (2 * .7)
    rhs = ((mass / dt - sigma * volume / 2) * eold
           - volume * total_j - weights * weights.dot(eold) / (2 * .7))
    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(field.flatten(), expected, rtol=1.e-12, atol=1.e-13)
    torch.testing.assert_close(runtime.last_voltage_after, weights.dot(expected), rtol=1.e-12, atol=1.e-13)
    midpoint = (eold + expected) / 2
    delta_field = .5 * torch.sum(mass * (expected.square() - eold.square()))
    conductive = dt * torch.sum(sigma * volume * midpoint.square())
    polarization_work = dt * torch.sum(volume * total_j * midpoint)
    residual = delta_field + conductive + polarization_work + dt * .7 * branch.square()
    torch.testing.assert_close(residual, residual.new_zeros(()), rtol=0, atol=1.e-13)


def _material(kind, sigma):
    poles = {
        "debye": {"debye_poles": (mw.DebyePole(delta_eps=1.5, tau=1.e-10),)},
        "drude": {"drude_poles": (mw.DrudePole(plasma_frequency=2.e9, gamma=.2e9),)},
        "lorentz": {"lorentz_poles": (mw.LorentzPole(delta_eps=1.5, resonance_frequency=2.5e9, gamma=.3e9),)},
    }
    return mw.Material(eps_r=2., sigma_e=sigma, **poles[kind])


def _scene(kind, sigma, remote, mna):
    port = mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
                         voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                             position=(0., 0., -.0025), size=(.015, .015, 0.)),
                         termination=None if mna else mw.SeriesRLC(r=25., l=.5e-9, c=1.e-12))
    circuits = ()
    if mna:
        network = mw.Circuit("load")
        a, b, c = network.node("a"), network.node("b"), network.node("c")
        network.add(mw.Resistor("r", a, b, 25.))
        network.add(mw.Inductor("l", b, c, .5e-9))
        network.add(mw.Capacitor("c", c, network.ground, 1.e-12))
        network.bind_port("feed", positive=a, negative=network.ground)
        circuits = (network,)
    return mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
                    boundary=mw.BoundarySpec.none(), ports=(port,), circuits=circuits,
                    structures=(mw.Structure(geometry=mw.Box(
                        position=(.0125, 0., 0.) if remote else (0., 0., 0.),
                        size=(.005, .02, .02) if remote else (.03, .03, .03)),
                        material=_material(kind, sigma)),), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
@pytest.mark.parametrize("sigma", (0., .2))
@pytest.mark.parametrize("remote", (False, True))
@pytest.mark.parametrize("mna", (False, True))
def test_gpu_anticipated_history_matches_apply_material_first_reference(kind, sigma, remote, mna):
    from witwin.maxwell.fdtd.circuits import prepare_circuit_time_series
    from witwin.maxwell.fdtd.ports import apply_port_runtimes
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block
    solvers = [mw.Simulation.fdtd(_scene(kind, sigma, remote, mna), frequency=3.e9).prepare().solver
               for _ in range(2)]
    production, reference = solvers
    for solver in solvers:
        if mna:
            prepare_circuit_time_series(solver, 64)
        solver.Ez[4, 4, 3:5] = 1.
        port = solver._port_runtimes[0]
        state = port.circuit_port.field if mna else port.lumped
        state.last_voltage_after.copy_(state.voltage_weights.dot(
            solver.Ez.flatten().index_select(0, state.linear_indices)))
    ref_port = reference._port_runtimes[0]
    (ref_port.circuit_port.field if mna else ref_port.lumped).electric_history = None
    production_port = production._port_runtimes[0]
    state = production_port.circuit_port.field if mna else production_port.lumped
    assert (state.electric_history is None) == remote
    transferred = 0.
    for step in range(64):
        for solver in solvers:
            _field_update_block(solver, step * solver.dt)
        # This reference physically deposits each material current before the
        # existing circuit solve; production forecasts the same voltage and
        # deposits the current in the original material tail.
        reference._apply_dispersive_corrections()
        apply_port_runtimes(reference)
        apply_port_runtimes(production)
        production._apply_dispersive_corrections()
        assert float(torch.max(torch.abs(production.Ez - reference.Ez))) < 1.e-5
        actual_voltage = state.voltage_weights.dot(production.Ez.flatten().index_select(0, state.linear_indices))
        torch.testing.assert_close(state.last_voltage_after, actual_voltage, rtol=2.e-5, atol=2.e-9)
        transferred += abs(float(state.last_current if mna else state.last_branch_current))
    assert transferred > 1.e-7
    assert torch.isfinite(production.Ez).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_public_dispersive_lossy_source_run_returns_nonzero_power(kind):
    scene = _scene(kind, .2, False, False)
    # Use an un-terminated source port with the same geometric support.
    old = scene.ports[0]
    scene.ports = (mw.LumpedPort(name=old.name, positive=old.positive, negative=old.negative,
                                voltage_path=old.voltage_path, current_surface=old.current_surface),)
    result = mw.Simulation.fdtd(scene, frequencies=(2.5e9, 3.e9),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=1.e9)),
        run_time=mw.TimeConfig(time_steps=256), spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False).run()
    port = result.port("feed")
    assert bool(torch.all(torch.isfinite(port.voltage)))
    assert bool(torch.all(port.accepted_power > 0.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_dispersive_port_graph_and_disk_resume_preserve_history(kind, tmp_path):
    def simulation(graph):
        scene = _scene(kind, .2, False, False)
        old = scene.ports[0]
        scene.ports = (mw.LumpedPort(name=old.name, positive=old.positive, negative=old.negative,
                                    voltage_path=old.voltage_path, current_surface=old.current_surface),)
        return mw.Simulation.fdtd(scene, frequency=3.e9,
            excitations=mw.PortExcitation("feed", source_time=mw.CW(3.e9)),
            run_time=mw.TimeConfig(time_steps=96), spectral_sampler=mw.SpectralSampler(window="none"),
            cuda_graph=graph, full_field_dft=False)
    eager = simulation(False).run()
    graph = simulation(True).run()
    assert graph.stats()["cuda_graph_active"]
    torch.testing.assert_close(graph.port("feed").voltage, eager.port("feed").voltage, rtol=2.e-5, atol=1.e-8)
    prepared = simulation(True).prepare()
    checkpoint = prepared.run_until(37)
    path = tmp_path / f"{kind}.pt"
    checkpoint.save(path)
    resumed = simulation(True).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    torch.testing.assert_close(resumed.port("feed").voltage, graph.port("feed").voltage, rtol=2.e-5, atol=1.e-8)
    torch.testing.assert_close(resumed.port("feed").current, graph.port("feed").current, rtol=2.e-5, atol=1.e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_uncoupled_conductive_ade_material_adjoint_matches_finite_difference(kind):
    class Design(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.zeros((1, 1, 1), device="cuda"))

        def to_scene(self):
            return mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.24, .24),) * 3),
                grid=mw.GridSpec.uniform(.12), boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.),
                structures=(mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(.24,) * 3),
                                         material=_material(kind, .1)),),
                material_regions=(mw.MaterialRegion(name="design", geometry=mw.Box(
                    position=(0., 0., .06), size=(.12,) * 3), density=torch.sigmoid(self.logit),
                    eps_bounds=(1., 6.)),),
                sources=(mw.PointDipole(position=(0., 0., -.06), polarization="Ez", width=.04,
                    source_time=mw.GaussianPulse(frequency=1.e9, fwidth=.25e9, amplitude=50.)),),
                monitors=(mw.PointMonitor("probe", (0., 0., .06), fields=("Ez",)),), device="cuda")
    model = Design()
    def objective():
        data = mw.Simulation.fdtd(model, frequency=1.e9, run_time=mw.TimeConfig(time_steps=36),
                                 spectral_sampler=mw.SpectralSampler(window="none"),
                                 full_field_dft=False).run().monitor("probe")["data"]
        return data.abs().square().sum()
    loss = objective()
    loss.backward()
    actual = model.logit.grad.detach().clone()
    delta = .01
    with torch.no_grad():
        model.logit.fill_(delta)
    plus = objective().detach()
    with torch.no_grad():
        model.logit.fill_(-delta)
    minus = objective().detach()
    expected = (plus - minus) / (2 * delta)
    assert float(actual.abs().max()) > 0.
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=.01, atol=1.e-15)
