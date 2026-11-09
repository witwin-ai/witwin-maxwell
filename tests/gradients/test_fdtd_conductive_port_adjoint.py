"""Fixed-topology conductive material gradients through RF sources and MNA."""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class ConductiveDesign(mw.SceneModule):
    def __init__(self, *, ade=False, mna=False, graph=False, sigma_zero=False, checkpoint_stride=None):
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.tensor(math.log(.2), device="cuda"))
        self.log_eps = torch.nn.Parameter(torch.tensor(math.log(2.), device="cuda"))
        self.log_resistance = torch.nn.Parameter(torch.tensor(math.log(25.), device="cuda"))
        self.ade, self.mna, self.graph = ade, mna, graph
        self.sigma_zero = sigma_zero
        self.checkpoint_stride = checkpoint_stride

    def to_scene(self):
        circuits = ()
        if self.mna:
            network = mw.Circuit("load")
            node = network.node("input")
            network.add(mw.Resistor("r", node, network.ground, self.log_resistance.exp()))
            network.add(mw.CurrentSource("source", network.ground, node,
                                        waveform=mw.SineWaveform(0., .01, 3.e9)))
            network.bind_port("feed", positive=node, negative=network.ground)
            circuits = (network,)
        return mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.none(), device="cuda", circuits=circuits,
            ports=(mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
                voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                    position=(0., 0., -.0025), size=(.015, .015, 0.))),),
            structures=(mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(.03, .03, .03)),
                material=mw.Material(eps_r=self.log_eps.exp(), sigma_e=0. if self.sigma_zero else self.log_sigma.exp(),
                    **({"debye_poles": (mw.DebyePole(delta_eps=1.5, tau=1.e-10),)} if self.ade == "debye" else
                       {"drude_poles": (mw.DrudePole(plasma_frequency=2.e9, gamma=.2e9),)} if self.ade == "drude" else
                       {"lorentz_poles": (mw.LorentzPole(delta_eps=1.5, resonance_frequency=2.5e9, gamma=.3e9),)} if self.ade else {}))),))

    def simulation(self, *, checkpoint_stride=None):
        kwargs = {} if self.mna else {"excitations": mw.PortExcitation("feed", source_time=mw.CW(3.e9))}
        simulation = mw.Simulation.fdtd(self, frequency=3.e9, run_time=mw.TimeConfig(time_steps=32),
                                  cuda_graph=self.graph, spectral_sampler=mw.SpectralSampler(window="none"), **kwargs)
        simulation.config.adjoint_checkpoint_stride = checkpoint_stride or self.checkpoint_stride
        return simulation



def _objective(model, kind="voltage"):
    result = model.simulation().run()
    data = result.port("feed")
    if kind == "voltage":
        return data.voltage.abs().square().sum()
    if kind == "power":
        return data.accepted_power.sum()
    reflection = data.b / data.a
    if kind == "complex_reflection":
        return reflection.real.sum() + .37 * reflection.imag.sum()
    return reflection.abs().square().sum()


@pytest.mark.parametrize("ade", (False, "debye", "drude", "lorentz"))
@pytest.mark.parametrize("mna", (False, True))
def test_sigma_eps_and_circuit_gradients_match_step_scan(ade, mna, record_property):
    model = ConductiveDesign(ade=ade, mna=mna)
    loss = _objective(model)
    loss.backward()
    names = ("log_sigma", "log_eps", "log_resistance") if mna else ("log_sigma", "log_eps")
    gradients = {name: getattr(model, name).grad.detach().clone() for name in names}
    model.requires_grad_(False)
    normal = _objective(model)
    torch.testing.assert_close(normal.to(loss), loss.detach(), rtol=2e-4, atol=1e-12)
    for name in names:
        parameter = getattr(model, name)
        actual = gradients[name]
        base = parameter.detach().clone()
        assert actual.isfinite() and abs(float(actual)) > 1.e-12
        errors = []
        for delta in (.02, .01, .005):
            with torch.no_grad():
                parameter.copy_(base + delta)
            plus = _objective(model).detach()
            with torch.no_grad():
                parameter.copy_(base - delta)
            minus = _objective(model).detach()
            with torch.no_grad():
                parameter.copy_(base)
            expected = (plus - minus) / (2 * delta)
            errors.append(float((actual - expected).abs() / expected.abs().clamp_min(1.e-12)))
        record_property(name + "_relative_errors", repr(errors))
        assert sum(error < .01 for error in errors) >= 2, (name, actual, errors)


@pytest.mark.parametrize("mna", (False, True))
@pytest.mark.parametrize("kind", ("power", "reflection", "complex_reflection"))
def test_lossy_network_objective_sigma_gradient_matches_finite_difference(mna, kind, record_property):
    model = ConductiveDesign(ade="debye", mna=mna)
    objective = _objective(model, kind)
    objective.backward()
    actual = model.log_sigma.grad.detach().clone()
    model.requires_grad_(False)
    normal = _objective(model, kind)
    torch.testing.assert_close(normal.to(objective), objective.detach(), rtol=2e-4, atol=1e-12)
    base = model.log_sigma.detach().clone()
    errors = []
    for delta in (.02, .01, .005):
        with torch.no_grad():
            model.log_sigma.copy_(base + delta)
        plus = _objective(model, kind)
        with torch.no_grad():
            model.log_sigma.copy_(base - delta)
        minus = _objective(model, kind)
        with torch.no_grad():
            model.log_sigma.copy_(base)
        expected = (plus - minus) / (2 * delta)
        errors.append(float((actual - expected.to(actual)).abs() / expected.abs().clamp_min(1e-30)))
        if delta == .01:
            # Preserve the original single-step gate as well as adding a scan.
            torch.testing.assert_close(actual, expected.to(actual), rtol=.01, atol=1.e-10)
    record_property("log_sigma_relative_errors", repr(errors))
    assert sum(error < .01 for error in errors) >= 2, (kind, mna, actual, errors)


@pytest.mark.parametrize("mna", (False, True))
def test_graph_checkpoint_and_lossless_ade_material_gradients(mna):
    values = []
    for graph, stride in ((False, 1), (True, 5), (True, 11)):
        model = ConductiveDesign(ade="debye", mna=mna, graph=graph, checkpoint_stride=stride)
        result = model.simulation().run()
        result.port("feed").voltage.abs().square().sum().backward()
        if graph:
            assert result.stats()["cuda_graph_active"]
        values.append(torch.stack([model.log_sigma.grad, model.log_eps.grad]))
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], rtol=.001, atol=1.e-9)
    model = ConductiveDesign(ade="debye", mna=mna, sigma_zero=True)
    loss = _objective(model)
    loss.backward()
    actual = model.log_eps.grad.detach().clone()
    model.requires_grad_(False)
    normal = _objective(model)
    torch.testing.assert_close(normal.to(loss), loss.detach(), rtol=2e-4, atol=1e-12)
    base = model.log_eps.detach().clone()
    with torch.no_grad():
        model.log_eps.copy_(base + .01)
    plus = _objective(model).detach()
    with torch.no_grad():
        model.log_eps.copy_(base - .01)
    minus = _objective(model).detach()
    torch.testing.assert_close(actual, ((plus-minus)/.02).to(actual), rtol=.01, atol=1.e-10)


def test_direct_scene_retains_scalar_conductivity_tensor_connection():
    model = ConductiveDesign(ade="debye")
    first = _objective(model)
    first.backward()
    expected = model.log_sigma.grad.detach().clone()
    model.zero_grad()
    simulation = mw.Simulation.fdtd(model.to_scene(), frequency=3.e9,
        run_time=mw.TimeConfig(time_steps=32), spectral_sampler=mw.SpectralSampler(window="none"),
        excitations=mw.PortExcitation("feed", source_time=mw.CW(3.e9)))
    simulation.run().port("feed").voltage.abs().square().sum().backward()
    torch.testing.assert_close(model.log_sigma.grad, expected, rtol=.001, atol=1.e-9)


def test_lossy_matching_optimization_improves_and_rechecks_forward(record_property):
    model = ConductiveDesign(ade="debye")
    optimizer = torch.optim.Adam((model.log_sigma, model.log_eps), lr=.05)
    initial = float(_objective(model, "reflection").detach())
    for _ in range(8):
        optimizer.zero_grad()
        objective = _objective(model, "reflection")
        objective.backward()
        optimizer.step()
    final = float(_objective(model, "reflection").detach())
    record_property("initial_reflected_power_fraction", initial)
    record_property("final_reflected_power_fraction", final)
    assert final < initial * .98
    model.requires_grad_(False)
    independent_forward = float(_objective(model, "reflection"))
    assert abs(independent_forward - final) < 1.e-5


def test_zero_initial_conductivity_has_nonzero_one_sided_material_gradient():
    class ZeroConductivity(ConductiveDesign):
        def __init__(self):
            super().__init__()
            self.sigma = torch.nn.Parameter(torch.zeros((), device="cuda"))

        def to_scene(self):
            scene = super().to_scene()
            scene.structures = (mw.Structure(geometry=scene.structures[0].geometry,
                material=mw.Material(eps_r=2., sigma_e=self.sigma)),)
            return scene
    model = ZeroConductivity()
    base = _objective(model)
    base.backward()
    actual = model.sigma.grad.detach().clone()
    model.requires_grad_(False)
    normal = _objective(model)
    torch.testing.assert_close(normal.to(base), base.detach(), rtol=2e-4, atol=1e-12)
    with torch.no_grad():
        model.sigma.fill_(.002)
    first = _objective(model).detach()
    with torch.no_grad():
        model.sigma.fill_(.004)
    second = _objective(model).detach()
    expected = (-3 * base.detach() + 4 * first - second) / .004
    assert abs(float(actual)) > 1.e-10
    torch.testing.assert_close(actual, expected.to(actual), rtol=.01, atol=1.e-9)
