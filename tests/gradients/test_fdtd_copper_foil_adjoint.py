"""Two-sided foil state and continuous-parameter gradients on a fixed planar grid."""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class FoilDesign(mw.SceneModule):
    def __init__(self, *, graph=False, stride=5, thickness=1.e-6, conductivity=1.e6, normal_axis=0):
        super().__init__()
        self.log_thickness = torch.nn.Parameter(torch.tensor(math.log(thickness), device="cuda", dtype=torch.float64))
        self.log_conductivity = torch.nn.Parameter(torch.tensor(math.log(conductivity), device="cuda", dtype=torch.float64))
        self.graph, self.stride = graph, stride
        self.normal_axis = normal_axis

    def to_scene(self):
        def rotate(values):
            return tuple(values[(axis - self.normal_axis) % 3] for axis in range(3))
        field_axis = "xyz"[(self.normal_axis + 2) % 3]
        return mw.Scene(domain=mw.Domain(bounds=((-0.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.pec(), device="cuda",
            structures=(mw.Structure(name="foil", geometry=mw.Box(position=(0., 0., 0.), size=rotate((0., .02, .02))),
                material=mw.CopperFoil(thickness=self.log_thickness.exp(), conductivity=self.log_conductivity.exp(),
                                       frequency_range=(2.e9, 4.e9), mode_count=16)),),
            ports=(mw.LumpedPort(name="feed", positive=rotate((-.01, 0., .005)), negative=rotate((-.01, 0., -.005)),
                voltage_path=mw.AxisPath(field_axis), current_surface=mw.Box(
                    position=rotate((-.01, 0., -.0025)), size=rotate((.015, .015, 0.)))),),
            monitors=(mw.PointMonitor("probe", rotate((.01, 0., 0.)), fields=("E" + field_axis,)),))

    def simulation(self):
        simulation = mw.Simulation.fdtd(self, frequency=3.e9, run_time=mw.TimeConfig(time_steps=48),
            cuda_graph=self.graph, spectral_sampler=mw.SpectralSampler(window="none"),
            excitations=mw.PortExcitation("feed", source_time=mw.CW(3.e9)))
        simulation.config.adjoint_checkpoint_stride = self.stride
        return simulation


def _objective(model, kind):
    result = model.simulation().run()
    if kind == "field":
        return result.monitor("probe")["data"].abs().square().sum(), result
    if kind in {"dissipated_energy", "stored_energy"}:
        return result.conductor_losses["foil"][kind], result
    port = result.port("feed")
    return (port.b / port.a).abs().square().sum(), result


@pytest.mark.parametrize("kind", ("field", "reflection", "dissipated_energy", "stored_energy"))
def test_foil_parameter_gradients_match_finite_difference_scan(kind, record_property):
    model = FoilDesign()
    value, _ = _objective(model, kind)
    assert value.requires_grad
    value.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    normal, _ = _objective(model, kind)
    torch.testing.assert_close(torch.as_tensor(normal, device=value.device, dtype=value.dtype),
                               value.detach(), rtol=2e-4, atol=1e-30)
    for name in ("log_thickness", "log_conductivity"):
        parameter = getattr(model, name)
        actual = gradients[name]
        base = parameter.detach().clone()
        errors = []
        # Include larger steps to distinguish a derivative plateau from float32
        # field noise in the small reflection derivative; retain all scan values.
        for delta in (.08, .04, .02, .01, .005):
            with torch.no_grad():
                parameter.copy_(base + delta)
            plus, _ = _objective(model, kind)
            with torch.no_grad():
                parameter.copy_(base - delta)
            minus, _ = _objective(model, kind)
            with torch.no_grad():
                parameter.copy_(base)
            expected = (torch.as_tensor(plus, device=actual.device, dtype=actual.dtype)
                        - torch.as_tensor(minus, device=actual.device, dtype=actual.dtype)) / (2 * delta)
            errors.append(float((actual - expected).abs() / expected.abs().clamp_min(1.e-30)))
        record_property(name + "_relative_errors", repr(errors))
        assert actual.isfinite() and abs(float(actual)) > 1.e-30
        assert sum(error < .01 for error in errors) >= 2, (kind, name, actual, errors)


def test_foil_graph_checkpoint_gradient_and_forward_reconstruction_agree():
    gradients = []
    losses = []
    for graph, stride in ((False, 1), (True, 7), (True, 13)):
        model = FoilDesign(graph=graph, stride=stride)
        value, result = _objective(model, "dissipated_energy")
        value.backward()
        if graph:
            assert result.stats()["cuda_graph_active"]
        gradients.append(torch.stack((model.log_thickness.grad, model.log_conductivity.grad)))
        losses.append(value.detach())
    for gradient, value in zip(gradients[1:], losses[1:]):
        torch.testing.assert_close(gradient, gradients[0], rtol=.001, atol=1.e-25)
        torch.testing.assert_close(value, losses[0], rtol=.001, atol=1.e-25)
    model.requires_grad_(False)
    normal, _ = _objective(model, "dissipated_energy")
    torch.testing.assert_close(torch.as_tensor(normal, device=losses[0].device, dtype=losses[0].dtype), losses[0], rtol=.001, atol=1.e-25)


def test_copper_skin_depth_regime_heat_gradient_matches_finite_difference(record_property):
    model = FoilDesign(thickness=2.e-6, conductivity=5.8e7)
    energy, _ = _objective(model, "dissipated_energy")
    energy.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    normal, _ = _objective(model, "dissipated_energy")
    torch.testing.assert_close(torch.as_tensor(normal, device=energy.device, dtype=energy.dtype),
                               energy.detach(), rtol=2e-4, atol=1e-30)
    for name in ("log_thickness", "log_conductivity"):
        parameter = getattr(model, name)
        actual = gradients[name]
        base = parameter.detach().clone()
        errors = []
        for delta in (.08, .04, .02, .01):
            with torch.no_grad():
                parameter.copy_(base + delta)
            plus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base - delta)
            minus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base)
            expected = (torch.as_tensor(plus, device=actual.device, dtype=actual.dtype)
                        - torch.as_tensor(minus, device=actual.device, dtype=actual.dtype)) / (2 * delta)
            errors.append(float((actual - expected).abs() / expected.abs().clamp_min(1.e-30)))
        record_property(name + "_relative_errors", repr(errors))
        assert sum(error < .01 for error in errors) >= 2, (name, actual, errors)


@pytest.mark.parametrize("axis", (1, 2))
def test_foil_material_gradients_are_consistent_under_axis_rotation(axis):
    evaluated = []
    for normal_axis in (0, axis):
        model = FoilDesign(normal_axis=normal_axis)
        energy, _ = _objective(model, "dissipated_energy")
        energy.backward()
        evaluated.append(torch.stack((energy.detach(), model.log_thickness.grad, model.log_conductivity.grad)))
    torch.testing.assert_close(evaluated[1], evaluated[0], rtol=.001, atol=1.e-25)


@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz", "magnetic"))
def test_local_foil_material_history_and_parameter_gradients(kind, record_property):
    from tests.materials.surface_impedance.test_copper_foil_material_ade import _local_material
    class LocalMaterialDesign(FoilDesign):
        def __init__(self):
            super().__init__(graph=True)
            self.log_substrate_eps = torch.nn.Parameter(torch.tensor(math.log(2.), device="cuda", dtype=torch.float64))

        def to_scene(self):
            scene = super().to_scene()
            medium = _local_material(kind)
            scene.structures = (*scene.structures, mw.Structure(name="substrate", geometry=mw.Box(
                position=(.01, 0., 0.), size=(.02, .04, .04)),
                material=mw.Material(eps_r=self.log_substrate_eps.exp(), sigma_e=.1,
                    debye_poles=medium.debye_poles, drude_poles=medium.drude_poles,
                    lorentz_poles=medium.lorentz_poles, mu_lorentz_poles=medium.mu_lorentz_poles)))
            return scene
    model = LocalMaterialDesign()
    energy, result = _objective(model, "dissipated_energy")
    assert result.stats()["cuda_graph_active"]
    energy.backward()
    gradients = {name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.requires_grad_(False)
    normal, _ = _objective(model, "dissipated_energy")
    torch.testing.assert_close(torch.as_tensor(normal, dtype=energy.dtype, device=energy.device),
                               energy.detach(), rtol=2e-4, atol=1e-25)
    for name in ("log_thickness", "log_conductivity", "log_substrate_eps"):
        parameter = getattr(model, name)
        actual = gradients[name]
        base = parameter.detach().clone()
        errors = []
        # Retain the original large-step controls. Near a magnetic resonance
        # the epsilon derivative needs smaller perturbations to expose the
        # expected central-difference convergence plateau.
        finite_values = []
        for delta in (.04, .02, .01, .005, .0025):
            with torch.no_grad():
                parameter.copy_(base + delta)
            plus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base - delta)
            minus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base)
            expected = (torch.as_tensor(plus, dtype=actual.dtype, device=actual.device)
                        - torch.as_tensor(minus, dtype=actual.dtype, device=actual.device)) / (2 * delta)
            finite_values.append(float(expected))
            errors.append(float((actual - expected).abs() / expected.abs().clamp_min(1.e-30)))
        record_property(name + "_adjoint", float(actual))
        record_property(name + "_finite_differences", repr(finite_values))
        record_property(name + "_relative_errors", repr(errors))
        assert sum(error < .01 for error in errors) >= 2, (kind, name, actual, errors)


class BlochFoilDesign(FoilDesign):
    def __init__(self, *, phase=True, graph=True, stride=7):
        super().__init__(graph=graph, stride=stride)
        self.phase = phase

    def to_scene(self):
        scene = super().to_scene()
        foil = mw.Structure(name="foil", geometry=mw.Box(position=(0., 0., 0.), size=(0., .06, .06)),
                            material=scene.structures[0].material)
        return scene.clone(boundary=(mw.BoundarySpec.faces(default="bloch", x="pml", num_layers=2,
                      bloch_wavevector=(0., 7., 11.)) if self.phase == "pml" else
                      mw.BoundarySpec.bloch(wavevector=(0., 7., 11.)) if self.phase else mw.BoundarySpec.periodic()),
            ports=(), structures=(foil,), sources=(mw.UniformCurrentSource(name="drive", center=(-.015, 0., -.0025),
                size=(.005, .005, .005), polarization="Ez", source_time=mw.CW(3.e9, amplitude=1e4)),))

    def simulation(self):
        simulation = mw.Simulation.fdtd(self, frequency=3.e9, run_time=mw.TimeConfig(time_steps=48),
            cuda_graph=self.graph, spectral_sampler=mw.SpectralSampler(window="none"))
        simulation.config.adjoint_checkpoint_stride = self.stride
        return simulation


@pytest.mark.parametrize("phase", (False, True, "pml"))
def test_periodic_and_bloch_foil_gradient_matches_finite_difference(phase, record_property):
    model = BlochFoilDesign(phase=phase)
    energy, result = _objective(model, "dissipated_energy")
    energy.backward()
    assert result.stats()["cuda_graph_active"]
    assert bool(result.solver.complex_fields_enabled) == bool(phase)
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    normal, _ = _objective(model, "dissipated_energy")
    torch.testing.assert_close(torch.as_tensor(normal, device=energy.device, dtype=energy.dtype),
                               energy.detach(), rtol=2e-4, atol=1e-30)
    for name in ("log_thickness", "log_conductivity"):
        parameter = getattr(model, name)
        base, actual = parameter.detach().clone(), gradients[name]
        errors = []
        for delta in (.04, .02, .01):
            with torch.no_grad():
                parameter.copy_(base + delta)
            plus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base - delta)
            minus, _ = _objective(model, "dissipated_energy")
            with torch.no_grad():
                parameter.copy_(base)
            expected = (torch.as_tensor(plus, device=actual.device, dtype=actual.dtype)
                        - torch.as_tensor(minus, device=actual.device, dtype=actual.dtype))/(2*delta)
            errors.append(float((actual-expected).abs()/expected.abs().clamp_min(1.e-30)))
        record_property(name+"_relative_errors", repr(errors))
        assert actual.isfinite() and actual.abs() > 0
        assert sum(error < .01 for error in errors) >= 2, (phase, name, actual, errors)


@pytest.mark.parametrize("phase", (False, True, "pml"))
def test_periodic_foil_graph_and_checkpoint_partition_gradient_agree(phase):
    evaluated = []
    for graph, stride in ((False, 1), (True, 7), (True, 13)):
        model = BlochFoilDesign(phase=phase, graph=graph, stride=stride)
        energy, result = _objective(model, "dissipated_energy")
        field = result.monitor("probe")["data"].abs().square().sum()
        objective = energy / energy.detach() + field / field.detach()
        objective.backward()
        evaluated.append(torch.stack((energy.detach(), field.detach(), model.log_thickness.grad, model.log_conductivity.grad)))
        if graph:
            assert result.stats()["cuda_graph_active"]
    for actual in evaluated[1:]:
        torch.testing.assert_close(actual, evaluated[0], rtol=.001, atol=1.e-24)
