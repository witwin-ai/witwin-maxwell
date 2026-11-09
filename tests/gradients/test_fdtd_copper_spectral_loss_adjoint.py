"""Independent normal-forward scans of both foil traces and direct loss factors."""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class SpectralLossDesign(mw.SceneModule):
    def __init__(self, *, pole="none", window="none", normalized=False):
        super().__init__()
        for name, value in (("thickness", 1e-6), ("copper_sigma", 1e6), ("bulk_sigma", .15), ("epsilon", 2.)):
            self.register_parameter("log_"+name, torch.nn.Parameter(
                torch.tensor(math.log(value), device="cuda", dtype=torch.float64)))
        self.pole, self.window, self.normalized = pole, window, normalized

    def to_scene(self):
        poles = {
            "none": {},
            "debye": dict(debye_poles=(mw.DebyePole(delta_eps=1., tau=3e-11),)),
            "drude": dict(drude_poles=(mw.DrudePole(plasma_frequency=2e9, gamma=.4e9),)),
            "lorentz": dict(lorentz_poles=(mw.LorentzPole(delta_eps=1., resonance_frequency=4e9, gamma=.8e9),)),
        }
        return mw.Scene(domain=mw.Domain(bounds=((- .03,.03),)*3), grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.pec(), device="cuda", structures=(
                mw.Structure(mw.Box(size=(0., .02, .02)), mw.CopperFoil(
                    thickness=self.log_thickness.exp(), conductivity=self.log_copper_sigma.exp(),
                    frequency_range=(2e9,4e9), mode_count=16), name="foil"),
                mw.Structure(mw.Box(position=(.015,0.,0.), size=(.03,.06,.06)),
                    mw.Material(eps_r=self.log_epsilon.exp(), sigma_e=self.log_bulk_sigma.exp(),
                                **poles[self.pole]), name="substrate")),
            sources=(mw.UniformCurrentSource(name="drive", center=(-.015,0.,-.0025), size=(.005,)*3, polarization="Ez",
                source_time=mw.CW(3e9, phase=.37, amplitude=1e-5)),),
            monitors=(mw.PowerLossMonitor("bulk", position=(0.,0.,0.), size=(.04,)*3,
                channels=("conduction", "electric_dispersion")),))

    def simulation(self, *, graph=False, stride=7):
        simulation = mw.Simulation.fdtd(self, frequencies=(2.5e9,3e9), cuda_graph=graph,
            full_field_dft=True, run_time=mw.TimeConfig(time_steps=72 if self.window=="none" else 512,
                dt=None if self.window=="none" else 5e-12, transient_cycles=0, steady_cycles=1),
            spectral_sampler=mw.SpectralSampler(window=self.window, normalize_source=self.normalized))
        simulation.config.adjoint_checkpoint_stride = stride
        return simulation


def objective(model, kind, **kwargs):
    result = model.simulation(**kwargs).run()
    if kind == "surface":
        value = result.conductor_losses["foil"]["surface_power"].sum()
    elif kind == "combined":
        value = result.power_loss("bulk").total.sum()+result.conductor_losses["foil"]["surface_power"].sum()
    else:
        value = result.power_loss("bulk").channel_power[kind].sum()
    return value, result


def test_multifrequency_pulse_absolute_copper_loss_gradient(record_property):
    class PulsedLossDesign(SpectralLossDesign):
        def to_scene(self):
            scene = super().to_scene()
            return scene.clone(sources=(mw.UniformCurrentSource(name="drive",
                center=(-.015, 0., -.0025), size=(.005,)*3, polarization="Ez",
                source_time=mw.GaussianPulse(3e9, fwidth=1.5e9, amplitude=1e-5)),))

        def simulation(self):
            simulation = mw.Simulation.fdtd(self, frequencies=(2.5e9, 3e9),
                cuda_graph=True, full_field_dft=True,
                run_time=mw.TimeConfig(time_steps=256), shutoff=0.,
                spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))
            simulation.config.adjoint_checkpoint_stride = 7
            return simulation

    model = PulsedLossDesign()
    model.requires_grad_(False)
    parameter = model.log_copper_sigma
    parameter.requires_grad_(True)
    value, result = objective(model, "surface")
    assert result.stats()["reference_frequency"] == "per-frequency"
    assert result.stats()["normalize_source"] is False
    assert result.stats()["cuda_graph_active"]
    assert bool(torch.all(result.conductor_losses["foil"]["surface_power"] > 0))
    value.backward()
    adjoint = parameter.grad.detach().clone()
    assert adjoint.isfinite() and adjoint.abs() > 0

    # Detached ordinary FDTD checks the reconstructed adjoint forward and FD.
    model.requires_grad_(False)
    normal, _ = objective(model, "surface")
    torch.testing.assert_close(normal.to(value.device), value.detach(), rtol=3e-4, atol=0.)
    record_property("absolute_copper_loss_W", float(normal))
    record_property("d_loss_d_log_conductivity_adjoint", float(adjoint))
    base = parameter.detach().clone()
    errors = []
    for step in (.02, .01):
        with torch.no_grad():
            parameter.copy_(base + step)
            plus = objective(model, "surface")[0]
            parameter.copy_(base - step)
            minus = objective(model, "surface")[0]
            parameter.copy_(base)
        finite = (plus - minus) / (2 * step)
        error = float((adjoint - finite).abs() / finite.abs().clamp_min(1e-300))
        record_property(f"central_fd_step_{step}", dict(plus_W=float(plus), minus_W=float(minus),
            derivative=float(finite), relative_error=error))
        errors.append(error)
    assert all(error < .01 for error in errors), (adjoint, errors)


@pytest.mark.parametrize("kind,pole", [("conduction","none"), ("surface","none"),
    ("combined","debye"), ("electric_dispersion","drude"), ("electric_dispersion","lorentz")])
def test_two_sided_spectral_loss_gradients_match_independent_forward_scan(kind, pole, record_property):
    model = SpectralLossDesign(pole=pole)
    value, result = objective(model, kind)
    assert value.requires_grad
    traces = result.conductor_losses["foil"]["bulk_electric_traces"]
    assert traces and all(trace["electric_fields"].requires_grad for trace in traces)
    assert any(float((trace["electric_fields"][:,0]-trace["electric_fields"][:,1]).detach().abs().max())>0 for trace in traces)
    value.backward()
    gradients = {name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    # The reference uses detached ordinary FDTD rather than the custom forward.
    model.requires_grad_(False)
    normal, _ = objective(model, kind)
    torch.testing.assert_close(normal.to(value.device), value.detach(), rtol=3e-4, atol=1e-25)
    for name, parameter in model.named_parameters():
        base = parameter.detach().clone()
        errors, finite_values = [], []
        for step in (.08,.04,.02,.01):
            with torch.no_grad():
                parameter.copy_(base+step)
                plus = objective(model, kind)[0]
                parameter.copy_(base-step)
                minus = objective(model, kind)[0]
                parameter.copy_(base)
            finite = (plus-minus)/(2*step)
            finite_values.append(float(finite))
            errors.append(float((gradients[name]-finite).abs()/finite.abs().clamp_min(1e-30)))
        record_property(name+"_fd",repr(finite_values))
        record_property(name+"_relative_errors",repr(errors))
        assert gradients[name].isfinite() and abs(float(gradients[name]))>1e-30
        assert sum(error<.01 for error in errors)>=2,(kind,pole,name,gradients[name],finite_values,errors)


@pytest.mark.parametrize("window,normalized", [("none",False), ("hanning",False), ("ramp",True)])
def test_side_spectral_loss_graph_checkpoint_clock_and_normalization_parity(window, normalized):
    model = SpectralLossDesign(pole="debye", window=window, normalized=normalized)
    reference = None
    for graph,stride in ((False,1),(True,7),(True,13)):
        model.zero_grad(set_to_none=True)
        value,result = objective(model,"combined",graph=graph,stride=stride)
        value.backward()
        actual = (value.detach(),torch.stack([parameter.grad for parameter in model.parameters()]))
        if reference is None:
            reference = actual
        else:
            assert result.stats()["cuda_graph_active"]
            for a,b in zip(actual,reference):
                torch.testing.assert_close(a,b,rtol=.001,atol=1e-25)
    model.requires_grad_(False)
    normal,_ = objective(model,"combined")
    torch.testing.assert_close(normal.to(reference[0].device),reference[0],rtol=3e-4,atol=1e-25)
    if normalized:
        parameter = model.log_bulk_sigma
        base = parameter.detach().clone()
        for step in (.04,.02):
            with torch.no_grad():
                parameter.copy_(base+step)
                plus = objective(model,"combined")[0]
                parameter.copy_(base-step)
                minus = objective(model,"combined")[0]
                parameter.copy_(base)
            finite = (plus-minus)/(2*step)
            torch.testing.assert_close(finite,reference[1][2],rtol=.01,atol=1e-25)


def test_bloch_complex_partner_spectral_surface_work_gradient(record_property):
    from tests.gradients.test_fdtd_copper_foil_adjoint import BlochFoilDesign
    model = BlochFoilDesign(phase=True, graph=True, stride=7)
    def evaluate():
        result = model.simulation().run()
        return result.conductor_losses["foil"]["surface_power"].sum(),result
    value,result = evaluate()
    assert result.solver.complex_fields_enabled and result.stats()["cuda_graph_active"]
    value.backward()
    gradients = {name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.requires_grad_(False)
    normal,_ = evaluate()
    torch.testing.assert_close(normal.to(value.device),value.detach(),rtol=3e-4,atol=1e-25)
    for name,parameter in model.named_parameters():
        base = parameter.detach().clone()
        errors=[]
        for step in (.08,.04,.02,.01):
            with torch.no_grad():
                parameter.copy_(base+step)
                plus = evaluate()[0]
                parameter.copy_(base-step)
                minus = evaluate()[0]
                parameter.copy_(base)
            finite=(plus-minus)/(2*step)
            errors.append(float((gradients[name]-finite).abs()/finite.abs().clamp_min(1e-30)))
        record_property(name+"_relative_errors",repr(errors))
        assert sum(error<.01 for error in errors)>=2,(name,errors)


def test_native_rf_batch_retains_side_loss_graph_after_reuse_and_close():
    class RFDesign(SpectralLossDesign):
        def to_scene(self):
            return super().to_scene().clone(sources=(), ports=(mw.LumpedPort("feed",
                positive=(-.01,0.,.005), negative=(-.01,0.,-.005), voltage_path=mw.AxisPath("z"),
                current_surface=mw.Box(position=(-.01,0.,-.0025),size=(.015,.015,0.))),))

        def simulation(self, *, graph=False, stride=7):
            simulation = mw.Simulation.fdtd(self, frequencies=(2.5e9,3e9), cuda_graph=graph,
                full_field_dft=True, run_time=mw.TimeConfig(time_steps=72,transient_cycles=0,steady_cycles=1),
                spectral_sampler=mw.SpectralSampler(window="none",normalize_source=False),
                excitations=mw.PortExcitation("feed",source_time=mw.CW(3e9,phase=.37)))
            simulation.config.adjoint_checkpoint_stride = stride
            return simulation

    models = [RFDesign(pole="debye"), RFDesign(pole="debye")]
    with torch.no_grad():
        models[1].log_bulk_sigma.add_(.2)
    simulations = [model.simulation() for model in models]
    parameters = tuple(parameter for model in models for parameter in model.parameters())
    batch = mw.Simulation.batch(simulations, outputs=mw.BatchOutput(fields=("Ex","Ey","Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",),batch_size=2,graph="required",graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run().results()
        second = prepared.run().results()

    def evaluate(results):
        return sum(result.power_loss("bulk").total.sum()+result.conductor_losses["foil"]["surface_power"].sum()
                   for result in results)

    expected_value = evaluate([simulation.run() for simulation in simulations])
    expected = torch.autograd.grad(expected_value, parameters)
    for results in (first,second):
        value = evaluate(results)
        torch.testing.assert_close(value,expected_value,rtol=3e-4,atol=1e-25)
        actual = torch.autograd.grad(value,parameters)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=.002,atol=1e-25)
        assert all(result.stats()["batch_adjoint"]=="native_checkpoint_per_lane" for result in results)
