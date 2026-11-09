"""Fixed-geometry pole-value pullbacks on centre-owned Yee constituents."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint.core import accumulate_pole_coefficient_gradients
from witwin.maxwell.fdtd.checkpoint import dispersive_state_name


@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz"])
@pytest.mark.parametrize("component", ["Ex", "Hy"])
def test_local_pole_coefficient_vjp_matches_autograd(kind, component):
    generator = torch.Generator().manual_seed(41)
    def random():
        return torch.randn(7, dtype=torch.float64, generator=generator)
    e, p, j = random(), random(), random()
    drive = random().requires_grad_()
    decay = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
    restoring = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    dt = .12
    if kind == "debye":
        pnext = decay*p + drive*e
        jnext = (pnext-p)/dt
    else:
        jnext = decay*j + drive*e - (restoring*p if kind == "lorentz" else 0.)
        pnext = p + dt*jnext
    le, lp, lj = random(), random(), random()
    correction = .4
    objective = (-correction*jnext*le + jnext*lj).sum()
    if kind != "drude":
        objective += (pnext*lp).sum()
    coefficients = (drive, decay, restoring) if kind == "lorentz" else (drive, decay)
    expected = torch.autograd.grad(objective, coefficients)
    names = tuple(("drive", "decay", "restoring")[:len(coefficients)])
    gradients = {name: torch.zeros_like(value) for name, value in zip(names, coefficients)}
    solver = SimpleNamespace(dt=dt, _pole_coefficient_gradients={(component, kind, 0): gradients})
    jname = dispersive_state_name(component, kind, 0, "current")
    pname = dispersive_state_name(component, kind, 0, "polarization")
    forward = {component: e, jname: j, pname: p}
    adjoint = {jname: lj, pname: lp}
    accumulate_pole_coefficient_gradients(solver, component, kind, 0, forward, adjoint, lj-correction*le)
    for name, value in zip(names, expected):
        torch.testing.assert_close(gradients[name], value)


_PARAMETERS = [("debye", "delta_eps"), ("debye", "tau"), ("drude", "plasma_frequency"),
               ("drude", "gamma"), ("lorentz", "delta_eps"),
               ("lorentz", "resonance_frequency"), ("lorentz", "gamma")]


def test_split_trace_pole_parameter_gradients_fail_closed_for_scene_modules():
    from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge, _require_fixed_trace_poles
    class Design(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))

        def to_scene(self):
            return mw.Scene(domain=mw.Domain(bounds=((-0.2, .2),)*3), grid=mw.GridSpec.uniform(.1),
                device="cpu", structures=[mw.Structure(geometry=mw.Sphere(radius=.1),
                    material=mw.Material.drude(eps_inf=2., plasma_frequency=2e9, gamma=self.scale*1e8))])
    model = Design()
    bridge = SimpleNamespace(_material_graph_scene=model.to_scene)
    for histories in ((), ({"kind": "drude"},)):
        solver = SimpleNamespace(_copper_foils=({"material_ade": histories},))
        with torch.no_grad(), pytest.raises(ValueError, match="split-trace conductor histories"):
            _FDTDGradientBridge._validate_supported_configuration(bridge, solver)
    static = model.to_scene()
    static.structures = [mw.Structure(geometry=static.structures[0].geometry,
        material=mw.Material.drude(eps_inf=2., plasma_frequency=2e9, gamma=1e8))]
    _require_fixed_trace_poles(static)


@pytest.mark.parametrize("kind,parameter", _PARAMETERS)
def test_pole_material_pullback_matches_fd_on_fixed_sphere(kind, parameter):
    from witwin.maxwell.adjoint_inputs import material_dependent_inputs, scene_trainable_material_tensors
    from witwin.maxwell.fdtd.material_pullback import pullback_material_input_gradients
    from witwin.maxwell.fdtd.runtime.materials import pole_update_coefficients
    from witwin.maxwell.constants import EPSILON_0
    from witwin.maxwell.scene import prepare_scene
    values = {"debye": dict(delta_eps=2., tau=1e-10), "drude": dict(plasma_frequency=2e9, gamma=2e8),
              "lorentz": dict(delta_eps=2., resonance_frequency=1.8e9, gamma=2e8)}[kind]
    pole_type = {"debye": mw.DebyePole, "drude": mw.DrudePole, "lorentz": mw.LorentzPole}[kind]
    dt = 3e-12
    def build(scale):
        parameters = dict(values)
        parameters[parameter] *= scale
        return mw.Scene(domain=mw.Domain(bounds=((-0.2, .2),)*3), grid=mw.GridSpec.uniform(.08),
            boundary=mw.BoundarySpec.pec(), device="cpu",
            structures=[mw.Structure(geometry=mw.Sphere(radius=.143),
                material=mw.Material(eps_r=2., **{kind+"_poles": (pole_type(**parameters),)}))])
    scale = torch.tensor(1., requires_grad=True)
    scene = build(scale)
    inputs = material_dependent_inputs(scene, scene_trainable_material_tensors(scene))
    assert len(inputs) == 1
    prepared = prepare_scene(scene)
    model = prepared.compile_materials()
    entry = model[kind+"_poles"][0]
    coefficients = pole_update_coefficients(entry["pole"], kind, EPSILON_0, dt)
    gradients = {}
    for c, weight in entry["edge_weights"].items():
        gradients[(c, kind, 0)] = {key: (torch.ones_like(weight)/abs(float(torch.as_tensor(value).detach())) if key == "drive" else
            torch.ones(())/abs(float(torch.as_tensor(value).detach()))) for key, value in coefficients.items()}
    zeros = [torch.zeros_like(model["edge_components"]["eps"][c]) for c in ("Ex", "Ey", "Ez")]
    output, = pullback_material_input_gradients(scene, inputs=inputs, grad_eps_ex=zeros[0],
        grad_eps_ey=zeros[1], grad_eps_ez=zeros[2], eps0=EPSILON_0, dt=dt,
        grad_pole_coefficients=gradients)
    derivative, = torch.autograd.grad(inputs[0], scale, grad_outputs=output)
    def objective(x):
        local = prepare_scene(build(x)).compile_materials()[kind+"_poles"][0]
        coeff = pole_update_coefficients(local["pole"], kind, EPSILON_0, dt)
        return sum((coeff[key] * (local["edge_weights"][c] if key == "drive" else 1.) * grad).sum()
                   for (c, _, _), mapping in gradients.items() for key, grad in mapping.items())
    fd = (objective(torch.tensor(1.002))-objective(torch.tensor(.998)))/.004
    torch.testing.assert_close(derivative, fd, rtol=.005, atol=.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the FDTD adjoint")
@pytest.mark.parametrize("kind,parameter", _PARAMETERS)
@pytest.mark.parametrize("magnetic", [False, True])
def test_binary_interface_pole_parameter_gradient_matches_fd(kind, parameter, magnetic):
    def loss(scale):
        values = {"debye": dict(delta_eps=2., tau=1e-10),
                  "drude": dict(plasma_frequency=2e9, gamma=2e8),
                  "lorentz": dict(delta_eps=2., resonance_frequency=1.8e9, gamma=2e8)}[kind]
        values[parameter] = values[parameter] * scale
        pole = {"debye": mw.DebyePole, "drude": mw.DrudePole, "lorentz": mw.LorentzPole}[kind](**values)
        material = mw.Material(eps_r=2., **{("mu_" if magnetic else "")+kind+"_poles": (pole,)})
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, .3),)*3), grid=mw.GridSpec.uniform(.06),
            boundary=mw.BoundarySpec.pec(), device="cuda",
            structures=[mw.Structure(geometry=mw.Sphere(radius=.143), material=material)],
            sources=[mw.PointDipole(position=(-.12, 0., 0.), polarization="Ez", width=.06,
                source_time=mw.CW(frequency=1e9))])
        result = mw.Simulation.fdtd(scene, frequencies=(1e9,),
            run_time=mw.TimeConfig(time_steps=240, dt=3e-12), full_field_dft=True, cuda_graph=False,
            spectral_sampler=mw.SpectralSampler(normalize_source=False, window="none")).run()
        field = result.field("Ez")
        if isinstance(field, dict):
            field = field["data"]
        return field.abs().square().sum()
    scale = torch.tensor(1., device="cuda", requires_grad=True)
    derivative, = torch.autograd.grad(loss(scale), scale)
    with torch.no_grad():
        fd = (loss(scale+.002)-loss(scale-.002))/.004
    assert torch.isfinite(derivative) and abs(float(fd)) > 0
    torch.testing.assert_close(derivative, fd, rtol=.03, atol=abs(float(fd))*.003)
