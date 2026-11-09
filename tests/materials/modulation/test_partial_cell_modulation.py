"""Constitutive derivatives and complex sidebands of partial-cell modulation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import quad

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.materials import compile_material_model
from witwin.maxwell.fdtd.runtime.materials import average_node_to_component, build_modulation


C = 299792458.0


def _compiled_runtime(scene, *, dtype=torch.float64):
    model = prepare_scene(scene, material_dtype=dtype).compile_materials()
    runtime = SimpleNamespace(dt=1e-12, device=scene.device, Ex=torch.empty(0, dtype=dtype))
    build_modulation(runtime, model)
    return model, runtime


@pytest.mark.parametrize("averaging", ("arithmetic", "polarized", "polarized_tensor", "kottke_edge", "kottke_smooth"))
@pytest.mark.parametrize("passive_conductivity", (0., .2))
@pytest.mark.parametrize("normal", range(3))
@pytest.mark.parametrize("fraction", (0.25, 0.5, 0.75))
def test_partial_cell_constitutive_derivative(averaging, passive_conductivity, normal, fraction):
    eps, depth, phase = 4.0, 1e-4, np.pi / 3
    samples = 3 if averaging == "kottke_smooth" else 4
    for axis, component in enumerate(("Ex", "Ey", "Ez")):
        # The Yee component is centered at .125 on its staggered axis and .1
        # on the others. Its dual window has width .05 in every direction.
        edge_coordinate = .125 if axis == normal else .1
        interface = edge_coordinate + (fraction - .5) * .05
        size, position = [2.0] * 3, [0.0] * 3
        size[normal], position[normal] = interface + 1, (interface - 1) / 2
        scene = mw.Scene(
            domain=mw.Domain(bounds=((0, .2),) * 3), grid=mw.GridSpec.uniform(.05), device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=tuple(samples if i == normal else 1 for i in range(3)), averaging=averaging),
            structures=[mw.Structure(geometry=mw.Box(position=torch.tensor(position, dtype=torch.float64),
                                                    size=torch.tensor(size, dtype=torch.float64)), material=mw.Material(
                eps_r=eps, modulation=mw.ModulationSpec(frequency=1e8, amplitude=depth, phase=phase))),
                # A distant passive conductor selects the general Kottke call
                # site without adding conductivity to the modulated material.
                mw.Structure(geometry=mw.Box(position=(10, 0, 0), size=(1, 1, 1)),
                             material=mw.Material(sigma_e=passive_conductivity))],
        )
        model, runtime = _compiled_runtime(scene)
        index = (2, 2, 2)
        static = model["edge_components"]["eps"][component][index].item()
        mc = getattr(runtime, "mod_cos_" + component)[index].item()
        ms = getattr(runtime, "mod_sin_" + component)[index].item()
        harmonic = axis == normal and averaging != "arithmetic"
        # Three smooth samples retain fractional sub-sample coverage, unlike
        # the binary four-sample windows aligned with these fractions.
        fill = np.clip(samples * fraction - np.arange(samples), 0, 1)
        constituents = 1 + fill * (eps - 1)
        expected_static = 1 / np.mean(1 / constituents) if harmonic else np.mean(constituents)
        assert static == pytest.approx(expected_static, abs=2e-10)
        for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            local = eps * (1 + depth * np.cos(angle + phase))
            instantaneous = 1 + fill * (local - 1)
            exact = 1 / np.mean(1 / instantaneous) if harmonic else np.mean(instantaneous)
            derivative = (expected_static**2 * np.mean(fill * eps / constituents**2)
                          if harmonic else fraction * eps)
            linear = static * (1 + mc * np.cos(angle) - ms * np.sin(angle))
            assert linear == pytest.approx(expected_static + derivative * depth * np.cos(angle + phase), abs=2e-10)
            assert abs(linear - exact) < 2 * depth**2


def test_modulation_interior_identity_and_parameter_gradients():
    depth = torch.tensor([[[.13]]], dtype=torch.float64, requires_grad=True)
    phase = torch.tensor([[[.7]]], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, .1),) * 3), grid=mw.GridSpec.uniform(.05), device="cpu",
        structures=[mw.Structure(geometry=mw.Box(size=(2, 2, 2)), material=mw.Material(
            eps_r=eps, modulation=mw.ModulationSpec(frequency=1e8, amplitude=depth, phase=phase)))],
    )
    model, runtime = _compiled_runtime(scene)
    for component in ("Ex", "Ey", "Ez"):
        for name in ("cos", "sin"):
            assert torch.equal(getattr(runtime, f"mod_{name}_{component}"),
                               average_node_to_component(None, model[f"modulation_{name}"], component))
    value = (model["edge_components"]["eps"]["Ex"] * runtime.mod_cos_Ex).mean()
    actual = torch.autograd.grad(value, (eps, depth, phase))
    expected = (depth.item() * np.cos(phase.item()), eps.item() * np.cos(phase.item()),
                -eps.item() * depth.item() * np.sin(phase.item()))
    for gradient, target in zip(actual, expected):
        assert gradient.item() == pytest.approx(target, rel=1e-6)


@pytest.mark.parametrize("eps_value", (1.0, 4.0))
def test_partial_cell_keeps_depth_phase_and_permittivity_gradients(eps_value):
    eps = torch.tensor(eps_value, requires_grad=True)
    depth = torch.tensor([[[.1]]], requires_grad=True)
    phase = torch.tensor([[[.7]]], requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, .1),) * 3), grid=mw.GridSpec.uniform(.05), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 4), averaging="polarized"),
        structures=[mw.Structure(geometry=mw.Box(position=(0, 0, -.5), size=(2, 2, 1)), material=mw.Material(
            eps_r=eps, modulation=mw.ModulationSpec(frequency=1e8, amplitude=depth, phase=phase)))],
    )
    model, runtime = _compiled_runtime(scene)
    value = (model["edge_components"]["eps"]["Ex"] * runtime.mod_cos_Ex)[1, 2, 2]
    actual = torch.autograd.grad(value, (eps, depth, phase))
    expected = (.5 * .1 * np.cos(.7), .5 * eps_value * np.cos(.7), -.5 * eps_value * .1 * np.sin(.7))
    for gradient, target in zip(actual, expected):
        assert gradient.item() == pytest.approx(target, rel=1e-6)


@pytest.mark.parametrize("samples", (2, 4))
def test_node_variation_includes_the_perturbed_background(samples):
    depth, phase = .1, .5

    def compile(scale, modulation):
        material = mw.PerturbationMedium(
            mw.Material(eps_r=4 * scale, modulation=modulation),
            perturbation=torch.full((1, 1, 1), .5 * scale, dtype=torch.float64),
        )
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.1, .1),) * 3), grid=mw.GridSpec.uniform(.05), device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=samples, averaging="polarized"),
            structures=[mw.Structure(geometry=mw.Box(position=(-.5, 0, 0), size=(1, 2, 2)), material=material)],
        )
        prepared = prepare_scene(scene, material_dtype=torch.float64)
        return compile_material_model(prepared, subpixel=scene.subpixel, include_edges=False)

    model = compile(1., mw.ModulationSpec(frequency=1e8, amplitude=depth, phase=phase))
    h = 1e-3
    plus, minus = compile(1 + h, None), compile(1 - h, None)
    for axis in ("x", "y", "z"):
        derivative = (plus["eps_components"][axis] - minus["eps_components"][axis]) / (2 * h)
        torch.testing.assert_close(model["modulation_absolute"]["cos"][axis],
                                   derivative * depth * np.cos(phase), rtol=2e-5, atol=1e-8)


def test_constitutive_variations_are_computed_in_inference_mode():
    def compile():
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.1, .1),) * 3), grid=mw.GridSpec.uniform(.05), device="cpu",
            subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"),
            structures=[mw.Structure(geometry=mw.Box(position=(-.5, 0, 0), size=(1, 2, 2)),
                                    material=mw.Material(eps_r=torch.tensor(4., dtype=torch.float64),
                                    modulation=mw.ModulationSpec(frequency=1e8,
                                        amplitude=torch.full((1, 1, 1), .1, dtype=torch.float64),
                                        phase=torch.full((1, 1, 1), .5, dtype=torch.float64))))],
        )
        return _compiled_runtime(scene)[1]

    regular = compile()
    with torch.inference_mode():
        inference = compile()
        assert torch.is_inference_mode_enabled()
    for component in ("Ex", "Ey", "Ez"):
        for name in ("cos", "sin"):
            assert torch.equal(getattr(regular, f"mod_{name}_{component}"),
                               getattr(inference, f"mod_{name}_{component}"))


def born_transmission(eps, depth, phase, pump, thickness):
    """Outgoing static-slab Green integral, with exp(-i omega t) phasors."""
    n, k0 = np.sqrt(eps), 2 * np.pi * 2e9 / C
    d = thickness
    t0 = 1 / (np.cos(n * k0 * d) - .5j * (n + 1 / n) * np.sin(n * k0 * d))
    carrier = lambda z: t0 * (np.cos(n * k0 * (d / 2 - z)) - 1j / n * np.sin(n * k0 * (d / 2 - z)))
    out = []
    for sign in (-1, 1):
        k = 2 * np.pi * (2e9 + sign * pump) / C
        ul = lambda z: np.cos(n * k * (z + d / 2)) - 1j / n * np.sin(n * k * (z + d / 2))
        wronskian = ul(d / 2) * 1j * k + n * k * np.sin(n * k * d) + 1j * k * np.cos(n * k * d)
        integral = (quad(lambda z: (ul(z) * carrier(z)).real, -d / 2, d / 2)[0]
                    + 1j * quad(lambda z: (ul(z) * carrier(z)).imag, -d / 2, d / 2)[0])
        out.append(-k**2 * eps * depth / 2 * np.exp(-1j * sign * phase) * integral / wronskian)
    return np.asarray(out)


def weak_slab_scene(dx, *, depth=.01, phase=np.pi / 3, thickness=.005, vacuum=False, eps=4.0, center=0.0, device="cuda", pml_layers=8):
    return mw.Scene(
        domain=mw.Domain(bounds=((-2 * dx, 2 * dx), (-2 * dx, 2 * dx), (-.4, .4))),
        grid=mw.GridSpec.uniform(dx), device=device,
        boundary=mw.BoundarySpec.pml(num_layers=pml_layers).with_faces(
            x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"),
        subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"),
        structures=[] if vacuum else [mw.Structure(geometry=mw.Box(position=(0, 0, center), size=(2, 2, thickness)), material=mw.Material(
            eps_r=eps, modulation=mw.ModulationSpec(frequency=.2e9, amplitude=depth, phase=phase)))],
        sources=[mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0), source_time=mw.CW(frequency=2e9))],
        monitors=[mw.FieldTimeMonitor(name=name, components=("Ex",), position=(0, 0, z))
                  for name, z in (("left", -.2), ("right", .2))],
    )


def run_weak_slab(dx, *, vacuum=False, duration=60e-9, **kwargs):
    simulation = mw.Simulation.fdtd(
        weak_slab_scene(dx, vacuum=vacuum, **kwargs), absorber="cpml", frequency=2e9,
        run_time=mw.TimeConfig(time_steps=1), full_field_dft=False,
    )
    prepared = simulation.prepare()
    simulation.config.run_time = mw.TimeConfig(time_steps=int(np.ceil(duration / prepared.solver.dt)))
    result = prepared.run()
    traces = {name: result.monitor(name) for name in ("left", "right")}
    return traces, float(result.solver.dt)


def fit_harmonics(trace, dt, *, start=.5, order=3):
    field = np.asarray(trace["field"].detach().cpu()).reshape(-1)
    t = np.arange(len(field)) * dt
    selected = t >= start * t[-1]
    frequencies = 2e9 + np.arange(-order, order + 1) * .2e9
    phase = 2 * np.pi * t[selected, None] * frequencies
    basis = np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
    coefficients = np.linalg.lstsq(basis, field[selected], rcond=None)[0]
    return coefficients[:len(frequencies)] + 1j * coefficients[len(frequencies):]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weak_partial_cell_slab_born_amplitude_and_phase():
    # Both slab faces cut the outer tangential control volumes in half.
    dx = .0025
    traces, dt = run_weak_slab(dx)
    vacuum, vacuum_dt = run_weak_slab(dx, vacuum=True)
    incident = fit_harmonics(vacuum["right"], vacuum_dt)[3] * np.exp(-2j * np.pi * 2e9 / C * (.2 + .0025))
    actual = fit_harmonics(traces["right"], dt)[[2, 4]] / incident
    actual *= np.exp(-2j * np.pi * np.asarray([1.8e9, 2.2e9]) / C * (.2 - .0025))
    expected = born_transmission(4, .01, np.pi / 3, .2e9, .005)
    np.testing.assert_allclose(actual, expected, rtol=.06, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_cell_time_step_bounds_the_actual_modulated_mass():
    scene = weak_slab_scene(.01, eps=.4, depth=.2, center=.005, pml_layers=40)
    solver = mw.Simulation.fdtd(scene, frequency=2e9, run_time=mw.TimeConfig(time_steps=1),
                                full_field_dft=False).prepare().solver
    minimum = min(float((getattr(solver, f"eps_{component}") / solver.eps0 *
                         (1 - torch.hypot(getattr(solver, f"mod_cos_{component}"),
                                          getattr(solver, f"mod_sin_{component}")))).min())
                  for component in ("Ex", "Ey", "Ez"))
    assert solver._time_step_metadata["epsilon_lower_bound"] == pytest.approx(minimum, rel=1e-6)
