"""Every trainable FDFD execution path goes through the adjoint bridge or refuses by name.

CPU tests replace the CuPy solver with a stub: routing and admission run before
any CUDA work. GPU tests prove the gradients against central differences.
"""

import re

import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.simulation as simulation_module
from witwin.maxwell.scene import prepare_scene

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9
FREQS = (1.0e9, 1.15e9)


# --------------------------------------------------------------------------- CPU


class _StubSolver:
    def __init__(self, scene):
        self.scene = scene
        self.E_field = None

    def _release_direct_solver(self):
        pass

    def solve(self, **_kwargs):
        shapes = ((getattr(self.scene, f"N{axis}_e{component}") for axis in "xyz") for component in "xyz")
        self.E_field = tuple(torch.zeros(*shape, dtype=torch.complex64) for shape in shapes)


@pytest.fixture
def stub_solver(monkeypatch):
    builds = []

    def build(simulation):
        builds.append(simulation.frequency)
        return _StubSolver(prepare_scene(simulation.scene, device=simulation.config.device))

    monkeypatch.setattr(simulation_module.Simulation, "_build_fdfd_solver", build)
    return builds


def _cpu_scene(structures=(), regions=(), averaging="kottke_edge"):
    # Geometry gradients need a differentiable averaging; kottke_edge refuses them itself.
    scene = mw.Scene(
                subpixel=mw.SubpixelSpec(averaging=averaging),
                domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                grid=mw.GridSpec.uniform(0.1),
                boundary=mw.BoundarySpec.pml(num_layers=4),
                sources=(mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=None, name="d"),),
            )
    scene = scene.updated(structures=(*scene.structures, *structures))
    scene = scene.updated(material_regions=(*scene.material_regions, *regions))
    return scene


def _box(material, position=(0.2, 0.0, 0.0), **kwargs):
    return mw.Structure(geometry=mw.Box(position=position, size=(0.2, 0.2, 0.2)), material=material, **kwargs)


def _trainable_position(x):
    return torch.stack((torch.tensor(x, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0)))


_PATHS = {
    "run": lambda sim: sim.run(),
    "prepare_run": lambda sim: sim.prepare().run(),
}


@pytest.mark.parametrize("frequencies", [(FREQ,), FREQS], ids=["single", "multi"])
@pytest.mark.parametrize("path", sorted(_PATHS))
def test_trainable_fdfd_paths_return_grad_carrying_fields(stub_solver, path, frequencies):
    eps = torch.tensor(2.0, requires_grad=True)
    simulation = mw.Simulation.fdfd(_cpu_scene([_box(mw.Material(eps_r=eps))]), frequencies=list(frequencies), config=mw.FDFDConfig(device="cpu"))
    result = _PATHS[path](simulation)
    assert stub_solver == list(frequencies)
    for name in ("EX", "EY", "EZ"):
        assert result.fields[name].values.requires_grad, name
        assert result.fields[name].values.grad_fn is not None


def test_detached_fdfd_paths_stay_detached(stub_solver):
    fixed = mw.Simulation.fdfd(_cpu_scene([_box(mw.Material(eps_r=2.0))]), frequencies=list(FREQS), config=mw.FDFDConfig(device="cpu")).run()
    assert not fixed.fields["EX"].values.requires_grad
    eps = torch.tensor(2.0, requires_grad=True)
    with torch.no_grad():
        result = mw.Simulation.fdfd(_cpu_scene([_box(mw.Material(eps_r=eps))]), frequencies=FREQ, config=mw.FDFDConfig(device="cpu")).run()
    assert not result.fields["EX"].values.requires_grad


class _UnusedParameterScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor(2.0))
        self.unused = torch.nn.Parameter(torch.tensor(1.0))

    def to_scene(self):
        return _cpu_scene([_box(mw.Material(eps_r=self.eps))])


class _UsedParameterScene(_UnusedParameterScene):
    def __init__(self):
        super().__init__()
        del self.unused


def test_scene_module_under_no_grad_runs_detached(stub_solver):
    with torch.no_grad():
        result = mw.Simulation.fdfd(_UsedParameterScene(), frequencies=FREQ, config=mw.FDFDConfig(device="cpu")).run()
    assert not result.fields["EX"].values.requires_grad


_REFUSALS = {
    "pec_geometry_next_to_dielectric": (
        lambda: _cpu_scene([
            _box(mw.Material(pec=True), position=_trainable_position(0.2)),
            _box(mw.Material(eps_r=torch.tensor(2.0, requires_grad=True)), position=(-0.2, 0.0, 0.0)),
        ], averaging="polarized"),
        re.escape("- structures[0].geometry.position"),
    ),
    "disabled_structure": (
        lambda: _cpu_scene([_box(mw.Material(eps_r=torch.tensor(2.0, requires_grad=True)), enabled=False)]),
        re.escape("- structures[0].material.eps_r"),
    ),
    "trainable_permeability": (
        lambda: _cpu_scene([_box(mw.Material(mu_r=torch.tensor(2.0, requires_grad=True)))]),
        "fixed permeability.*structures\\[0\\]\\.material\\.mu_r",
    ),
    "dispersive_medium_in_trainable_scene": (
        lambda: _cpu_scene([
            _box(mw.Material(eps_r=torch.tensor(2.0, requires_grad=True))),
            _box(mw.Material(eps_r=2.0, drude_poles=(mw.DrudePole(plasma_frequency=1e9, gamma=1e8),)),
                 position=(-0.2, 0.0, 0.0)),
        ]),
        "FDFD adjoint does not support dispersive media yet",
    ),
    "unused_scene_module_parameter": (
        _UnusedParameterScene,
        re.escape("- SceneModule parameter 'unused'"),
    ),
}


@pytest.mark.parametrize("frequencies", [(FREQ,), FREQS], ids=["single", "multi"])
@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_non_differentiable_trainable_inputs_refuse_by_name_before_solver_build(stub_solver, case, path, frequencies):
    build, match = _REFUSALS[case]
    simulation = mw.Simulation.fdfd(build(), frequencies=list(frequencies), config=mw.FDFDConfig(device="cpu"))
    with pytest.raises(NotImplementedError, match=match):
        _PATHS[path](simulation)
    assert stub_solver == []


def test_trainable_zero_conductivity_is_admitted_as_a_material_input():
    from witwin.maxwell.fdfd.adjoint.bridge import _FDFDGradientBridge

    sigma = torch.tensor(0.0, requires_grad=True)
    simulation = mw.Simulation.fdfd(_cpu_scene([_box(mw.Material(eps_r=2.0, sigma_e=sigma))]), frequencies=FREQ, config=mw.FDFDConfig(device="cpu"))
    bridge = _FDFDGradientBridge(simulation)
    assert bridge.input_names == ("structures[0].material.sigma_e",)
    assert bridge.material_inputs[0] is sigma


def test_material_region_geometry_outside_the_census_is_routed_through_the_bridge(stub_solver):
    # MaterialRegion geometry was missing from the trainable census (added in
    # api070/p2f), so it used to take the detached CuPy path. It reaches the
    # permittivity and is differentiated.
    region = mw.MaterialRegion(name="r", geometry=mw.Box(position=_trainable_position(0.0), size=(0.2, 0.2, 0.2)),
                               density=torch.full((2, 2, 2), 0.5), eps_bounds=(1.0, 4.0))
    simulation = mw.Simulation.fdfd(_cpu_scene(regions=[region], averaging="polarized"), frequencies=FREQ, config=mw.FDFDConfig(device="cpu"))
    assert simulation.has_trainable_parameters
    assert simulation.run().fields["EZ"].values.requires_grad


def test_repeated_runs_of_one_direct_scene_stay_admitted(stub_solver):
    eps = torch.tensor(2.0, requires_grad=True)
    scene = _cpu_scene([_box(mw.Material(eps_r=eps))])
    for _ in range(2):
        assert mw.Simulation.fdfd(scene, frequencies=FREQ, config=mw.FDFDConfig(device="cpu")).prepare().run().fields["EZ"].values.requires_grad
        assert mw.Simulation.fdfd(scene, frequencies=FREQ, config=mw.FDFDConfig(device="cpu")).run().fields["EZ"].values.requires_grad


def test_trainable_copper_foil_is_refused_at_construction():
    # Row 12: a copper-foil thickness/conductivity gradient can no longer come back
    # silently zero, because FDFD refuses CopperFoil itself at Simulation.fdfd(...).
    foil = mw.media.CopperFoil(
        thickness=torch.tensor(35e-6, requires_grad=True),
        conductivity=torch.tensor(5.8e7, requires_grad=True),
        frequency_range=(0.5e9, 2e9),
        mode_count=4,
    )
    scene = _cpu_scene([mw.Structure(geometry=mw.Box(position=(0, 0, 0.2), size=(0.4, 0.4, 0.0)), material=foil)])
    with pytest.raises(NotImplementedError, match="no CopperFoil conductor model"):
        mw.Simulation.fdfd(scene, frequencies=FREQ, config=mw.FDFDConfig(device="cpu"))


# --------------------------------------------------------------------------- GPU


class _DensityScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        scene = mw.Scene(
                    subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
                    domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
                    grid=mw.GridSpec.uniform(0.12),
                    boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                    material_regions=(
                mw.MaterialRegion(
                    name="design",
                    geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                    density=torch.sigmoid(self.logits),
                    eps_bounds=(1.0, 6.0),
                    mu_bounds=(1.0, 1.0),
                ),
                ),
                    sources=(
                mw.PointDipole(position=(0.0, 0.0, -0.06), polarization="Ez", width=0.04,
                               source_time=None),
                ),
                )
        return scene


def _conductive_scene(sigma):
    scene = mw.Scene(
                subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
                domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
                grid=mw.GridSpec.uniform(0.12),
                boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                structures=(
            mw.Structure(
                name="lossy", geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                material=mw.Material(eps_r=4.0, sigma_e=sigma),
            ),
            ),
                sources=(mw.PointDipole(position=(0.0, 0.0, -0.06), polarization="Ez", width=0.04, source_time=None),),
            )
    return scene


# Finite-difference checks use the double-precision direct solve. It is not
# bitwise repeatable run to run (about 1e-16 relative, also on the parent
# commit), so torch.equal checks use a fixed-iteration GMRES instead: they test
# that routing leaves the arithmetic unchanged, which needs repeatability, not
# convergence.
_DIRECT = mw.fdfd.FDFDSolverConfig(solver_type="direct", precision="double")
_REPEATABLE = mw.fdfd.FDFDSolverConfig(solver_type="gmres", precision="double", max_iter=200, restart=50)


def _simulation(scene, frequencies, solver=_DIRECT):
    return mw.Simulation.fdfd(scene, frequencies=list(frequencies), config=mw.FDFDConfig(solver=solver))


def _probe_power(ez):
    return ez[..., ez.shape[-3] // 2, ez.shape[-2] // 2, ez.shape[-1] - 2].abs().square().sum()


def _gradient(model, frequencies, path, solver=_DIRECT):
    model.logits.grad = None
    loss = _probe_power(_PATHS[path](_simulation(model, frequencies, solver)).fields["EZ"].values)
    loss.backward()
    return loss.detach(), model.logits.grad.detach().reshape(())


def _central_difference(frequencies, init=0.0, delta=1.0e-2):
    with torch.no_grad():
        plus = _probe_power(_simulation(_DensityScene(init + delta), frequencies).run().fields["EZ"].values)
        minus = _probe_power(_simulation(_DensityScene(init - delta), frequencies).run().fields["EZ"].values)
    return (plus - minus) / (2.0 * delta)


@requires_cuda
@pytest.mark.parametrize("path", sorted(_PATHS))
def test_multi_frequency_gradient_matches_central_difference_and_frequency_sum(path):
    model = _DensityScene(0.3)
    _, gradient = _gradient(model, FREQS, path)
    per_frequency = sum(_gradient(model, (frequency,), "run")[1] for frequency in FREQS)
    torch.testing.assert_close(gradient, per_frequency, rtol=1e-6, atol=0.0)
    finite_difference = _central_difference(FREQS, init=0.3)
    torch.testing.assert_close(gradient, finite_difference.to(gradient), rtol=1e-2, atol=0.0)


@requires_cuda
def test_single_frequency_prepare_run_gradient_equals_run_and_central_difference():
    model = _DensityScene(0.3)
    loss_run, gradient_run = _gradient(model, (FREQ,), "run", _REPEATABLE)
    loss_prepared, gradient_prepared = _gradient(model, (FREQ,), "prepare_run", _REPEATABLE)
    assert torch.equal(loss_prepared, loss_run)
    assert torch.equal(gradient_prepared, gradient_run)
    _, gradient_prepared = _gradient(model, (FREQ,), "prepare_run")
    torch.testing.assert_close(gradient_prepared, _central_difference((FREQ,), init=0.3).to(gradient_prepared),
                               rtol=1e-2, atol=0.0)


@requires_cuda
@pytest.mark.parametrize("frequencies", [(FREQ,), FREQS], ids=["single", "multi"])
@pytest.mark.parametrize("path", sorted(_PATHS))
def test_bridge_forward_fields_equal_detached_forward(path, frequencies):
    trained = _PATHS[path](_simulation(_DensityScene(0.3), frequencies, _REPEATABLE))
    with torch.no_grad():
        detached = _PATHS[path](_simulation(_DensityScene(0.3), frequencies, _REPEATABLE))
    for name in ("EX", "EY", "EZ"):
        assert trained.fields[name].values.requires_grad
        assert not detached.fields[name].values.requires_grad
        assert torch.equal(trained.fields[name].values.detach(), detached.fields[name].values), name


@requires_cuda
def test_trainable_zero_conductivity_gradient_matches_central_difference():
    sigma = torch.tensor(0.0, device="cuda", requires_grad=True)
    loss = _probe_power(_simulation(_conductive_scene(sigma), (FREQ,)).run().fields["EZ"].values)
    loss.backward()
    # sigma_e >= 0, so use the second-order one-sided difference at the boundary.
    delta = 1.0e-3
    with torch.no_grad():
        f0, f1, f2 = (_probe_power(_simulation(_conductive_scene(value), (FREQ,)).run().fields["EZ"].values)
                      for value in (0.0, delta, 2.0 * delta))
    finite_difference = (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * delta)
    assert bool(finite_difference.abs() > 0)
    torch.testing.assert_close(sigma.grad, finite_difference.to(sigma.grad), rtol=1e-2, atol=0.0)


def _region_scene(x):
    position = torch.stack((x, x.new_tensor(0.0), x.new_tensor(0.06)))
    density = torch.linspace(0.2, 0.9, 27, device="cuda").reshape(3, 3, 3)
    scene = mw.Scene(
                subpixel=mw.SubpixelSpec(samples=5, averaging="polarized"),
                domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
                grid=mw.GridSpec.uniform(0.06),
                boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                material_regions=(
            mw.MaterialRegion(name="design", geometry=mw.Box(position=position, size=(0.18,) * 3),
                              density=density, eps_bounds=(1.0, 6.0)),
            ),
                sources=(mw.PointDipole(position=(0.0, 0.0, -0.06), polarization="Ez", width=0.04, source_time=None),),
            )
    return scene


@requires_cuda
def test_material_region_geometry_gradient_matches_central_difference():
    x = torch.tensor(0.0137, device="cuda", requires_grad=True)
    loss = _probe_power(_simulation(_region_scene(x), (FREQ,)).run().fields["EZ"].values)
    loss.backward()
    delta = 1.0e-4
    with torch.no_grad():
        plus, minus = (_probe_power(_simulation(_region_scene(x.detach() + step), (FREQ,)).run().fields["EZ"].values)
                       for step in (delta, -delta))
    finite_difference = (plus - minus) / (2.0 * delta)
    torch.testing.assert_close(x.grad, finite_difference.to(x.grad), rtol=2e-2, atol=0.0)
