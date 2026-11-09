"""Native batch tapes preserve gradients across slot reuse and closure."""
import pytest
import torch

import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def density_batch(density, *, capacity=2, slots=1, devices=("cuda:0",), graph="required"):
    sim = simulation(steps=27, frequencies=(.8e9, 1e9))
    sim.scene.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
    sim.scene.add_material_region(mw.MaterialRegion(name="rho",
        geometry=mw.Box(position=(.05, 0, 0), size=(.1, .1, .1), device="cuda:0"),
        density=torch.zeros(2, 2, 2, device="cuda:0"), eps_bounds=(1., 4.)))
    sim.config.adjoint_checkpoint_stride = 7
    return mw.Simulation.batch(sim, parameters=mw.BatchParameters(region_densities={"rho": density}),
        execution=mw.BatchExecution(devices=devices, batch_size=capacity, slots_per_device=slots,
            graph=graph, graph_steps=8, memory_limit_bytes=256*1024**2, reserve_bytes=0))


def objective(results):
    return sum(result.raw_monitor("probe")["components"]["Ez"].abs().square().sum()
               for result in results)


@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("graph", ["off", "required"])
def test_batch_density_adjoint_matches_independent_after_reuse_and_close(slots, graph):
    density = torch.full((3, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density, slots=slots, graph=graph)
    with batch.prepare() as prepared:
        first = prepared.run()
        first.results()
        second = prepared.run()
        second.results()
    assert all(result.solver_stats["batch_adjoint"] == "native_checkpoint_per_lane" for result in first)
    actual, = torch.autograd.grad(objective(first), density)
    repeated, = torch.autograd.grad(objective(second), density)
    for result in (*first, *second):
        assert result.solver_stats["batch_adjoint_checkpoint_bytes"] > 0
        memory = result.solver_stats["batch_backward_memory"]
        assert memory["peak_charged_bytes"] <= memory["limit_bytes"]
    reference_density = density.detach().clone().requires_grad_()
    reference_batch = density_batch(reference_density)
    results = []
    for index in range(3):
        sim = mw.Simulation.fdtd(reference_batch.scene(index), frequencies=batch.template.frequencies,
            run_time=batch.template.config.run_time, spectral_sampler=batch.template.config.spectral_sampler,
            cuda_graph=False)
        sim.config.adjoint_checkpoint_stride = 7
        results.append(sim.run())
    expected, = torch.autograd.grad(objective(results), reference_density)
    assert expected.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-6)
    torch.testing.assert_close(repeated, expected, rtol=2e-3, atol=1e-6)


def test_batch_tape_rejects_in_place_parameter_changes():
    density = torch.full((1, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    results = density_batch(density).run()
    results.results()
    with torch.no_grad():
        density.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        objective(results).backward()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_batch_material_adjoint_gathers_gradients_from_two_devices():
    density = torch.full((2, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density, capacity=1, devices=("cuda:0", "cuda:1"))
    results = batch.run()
    results.results()
    loss = sum(result.raw_monitor("probe")["components"]["Ez"].abs().square().sum().to("cuda:0")
               for result in results)
    loss.backward()
    assert density.grad.abs().min() > 0
    torch.testing.assert_close(density.grad[0], density.grad[1], rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("normalize", [False])
def test_batch_source_scale_gradient_matches_finite_difference(normalize):
    sim = simulation(steps=35)
    sim.config.spectral_sampler = mw.SpectralSampler(window="none", normalize_source=normalize)
    sim.scene.add_source(mw.PointDipole(name="second", position=(0., -.05, 0.), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=.7)))
    scales = torch.tensor([.7, -1.2, 1.3 if normalize else 0.], device="cuda:0", requires_grad=True)
    other = torch.tensor([.4, .8, -.2], device="cuda:0", requires_grad=True)
    def make(a, b):
        return mw.Simulation.batch(sim,
            parameters=mw.BatchParameters(source_scales={"drive": a, "second": b}),
            execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2,
                graph="required", graph_steps=8, memory_limit_bytes=256*1024**2, reserve_bytes=0))
    result = make(scales, other).run()
    result.results()
    actual = torch.autograd.grad(objective(result), (scales, other))
    for parameter_index, parameter in enumerate((scales, other)):
        for lane in range(3):
            values = []
            for direction in (-1, 1):
                perturbed = [scales.detach().clone(), other.detach().clone()]
                perturbed[parameter_index][lane] += direction * .002
                trial = make(*perturbed).run()
                trial.results()
                values.append(objective(trial))
            expected = (values[1] - values[0]) / .004
            torch.testing.assert_close(actual[parameter_index][lane], expected, rtol=8e-3, atol=1e-5)


def test_normalized_source_preserves_physical_amplitude_gradient():
    sim = simulation(steps=35, frequencies=(.8e9, 1e9))
    sim.config.spectral_sampler = mw.SpectralSampler(window="none", normalize_source=True)
    scales = torch.tensor([.7, -1.2], device="cuda:0", requires_grad=True)
    batch = mw.Simulation.batch(sim, parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required"))
    result = batch.run()
    result.results()
    loss = objective(result)
    gradient, = torch.autograd.grad(loss, scales)
    expected = torch.stack([2*objective([item]).detach()/scale.detach()
                            for item, scale in zip(result, scales)])
    torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(result[0].raw_monitor("probe")["components"]["Ez"]/scales[0],
                               result[1].raw_monitor("probe")["components"]["Ez"]/scales[1], rtol=1e-4, atol=1e-6)


def test_shared_parameter_material_and_source_channels_reduce_once():
    parameter = torch.tensor(.4, device="cuda:0", requires_grad=True)
    def run(value):
        batch = density_batch(value.expand(2, 2, 2, 2))
        batch = mw.Simulation.batch(batch.template,
            parameters=mw.BatchParameters(region_densities={"rho": value.expand(2, 2, 2, 2)},
                source_scales={"drive": value.square().expand(2)}), execution=batch.execution)
        results = batch.run()
        results.results()
        return objective(results)
    actual, = torch.autograd.grad(run(parameter), parameter)
    expected = (run(parameter.detach() + .002) - run(parameter.detach() - .002)) / .004
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=1e-5)


def test_batch_tapes_release_without_cyclic_collection(monkeypatch):
    import gc
    import weakref
    from witwin.maxwell.execution.adjoint import BatchAdjointTape
    references = []
    original = BatchAdjointTape.__init__
    def tracked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        references.append(weakref.ref(self))
    monkeypatch.setattr(BatchAdjointTape, "__init__", tracked)
    density = torch.full((2, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = density_batch(density).run()
        result.results()
        assert len(references) == 2
        objective(result).backward()
        del result
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("slots", [1, 2])
def test_no_grad_batch_skips_tapes_and_can_resume_training(slots):
    density = torch.full((2, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    with density_batch(density, slots=slots).prepare() as prepared:
        with torch.no_grad():
            forward = prepared.run()
        forward.results()
        assert not objective(forward).requires_grad
        training = prepared.run()
        training.results()
        objective(training).backward()
        assert density.grad.abs().max() > 0


def test_scene_module_optimizer_updates_recompile_current_parameters():
    class Design(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.rho = torch.nn.Parameter(torch.full((2, 2, 2), .3, device="cuda:0"))
        def to_scene(self):
            scene = simulation(steps=27).scene
            scene.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
            scene.add_material_region(mw.MaterialRegion(name="rho",
                geometry=mw.Box(position=(.05, 0., 0.), size=(.1, .1, .1), device="cuda:0"),
                density=self.rho.sigmoid(), eps_bounds=(1., 4.)))
            return scene
    module = Design()
    sim = mw.Simulation.fdtd(module, frequency=1e9, run_time=mw.TimeConfig(time_steps=27, dt=2e-11),
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    batch = mw.Simulation.batch([sim], execution=mw.BatchExecution(devices=("cuda:0",),
        batch_size=1, graph="required"))
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    with batch.prepare() as prepared:
        before = prepared.run()
        before.results()
        objective(before).backward()
        assert module.rho.grad.abs().max() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = prepared.run()
        after.results()
        objective(after).backward()
    reference = sim.run()
    torch.testing.assert_close(after[0].raw_monitor("probe")["components"]["Ez"],
        reference.raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_gathered_training_outputs_survive_close_and_preserve_source_gradients():
    scales = torch.tensor([.4, -.7, 1.2, 0.], device="cuda:0", requires_grad=True)
    batch = mw.Simulation.batch(simulation(steps=29),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0", "cuda:1"), slots_per_device=2,
            batch_size=1, scheduling="static", graph="required", memory_limit_bytes=128*1024**2, reserve_bytes=0),
        outputs=mw.BatchOutput(result_device="cuda:0"))
    result = batch.run()
    result.results()
    assert {record.device for record in result.records} == {"cuda:0", "cuda:1"}
    values = result.stack(lambda r: r.raw_monitor("probe")["components"]["Ez"])
    values.real.sum().backward()
    assert scales.grad.abs().min() > 0
    torch.testing.assert_close(scales.grad, scales.grad[0].expand_as(scales), rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("channel", ["permeability", "conductivity", "region_geometry"])
def test_unsupported_material_gradient_channels_fail_explicitly(channel):
    from dataclasses import replace
    density = torch.full((1, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density)
    if channel == "permeability":
        region = batch.template.scene.material_regions[0]
        batch.template.scene.material_regions[0] = replace(region, mu_bounds=(1., 2.))
    elif channel == "conductivity":
        structure = batch.template.scene.structures[0]
        structure.geometry.size.requires_grad_(True)
        batch.template.scene.structures[0] = mw.Structure(name=structure.name, geometry=structure.geometry,
            material=mw.Material(eps_r=2., sigma_e=.01))
    else:
        batch.template.scene.material_regions[0].geometry.size.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="gradient channel|region geometry"):
        batch.prepare()
    with torch.no_grad(), batch.prepare() as prepared:
        result = prepared.run()
        result.results()
        assert not objective(result).requires_grad
        with torch.enable_grad():
            rejected = prepared.run()
            assert rejected.failed
            assert any(text in str(rejected.records[0].failure) for text in ("gradient channel", "region geometry"))


def test_dielectric_geometry_batch_gradient_matches_finite_difference():
    width = torch.tensor([.1, .08, .09], device="cuda:0", requires_grad=True)
    def run(size):
        sim = simulation(steps=35)
        # Preserve the differentiable arithmetic interface used by this finite-difference oracle.
        sim.scene.subpixel = mw.SubpixelSpec(samples=1, averaging="arithmetic")
        sim.scene.structures = [mw.Structure(name="design",
            geometry=mw.Box(position=(.05, 0., 0.), size=size, device="cuda:0"), material=mw.Material(eps_r=2.))]
        result = mw.Simulation.batch([sim], execution=mw.BatchExecution(devices=("cuda:0",),
            batch_size=1, graph="required")).run()
        result.results()
        return objective(result)
    gradient, = torch.autograd.grad(run(width), width)
    delta = torch.tensor([.00005, 0., 0.], device="cuda:0")
    finite = (run(width.detach()+delta)-run(width.detach()-delta))/(2*delta[0])
    assert finite.abs() > 0
    torch.testing.assert_close(gradient[0], finite, rtol=1e-2, atol=1e-5)


def test_requires_grad_changes_invalidate_cached_source_materialization():
    scales = torch.tensor([.4], device="cuda:0")
    batch = mw.Simulation.batch(simulation(steps=27),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="required"))
    with batch.prepare() as prepared:
        first = prepared.run()
        assert not objective(first).requires_grad
        scales.requires_grad_(True)
        second = prepared.run()
        objective(second).backward()
        assert scales.grad.abs().max() > 0
        scales.requires_grad_(False)
        last = prepared.run()
        assert not objective(last).requires_grad


def test_cpu_output_revalidates_new_trainable_parameters():
    scales = torch.tensor([.4], device="cuda:0")
    batch = mw.Simulation.batch(simulation(steps=27),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="off"),
        outputs=mw.BatchOutput(result_device="cpu"))
    with batch.prepare() as prepared:
        prepared.run().results()
        scales.requires_grad_(True)
        with pytest.raises(ValueError, match="detached forward"):
            prepared.run()
