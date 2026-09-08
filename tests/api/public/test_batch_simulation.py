from dataclasses import FrozenInstanceError, replace
import json

import pytest
import torch
import witwin.maxwell as mw


def simulation(*, frequency=1e9, boundary=None, grid=None, dt=1e-11):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
        grid=grid or mw.GridSpec.uniform(0.02),
        boundary=boundary or mw.BoundarySpec.none(), device="cpu",
    )
    scene.add_source(mw.PointDipole(
        name="drive", position=(0.0, 0.0, 0.0), polarization="Ez",
        source_time=mw.CW(frequency=1e9),
    ))
    scene.add_monitor(mw.PointMonitor(name="probe", position=(0.1, 0.0, 0.0), fields=("Ez",)))
    scene.add_material_region(mw.MaterialRegion(
        name="design", geometry=mw.Box(size=(0.1, 0.1, 0.1), device="cpu"),
        density=torch.zeros(2, 3, 1), eps_bounds=(1.0, 4.0),
    ))
    return mw.Simulation.fdtd(scene, frequency=frequency, run_time=mw.TimeConfig(time_steps=10, dt=dt))


def execution(**kwargs):
    return mw.BatchExecution(devices=("cuda:0",), **kwargs)


@pytest.mark.parametrize("kwargs,match", [
    ({"batch_size": True}, "batch_size"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 33}, "max_batch_size"),
    ({"max_batch_size": 0}, "max_batch_size"),
    ({"slots_per_device": 3}, "slots_per_device"),
    ({"slots_per_device": "auto"}, "tune"),
    ({"slots_per_device": True}, "slots_per_device"),
    ({"graph_steps": 0}, "graph_steps"),
    ({"memory_limit_bytes": -1}, "memory_limit_bytes"),
    ({"memory_limit_bytes": {"cuda:1": 100}}, "every execution device"),
    ({"reserve_bytes": -1}, "reserve_bytes"),
    ({"tune": 1}, "tune"),
    ({"fail_fast": 1}, "fail_fast"),
    ({"grouping": "guess"}, "grouping"),
    ({"scheduling": "random"}, "scheduling"),
    ({"graph": "maybe"}, "graph"),
])
def test_execution_validation(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        execution(**kwargs)


@pytest.mark.parametrize("devices", [("cpu",), ("cuda",), ("cuda:0", "cuda:0"), ()])
def test_explicit_unique_cuda_devices(devices):
    with pytest.raises(ValueError):
        mw.BatchExecution(devices=devices)


def test_execution_owns_immutable_device_limits():
    limits = {"cuda:0": 1024}
    config = execution(memory_limit_bytes=limits)
    limits["cuda:0"] = 10
    assert config.memory_limit_bytes["cuda:0"] == 1024
    with pytest.raises(TypeError):
        config.memory_limit_bytes["cuda:0"] = 20
    with pytest.raises(FrozenInstanceError):
        config.batch_size = 2


@pytest.mark.parametrize("kwargs", [
    {}, {"region_densities": {"design": torch.ones(2, 3, 1)}},
    {"source_scales": {"drive": torch.ones(2, dtype=torch.complex64)}},
    {"source_scales": {"drive": torch.ones(2, 1)}},
    {"source_scales": {"drive": torch.ones(2, dtype=torch.int64)}},
    {"source_scales": {"drive": torch.empty(0)}},
    {"source_scales": {"drive": torch.ones(3)}, "region_densities": {"design": torch.ones(2, 2, 3, 1)}},
])
def test_parameter_schema_rejects_ambiguous_batch_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        mw.BatchParameters(**kwargs)


def test_template_views_keep_gradients_and_own_scene_containers():
    sim = simulation()
    rho = torch.rand(4, 2, 3, 1, requires_grad=True)
    parameters = mw.BatchParameters(region_densities={"design": rho})
    batch = mw.Simulation.batch(sim, parameters=parameters, execution=execution())
    scene = batch.scene(2)
    assert isinstance(scene, mw.Scene)
    assert scene is not sim.scene
    assert scene.material_regions is not sim.scene.material_regions
    scene.material_regions[0].density.square().sum().backward()
    assert torch.equal(rho.grad[2], 2 * rho.detach()[2])
    assert torch.count_nonzero(rho.grad[[0, 1, 3]]) == 0
    assert not sim.scene.material_regions[0].density.requires_grad
    scene.monitors.clear()
    assert len(sim.scene.monitors) == 1


def test_noncontiguous_and_broadcast_parameter_views_are_not_copied():
    rho = torch.ones(1, 2, 3, 1, requires_grad=True).expand(4, -1, -1, -1)
    parameters = mw.BatchParameters(region_densities={"design": rho})
    assert parameters.region_densities["design"] is rho
    assert parameters.variant_count == 4


def test_source_scale_materialization_is_independent_and_trainable_scale_fails_closed():
    sim = simulation()
    scales = torch.tensor([1.0, 2.0])
    batch = mw.Simulation.batch(sim, parameters=mw.BatchParameters(source_scales={"drive": scales}), execution=execution())
    assert batch.scene(1).sources[0].source_time.amplitude == 2.0
    assert sim.scene.sources[0].source_time.amplitude == 1.0
    scales.requires_grad_()
    with pytest.raises(NotImplementedError, match="P6"):
        batch.scene(0)


def test_list_and_template_input_guards():
    sim = simulation()
    parameters = mw.BatchParameters(source_scales={"drive": torch.ones(2)})
    cases = [([], {}), ([sim], {"parameters": parameters}), (sim, {}),
             ([object()], {}), ([sim], {"variant_ids": ["a", "b"]}),
             ([sim, sim], {"variant_ids": ["a", "a"]})]
    for values, kwargs in cases:
        with pytest.raises((TypeError, ValueError)):
            mw.Simulation.batch(values, execution=execution(), **kwargs)


def test_target_and_shape_validation():
    sim = simulation()
    with pytest.raises(ValueError, match="missing"):
        mw.Simulation.batch(sim, parameters=mw.BatchParameters(source_scales={"missing": torch.ones(2)}), execution=execution())
    with pytest.raises(ValueError, match="spatial shape"):
        mw.Simulation.batch(sim, parameters=mw.BatchParameters(region_densities={"design": torch.ones(2, 3, 2, 1)}), execution=execution())
    with pytest.raises(ValueError, match="outputs.monitors"):
        mw.Simulation.batch([sim], execution=execution(), outputs=mw.BatchOutput(monitors=("missing",)))
    with pytest.raises(ValueError, match="material_regions"):
        mw.Simulation.batch([sim], execution=execution(), variation=mw.BatchVariation(material_regions=("missing",)))


def test_output_conflict_and_gradient_export():
    sim = simulation()
    sim.config.full_field_dft = True
    with pytest.raises(ValueError, match="conflicts"):
        mw.Simulation.batch([sim], execution=execution())
    mw.Simulation.batch([sim], execution=execution(), outputs=mw.BatchOutput(fields=("Ez",)))
    sim.config.full_field_dft = False
    with pytest.raises(ValueError, match="CPU result export"):
        mw.Simulation.batch(sim, execution=execution(),
            parameters=mw.BatchParameters(region_densities={"design": torch.ones(2, 2, 3, 1, requires_grad=True)}),
            outputs=mw.BatchOutput(result_device="cpu"))


@pytest.mark.parametrize("change,match", [
    ({"frequency": 2e9}, "frequencies"),
    ({"dt": 2e-11}, "dt"),
    ({"boundary": mw.BoundarySpec.pml(num_layers=3)}, "boundary"),
    ({"grid": mw.GridSpec.uniform(0.01)}, "grid"),
])
def test_strict_reports_specific_physics_difference(change, match):
    batch = mw.Simulation.batch([simulation(), simulation(**change)], execution=execution())
    with pytest.raises(ValueError, match=match):
        batch.plan()


def test_custom_grid_same_shape_different_coordinates_is_incompatible():
    a = [-0.2, -0.1, 0.1, 0.2]
    b = [-0.2, -0.05, 0.05, 0.2]
    first = simulation(grid=mw.GridSpec.custom(a, a, a))
    second = simulation(grid=mw.GridSpec.custom(b, a, a))
    with pytest.raises(ValueError, match="x_coords"):
        mw.Simulation.batch([first, second], execution=execution()).plan()


def test_compatible_groups_and_reports_are_stable_independent_snapshots():
    sims = [simulation(), simulation(frequency=2e9), simulation()]
    batch = mw.Simulation.batch(sims, execution=execution(grouping="compatible"))
    report = batch.plan()
    assert [g["variant_indices"] for g in report.groups] == [[0, 2], [1]]
    assert not report.resolved
    assert report.to_dict() == batch.plan().to_dict()
    exported = report.to_dict()
    exported["groups"].clear()
    assert len(report.groups) == 2
    json.dumps(report.to_dict())
    assert "planning_only" in report.to_markdown()


def test_plan_never_prepares_or_queries_gpu_and_does_not_infer_full_field_dft(monkeypatch):
    sim = simulation()
    sim.frequencies = tuple(1e9 + i * 1e7 for i in range(31))
    def forbidden(*args, **kwargs):
        raise AssertionError("dry planning must not allocate or prepare")
    monkeypatch.setattr(mw.Simulation, "prepare", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    report = mw.Simulation.batch([sim] * 100, execution=execution()).plan()
    assert report.variant_count == 100
    assert report.groups[0]["output_plan"]["full_field_dft"] is False
    assert report.groups[0]["shared_artifacts"] == []
    assert report.devices[0]["estimated_peak_bytes"] is None


@pytest.mark.parametrize("policy,match", [
    ({"tune": True}, "autotuning"),
    ({"slots_per_device": 2}, "one execution slot"),
])
def test_pending_execution_policies_fail_before_gpu_allocation(policy, match):
    batch = mw.Simulation.batch([simulation()], execution=execution(**policy))
    with pytest.raises(NotImplementedError, match=match):
        batch.prepare()


def test_trainable_batch_forward_is_not_silently_detached():
    sim = simulation()
    parameters = mw.BatchParameters(region_densities={"design": torch.ones(2, 2, 3, 1, requires_grad=True)})
    batch = mw.Simulation.batch(sim, parameters=parameters, execution=execution())
    with pytest.raises(NotImplementedError, match="adjoint"):
        batch.prepare()


def test_scene_module_still_materializes_to_the_standard_scene():
    class Module(mw.SceneModule):
        def to_scene(self):
            return simulation().scene
    sim = mw.Simulation.fdtd(Module(), frequency=1e9, run_time=mw.TimeConfig(time_steps=10))
    batch = mw.Simulation.batch([sim], execution=execution())
    assert isinstance(batch.scene(0), mw.Scene)
    assert batch.plan().variant_count == 1
