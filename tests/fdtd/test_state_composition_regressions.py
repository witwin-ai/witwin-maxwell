"""Physical-state regressions for composed boundaries, materials, and RF loads."""
import pytest
import torch

import witwin.maxwell as mw
from tests.electrostatic.test_initial_condition import _uniform_dc_result
from tests.rf.network.test_network_observer_hot_path import _prepared_embedded_solver
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state, restore_checkpoint_state
from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators
from witwin.maxwell.fdtd.resume import _configuration_fingerprint
from witwin.maxwell.fdtd.runtime.stepping import _make_full_embedded_step_runner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _material_scene(material):
    scene = _prepared_embedded_solver().scene.clone()
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0.,) * 3, size=(0.04,) * 3), material=material))
    return scene


def _ferrite(bias=1.75e5):
    return mw.GyromagneticFerrite(eps_r=2., saturation_magnetization=1.4e5,
        bias_field=(0., 0., bias), gilbert_damping=0.01)


def _simulation(scene, *, graph=False, initial=None):
    return mw.Simulation.fdtd(scene, frequencies=(2.5e9, 3e9),
        run_time=mw.TimeConfig(time_steps=40), spectral_sampler=mw.SpectralSampler(window="none"),
        initial_condition=initial, cuda_graph=graph, full_field_dft=False)


def _dc_initial():
    return mw.ElectrostaticInitialCondition.from_result(_uniform_dc_result(
        device="cuda", bounds=((-0.02, 0.02),) * 3, h=0.005))


@pytest.mark.parametrize("fail_capture", [False, True])
def test_embedded_graph_restores_nonzero_ferrite_memory(fail_capture, monkeypatch):
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner

    solver = _simulation(_material_scene(_ferrite())).prepare().solver
    prepare_port_spectral_accumulators(solver, 40, "none")
    solver.Hx.fill_(0.01)
    before = capture_checkpoint_state(solver, 0)
    if fail_capture:
        def fail(self, function):
            function()
            function()
            raise RuntimeError("injected capture failure")
        monkeypatch.setattr(CudaGraphRunner, "capture", fail)
    runner = _make_full_embedded_step_runner(solver, use_cuda_graph=True)
    assert (runner is None) == fail_capture
    after = capture_checkpoint_state(solver, 0)
    for name, value in before.tensors.items():
        torch.testing.assert_close(after.tensors[name], value, rtol=0, atol=0)


def test_public_dc_ferrite_network_graph_matches_eager():
    scene = _material_scene(_ferrite())
    initial = _dc_initial()
    eager = _simulation(scene, initial=initial).run()
    captured = _simulation(scene, graph=True, initial=initial).run()
    assert initial.gauss_residual < initial.tolerance
    assert captured.stats()["cuda_graph_active"]
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(captured.solver, name), getattr(eager.solver, name), rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(captured.port("feed").voltage, eager.port("feed").voltage, rtol=2e-5, atol=2e-7)


def test_resume_rejects_changed_ferrite_bias_and_ignores_magnetization_history(tmp_path):
    prepared = _simulation(_material_scene(_ferrite()), initial=_dc_initial()).prepare()
    before = _configuration_fingerprint(prepared.solver)
    checkpoint = prepared.run_until(12)
    assert _configuration_fingerprint(prepared.solver) == before
    checkpoint.save(tmp_path / "ferrite.pt")
    checkpoint = mw.FDTDResumeCheckpoint.load(tmp_path / "ferrite.pt")
    changed = _simulation(_material_scene(_ferrite(2.75e5)), initial=_dc_initial()).prepare()
    assert changed.solver.dt == prepared.solver.dt
    with pytest.raises(ValueError, match="layout"):
        changed.run(resume_from=checkpoint)


def _breakdown_scene(*, minimum_duration=2e-10):
    scene = _material_scene(mw.Material(eps_r=2., breakdown=mw.DielectricBreakdown(
        critical_field=1., minimum_duration=minimum_duration, post_breakdown_conductivity=0.1)))
    # A changing conductivity needs a refreshed RF coupling operator, which is
    # explicitly unsupported. Exercise the complete breakdown state separately.
    return scene.clone(ports=(), networks=(), monitors=(
        mw.PointMonitor("breakdown_field", (0., 0., 0.), fields=("Ez",)),))


def test_dynamic_breakdown_rejects_a_stale_rf_coupling_operator():
    scene = _material_scene(mw.Material(eps_r=2., breakdown=mw.DielectricBreakdown(
        critical_field=1., minimum_duration=2e-10, post_breakdown_conductivity=0.1)))
    with pytest.raises(NotImplementedError, match="refreshing.*coupled field/circuit"):
        _simulation(scene, initial=_dc_initial()).prepare()


@pytest.mark.parametrize("stop", [0, 12, 24, 34])
def test_breakdown_disk_resume_preserves_timer_ramp_events_and_energy(tmp_path, stop):
    scene = _breakdown_scene()
    initial = _dc_initial()
    expected = _simulation(scene, initial=initial).run()
    prepared = _simulation(scene, initial=initial).prepare()
    fingerprint = _configuration_fingerprint(prepared.solver)
    checkpoint = prepared.run_until(stop)
    assert _configuration_fingerprint(prepared.solver) == fingerprint
    assert len(checkpoint.physics.schema.breakdown_state_names) == 7
    checkpoint.save(tmp_path / "breakdown.pt")
    loaded = mw.FDTDResumeCheckpoint.load(tmp_path / "breakdown.pt")
    assert loaded.physics.schema == checkpoint.physics.schema

    # Restoring coefficients itself must not advance the timer or duplicate heat.
    target = _simulation(scene, initial=initial).prepare().solver
    restore_checkpoint_state(target, loaded.physics)
    restored = capture_checkpoint_state(target, stop)
    for name in loaded.physics.schema.breakdown_state_names:
        torch.testing.assert_close(restored.tensors[name].cpu(), loaded.physics.tensors[name], rtol=0, atol=0)
    for name in ("cex_decay", "cex_curl", "cey_decay", "cey_curl", "cez_decay", "cez_curl"):
        torch.testing.assert_close(getattr(target, name), getattr(prepared.solver, name), rtol=0, atol=0)

    actual = _simulation(scene, initial=initial).prepare().run(resume_from=loaded)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(actual.solver, name), getattr(expected.solver, name), rtol=0, atol=0)
    assert actual.breakdown_events == expected.breakdown_events
    assert int(actual.solver.breakdown_runtime["state"].sum()) > 0
    for name in ("state", "timer", "trigger_step", "field_before", "deposited_at_trigger", "dissipated_energy", "total_dissipated"):
        torch.testing.assert_close(actual.solver.breakdown_runtime[name], expected.solver.breakdown_runtime[name])
    torch.testing.assert_close(actual.monitor("breakdown_field")["data"],
                               expected.monitor("breakdown_field")["data"], rtol=0, atol=0)


def test_breakdown_resume_rejects_changed_duration_before_trigger():
    checkpoint = _simulation(_breakdown_scene(), initial=_dc_initial()).prepare().run_until(12)
    changed = _simulation(_breakdown_scene(minimum_duration=3e-10), initial=_dc_initial()).prepare()
    with pytest.raises(ValueError, match="layout"):
        changed.run(resume_from=checkpoint)


@pytest.mark.parametrize("objective_kind", ["field", "port"])
def test_mur_network_material_gradient_matches_central_difference(objective_kind):
    from tests.gradients.test_fdtd_network_adjoint import _rational_network
    from tests.rf.network.test_network_runtime import _fdtd_scene

    def evaluate(value, trainable=False):
        block, _, _ = _rational_network(residue_value=2e8, direct_value=1e-2, trainable=False)
        logit = torch.tensor(value, device="cuda", dtype=torch.float64, requires_grad=trainable)
        scene = _fdtd_scene(network=block).clone(
            boundary=mw.BoundarySpec.mur(), subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"))
        scene.add_material_region(mw.MaterialRegion(name="design",
            geometry=mw.Box(position=(0.005, 0., 0.005), size=(0.005,) * 3),
            density=torch.sigmoid(logit).reshape(1, 1, 1), eps_bounds=(1., 4.), mu_bounds=(1., 1.)))
        scene.add_monitor(mw.PointMonitor("probe", (0.005, 0., 0.005), fields=("Ez",)))
        simulation = mw.Simulation.fdtd(scene, frequency=2.75e9,
            run_time=mw.TimeConfig(time_steps=200), spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
        simulation.config.adjoint_checkpoint_stride = 31
        result = simulation.run()
        spectrum = result.port("feed").voltage if objective_kind == "port" else result.monitor("probe")["data"]
        return spectrum.abs().square().sum(), logit

    objective, logit = evaluate(0.3, trainable=True)
    fixed, _ = evaluate(0.3)
    torch.testing.assert_close(objective, fixed, rtol=1e-6, atol=0)
    objective.backward()
    for delta in (0.004, 0.02):
        plus, _ = evaluate(0.3 + delta)
        minus, _ = evaluate(0.3 - delta)
        finite_difference = (plus - minus) / (2 * delta)
        torch.testing.assert_close(logit.grad, finite_difference.to(logit.grad.dtype), rtol=2e-3, atol=0)


def test_full_tensor_absorber_disk_resume_preserves_offdiagonal_memory(tmp_path):
    scene = _material_scene(mw.Material(epsilon_tensor=mw.Tensor3x3(
        ((2., 0.2, 0.1), (0.2, 3., 0.3), (0.1, 0.3, 2.5)))))
    scene = scene.clone(boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.), ports=(), networks=())
    scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
        source_time=mw.CW(frequency=2.5e9)))

    def simulation():
        return mw.Simulation.fdtd(scene, frequency=2.5e9,
            run_time=mw.TimeConfig(time_steps=40), spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False, cuda_graph=False)

    expected = simulation().prepare()
    expected.simulation._execute_fdtd_solve(expected.solver)
    checkpoint = simulation().prepare().run_until(24)
    names = tuple(name for name in checkpoint.physics.schema.cpml_state_names if "_aniso_" in name)
    assert len(names) == 9
    assert any(bool(checkpoint.physics.tensors[name].abs().max() > 0) for name in names)
    path = tmp_path / "tensor_absorber.pt"
    checkpoint.save(path)
    restored = mw.FDTDResumeCheckpoint.load(path)
    actual = simulation().prepare()
    actual.simulation._execute_fdtd_solve(actual.solver, resume_from=restored)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", *names):
        torch.testing.assert_close(getattr(actual.solver, name), getattr(expected.solver, name), rtol=0, atol=0)
    for name in names:
        restored.physics.tensors[name].zero_()
    corrupted = simulation().prepare()
    corrupted.simulation._execute_fdtd_solve(corrupted.solver, resume_from=restored)
    assert not torch.equal(corrupted.solver.Ez, expected.solver.Ez)
