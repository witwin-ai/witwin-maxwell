"""ADE closure ordering, coefficient packing and fused-path resume identity."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import core_path, core_tables, materials, stepping


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **kw: pytest.fail("Host check initialized CUDA."))


@pytest.mark.host_validation
@pytest.mark.parametrize("tfsf", [False, True])
def test_ordinary_ade_order(monkeypatch, tfsf):
    events = []
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.magnetic = lambda: events.append("yee_h")
    runtime.electric = lambda: events.append("yee_e")
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda *args:
        SimpleNamespace(launch=lambda: events.append("commit")))
    groups = ("magnetic", "electric", "ordinary", "tfsf_magnetic", "tfsf_electric")
    runtime.sources = SimpleNamespace(auxiliary=None, groups={
        name: [SimpleNamespace(launch=lambda n=name: events.append(n))] for name in groups})
    solver = SimpleNamespace(scene=SimpleNamespace(), tfsf_enabled=tfsf, observers_enabled=False,
        _advance_magnetic_dispersive_state=lambda: events.append("ade_h"),
        _apply_magnetic_dispersive_corrections=lambda: events.append("correct_h"),
        _advance_dispersive_state=lambda: events.append("ade_e"),
        _apply_dispersive_corrections=lambda: events.append("correct_e"))
    monkeypatch.setattr(core_tables, "compile_spectral_tables", lambda *a:
        SimpleNamespace(launch=lambda: events.append("dft")))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda *a: None)
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("pec"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    from witwin.maxwell.fdtd import cut_surface
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=True, time_steps=32)
    step()
    assert events == (["ade_h", "yee_h"] + (["tfsf_magnetic"] if tfsf else [])
        + ["magnetic", "correct_h", "ade_e", "yee_e"] + (["tfsf_electric"] if tfsf else [])
        + ["electric", "ordinary", "pec", "correct_e"] + ([] if tfsf else ["pec"])
        + ["mur", "cut", "dft", "commit"])


@pytest.mark.host_validation
@pytest.mark.parametrize("conductive", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_final_ade_coefficients_pack_bitwise(conductive, uniform):
    solver = SimpleNamespace(Nx=4, Ny=5, Nz=6, device="cpu", dt=1e-11,
        conductive_enabled=conductive, magnetically_conductive_enabled=False)
    # eps_inf and sigma are the background values used by the ordinary Yee launch.
    for axis, shape in zip("xyz", ((3, 5, 6), (4, 4, 6), (4, 5, 5))):
        eps = torch.full(shape, 2. * 8.8541878128e-12)
        if not uniform:
            eps[1:3] *= 1.5
        decay, curl = materials._electric_update_coefficients(solver, eps, torch.full(shape, .01), None)
        setattr(solver, f"ce{axis}_decay", decay)
        setattr(solver, f"ce{axis}_curl", curl)
    solver._coefficient_uniformity = {name: materials._uniform_coefficient_value(value)
        for name, value in vars(solver).items() if name.startswith("ce")}
    indices, table, scalars, packed_uniform = core_path._compress_coefficients(solver, "e")
    assert packed_uniform is uniform
    names = [f"ce{axis}_{kind}" for axis in "xyz" for kind in ("decay", "curl")]
    for column, name in enumerate(names):
        original = getattr(solver, name)
        restored = (torch.full_like(original, scalars[column]) if packed_uniform else
            table[indices.to(torch.int64), column][:original.shape[0], :original.shape[1], :original.shape[2]])
        assert torch.equal(restored.view(torch.int32), original.view(torch.int32)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core resume requires CUDA.")
def test_lorentz_two_pole_core_resume_bitwise(tmp_path):
    from benchmark.core_path.gates import make_simulation, tensors
    from benchmark.core_path.scenes import build_core_case

    steps = 3 * 32 + 13
    case = build_core_case("lorentz_two_pole_slab")

    def prepare():
        simulation = make_simulation(case, arm="candidate", steps=steps, full_field_dft=True)
        assert simulation.config.fdtd_core_path
        execution = simulation.prepare()
        execution.solver._core_runtime.chunk_size = 32
        return execution

    direct = prepare().run()
    assert direct.solver._core_step_graph_active
    assert direct.solver._core_step_graph_k == 32
    assert direct.solver._core_step_graph_chunks == [32, 32, 32, 13]

    partial = prepare()
    checkpoint = partial.run_until(45)
    assert checkpoint.step == checkpoint.physics.step == 45
    assert partial.solver._core_step_graph_active
    assert partial.solver._core_step_graph_k == 32
    assert partial.solver._core_step_graph_chunks == [32, 13]
    path = tmp_path / "lorentz-two-pole.pt"
    checkpoint.save(path)
    resumed = prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    # The resumed suffix uses per-step graphs, which stay eager for magnetic surface sources.
    assert resumed.solver._cuda_graph_active is not bool(resumed.solver._magnetic_source_terms)
    assert direct.stats()["steps_run"] == resumed.stats()["steps_run"] == steps

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(direct.solver, name), getattr(resumed.solver, name)), name
    expected_monitors = tensors(direct.monitors, "monitors")
    actual_monitors = tensors(resumed.monitors, "monitors")
    assert expected_monitors and expected_monitors.keys() == actual_monitors.keys()
    for name, value in expected_monitors.items():
        assert torch.equal(value, actual_monitors[name]), name

    expected_templates = direct.solver._dispersive_templates
    assert expected_templates and all(len(kinds["lorentz"]) == 2 for kinds in expected_templates.values())
    expected_ade = tensors(expected_templates, "ade")
    actual_ade = tensors(resumed.solver._dispersive_templates, "ade")
    assert expected_ade.keys() == actual_ade.keys()
    for name, value in expected_ade.items():
        assert torch.equal(value, actual_ade[name]), name
