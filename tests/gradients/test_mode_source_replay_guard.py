"""Mode launch material gradients must differentiate the forward profile family."""
import copy
import math
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.core.material import VACUUM_PERMITTIVITY as EPS_0
from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge, _validate_mode_source_replay
from witwin.maxwell.fdtd.adjoint import core
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.fdtd.excitation.modes.common import _average_node_tensor_to_component
from witwin.maxwell.scene import prepare_scene


class _Guide(mw.SceneModule):
    def __init__(self, case, value=12.0, *, device="cpu", launch="source", scale=1., samples=1):
        super().__init__()
        self.case, self.device, self.launch, self.scale = case, device, launch, scale
        self.samples = samples
        self.eps = torch.nn.Parameter(torch.tensor(float(value), device=device))

    def to_scene(self):
        scale = self.scale
        scene = mw.Scene(
            domain=mw.Domain(bounds=tuple((-.6*scale, .6*scale) if axis == 0 else (-.3*scale, .3*scale) for axis in range(3))),
            grid=mw.GridSpec.uniform(0.02*scale), boundary=mw.BoundarySpec.pml(num_layers=8),
            device=self.device, subpixel_samples=mw.SubpixelSpec(samples=self.samples, averaging="arithmetic"),
        )
        lo, hi = {"source": (-0.26, -0.14), "control": (0.04, 0.16)}[self.case]
        for a, b, eps in ((-1., lo, 12.), (lo, hi, self.eps), (hi, 1., 12.)):
            scene.add_structure(mw.Structure(
                geometry=mw.Box(position=((a + b) / 2*scale, 0, 0), size=((b - a)*scale, 0.16*scale, 0.08*scale)),
                material=mw.Material(eps_r=eps),
            ))
        kwargs = dict(position=(-0.2*scale, 0, 0), size=(0, 0.4*scale, 0.4*scale), direction="+", polarization="Ey",
                      source_time=mw.GaussianPulse(frequency=1e9/scale, fwidth=0.2e9/scale), name="launch")
        if self.launch == "port":
            scene.add_port(mw.ModePort(**kwargs))
        else:
            scene.add_source(mw.ModeSource(**kwargs))
        scene.add_monitor(mw.PointMonitor("probe", (0.3*scale, 0, 0), fields=("Ey",)))
        scene.add_monitor(mw.FluxMonitor("fwd", axis="x", position=0.3*scale, frequencies=[1e9/scale]))
        return scene


def _forward_context(scene, *, spacing=.04, dtype=torch.float32):
    # Small host-only mode solves exercise the same material compiler and
    # no-grad launch as the native FDTD bridge, without allocating CUDA fields.
    scene.grid = mw.GridSpec.uniform(spacing)
    scene.boundary = mw.BoundarySpec.none()
    prepared = prepare_scene(scene, material_dtype=dtype)
    from witwin.maxwell.fdtd.runtime.materials import compile_forward_materials
    sources = tuple(_compile_mode_source(source, default_frequency=1e9)
                    for source in scene.resolved_sources())
    with torch.no_grad():
        model = compile_forward_materials(prepared, sources)
    context = SimpleNamespace(scene=prepared, _compiled_material_model=model, c=299792458.,
                              Ex=torch.empty(1, device=prepared.device, dtype=dtype),
                              eps0=EPS_0, device=prepared.device)
    for field, axis in zip(("Ex", "Ey", "Ez"), "xyz"):
        setattr(context, "eps_" + field,
                _average_node_tensor_to_component(model["eps_components"][axis], field) * EPS_0)
    context._compiled_sources = sources
    for source in context._compiled_sources:
        with torch.no_grad():
            data = solve_mode_source_profile(context, source)
        source["mode_solver_kind"] = data["mode_solver_kind"]
    return context


@pytest.mark.parametrize("launch", ["source", "port"])
@pytest.mark.parametrize("case", ["source", "control"])
@pytest.mark.parametrize("trainable", [True, False])
def test_plane_material_guard_is_spatial(case, launch, trainable):
    model = _Guide(case, launch=launch)
    model.eps.requires_grad_(trainable)
    scene = model.to_scene()
    context = _forward_context(scene)
    source = context._compiled_sources[0]
    assert source["mode_solver_kind"].startswith("vector_")
    inputs = (model.eps,) if trainable else ()
    _validate_mode_source_replay(context, scene, inputs)
    assert (source["mode_solver_kind"] == "vector_dense_torch") == (trainable and case == "source")
    assert source["mode_solver_kind"].startswith("vector_")


@pytest.mark.parametrize("forward_kind", ["scalar_dense", "scalar_dense_torch", "scalar_sparse",
                                          "scalar_sparse_implicit", "scalar_complex_dense"])
def test_matching_scalar_families_remain_admitted(forward_kind):
    model = _Guide("source")
    scene = model.to_scene()
    context = _forward_context(scene)
    context._compiled_sources[0]["mode_solver_kind"] = forward_kind
    if "complex" in forward_kind:
        context.eps_Ey = context.eps_Ey.to(torch.complex64) * (1 + 0.01j)
    _validate_mode_source_replay(context, scene, (model.eps,))


def test_complex_forward_real_replay_rejects_trainable_plane():
    model = _Guide("source")
    scene = model.to_scene()
    context = _forward_context(scene)
    context._compiled_sources[0]["mode_solver_kind"] = "scalar_complex_dense"
    with pytest.raises(NotImplementedError, match="launch.*scalar_complex_dense.*scalar_real"):
        _validate_mode_source_replay(context, scene, (model.eps,))


def test_trainable_launch_uses_torch_even_above_scipy_dense_cutoff(monkeypatch):
    import witwin.maxwell.fdtd.excitation.modes as modes

    monkeypatch.setattr(modes, "_FULL_VECTOR_DENSE_LIMIT", 1)
    model = _Guide("source")
    scene = model.to_scene()
    context = _forward_context(scene)
    assert context._compiled_sources[0]["mode_solver_kind"] == "vector_dense_torch"
    _validate_mode_source_replay(context, scene, (model.eps,))


def test_only_material_dependent_source_in_a_multisource_scene_switches():
    model = _Guide("source")
    scene = model.to_scene()
    scene.sources.insert(0, mw.ModeSource(
        position=(0.2, 0, 0), size=(0, 0.4, 0.4), polarization="Ey",
        source_time=mw.CW(1e9), name="fixed_plane",
    ))
    context = _forward_context(scene)
    _validate_mode_source_replay(context, scene, (model.eps,))
    assert context._compiled_sources[0]["mode_solver_kind"] == "vector_dense"
    assert context._compiled_sources[1]["mode_solver_kind"] == "vector_dense_torch"


def _waveport_guide(case="control", *, device="cpu"):
    model = _Guide(case, device=device)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.4, 0.4), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.05), boundary=mw.BoundarySpec.none(), device=device,
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="arithmetic"),
    )
    lo, hi = {"source": (-0.26, -0.14), "control": (0.04, 0.16)}[case]
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=((lo + hi) / 2, 0, 0), size=(hi - lo, 0.16, 0.08)),
        material=mw.Material(eps_r=model.eps),
    ))
    scene.add_port(mw.WavePort(
        "input", position=(-0.2, 0, 0), size=(0, 0.6, 0.3), direction="+",
        reference_plane=-0.2, modes=(mw.WaveModeSpec("te", polarization="Ez"),),
    ))
    return model, scene


def _waveport_simulation(scene):
    return mw.Simulation.fdtd(
        scene, frequencies=[1e9], excitations=mw.PortSweep(),
        run_time=mw.TimeConfig(time_steps=240),
        spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=False,
    )


@pytest.mark.parametrize("launch", ["source", "port"])
@pytest.mark.parametrize("trainable", [True, False])
def test_waveport_sweep_rejects_independent_mode_launches(launch, trainable):
    """Pins an upstream refusal that keeps fixed launch terms away from trainable planes.

    WavePort runs rewrite mode launches to fixed forward terms, which carry no
    launch-plane material graph. If this refusal is ever relaxed, the mode-replay
    guard in fdtd/adjoint/bridge.py must be extended to refuse fixed launch terms
    on trainable planes, or their material derivative is silently dropped.
    """
    model, scene = _waveport_guide("source")
    model.eps.requires_grad_(trainable)
    constructor = mw.ModeSource if launch == "source" else mw.ModePort
    launch_object = constructor(
        position=(-0.2, 0, 0), size=(0, 0.4, 0.4), polarization="Ey",
        source_time=mw.CW(1e9), name="independent_launch",
    )
    if launch == "source":
        scene.add_source(launch_object)
    else:
        scene.add_port(launch_object)
    with pytest.raises(ValueError, match="requires a Scene without independent field sources"):
        _waveport_simulation(scene).run()


def test_waveport_sweep_rejects_trainable_cross_section_through_simulation():
    """Pins an upstream refusal that keeps fixed launch terms away from trainable planes.

    WavePort runs rewrite mode launches to fixed forward terms, which carry no
    launch-plane material graph. If this refusal is ever relaxed, the mode-replay
    guard in fdtd/adjoint/bridge.py must be extended to refuse fixed launch terms
    on trainable planes, or their material derivative is silently dropped.
    """
    _, scene = _waveport_guide("source")
    with pytest.raises(NotImplementedError, match="fixed port cross-sections"):
        _waveport_simulation(scene).run()


def test_fixed_waveport_column_records_keep_forward_terms():
    from witwin.maxwell.waveport_sweep import _column_scene, resolve_waveport_run_manifest

    model, scene = _waveport_guide()
    manifest = resolve_waveport_run_manifest(scene, mw.PortSweep(), (1e9,))
    column = _column_scene(scene, manifest, active_port_index=0, active_mode_index=0,
                           frequency_index=0, amplitude=1.0)
    # Construct the real column's records on the forward-only CPU profile;
    # the native public run and backward are covered by the CUDA test below.
    model.eps.requires_grad_(False)
    simulation = mw.Simulation.fdtd(column, frequency=1e9, backend="cpu_reference",
                                    run_time=mw.TimeConfig(time_steps=2))
    simulation._fixed_waveport_mode_sources = True
    with torch.no_grad():
        solver = simulation.prepare().solver
    model.eps.requires_grad_(True)
    assert solver._compiled_sources
    assert all(source["kind"] == "fixed_waveport_mode_source"
               for source in solver._compiled_sources)
    assert not core._scene_has_mode_source(solver)
    terms = (solver._source_terms, solver._electric_source_terms, solver._magnetic_source_terms)
    assert any(terms)
    resolved = core._resolved_source_term_lists(solver, *(
        getattr(solver, "eps_" + field).detach().requires_grad_() for field in ("Ex", "Ey", "Ez")))
    assert all(actual is expected for actual, expected in zip(resolved, terms))
    _validate_mode_source_replay(solver, column, (model.eps,))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native WavePort adjoint")
def test_public_fixed_waveport_column_admits_backward():
    model, scene = _waveport_guide(device="cuda")
    result = _waveport_simulation(scene).run()
    assert result.solver._compiled_sources
    assert all(source["kind"] == "fixed_waveport_mode_source"
               for source in result.solver._compiled_sources)
    assert not core._scene_has_mode_source(result.solver)
    loss = result.network.s.abs().square().sum()
    assert loss.grad_fn is not None
    loss.backward()
    assert model.eps.grad is not None
    assert torch.isfinite(model.eps.grad)


def test_mode_port_excitation_is_not_an_rf_port():
    scene = _Guide("source", launch="port").to_scene()
    with pytest.raises(TypeError, match="requires an RF port.*ModePort"):
        mw.Simulation.fdtd(scene, frequencies=[1e9],
                           excitations=[mw.PortExcitation("launch", source_time=mw.CW(1e9))]).prepare()


def _edge_scene(case, parameter, *, material="eps"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
        grid=mw.GridSpec.uniform(0.1), boundary=mw.BoundarySpec.none(), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="arithmetic"),
    )
    lo, hi = {"one_cell_overlap": (-0.05, 0.05), "density_neighbor": (-0.2, 0.2),
              "padded_box": (0.025, 0.125), "away": (0.25, 0.35)}[case]
    density = parameter.expand(2, 2, 2)
    if case == "density_neighbor":
        # The only trainable texture layer is centered at x=+0.1, exactly
        # one grid cell from the plane. Trilinear sampling at x=0 reaches it.
        density = torch.stack((torch.full((2, 2), 0.5), parameter.expand(2, 2)))
    scene.add_material_region(mw.MaterialRegion(
        name=case, geometry=mw.Box(position=((lo + hi) / 2, 0, 0), size=(hi - lo, 0.2, 0.2)),
        density=density, eps_bounds=(1., 12.) if material == "eps" else (2., 2.),
        mu_bounds=(1., 2.) if material == "mu" else (1., 1.),
    ))
    scene.add_source(mw.ModeSource(position=(0, 0, 0), size=(0, 0.4, 0.4), polarization="Ey",
                                   source_time=mw.CW(1e9), name=case))
    return scene


def _exact_plane_dependencies(scene, parameter):
    """Check every plane entry's derivative, without a weighted VJP reduction."""
    from witwin.maxwell.fdtd.excitation.modes.common import (
        _field_component_axis_coords, _mode_slice, _mode_source_axes,
        _mode_source_relative_material_slices, _resolve_tangential_bounds,
    )
    from witwin.maxwell.fdtd.excitation.tfsf_common import nearest_index

    prepared = prepare_scene(scene)
    model = prepared.compile_materials()
    source = _compile_mode_source(scene.sources[0], default_frequency=1e9)
    axis, tangents = _mode_source_axes(source)
    plane = nearest_index(getattr(prepared, axis), source["position"]["xyz".index(axis)])
    bounds, _ = _resolve_tangential_bounds(prepared, source, axis_coords_by_axis={
        tangent: getattr(prepared, tangent) for tangent in tangents})
    eps, mu = _mode_source_relative_material_slices(
        SimpleNamespace(_compiled_material_model=model), frequency=1e9,
        normal_axis=axis, plane_index=plane, tangential_bounds=bounds)
    bounds, _ = _resolve_tangential_bounds(prepared, source, axis_coords_by_axis={
        tangent: _field_component_axis_coords(prepared, "Ey", tangent) for tangent in tangents})
    component = _mode_slice(model["edge_components"]["eps"]["Ey"], axis=axis,
                            plane_index=plane, tangential_bounds=bounds)

    def depends(values):
        found = False
        for value in values:
            if not value.requires_grad:
                continue
            for entry in value.flatten():
                (gradient,) = torch.autograd.grad(entry, parameter, allow_unused=True, retain_graph=True)
                found |= gradient is not None and bool(torch.any(gradient != 0))
        return found

    verdict = (depends(eps.values()), depends(mu.values()), depends((component,)))
    prepared.release_meshgrid()
    return verdict


@pytest.mark.parametrize("material", ["eps", "mu"])
@pytest.mark.parametrize("case", ["one_cell_overlap", "density_neighbor", "padded_box", "away"])
def test_plane_guard_matches_exact_autograd_at_sampling_edges(case, material):
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    scene = _edge_scene(case, parameter, material=material)
    truth = _exact_plane_dependencies(scene, parameter)
    dependent = case != "away"
    assert truth == (dependent and material == "eps", dependent and material == "mu",
                     dependent and material == "eps")
    context = _edge_forward_context(scene)
    if any(truth):
        with pytest.raises(NotImplementedError, match=case + ".*plane material is trainable"):
            _validate_mode_source_replay(context, scene, (parameter,))
    else:
        _validate_mode_source_replay(context, scene, (parameter,))


def _edge_forward_context(scene):
    prepared = prepare_scene(scene)
    with torch.no_grad():
        model = prepared.compile_materials()
    # Pin a PEC-aware family that still mismatches scalar replay. Lossless
    # conductor-free trainable vector planes now use matching torch replay.
    # Only solver-family selection is needed here; no eigenproblem or FDTD step.
    source = _compile_mode_source(scene.sources[0], default_frequency=1e9)
    source["mode_solver_kind"] = "vector_pec_dense_torch"
    return SimpleNamespace(scene=prepared, _compiled_material_model=model, eps0=EPS_0,
                           eps_Ey=model["edge_components"]["eps"]["Ey"] * EPS_0,
                           _compiled_sources=(source,))


def test_bridge_caches_verdict_and_invalidates_design_graph(monkeypatch):
    import witwin.maxwell.fdtd.adjoint.bridge as bridge_module
    from witwin.maxwell.scene import PreparedScene

    parameter = torch.nn.Parameter(torch.tensor(0.5))
    scene = _edge_scene("away", parameter)
    solver = _edge_forward_context(scene)
    bridge = _FDTDGradientBridge.__new__(_FDTDGradientBridge)
    bridge.simulation = SimpleNamespace(scene=scene)
    bridge._mode_replay_cache = None
    bridge._material_graph_scene = lambda: bridge.simulation.scene
    counts = dict(probe=0, compile=0)
    original_probe = bridge_module._mode_source_replay_error
    original_compile = PreparedScene.compile_materials

    def probe(*args):
        counts["probe"] += 1
        return original_probe(*args)

    def compile_materials(*args, **kwargs):
        counts["compile"] += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "_mode_source_replay_error", probe)
    monkeypatch.setattr(PreparedScene, "compile_materials", compile_materials)
    bridge._validate_mode_source_replay(solver, (parameter,))
    bridge._validate_mode_source_replay(solver, (parameter,))
    assert counts == dict(probe=1, compile=1)
    with torch.no_grad():
        parameter.add_(0.01)
    bridge._validate_mode_source_replay(solver, (parameter,))
    assert counts == dict(probe=2, compile=2)
    # A replacement material graph with the same input leaf must be rechecked.
    bridge.simulation.scene = _edge_scene("one_cell_overlap", parameter)
    for _ in range(2):
        with pytest.raises(NotImplementedError, match="plane material is trainable"):
            bridge._validate_mode_source_replay(solver, (parameter,))
    assert counts == dict(probe=3, compile=3)


def _source_snapshot(value):
    from witwin.maxwell.waveforms import WaveformTable

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _source_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_source_snapshot(item) for item in value)
    if is_dataclass(value):
        return {field.name: _source_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, WaveformTable):
        # deepcopy deliberately shares WaveformTable, so snapshot its arrays
        # and nested device-tensor cache explicitly to detect accidental writes.
        return {name: _source_snapshot(getattr(value, name)) for name in value.__slots__}
    return copy.deepcopy(value)


def _assert_deep_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_deep_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_deep_equal(left, right)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def test_replay_initialization_preserves_nested_sources_and_resume_fingerprint():
    from benchmark.core_path.gates import tensors as benchmark_tensors
    from witwin.maxwell.fdtd.excitation import initialize_source_terms
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint

    model = _Guide("control")
    model.eps.requires_grad_(False)
    scene = model.to_scene()
    scene.grid = mw.GridSpec.uniform(0.04)
    scene.boundary = mw.BoundarySpec.none()
    solver = mw.Simulation.fdtd(scene, frequency=1e9, backend="cpu_reference",
                                run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    before = _source_snapshot(solver._compiled_sources)
    benchmark_before = benchmark_tensors(solver._compiled_sources, "sources")
    fingerprint = _configuration_fingerprint(solver)
    replay = core._build_source_replay_solver(solver, solver._compiled_sources, *(
        getattr(solver, "eps_" + field).detach().requires_grad_() for field in ("Ex", "Ey", "Ez")))
    initialize_source_terms(replay)
    assert replay._compiled_sources[0]["mode_solver_kind"] == "scalar_dense_torch"
    assert solver._compiled_sources[0]["mode_solver_kind"] == "vector_dense"
    assert replay._compiled_sources[0]["beta"] != before[0]["beta"]
    _assert_deep_equal(_source_snapshot(solver._compiled_sources), before)
    assert _configuration_fingerprint(solver) == fingerprint
    _assert_deep_equal(benchmark_tensors(solver._compiled_sources, "sources"), benchmark_before)


@pytest.mark.parametrize("kind", sorted(core._SOURCE_REPLAY_REBUILD_KINDS - {"mode_source"}))
def test_other_replay_initializers_preserve_nested_source_payloads(kind):
    from witwin.maxwell.fdtd.excitation import initialize_source_terms

    waveform = mw.CustomSourceTime((0., 1e-8, 2e-8), (0., 1., 0.),
                                   characteristic_frequency=1e8, frequency=1e8)
    waveform.table.tensors("cpu")
    if kind == "custom_current":
        source = mw.CustomCurrentSource(mw.CurrentDataset(
            ((-0.1, 0.1),) * 3, {"Jy": np.ones((2, 2, 2))}), source_time=waveform)
    elif kind == "custom_field":
        source = mw.CustomFieldSource(mw.FieldDataset(
            ((0.,), (-0.1, 0.1), (-0.1, 0.1)),
            {"Ey": np.ones((1, 2, 2)), "Hz": np.ones((1, 2, 2))}), source_time=waveform)
    else:
        constructor = {"point_dipole": mw.PointDipole, "plane_wave": mw.PlaneWave,
                       "gaussian_beam": mw.GaussianBeam, "uniform_current": mw.UniformCurrentSource,
                       "astigmatic_gaussian_beam": mw.AstigmaticGaussianBeam}[kind]
        kwargs = {"position": (0., 0., 0.)} if kind == "point_dipole" else (
            {"size": (0.2, 0.2, 0.2)} if kind == "uniform_current" else {})
        source = constructor(source_time=waveform, **kwargs)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3), grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2), device="cpu", sources=[source])
    solver = mw.Simulation.fdtd(scene, frequency=1e9, backend="cpu_reference",
                                run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    before = _source_snapshot(solver._compiled_sources)
    replay = core._build_source_replay_solver(solver, solver._compiled_sources, *(
        getattr(solver, "eps_" + field).detach().requires_grad_() for field in ("Ex", "Ey", "Ez")))
    initialize_source_terms(replay)
    _assert_deep_equal(_source_snapshot(solver._compiled_sources), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD replay")
def test_second_backward_reuses_plane_verdict(monkeypatch):
    import witwin.maxwell.fdtd.adjoint.bridge as bridge_module
    from witwin.maxwell.scene import PreparedScene

    model = _Guide("control", device="cuda")
    result = mw.Simulation.fdtd(model, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=8),
                                 full_field_dft=False).run()
    loss = result.monitor("probe")["data"].real.sum()
    before = _source_snapshot(result.solver._compiled_sources)
    counts = dict(probe=0, compile=0)
    probing = False
    original_probe = bridge_module._mode_source_replay_error
    original_compile = PreparedScene.compile_materials

    def probe(*args):
        nonlocal probing
        counts["probe"] += 1
        probing = True
        try:
            return original_probe(*args)
        finally:
            probing = False

    def compile_materials(*args, **kwargs):
        if probing:
            counts["compile"] += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "_mode_source_replay_error", probe)
    monkeypatch.setattr(PreparedScene, "compile_materials", compile_materials)
    for _ in range(2):
        torch.autograd.grad(loss, model.eps, retain_graph=True)
        assert counts == dict(probe=1, compile=1)
        _assert_deep_equal(_source_snapshot(result.solver._compiled_sources), before)


def _objective(model):
    result = mw.Simulation.fdtd(model, frequencies=[1e9/model.scale],
        run_time=mw.TimeConfig.auto(transient_cycles=40), full_field_dft=False).run()
    data = result.monitor("probe")["data"].sum()
    loss = model.scale**2 * (data.real.square() + data.imag.square()) + result.monitor("fwd")["flux"].real.sum()
    return loss, result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD replay")
@pytest.mark.parametrize("case", ["source", "control"])
def test_rectangular_guide_replay_against_richardson_bound(case):
    model = _Guide(case, device="cuda")
    loss, result = _objective(model)
    source = result.solver._compiled_sources[0]
    before = _source_snapshot(result.solver._compiled_sources)
    forward_kind = source["mode_solver_kind"]
    assert forward_kind == ("vector_dense_torch" if case == "source" else "vector_dense")
    (gradient,) = torch.autograd.grad(loss, model.eps)
    assert source["mode_solver_kind"] == forward_kind
    _assert_deep_equal(_source_snapshot(result.solver._compiled_sources), before)
    values = {}
    for h in (0.05, 0.025):
        for sign in (-1, 1):
            with torch.no_grad():
                value, _ = _objective(_Guide(case, 12.0 + sign * h, device="cuda"))
                values[h, sign] = float(value)
    differences = {h: (values[h, 1] - values[h, -1]) / (2 * h) for h in (0.05, 0.025)}
    truncation = abs(differences[0.05] - differences[0.025]) / 3
    roundoff = 2 * torch.finfo(torch.float32).eps * math.sqrt(2000) * max(
        abs(value) for value in values.values()) / 0.025
    bound = truncation + roundoff
    assert math.isfinite(bound)
    assert abs(differences[0.025]) > bound
    assert abs(float(gradient) - differences[0.025]) <= bound
    coarse_bound = 4 * truncation + 2 * torch.finfo(torch.float32).eps * math.sqrt(2000) * max(
        abs(value) for value in values.values()) / 0.05
    assert abs(float(gradient) - differences[0.05]) <= coarse_bound
