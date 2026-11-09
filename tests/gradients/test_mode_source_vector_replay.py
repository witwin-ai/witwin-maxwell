"""Dense vector launch assembly, selection, gauge and source replay derivatives."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import linalg

from witwin.maxwell.fdtd.excitation import modes
from witwin.maxwell.fdtd.excitation.modes.vector import (
    _build_vector_operator_sparse, _build_vector_operator_torch,
    _relative_vector_residual, _vector_mode_power_overlap_numpy,
)
from witwin.maxwell.fdtd.excitation.modes.yee import (
    _build_yee_transverse_operator_sparse, _build_yee_transverse_operator_torch,
    _yee_stagger_eps_from_nodes,
)

FIELDS = ('Ey', 'Ez', 'Hy', 'Hz')
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA native FDTD replay')


def test_candidate_power_overlap_and_residual_have_no_absolute_floor():
    gram = np.array([[2., .25, 0.], [.25, 3., 0.], [0., 0., 0.]], dtype=np.complex128)
    expected = np.array([[1., .25 / math.sqrt(2.) / math.sqrt(3.), 0.],
                         [.25 / math.sqrt(3.) / math.sqrt(2.), 1., 0.], [0., 0., 0.]])
    for scale in (1., 1e-40):
        np.testing.assert_allclose(_vector_mode_power_overlap_numpy(gram * scale), expected,
                                   rtol=4*np.finfo(float).eps, atol=0)
        term = np.array([scale, 2*scale])
        assert _relative_vector_residual(.1*term, term) == pytest.approx(.1)
    assert _relative_vector_residual(np.zeros(2), np.zeros(2)) == 0.


def _planes(uniform=False, device='cpu'):
    rng = np.random.default_rng(12)
    eps = tuple(torch.tensor(np.full((7, 5), 3.) if uniform else 2 + rng.random((7, 5)),
                             dtype=torch.float64, device=device) for _ in range(3))
    mu = tuple(torch.ones_like(eps[0]) if uniform else 1 + 0.1 * torch.tensor(
        rng.random((7, 5)), device=device) for _ in range(3))
    return eps, mu


@pytest.mark.parametrize('uniform', [False, True])
def test_nodal_operator_entries(uniform):
    eps, mu = _planes(uniform)
    actual, _, _ = _build_vector_operator_torch(eps, mu, k0=30., du=.02, dv=.03)
    expected, _, _ = _build_vector_operator_sparse(tuple(x.numpy() for x in eps),
        tuple(x.numpy() for x in mu), k0=30., du=.02, dv=.03)
    np.testing.assert_allclose(actual.numpy(), expected.toarray(), rtol=8*np.finfo(float).eps,
                               atol=8*np.finfo(float).eps*np.max(abs(expected.data)))


@pytest.mark.parametrize('uniform', [False, True])
def test_yee_operator_entries(uniform):
    eps, _ = _planes(uniform)
    actual, _, _ = _build_yee_transverse_operator_torch(eps, k0=30., du=.02, dv=.03)
    staggered = _yee_stagger_eps_from_nodes(tuple(x.numpy() for x in eps), nu_cells=6, nv_cells=4)
    expected, _ = _build_yee_transverse_operator_sparse(nu_cells=6, nv_cells=4, k0=30.,
        du=.02, dv=.03, eps_uu=staggered[0], eps_vv=staggered[1], eps_ww=staggered[2])
    np.testing.assert_allclose(actual.numpy(), expected.toarray(), rtol=8*np.finfo(float).eps,
                               atol=8*np.finfo(float).eps*np.max(abs(expected.data)))


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('length', [1e-6, 1e6])
def test_vector_candidate_selection_is_length_covariant(uniform, length):
    eps, mu = _planes(uniform)

    def solve(scale):
        return modes._solve_vector_mode_torch(eps, mu,
            source=dict(name='scaled', mode_index=0), k0=90./scale,
            du=.02*scale, dv=.03*scale, field_names=FIELDS, preferred_field_name='Ey')

    beta, profiles, _ = solve(1.)
    scaled_beta, scaled_profiles, _ = solve(length)
    torch.testing.assert_close(beta, scaled_beta*length, rtol=1e-12, atol=0)
    for name in profiles:
        torch.testing.assert_close(profiles[name], scaled_profiles[name], rtol=2e-11, atol=2e-12)


def test_gauge_invariance():
    generator = torch.Generator().manual_seed(21)
    profiles = {name: torch.randn((5, 3), dtype=torch.complex128, generator=generator) for name in FIELDS}
    fixed = modes._gauge_vector_profiles(profiles, 'Ey')
    rotated = modes._gauge_vector_profiles({name: value * np.exp(.73j) for name, value in profiles.items()}, 'Ey')
    for name in FIELDS:
        torch.testing.assert_close(fixed[name], rotated[name], atol=1e-15, rtol=1e-15)
    peak = fixed['Ey'].flatten()[profiles['Ey'].abs().argmax()]
    torch.testing.assert_close(peak, torch.ones_like(peak))


def test_gauge_keeps_opposite_sign_peak_until_half_peak_threshold():
    source = {'name': 'sticky'}

    def gauge(values, phase=1.):
        preferred = torch.tensor(values, dtype=torch.complex128) * phase
        profiles = {name: preferred for name in FIELDS}
        return modes._gauge_vector_profiles(profiles, 'Ey', state=source)['Ey']

    gauge([1., -.99])
    assert source['gauge'][2] == 0
    rebuilt = dict(source)
    result = gauge([.99, -1.])
    assert source['gauge'][2] == 0
    assert result[0].real > 0 and result[1].real < 0
    rotated = gauge([.99, -1.], np.exp(.73j))
    torch.testing.assert_close(result, rotated, rtol=1e-15, atol=1e-15)
    replay = modes._gauge_vector_profiles(
        {name: torch.tensor([.99, -1.], dtype=torch.complex128) for name in FIELDS},
        'Ey', state=rebuilt)
    assert torch.equal(result, replay['Ey'])
    gauge([.5, -1.])
    assert source['gauge'][2] == 0
    result = gauge([.49, -1.])
    assert source['gauge'][2] == 1
    assert result[1].real > 0


@pytest.mark.parametrize('component', range(6))
@pytest.mark.parametrize('entrypoint', ['solve', 'operator'])
def test_lossy_trainable_vector_planes_refuse_before_real_projection(component, entrypoint):
    eps, mu = map(list, _planes())
    planes = eps if component < 3 else mu
    planes[component % 3] = planes[component % 3].to(torch.complex128).requires_grad_()
    planes[component % 3] = planes[component % 3] - 1e-3j
    with pytest.raises(NotImplementedError, match='lossy.*lossless.*imaginary magnitude > 1e-7'):
        if entrypoint == 'operator':
            _build_vector_operator_torch(eps, mu, k0=90., du=.02, dv=.03, source_name='lossy')
        else:
            modes._solve_vector_mode_torch(eps, mu,
                source={'name': 'lossy', 'mode_index': 0}, k0=90., du=.02, dv=.03,
                field_names=FIELDS, preferred_field_name='Ey')


def test_yee_operator_refuses_lossy_plane():
    eps, _ = _planes(True)
    with pytest.raises(NotImplementedError, match='lossy-yee.*lossless'):
        _build_yee_transverse_operator_torch(tuple(value - 1e-3j for value in eps),
            k0=40., du=.02, dv=.03, source_name='lossy-yee')


def test_lossless_tolerance_matches_forward_router():
    eps, mu = _planes()
    real_operator, _, _ = _build_vector_operator_torch(eps, mu, k0=90., du=.02, dv=.03)
    operator, _, _ = _build_vector_operator_torch(
        tuple(value - 1e-7j for value in eps), mu, k0=90., du=.02, dv=.03)
    assert torch.equal(operator, real_operator)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_lossless_threshold_uses_router_precision(dtype):
    eps = torch.full((3, 3), 2. - 1e-7j, dtype=dtype)
    if modes._max_component_imag({'x': eps}) > 1e-7:
        with pytest.raises(NotImplementedError, match='threshold.*lossless'):
            modes._require_lossless_vector_planes((eps,), (), source_name='threshold')
    else:
        modes._require_lossless_vector_planes((eps,), (), source_name='threshold')


@pytest.mark.parametrize('relative_imag,refuses', [(0., False), (1e-9, False), (1e-6, True)])
def test_selected_eigenpair_checks_imaginary_part(relative_imag, refuses):
    operator = torch.diag(torch.tensor([2., 3.], dtype=torch.float64))
    value = torch.tensor(2. * (1 + relative_imag*1j), dtype=torch.complex128)
    vector = torch.tensor([1., 0.], dtype=torch.complex128)
    if refuses:
        with pytest.raises(NotImplementedError, match='complex-pair.*selected vector eigenvalue is complex'):
            modes._IsolatedVectorEigenpair.apply(operator, value, vector, 'complex-pair')
    else:
        result, _ = modes._IsolatedVectorEigenpair.apply(operator, value, vector, 'complex-pair')
        assert result == value.real


@pytest.mark.parametrize('trainable', ['epsilon', 'sigma'])
def test_conductive_trainable_launch_plane_refuses(trainable):
    from dataclasses import replace
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    model = _Guide('source')
    scene = model.to_scene()
    sigma = torch.nn.Parameter(torch.tensor(.01)) if trainable == 'sigma' else .01
    scene.structures[1] = replace(scene.structures[1], material=mw.Material(
        eps_r=model.eps if trainable == 'epsilon' else 12., sigma_e=sigma))
    with pytest.raises(NotImplementedError, match='launch.*lossless.*conductivity'):
        _forward_context(scene)


def test_two_forwards_reuse_own_material_compile(monkeypatch):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    from witwin.maxwell import scene as scene_module
    from witwin.maxwell.compiler import materials
    from witwin.maxwell.compiler.sources import _compile_mode_source
    from witwin.maxwell.fdtd.runtime.materials import build_materials
    from witwin.maxwell.fdtd.runtime import timestep
    from witwin.maxwell.scene import prepare_scene
    calls = []
    compile_material_model = materials.compile_material_model

    def counted(*args, **kwargs):
        calls.append(1)
        return compile_material_model(*args, **kwargs)

    monkeypatch.setattr(materials, 'compile_material_model', counted)
    monkeypatch.setattr(scene_module, 'compile_material_model', counted)
    # Exercise production material initialization without native field allocation.
    monkeypatch.setattr(timestep, 'compile_fdtd_time_step', lambda *args: None)
    model = _Guide('source')
    scene = model.to_scene()
    scene.grid = mw.GridSpec.uniform(.04)
    scene.boundary = mw.BoundarySpec.none()
    for forward_number in (1, 2):
        prepared = prepare_scene(scene)
        source = _compile_mode_source(scene.sources[0], default_frequency=1e9)
        context = SimpleNamespace(scene=prepared, _compiled_sources=(source,), source_frequency=1e9,
                                  eps0=8.8541878128e-12, mu0=1.25663706212e-6, c=299792458.,
                                  Ex=torch.empty(1), device='cpu')
        with torch.no_grad():
            build_materials(context, prepared)
            data = modes.solve_mode_source_profile(context, source)
        assert data['mode_solver_kind'] == 'vector_dense_torch'
        source['mode_solver_kind'] = data['mode_solver_kind']
        from witwin.maxwell.fdtd.adjoint.bridge import _validate_mode_source_replay
        _validate_mode_source_replay(context, scene, (model.eps,))
        assert len(calls) == forward_number


@pytest.mark.parametrize('launch', ['source', 'port'])
@pytest.mark.parametrize('scale', [1., 1e-6])
def test_preparation_replay_receives_forward_materials_before_guard(monkeypatch, launch, scale):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    from witwin.maxwell.fdtd.adjoint.core import _build_source_replay_solver
    from witwin.maxwell.fdtd.excitation.injection import initialize_source_terms
    from witwin.maxwell.fdtd import cpu_profile

    # Exercise initialization only; native differentiable stepping remains CUDA-only.
    monkeypatch.setattr(cpu_profile, 'admit_cpu_reference_scene',
                        lambda simulation, prepared: setattr(prepared, '_fdtd_cpu_reference', True))
    model = _Guide('source', launch=launch, scale=scale, samples=2)
    material_graphs = []
    for _ in range(2):
        scene = model.to_scene()
        scene.grid = mw.GridSpec.uniform(.04*scale)
        scene.boundary = mw.BoundarySpec.none()
        if scale == 1.:
            scene.add_source(mw.ModeSource(
                position=(.2, 0, 0), size=(0, .4, .4),
                polarization='Ey', source_time=mw.CW(1e9), name='fixed'))
        with torch.no_grad():
            simulation = mw.Simulation.fdtd(scene, frequency=1e9/scale, backend='cpu_reference',
                                            run_time=mw.TimeConfig(time_steps=2))
            solver = simulation._build_fdtd_solver(initialize=True)
        source_name = 'launch' if launch == 'source' else 'launch::source'
        source_index = next(index for index, source in enumerate(solver._compiled_sources)
                            if source['name'] == source_name)
        assert set(solver._source_vector_materials) == {source_name}
        materials = solver._source_vector_materials[source_name]
        material_graphs.append(materials)
        assert any(value.requires_grad for value in materials[0].values())
        assert solver._compiled_sources[source_index]['mode_solver_kind'] == 'vector_dense_torch'
        if scale == 1.:
            fixed = next(source for source in solver._compiled_sources if source['name'] == 'fixed')
            assert fixed['mode_solver_kind'] == 'vector_dense'
        forward_terms = tuple(term for term in solver._electric_source_terms
                              if term['source_index'] == source_index)
        replay = _build_source_replay_solver(
            solver, solver._compiled_sources, solver.eps_Ex, solver.eps_Ey, solver.eps_Ez)
        assert replay._source_vector_materials[source_name] is materials
        with torch.enable_grad():
            initialize_source_terms(replay)
        replay_terms = tuple(term for term in replay._electric_source_terms
                             if term['source_index'] == source_index)
        assert len(forward_terms) == len(replay_terms) > 0
        for forward, rebuilt in zip(forward_terms, replay_terms):
            assert forward['field_name'] == rebuilt['field_name']
            assert forward['offsets'] == rebuilt['offsets']
            assert torch.equal(forward['patch'], rebuilt['patch'])
        assert replay._mode_source_vector_state == solver._mode_source_vector_state
        assert replay._mode_source_vector_state[source_name] is not solver._mode_source_vector_state[source_name]
        objective = sum(term['patch'].square().sum() for term in replay._electric_source_terms
                        if term['source_index'] == source_index)
        gradient, = torch.autograd.grad(objective, model.eps)
        assert torch.isfinite(gradient) and gradient != 0
    assert material_graphs[0] is not material_graphs[1]


def test_fixed_mode_source_resume_fingerprint_excludes_vector_runtime_state():
    import copy
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint

    model = _Guide('source')
    model.eps.requires_grad_(False)
    scene = model.to_scene()
    scene.grid = mw.GridSpec.uniform(.04)
    scene.boundary = mw.BoundarySpec.none()
    solver = mw.Simulation.fdtd(scene, frequency=1e9, backend='cpu_reference',
                                run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    current = _configuration_fingerprint(solver)
    # d3b11a6b source records contain neither routing flags nor sticky gauges.
    baseline = copy.copy(solver)
    runtime_keys = {'_torch_vector_plane', '_vector_gauge'}
    baseline._compiled_sources = tuple(
        {key: value for key, value in source.items() if key not in runtime_keys}
        for source in solver._compiled_sources)
    assert _configuration_fingerprint(baseline) == current
    assert all(runtime_keys.isdisjoint(source) for source in solver._compiled_sources)
    solver._mode_source_vector_state['launch']['gauge'] = ('Ey', (3, 3), 4)
    solver._source_vector_materials = {'launch': ({'x': torch.ones(3)}, {})}
    assert _configuration_fingerprint(solver) == current


@pytest.mark.parametrize('owner', ['scene', 'module'])
def test_scene_retains_source_gauge_across_solver_and_simulation_rebuilds(monkeypatch, owner):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    model = _Guide('source')
    model.eps.requires_grad_(False)
    to_scene = model.to_scene

    def host_scene():
        scene = to_scene()
        scene.grid = mw.GridSpec.uniform(.04)
        scene.boundary = mw.BoundarySpec.none()
        return scene

    model.to_scene = host_scene
    scene = model.to_scene()
    simulation_input = scene if owner == 'scene' else model
    # Select the vector implementation on the host forward-only profile; this
    # tests Simulation's gauge ownership without a native adjoint or CUDA fields.
    monkeypatch.setattr(modes, '_mode_source_trainable_plane', lambda *args, **kwargs: True)
    simulation = mw.Simulation.fdtd(simulation_input, frequency=1e9, backend='cpu_reference',
                                   run_time=mw.TimeConfig(time_steps=2))
    seen = []
    gauge = modes._gauge_vector_profiles

    def capture(profiles, field, *, state):
        seen.append(profiles[field].detach().abs().flatten())
        return gauge(profiles, field, state=state)

    monkeypatch.setattr(modes, '_gauge_vector_profiles', capture)
    with torch.no_grad():
        first = simulation.prepare().solver
    record = first._mode_source_vector_state['launch']['gauge']
    candidates = torch.where(seen[0] >= .5 * seen[0].max())[0].tolist()
    alternative = next(index for index in candidates if index != record[2])
    simulation_input._mode_source_gauges['launch'] = (*record[:2], alternative)
    for rebuilt in (simulation, mw.Simulation.fdtd(
            simulation_input, frequency=1e9, backend='cpu_reference', run_time=mw.TimeConfig(time_steps=2))):
        with torch.no_grad():
            second = rebuilt.prepare().solver
        assert second is not first
        assert second._mode_source_vector_state['launch']['gauge'][2] == alternative


def test_retained_material_graph_preserves_fixed_pec_launch_admission(monkeypatch):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    model = _Guide('control')
    scene = model.to_scene()
    scene.add_structure(mw.Structure(
        mw.Box(position=(0., 0., 0.), size=(1.2, .08, .08)), mw.Material(pec=True)))
    admitted = []

    def assemble(solver, source, **kwargs):
        assert bool(torch.any(kwargs['pec_occupancy'][1:-1, 1:-1] >= .5))
        assert not solver._mode_source_vector_state[source['name']]['trainable']
        assert not any(value.requires_grad for value in kwargs['eps_by_axis'].values())
        admitted.append(source['name'])
        return {'mode_solver_kind': 'vector_pec_dense_torch'}

    monkeypatch.setattr(modes, '_assemble_vector_mode_data', assemble)
    _forward_context(scene)
    assert admitted == ['launch']


def test_square_guide_refuses_degenerate_polarizations():
    eps = tuple(torch.ones((7, 7), dtype=torch.float64, requires_grad=True) for _ in range(3))
    with pytest.raises(NotImplementedError, match='square.*relative gap=.*delta='):
        modes._solve_vector_mode_torch(eps, tuple(torch.ones_like(x) for x in eps),
            source={'name': 'square', 'mode_index': 0, 'wave_family': 'guided'},
            k0=40., du=.02, dv=.02, field_names=FIELDS, preferred_field_name='Ey')


@pytest.mark.parametrize('scale', [1., 1e-6, 1e6])
def test_nodal_selection_and_profile_derivative(scale):
    eps, mu = _planes()
    parameter = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    mask = torch.linspace(.1, 1., eps[0].numel(), dtype=torch.float64).reshape(eps[0].shape)
    source = {'name': 'graded', 'mode_index': 0}

    def solve(value):
        return modes._solve_vector_mode_torch(tuple(x + value * mask for x in eps), mu,
            source=source, k0=90./scale, du=.02*scale, dv=.03*scale,
            field_names=FIELDS, preferred_field_name='Ey')

    beta, profiles, diagnostics = solve(parameter)
    operator, nu, nv = _build_vector_operator_sparse(tuple(x.numpy() for x in eps),
        tuple(x.numpy() for x in mu), k0=90./scale, du=.02*scale, dv=.03*scale)
    values, vectors = linalg.eig(operator.toarray())
    expected, _, arrays, reference = modes._select_and_normalize_vector_mode_numpy(
        values, vectors, interior_u=nu, interior_v=nv, mode_index=0, field_names=FIELDS,
        preferred_field_name='Ey', operator=operator, eps_planes=tuple(x.numpy() for x in eps),
        mu_planes=tuple(x.numpy() for x in mu), k0=90./scale, du=.02*scale, dv=.03*scale)
    assert float(beta.detach()) == pytest.approx(expected.real, rel=1e-13)
    for name in FIELDS:
        np.testing.assert_allclose(profiles[name].detach().real.numpy(), arrays[name].real, atol=2e-12)
    # A spatial cotangent checks mode-shape changes that a profile norm can miss.
    weights = {name: torch.sin(torch.arange(value.numel(), dtype=torch.float64)
                              .reshape(value.shape) * .37)
               for name, value in profiles.items()}
    def contract(fields):
        return sum((value.real * weights[name]).sum() for name, value in fields.items())
    loss = contract(profiles)
    gradient, = torch.autograd.grad(loss, parameter)
    def objective(value):
        return contract(solve(value)[1])
    h = 1e-5
    fd = (objective(h) - objective(-h)) / (2*h)
    torch.testing.assert_close(gradient, fd, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
def test_above_cap_refuses(device):
    eps = tuple(torch.ones((49, 49), device=device, requires_grad=True) for _ in range(3))
    with pytest.raises(NotImplementedError, match='large.*plane material is trainable.*cap n <= 4096'):
        modes._solve_vector_mode_torch(eps, tuple(torch.ones_like(x) for x in eps),
            source={'name': 'large', 'mode_index': 0}, k0=40., du=.02, dv=.02,
            field_names=FIELDS, preferred_field_name='Ey')


def test_yee_rectangular_profile_derivative():
    parameter = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    eps = tuple(parameter.expand(7, 5) for _ in range(3))
    beta, profiles, _ = modes._solve_vector_mode_torch(eps, tuple(torch.ones_like(x) for x in eps),
        source={'name': 'rectangle', 'mode_index': 0, 'wave_family': 'guided'},
        k0=40., du=.02, dv=.02, field_names=FIELDS, preferred_field_name='Ey')
    expected_beta, expected_profiles, _ = modes._solve_yee_transverse_vector_mode(
        tuple(x.detach().numpy() for x in eps), k0=40., du=.02, dv=.02,
        mode_index=0, field_names=FIELDS, preferred_field_name='Ey',
        wave_family='guided', uniform=True, use_dense=True)
    assert float(beta.detach()) == pytest.approx(expected_beta, rel=1e-13)
    for name in FIELDS:
        np.testing.assert_allclose(profiles[name].detach().numpy(), expected_profiles[name], atol=1e-12)
    loss = profiles['Hz'].real.square().sum() + beta
    gradient, = torch.autograd.grad(loss, parameter)
    assert torch.isfinite(gradient) and gradient != 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('launch', ['source', 'port'])
@pytest.mark.parametrize('scale', [1., 1e-6])
def test_forward_replay_component_profiles_equal(device, launch, scale):
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    from witwin.maxwell.fdtd.adjoint.bridge import _validate_mode_source_replay
    model = _Guide('source', device=device, launch=launch, scale=scale)
    scene = model.to_scene()
    context = _forward_context(scene, spacing=.04*scale)
    context.device = device
    context.Ex = torch.empty(1, device=device)
    source = context._compiled_sources[0]
    with torch.no_grad():
        forward = modes.solve_mode_source_profile(context, source)
    _validate_mode_source_replay(context, scene, (model.eps,))
    context._mode_source_rebuild_from_fields = True
    replay = modes.solve_mode_source_profile(context, dict(source))
    assert forward['mode_solver_kind'] == replay['mode_solver_kind'] == 'vector_dense_torch'
    for name in forward['component_profiles']:
        assert torch.equal(forward['component_profiles'][name], replay['component_profiles'][name])
    objective = sum(value.square().sum() for value in replay['component_profiles'].values())
    gradient, = torch.autograd.grad(objective, model.eps)
    assert torch.isfinite(gradient) and gradient != 0


@pytest.mark.parametrize('length', [1., 1e-6])
@pytest.mark.parametrize('amplitude', [1., 1e-20])
@pytest.mark.parametrize('case', ['valid', 'zero', 'cancellation'])
def test_vector_finalization_shares_scale_relative_power_guard(length, amplitude, case):
    from witwin.maxwell.fdtd.excitation.modes.common import _discrete_mode_profile_power
    coordinates = torch.linspace(0., length, 5, dtype=torch.float64)
    electric = torch.full((5, 5), amplitude, dtype=torch.float64)
    profiles = dict(Ey=electric, Ez=electric / 2, Hy=-electric / 100, Hz=electric / 200)
    if case == 'zero':
        profiles = {name: value * 0 for name, value in profiles.items()}
    elif case == 'cancellation':
        profiles['Hy'] = -profiles['Hy']
    solver = SimpleNamespace(device='cpu', scene=SimpleNamespace(device='cpu'),
                             Ex=torch.empty(1, dtype=torch.float64))
    results = []
    for kind in ('vector_dense', 'vector_dense_torch'):
        # Both solver families must normalize the unscaled fields, even when
        # their dimensional power and propagation constant are extremely small.
        kwargs = dict(beta_value=torch.tensor(2e-40, dtype=torch.float64),
            component_arrays={name: value.numpy() if kind == 'vector_dense' else value
                              for name, value in profiles.items()},
            pec_node_count=0, solver_kind=kind,
            candidate_diagnostics=dict(candidates=[], overlap_matrix=[], selected_candidate_index=0),
            k0=1e-40, eps_by_axis={}, mu_by_axis={}, tensor_eps_planes=(electric,),
            field_names=FIELDS, preferred_field_name='Ey', normal_axis='x',
            tangential_axes=('y', 'z'), tangential_bounds=((0, 4), (0, 4)),
            tangential_coord_map=dict(y=coordinates, z=coordinates), plane_index=1,
            native_tem=False, du=length / 4, tem_coordinates={})
        if case != 'valid':
            with pytest.raises(RuntimeError, match='zero integrated Poynting'):
                modes._finalize_vector_mode_data(solver, dict(direction_sign=1), **kwargs)
            continue
        result = modes._finalize_vector_mode_data(solver, dict(direction_sign=1), **kwargs)
        assert result['effective_index'] == 2.
        complete = dict(profiles, Ex=torch.zeros_like(electric), Hx=torch.zeros_like(electric))
        power = _discrete_mode_profile_power(complete, coords_u=coordinates,
                                            coords_v=coordinates, normal_axis='x')
        for name, profile in complete.items():
            assert torch.equal(result['component_profiles'][name], profile * power.abs().rsqrt())
        results.append(result)
    if results:
        for name in results[0]['component_profiles']:
            assert torch.equal(results[0]['component_profiles'][name], results[1]['component_profiles'][name])


@pytest.mark.parametrize('length', [1., 1e-6])
@pytest.mark.parametrize('ratio, count', [(1e-16, 1), (1e-6, 2)])
def test_vector_injection_prunes_only_float64_relative_roundoff(monkeypatch, length, ratio, count):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    from witwin.maxwell.fdtd.excitation import injection
    model = _Guide('source', scale=length)
    model.eps.requires_grad_(False)
    scene = model.to_scene()
    scene.grid = mw.GridSpec.uniform(.04*length)
    scene.boundary = mw.BoundarySpec.none()
    solver = mw.Simulation.fdtd(scene, frequency=1e9/length, backend='cpu_reference',
                                run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    source = solver._compiled_sources[0]
    mode = modes.solve_mode_source_profile(solver, source)
    # Both E and H families sit below the former absolute threshold, with a
    # deliberate weak component above float64 roundoff but below float32 eps.
    electric = mode['component_profiles']['Ey'] * 1e-20
    magnetic = electric * 1e-5
    profiles = dict(Ey=electric, Ez=electric*ratio, Hz=magnetic, Hy=magnetic*ratio,
                    Ex=torch.zeros_like(electric), Hx=torch.zeros_like(electric))
    for kind in ('vector_dense', 'vector_dense_torch'):
        resolved = dict(mode, component_profiles=profiles, mode_solver_kind=kind,
                        beta_tensor=torch.tensor(mode['beta'], dtype=torch.float64))
        monkeypatch.setattr(injection, 'solve_mode_source_profile', lambda *args: resolved)
        solver._electric_source_terms.clear()
        solver._magnetic_source_terms.clear()
        injection._prepare_mode_surface_source(solver, dict(source), source_index=0)
        assert len(solver._electric_source_terms) == len(solver._magnetic_source_terms) == count
        for term in (*solver._electric_source_terms, *solver._magnetic_source_terms):
            assert torch.isfinite(term['patch']).all() and term['patch'].abs().max() > 0


@CUDA
def test_control_gradient_bitwise_matches_fixed_plane_path(monkeypatch):
    from tests.gradients.test_mode_source_replay_guard import _Guide, _objective
    model = _Guide('control', device='cuda')
    loss, _ = _objective(model)
    actual, = torch.autograd.grad(loss, model.eps)
    # Recreate the base branch's dispatch: no trainable vector launch path.
    monkeypatch.setattr(modes, '_mode_source_trainable_plane', lambda *args, **kwargs: False)
    baseline = _Guide('control', device='cuda')
    baseline_loss, _ = _objective(baseline)
    expected, = torch.autograd.grad(baseline_loss, baseline.eps)
    assert torch.equal(loss, baseline_loss)
    assert torch.equal(actual, expected)


@CUDA
def test_metre_micrometre_gradient_scaling(monkeypatch):
    from tests.gradients.test_mode_source_replay_guard import _Guide, _objective
    from witwin.maxwell.fdtd.excitation import injection
    kinds = []

    def record_solver_kind(solver, source):
        data = modes.solve_mode_source_profile(solver, source)
        kinds.append(('replay' if getattr(solver, '_mode_source_rebuild_from_fields', False)
                      else 'forward', data['mode_solver_kind']))
        return data

    monkeypatch.setattr(injection, 'solve_mode_source_profile', record_solver_kind)
    gradients, errors = [], []
    for scale in (1., 1e-6):
        kinds.clear()
        model = _Guide('source', device='cuda', scale=scale, samples=2)
        loss, _ = _objective(model)
        gradient, = torch.autograd.grad(loss, model.eps)
        assert set(kinds) == {('forward', 'vector_dense_torch'),
                              ('replay', 'vector_dense_torch')}, f'{scale=}: {kinds=}'
        gradients.append(float(gradient))
        values = {}
        for h in (.05, .025):
            for sign in (-1, 1):
                kinds.clear()
                with torch.no_grad():
                    value, _ = _objective(_Guide('source', 12 + sign*h, device='cuda', scale=scale, samples=2))
                    values[h, sign] = float(value)
                assert kinds and all(kind == 'vector_dense_torch' for _, kind in kinds), (
                    f'{scale=}, {h=}, {sign=}: {kinds=}')
        differences = {h: (values[h, 1] - values[h, -1])/(2*h) for h in (.05, .025)}
        # J = scale^2 |E|^2 + flux is invariant for a unit-power launch.
        # Each central derivative differs from its continuum value by the
        # Richardson estimate plus accumulated FP32 roundoff.
        error = abs(differences[.05] - differences[.025])/3 + 2*torch.finfo(torch.float32).eps*math.sqrt(2000)*max(abs(x) for x in values.values())/.025
        assert abs(float(gradient) - differences[.025]) <= error, (
            f'{scale=}: adjoint={float(gradient)}, FD(h/2)={differences[.025]}, '
            f'bound={error}, {kinds=}')
        errors.append(error)
    assert abs(gradients[0] - gradients[1]) <= sum(errors), (
        f'scales=(1, 1e-6): {gradients=}, {errors=}')


@CUDA
def test_bend_material_region_crosses_launch_plane(monkeypatch):
    import importlib.util
    from pathlib import Path
    import witwin.maxwell as mw
    path = Path(__file__).resolve().parents[2] / 'examples/invdes/waveguide_bend_topology.py'
    spec = importlib.util.spec_from_file_location('vector_replay_bend', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    # Reuse the example with a coarser design and extend its box over x=-1.4 um.
    monkeypatch.setattr(example, 'CELL', .1*example.UM)
    monkeypatch.setattr(example, 'DESIGN_SIZE', 3.*example.UM)
    monkeypatch.setattr(example, 'FILTER_RADIUS', .1*example.UM)
    model = example.BendDesign(device='cuda')
    result = mw.Simulation.fdtd(model, frequencies=[example.FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=3, transient_cycles=5), full_field_dft=False).run()
    loss = result.monitor('output')['flux'].real.sum()
    loss.backward()
    gradient = model.design.params.grad
    assert torch.isfinite(gradient).all()
    # Source plane is one design cell from the box's left face; filtering also
    # spreads the plane cotangent to its immediate neighbours.
    assert gradient[:3].abs().sum() > 0


def test_source_profile_pullback_reaches_material_input_frontier():
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    from witwin.maxwell.fdtd.adjoint.bridge import _validate_mode_source_replay
    from witwin.maxwell.fdtd.adjoint.core import _mode_source_profile_pullback
    model = _Guide('source')
    scene = model.to_scene()
    context = _forward_context(scene)
    _validate_mode_source_replay(context, scene, (model.eps,))
    context._mode_source_rebuild_from_fields = True
    replay = modes.solve_mode_source_profile(context, dict(context._compiled_sources[0]))
    patches = tuple(replay['component_profiles'].values())
    weights = tuple(torch.linspace(.1, 1., x.numel()).reshape(x.shape) for x in patches)
    objective = sum((x*w).sum() for x, w in zip(patches, weights))
    expected, = torch.autograd.grad(objective, model.eps, retain_graph=True)
    context._mode_source_cotangent_accum = {('E_surface', i, 0): w for i, w in enumerate(weights)}
    terms = ((), tuple({'literal_patch': x} for x in patches), ())
    edge_leaves = tuple(torch.zeros(1, requires_grad=True) for _ in range(3))
    assert _mode_source_profile_pullback(context, terms, *edge_leaves) == (None, None, None)
    torch.testing.assert_close(context._source_vector_gradients[0], expected, rtol=0, atol=0)
    assert expected.abs() > 0


@pytest.mark.parametrize('scale', [1., 1e-6])
@pytest.mark.parametrize('launch', ['source', 'port'])
@pytest.mark.parametrize('samples', [1, 2])
def test_no_grad_replay_retains_injection_coefficient_derivative(monkeypatch, scale, launch, samples):
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    from witwin.maxwell.fdtd import cpu_profile
    from witwin.maxwell.fdtd.adjoint import core
    from witwin.maxwell.fdtd.material_pullback import pullback_material_input_gradients

    # Initialize the production launch/replay graphs on the host; no time stepping.
    monkeypatch.setattr(cpu_profile, 'admit_cpu_reference_scene',
                        lambda simulation, prepared: setattr(prepared, '_fdtd_cpu_reference', True))

    def prepare(value):
        model = _Guide('source', value, launch=launch, scale=scale, samples=samples)
        scene = model.to_scene()
        scene.grid = mw.GridSpec.uniform(.04*scale)
        scene.boundary = mw.BoundarySpec.none()
        with torch.no_grad():
            solver = mw.Simulation.fdtd(scene, frequency=1e9/scale, backend='cpu_reference',
                run_time=mw.TimeConfig(time_steps=2))._build_fdtd_solver(initialize=True)
        assert solver._compiled_sources[0]['mode_solver_kind'] == 'vector_dense_torch'
        return model, solver

    model, solver = prepare(12.)
    leaves = tuple(getattr(solver, 'eps_' + field).detach().requires_grad_()
                   for field in ('Ex', 'Ey', 'Ez'))
    # This is the grad mode of a custom Function.backward and checkpoint replay.
    with torch.no_grad():
        terms = core._resolved_source_term_lists(solver, *leaves)
    for forward, replay in zip(solver._electric_source_terms, terms[1]):
        torch.testing.assert_close(forward['patch'], replay['patch'],
                                   rtol=64*torch.finfo(torch.float32).eps, atol=0)
    solver._source_vector_inputs = (model.eps,)
    solver._mode_source_cotangent_accum = {
        ('E_surface', index, 0): 2*term['patch'].detach()*scale**2
        for index, term in enumerate(terms[1])}
    with torch.no_grad():
        edge_gradients = core._mode_source_profile_pullback(solver, terms, *leaves)
    # Missing the 1/eps branch used to leave every edge gradient as None.
    assert any(value is not None and bool(value.abs().max() > 0) for value in edge_gradients)
    coefficient_gradient, = pullback_material_input_gradients(solver.scene, inputs=(model.eps,),
        grad_eps_ex=torch.zeros_like(leaves[0]) if edge_gradients[0] is None else edge_gradients[0],
        grad_eps_ey=edge_gradients[1], grad_eps_ez=edge_gradients[2], eps0=solver.eps0)
    profile_gradient, = solver._source_vector_gradients
    assert coefficient_gradient.abs() > 0 and profile_gradient.abs() > 0
    gradient = coefficient_gradient + profile_gradient

    def objective(value):
        _, perturbed = prepare(value)
        return sum(term['patch'].square().sum() for term in perturbed._electric_source_terms)*scale**2

    values = {h: (objective(12+h), objective(12-h)) for h in (.05, .025)}
    fd = {h: (pos-neg)/(2*h) for h, (pos, neg) in values.items()}
    # Richardson truncation plus the FP32 profile contraction's roundoff.
    entries = sum(term['patch'].numel() for term in terms[1])
    bound = abs(fd[.05] - fd[.025])/3 + 2*torch.finfo(torch.float32).eps*math.sqrt(entries)*max(
        value.abs() for pair in values.values() for value in pair)/.025
    assert abs(gradient - fd[.025]) <= bound, (
        f'{scale=}, {launch=}, {samples=}: profile={profile_gradient}, coefficient={coefficient_gradient}, '
        f'FD={fd[.025]}, {bound=}')


def test_cached_vector_graph_survives_backward_cleanup(monkeypatch):
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    from witwin.maxwell.fdtd.adjoint import bridge as bridge_module
    from witwin.maxwell.fdtd.adjoint.core import _mode_source_profile_pullback

    model = _Guide('source')
    scene = model.to_scene()
    context = _forward_context(scene)
    bridge = bridge_module._FDTDGradientBridge.__new__(bridge_module._FDTDGradientBridge)
    bridge.simulation = SimpleNamespace(scene=scene)
    bridge._mode_replay_cache = None
    bridge._material_graph_scene = lambda: scene
    probe = bridge_module._mode_source_replay_error
    calls = []

    def counted_probe(*args):
        calls.append(1)
        return probe(*args)

    monkeypatch.setattr(bridge_module, '_mode_source_replay_error', counted_probe)
    gradients = []
    for _ in range(2):
        bridge._validate_mode_source_replay(context, (model.eps,))
        context._mode_source_rebuild_from_fields = True
        replay = modes.solve_mode_source_profile(context, dict(context._compiled_sources[0]))
        patches = tuple(replay['component_profiles'].values())
        weights = tuple(torch.linspace(.1, 1., x.numel()).reshape(x.shape) for x in patches)
        context._mode_source_cotangent_accum = {('E_surface', i, 0): w for i, w in enumerate(weights)}
        terms = ((), tuple({'literal_patch': x} for x in patches), ())
        leaves = tuple(torch.zeros(1, requires_grad=True) for _ in range(3))
        _mode_source_profile_pullback(context, terms, *leaves)
        gradients.append(context._source_vector_gradients[0])
        for name in ('_source_vector_materials', '_source_vector_inputs', '_source_vector_gradients'):
            delattr(context, name)
    assert len(calls) == 1
    assert torch.isfinite(gradients[0]) and gradients[0] != 0
    assert torch.equal(*gradients)


@CUDA
def test_above_cap_source_refuses_before_stepping():
    import witwin.maxwell as mw
    from tests.gradients.test_mode_source_replay_guard import _Guide
    model = _Guide('source', device='cuda')
    scene = model.to_scene()
    scene.grid = mw.GridSpec.uniform(.01)
    with pytest.raises(NotImplementedError, match='launch.*plane material is trainable.*cap n <= 4096'):
        mw.Simulation.fdtd(scene, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=1), full_field_dft=False).run()


def test_pulse_delay_cotangent_is_included_in_source_pullback():
    from witwin.maxwell.fdtd.adjoint import core
    parameter = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    patch = (1 + parameter).expand(2, 3)
    delay = (parameter * .05).expand(2, 3)
    source_time = dict(kind='gaussian_pulse', kind_code=1, amplitude=1., frequency=2., fwidth=1., delay=.1, phase=.3)
    term = dict(patch=patch, delay_patch=delay, cw_cos_patch=None, activation_delay_patch=None,
                field_name='Ey', offsets=(0, 0, 0), source_time=source_time)
    adjoint = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    solver = SimpleNamespace(dt=.02, _source_vector_inputs=(parameter,))
    with torch.no_grad():
        core._accumulate_mode_source_cotangents(solver, electric_terms=(term,),
            volume_electric_terms=(), magnetic_terms=(), electric_adjoint={'Ey': adjoint},
            magnetic_adjoint={}, source_time=source_time, omega=4*math.pi, time_value=.15)
    direct = (patch * adjoint * core._evaluate_source_time_tensor(source_time, .16-delay)).sum()
    expected, = torch.autograd.grad(direct, parameter, retain_graph=True)
    leaves = tuple(torch.zeros(1, requires_grad=True) for _ in range(3))
    core._mode_source_profile_pullback(solver, ((), (term,), ()), *leaves)
    torch.testing.assert_close(solver._source_vector_gradients[0], expected, rtol=1e-14, atol=1e-14)


def test_axial_power_correction_preserves_beta_derivative():
    from witwin.maxwell.fdtd.excitation.injection import _mode_normal_stagger_power_factor
    beta = torch.tensor(7., dtype=torch.float64, requires_grad=True)
    solver = SimpleNamespace(scene=SimpleNamespace(x_half=torch.tensor([-.01, .01])),
                             _plane_coordinate=lambda axis, index: 0.)
    source = dict(normal_axis='x', direction_sign=1)
    mode = dict(plane_index=1, mode_solver_kind='vector_dense_torch', beta_tensor=beta)
    correction = _mode_normal_stagger_power_factor(solver, source, mode)
    actual, = torch.autograd.grad(correction, beta)
    offset = float(solver.scene.x_half[0].abs())
    torch.testing.assert_close(actual, -offset*torch.sin(beta*offset))


def test_cpu_metre_micrometre_profiles_and_gradients_scale():
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    from witwin.maxwell.fdtd.adjoint.bridge import _validate_mode_source_replay
    gradients, profiles = [], []
    for scale in (1., 1e-6):
        model = _Guide('source', scale=scale, samples=2)
        scene = model.to_scene()
        context = _forward_context(scene, spacing=.04*scale)
        _validate_mode_source_replay(context, scene, (model.eps,))
        context._mode_source_rebuild_from_fields = True
        data = modes.solve_mode_source_profile(context, dict(context._compiled_sources[0]))
        profile = data['component_profiles']['Ey'] * scale
        profiles.append(profile)
        gradient, = torch.autograd.grad(profile.square().sum(), model.eps)
        gradients.append(gradient)
    tolerance = 8*torch.finfo(torch.float32).eps*math.sqrt(profiles[0].numel())
    torch.testing.assert_close(profiles[0], profiles[1], rtol=tolerance, atol=tolerance*float(profiles[0].detach().abs().max()))
    torch.testing.assert_close(gradients[0], gradients[1], rtol=tolerance, atol=0)


def test_fp64_material_compile_is_preserved_in_replay():
    from tests.gradients.test_mode_source_replay_guard import _Guide, _forward_context
    from witwin.maxwell.fdtd.adjoint.bridge import _validate_mode_source_replay
    model = _Guide('source', scale=1e-6)
    scene = model.to_scene()
    context = _forward_context(scene, spacing=.04e-6, dtype=torch.float64)
    source = context._compiled_sources[0]
    with torch.no_grad():
        forward = modes.solve_mode_source_profile(context, source)
    _validate_mode_source_replay(context, scene, (model.eps,))
    context._mode_source_rebuild_from_fields = True
    replay = modes.solve_mode_source_profile(context, dict(source))
    for name in forward['component_profiles']:
        assert torch.equal(forward['component_profiles'][name], replay['component_profiles'][name])
    gradient, = torch.autograd.grad(replay['profile'].square().sum(), model.eps)
    assert torch.isfinite(gradient) and gradient != 0
