"""Real Scene metadata and independent local algebra; GPU remains gated.

No fake FDTD solver and no CPU field time integration are used. CUDA tests
construct the actual public Scene -> Simulation.prepare() background.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.copper_foil import EPSILON_0, MU_0
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from scripts.dev.fdtd.native_split_copper_plane import (
    NativeSplitPlane, compile_native_plane_layout, compile_outside_dependency_audit,
    require_native_plane_execution, require_native_solve_options,
)
from scripts.dev.fdtd.native_split_copper_validation import build_native_simulation, compile_preflight, main
from scripts.dev.fdtd.split_copper_plane import compiled_local_cn_algebra
from tests.materials.surface_impedance.test_split_copper_local_cn import high_precision_midpoint


def foil():
    return mw.CopperFoil(thickness=math.sqrt(2/(2*math.pi*1e9*MU_0*5.8e7)), frequency_range=(.5e9, 2e9))


def layout_case(polarization='Ez', alpha=.3, complex_fields=True):
    simulation = build_native_simulation(device='cpu', polarization=polarization, complex_fields=complex_fields)
    scene = PreparedScene(simulation.scene, lazy_meshgrid=True)
    layout = compile_native_plane_layout(scene, cut=alpha*.005, polarization=polarization, implicit_bounds=(-.005, .01))
    return simulation, scene, layout


@pytest.mark.parametrize('polarization', ('Ey', 'Ez'))
@pytest.mark.parametrize('alpha', (0., 1e-12, .3, .7, 1-1e-12, 1.))
@pytest.mark.parametrize('complex_fields', (False, True))
def test_actual_scene_grid_source_and_physical_split_ownership(polarization, alpha, complex_fields):
    before = torch.cuda.is_initialized()
    simulation, scene, layout = layout_case(polarization, alpha, complex_fields)
    assert torch.cuda.is_initialized() == before
    assert isinstance(scene, PreparedScene) and isinstance(simulation.scene, mw.Scene)
    assert layout.source_metadata[0]['injection']['mode'] == 'slab'
    assert layout.source_metadata[0]['source_time']['kind'] == 'cw'
    assert len(layout.discarded_native_magnetic) == (0 if layout.geometry.aligned else 1)
    assert layout.dependency_audit['discarded_to_exterior_electric_nonzeros'] == 0
    assert not layout.dependency_audit['provisional_samples_are_physical']
    assert math.fsum(layout.physical_line_area.flatten().tolist()) == pytest.approx(.0004, abs=2e-19)
    assert bool((layout.physical_line_area > 0).all())
    positions = layout.geometry.electric_positions[layout.support['electric']]
    torch.testing.assert_close(positions[layout.local_electric], layout.geometry.original_nodes[layout.native_electric], rtol=0, atol=0)
    if alpha == 1e-12:
        assert float(layout.geometry.magnetic_lengths.min()) == .005e-12


@pytest.mark.parametrize('polarization', ('Ey', 'Ez'))
def test_both_native_magnetic_signs_and_integrated_area_work_are_physical(polarization):
    _, _, layout = layout_case(polarization)
    electric_vector = np.eye(3)[1 if polarization == 'Ey' else 2]
    native_h = np.cross([1., 0., 0.], electric_vector)
    component = 2 if layout.magnetic_name == 'Hz' else 1
    assert native_h[component] == layout.magnetic_sign
    assert np.cross(electric_vector, native_h)[0] == 1
    C = layout.support['local_curl']
    generator = torch.Generator().manual_seed(984)
    e = torch.randn((C.shape[1], layout.physical_line_area.numel()), generator=generator, dtype=torch.float64)
    h = torch.randn((C.shape[0], layout.physical_line_area.numel()), generator=generator, dtype=torch.float64)
    area = layout.physical_line_area.flatten()
    electric_work = (e*(C.T@(h*area))).sum()
    magnetic_work = ((C@e)*h*area).sum()
    torch.testing.assert_close(electric_work, magnetic_work, rtol=2e-15, atol=2e-18)


@pytest.mark.parametrize('side', (0, 1))
def test_discarding_a_real_exterior_h_sample_is_detected_by_the_actual_stencil(side):
    _, _, layout = layout_case()
    with pytest.raises(ValueError, match='real exterior stencil dependency'):
        compile_outside_dependency_audit(layout.geometry.original_nodes, layout.support['bounds'], [layout.outside_magnetic[side]])


@pytest.mark.parametrize('x', (0., .0015, -.005, .01, -.006))
def test_inside_ordinary_query_cannot_publish_discarded_or_one_sided_workspace(x):
    _, _, layout = layout_case()
    with pytest.raises(NotImplementedError, match='side-aware cut records'):
        layout.assert_exterior_query(x)


def test_ordinary_inside_monitor_rejects_before_native_prepare():
    simulation = build_native_simulation(device='cpu')
    simulation.scene.monitors = (mw.PointMonitor('invalid-inside', (.0015, 0., 0.), fields=('Ez',)),)
    with pytest.raises(NotImplementedError, match='side-aware cut records'):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))


def test_real_dielectric_geometry_is_not_silently_removed_to_make_vacuum():
    simulation = build_native_simulation(device='cpu')
    body = mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(.01, .01, .01)), material=mw.Material(eps_r=2.))
    simulation.scene.structures = (body,)
    with pytest.raises(NotImplementedError, match='rejects structures'):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))
    assert simulation.scene.structures == (body,)


def test_actual_feed_is_rejected_instead_of_decoupled():
    simulation = build_native_simulation(device='cpu')
    port = mw.LumpedPort(name='feed', positive=(0., 0., .005), negative=(0., 0., -.005),
        voltage_path=mw.AxisPath('z'), current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)))
    simulation.scene.ports = (port,)
    with pytest.raises(NotImplementedError, match='rejects ports'):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))
    assert simulation.scene.ports == (port,)


def test_tfsf_overlap_rejects_actual_injection_faces():
    simulation = build_native_simulation(device='cpu')
    simulation.scene.sources = (mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.),
        source_time=mw.CW(1e9), injection=mw.TFSF.slab(axis='x', bounds=(-.005, .1))),)
    with pytest.raises(NotImplementedError, match='TFSF correction face'):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))


def test_pml_patch_rejects_actual_compiled_extension_nodes():
    simulation = build_native_simulation(device='cpu')
    scene = PreparedScene(simulation.scene, lazy_meshgrid=True)
    bound = float(scene.x_nodes64[3])
    with pytest.raises(NotImplementedError, match='outside PML'):
        compile_native_plane_layout(scene, cut=.0015, polarization='Ez', implicit_bounds=(bound, .01))


@pytest.mark.parametrize('kind', ('oblique', 'phase', 'soft'))
def test_nonuniform_or_unqualified_sources_fail_before_gpu(kind):
    simulation = build_native_simulation(device='cpu')
    if kind == 'oblique':
        simulation.scene.sources = (mw.PlaneWave(direction=(1., .1, 0.), polarization=(0., 0., 1.),
            source_time=mw.CW(1e9), injection=mw.TFSF.slab(axis='x', bounds=(-.15, .15))),)
    elif kind == 'phase':
        simulation.scene.boundary = mw.BoundarySpec.faces(default='pml', num_layers=12, y='bloch', z='bloch',
            bloch_wavevector=(0., 10., 0.))
    else:
        simulation.scene.sources = (mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.), source_time=mw.CW(1e9)),)
    with pytest.raises(NotImplementedError):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))


@pytest.mark.parametrize('option', ('graph', 'fields', 'shutoff', 'gradient', 'normalization'))
def test_execution_guards_use_real_declarative_simulation(option):
    simulation = build_native_simulation(device='cpu')
    material = foil()
    if option == 'graph':
        simulation.config.cuda_graph = True
    elif option == 'fields':
        simulation.config.full_field_dft = True
    elif option == 'shutoff':
        simulation.config.shutoff = .01
    elif option == 'gradient':
        material = mw.CopperFoil(thickness=torch.tensor(2e-6, requires_grad=True), mode_count=16, frequency_range=(.5e9, 2e9))
    else:
        simulation.config.spectral_sampler = mw.SpectralSampler(normalize_source=True)
    with pytest.raises(NotImplementedError):
        require_native_plane_execution(simulation, material)


@pytest.mark.parametrize('options', ({'stop_step': 1}, {'resume_from': 'prior'}, {'use_cuda_graph': True},
                                    {'full_field_dft': True}, {'normalize_source': True}))
def test_unsupported_real_native_lifecycle_options_reject_before_kernel_execution(options):
    with pytest.raises(NotImplementedError):
        require_native_solve_options(options)


def test_zero_actual_carrier_cannot_produce_a_scattering_qualification():
    simulation = build_native_simulation(device='cpu')
    simulation.scene.sources = (mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.), source_time=mw.CW(1e9, amplitude=0.),
        injection=mw.TFSF.slab(axis='x', bounds=(-.15, .15))),)
    with pytest.raises(ValueError, match='nonzero actual incident carrier'):
        compile_native_plane_layout(simulation.scene, cut=.0015, polarization='Ez', implicit_bounds=(-.005, .01))


def test_adapter_never_accepts_a_fake_or_standalone_solver():
    with pytest.raises(TypeError, match='actual Scene'):
        NativeSplitPlane(object(), cut=.0015, polarization='Ez', material=foil(), implicit_bounds=(-.005, .01))


@pytest.mark.parametrize('polarization', ('Ey', 'Ez'))
@pytest.mark.parametrize('alpha', (1e-12, .3, 1-1e-12))
def test_positive_supplied_field_weights_match_independent_full_local_oracle(polarization, alpha):
    _, _, layout = layout_case(polarization, alpha)
    g, support = layout.geometry, layout.support
    dt = g.bulk_stable_dt(.5)
    # Deliberately distinct positive host coefficients test the mass-input
    # contract. They are not claimed to be native prepared CUDA coefficients.
    eps, mu = EPSILON_0*1.00000007, MU_0*.99999995
    me, mh = eps*g.electric_lengths, mu*g.magnetic_lengths
    p = compiled_local_cn_algebra(g, foil(), dt, support['bounds'], electric_mass=me,
                                 magnetic_mass=mh, impedance=math.sqrt(mu/eps))
    local_me, local_mh = me[support['electric']], mh[support['magnetic']]
    response = compile_copper_response(foil())
    C, T = support['local_curl'], support['trace_selector']
    ne, nh, ns = len(local_me), len(local_mh), len(response.rates)
    Q = torch.diag(torch.cat((local_me, local_mh, torch.ones(ns, dtype=torch.float64))))
    A = torch.zeros_like(Q)
    A[:ne, :ne] = -T.T@response.state_space.D@T
    A[:ne, ne:ne+nh], A[ne:ne+nh, :ne] = C.T, -C
    A[:ne, ne+nh:] = -T.T@response.state_space.B.T
    A[ne+nh:, :ne] = response.state_space.B@T
    A[ne+nh:, ne+nh:] = -torch.diag(response.rates)
    L = Q.diag().sqrt()
    generator = torch.Generator().manual_seed(723)
    old = torch.randn(len(Q), generator=generator, dtype=torch.complex128)/L
    expected = high_precision_midpoint(Q, A, dt, old)
    a = p['material']
    history = a['B'].T.to(old.dtype)@(a['inverse_denominator']*old[ne+nh:])
    rhs = torch.cat((p['electric_diagonal']*old[:ne]-p['eta']*T.T.to(old.dtype)@history,
                     p['magnetic_diagonal']*p['eta']*old[ne:ne+nh]))
    mid = torch.linalg.lu_solve(p['factor'], p['pivots'], rhs[:, None])[:, 0]
    v = T.to(old.dtype)@mid[:ne]
    znew = a['decay']*old[ne+nh:]+a['drive'].to(old.dtype)@v
    actual = torch.cat((2*mid[:ne]-old[:ne], 2*mid[ne:]/p['eta']-old[ne:ne+nh], znew))
    assert float(torch.linalg.vector_norm(L*(actual-expected))/torch.linalg.vector_norm(L*expected)) < 3e-11


def test_preflight_json_cannot_claim_native_gpu_preparation(tmp_path, monkeypatch):
    import sys
    target = tmp_path/'native-preflight.json'
    before = torch.cuda.is_initialized()
    monkeypatch.setattr(sys, 'argv', ['native-split', '--cpu-preflight', '--offset', '1e-12', '--output', str(target)])
    main()
    report = json.loads(target.read_text())
    assert torch.cuda.is_initialized() == before
    assert not report['native_gpu_execution_completed']
    assert report['actual_native_dt_s'] is None
    assert not report['public_surface_mesh_copper_enabled']
    assert report['source_unchanged']
    assert min(report['actual_split_intervals_m']) == .005e-12


def test_missing_gpu_window_fails_with_bound_evidence_without_native_preparation(tmp_path, monkeypatch):
    import sys
    target = tmp_path/'native-not-authorized.json'
    monkeypatch.setenv('WITWIN_RUN_NATIVE_SPLIT_CUDA', '0')
    monkeypatch.setattr(sys, 'argv', ['native-split', '--output', str(target)])
    before = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match='separately scheduled'):
        main()
    report = json.loads(target.read_text())
    assert torch.cuda.is_initialized() == before
    assert report['status'] == 'failed'
    assert not report['native_gpu_execution_completed']
    assert report['source_unchanged']
    assert report['source_sha256_before'] == report['source_sha256_after']


@pytest.mark.skipif(os.environ.get('WITWIN_RUN_NATIVE_SPLIT_CUDA') != '1',
                   reason='requires an explicitly scheduled actual prepared-native CUDA window')
@pytest.mark.parametrize('polarization', ('Ey', 'Ez'))
@pytest.mark.parametrize('alpha', (1e-12, .3, 1-1e-12))
def test_actual_prepared_native_kernel_and_joint_patch_step(polarization, alpha):
    simulation = build_native_simulation(device='cuda', polarization=polarization, complex_fields=True)
    simulation.config.run_time = mw.TimeConfig(time_steps=1, transient_cycles=0, steady_cycles=1)
    simulation.config.spectral_sampler = mw.SpectralSampler(window='none', normalize_source=False)
    prepared = simulation.prepare()
    adapter = NativeSplitPlane(prepared, cut=alpha*.005, polarization=polarization, material=foil(), implicit_bounds=(-.005, .01))
    # Seed a nontrivial transverse-uniform physical local state. The real native
    # kernels run once; the independent oracle conditions on their actual new H.
    generator = torch.Generator().manual_seed(704)
    old_e = torch.randn(len(adapter.electric), generator=generator, dtype=torch.complex128)
    old_h = torch.randn(len(adapter.magnetic), generator=generator, dtype=torch.complex128)/adapter.p['eta']
    old_z = torch.randn(len(adapter.state), generator=generator, dtype=torch.complex128)*1e-9
    adapter.electric.copy_(old_e[:, None].cuda())
    adapter.magnetic.copy_(old_h[:, None].cuda())
    adapter.state.copy_(old_z[:, None].cuda())
    l = adapter.layout
    adapter._scatter(l.electric_name, adapter.maps['native_electric'], adapter.electric[adapter.maps['local_electric']])
    adapter._scatter(l.magnetic_name, adapter.maps['native_magnetic'], l.magnetic_sign*adapter.magnetic[adapter.maps['local_magnetic']])
    with adapter.native_context():
        prepared.solver.solve(time_steps=1, dft_frequency=1e9, full_field_dft=False, use_cuda_graph=False)
    assert adapter.native_block_calls == 1
    assert prepared.solver.fdtd_module is not None
    assert adapter.last_force is not None
    assert float(adapter.last_force.abs().max().cpu()) > 0
    # Independently assemble full local field + all retained ADE states. The
    # nonzero actual native exterior force enters as a constant physical load.
    support, g = l.support, l.geometry
    response = compile_copper_response(foil())
    eps = adapter.weight_audit[l.electric_name]['effective_mass_coefficient_from_actual_gain']
    mu = adapter.weight_audit[l.magnetic_name]['effective_mass_coefficient_from_actual_gain']
    me, mh = eps*g.electric_lengths[support['electric']], mu*g.magnetic_lengths[support['magnetic']]
    ne, nh, ns = len(me), len(mh), len(response.rates)
    C, T = support['local_curl'], support['trace_selector']
    Q = torch.diag(torch.cat((me, mh, torch.ones(ns, dtype=torch.float64))))
    A = torch.zeros_like(Q)
    A[:ne, :ne] = -T.T@response.state_space.D@T
    A[:ne, ne:ne+nh], A[ne:ne+nh, :ne] = C.T, -C
    A[:ne, ne+nh:], A[ne+nh:, :ne] = -T.T@response.state_space.B.T, response.state_space.B@T
    A[ne+nh:, ne+nh:] = -torch.diag(response.rates)
    old = torch.cat((old_e, old_h, old_z))
    force = torch.zeros(len(Q), dtype=old.dtype)
    force[:ne] = adapter.last_force[:, 0].cpu()
    # Exact affine augmentation makes the independent 85-digit solver include
    # the real native forcing without reusing the runtime's Schur elimination.
    Qa = torch.block_diag(Q, torch.ones((1, 1), dtype=torch.float64)).to(old.dtype)
    Aa = torch.zeros_like(Qa); Aa[:-1, :-1] = A; Aa[:-1, -1] = force
    expected = high_precision_midpoint(Qa, Aa, adapter.dt, torch.cat((old, torch.ones(1, dtype=old.dtype))))[:-1]
    actual = torch.cat((adapter.electric[:, 0], adapter.magnetic[:, 0], adapter.state[:, 0])).cpu()
    scale = Q.diag().sqrt()
    assert float(torch.linalg.vector_norm(scale*(actual-expected))/torch.linalg.vector_norm(scale*expected)) < 3e-11
    with pytest.raises(NotImplementedError):
        l.assert_exterior_query(alpha*.005)
