"""Common physical queries and explicitly diagnostic cross-backend evidence."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from benchmark.mom_fdtd import add_query_stencils, make_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_yee_interpolation_reproduces_complex_affine_fields():
    scene = mw.Scene(domain=mw.Domain(bounds=((-1, 1),)*3), grid=mw.GridSpec.uniform(.25), device='cuda')
    prepared = prepare_scene(scene)
    fields = {}
    for c, name in enumerate(('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')):
        axes = [torch.as_tensor(axis, device='cuda', dtype=torch.float64) for axis in component_coords(prepared, name)]
        x, y, z = torch.meshgrid(*axes, indexing='ij')
        fields[name.upper()] = (c+1)*(x+2*y-3*z+2j)
    result = mw.Result(method='fdtd', scene=scene, prepared_scene=prepared, frequency=3e8, fields=fields)
    points = torch.tensor([[.13, -.27, .32], [-.4, .17, .08]], device='cuda', dtype=torch.float64)
    e, h = result.query_fields(points, region_id='exterior', field_kind='total')
    scalar = points[:, 0]+2*points[:, 1]-3*points[:, 2]+2j
    expected = scalar[:, None]*torch.arange(1, 7, device='cuda')
    torch.testing.assert_close(torch.cat((e,h), -1), expected)
    with pytest.raises(ValueError, match='reference'):
        result.query_fields(points, region_id='exterior', field_kind='scattered')
    with pytest.raises(ValueError, match='region'):
        result.query_fields(points, region_id='interior', field_kind='total')


def test_sparse_point_stencils_interpolate_on_actual_component_coordinates():
    points_tuple = ((.13, -.27, .32), (-.4, .17, .08))
    scene = mw.Scene(domain=mw.Domain(bounds=((-1, 1),)*3), grid=mw.GridSpec.uniform(.25), device='cuda')
    add_query_stencils(scene, points_tuple)
    prepared = prepare_scene(scene)
    monitors = {}
    for monitor in scene.monitors:
        component = monitor.fields[0].capitalize()
        axes = component_coords(prepared, component)
        indices = tuple(int(abs(axis-p).argmin()) for axis, p in zip(axes, monitor.position))
        x, y, z = monitor.position
        value = torch.tensor(x+2*y-3*z+2j, dtype=torch.complex128, device='cuda')
        monitors[monitor.name] = {'kind': 'point', 'components': {component: value},
                                  'field_indices': {component: indices}}
    result = mw.Result(method='fdtd', scene=scene, prepared_scene=prepared, frequency=3e8, monitors=monitors)
    points = torch.tensor(points_tuple, device='cuda', dtype=torch.float64)
    e, h = result.query_fields(points, region_id='exterior', field_kind='total')
    expected = (points[:, 0]+2*points[:, 1]-3*points[:, 2]+2j)[:, None].expand(-1, 3)
    torch.testing.assert_close(e, expected)
    torch.testing.assert_close(h, expected)
    result._monitors.clear()
    with pytest.raises(ValueError, match='No spectral'):
        result.query_fields(points, region_id='exterior', field_kind='total')


def test_shared_point_monitors_have_mom_physical_semantics():
    from witwin.maxwell.mom.operator import IntegrationConfig
    scene = mw.Scene(structures=[mw.Structure(geometry=mw.Sphere(radius=.1), material=mw.Material(eps_r=2.25))],
                     sources=[mw.PlaneWave()], monitors=[mw.PointMonitor('probe', (.3, .2, .1), fields=('Ex','Hy')),
                                                       mw.PointMonitor('duplicate', (.3,.2,.1), fields=('Ey',))])
    result = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(
        subdivisions=0, integration=IntegrationConfig(order=3, singular_order=6))).run()
    monitor = result.monitor('probe')
    point = torch.tensor([[.3, .2, .1]], device='cuda', dtype=torch.float64)
    e,h = result.query_fields(point, region_id='exterior', field_kind='total')
    torch.testing.assert_close(monitor['Ex'], e[0,0])
    torch.testing.assert_close(monitor['Hy'], h[0,1])
    assert monitor['field_kind'] == 'total' and monitor['region_id'] == 0
    assert monitor['sampling'] == 'physical_position'
    assert result.stats()['unique_monitor_positions'] == 1
    assert result.stats()['monitor_query_logs'][0]['query_count'] == 1


def test_matched_reference_subtraction_and_configuration_guard():
    scene = make_scene('lossless_sphere', .06)
    reference_scene = make_scene('lossless_sphere', .06, empty=True)
    from witwin.maxwell.result import _validate_fdtd_query_reference
    stats = dict(normalize_source=False, dt=1e-10, time_steps=5, steps_run=5, dft_window='hanning', observer_sample_counts=(5,))
    a = mw.Result(method='fdtd', scene=scene, frequency=3e8, solver_stats=stats)
    b = mw.Result(method='fdtd', scene=reference_scene, frequency=3e8, solver_stats=stats)
    _validate_fdtd_query_reference(a,b)
    b.frequencies = (4e8,)
    with pytest.raises(ValueError, match='frequency'):
        _validate_fdtd_query_reference(a,b)


def test_tfsf_query_never_mislabels_scattered_storage_as_total():
    from witwin.maxwell.result import _interpolate_fdtd_component
    scene = mw.Scene(domain=mw.Domain(bounds=((-1,1),)*3), grid=mw.GridSpec.uniform(.2),
        sources=[mw.PlaneWave(injection=mw.TFSF(bounds=((-0.3,.3),)*3))])
    prepared = prepare_scene(scene)
    shape = tuple(len(axis) for axis in component_coords(prepared,'Ex'))
    result = mw.Result(method='fdtd', scene=scene, prepared_scene=prepared, frequency=3e8,
                       fields={'EX': torch.ones(shape, device='cuda', dtype=torch.complex64)})
    points = torch.tensor([[.25, .0, .0]], device='cuda', dtype=torch.float64)
    with pytest.raises(NotImplementedError, match='TFSF'):
        _interpolate_fdtd_component(result, points, 'Ex')


@pytest.mark.host_validation
def test_reference_rejects_normalization_or_recording_mismatch():
    from witwin.maxwell.result import _validate_fdtd_query_reference
    scene = mw.Scene()
    timing = dict(dt=1e-10, time_steps=5, steps_run=5, dft_window='hanning')
    result = mw.Result(method='fdtd', scene=scene, frequency=3e8,
                       solver_stats={**timing, 'normalize_source': False, 'observer_sample_counts': (5,)})
    for stats in [{'normalize_source': True, 'observer_sample_counts': (5,)},
                  {'normalize_source': False, 'observer_sample_counts': (4,)}]:
        reference = mw.Result(method='fdtd', scene=scene, frequency=3e8, solver_stats={**timing, **stats})
        with pytest.raises(ValueError, match='reference must share'):
            _validate_fdtd_query_reference(result, reference)


def test_closed_sphere_quadrature_cancels_source_free_incident_radiation():
    import math
    from benchmark.mom_fdtd import control_sphere, _weighted_norm
    from witwin.maxwell.mom.physics import compile_medium, incident_fields
    from witwin.maxwell.mom.fields import poynting_flux
    from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_surface_samples
    from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
    from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
    points,normals,weights,solid_angles = control_sphere(order=8)
    assert float(weights.sum()) == pytest.approx(4*math.pi*.2**2)
    assert _weighted_norm(normals[:,2],solid_angles) == pytest.approx(math.sqrt(4*math.pi/3))
    medium = compile_medium(mw.Material(),3e8,background=True)
    e,h = incident_fields(mw.PlaneWave(),points,medium)
    currents = equivalent_surface_currents_from_surface_samples(frequency=3e8,
        points=points,normals=normals,areas=weights,E=e,H=h)
    far = NearFieldFarFieldTransformer(currents,device='cuda', c=C_0, eps0=EPSILON_0, mu0=MU_0).transform_directions(normals)
    assert float(far['E'].abs().max()) < 1e-9
    assert abs(float(poynting_flux(e,h,normals,weights))) < 1e-14


def test_tetrahedral_volume_rule_integrates_quadratic_moments():
    from benchmark.mom_fdtd import tetrahedral_volume_rule
    from witwin.maxwell.mom.mesh import SurfaceMesh
    vertices = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], device='cuda',dtype=torch.float64)
    faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
    points,weights = tetrahedral_volume_rule(SurfaceMesh(vertices,faces),degree=2)
    assert float(weights.sum()) == pytest.approx(1/6)
    assert float((points[:,0]*weights).sum()) == pytest.approx(1/24)
    assert float((points[:,0].square()*weights).sum()) == pytest.approx(1/60)
    assert float((points[:,0]*points[:,1]*weights).sum()) == pytest.approx(1/120)


def test_auxiliary_stencils_do_not_change_the_mom_physical_solution():
    from benchmark.mom_fdtd import physical_scene_without_stencils
    from witwin.maxwell.mom.operator import IntegrationConfig
    scene = mw.Scene(structures=[mw.Structure(geometry=mw.Sphere(radius=.1),material=mw.Material(eps_r=2.25))],
                     sources=[mw.PlaneWave()],monitors=[mw.PointMonitor('query_auxiliary',(.3,.2,.1),fields=('Ex',))])
    physical = physical_scene_without_stencils(scene)
    assert len(scene.monitors) == 1 and not physical.monitors
    assert physical.structures is scene.structures and physical.sources is scene.sources
    config = mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=3,singular_order=6))
    a = mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    b = mw.Simulation.mom(physical,frequency=3e8,config=config).run()
    torch.testing.assert_close(a.surface_currents.scattered_coefficients,b.surface_currents.scattered_coefficients,rtol=1e-12,atol=1e-12)
    points = torch.tensor([[.3,.2,.1]],device='cuda',dtype=torch.float64)
    for first,second in zip(a.query_fields(points,region_id='exterior',field_kind='total'),
                            b.query_fields(points,region_id='exterior',field_kind='total')):
        torch.testing.assert_close(first,second,rtol=1e-12,atol=1e-12)


@pytest.mark.host_validation
def test_resume_rejects_changed_sampling_and_labels_historical_norms():
    from benchmark.mom_fdtd import _resume_cases, _resume_fdtd_sweeps, INTEGRATION
    from dataclasses import asdict
    row = dict(family='lossless_sphere',spacing_m=.04,mom_subdivisions=2,status='measured',
        transient_cycles=16,steady_cycles=16,pml_m=.2,dt_fraction_of_uniform_cfl=.8,
        complex_far_relative_error=.01,mom_integration=asdict(INTEGRATION))
    expected = {('lossless_sphere',.04,2)}
    resumed = _resume_cases({'cases':[row]},expected,16)
    assert resumed[0]['error_norm_version'] == 'legacy_unweighted_sample_L2'
    assert 'error_norm_version' not in row
    with pytest.raises(ValueError,match='Resume cases'):
        _resume_cases({'cases':[row]},expected,8)
    with pytest.raises(ValueError,match='Resume cases'):
        _resume_cases({'cases':[{**row,'dt_fraction_of_uniform_cfl':.4}]},expected,16)
    sweep = {**row,'parameter':'cycles','value':20,'transient_cycles':20,'steady_cycles':20,'frequency_hz':3e8}
    assert len(_resume_fdtd_sweeps({'sensitivity_sweeps':[sweep]},.04,16)) == 1
    with pytest.raises(ValueError,match='fixed settings'):
        _resume_fdtd_sweeps({'sensitivity_sweeps':[{**sweep,'pml_m':.25}]},.04,16)
