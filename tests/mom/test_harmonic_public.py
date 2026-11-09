"""Shared Scene/Simulation/Result harmonic candidate and explicit scope guards."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup():
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume = mw.TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    geometry = mw.Mesh(volume.boundary.vertices,volume.boundary.faces,recenter=False,fill_mode='solid')
    scene = mw.Scene(structures=[mw.Structure(geometry=geometry,material=mw.Material(eps_r=2.,
        nonlinearity=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)))],sources=[mw.PlaneWave()])
    hb = mw.HarmonicBalanceConfig(harmonics=2,volumes=(volume,),branch_id='linear_start',pump_scales=(.5,1.))
    config = mw.MoMConfig(harmonic_balance=hb,integration=IntegrationConfig(order=3,singular_order=3),
        riesz_storage='compact_rwg_v1',query_config=FieldQueryConfig(
            relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    return scene,config,vertices.mean(0,keepdim=True)


def test_public_harmonic_result_frequency_unknowns_queries_and_continuation(tmp_path,monkeypatch):
    scene,config,points = setup()
    scene.add_monitor(mw.PointMonitor('inside',tuple(points[0].tolist()),fields=('Ex','Ey','Ez','Hx','Hy','Hz')))
    scene.add_monitor(mw.PointMonitor('magnetic_copy',tuple(points[0].tolist()),fields=('Hx',)))
    result = mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    assert isinstance(result,mw.Result)
    assert result.frequencies == (0.,3e8,6e8)
    state = result.harmonic_state
    assert state.receipt['solver']['branch_id'] == 'linear_start'
    assert state.receipt['pump_scale'] == 1.
    assert not state.receipt['physical_qualification']
    assert not state.receipt['harmonic_truncation_converged']
    assert len(state.surface_coefficients) == 2
    for harmonic in (1,2):
        electric,magnetic = result.query_fields(points,region_id=1,field_kind='total',harmonic=harmonic)
        relative = (electric-state.cell_electric[harmonic]).norm()/state.cell_electric[harmonic].norm()
        assert relative < 3e-4
        assert torch.isfinite(magnetic).all()
        monitor = result.monitor('inside',frequency=harmonic*3e8)
        for name,value in zip(('Ex','Ey','Ez','Hx','Hy','Hz'),torch.cat((electric[0],magnetic[0]))):
            torch.testing.assert_close(monitor['components'][name],value,rtol=0,atol=0)
    potential,electric = result.query_dc(points,region_id=1)
    torch.testing.assert_close(electric,state.cell_electric[0].real,rtol=1e-8,atol=1e-10)
    assert not potential.is_complex()
    monitor = result.monitor('inside')
    assert monitor['frequencies'] == (3e8,6e8)
    for name,value in zip(('Ex','Ey','Ez'),electric[0]):
        torch.testing.assert_close(monitor['dc']['components'][name],value,rtol=0,atol=0)
    assert result.monitor('magnetic_copy')['dc']['components'] == {}
    assert result.solver_stats['monitor_queries']['unique_monitor_positions'] == 1
    with pytest.raises(ValueError,match='positive harmonic'):
        result.query_fields(points,region_id=1,field_kind='total')
    path = tmp_path/'hb.pt'
    result.save(path)
    def forbidden(*args,**kwargs):
        raise AssertionError('Restore/query must not assemble or solve')
    from witwin.maxwell.mom import runtime
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(runtime,'assemble_system',forbidden)
        restored = mw.Result.load(path,scene=scene)
        for name in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            torch.testing.assert_close(restored.monitor('inside')['components'][name],
                                       result.monitor('inside')['components'][name].cpu(),rtol=0,atol=0)
        for name in ('frequencies','cell_electric','polarization','dissipative_current',
                     'total_volume_current','dc_surface_charge'):
            torch.testing.assert_close(getattr(restored.harmonic_state,name),getattr(state,name),rtol=0,atol=0)
        for harmonic in (1,2):
            actual = restored.query_fields(points,region_id=1,field_kind='total',harmonic=harmonic)
            expected = result.query_fields(points,region_id=1,field_kind='total',harmonic=harmonic)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(restored.query_dc(points,region_id=1),result.query_dc(points,region_id=1)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    restored.save(tmp_path/'second.pt')
    original = torch.load(path,weights_only=False)
    second = torch.load(tmp_path/'second.pt',weights_only=False)
    assert original['harmonic_data']['scene_signature'] == second['harmonic_data']['scene_signature']
    for mutation,match in (
            (lambda p: p['state']['receipt'].update(physical_qualification=True),'qualification'),
            (lambda p: p['state']['surface_coefficients'][0].add_(1),'surface state')):
        import copy
        damaged = copy.deepcopy(original)
        mutation(damaged['harmonic_data'])
        torch.save(damaged,tmp_path/'damaged.pt')
        with pytest.raises(ValueError,match=match):
            mw.Result.load(tmp_path/'damaged.pt',scene=scene)
    bad = torch.load(path,weights_only=False)
    bad['harmonic_data']['state']['total_volume_current'][1,0,0] += 1
    torch.save(bad,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='material/current'):
        mw.Result.load(tmp_path/'bad.pt',scene=scene)


def test_harmonic_public_configuration_and_overlapping_bodies_reject():
    scene,config,_ = setup()
    for values in ({'harmonics':True},{'branch_id':''},{'pump_scales':(float('nan'),)},
                   {'damping':0.},{'relative_tolerance':0.,'absolute_tolerance':0.}):
        with pytest.raises(ValueError):
            replace(config.harmonic_balance,**values)
    scene.structures.append(scene.structures[0])
    with pytest.raises(ValueError,match='touch|intersect|overlap|disjoint'):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()


def test_public_harmonic_full_field_power_uses_closed_flux_and_volume_queries():
    scene,config,_ = setup()
    result = mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    volume = config.harmonic_balance.volumes[0]
    from witwin.maxwell.mom.mesh import SurfaceMesh
    from witwin.maxwell.mom.quadrature import triangle_rule
    boundary = volume.boundary
    center = boundary.vertices.mean(0)
    control = SurfaceMesh(center+5*(boundary.vertices-center),boundary.faces.clone())
    fixed = FieldQueryConfig(adaptive=False)
    power = result.harmonic_power_balance(control,surface_order=2,volume_order=2,query_config=fixed)
    assert power['harmonics'] == (0,1,2)
    assert not power['global_power_balance_verified']
    assert not power['quadrature_convergence_verified']
    assert torch.count_nonzero(power['linear_absorption_W']) == 0
    assert torch.count_nonzero(power['nonlinear_dissipation_W']) == 0
    points,weights = triangle_rule(control.triangles,2)
    for harmonic in (1,2):
        electric,magnetic = result.query_fields(points.reshape(-1,3),region_id=0,field_kind='total',
                                               harmonic=harmonic,query_config=fixed)
        vector = .5*torch.linalg.cross(electric,magnetic.conj()).real.reshape(*points.shape)
        direct_flux = ((vector*control.normals[:,None]).sum(-1)*weights).sum()
        torch.testing.assert_close(power['outward_flux_W'][harmonic],direct_flux,rtol=1e-10,atol=1e-18)
    with pytest.raises(ValueError,match='strictly enclose'):
        result.harmonic_power_balance(boundary,surface_order=2,volume_order=2,query_config=fixed)
    print({'outward_flux_W':power['outward_flux_W'].tolist(),
           'polarization_work_W':power['polarization_work_W'].tolist(),
           'balance_residual_W':power['balance_residual_W'].tolist(),
           'quadrature_convergence_verified':False,'physical_qualification':False})


def test_harmonic_unsupported_linear_method_and_zero_pump_fail_before_assembly(monkeypatch):
    scene,config,_ = setup()
    from witwin.maxwell.mom import harmonic_runtime
    from witwin.maxwell.mom.linalg import StoppingRule
    def forbidden(*args,**kwargs):
        raise AssertionError('Unsupported settings must fail before AC preparation')
    monkeypatch.setattr(harmonic_runtime,'SystemOperator',forbidden)
    with pytest.raises(NotImplementedError,match='direct linear solves'):
        mw.Simulation.mom(scene,frequency=3e8,config=replace(config,
            stopping=StoppingRule(method='gmres'))).prepare()
    scene.structures[0]=replace(scene.structures[0],material=mw.Material(eps_r=2.,
        nonlinearity=mw.TwoPhotonAbsorption(beta=5.)))
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,pump_scales=(0.,1.)))
    with pytest.raises(NotImplementedError,match='zero-pump continuation'):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
