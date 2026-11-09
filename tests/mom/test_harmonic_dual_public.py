"""Public dual-conductive HB queries, live gradients and exact restoration."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def dual_setup():
    original,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    old=config.harmonic_balance.volumes[0]
    vertices=torch.cat((old.vertices,old.vertices.mean(0,keepdim=True)))
    cells=torch.cat((old.boundary.faces[:,[0,2,1]],torch.full((4,1),4,device='cuda',dtype=torch.int64)),dim=1)
    volume=mw.TetrahedralVolumeMesh(vertices,cells,region_id=1)
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,volumes=(volume,)))
    material=mw.Material(eps_r=2.,sigma_e=.001,sigma_m=.002,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    scene=mw.Scene(structures=[mw.Structure(geometry=original.structures[0].geometry,material=material)],
                   sources=[mw.PlaneWave()])
    inside=vertices[cells[0]].mean(0,keepdim=True)
    outside=vertices.new_tensor([[.16,.1,.09]])
    for name,point in (('inside',inside),('outside',outside)):
        scene.add_monitor(mw.PointMonitor(name,tuple(point[0].tolist()),fields=('Ex','Hx','Hy','Hz')))
    return scene,config,inside,outside


def test_dual_public_queries_monitors_gradients_and_restore(tmp_path,monkeypatch):
    scene,config,inside,outside=dual_setup()
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,
        source_max_order=64,source_max_inner_order=96))
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    prepared.solver.model.background.requires_grad_()
    result=prepared.run()
    expected={}
    for region,name,points in ((1,'inside',inside),(0,'outside',outside)):
        fields=result.query_dc_dual(points,region_id=region)
        expected[region]=fields
        assert fields['electric_V_per_m'].requires_grad
        assert fields['magnetic_A_per_m'].requires_grad
        monitor=result.monitor(name)['dc']
        assert monitor['magnetic_available']
        for i,key in enumerate(('Hx','Hy','Hz')):
            torch.testing.assert_close(monitor['components'][key],fields['magnetic_A_per_m'][0,i],rtol=0,atol=0)
    gradient=torch.autograd.grad(expected[0]['electric_V_per_m'].square().sum(),prepared.solver.model.background)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    assert expected[1]['magnetic_flux_density_T'] is None
    torch.testing.assert_close(expected[1]['magnetic_flux_pole_residue_T_per_s'],
                               1j*.002*expected[1]['magnetic_A_per_m'])
    with pytest.raises(NotImplementedError,match='flux pole'):
        result.query_dc_magnetic(inside,region_id=1)
    path=tmp_path/'dual.pt'
    result.save(path)
    settings=torch.load(path,weights_only=True)['harmonic_data']['harmonic_config']
    assert settings['source_max_order']==64 and settings['source_max_inner_order']==96
    def forbidden(*args,**kwargs):
        raise AssertionError('Restoration must not solve')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        restored_settings=restored._harmonic_data.config.harmonic_balance
        assert restored_settings.source_max_order==64 and restored_settings.source_max_inner_order==96
        for region,points in ((0,outside),(1,inside)):
            actual=restored.query_dc_dual(points,region_id=region)
            for key,value in expected[region].items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
    result.harmonic_state.dc_dual_live[0]['electric'].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        result.query_dc_dual(inside,region_id=1)
