"""Public experimental current HB with DC magnetic monitors and persistence."""
import runpy
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_current_material_dc_monitors_fields_and_result_restore(tmp_path,monkeypatch):
    original,config,_ = runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    material = mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    scene = mw.Scene(structures=[mw.Structure(geometry=original.structures[0].geometry,material=material)],
                     sources=[mw.PlaneWave()])
    scene.add_monitor(mw.PointMonitor('outside',(.16,.1,.09),fields=('Ex','Hx','Hy','Hz')))
    result = mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    runpy.run_path('tests/mom/test_harmonic_current_krylov.py')['check_saved_current_policy_codec'](
        result,scene,tmp_path,monkeypatch)
    assert not result.harmonic_state.receipt['physical_qualification']
    assert result.harmonic_state.receipt['dc_current']['dc_magnetic_fields_supported']
    points = config.harmonic_balance.volumes[0].vertices.new_tensor([[.16,.1,.09]])
    expected = result.query_dc_magnetic(points,region_id=0)
    monitor = result.monitor('outside')
    assert monitor['dc']['magnetic_available']
    for index,name in enumerate(('Hx','Hy','Hz')):
        torch.testing.assert_close(monitor['dc']['components'][name],expected[0][0,index],rtol=0,atol=0)
    path = tmp_path/'current-result.pt'
    result.save(path)
    replacement = mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    assert replacement.material_id != material.material_id
    scene = mw.Scene(structures=[mw.Structure(geometry=original.structures[0].geometry,material=replacement)],
                     sources=[mw.PlaneWave()])
    scene.add_monitor(mw.PointMonitor('outside',(.16,.1,.09),fields=('Ex','Hx','Hy','Hz')))
    from witwin.maxwell.mom.static_magnetic import MagnetostaticInterfaceResponse
    def forbidden(*args,**kwargs):
        raise AssertionError('Public restore and query must not solve an interface')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(MagnetostaticInterfaceResponse,'__init__',forbidden)
        restored = mw.Result.load(path,scene=scene)
        for actual,value in zip(restored.query_dc_magnetic(points,region_id=0),expected):
            torch.testing.assert_close(actual,value,rtol=0,atol=0)
    for name in ('Hx','Hy','Hz'):
        torch.testing.assert_close(restored.monitor('outside')['dc']['components'][name],
                                   monitor['dc']['components'][name].cpu(),rtol=0,atol=0)
    changed_material = mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.003,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    scene = mw.Scene(structures=[mw.Structure(geometry=original.structures[0].geometry,material=changed_material)],
                     sources=[mw.PlaneWave()])
    scene.add_monitor(mw.PointMonitor('outside',(.16,.1,.09),fields=('Ex','Hx','Hy','Hz')))
    with pytest.raises(ValueError,match='does not match the supplied Scene'):
        mw.Result.load(path,scene=scene)
