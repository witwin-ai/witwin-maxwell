"""Public disjoint-body HB coupling, region ordering and exact detached restore."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('current_dc',[False,True,'mixed'])
def test_disjoint_nonlinear_bodies_reversed_volumes_and_restore(tmp_path,monkeypatch,current_dc):
    _,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    first=config.harmonic_balance.volumes[0]
    second=mw.TetrahedralVolumeMesh(first.vertices+first.vertices.new_tensor([.18,.02,.01]),
                                    first.cells,region_id=2)
    structures=[]
    for index,volume in enumerate((first,second)):
        nonlinear=[mw.NonlinearSusceptibility(chi2=.002*(index+1),chi3=.001)]
        conducting = current_dc is True or (current_dc == 'mixed' and index == 0)
        if conducting:
            nonlinear.append(mw.TwoPhotonAbsorption(beta=5.))
        material=mw.Material(eps_r=2.+index,mu_r=1.+index,
                             sigma_e=.001 if conducting else 0.,nonlinearity=tuple(nonlinear))
        structures.append(mw.Structure(name=f'body{index+1}',material=material,
            geometry=mw.Mesh(volume.boundary.vertices,volume.boundary.faces,recenter=False,fill_mode='solid')))
    scene=mw.Scene(structures=structures,sources=[mw.PlaneWave()])
    if current_dc=='mixed':
        for volume in (first,second):
            scene.add_monitor(mw.PointMonitor(f'inside{volume.region_id}',tuple(volume.vertices.mean(0).tolist()),
                                              fields=('Ex','Hx','Hy','Hz')))
    volumes=(second,first)
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,volumes=volumes))
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    state=result.harmonic_state
    assert not state.receipt['physical_qualification']
    assert len(state.cell_electric[0])==2
    prepared=result._harmonic_data.decoder.prepared
    for response in prepared['harmonics']:
        # The bodies participate in a coupled operator, not independent solves.
        assert response['electric_response'][:3,3:].norm()>0
        assert response['electric_response'][3:,:3].norm()>0
    observations=[]
    for index,volume in enumerate(volumes):
        point=volume.vertices[volume.cells].mean(1)
        dc=result.query_dc(point,region_id=volume.region_id)
        torch.testing.assert_close(dc[1],state.cell_electric[0,index:index+1].real,rtol=1e-8,atol=1e-10)
        ac=result.query_fields(point,region_id=volume.region_id,field_kind='total',harmonic=2)
        torch.testing.assert_close(ac[0],state.cell_electric[2,index:index+1],rtol=3e-4,atol=1e-9)
        magnetic=result.query_dc_magnetic(point,region_id=volume.region_id) if current_dc else None
        observations.append((point,volume.region_id,dc,ac,magnetic))
    if current_dc=='mixed':
        assert state.dc_current_regions==(1,)
        for point,region,dc,ac,magnetic in observations:
            monitor=result.monitor(f'inside{region}')
            assert monitor['dc']['magnetic_available']
            torch.testing.assert_close(monitor['dc']['components']['Ex'],dc[1][0,0],rtol=0,atol=0)
            for i,name in enumerate(('Hx','Hy','Hz')):
                torch.testing.assert_close(monitor['dc']['components'][name],magnetic[0][0,i],rtol=0,atol=0)
    assert state.cell_electric[2].norm()>0
    path=tmp_path/'two-body.pt'
    result.save(path)
    def forbidden(*args,**kwargs):
        raise AssertionError('Detached multi-body restore must not assemble or solve')
    from witwin.maxwell.mom import runtime
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'lu_factor',forbidden)
        patch.setattr(runtime,'assemble_system',forbidden)
        restored=mw.Result.load(path,scene=scene)
        for point,region,dc,ac,magnetic in observations:
            actual_dc=restored.query_dc(point,region_id=region)
            actual_ac=restored.query_fields(point,region_id=region,field_kind='total',harmonic=2)
            for actual,expected in zip((*actual_dc,*actual_ac),(*dc,*ac)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            if magnetic is not None:
                for actual,expected in zip(restored.query_dc_magnetic(point,region_id=region),magnetic):
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if current_dc=='mixed':
        for region in (1,2):
            for name in ('Ex','Hx','Hy','Hz'):
                torch.testing.assert_close(restored.monitor(f'inside{region}')['dc']['components'][name],
                    result.monitor(f'inside{region}')['dc']['components'][name].cpu(),rtol=0,atol=0)
    print({'current_dc':current_dc,'cell_order':[v.region_id for v in volumes],
           'second_harmonic_norm':float(state.cell_electric[2].norm()),'physical_qualification':False})
