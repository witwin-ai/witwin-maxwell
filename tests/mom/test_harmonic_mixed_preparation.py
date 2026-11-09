"""Mixed DC preparation with actual heterogeneous SIE/VIE AC response maps."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance
from witwin.maxwell.mom.harmonic_current import surface_face_potentials
from witwin.maxwell.mom.harmonic_state import preparation_signature,recover_harmonic_state
from witwin.maxwell.mom.static_mixed import MixedStaticOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_mixed_material_harmonic_preparation_and_dc_coupling(monkeypatch,tmp_path):
    _,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    first=config.harmonic_balance.volumes[0]
    second=mw.TetrahedralVolumeMesh(first.vertices+first.vertices.new_tensor([.18,.02,.01]),first.cells,region_id=2)
    volumes=(second,first)
    physical=(mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.))),
        mw.Material(eps_r=3.,mu_r=2.,nonlinearity=mw.NonlinearSusceptibility(chi2=.004,chi3=.001)))
    materials=tuple(compile_harmonic_material(value,3e8,2) for value in physical)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    bases=tuple(RWGBasis(v.boundary) for v in (first,second))
    operators=tuple(SystemOperator(SurfaceSystem.create([
        (index,f'body{index}',basis,material.linear_samples[n])
        for index,(basis,material) in enumerate(zip(bases,materials))]),exterior.linear_samples[n],
        [mw.PlaneWave()] if n==0 else [],replace(config,harmonic_balance=None)) for n in range(2))
    model,prepared=prepare_harmonic_balance(operators,volumes,materials,exterior,query_config=config.query_config)
    assert prepared['receipt']['formulation']=='coupled_sie_vie_hb_mixed_candidate_v1'
    assert prepared['receipt']['dc_magnetic_fields_supported']
    assert prepared['receipt']['mixed_dc_state_recovery_supported']
    closure=model.dc_current_closure
    assert closure.maps['source_regions']==(2,)
    assert closure.operator.current_regions==(1,)
    def forbidden(*args,**kwargs):
        raise AssertionError('Prepared mixed solve must not rebuild its static response')
    with monkeypatch.context() as patch:
        patch.setattr(MixedStaticOperator,'solve',forbidden)
        solved=model.solve_ramp((.5,1.),branch_id='mixed_actual')[-1]
    assert solved.receipt['converged']
    assert solved.electric[0].norm()>0 and solved.electric[2].norm()>0
    assert model.polarization_response[1,:3,3:].norm()>0
    assert model.polarization_response[1,3:,:3].norm()>0
    current=closure.solve(solved.electric[1:],model.tpa_scale)[0]
    boundary=surface_face_potentials(closure.current.blocks[0],closure.current_bindings[0],current.potential)
    direct=closure.operator.solve(closure.dielectric_bindings,[solved.sources.polarization[0,:1].real],[boundary])
    point=second.vertices[second.cells].mean(1)
    torch.testing.assert_close(solved.electric[0,:1].real,direct.fields(point,2)[1],rtol=1e-7,atol=1e-10)
    total_current=physical[0].sigma_e*solved.electric[0,1:].real+solved.sources.dissipative_current[0,1:].real
    assert closure.current.blocks[0].weak_current(total_current).norm()<1e-14
    state=recover_harmonic_state(model,prepared,solved)
    assert state.dc_current_regions==(1,)
    assert len(state.dc_face_potentials)==1 and state.dc_potential_offsets.shape==(1,)
    assert state.receipt['formulation']=='coupled_sie_vie_hb_mixed_face_state_v1'
    torch.testing.assert_close(state.dc_surface_charge,direct.density,rtol=1e-10,atol=1e-14)
    torch.testing.assert_close(state.total_volume_current[0,1:].real,total_current,rtol=0,atol=0)
    assert torch.count_nonzero(state.total_volume_current[0,:1])==0
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    from witwin.maxwell.mom.static_magnetic import MagnetostaticInterfaceResponse
    decoder=HarmonicFieldDecoder(prepared,state)
    outside=first.vertices.new_tensor([[-.1,.15,.12]])
    expected_outside=direct.fields(outside,0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(MixedStaticOperator,'solve',forbidden)
        patch.setattr(type(closure),'solve',forbidden)
        patch.setattr(MagnetostaticInterfaceResponse,'__init__',forbidden)
        for index,volume in enumerate(volumes):
            cell_point=volume.vertices[volume.cells].mean(1)
            _,electric=decoder.query_dc(cell_point,volume.region_id)
            torch.testing.assert_close(electric,state.cell_electric[0,index:index+1].real,rtol=1e-7,atol=1e-10)
            magnetic,flux=decoder.query_dc_magnetic(cell_point,volume.region_id)
            assert torch.isfinite(magnetic).all() and torch.isfinite(flux).all()
            ac,_=decoder.query_fields(cell_point,volume.region_id,harmonic=2)
            torch.testing.assert_close(ac,state.cell_electric[2,index:index+1],rtol=3e-4,atol=1e-9)
        for actual,expected in zip(decoder.query_dc(outside,0),expected_outside):
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-14)
        decoder.query_dc_magnetic(outside,0)
    from witwin.maxwell.mom.harmonic_runtime import HarmonicData
    from witwin.maxwell.mom.harmonic_snapshot import snapshot,restore,tensors
    data=HarmonicData(prepared,state,config,())
    from witwin.maxwell.mom.harmonic_observables import harmonic_power_balance
    from witwin.maxwell.mom.mesh import SurfaceMesh
    from witwin.maxwell.mom.fields import FieldQueryConfig
    center=torch.cat((first.vertices,second.vertices)).mean(0)
    control=SurfaceMesh(center+20*(first.boundary.vertices-first.boundary.vertices.mean(0)),first.boundary.faces)
    cache={}
    fixed=FieldQueryConfig(adaptive=False)
    power=harmonic_power_balance(data,control,surface_order=2,volume_order=2,query_config=fixed,_cache=cache)
    assert (0,1) in power['per_region'] and (0,2) in power['per_region']
    assert power['per_region'][(0,2)]['linear_absorption_W']==0
    assert power['per_region'][(0,2)]['dissipative_work_W']==0
    expected_work=physical[0].sigma_e*state.cell_electric[0,1].real.square().sum()*first.volumes.sum()
    torch.testing.assert_close(power['per_region'][(0,1)]['linear_absorption_W'],expected_work,rtol=1e-7,atol=1e-25)
    assert not power['global_power_balance_verified']
    with monkeypatch.context() as patch:
        for name in ('query_dc','query_dc_magnetic','query_fields'):
            patch.setattr(data,name,forbidden)
        repeated=harmonic_power_balance(data,control,surface_order=2,volume_order=2,query_config=fixed,_cache=cache)
        torch.testing.assert_close(repeated['balance_residual_W'],power['balance_residual_W'],rtol=0,atol=0)
    print({'mixed_power_residual_W':power['balance_residual_W'].tolist(),'physical_qualification':False})
    structures=[mw.Structure(name=f'body{index}',geometry=mw.Mesh(v.boundary.vertices,v.boundary.faces,recenter=False,fill_mode='solid'),material=m)
                for index,(v,m) in enumerate(zip((first,second),physical))]
    scene=mw.Scene(structures=structures,sources=[mw.PlaneWave()])
    public_config=replace(config,harmonic_balance=replace(config.harmonic_balance,volumes=volumes,branch_id='mixed_actual'))
    from witwin.maxwell.mom.runtime import _signature
    from witwin.maxwell.mom.harmonic_material import material_signature
    signature=(_signature(scene,exterior.linear_samples[0],tuple(m.linear_samples[0] for m in materials)),
               tuple(material_signature(s.material) for s in scene.structures),3e8)
    data=HarmonicData(prepared,state,public_config,signature)
    payload=snapshot(data)
    assert payload['schema_version']==6 and payload['payload_kind']=='coupled_sie_vie_hb_mixed'
    path=tmp_path/'mixed-snapshot.pt'
    torch.save(payload,path)
    expected_dc=data.query_dc(outside,0)
    expected_h=data.query_dc_magnetic(outside,0)
    expected_ac=data.query_fields(outside,0,'total',harmonic=2)
    with monkeypatch.context() as patch:
        for name in ('solve','lu_factor','lu_solve'):
            patch.setattr(torch.linalg,name,forbidden)
        patch.setattr(MixedStaticOperator,'__init__',forbidden)
        restored=restore(torch.load(path,weights_only=False),scene)
        for actual,expected in zip((*restored.query_dc(outside,0),*restored.query_dc_magnetic(outside,0),
                                   *restored.query_fields(outside,0,'total',harmonic=2)),
                                  (*expected_dc,*expected_h,*expected_ac)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for volume in volumes:
            point=volume.vertices[volume.cells].mean(1)
            for actual,expected in zip(restored.query_dc(point,volume.region_id),data.query_dc(point,volume.region_id)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for kind in ('region','offset','points','weights','threshold','dc_threshold','dc_policy','map_null_direction'):
            bad=tensors(payload,'cpu')
            if kind=='region':
                bad['state']['dc_current_regions']=(2,)
            elif kind=='offset':
                bad['state']['dc_potential_offsets'][0]+=.001
            elif kind=='points':
                bad['prepared']['harmonics'][0]['points']+=1
            elif kind=='weights':
                bad['prepared']['harmonics'][0]['cell_volumes']*=2
            elif kind=='threshold':
                bad['prepared']['harmonics'][0]['background_electric']+=1
                bad['state']['receipt']['solver']['threshold_V_per_m_l2']=1e30
            elif kind=='dc_threshold':
                bad['state']['receipt']['dc_current']['current'][0]['threshold_A_l2']=1e30
            elif kind=='dc_policy':
                bad['prepared']['static']['current_closure']['absolute_tolerance']=1e30
                bad['state']['receipt']['dc_current']['current'][0]['threshold_A_l2']=1e30
            else:
                packed=torch.cat((state.polarization[0,:1].real.flatten(),boundary)).detach().cpu()
                direction=torch.ones_like(packed)
                direction-=packed*(direction@packed)/(packed@packed)
                direction/=direction.norm()
                bad['prepared']['static']['mixed_closure']['maps']['electric_response'][0]+=direction
            with pytest.raises(ValueError,match='current region order|layer or offset|volume geometry|threshold|tolerance|operator or mapping'):

                restore(bad,scene)
    receipt={'residual':solved.receipt['final_residual'],'dc_norm':float(solved.electric[0].norm()),
             'second_harmonic_norm':float(solved.electric[2].norm()),'physical_qualification':False}
    torch.save({'electric':solved.electric.detach(),'receipt':receipt},tmp_path/'mixed-algebra.pt')
    print(receipt)
    closure.operator.lu.add_(1.)
    assert preparation_signature(prepared)!=prepared['_signature']
    with pytest.raises(RuntimeError,match='operator was mutated'):
        model.residual(solved.electric)
