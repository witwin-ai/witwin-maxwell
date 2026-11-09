"""Actual AC response matrices coupled to nonlinear face-current DC closure."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material,compile_tensor_harmonic_material
from witwin.maxwell.mom.tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from witwin.maxwell.mom.harmonic_state import preparation_signature,recover_harmonic_state

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('reordered',[False,True])
def test_actual_ac_current_hb_preparation_and_ownership(monkeypatch,tmp_path,reordered,tensor_current=False):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    linear=TetrahedralVolumeMesh(vertices,cells,region_id=1)
    current=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.2,0.,0.]),cells,region_id=2)
    tensor=compile_tensor_harmonic_material(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.),mu_r=1.3),3e8,2,device='cuda')
    scalar=compile_harmonic_material(mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.))),3e8,2)
    if tensor_current:
        # Discrete DC integration control: keep the existing scalar AC maps.
        # This deliberately does not claim a consistent dispersive material.
        from dataclasses import replace
        from witwin.maxwell.mom.static_tensor_material import StaticTensorChannel
        from witwin.maxwell.constants import EPSILON_0
        identity=torch.eye(3,device='cuda',dtype=torch.float64)
        dc=StaticTensorChannel((2*identity).to(torch.complex128),vertices.new_tensor(
            [[.001,.0002,0.],[.0002,.002,.0001],[0.,.0001,.0015]]),torch.zeros_like(identity),EPSILON_0)
        scalar=replace(scalar,electric_dc=dc)
    volumes=(linear,current)
    materials=(tensor,scalar)
    selected=(current,)
    if reordered:
        second=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.4,0.,0.]),cells,region_id=3)
        dielectric=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.6,0.,0.]),cells,region_id=4)
        second_material=compile_harmonic_material(mw.Material(eps_r=3.,sigma_e=.002,nonlinearity=(
            mw.NonlinearSusceptibility(chi2=.001),mw.TwoPhotonAbsorption(beta=3.))),3e8,2)
        dielectric_material=compile_harmonic_material(mw.Material(eps_r=2.,nonlinearity=
            mw.NonlinearSusceptibility(chi2=.003)),3e8,2)
        volumes+=(second,dielectric)
        materials+=(second_material,dielectric_material)
        selected=(dielectric,second,current)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
        dyadic_integration=DyadicIntegrationConfig(),riesz_storage='compact_rwg_v1')
    operators=tuple(SystemOperator(SurfaceSystem.create([
        (i,f'body{i}',RWGBasis(v.boundary),m.linear_samples[n]) for i,(v,m) in enumerate(zip(volumes,materials))]),
        exterior.linear_samples[n],[mw.PlaneWave()] if n==0 else [],config) for n in range(2))
    query=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48)
    model,prepared=prepare_mixed_tensor_harmonic_balance(operators,selected,materials,exterior,query_config=query,tensor_current=tensor_current)
    closure=prepared['static']['tensor_current_closure']
    assert closure.operator.current_regions==((2,3) if reordered else (2,))
    assert all(p['electric_response'].norm()>0 for p in prepared['harmonics'])
    assert preparation_signature(prepared)==prepared['_signature']
    assert not prepared['receipt']['physical_qualification'] and not prepared['receipt']['public_runtime_supported']
    def forbidden(*args,**kwargs):raise AssertionError('HB must reuse retained AC/DC preparation')
    from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        solution=model.solve(relative_tolerance=1e-9,absolute_tolerance=1e-10)
    assert solution.receipt['converged'] and bool(solution.electric[0].norm()>0) and bool(solution.electric[2].norm()>0)
    assert model.residual(solution.electric).norm()<1e-8
    dc=closure.solve(solution.electric[1:],model.tpa_scale)
    torch.testing.assert_close(solution.electric[0,closure.current_indices].real,
                               torch.cat([s.electric for s in dc]),rtol=1e-8,atol=1e-10)
    assert solution.sources.dissipative_current[0].norm()>0
    for n in range(2):
        p=prepared['harmonics'][n]
        j=solution.sources.dissipative_current[n+1]-1j*operators[n].exterior.omega*solution.sources.polarization[n+1]
        expected=p['background_electric']+(p['electric_response']@j.flatten()).reshape(-1,3)
        torch.testing.assert_close(solution.electric[n+1],expected,rtol=1e-8,atol=1e-9)
    state=recover_harmonic_state(model,prepared,solution)
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    from witwin.maxwell.mom.tensor_harmonic_snapshot import snapshot_mixed_tensor_harmonic_state,restore_mixed_tensor_harmonic_state
    payload,owner=snapshot_mixed_tensor_harmonic_state(state,prepared)
    path=tmp_path/'tensor-current.pt';torch.save(payload,path)
    saved=torch.load(path,weights_only=True)
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        patch.setattr(type(closure),'solve',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=restore_mixed_tensor_harmonic_state(saved,prepared,expected_binding=owner)
    for name in ('cell_electric','polarization','dissipative_current','total_volume_current','dc_potential_offsets'):
        torch.testing.assert_close(getattr(restored,name),getattr(state,name),rtol=0,atol=0)
    for actual,expected in zip(restored.dc_face_potentials,state.dc_face_potentials):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    from witwin.maxwell.mom.tensor_harmonic_bundle import snapshot_mixed_tensor_harmonic_bundle,restore_mixed_tensor_harmonic_bundle
    bundle,bundle_owner=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
    bundle_path=tmp_path/'tensor-current-bundle.pt';torch.save(bundle,bundle_path)
    bundle_saved=torch.load(bundle_path,weights_only=True)
    from witwin.maxwell.mom.static_current import FaceCurrentBalance
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'__init__',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'_layers',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        patch.setattr(type(closure),'__init__',forbidden)
        patch.setattr(type(closure),'solve',forbidden)
        patch.setattr(FaceCurrentBalance,'__init__',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored_model,restored_prepared,bundled_state=restore_mixed_tensor_harmonic_bundle(
            bundle_saved,expected_binding=bundle_owner,device='cuda')
    assert restored_prepared['operators'][0].system is not prepared['operators'][0].system
    assert restored_model.dc_current_closure is restored_prepared['static']['tensor_current_closure']
    assert not restored_model.current_response[0].any() and not restored_model.polarization_response[0].any()
    torch.testing.assert_close(restored_model.residual(bundled_state.cell_electric),model.residual(state.cell_electric),rtol=0,atol=0)
    replay=HarmonicFieldDecoder(restored_prepared,bundled_state)
    decoder=HarmonicFieldDecoder(prepared,state)
    assert state.dc_current_regions==((3,2) if reordered else (2,)) and state.dc_surface_charge.numel()==0
    ohmic=torch.zeros_like(state.cell_electric[0]).index_copy(0,closure.current_indices,
        closure.current.ohmic_current(state.cell_electric[0,closure.current_indices].real).to(state.cell_electric.dtype))
    torch.testing.assert_close(state.total_volume_current[0],
        state.dissipative_current[0]+ohmic,rtol=1e-12,atol=1e-12)
    points=(vertices.new_tensor([[.4,.1,.1]]),)+tuple(v.vertices[v.cells].mean(1) for v in volumes)
    from witwin.maxwell.mom.harmonic_current import surface_face_potentials
    direct=closure.operator.solve_uniform(vertices.new_zeros(3),boundary_potentials={
        b.volume.region_id:surface_face_potentials(block,b,potential) for block,b,potential in
        zip(closure.current.blocks,closure.current_bindings,state.dc_face_potentials)},
        bindings=closure.dielectric_bindings,polarizations=tuple(
            state.polarization[0,closure.dielectric_indices].real.split(
                tuple(len(b.volume.cells) for b in closure.dielectric_bindings))))
    torch.testing.assert_close(state.dc_tensor_response.potential,direct.potential,rtol=1e-9,atol=1e-12)
    torch.testing.assert_close(state.dc_tensor_response.conormal,direct.conormal,rtol=1e-9,atol=1e-22)
    for i,region in enumerate(state.dc_current_regions):
        torch.testing.assert_close(state.dc_potential_offsets[i],
            direct.offsets[closure.operator.current_regions.index(region)],rtol=1e-9,atol=1e-12)
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        patch.setattr(type(closure),'solve',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        for region in range(len(points)):
            potential,electric=decoder.query_dc(points[region],region)
            for actual,expected in zip(replay.query_dc(points[region],region),(potential,electric)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert bool(torch.isfinite(potential).all() & torch.isfinite(electric).all())
            if region in state.dc_current_regions:
                index=state.dc_current_regions.index(region)
                torch.testing.assert_close(electric,dc[index].electric,rtol=1e-8,atol=1e-10)
                torch.testing.assert_close(potential,dc[index].potential[closure.current.blocks[index].cell_dofs].mean(-1)
                    +state.dc_potential_offsets[index],rtol=1e-10,atol=1e-12)
            else:
                for actual,expected in zip((potential,electric),direct.fields(points[region],region)):
                    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-12)
            e,h=decoder.query_fields(points[region],region,harmonic=1,query_config=query)
            for actual,expected in zip(replay.query_fields(points[region],region,harmonic=1,query_config=query),(e,h)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert bool(torch.isfinite(e).all() & torch.isfinite(h).all())
            if region==1:
                assert decoder.last_query_receipt['fields']['field_error_metric']=='separate_E_and_vacuum_impedance_H'
            if region in [v.region_id for v in selected]:
                index=[v.region_id for v in selected].index(region)
                torch.testing.assert_close(e,state.cell_electric[1,index:index+1],rtol=3e-4,atol=1e-7)
        with pytest.raises(NotImplementedError,match='magnetic response'):
            decoder.query_dc_magnetic(points[0],0)
        with pytest.raises(ValueError,match='geometry'):
            decoder.query_dc(points[0],2)
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    with pytest.raises(ValueError,match='budget'):
        restore_mixed_tensor_harmonic_bundle(bundle,expected_binding=bundle_owner,device='cuda',max_retained_bytes=1)
    for key in ('conductivities','blocks'):
        bad=copy.deepcopy(bundle)
        value=bad['static_maps']['current_closure']
        if key=='conductivities':value[key][0].add_(.001)
        else:value[key][0]['face_dofs'][0,0]=value[key][0]['face_dofs'][0,1]
        with pytest.raises(ValueError,match='owning binding'):
            restore_mixed_tensor_harmonic_bundle(bad,expected_binding=bundle_owner,device='cuda')
        with pytest.raises(ValueError,match='preparation differs|topology'):
            restore_mixed_tensor_harmonic_bundle(bad,expected_binding=dict(version=bad['version'],sha256=_digest(bad)),device='cuda')
    bundle_saved['static_maps']['current_closure']['conductivities'][0].add_(.001)
    replay.query_dc(vertices.new_tensor([[.4,.1,.1]]),0)
    for key in ('dc_potential_offsets','total_volume_current'):
        bad=copy.deepcopy(saved);bad['state'][key].add_(1.)
        with pytest.raises(ValueError,match='owning binding'):
            restore_mixed_tensor_harmonic_state(bad,prepared,expected_binding=owner)
        with pytest.raises(ValueError,match='partition|offsets'):
            restore_mixed_tensor_harmonic_state(bad,prepared,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    for key,value in (('threshold_A_l2',1.),('converged',False),('failure_reason','failed')):
        bad=copy.deepcopy(saved);bad['receipt']['dc_current']['current'][0][key]=value
        with pytest.raises(ValueError,match='independent current policy'):
            restore_mixed_tensor_harmonic_state(bad,prepared,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    bad=copy.deepcopy(saved);bad['state']['dc_face_potentials'][0].add_(.1)
    with pytest.raises(ValueError,match='gauge'):
        restore_mixed_tensor_harmonic_state(bad,prepared,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    saved['state']['dc_potential_offsets'].add_(1.)
    torch.testing.assert_close(restored.dc_potential_offsets,state.dc_potential_offsets,rtol=0,atol=0)
    state.dc_face_potentials[0].add_(1.)
    with pytest.raises(RuntimeError,match='mutated'):decoder.query_dc(points[2],2)
    with pytest.raises(ValueError,match='budget'):
        prepare_mixed_tensor_harmonic_balance(operators,selected,materials,exterior,
                                             query_config=query,max_static_response_bytes=1,tensor_current=tensor_current)
    closure.current.conductivities[0].add_(.001)
    with pytest.raises(RuntimeError,match='mutated'):preparation_signature(prepared)
    print({'actual_ac_matrices':True,'tensor_sample':'scalar_limit_control','harmonics':2,
           'dc_current_closure':solution.receipt['dc_formulation'],'physical_qualification':False})
