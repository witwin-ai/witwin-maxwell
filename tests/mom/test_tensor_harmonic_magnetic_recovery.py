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


@pytest.mark.parametrize('compressed',[False,True],ids=['dense','compressed'])
def test_mixed_hb_magnetic_volume_recovery_and_solve_free_queries(monkeypatch,tmp_path,compressed):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    linear=TetrahedralVolumeMesh(vertices,cells,region_id=1)
    current=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.2,0.,0.]),cells,region_id=2).refine(1)
    tensor=compile_tensor_harmonic_material(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.),mu_r=1.3),3e8,2,device='cuda')
    scalar=compile_harmonic_material(mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.))),3e8,2)
    volumes=(linear,current)
    materials=(tensor,scalar)
    selected=(current,)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
        dyadic_integration=DyadicIntegrationConfig(),riesz_storage='compact_rwg_v1')
    operator_type=SystemOperator
    options={}
    if compressed:
        from dataclasses import replace
        from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator
        from witwin.maxwell.mom.hmatrix import HMatrixConfig
        from witwin.maxwell.mom.linalg import StoppingRule
        config=replace(config,operator_backend='h2',hierarchical=HMatrixConfig(leaf_size=64),
            riesz_storage='local_rwg_v1',stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
        operator_type=HierarchicalH2SystemOperator
        options=dict(compressed_settings={})
    operators=tuple(operator_type(SurfaceSystem.create([
        (i,f'body{i}',RWGBasis(v.boundary),m.linear_samples[n]) for i,(v,m) in enumerate(zip(volumes,materials))]),
        exterior.linear_samples[n],[mw.PlaneWave()] if n==0 else [],config) for n in range(2))
    print(dict(phase='magnetic_surface_operators_ready',compressed=compressed),flush=True)
    query=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48)
    model,prepared=prepare_mixed_tensor_harmonic_balance(operators,selected,materials,exterior,
        query_config=query,dc_magnetic_volumes=volumes,**options)
    if compressed:
        assert all('electric_response' not in item for item in prepared['harmonics'])
        assert all(item['receipt']['converged'] for item in prepared['harmonics'])
    print(dict(phase='magnetic_hb_prepared',compressed=compressed),flush=True)

    solution=model.solve(relative_tolerance=1e-9,absolute_tolerance=1e-10)
    assert solution.receipt['converged']
    print(dict(phase='magnetic_hb_root_converged',compressed=compressed),flush=True)
    state=recover_harmonic_state(model,prepared,solution)
    magnetic=state.dc_magnetic_volume
    assert magnetic is not None and state.receipt['dc_current']['dc_magnetic_fields_supported']
    torch.testing.assert_close(magnetic.currents[0],torch.zeros_like(magnetic.currents[0]),rtol=0,atol=0)
    torch.testing.assert_close(magnetic.currents[1],state.total_volume_current[0].real,rtol=0,atol=0)
    assert magnetic.currents[1].norm()>1e-12
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    from witwin.maxwell.mom.static_magnetic_volume import MagneticVolumeContrastResponse
    from witwin.maxwell.mom.tensor_harmonic_snapshot import snapshot_mixed_tensor_harmonic_state
    decoder=HarmonicFieldDecoder(prepared,state)
    points=(vertices.new_tensor([[.4,.1,.1]]),linear.vertices[linear.cells].mean(1),current.vertices[current.cells].mean(1))
    expected=[magnetic.fields(p,r)[:2] for r,p in enumerate(points)]
    def forbidden(*args,**kwargs):raise AssertionError('Magnetic query repeated recovery or solving')
    with monkeypatch.context() as patch:
        patch.setattr(MagneticVolumeContrastResponse,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(type(model.dc_current_closure),'solve',forbidden)
        for region,p in enumerate(points):
            for actual,reference in zip(decoder.query_dc_magnetic(p,region),expected[region]):
                torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-28)
            assert not decoder.last_query_receipt['physical_qualification']
    from witwin.maxwell.mom.tensor_harmonic_bundle import snapshot_mixed_tensor_harmonic_bundle,restore_mixed_tensor_harmonic_bundle
    from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
    payload,owner=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
    path=tmp_path/'magnetic-hb.pt';torch.save(payload,path);saved=torch.load(path,weights_only=True)
    with monkeypatch.context() as patch:
        patch.setattr(MagneticVolumeContrastResponse,'__init__',forbidden)
        patch.setattr(MagneticVolumeContrastResponse,'_source',forbidden)
        patch.setattr(StaticTensorGreen,'panel_integrals',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(type(model.dc_current_closure),'solve',forbidden)
        replay_model,replay_prepared,replay_state=restore_mixed_tensor_harmonic_bundle(saved,expected_binding=owner,device='cuda')
    replay=HarmonicFieldDecoder(replay_prepared,replay_state)
    assert replay_prepared['operators'][0].system is not operators[0].system
    assert replay_prepared['static']['magnetic_volume']['bindings'][1].volume is replay_prepared['bindings'][0][0].volume
    for name in ('source_field','matrix','cell_fields','contrast'):
        torch.testing.assert_close(getattr(replay_state.dc_magnetic_volume,name),getattr(magnetic,name),rtol=0,atol=0)
    for region,p in enumerate(points):
        for actual,reference in zip(replay.query_dc_magnetic(p,region),expected[region]):
            torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-28)
        for actual,reference in zip(replay.query_dc(p,region),decoder.query_dc(p,region)):
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    bad=copy.deepcopy(saved);bad['static_maps']['magnetic_volume']['interiors'][0].mul_(2)
    with pytest.raises(ValueError,match='owning binding'):
        restore_mixed_tensor_harmonic_bundle(bad,expected_binding=owner,device='cuda')
    with pytest.raises(ValueError,match='preparation differs'):
        restore_mixed_tensor_harmonic_bundle(bad,expected_binding=dict(version=bad['version'],sha256=_digest(bad)),device='cuda')
    with pytest.raises(ValueError,match='budget'):
        restore_mixed_tensor_harmonic_bundle(saved,expected_binding=owner,device='cuda',max_retained_bytes=1)
    check_magnetic_bundle_rejects_cross_partition(saved)
    saved['state']['magnetic_response']['state']['cell_fields'].add_(1)
    replay.query_dc_magnetic(points[0],0)
    with pytest.raises(ValueError,match='region order'):
        prepare_mixed_tensor_harmonic_balance(operators,selected,materials,exterior,dc_magnetic_volumes=(current,))
    copied=TetrahedralVolumeMesh(current.vertices.clone(),current.cells.clone(),region_id=2)
    with pytest.raises(ValueError,match='exact selected'):
        prepare_mixed_tensor_harmonic_balance(operators,selected,materials,exterior,dc_magnetic_volumes=(linear,copied))
    prepared['static']['magnetic_volume']['interiors'][0].add_(torch.eye(3,device='cuda',dtype=vertices.dtype))
    with pytest.raises((RuntimeError,ValueError),match='changed|mutated|preparation'):
        decoder.query_dc_magnetic(points[0],0)
    magnetic.cell_fields.detach().add_(1)
    with pytest.raises(ValueError,match='changed'):state.assert_current()


def check_magnetic_bundle_rejects_cross_partition(saved):
    from witwin.maxwell.mom.tensor_harmonic_bundle import (restore_mixed_tensor_harmonic_bundle,
        CURRENT_VERSION,COMPRESSED_CURRENT_VERSION,COMPRESSED_MAGNETIC_VERSION)
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    bad=copy.deepcopy(saved)
    nested=bad['state']['magnetic_response']
    nested['currents'][1].mul_(2)
    nested['receipt']['source_current_jump_rms_A_per_m2'][1]*=2
    bad['state']['receipt']['dc_current']['magnetic']=copy.deepcopy(nested['receipt'])
    bad['state']['magnetic_binding']['sha256']=_digest(nested)
    bad['state_binding']['sha256']=_digest(bad['state'])
    with pytest.raises(ValueError,match='magnetic current partition'):
        restore_mixed_tensor_harmonic_bundle(bad,expected_binding=dict(version=bad['version'],sha256=_digest(bad)),device='cuda')
    bad=copy.deepcopy(saved)
    bad['version']=COMPRESSED_CURRENT_VERSION if saved['version']==COMPRESSED_MAGNETIC_VERSION else CURRENT_VERSION
    with pytest.raises(ValueError,match='magnetic bundle inventory'):
        restore_mixed_tensor_harmonic_bundle(bad,expected_binding=dict(version=bad['version'],sha256=_digest(bad)),device='cuda')
