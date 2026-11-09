"""Admitted compressed harmonic preparation and recovered physical states."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance
from witwin.maxwell.mom.tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeHConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('prepare,tensor_sources',[(prepare_harmonic_balance,False),(prepare_mixed_tensor_harmonic_balance,False),
    (prepare_mixed_tensor_harmonic_balance,True)],ids=['scalar_dc','tensor_dc_scalar_ac_limit','tensor_susceptibility'])
def test_actual_compressed_material_preparation_root_recovery_and_fields(monkeypatch,tmp_path,prepare,tensor_sources):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    basis=RWGBasis(volume.boundary)
    material=compile_harmonic_material(mw.Material(eps_r=2.,nonlinearity=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)),3e8,1)
    if tensor_sources:
        from dataclasses import replace
        from witwin.maxwell.mom.harmonic_material import compile_tensor_harmonic_material
        axis=vertices.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
        spec=mw.TensorNonlinearSusceptibility(chi2=.002*torch.einsum('i,j,k->ijk',axis,axis,axis),
            chi3=.001*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis))
        sampled=compile_tensor_harmonic_material(mw.Material(eps_r=2.,nonlinearity=spec),3e8,1,
            device='cuda',tensor_sources=True)
        # Keep the independently equivalent scalar AC representation here;
        # the nonlinear law itself has non-diagonal tensor coefficients.
        material=replace(material,chi2=0.,chi3=0.,chi2_tensor=sampled.chi2_tensor,chi3_tensor=sampled.chi3_tensor)
    exterior=compile_harmonic_material(mw.Material(),3e8,1)
    system=SurfaceSystem.create([(0,'body',basis,material.linear_samples[0])])
    integration=IntegrationConfig(order=3,singular_order=3)
    source=(mw.PlaneWave(direction=(.3,.4,.5),polarization=(.8,-.6,0.),
        source_time=mw.CW(frequency=3e8,amplitude=.73,phase=.27))
        if prepare is prepare_mixed_tensor_harmonic_balance else mw.PlaneWave())
    dense=SystemOperator(system,exterior.linear_samples[0],[source],mw.MoMConfig(integration=integration))
    config=mw.MoMConfig(operator_backend='h2',integration=integration,hierarchical=HMatrixConfig(leaf_size=64),
        riesz_storage='local_rwg_v1',stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
    operator=HierarchicalH2SystemOperator(system,exterior.linear_samples[0],dense.sources,config)
    options=dict(source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),
        volume_config=VolumeHConfig(leaf_size=64))
    with pytest.raises(ValueError,match='separate compressed settings'):
        prepare([operator],[volume],[material],exterior,compressed_settings=options,max_order=64)
    model,prepared=prepare([operator],[volume],[material],exterior,compressed_settings=options)
    reference,original=prepare([dense],[volume],[material],exterior)
    if tensor_sources:
        assert set(prepared['nonlinear_coefficients'])=={'chi2','chi3','tpa_scale','chi2_tensor','chi3_tensor'}
    assert prepared['response_backend']=='compressed'
    assert 'electric_response' not in prepared['harmonics'][0]
    assert not hasattr(model,'current_response')
    solution=model.solve(pump_scale=.5);control=reference.solve(pump_scale=.5)
    torch.testing.assert_close(solution.electric,control.electric,rtol=2e-7,atol=1e-9)
    state=recover_harmonic_state(model,prepared,solution)
    expected=recover_harmonic_state(reference,original,control)
    torch.testing.assert_close(state.surface_coefficients[0],expected.surface_coefficients[0],rtol=2e-7,atol=1e-10)
    torch.testing.assert_close(state.dc_surface_charge,expected.dc_surface_charge,rtol=2e-7,atol=1e-10)
    decoder=HarmonicFieldDecoder(prepared,state);baseline=HarmonicFieldDecoder(original,expected)
    points=vertices.new_tensor([[.12,.03,.02]])
    def forbidden(*args,**kwargs):raise AssertionError('Queries must not solve the interface again')
    with monkeypatch.context() as patch:
        patch.setattr(prepared['harmonics'][0]['electric_action'].inverse,'apply',forbidden)
        e,h=decoder.query_fields(points,0,harmonic=1)
    er,hr=baseline.query_fields(points,0,harmonic=1)
    torch.testing.assert_close(e,er,rtol=2e-7,atol=1e-9)
    torch.testing.assert_close(h,hr,rtol=2e-7,atol=1e-11)
    assert not solution.receipt['physical_qualification']
    if prepare is prepare_mixed_tensor_harmonic_balance:
        import copy
        from witwin.maxwell.mom.tensor_harmonic_snapshot import (
            snapshot_mixed_tensor_harmonic_state,restore_mixed_tensor_harmonic_state)
        from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
        from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
        from witwin.maxwell.mom.volume_h_snapshot import _digest
        payload,owner=snapshot_mixed_tensor_harmonic_state(state,prepared)
        path=tmp_path/'mixed-compressed-state.pt'
        torch.save(dict(payload=payload,owner=owner),path)
        saved=torch.load(path,weights_only=True)
        action=prepared['harmonics'][0]['electric_action']
        diagnostics=_digest(action.inverse.receipt)
        probe=state.cell_electric.new_tensor([.3+.2j,.1-.4j,-.2j])
        action.apply(probe);action.adjoint(probe)
        assert _digest(action.inverse.receipt)!=diagnostics
        with monkeypatch.context() as patch:
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(ScalarSystemBlocks,'__call__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=restore_mixed_tensor_harmonic_state(saved['payload'],prepared,expected_binding=saved['owner'])
            for name in ('cell_electric','polarization','total_volume_current'):
                torch.testing.assert_close(getattr(restored,name),getattr(state,name),rtol=0,atol=0)
            assert snapshot_mixed_tensor_harmonic_state(restored,prepared)[1]==owner
            for name in ('surface_coefficients','induced_surface_coefficients'):
                bad=copy.deepcopy(payload);bad['state'][name][0].add_(1e-3)
                with pytest.raises(ValueError,match='surface|tolerance'):
                    restore_mixed_tensor_harmonic_state(bad,prepared,
                        expected_binding=dict(version=owner['version'],sha256=_digest(bad)))
            bad=copy.deepcopy(payload);bad['state']['induced_surface_coefficients']=()
            with pytest.raises(ValueError,match='inventory'):
                restore_mixed_tensor_harmonic_state(bad,prepared,
                    expected_binding=dict(version=owner['version'],sha256=_digest(bad)))
        from witwin.maxwell.mom.tensor_harmonic_bundle import (
            snapshot_mixed_tensor_harmonic_bundle,restore_mixed_tensor_harmonic_bundle)
        bundle,binding=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
        path=tmp_path/'mixed-compressed-bundle.pt'
        torch.save(dict(payload=bundle,owner=binding),path)
        saved=torch.load(path,weights_only=True)
        with monkeypatch.context() as patch:
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(ScalarSystemBlocks,'__call__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            reopened,new_prepared,new_state=restore_mixed_tensor_harmonic_bundle(
                saved['payload'],expected_binding=saved['owner'],device='cuda')
            assert snapshot_mixed_tensor_harmonic_bundle(new_state,new_prepared)[1]==binding
            if tensor_sources:
                for name in ('chi2_tensor','chi3_tensor'):
                    torch.testing.assert_close(new_prepared['nonlinear_coefficients'][name],prepared['nonlinear_coefficients'][name],rtol=0,atol=0)
                    torch.testing.assert_close(getattr(reopened,name),getattr(model,name),rtol=0,atol=0)
            assert new_prepared['operators'][0].incident_carrier.identity==operator.incident_carrier.identity
            torch.testing.assert_close(new_state.cell_electric,state.cell_electric,rtol=0,atol=0)
            for x,y in zip(HarmonicFieldDecoder(new_prepared,new_state).query_fields(points,0,harmonic=1),(e,h)):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            with pytest.raises(ValueError,match='budget'):
                restore_mixed_tensor_harmonic_bundle(bundle,expected_binding=binding,device='cuda',max_retained_bytes=1)
            bad=copy.deepcopy(bundle);bad['operators'][0]['sources'][0]['waveform']['phase']+=.1
            with pytest.raises(ValueError,match='carrier'):
                restore_mixed_tensor_harmonic_bundle(bad,
                    expected_binding=dict(version=binding['version'],sha256=_digest(bad)),device='cuda')
    with torch.no_grad():prepared['harmonics'][0]['electric_action'].indices[0][0]+=1
    with pytest.raises(ValueError,match='changed'):recover_harmonic_state(model,prepared,solution)
