"""Mixed tensor cutoff extension reuses real AC/DC maps without qualification."""
from dataclasses import replace
import copy
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup
from witwin.maxwell.mom.harmonic_runtime import HarmonicCutoffError
from witwin.maxwell.mom.harmonic_state import preparation_signature

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('backend',['dense','h2'])
@pytest.mark.parametrize('magnetic', [False,True],ids=['linear_spd_success','nonlinear_current_magnetic_budget'])
def test_mixed_cutoff_prefix_reuse_persistence_and_budget(tmp_path,monkeypatch,magnetic,backend):
    scene,config,point=setup(current=magnetic,magnetic=magnetic)
    if backend=='h2':
        from witwin.maxwell.mom.hmatrix import HMatrixConfig
        from witwin.maxwell.mom.linalg import StoppingRule
        config=replace(config,operator_backend='h2',riesz_storage='local_rwg_v1',
            hierarchical=HMatrixConfig(leaf_size=64),stopping=StoppingRule(method='block_gmres',
                preconditioner='row_column_norm',rtol=1e-11,atol=1e-20,stagnation_window=48))
    if not magnetic:
        scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.))
    policy=mw.HarmonicCutoffConfig(max_harmonics=2 if magnetic else 3,
        failure='record' if magnetic else 'raise')
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,
        harmonics=1,pump_scales=(1.,),cutoff=policy))
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    original=prepared.solver.prepared
    import witwin.maxwell.mom.tensor_harmonic_preparation as preparation
    response_name='_prepare_compressed_harmonic' if backend=='h2' else 'prepare_cell_field_response'
    import witwin.maxwell.mom.harmonic_preparation as scalar_preparation
    response_module=scalar_preparation if backend=='h2' else preparation
    response=getattr(response_module,response_name);calls=[]
    def appended(operator,*args,**kwargs):
        calls.append(operator.exterior.frequency)
        return response(operator,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Cutoff must reuse existing DC preparation')
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
    import witwin.maxwell.mom.harmonic_convergence as convergence
    comparison=convergence.compare_harmonic_cells;compared=[]
    def record_comparison(previous,current,**kwargs):
        compared.append((previous,current))
        return comparison(previous,current,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(convergence,'compare_harmonic_cells',record_comparison)
        patch.setattr(response_module,response_name,appended)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'__init__',forbidden)
        result=prepared.run()
    native=result._harmonic_data;extended=native.decoder.prepared
    assert calls==([6e8] if magnetic else [6e8,9e8])
    assert extended['static'] is original['static']
    assert extended['harmonics'][0] is original['harmonics'][0]
    assert extended['bindings'][0] is original['bindings'][0]
    assert prepared.solver.prepared is original
    assert preparation_signature(original)==original['_signature']
    assert 'cutoff' not in original['receipt']
    cutoff=result.solver_stats['cutoff']
    assert cutoff['cell_truncation_estimate_passed'] is (not magnetic)
    assert cutoff['final_harmonics']==policy.max_harmonics
    assert cutoff['physical_qualification'] is False and cutoff['harmonic_truncation_converged'] is False
    assert result._metadata['cell_cutoff_estimate_passed'] is (not magnetic)
    assert result.frequencies==tuple(i*3e8 for i in range(policy.max_harmonics+1))
    if magnetic:
        assert cutoff['failure_reason']=='harmonic_budget_exhausted'
        assert result.harmonic_state.dc_magnetic_volume is not None
        assert bool(result.harmonic_state.cell_electric[2].norm()>0)
    else:assert torch.count_nonzero(result.harmonic_state.cell_electric[2:])==0
    dc=result.query_dc(point,region_id=2)
    ac=result.query_fields(point,region_id=2,field_kind='total',harmonic=policy.max_harmonics)
    path=tmp_path/'cutoff.pt';result.save(path)
    from witwin.maxwell.mom.runtime import SystemOperator
    from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'__init__',forbidden)
        patch.setattr(DenseHarmonicBalance,'solve',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        if backend=='h2':
            from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
            from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(GeneralizedSystemBlocks,'__call__',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.solver_stats['cutoff']==cutoff
        assert restored._harmonic_data.config.harmonic_balance.cutoff==policy
        assert restored.frequencies==result.frequencies
        replay_comparison=comparison(compared[0][0],restored._harmonic_data)
        assert replay_comparison['current_max_harmonic']==policy.max_harmonics
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(restored.harmonic_state,name),getattr(result.harmonic_state,name),rtol=0,atol=0)
        for a,b in zip(restored.query_dc(point,region_id=2),dc):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(restored.query_fields(point,region_id=2,field_kind='total',harmonic=policy.max_harmonics),ac):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Export this already solved state for cross-repository archive validation.
    import os
    from pathlib import Path
    if os.environ.get('MIXED_CUTOFF_ARTIFACT_ROOT'):
        export=Path(os.environ['MIXED_CUTOFF_ARTIFACT_ROOT']);export.mkdir(exist_ok=True)
        filename='magnetic.pt' if magnetic else 'spd.pt'
        result.save(export/(filename if backend=='dense' else backend+'-'+filename))
    payload=torch.load(path,weights_only=False)
    for target in ('metadata','solver_stats'):
        broken=copy.deepcopy(payload)
        if target=='metadata':broken[target]['cell_cutoff_estimate_passed']=magnetic
        else:broken[target]['cutoff']['cell_truncation_estimate_passed']=magnetic
        damaged=tmp_path/(target+'.pt');torch.save(broken,damaged)
        with pytest.raises(ValueError,match='metadata differs|statistics differ'):mw.Result.load(damaged,scene=scene)
    before=copy.deepcopy(extended['receipt'])
    with pytest.raises(HarmonicCutoffError):prepared.solver._extend_cutoff(native,replace(policy,failure='raise'))
    again,failed=prepared.solver._extend_cutoff(native,replace(policy,failure='record'))
    assert failed['checks']==[] and not failed['cell_truncation_estimate_passed']
    assert extended['receipt']==before and again.decoder.prepared is not extended
    # Reuse rejects changed physical inputs before reaching the appended operator.
    from witwin.maxwell.mom.tensor_harmonic_runtime import _materials
    from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
    materials=list(_materials(scene,3e8,2,config.dtype))
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    opts=dict(reuse_prepared=original,dc_policy=config.harmonic_balance.dc_policy() if magnetic else None,
              dc_magnetic_volumes=config.harmonic_balance.dc_magnetic_volumes)
    from witwin.maxwell.mom.harmonic_runtime import _response_settings
    opts.update(_response_settings(config))
    for policy_change in ({'static_pair_batch':32},{'max_static_matrix_bytes':1},{'max_static_response_bytes':1}):
        with pytest.raises(ValueError,match='static preparation policy'):
            preparation.prepare_mixed_tensor_harmonic_balance((*original['operators'],object()),config.harmonic_balance.volumes,
                materials,exterior,**opts,**policy_change)
    changed=list(materials);changed[1]=replace(changed[1],chi2=changed[1].chi2+.1)
    with pytest.raises(ValueError,match='nonlinear coefficients'):
        preparation.prepare_mixed_tensor_harmonic_balance((*original['operators'],object()),config.harmonic_balance.volumes,
            changed,exterior,**opts)
    changed=list(materials);channel=changed[0].electric_dc
    changed[0]=replace(changed[0],electric_dc=type(channel)(channel.finite_relative_part*1.1,channel.conductivity,channel.plasma_angular_frequency_squared,channel.absolute_scale))
    with pytest.raises(ValueError,match='DC electric samples'):
        preparation.prepare_mixed_tensor_harmonic_balance((*original['operators'],object()),config.harmonic_balance.volumes,
            changed,exterior,**opts)
