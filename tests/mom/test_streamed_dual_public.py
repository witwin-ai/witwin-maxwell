"""Public nonlinear dual DC policy, state ownership and decode-only archive."""
from dataclasses import replace
import copy
import pytest
import torch
import witwin.maxwell as mw
from test_harmonic_dual_public import dual_setup
from test_dual_action_snapshot import deterministic
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse
from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_streamed_dual_root_and_archive(tmp_path,monkeypatch,deterministic):
    scene,config,inside,outside=dual_setup()
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-22,maxiter=100,restart=24)
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,
        dc_tangent_rule=rule,dc_tangent_column_batch=12,source_max_order=64,source_max_inner_order=96))
    prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    assert prepared.solver.prepared['static']['current_closure'].streamed_rule==rule
    result=prepared.run()
    path=tmp_path/'result.pt';result.save(path)
    original=torch.load(path,weights_only=True)
    assert result.harmonic_state.dc_dual_binding is not None
    expected={region:result.query_dc_dual(point,region_id=region) for region,point in ((1,inside),(0,outside))}
    def forbidden(*args,**kwargs):raise AssertionError('Public dual restore cannot solve or prepare')
    with monkeypatch.context() as patch:
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        patch.setattr(DualHarmonicCurrentClosure,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        for region,point in ((1,inside),(0,outside)):
            fields=restored.query_dc_dual(point,region_id=region)
            for key,value in expected[region].items():
                if isinstance(value,torch.Tensor):torch.testing.assert_close(fields[key],value,rtol=0,atol=0)
        from witwin.maxwell.mom.harmonic_snapshot import snapshot,restore
        from witwin.maxwell.mom.volume_h_snapshot import _digest
        assert _digest(snapshot(restored._harmonic_data))==_digest(original['harmonic_data'])
        for mutation in ('binding','rule','historical_bound','historical_threshold','source'):
            changed=copy.deepcopy(original['harmonic_data'])
            if mutation=='binding':changed['state']['dc_dual_binding']['sha256']='invalid'
            elif mutation=='rule':changed['harmonic_config']['dc_tangent_column_batch']+=1
            elif mutation=='historical_bound':
                changed['state']['receipt']['solver']['dc_current_closure'][0]['tangent_checks'][0]['bound']['columns_checked']-=1
            elif mutation=='historical_threshold':
                changed['state']['receipt']['solver']['dc_current_closure'][0]['threshold_V_per_m_l2']=1.
            else:changed['state']['dc_dual_state']['witness']['electric_source'][0,0]+=1.
            with pytest.raises(ValueError):restore(changed,scene)
