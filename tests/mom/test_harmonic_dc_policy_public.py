"""Public DC tangent selection, exact policy binding and saved-record rejection."""
import copy
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.static_current import NodalCurrentBalance
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('kind',['current','mixed'])
def test_public_current_dc_policy_restore(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        scene,config=runpy.run_path('tests/mom/test_compressed_hb_public.py')['setup'](kind,'hmatrix')
        config=replace(config,operator_backend='dense',hierarchical=None,riesz_storage='compact_rwg_v1',
            stopping=StoppingRule(method='direct'))
        rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-22,maxiter=100,restart=20)
        hb=replace(config.harmonic_balance,dc_tangent_rule=rule,dc_tangent_column_batch=3)
        config=replace(config,harmonic_balance=hb)
        def forbidden(*args,**kwargs):raise AssertionError('DC dense assembly or restore solve forbidden')
        monkeypatch.setattr(NodalCurrentBalance,'_matrix',forbidden)
        result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
        path=tmp_path/'result.pt';result.save(path)
        raw=torch.load(path,weights_only=True)
        assert raw['harmonic_data']['harmonic_config']==hb.numerical_payload()
        assert result.harmonic_state.receipt['dc_current']['current'][0]['linear_solver']=='streamed_inverse_checked_gmres'
        point=hb.volumes[0].vertices.new_tensor([[.16,.1,.09]])
        expected=result.query_dc(point,region_id=0)
        with monkeypatch.context() as patch:
            patch.setattr(NodalCurrentBalance,'solve',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(path,scene=scene)
            for a,b in zip(restored.query_dc(point,region_id=0),expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            restored.save(tmp_path/'second.pt')
            assert _digest(torch.load(tmp_path/'second.pt',weights_only=True)['harmonic_data'])==_digest(raw['harmonic_data'])
            for mutation in (
                lambda p:p['harmonic_config']['dc_tangent_rule'].update(rtol=1e-6),
                lambda p:p['prepared']['static']['current_closure'].update(tangent_column_batch=2),
                lambda p:p['state']['receipt']['dc_current']['current'][0].update(tangent_column_batch=2),
                lambda p:p['state']['receipt']['solver']['dc_current_closure'][0]['tangent_checks'][0]['bound'].update(complete=False),
                lambda p:p['state']['receipt']['dc_current']['current'][0]['tangent_checks'][-1]['bound'].update(inverse_residual_norm_upper=1.)):
                damaged=copy.deepcopy(raw);mutation(damaged['harmonic_data'])
                torch.save(damaged,tmp_path/'damaged.pt')
                with pytest.raises(ValueError,match='policy|bound'):mw.Result.load(tmp_path/'damaged.pt',scene=scene)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def test_unsupported_dc_policy_fails_before_operator_assembly(monkeypatch):
    scene,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,
        dc_tangent_rule=StoppingRule(preconditioner='none')))
    import witwin.maxwell.mom.harmonic_runtime as runtime
    def forbidden(*args,**kwargs):raise AssertionError('Unsupported policy must fail before assembly')
    monkeypatch.setattr(runtime,'_compile_basis',forbidden)
    with pytest.raises(NotImplementedError,match='electric current or mixed'):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
