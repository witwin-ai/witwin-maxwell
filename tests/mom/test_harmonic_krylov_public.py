"""Public Newton-Krylov policy, exact restore and policy-tampering rejection."""
import copy
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.volume_h_snapshot import _digest
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('backend,implicit',[('dense','dense'),('h2','streamed_krylov'),
    ('dense','streamed_inverse_krylov')])
def test_public_krylov_policy_restore(backend,implicit,tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        if backend=='dense':scene,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
        else:scene,config=runpy.run_path('tests/mom/test_compressed_hb_public.py')['setup']('current',backend)
        config=replace(config,harmonic_balance=replace(config.harmonic_balance,method='newton_krylov',
            implicit_solver=implicit,tangent_column_batch=4,pump_scales=(1.,)))
        hb=config.harmonic_balance
        result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
        path=tmp_path/'result.pt';result.save(path)
        raw=torch.load(path,weights_only=True)
        policy=raw['harmonic_data']['harmonic_config']
        assert policy==hb.numerical_payload()
        assert policy['tangent_rule']['method']=='gmres'
        point=hb.volumes[0].vertices.new_tensor([[.16,.1,.09]])
        expected=result.query_fields(point,region_id=0,field_kind='total',harmonic=2)
        def forbidden(*args,**kwargs):raise AssertionError('Policy restore must not solve')
        with monkeypatch.context() as patch:
            patch.setattr(DenseHarmonicBalance,'solve',forbidden)
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            restored=mw.Result.load(path,scene=scene)
            assert restored._harmonic_data.config.harmonic_balance.numerical_payload()==policy
            for a,b in zip(restored.query_fields(point,region_id=0,field_kind='total',harmonic=2),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            restored.save(tmp_path/'second.pt')
            assert _digest(torch.load(tmp_path/'second.pt',weights_only=True)['harmonic_data'])==_digest(raw['harmonic_data'])
            for mutation,match in (
                (lambda p:p['harmonic_config']['tangent_rule'].update(rtol=1e-6),'policy mismatch'),
                (lambda p:p['harmonic_config'].pop('tangent_rule'),'not canonical'),
                (lambda p:p['state']['receipt']['solver'].update(method='picard'),'solver receipt')):
                damaged=copy.deepcopy(raw);mutation(damaged['harmonic_data'])
                torch.save(damaged,tmp_path/'damaged.pt')
                with pytest.raises(ValueError,match=match):mw.Result.load(tmp_path/'damaged.pt',scene=scene)
            if implicit=='streamed_inverse_krylov':
                for mutation in (
                    lambda c:c.update(complete=False),
                    lambda c:c.update(inverse_residual_norm_upper=1.),
                    lambda c:c.update(inverse_norm_upper=c['inverse_norm_upper']*2),
                    lambda c:c['tangent_rule'].update(rtol=1e-6),
                    lambda c:c.update(method='complete_streamed_diagonal_dominance')):
                    damaged=copy.deepcopy(raw)
                    mutation(damaged['harmonic_data']['state']['receipt']['solver']['implicit_tangent_checks'][0])
                    torch.save(damaged,tmp_path/'damaged.pt')
                    with pytest.raises(ValueError,match='bound|checks mismatch'):
                        mw.Result.load(tmp_path/'damaged.pt',scene=scene)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
