"""Projector codec round trips without repeating assembly, admission or solves."""
import copy
import runpy
import pytest
import torch
from witwin.maxwell.mom.static_current_projector import StaticCurrentProjector
from witwin.maxwell.mom.static_projector_snapshot import snapshot_projector,restore_projector,_digest
from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse
from witwin.maxwell.mom.static_current import NodalCurrentBalance
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(autouse=True)
def deterministic_projector(monkeypatch):
    enabled=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def test_projector_decode_only_rebind_and_policy_tampering(monkeypatch,tmp_path):
    make_block=runpy.run_path('tests/mom/test_static_current_projector.py')['block']
    block=make_block();fresh=make_block()
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    projector=StaticCurrentProjector(block,rule,column_batch=3)
    value=torch.arange(12,device='cuda',dtype=torch.float64).cos().requires_grad_()
    expected=projector.apply(value)
    recovery=projector.recover(value)
    expected_adjoint=projector.adjoint(value)
    reduced=value[:block.n_dofs-1].detach()
    expected_recovery_adjoint=projector.recover_adjoint(reduced)
    expected_gradient,=torch.autograd.grad(expected.square().sum()+recovery.square().sum(),value)
    record,binding=snapshot_projector(projector)
    path=tmp_path/'projector.pt';torch.save({'record':record,'binding':binding},path)
    loaded=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Decode-only projector cannot assemble, admit or solve')
    with monkeypatch.context() as patch:
        patch.setattr(StaticCurrentProjector,'__init__',forbidden)
        patch.setattr(NodalCurrentBalance,'__init__',forbidden)
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        for name in ('solve','inv','svdvals'):patch.setattr(torch.linalg,name,forbidden)
        import witwin.maxwell.mom.static_current_projector as native
        patch.setattr(native,'_streamed_inverse_bounds',forbidden)
        restored=restore_projector(loaded['record'],fresh,expected_binding=loaded['binding'])
        assert restored.receipt==projector.receipt
        assert restored.block is fresh
        for name in ('rule','receipt','block'):
            changed=copy.deepcopy(record)
            if name=='rule':changed[name]['rtol']*=2
            elif name=='receipt':changed[name]['tangent_bound']['sigma_min_lower']*=.5
            else:changed[name]['region_id']+=1
            with pytest.raises(ValueError,match='binding'):
                restore_projector(changed,fresh,expected_binding=binding)
        changed=copy.deepcopy(record)
        changed['receipt']['tangent_bound']['sigma_min_lower']*=.5
        rebound={'schema_version':changed['schema_version'],'sha256':_digest(changed)}
        with pytest.raises(ValueError,match='admission'):
            restore_projector(changed,fresh,expected_binding=rebound)
    torch.testing.assert_close(restored.apply(value),expected,rtol=0,atol=0)
    torch.testing.assert_close(restored.recover(value),recovery,rtol=0,atol=0)
    torch.testing.assert_close(restored.adjoint(value),expected_adjoint,rtol=0,atol=0)
    torch.testing.assert_close(restored.recover_adjoint(reduced),expected_recovery_adjoint,rtol=0,atol=0)
    gradient,=torch.autograd.grad(restored.apply(value).square().sum()+restored.recover(value).square().sum(),value)
    torch.testing.assert_close(gradient,expected_gradient,rtol=0,atol=0)
    delayed=restored.apply(value)
    restored.inverse.tangent.tangent[0,0,0]+=1
    with pytest.raises(ValueError,match='binding changed'):restored.apply(value)

    with pytest.raises(ValueError,match='binding changed'):torch.autograd.grad(delayed.sum(),value)
