"""Native HB current elimination propagates the selected DC tangent policy."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_current import HarmonicCurrentClosure
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_hb_current_elimination_with_iterative_dc_and_material_gradient(monkeypatch):
    base=runpy.run_path('tests/mom/test_harmonic_balance.py')['model'](harmonics=2,chi2=.02,chi3=.03)
    vertices=base.background.real.new_tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]])
    volume=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    sigma=vertices.new_tensor(.1,requires_grad=True)
    tpa=vertices.new_tensor(.0001,requires_grad=True)
    dense=HarmonicCurrentClosure([volume],[sigma])
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-22,maxiter=30,restart=8)
    iterative=HarmonicCurrentClosure([volume],[sigma],tangent_rule=rule,tangent_column_batch=2)
    polarization=base.polarization_response.clone();polarization[0]=0
    def model(closure):
        return DenseHarmonicBalance(base.background,polarization,base.current_response,volume.volumes,
            base.frequency,chi2=base.chi2,chi3=base.chi3,tpa_scale=tpa,dc_current_closure=closure)
    expected=model(dense).solve()
    gradients=torch.autograd.grad(expected.electric.abs().square().sum(),(sigma,tpa),retain_graph=True)
    def forbidden(*args,**kwargs):raise AssertionError('No global DC tangent assembly')
    monkeypatch.setattr(iterative.blocks[0],'_matrix',forbidden)
    actual=model(iterative).solve()
    torch.testing.assert_close(actual.electric,expected.electric,rtol=1e-9,atol=1e-10)
    actual_gradients=torch.autograd.grad(actual.electric.abs().square().sum(),(sigma,tpa))
    for a,b in zip(actual_gradients,gradients):torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-10)
    dc=iterative.solve(actual.electric[1:].detach(),tpa.detach())[0]
    assert dc.receipt['linear_solver']=='streamed_inverse_checked_gmres'
    assert not dc.receipt['global_current_jacobian_stored']
    assert dense.numerical_policy()=={}
    assert iterative.numerical_policy()['tangent_column_batch']==2
    iterative.tangent_column_batch=1
    with pytest.raises(RuntimeError,match='mutated'):iterative.assert_current()


def test_current_policy_rejects_inactive_or_unsupported_options():
    with pytest.raises(ValueError,match='active tangent rule'):
        HarmonicCurrentClosure([],[],tangent_column_batch=2)
    with pytest.raises(ValueError,match='unpreconditioned'):
        HarmonicCurrentClosure([],[],tangent_rule=StoppingRule(preconditioner='row_norm'))


def check_saved_current_policy_codec(result,scene,tmp_path,monkeypatch):
    """Reuse a solved public current state; never repeat its EM preparation."""
    import copy
    import witwin.maxwell as mw
    from witwin.maxwell.mom.harmonic_current_snapshot import encode,restore
    from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    prepared=result._harmonic_data.decoder.prepared
    static=prepared['static'];old=static['current_closure']
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-22)
    closure=HarmonicCurrentClosure(old.volumes,old.conductivities,tangent_rule=rule,tangent_column_batch=2)
    raw=encode(dict(static,current_closure=closure))
    before=_digest(raw)
    hb=result._harmonic_data.config.harmonic_balance
    frequency=float(result.harmonic_state.frequencies[1])
    materials=tuple(compile_harmonic_material(s.material,frequency,hb.harmonics) for s in scene.structures)
    path=tmp_path/'source.pt';result.save(path)
    payload=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Policy codec must not solve or rebuild current blocks')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(HarmonicCurrentClosure,'__init__',forbidden)
        patch.setattr(HarmonicCurrentClosure,'solve',forbidden)
        restored=restore(copy.deepcopy(raw),prepared['bindings'][0],materials,hb.dc_testing_order)
        assert restored['current_closure'].numerical_policy()==closure.numerical_policy()
        assert _digest(encode(restored))==before
        for change in (lambda c:c.pop('tangent_column_batch'),
                       lambda c:c['tangent_rule'].pop('maxiter'),
                       lambda c:c['tangent_rule'].update(preconditioner='row_norm'),
                       lambda c:c.update(tangent_column_batch=True)):
            damaged=copy.deepcopy(raw);change(damaged['current_closure'])
            with pytest.raises(ValueError,match='snapshot fields|tangent'):
                restore(damaged,prepared['bindings'][0],materials,hb.dc_testing_order)
        # Public defaults must not silently admit an injected inner policy.
        payload['harmonic_data']['prepared']['static']['current_closure'].update(closure.numerical_policy())
        torch.save(payload,tmp_path/'changed.pt')
        with pytest.raises(ValueError,match='public preparation policy'):
            mw.Result.load(tmp_path/'changed.pt',scene=scene)
