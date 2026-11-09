"""Public tensor-source cutoff with scalar linear media and retained responses."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_sources_cutoff_prefix_and_exact_restore(tmp_path,monkeypatch):
    scene,config,point=setup()
    scene.structures[0]=replace(scene.structures[0],material=mw.Material(eps_r=2.,mu_r=1.3))
    axis=point.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
    q=.002*torch.einsum('i,j,k->ijk',axis,axis,axis)
    c=.001*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis)
    scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.,
        nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q,chi3=c)))
    policy=mw.HarmonicCutoffConfig(max_harmonics=3,failure='record')
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,harmonics=1,cutoff=policy))
    public=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    original=public.solver.prepared
    from witwin.maxwell.mom.physics import Medium
    assert all(isinstance(m,Medium) for body in public.solver.materials for m in body.linear_samples)
    from witwin.maxwell.mom.harmonic_state import preparation_signature
    import witwin.maxwell.mom.tensor_harmonic_preparation as preparation
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    response=preparation.prepare_cell_field_response;calls=[]
    def appended(operator,*args,**kwargs):
        calls.append(operator.exterior.frequency)
        return response(operator,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Cutoff must reuse DC and stored replay must not assemble or solve')
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_cell_field_response',appended)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        result=public.run()
    extended=result._harmonic_data.decoder.prepared
    assert calls==[6e8,9e8]
    assert extended['static'] is original['static'] and extended['harmonics'][0] is original['harmonics'][0]
    assert extended['bindings'][0] is original['bindings'][0]
    assert public.solver.prepared is original and preparation_signature(original)==original['_signature']
    assert 'cutoff' not in original['receipt']
    for name in ('chi2_tensor','chi3_tensor'):
        torch.testing.assert_close(extended['nonlinear_coefficients'][name],original['nonlinear_coefficients'][name],rtol=0,atol=0)
    cutoff=result.solver_stats['cutoff']
    assert cutoff['final_harmonics']==3 and len(cutoff['checks'])==2
    assert not cutoff['physical_qualification'] and not cutoff['harmonic_truncation_converged']
    assert result._metadata['cell_cutoff_estimate_passed']==cutoff['cell_truncation_estimate_passed']
    if not cutoff['cell_truncation_estimate_passed']:
        assert cutoff['failure_reason']=='harmonic_budget_exhausted'
    assert bool(result.harmonic_state.cell_electric[2:].norm()>0)
    fields=result.query_fields(point,region_id=2,field_kind='total',harmonic=3)
    path=tmp_path/'tensor-cutoff.pt';result.save(path)
    from witwin.maxwell.mom.runtime import SystemOperator
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.solver_stats['cutoff']==cutoff
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(restored.harmonic_state,name),getattr(result.harmonic_state,name),rtol=0,atol=0)
        for actual,expected in zip(restored.query_fields(point,region_id=2,field_kind='total',harmonic=3),fields):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    payload=torch.load(path,weights_only=False)
    damaged=copy.deepcopy(payload)
    damaged['solver_stats']['cutoff']['cell_truncation_estimate_passed']=not cutoff['cell_truncation_estimate_passed']
    altered=tmp_path/'altered.pt';torch.save(damaged,altered)
    with pytest.raises(ValueError,match='statistics differ'):mw.Result.load(altered,scene=scene)
    print({'cutoff':cutoff,'physical_qualification':False})
