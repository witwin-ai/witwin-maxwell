"""Internal cutoff continuation adapter over a validated public H3 Result."""
import copy
import os
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_saved_h5_native_query_replay(tmp_path,monkeypatch):
    source=os.environ.get('GSP_TENSOR_H5_NATIVE')
    if not source:
        pytest.skip('Requires a previously collected native H5 continuation artifact')
    scene,_,point=setup()
    scene.structures[0]=replace(scene.structures[0],material=mw.Material(eps_r=2.,mu_r=1.3))
    axis=point.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
    scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.,nonlinearity=
        mw.TensorNonlinearSusceptibility(chi2=.002*torch.einsum('i,j,k->ijk',axis,axis,axis),
            chi3=.001*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis))))
    from witwin.maxwell.mom.tensor_harmonic_runtime import restore,snapshot
    from witwin.maxwell.mom.runtime import SystemOperator
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
    def forbidden(*args,**kwargs):
        raise AssertionError('Native query replay must not assemble or solve')
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        original=restore(torch.load(source,weights_only=False),scene)
        assert original.state.cell_electric.shape[0]==6
        fields={h:original.query_fields(point,region_id=2,field_kind='total',harmonic=h)
                for h in (4,5)}
        path=tmp_path/'native-replay.pt';torch.save(snapshot(original),path)
        replay=restore(torch.load(path,weights_only=False),scene)
        for h,expected in fields.items():
            actual=replay.query_fields(point,region_id=2,field_kind='total',harmonic=h)
            for a,b in zip(actual,expected):
                assert bool(torch.isfinite(b).all())
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert replay.decoder.prepared['receipt']==original.decoder.prepared['receipt']


def test_restored_tensor_sources_extend_only_h4_h5(tmp_path,monkeypatch):
    scene,config,point=setup()
    scene.structures[0]=replace(scene.structures[0],material=mw.Material(eps_r=2.,mu_r=1.3))
    axis=point.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
    scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.,nonlinearity=
        mw.TensorNonlinearSusceptibility(chi2=.002*torch.einsum('i,j,k->ijk',axis,axis,axis),
            chi3=.001*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis))))
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,harmonics=1,
        cutoff=mw.HarmonicCutoffConfig(max_harmonics=3,failure='record')))
    source=os.environ.get('GSP_TENSOR_CUTOFF_PREFIX')
    if not source:
        source=tmp_path/'prefix.pt';mw.Simulation.mom(scene,frequency=3e8,config=config).run().save(source)
    result=mw.Result.load(source,scene=scene)
    native=result._harmonic_data;prefix=native.decoder.prepared
    previous=copy.deepcopy(prefix['receipt'])
    assert result.solver_stats['cutoff']['failure_reason']=='harmonic_budget_exhausted'
    policy=replace(native.config.harmonic_balance.cutoff,max_harmonics=5)
    continued_config=replace(native.config,harmonic_balance=replace(native.config.harmonic_balance,cutoff=policy))
    from witwin.maxwell.mom.tensor_harmonic_runtime import MixedTensorHarmonicRuntime,snapshot,restore
    # Test-only adapter: all ownership comes from the validated restored data.
    # This does not provide or exercise a public resume API.
    runtime=object.__new__(MixedTensorHarmonicRuntime)
    runtime.simulation=mw.Simulation.mom(scene,frequency=3e8,config=continued_config)
    runtime.config=continued_config;runtime.signature=native.scene_signature
    import witwin.maxwell.mom.tensor_harmonic_preparation as preparation
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    response=preparation.prepare_cell_field_response;calls=[]
    def appended(operator,*args,**kwargs):
        calls.append(operator.exterior.frequency)
        assert operator.exterior.frequency in (1.2e9,1.5e9)
        return response(operator,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Restored response prefix must not be rebuilt')
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_cell_field_response',appended)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        extended,receipt=runtime._extend_cutoff(native,policy)
    assert calls==[1.2e9,1.5e9]
    assert receipt['initial_harmonics']==3 and receipt['final_harmonics']==5 and len(receipt['checks'])==2
    assert not receipt['physical_qualification'] and not receipt['harmonic_truncation_converged']
    prepared=extended.decoder.prepared
    assert prepared['static'] is prefix['static']
    for key in ('harmonics','operators','bindings'):
        assert all(a is b for a,b in zip(prepared[key][:3],prefix[key]))
    from witwin.maxwell.mom.harmonic_state import preparation_signature
    assert prefix['receipt']==previous and preparation_signature(prefix)==prefix['_signature']
    assert extended.config.harmonic_balance.cutoff==policy
    path=tmp_path/'extended-native.pt';torch.save(snapshot(extended),path)
    from witwin.maxwell.mom.runtime import SystemOperator
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        replay=restore(torch.load(path,weights_only=False),scene)
    assert replay.decoder.prepared['receipt']['cutoff']==receipt
    for name in ('cell_electric','polarization','total_volume_current'):
        torch.testing.assert_close(getattr(replay.state,name),getattr(extended.state,name),rtol=0,atol=0)
    print({'cutoff':receipt,'public_resume_api':False,'physical_qualification':False})
