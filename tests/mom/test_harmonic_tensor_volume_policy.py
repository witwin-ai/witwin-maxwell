"""Explicit volume policy identity and retained-response binding."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from test_harmonic_public import setup
from witwin.maxwell.mom.harmonic_runtime import _response_settings,_validate_dense_response_policy

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_policy_canonical_defaults_forwarding_and_receipt_binding():
    _,config,_=setup();hb=config.harmonic_balance
    default=hb.numerical_payload()
    assert 'tensor_face_rule' not in default and 'tensor_kernel_batch' not in default
    active=replace(hb,tensor_face_rule='distance_scaled_duffy',tensor_kernel_batch=8192)
    payload=active.numerical_payload()
    assert payload['tensor_face_rule']=='distance_scaled_duffy' and payload['tensor_kernel_batch']==8192
    assert mw.HarmonicBalanceConfig(volumes=hb.volumes,**payload).numerical_payload()==payload
    configured=replace(config,harmonic_balance=active)
    settings=_response_settings(configured)
    assert settings['tensor_kernel_batch']==8192 and settings['tensor_face_rule']=='distance_scaled_duffy'
    for value in (False,0,-1,1.5,None):
        with pytest.raises(ValueError,match='kernel batch'):replace(hb,tensor_kernel_batch=value)
    with pytest.raises(ValueError,match='face rule'):replace(hb,tensor_face_rule='unknown')
    with pytest.raises(NotImplementedError,match='dense HB'):
        _response_settings(SimpleNamespace(operator_backend='h2',harmonic_balance=active))
    from test_tensor_volume_source import setup as tensor_setup
    operator,binding=tensor_setup()
    row=dict(face_rule=active.tensor_face_rule,kernel_batch=active.tensor_kernel_batch)
    source=dict(max_outer_order=active.source_max_order,max_inner_order=active.source_max_inner_order,
        tensor_face_rule=active.tensor_face_rule,tensor_kernel_batch=active.tensor_kernel_batch,
        matrix={'tensor_kernel_batch':8192,'bodies':[{'region_id':1,'tensor_source_block':{'region_id':1,
            'tensor_kernel_batch':8192,'tensor_volume_evaluations':[{'receipt':row}]}}]})
    part={'receipt':{'source_matrix':source,'tensor_field_evaluations':[{'region_id':1,'order':2,'volume':dict(row)}]}}
    _validate_dense_response_policy(part,configured,operator=operator,bindings=(binding,))
    for kind in ('budget','top_batch','nested_source','direct'):
        bad=copy.deepcopy(part)
        if kind=='budget':bad['receipt']['source_matrix']['max_outer_order']+=1
        elif kind=='top_batch':bad['receipt']['source_matrix']['tensor_kernel_batch']=128
        elif kind=='nested_source':
            bad['receipt']['source_matrix']['matrix']['bodies'][0]['tensor_source_block']['tensor_volume_evaluations'][0]['receipt']['kernel_batch']=128
        else:bad['receipt']['tensor_field_evaluations'][0]['volume']['face_rule']='triangle'
        with pytest.raises(ValueError,match='policy mismatch'):_validate_dense_response_policy(bad,configured,operator=operator,bindings=(binding,))

    for missing in ('bodies','source_evaluations','direct_evaluations'):
        bad=copy.deepcopy(part)
        if missing=='bodies':bad['receipt']['source_matrix']['matrix']['bodies']=[]
        elif missing=='source_evaluations':
            bad['receipt']['source_matrix']['matrix']['bodies'][0]['tensor_source_block']['tensor_volume_evaluations']=[]
        else:bad['receipt']['tensor_field_evaluations']=[]
        with pytest.raises(ValueError,match='coverage mismatch'):
            _validate_dense_response_policy(bad,configured,operator=operator,bindings=(binding,))


def test_public_dense_policy_native_and_bench_identity(tmp_path,monkeypatch):
    from maxwell_bench.harmonic_state import harmonic_simulation_binding
    scene,config,points=setup()
    default=mw.Simulation.mom(scene,frequency=3e8,config=config)
    configured=replace(config,harmonic_balance=replace(config.harmonic_balance,
        tensor_face_rule='distance_scaled_duffy',tensor_kernel_batch=8192))
    simulation=mw.Simulation.mom(scene,frequency=3e8,config=configured)
    before=harmonic_simulation_binding(default);after=harmonic_simulation_binding(simulation)
    assert before!=after and after['harmonic_config']['tensor_kernel_batch']==8192
    # This public source region is scalar: the control proves policy storage,
    # not public tensor-source admission or tensor nonlinear physics.
    result=simulation.run();path=tmp_path/'policy.pt';result.save(path)
    data=torch.load(path,weights_only=False)
    assert data['harmonic_data']['harmonic_config']['tensor_kernel_batch']==8192
    def forbidden(*args,**kwargs):raise AssertionError('Policy replay cannot assemble or solve')
    from witwin.maxwell.mom import runtime
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'assemble_system',forbidden);patch.setattr(torch.linalg,'solve',forbidden)
        replay=mw.Result.load(path,scene=scene)
        torch.testing.assert_close(replay.harmonic_state.cell_electric,result.harmonic_state.cell_electric,rtol=0,atol=0)
    broken=copy.deepcopy(data)
    broken['harmonic_data']['harmonic_config']['tensor_kernel_batch']=4096
    torch.save(broken,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='source quadrature policy mismatch'):
        mw.Result.load(tmp_path/'bad.pt',scene=scene)
