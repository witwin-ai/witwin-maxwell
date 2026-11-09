"""Public mixed replay binds retained solver policy independently of hashes."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup
from witwin.maxwell.mom.tensor_harmonic_runtime import snapshot,restore
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('method',['newton','newton_krylov'])
def test_public_mixed_solver_policy_binding_without_resolve(monkeypatch,method):
    scene,config,_=setup()
    scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.))
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,
        harmonics=1,pump_scales=(.5,),method=method))
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    payload=snapshot(result._harmonic_data)
    def forbidden(*args,**kwargs):raise AssertionError('Policy binding cannot assemble or solve')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    from witwin.maxwell.mom.runtime import SystemOperator
    monkeypatch.setattr(SystemOperator,'__init__',forbidden)
    decoded=restore(payload,scene)
    torch.testing.assert_close(decoded.state.cell_electric,result.harmonic_state.cell_electric,rtol=0,atol=0)
    faults=[('threshold','configured tolerances'),('field_and_threshold','configured tolerances'),
            ('method','solver method mismatch')]
    if method=='newton_krylov':faults.append(('tangent','tangent solver policy mismatch'))
    for fault,message in faults:
        bad=copy.deepcopy(payload);bundle=bad['bundle'];state=bundle['state']
        solver=state['receipt']['solver']
        if fault in ('threshold','field_and_threshold'):
            solver['threshold_V_per_m_l2']=1e6
            if fault=='field_and_threshold':state['state']['cell_electric'][1,0,0]+=1
        elif fault=='method':solver['method']='picard'
        else:solver['tangent_rule']['rtol']*=2
        bundle['state_binding']['sha256']=_digest(state)
        bad['binding']['sha256']=_digest(bundle)
        with pytest.raises(ValueError,match=message):restore(bad,scene)
