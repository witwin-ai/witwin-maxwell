"""Exact owner hashes do not replace retained mixed-HB equation checks."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup
from witwin.maxwell.mom.tensor_harmonic_snapshot import (
    snapshot_mixed_tensor_harmonic_state,restore_mixed_tensor_harmonic_state)
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_owned_mixed_response_corruption_rejected_without_solving(monkeypatch):
    scene,config,_=setup()
    # Linear source-region coefficients isolate field equations from the
    # nonlinear constitutive checks; finite altered AC fields still give J=0.
    scene.structures[1]=replace(scene.structures[1],material=mw.Material(eps_r=2.))
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,harmonics=1,pump_scales=(.5,)))
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    prepared=result._harmonic_data.decoder.prepared
    payload,owner=snapshot_mixed_tensor_harmonic_state(result.harmonic_state,prepared)
    def forbidden(*args,**kwargs):raise AssertionError('Retained response validation must not solve')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    from witwin.maxwell.mom.runtime import SystemOperator
    monkeypatch.setattr(SystemOperator,'__init__',forbidden)
    restored=restore_mixed_tensor_harmonic_state(payload,prepared,expected_binding=owner)
    torch.testing.assert_close(restored.cell_electric,result.harmonic_state.cell_electric,rtol=0,atol=0)
    for fault,message in [('surface','surface state'),('field','field violates'),('dc_field','field violates'),
            ('unconverged','solver convergence'),('threshold','solver convergence'),('pump','solver convergence'),
            ('overflow','residual is not finite')]:
        bad=copy.deepcopy(payload)
        if fault=='surface':bad['state']['surface_coefficients'][0][0]+=1
        elif fault=='field':bad['state']['cell_electric'][1,0,0]+=1
        elif fault=='dc_field':bad['state']['cell_electric'][0,0,0]+=1
        elif fault=='unconverged':bad['receipt']['solver']['converged']=False
        elif fault=='threshold':bad['receipt']['solver']['threshold_V_per_m_l2']=-1.
        elif fault=='pump':bad['receipt']['solver']['pump_scale']=.75
        else:bad['state']['surface_coefficients'][0].fill_(1e308)
        # Rebind altered producer bytes: rejection must come from retained
        # equations, not merely an unchanged outer checksum.
        with pytest.raises(ValueError,match=message):
            restore_mixed_tensor_harmonic_state(bad,prepared,
                expected_binding=dict(version=owner['version'],sha256=_digest(bad)))
