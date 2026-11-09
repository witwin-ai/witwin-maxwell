"""Public replay rejects coefficients absent from the declared Scene."""
import os
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_restore_rejects_extra_internal_tensor_coefficients(tmp_path,monkeypatch):
    from dataclasses import replace
    from witwin.maxwell.mom.hmatrix import HMatrixConfig
    from witwin.maxwell.mom.linalg import StoppingRule
    from witwin.maxwell.mom.harmonic_state import preparation_signature
    import witwin.maxwell.mom.tensor_harmonic_bundle as bundle
    scene,config,point=setup()
    scene.add_monitor(mw.PointMonitor('nonlinear_cell',tuple(point[0].tolist()),fields=('Ex','Ey','Ez','Hx','Hy','Hz')))
    config=replace(config,operator_backend='h2',riesz_storage='local_rwg_v1',hierarchical=HMatrixConfig(leaf_size=64),
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-20,stagnation_window=48))
    source=os.environ.get('GSP_MIXED_H2_RESULT')
    if not source:
        source=tmp_path/'base.pt';mw.Simulation.mom(scene,frequency=3e8,config=config).run().save(source)
    # Real internal restore, then emulate an additional supported internal
    # constitutive coefficient. This isolates the public Scene contract gate;
    # it is not a corrupted-byte validation or a new numerical producer test.
    restore=bundle.restore_mixed_tensor_harmonic_bundle
    reached=[]
    def extra(*args,**kwargs):
        model,prepared,state=restore(*args,**kwargs)
        prepared['nonlinear_coefficients']['chi2_tensor']=state.cell_electric.real.new_zeros(state.cell_electric.shape[1],3,3,3)
        prepared['_signature']=preparation_signature(prepared)
        reached.append(True)
        return model,prepared,state
    monkeypatch.setattr(bundle,'restore_mixed_tensor_harmonic_bundle',extra)
    with pytest.raises(ValueError,match='nonlinear coefficients differ'):
        mw.Result.load(source,scene=scene)
    assert reached==[True]


def test_public_restore_rejects_tensor_current_for_scalar_scene(tmp_path,monkeypatch):
    from witwin.maxwell.mom.harmonic_state import preparation_signature
    import witwin.maxwell.mom.tensor_harmonic_bundle as bundle
    scene,config,point=setup(current=True)
    scene.add_monitor(mw.PointMonitor('nonlinear_cell',tuple(point[0].tolist()),fields=('Ex','Ey','Ez','Hx','Hy','Hz')))
    source=os.environ.get('GSP_MIXED_CURRENT_RESULT')
    if not source:
        source=tmp_path/'current.pt';mw.Simulation.mom(scene,frequency=3e8,config=config).run().save(source)
    restore=bundle.restore_mixed_tensor_harmonic_bundle
    reached=[]
    def tensor_conductivity(*args,**kwargs):
        model,prepared,state=restore(*args,**kwargs)
        current=prepared['static']['tensor_current_closure'].current
        # Emulate an internal tensor coefficient after real decoding. This
        # isolates Scene-law admission, not adversarial byte validation.
        current.conductivities=tuple(s[:,None,None].expand(-1,3,3).clone() for s in current.conductivities)
        current._versions=current._state()
        prepared['_signature']=preparation_signature(prepared)
        reached.append(True)
        return model,prepared,state
    monkeypatch.setattr(bundle,'restore_mixed_tensor_harmonic_bundle',tensor_conductivity)
    with pytest.raises(ValueError,match='DC conductivity differs from Scene'):
        mw.Result.load(source,scene=scene)
    assert reached==[True]
