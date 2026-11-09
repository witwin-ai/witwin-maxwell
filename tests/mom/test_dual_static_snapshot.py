"""Exact state reopening forbids assembly/solve and rejects changed equations."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance,DualStaticExterior
from witwin.maxwell.mom.dual_static_snapshot import snapshot_dual_state,restore_dual_state
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from test_dual_static_current import volume

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_exact_dual_state_without_resolve_and_with_adversarial_changes(monkeypatch,tmp_path):
    volumes=(volume(),volume(2,.3)); medium=compile_medium(mw.Material(eps_r=2.),3e8)
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),medium) for i,v in enumerate(volumes)])
    bindings=tuple(bind_volume_region(v,system) for v in volumes)
    laplace=FloatingPotentialOperator(system,testing_order=6)
    closure=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    source=torch.arange(24,device='cuda',dtype=torch.float64).reshape(8,3).sin()
    state=closure.solve(source,source.cos())
    exterior=DualStaticExterior(closure,state,bindings,laplace)
    points=source.new_tensor([[.8,.7,.9],[-.3,-.2,.4]])
    expected=exterior.fields(points)
    payload=snapshot_dual_state(closure,state,exterior)
    path=tmp_path/'dual-state.pt'
    torch.save(payload,path)
    payload=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs): raise AssertionError('Restore cannot solve or assemble')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch.linalg,'lu_solve',forbidden)
    monkeypatch.setattr(DualStaticCurrentBalance,'__init__',forbidden)
    monkeypatch.setattr(FloatingPotentialOperator,'__init__',forbidden)
    restored,decoded=restore_dual_state(payload,closure,bindings,laplace)
    for name in ('electric','magnetic','electric_current','magnetic_current'):
        torch.testing.assert_close(restored[name],state[name],rtol=0,atol=0)
    for name in expected: torch.testing.assert_close(decoded.fields(points)[name],expected[name],rtol=0,atol=0)
    for name in ('magnetic_current','electric','magnetic_density','electric_offsets'):
        changed=copy.deepcopy(payload); changed['state'][name].flatten()[0]+=.1
        with pytest.raises(ValueError,match='equations'):
            restore_dual_state(changed,closure,bindings,laplace)
    changed=copy.deepcopy(payload); changed['preparation']['current']['curl']['sha256']='0'*64
    with pytest.raises(ValueError,match='preparation'): restore_dual_state(changed,closure,bindings,laplace)
    changed=copy.deepcopy(payload); changed['physical_qualification']=True
    with pytest.raises(ValueError,match='qualification'): restore_dual_state(changed,closure,bindings,laplace)
