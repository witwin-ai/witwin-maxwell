"""Fresh preparation rebinding without quadrature, inverses, factors or solves."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance,DualStaticExterior
from witwin.maxwell.mom.dual_preparation_snapshot import snapshot_dual_preparation,restore_dual_preparation
from witwin.maxwell.mom.dual_static_snapshot import snapshot_dual_state,restore_dual_state
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from test_dual_static_current import volume
from test_harmonic_dual_current import ac_fields

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def bindings_for(volumes):
    medium=compile_medium(mw.Material(eps_r=2.),3e8)
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),medium) for i,v in enumerate(volumes)])
    return tuple(bind_volume_region(v,system) for v in volumes)


@pytest.mark.parametrize("reverse",[False,True])
def test_fresh_preparation_and_nonlinear_state_restore_without_solves(tmp_path,monkeypatch,reverse):
    volumes=(volume(),volume(2,.3)); bindings=bindings_for(volumes)
    if reverse: volumes=volumes[::-1]; bindings=bindings[::-1]
    closure=DualHarmonicCurrentClosure(volumes,[2.,3.],[5.,7.])
    laplace=FloatingPotentialOperator(bindings[0].surface_system,testing_order=6)
    result=closure.solve(ac_fields(8),.4)
    exterior=DualStaticExterior(closure.current,result,bindings,laplace)
    prepared,identity=snapshot_dual_preparation(closure,bindings,laplace)
    state=snapshot_dual_state(closure.current,result,exterior)
    path=tmp_path/'dual-prepared.pt'; torch.save({'prepared':prepared,'state':state},path)
    raw=torch.load(path,weights_only=True)
    new_bindings=bindings_for((volume(),volume(2,.3)))
    if reverse: new_bindings=new_bindings[::-1]
    points=closure.current.points.new_tensor([[.8,.7,.9],[-.3,-.2,.4]])
    expected=exterior.fields(points)
    def forbidden(*args,**kwargs): raise AssertionError('Decode-only preparation must not assemble or solve')
    with monkeypatch.context() as patch:
        for name in ('solve','inv','lu_factor','lu_solve'): patch.setattr(torch.linalg,name,forbidden)
        patch.setattr(DualStaticCurrentBalance,'__init__',forbidden)
        patch.setattr(FloatingPotentialOperator,'__init__',forbidden)
        from witwin.maxwell.mom import vie_operator
        patch.setattr(vie_operator,'cell_green_potentials',forbidden)
        restored,floating=restore_dual_preparation(raw['prepared'],new_bindings,sigma_e=[2.,3.],sigma_m=[5.,7.],expected_binding=identity)
        actual,decoded=restore_dual_state(raw['state'],restored.current,new_bindings,floating)
    for name in expected: torch.testing.assert_close(decoded.fields(points)[name],expected[name],rtol=0,atol=0)
    repeated=restored.solve(ac_fields(8),.4)
    for key in ('electric','magnetic','electric_current','magnetic_current'):
        torch.testing.assert_close(actual[key],result[key],rtol=0,atol=0)
        torch.testing.assert_close(repeated[key],result[key],rtol=0,atol=0)
    for group,key in [('current','curl'),('current','matrix'),('nonlinear','response'),('laplace','single_layer')]:
        changed=copy.deepcopy(prepared); changed[group][key][0,0]+=.1
        with pytest.raises(ValueError): restore_dual_preparation(changed,new_bindings,sigma_e=[2.,3.],sigma_m=[5.,7.],expected_binding=identity)
    changed=copy.deepcopy(prepared); changed['nonlinear']['absolute_tolerance']*=10
    with pytest.raises(ValueError,match='settings'): restore_dual_preparation(changed,new_bindings,sigma_e=[2.,3.],sigma_m=[5.,7.],expected_binding=identity)
    with pytest.raises(ValueError,match='conductivity'): restore_dual_preparation(prepared,new_bindings,sigma_e=[2.1,3.],sigma_m=[5.,7.],expected_binding=identity)
