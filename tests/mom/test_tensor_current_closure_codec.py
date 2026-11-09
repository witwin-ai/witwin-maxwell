"""Tensor current elimination and exact geometry-bound closure persistence."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_current import HarmonicCurrentClosure
from witwin.maxwell.mom.harmonic_current_snapshot import encode_current_closure,restore_current_closure
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_current_closure_gradient_and_decode_only_codec(tmp_path,monkeypatch):
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    volume=mw.TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    system=SurfaceSystem.create([(0,'body',RWGBasis(volume.boundary),compile_medium(mw.Material(),3e8))])
    binding=bind_volume_region(volume,system)
    strength=vertices.new_tensor(.1,requires_grad=True)
    sigma=vertices.new_tensor([[.3,.02,.01],[.02,.2,.03],[.01,.03,.4]])+strength*torch.eye(3,device='cuda',dtype=torch.float64)
    closure=HarmonicCurrentClosure([volume],[sigma])
    ac=torch.tensor([[[.4,.1,.2]],[[.1,.2,-.1]]],device='cuda',dtype=torch.complex128)
    tpa=.8
    solved=closure.solve(ac,tpa)[0]
    assert solved.electric.norm()>0
    derivative,=torch.autograd.grad(solved.electric.square().sum(),strength)
    step=1e-5
    losses=[HarmonicCurrentClosure([volume],[sigma.detach()+sign*step*torch.eye(3,device='cuda')]).solve(ac,tpa)[0].electric.square().sum() for sign in (-1,1)]
    torch.testing.assert_close(derivative,(losses[1]-losses[0])/(2*step),rtol=2e-5,atol=1e-10)
    raw=encode_current_closure(closure)
    path=tmp_path/'closure.pt';torch.save(raw,path);raw=torch.load(path,weights_only=True)
    expected={1:sigma.detach()}
    def forbidden(*args,**kwargs):raise AssertionError('Restore must not construct or solve current blocks')
    with monkeypatch.context() as patch:
        patch.setattr(HarmonicCurrentClosure,'__init__',forbidden)
        patch.setattr(HarmonicCurrentClosure,'solve',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'inv',forbidden)
        restored=restore_current_closure(raw,[binding],expected_conductivities=expected)
        assert _digest(encode_current_closure(restored))==_digest(raw)
        with pytest.raises(ValueError,match='conductivity mismatch'):
            restore_current_closure(raw,[binding],expected_conductivities={1:.4})
        damaged=copy.deepcopy(raw)
        with torch.no_grad():damaged['conductivities'][0][0,0,1]+=.1
        with pytest.raises(ValueError,match='symmetric'):
            restore_current_closure(damaged,[binding],expected_conductivities=expected)
    actual=restored.solve(ac,tpa)[0]
    torch.testing.assert_close(actual.electric,solved.electric,rtol=0,atol=0)
    with torch.no_grad():restored.conductivities[0].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):restored.assert_current()


def test_mixed_scalar_tensor_ohmic_order_and_gradient():
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    volumes=[mw.TetrahedralVolumeMesh(vertices+i*.2,cells,region_id=i+1) for i in range(2)]
    sigma=vertices.new_tensor([[.3,.02,.01],[.02,.2,.03],[.01,.03,.4]],requires_grad=True)
    closure=HarmonicCurrentClosure(volumes,[.1,sigma])
    field=vertices.new_tensor([[.2,.3,.1],[-.4,.1,.5]],requires_grad=True)
    actual=closure.ohmic_current(field)
    # The admitted law uses the symmetric part, including its coefficient VJP.
    expected=torch.stack((.1*field[0],(.5*(sigma+sigma.T))@field[1]))
    torch.testing.assert_close(actual,expected,rtol=1e-15,atol=1e-16)
    for a,b in zip(torch.autograd.grad(actual.square().sum(),(sigma,field),retain_graph=True),
                   torch.autograd.grad(expected.square().sum(),(sigma,field))):
        torch.testing.assert_close(a,b,rtol=1e-15,atol=1e-16)
