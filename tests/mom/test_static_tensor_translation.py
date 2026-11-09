"""Relative collocation preserves local PV admission under global translation."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis,sphere_mesh,SurfaceMesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('floating',[False,True])
def test_translated_actual_collocation_and_field_queries(floating):
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    coefficient=eye.new_tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]])
    drive=eye.new_tensor([1.,-.3,.5])
    translated=sphere_mesh(radius=.1,subdivisions=0,center=(100.,0.,0.))
    shift=eye.new_tensor([100.,0.,0.])
    # Compare the same rounded vertices, not two independently rounded meshes.
    local=SurfaceMesh(translated.vertices-shift,translated.faces)
    operators=[]
    for mesh in (local,translated):
        system=SurfaceSystem.create([(0,'body',RWGBasis(mesh),compile_medium(mw.Material(),3e8))])
        cls=StaticTensorFloatingOperator if floating else StaticTensorInterfaceOperator
        operators.append(cls(system,eye,(None if floating else coefficient,),pair_batch=37))
    torch.testing.assert_close(operators[1].matrix,operators[0].matrix,rtol=1e-12,atol=1e-12)
    responses=[op.solve_uniform(drive) for op in operators]
    torch.testing.assert_close(responses[1].conormal,responses[0].conormal,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(responses[1].potential+shift@drive,responses[0].potential,rtol=1e-10,atol=1e-10)
    point=eye.new_tensor([[.27,.13,.09]])
    for region in ((0,) if floating else (0,1)):
        query=point if region==0 else point*.1
        a=responses[0].fields(query,region);b=responses[1].fields(query+shift,region)
        torch.testing.assert_close(a[1],b[1],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(a[0],b[0]+shift@drive,rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError,match='face-center'):
        operators[1]._layers(operators[1].exterior,operators[1].centers+.001,
            operators[1].triangles,operators[1].normals,diagonal=True)


def test_translated_panel_preserves_resolved_offplane_and_edges():
    eye=torch.eye(3,device='cuda',dtype=torch.float64);kernel=StaticTensorGreen(eye)
    panel=eye.new_tensor([[[0.,0.,0.],[.125,0.,0.],[0.,.125,0.]]])
    point=eye.new_tensor([[.03125,.03125,.001]])
    shift=eye.new_tensor([100.,-64.,32.])
    expected=kernel.panel_integrals(point,panel)
    actual=kernel.panel_integrals(point+shift,panel+shift)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    mask=torch.ones(1,device='cuda',dtype=torch.bool)
    with pytest.raises(ValueError,match='principal values'):
        kernel.panel_integrals(point+shift,panel+shift,principal_value=mask)
    with pytest.raises(ValueError,match='edges'):
        kernel.panel_integrals(panel[:,0]+shift,panel+shift,principal_value=mask)
