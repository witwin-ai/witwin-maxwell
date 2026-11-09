"""Exact magnetic contrast storage without matrix assembly or magnetic solve."""
import copy
import pytest
import torch
from test_magnetostatic import mesh
from test_static_coupling import system_from_volume
from test_tensor_magnetostatic import coefficient
from witwin.maxwell.constants import MU_0
from witwin.maxwell.mom.static_magnetic_volume import MagneticVolumeContrastResponse
from witwin.maxwell.mom.static_magnetic_snapshot import snapshot_magnetic_volume_response,restore_magnetic_volume_response,ARRAYS
from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('body_count',[1,2])
def test_exact_magnetic_response_storage_and_independent_owner(monkeypatch,tmp_path,body_count):
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh,bind_volume_region
    def geometry():
        first=mesh()
        volumes=[first]+[TetrahedralVolumeMesh(first.vertices+first.vertices.new_tensor([.3,0.,0.]),first.cells,region_id=2)]*(body_count-1)
        system=SurfaceSystem.create([(i,'body'+str(i),RWGBasis(v.boundary),compile_medium(mw.Material(),3e8)) for i,v in enumerate(volumes)])
        return system,tuple(bind_volume_region(v,system) for v in volumes)
    system,bindings=geometry();volume=bindings[0].volume;sample=volume.vertices
    ext=MU_0*torch.eye(3,device='cuda',dtype=sample.dtype)
    current=sample.new_tensor([1.,-.4,.3]).expand(len(volume.cells),3)
    response=MagneticVolumeContrastResponse(system,bindings,tuple(current*(i+1) for i in range(body_count)),ext,
        tuple(coefficient(sample)*(i+1) for i in range(body_count)),background_h=sample.new_tensor([.1,.2,.3]))
    import witwin.maxwell.mom.static_magnetic_snapshot as codec
    observed=[]
    def validate(*args,**kwargs):
        observed.append(kwargs.get('max_matrix_bytes'))
        return restore_magnetic_volume_response(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(codec,'restore_magnetic_volume_response',validate)
        payload,owner=snapshot_magnetic_volume_response(response)
    assert observed==[response.matrix.numel()*response.matrix.element_size()]
    path=tmp_path/'magnetic.pt';torch.save(payload,path);saved=torch.load(path,weights_only=True)
    other,other_bindings=geometry()
    def forbidden(*args,**kwargs):raise AssertionError('Restore assembled or solved the magnetic system')
    with monkeypatch.context() as patch:
        patch.setattr(MagneticVolumeContrastResponse,'__init__',forbidden)
        patch.setattr(MagneticVolumeContrastResponse,'_source',forbidden)
        patch.setattr(StaticTensorGreen,'panel_integrals',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=restore_magnetic_volume_response(saved,other,other_bindings,expected_binding=owner)
    for name in ARRAYS:torch.testing.assert_close(getattr(restored,name),getattr(response,name),rtol=0,atol=0)
    for old,new in zip((response.exterior,*response.interiors),(restored.exterior,*restored.interiors)):
        for name in ('coefficient','factor','root_determinant'):
            torch.testing.assert_close(getattr(new,name),getattr(old,name),rtol=0,atol=0)
    for a,b in zip(restored.currents,response.currents):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(restored.sources,response.sources):torch.testing.assert_close(a.polarization,b.polarization,rtol=0,atol=0)
    for region,q in [(0,[[.2,.12,.14]])]+[(i+1,[[.021+.3*i,.019,.023]]) for i in range(body_count)]:
        for a,b in zip(restored.fields(sample.new_tensor(q),region)[:2],response.fields(sample.new_tensor(q),region)[:2]):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-18)
    for key in ('matrix','cell_fields','contrast'):
        bad=copy.deepcopy(saved);bad['state'][key].add_(.1)
        with pytest.raises(ValueError,match='owning binding'):
            restore_magnetic_volume_response(bad,other,other_bindings,expected_binding=owner)
        with pytest.raises(ValueError,match='contrast|partition|residual'):
            restore_magnetic_volume_response(bad,other,other_bindings,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    bad=copy.deepcopy(saved);bad['kernels'][0]['factor'][0,0]*=2
    with pytest.raises(ValueError,match='factors'):
        restore_magnetic_volume_response(bad,other,other_bindings,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    with pytest.raises(ValueError,match='budget'):
        restore_magnetic_volume_response(saved,other,other_bindings,expected_binding=owner,max_matrix_bytes=1)
    saved['state']['cell_fields'].add_(1)
    restored.assert_current()
    restored.source_field.add_(1)
    with pytest.raises(ValueError,match='changed'):restored.assert_current()
