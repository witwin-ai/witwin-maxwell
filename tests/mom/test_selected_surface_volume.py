"""Selected weak cross blocks retain complete RWG support and cell ownership."""
from types import SimpleNamespace
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom import volume_response_matrix as module

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def problem():
    volume=runpy.run_path('tests/mom/test_volume_green.py')['mesh']().refine(1)
    basis=RWGBasis(volume.boundary)
    system=SurfaceSystem.create([(0,'body',basis,compile_medium(mw.Material(eps_r=2.),3e8))])
    operator=SimpleNamespace(system=system,device=volume.vertices.device,dtype=torch.complex128,shape=(2*basis.n_edges,)*2)
    return operator,bind_volume_region(volume,system)


def test_selected_weak_source_order_repetitions_and_full_support(monkeypatch):
    operator,binding=problem();basis=operator.system.bodies[0].basis
    full,_=module.build_volume_source_matrix(operator,[binding],order=4,inner_order=5)
    rows=torch.tensor([5,0,5],device='cuda');cells=torch.tensor([7,1,7],device='cuda')
    calls=[];original=module.cell_green_potentials
    def selected_only(mesh,points,k,**kwargs):
        assert torch.equal(kwargs['cell_indices'],cells)
        calls.append(len(points));return original(mesh,points,k,**kwargs)
    monkeypatch.setattr(module,'cell_green_potentials',selected_only)
    block,receipt=module.build_volume_source_block(operator,binding,rows,cells,order=4,inner_order=5)
    rr=torch.cat((rows,rows+basis.n_edges));cc=(3*cells[:,None]+torch.arange(3,device='cuda')).flatten()
    torch.testing.assert_close(block,full[rr[:,None],cc[None,:]],rtol=2e-12,atol=1e-13)
    faces=torch.unique(torch.cat((basis.plus_faces[rows],basis.minus_faces[rows])))
    assert len(calls)==len(faces)==receipt['integrated_support_faces']
    assert len(faces)<len(basis.mesh.faces)
    assert not receipt['linear_contrast_added'] and not receipt['physical_qualification']
    x=torch.randn(9,device='cuda',dtype=torch.complex128,requires_grad=True)
    y=torch.randn(6,device='cuda',dtype=torch.complex128)
    grad,=torch.autograd.grad((y.conj()*(block@x)).real.sum(),x)
    torch.testing.assert_close(grad,block.mH@y,rtol=1e-12,atol=1e-12)


def test_selected_weak_source_empty_indices_ownership_and_validation():
    operator,binding=problem();ids=torch.tensor([0],device='cuda');empty=ids[:0]
    for rows,cells,shape in ((empty,ids,(0,3)),(ids,empty,(2,0)),(empty,empty,(0,0))):
        value,receipt=module.build_volume_source_block(operator,binding,rows,cells)
        assert value.shape==shape and receipt['integrated_support_faces']==0
    for invalid in ([-1],ids.cpu(),ids.float(),ids[:,None],ids-1):
        with pytest.raises(ValueError,match='indices'):
            module.build_volume_source_block(operator,binding,invalid,ids)
    for invalid in (ids+len(binding.volume.cells),ids.float()):
        with pytest.raises(ValueError,match='indices'):
            module.build_volume_source_block(operator,binding,ids,invalid)
    another,_=problem()
    with pytest.raises(ValueError,match='owning'):
        module.build_volume_source_block(another,binding,ids,ids)


def test_support_selection_ignores_separate_plus_minus_face_caches():
    operator,binding=problem();basis=operator.system.bodies[0].basis
    rows=torch.tensor([0],device='cuda');cells=torch.tensor([0],device='cuda')
    expected,receipt=module.build_volume_source_block(operator,binding,rows,cells,order=3,inner_order=4)
    with torch.no_grad():basis.plus_faces[0]=basis.minus_faces[0]
    actual,changed=module.build_volume_source_block(operator,binding,rows,cells,order=3,inner_order=4)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert changed['integrated_support_faces']==receipt['integrated_support_faces']==2
