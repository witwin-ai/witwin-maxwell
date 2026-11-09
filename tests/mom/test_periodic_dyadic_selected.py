"""Selected periodic tensor correction: actual lossy scalar-limit images."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
from witwin.maxwell.mom.operator import IntegrationConfig
import witwin.maxwell.mom.periodic_dyadic_operator as assembly

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dimension',[1,2,3])
def test_selected_regular_blocks_preserve_support_order_and_bound_panel_work(monkeypatch,dimension):
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    system=handle.system
    sample=system.bodies[0].mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    scalar=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),handle.exterior.frequency)
    medium=TensorMedium(scalar.frequency,scalar.epsilon_r*eye,scalar.mu_r*eye)
    vectors=torch.eye(3,device=sample.device,dtype=sample.dtype)[:dimension]
    lattice=LatticeSpec(vectors,sample.new_tensor([.2,0.,0.]))
    reference=LossyDyadicImageReference(lattice,medium,1)
    config=IntegrationConfig(order=2,singular_order=2,pair_batch=3)
    dense,_=assembly.assemble_periodic_dyadic_correction(system,reference=reference,config=config)
    def forbidden(*args,**kwargs):raise AssertionError('Selected block must not assemble a dense correction')
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_correction',forbidden)
    row_edges=torch.tensor([4,0],device=system.device)
    column_edges=torch.tensor([5,1,3],device=system.device)
    for i,j in ((0,0),(0,1),(1,0),(1,1)):
        value,receipt=assembly.periodic_dyadic_regular_block(system,i,j,row_edges,column_edges,
            reference=reference,config=config)
        rows=torch.cat((row_edges+system.J_slices[i].start,row_edges+system.M_slices[i].start))
        cols=torch.cat((column_edges+system.J_slices[j].start,column_edges+system.M_slices[j].start))
        torch.testing.assert_close(value,dense[rows[:,None],cols],rtol=2e-13,atol=1e-15)
        assert receipt['maximum_panel_pairs']<=3
        assert receipt['matrix_bytes']==value.numel()*value.element_size()
        assert not receipt['primary_included'] and not receipt['interiors_periodic']
        assert not receipt['complete_operator'] and not receipt['physical_qualification']
    for invalid in (row_edges[:0],row_edges.repeat(2),row_edges+100,row_edges.to(torch.float64)):
        with pytest.raises(ValueError):
            assembly.periodic_dyadic_regular_block(system,0,1,invalid,column_edges,reference=reference,config=config)
    with pytest.raises(ValueError,match='body index'):
        assembly.periodic_dyadic_regular_block(system,-1,1,row_edges,column_edges,reference=reference,config=config)


def test_single_edge_visits_only_its_two_support_faces(monkeypatch):
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    system=handle.system
    sample=system.bodies[0].mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    scalar=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),handle.exterior.frequency)
    medium=TensorMedium(scalar.frequency,scalar.epsilon_r*eye,scalar.mu_r*eye)
    lattice=LatticeSpec(sample.new_tensor([[1.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    reference=LossyDyadicImageReference(lattice,medium,1)
    original=LossyDyadicImageReference.evaluate
    calls=[]
    def evaluate(self,displacements,**kwargs):
        calls.append(len(displacements))
        return original(self,displacements,**kwargs)
    monkeypatch.setattr(LossyDyadicImageReference,'evaluate',evaluate)
    def forbidden(*args,**kwargs):raise AssertionError('Single-edge work must not assemble a dense correction')
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_correction',forbidden)
    config=IntegrationConfig(order=1,singular_order=1,pair_batch=2)
    value,receipt=assembly.periodic_dyadic_regular_block(system,0,1,
        torch.tensor([4],device=system.device),torch.tensor([5],device=system.device),reference=reference,config=config)
    assert value.shape==(2,2) and torch.isfinite(value).all() and value.norm()>0
    assert receipt['panel_pairs']==4<len(system.bodies[0].mesh.faces)*len(system.bodies[1].mesh.faces)
    assert receipt['maximum_panel_pairs']==2 and calls==[2,2]


@pytest.mark.parametrize('selected',[False,True])
def test_failed_regular_assembly_records_reconstructible_panel_context(monkeypatch,selected):
    from witwin.maxwell.mom.periodic_dyadic import PeriodicDyadicConvergenceError
    from witwin.maxwell.mom.quadrature import triangle_rule
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    system=handle.system
    sample=system.bodies[0].mesh.vertices
    eye=torch.eye(3,device=sample.device,dtype=torch.complex128)
    scalar=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),handle.exterior.frequency)
    medium=TensorMedium(scalar.frequency,scalar.epsilon_r*eye,scalar.mu_r*eye)
    reference=LossyDyadicImageReference(LatticeSpec(sample.new_tensor([[1.,0.,0.]]),
        sample.new_tensor([.2,0.,0.])),medium,1)
    original={'failure_reason':'manufactured_cutoff_failure','converged':False}
    queried=[]
    def fail(self,points,**kwargs):
        queried.append(points.clone())
        raise PeriodicDyadicConvergenceError(original)
    monkeypatch.setattr(LossyDyadicImageReference,'evaluate',fail)
    config=IntegrationConfig(order=2,singular_order=2,pair_batch=3)
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        if selected:
            assembly.periodic_dyadic_regular_block(system,1,0,
                torch.tensor([4,0],device=system.device),torch.tensor([5,1],device=system.device),
                reference=reference,config=config)
        else:
            assembly.assemble_periodic_dyadic_correction(system,reference=reference,config=config)
    record=failure.value.receipt
    context=record['regular_assembly_context']
    assert record['failure_reason']==original['failure_reason']
    assert 'regular_assembly_context' not in original and failure.value.__cause__.receipt is original
    assert (context['test_body_index'],context['source_body_index'])==((1,0) if selected else (0,0))
    rows=torch.tensor(context['test_face_indices'],device=system.device)
    cols=torch.tensor(context['source_face_indices'],device=system.device)
    x,_=triangle_rule(system.bodies[context['test_body_index']].basis.mesh.triangles[rows],context['quadrature_order'])
    y,_=triangle_rule(system.bodies[context['source_body_index']].basis.mesh.triangles[cols],context['quadrature_order'])
    rebuilt=(x[:,:,None]-y[:,None,:]).reshape(-1,3)
    assert torch.equal(rebuilt,queried[0])
    assert context['query_count']==len(rebuilt)==3*4*4
