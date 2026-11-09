"""Actual lossy weak PMCHWT streaming actions and independent solve checks."""
import pytest
import torch
import witwin.maxwell as mw
from benchmark.mom_matrix_free import StreamedWeakOperator, experimental_gmres
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, build_rhs

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA PMCHWT required')


@pytest.mark.parametrize('order', [4, 6])
def test_streamed_lossy_forward_adjoint_and_gmres(order):
    basis=RWGBasis(sphere_mesh(radius=.1,subdivisions=0))
    exterior=compile_medium(mw.Material(),3e8)
    interior=compile_medium(mw.Material(eps_r=2.25,sigma_e=.004),3e8)
    config=IntegrationConfig(order=order,singular_order=order)
    matrix,metadata=assemble_weak(basis,exterior,interior,config)
    operator=StreamedWeakOperator(basis,exterior,interior,config,pair_chunk=137)
    generator=torch.Generator(device='cuda').manual_seed(7619)
    x=torch.randn(2,len(matrix),device='cuda',dtype=matrix.dtype,generator=generator)
    y=torch.randn(2,len(matrix),device='cuda',dtype=matrix.dtype,generator=generator)
    forward,adjoint=operator.apply(x),operator.adjoint(y)
    torch.testing.assert_close(forward,x@matrix.mT,rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(adjoint,y@matrix.conj(),rtol=1e-11,atol=1e-13)
    torch.testing.assert_close((forward.conj()*y).sum(),(x.conj()*adjoint).sum(),rtol=1e-11,atol=1e-13)
    assert operator.last_action['triangle_pair_counts']==metadata['triangle_pair_counts']
    assert operator.last_action['stored_operator_matrix_bytes']==0
    assert operator.last_action['stored_global_pair_list_bytes']==0
    source=mw.PlaneWave(source_time=mw.CW(frequency=3e8,phase=.23))
    rhs=build_rhs(basis,exterior,[source])
    actual,stats=experimental_gmres(operator,rhs,rtol=1e-9)
    assert stats['converged'],stats
    expected=torch.linalg.solve(matrix,rhs)
    eta=exterior.eta.real
    scale=torch.cat((torch.full((basis.n_edges,),eta,device='cuda',dtype=torch.float64),torch.ones(basis.n_edges,device='cuda',dtype=torch.float64)))
    assert float(torch.linalg.vector_norm(scale*(actual-expected))/torch.linalg.vector_norm(scale*expected))<1e-7


def test_streamed_input_contract_rejects_bad_shapes_and_chunks():
    basis=RWGBasis(sphere_mesh(radius=.1,subdivisions=0))
    medium=compile_medium(mw.Material(),3e8)
    for bad in [0,-1,True,1.5]:
        with pytest.raises(ValueError,match='pair_chunk'):
            StreamedWeakOperator(basis,medium,medium,pair_chunk=bad)
    operator=StreamedWeakOperator(basis,medium,medium)
    with pytest.raises(ValueError,match='dimension'):
        operator.apply(torch.zeros(3,device='cuda',dtype=torch.complex128))
    with pytest.raises(ValueError,match='dtype'):
        operator.apply(torch.zeros(2*basis.n_edges,device='cuda',dtype=torch.float32))
