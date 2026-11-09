"""Explicit finite projected resolvents, never ordinary forced solutions at poles."""
import cmath
from dataclasses import replace
import math
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_limits import projected_bloch_green

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('degenerate',[False,True])
def test_projected_resolvent_pde_residue_two_limits_and_splitting(degenerate):
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    beta=vectors.new_zeros(3) if degenerate else vectors.new_tensor([.2,.1,.05])
    lattice=LatticeSpec(vectors,beta)
    kernel=EwaldHelmholtzKernel(lattice,3.,real_shells=4,reciprocal_shells=6,quadrature_order=128)
    selected=torch.tensor([1,0,0] if degenerate else [0,0,0],device='cuda')
    points=vectors.new_tensor([[.13,.17,.21],[.24,-.18,.12]],requires_grad=True)
    result=projected_bloch_green(kernel,points,eigen_order=selected)
    assert len(result['projected_orders'])==(6 if degenerate else 1)
    eigenvalue=float(result['eigenvalue_per_m2'])
    with pytest.raises(ValueError,match='spectral pole'):
        kernel.evaluate(points,math.sqrt(eigenvalue))
    for component in ('real','imag'):
        derivative,=torch.autograd.grad(getattr(result['green'],component).sum(),points,retain_graph=True)
        torch.testing.assert_close(derivative,getattr(result['gradient'],component),rtol=1e-10,atol=1e-10)
        laplace=torch.zeros(len(points),device='cuda',dtype=points.dtype)
        for axis in range(3):
            row,=torch.autograd.grad(getattr(result['gradient'][:,axis],component).sum(),points,retain_graph=True)
            laplace=laplace+row[:,axis]
        torch.testing.assert_close(laplace+eigenvalue*getattr(result['green'],component),
            getattr(result['projector_kernel'],component),rtol=1e-8,atol=1e-8)
    alternative=projected_bloch_green(replace(kernel,eta=2.),points.detach(),eigen_order=selected)
    for name in ('green','gradient'):
        torch.testing.assert_close(result[name],alternative[name],rtol=1e-9,atol=1e-9)
    shifted=projected_bloch_green(kernel,points.detach()+vectors[0],eigen_order=selected)
    for name in ('green','gradient','residue_z'):
        torch.testing.assert_close(shifted[name],result[name]*torch.exp(1j*beta[0]),rtol=1e-10,atol=1e-10)
    errors=[]
    for direction in (1j,1+1j):
        path=[]
        for delta in (.03,.01,.003):
            z=eigenvalue+delta*direction
            g,dg=kernel.evaluate(points.detach(),cmath.sqrt(z))
            finite=g-result['projector_kernel']/(eigenvalue-z)
            finite_gradient=dg-result['projector_gradient']/(eigenvalue-z)
            difference=torch.cat(((finite-result['green']).reshape(-1),
                (finite_gradient-result['gradient']).reshape(-1))).norm()
            path.append(float(difference.detach()))
            torch.testing.assert_close((z-eigenvalue)*g,result['residue_z'],rtol=0,atol=.02)
        assert path[2]<.15*path[0] and path[2]<.003
        errors.append(path)
    assert not result['ordinary_forced_solution'] and not result['physical_qualification']
    print({'degenerate':degenerate,'two_projected_limit_path_errors':errors})


def test_projected_resolvent_requires_complete_eigenspace_scan():
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    kernel=EwaldHelmholtzKernel(LatticeSpec(vectors,vectors.new_zeros(3)),2.,reciprocal_shells=1)
    with pytest.raises(ValueError,match='complete degenerate'):
        projected_bloch_green(kernel,vectors.new_tensor([[.1,.2,.3]]),
            eigen_order=torch.tensor([2,0,0],device='cuda'))


def test_projected_resolvent_rejects_ambiguous_near_degeneracy():
    vectors=torch.eye(3,device='cuda',dtype=torch.float64)
    lattice=LatticeSpec(vectors,vectors.new_tensor([1e-12,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,3.,reciprocal_shells=3)
    with pytest.raises(ValueError,match='projection is ambiguous'):
        projected_bloch_green(kernel,vectors.new_tensor([[.1,.2,.3]]),
            eigen_order=torch.tensor([1,0,0],device='cuda'))
