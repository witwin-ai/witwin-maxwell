"""Tensor constitutive HB roots with analytic scalar-direction references."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_rotated_tensor_cubic_root_implicit_gradient_and_ownership():
    direction=torch.tensor([1.,2.,3.],device='cuda',dtype=torch.float64)
    direction=direction/direction.norm()
    chi=torch.tensor(.2,device='cuda',dtype=torch.float64,requires_grad=True)
    tensor=chi*torch.einsum('i,j,k,l->ijkl',direction,direction,direction,direction)
    background=torch.zeros(2,1,3,device='cuda',dtype=torch.complex128)
    background[1,0]=direction
    maps=-torch.eye(3,device='cuda',dtype=torch.complex128).repeat(2,1,1)/EPSILON_0
    current=torch.zeros_like(maps);current[1]=maps[1]/(-2j*math.pi*3e8)
    model=DenseHarmonicBalance(background,maps,current,background.real.new_ones(1),3e8,chi3_tensor=tensor)
    assert model.chi3_tensor is not tensor
    solution=model.solve()
    amplitude=(solution.electric[1,0].real*direction).sum()
    torch.testing.assert_close(solution.electric[1,0],amplitude*direction.to(background.dtype),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(amplitude+.75*chi*amplitude**3,amplitude.new_tensor(1.),rtol=1e-10,atol=1e-10)
    derivative,=torch.autograd.grad(amplitude,chi)
    torch.testing.assert_close(derivative,-.75*amplitude**3/(1+2.25*chi*amplitude**2),rtol=1e-8,atol=1e-10)
    # Caller edits do not change the owned response; edits to owned entries reject.
    before=model.residual(solution.electric.detach())
    with torch.no_grad():tensor.add_(1)
    torch.testing.assert_close(model.residual(solution.electric.detach()),before,rtol=0,atol=0)
    with torch.no_grad():model.chi3_tensor.add_(.1)
    with pytest.raises(RuntimeError,match='mutated'):model.residual(solution.electric.detach())
