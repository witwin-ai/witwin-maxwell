"""Independent signed-frequency contraction controls for tensor source laws."""
import itertools
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.nonlinear import instantaneous_harmonic_sources,rotate_nonlinear_susceptibility

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def convolution(electric,q,c):
    # Direct signed-frequency polynomial sums, independent of FFT collocation.
    highest=len(electric)-1
    modes={0:electric[0]}
    for n in range(1,highest+1):
        modes[n]=electric[n]/2
        modes[-n]=electric[n].conj()/2
    output=[]
    for n in range(3*highest+1):
        value=torch.zeros_like(electric[0])
        for a,b in itertools.product(modes,repeat=2):
            if a+b==n:value=value+torch.einsum('ijk,...j,...k->...i',q.to(electric.dtype),modes[a],modes[b])
        for a,b,d in itertools.product(modes,repeat=3):
            if a+b+d==n:value=value+torch.einsum('ijkl,...j,...k,...l->...i',c.to(electric.dtype),modes[a],modes[b],modes[d])
        output.append(value*EPSILON_0*(1 if n==0 else 2))
    return torch.stack(output)


def test_tensor_sources_signed_convolution_gradients_and_rotation():
    torch.manual_seed(194)
    dc=torch.randn(2,3,device='cuda',dtype=torch.float64,requires_grad=True)
    ac=torch.randn(1,2,3,device='cuda',dtype=torch.complex128,requires_grad=True)
    electric=torch.cat((dc.to(ac.dtype)[None],ac))
    q=torch.randn(3,3,3,device='cuda',dtype=torch.float64,requires_grad=True)
    c=torch.randn(3,3,3,3,device='cuda',dtype=torch.float64,requires_grad=True)
    result=instantaneous_harmonic_sources(electric,chi2_tensor=q,chi3_tensor=c)
    reference=convolution(electric,q,c)
    torch.testing.assert_close(result.polarization,reference,rtol=2e-13,atol=1e-25)
    weight=torch.randn_like(reference)
    losses=[(v/EPSILON_0*weight.conj()).real.sum() for v in (result.polarization,reference)]
    gradients=[torch.autograd.grad(loss,(dc,ac,q,c),retain_graph=True) for loss in losses]
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)
    angle=torch.tensor(.37,device='cuda',dtype=torch.float64,requires_grad=True)
    zero=angle*0;one=zero+1
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin(),zero)),
        torch.stack((angle.sin(),angle.cos(),zero)),torch.stack((zero,zero,one))))
    lab=instantaneous_harmonic_sources(electric,
        chi2_tensor=rotate_nonlinear_susceptibility(q,rotation),
        chi3_tensor=rotate_nonlinear_susceptibility(c,rotation)).polarization
    material=instantaneous_harmonic_sources(electric@rotation.to(electric.dtype),chi2_tensor=q,chi3_tensor=c)
    expected=material.polarization@rotation.T.to(electric.dtype)
    torch.testing.assert_close(lab,expected,rtol=2e-13,atol=1e-25)
    derivatives=[torch.autograd.grad((v/EPSILON_0*weight.conj()).real.sum(),angle,retain_graph=True)[0] for v in (lab,expected)]
    torch.testing.assert_close(*derivatives,rtol=2e-12,atol=2e-12)


def test_tensor_source_scalar_reduction_broadcast_and_rejections():
    torch.manual_seed(195)
    electric=torch.randn(3,2,3,device='cuda',dtype=torch.complex128)
    electric[0]=electric[0].real.clone()
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    q=torch.zeros(3,3,3,device='cuda',dtype=torch.float64)
    q[range(3),range(3),range(3)]=.2
    c=.3*(torch.einsum('ij,kl->ijkl',eye,eye)+torch.einsum('ik,jl->ijkl',eye,eye)+torch.einsum('il,jk->ijkl',eye,eye))/3
    tensor=instantaneous_harmonic_sources(electric,chi2_tensor=q.expand(2,3,3,3),chi3_tensor=c)
    scalar=instantaneous_harmonic_sources(electric,chi2=.2,chi3=.3)
    torch.testing.assert_close(tensor.polarization,scalar.polarization,rtol=2e-13,atol=1e-25)
    for invalid in (q.to(torch.complex128),q.float(),q.new_zeros(4,3,3,3),q.new_full(q.shape,float('nan'))):
        with pytest.raises(ValueError):instantaneous_harmonic_sources(electric,chi2_tensor=invalid)
    with pytest.raises(ValueError,match='proper orthogonal'):
        rotate_nonlinear_susceptibility(q,eye*2)
    with pytest.raises(ValueError,match='proper orthogonal'):
        rotate_nonlinear_susceptibility(q,torch.diag(eye.new_tensor([-1.,1.,1.])))


def test_tensor_constitutive_power_distinguishes_conservative_coefficients():
    from witwin.maxwell.mom.nonlinear import harmonic_power_ledger
    torch.manual_seed(196)
    vectors=torch.randn(4,3,device='cuda',dtype=torch.float64)
    # Fully symmetric derivatives of cubic/quartic energy polynomials.
    q=torch.einsum('ai,aj,ak->ijk',vectors,vectors,vectors)
    c=torch.einsum('ai,aj,ak,al->ijkl',vectors,vectors,vectors,vectors)
    electric=torch.randn(3,2,3,device='cuda',dtype=torch.complex128)
    electric[0]=electric[0].real.clone()
    sources=instantaneous_harmonic_sources(electric,chi2_tensor=q,chi3_tensor=c,tpa_scale=.4)
    ledger=harmonic_power_ledger(electric,sources,3e7,electric.real.new_ones(2))
    assert ledger['polarization_conservative'] and ledger['dissipation_parseval_closed']
    # An arbitrary real tensor is a supported algebraic source, not a claim of
    # conservative or passive material physics: P_x=eps0*E_y^3 does net work.
    c=torch.zeros_like(c);c[0,1,1,1]=1
    field=electric.new_zeros(2,1,3);field[1,0,0]=1;field[1,0,1]=1j
    sources=instantaneous_harmonic_sources(field,chi3_tensor=c)
    ledger=harmonic_power_ledger(field,sources,3e7,field.real.new_ones(1))
    assert not ledger['polarization_conservative']
    # E_y=sin(wt), so <E_x*d(E_y^3)/dt>=3*w/8.
    expected=field.real.new_tensor(3*EPSILON_0*2*torch.pi*3e7/8)
    torch.testing.assert_close(ledger['polarization_closure_W'],expected,rtol=2e-13,atol=0)
