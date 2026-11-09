"""Cylindrical amplitudes against independent Hankel near-field potentials."""
import copy
import math
import runpy
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_cylindrical import cylindrical_orders
from witwin.maxwell.mom.fields import _potentials
from witwin.maxwell.mom.lifting_fields import system_local_coefficients

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('negative_index',[False,True])
def test_cylindrical_asymptotics_hankel_fields_power_and_evanescent_orders(negative_index):
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    if negative_index:
        import witwin.maxwell as mw
        from witwin.maxwell.mom.runtime import SystemDecodeHandle
        from witwin.maxwell.mom.physics import compile_medium
        pole=mw.DrudePole(plasma_frequency=math.sqrt(2)*handle.exterior.frequency,gamma=0.)
        exterior=compile_medium(mw.Material(drude_poles=(pole,),mu_drude_poles=(pole,)),handle.exterior.frequency)
        handle=SystemDecodeHandle(handle.system,exterior,[mw.PlaneWave()],mw.MoMConfig(integration=handle.integration),{})
        assert handle.exterior.k.real<0
    sample=handle.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[.9,0.,0.]]),sample.new_zeros(3))
    angles=torch.arange(8,device=sample.device,dtype=sample.dtype)*2*math.pi/8
    directions=torch.stack((torch.zeros_like(angles),angles.cos(),angles.sin()),-1)
    orders=torch.tensor([[0],[1],[-1]],device=sample.device)
    torch.manual_seed(31)
    coefficients=torch.randn(2*handle.system.total_edges,device=sample.device,dtype=torch.complex128,requires_grad=True)
    result=cylindrical_orders(handle,coefficients,lattice=lattice,orders=orders,directions=directions)
    assert result['propagating_lossless'].tolist()==[True,False,False]
    assert torch.count_nonzero(result['radiated_power_W_per_radian'][1:])==0
    p=result['wavevectors']
    torch.testing.assert_close((p*result['electric']).sum(-1),torch.zeros_like(p[:,:,0]),rtol=0,atol=1e-11)
    torch.testing.assert_close(1j*torch.linalg.cross(p,result['electric']),
        1j*handle.exterior.omega*handle.exterior.mu*result['magnetic'],rtol=1e-11,atol=1e-10)
    expected_power=.5*lattice.cell_measure*result['electric'][0].abs().square().sum(-1)/handle.exterior.eta.real
    torch.testing.assert_close(result['radiated_power_W_per_radian'][0],expected_power,rtol=1e-11,atol=1e-12)
    class HankelZeroMode:
        def evaluate(self,delta,k):
            radial=delta.clone()
            radial[...,0]=0
            rho=radial.norm(dim=-1)
            sign=-1 if k.real<0 else 1
            x=abs(k.real)*rho
            h0=torch.special.bessel_j0(x)+sign*1j*torch.special.bessel_y0(x)
            h1=torch.special.bessel_j1(x)+sign*1j*torch.special.bessel_y1(x)
            g=sign*1j*h0/(4*lattice.cell_measure)
            derivative=-sign*1j*abs(k.real)*h1/(4*lattice.cell_measure)
            return g,derivative[...,None]*radial/rho[...,None]
    errors=[]
    for radius in (200.,400.):
        points=radius*directions
        fields=[torch.zeros_like(points,dtype=coefficients.dtype) for _ in range(2)]
        for index,local in enumerate(handle.body_handles):
            oracle=copy.copy(local)
            oracle.kernel=HankelZeroMode()
            value=_potentials(oracle,system_local_coefficients(handle,coefficients,index),points,
                handle.exterior,order=6,singular_order=6)
            fields=[a+b for a,b in zip(fields,value)]
        amplitude=[value*math.sqrt(radius)*torch.exp(-1j*result['radial_wavenumbers'][0]*radius) for value in fields]
        error=torch.cat((amplitude[0]-result['electric'][0],
            abs(handle.exterior.eta)*(amplitude[1]-result['magnetic'][0])),-1).norm()
        scale=torch.cat((result['electric'][0],abs(handle.exterior.eta)*result['magnetic'][0]),-1).norm()
        errors.append(float((error/scale).detach()))
    assert errors[1]<.6*errors[0] and errors[1]<.003
    gradient,=torch.autograd.grad(result['radiated_power_W_per_radian'].sum(),coefficients)
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    print({'cylindrical_asymptotic_relative_errors':errors,'physical_qualification':False})
    grazing=LatticeSpec(sample.new_tensor([[2*math.pi/abs(handle.exterior.k.real),0.,0.]]),sample.new_zeros(3))
    with pytest.raises(ValueError,match='grazing'):
        cylindrical_orders(handle,coefficients,lattice=grazing,orders=orders,directions=directions)
    with pytest.raises(ValueError,match='perpendicular'):
        cylindrical_orders(handle,coefficients,lattice=lattice,orders=orders,directions=sample.new_tensor([[1.,0.,0.]]))
