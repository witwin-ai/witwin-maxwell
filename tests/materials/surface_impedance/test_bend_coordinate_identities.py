"""Independent metric/Maxwell/work identities; no public curved solver claim."""
import math
import numpy as np
import pytest
import torch

from scripts.dev.fdtd.bend_coordinates_reference import ConstantCurvatureChart,curl_from_autograd

_ROTATED=((math.cos(.7),-math.sin(.7),0.),(math.sin(.7),math.cos(.7),0.),(0.,0.,1.))

@pytest.mark.parametrize("curvature",(0.,-3.,3.))
@pytest.mark.parametrize("frame",(None,_ROTATED))
def test_chart_inverse_jacobian_flat_limit_and_focal_wrap_guards(curvature,frame):
    chart=ConstantCurvatureChart(curvature,origin=(.2,-.1,.3),frame=frame)
    chart.validate_bounds(((-.2,.2),(-.05,.05),(-.1,.1)))
    q=torch.tensor(((.1,.02,.03),(-.12,-.04,.02)),dtype=torch.float64,requires_grad=True)
    r=chart.position(q)
    jacobian=torch.stack([torch.autograd.grad(r[:,i].sum(),q,retain_graph=True)[0] for i in range(3)],1)
    torch.testing.assert_close(chart.jacobian(q),jacobian,rtol=3e-14,atol=2e-15)
    torch.testing.assert_close(torch.linalg.det(jacobian),chart.metric(q),rtol=3e-14,atol=2e-15)
    torch.testing.assert_close(chart.inverse(r),q,rtol=3e-14,atol=2e-15)
    if curvature==0:
        expected=q if frame is None else q@q.new_tensor(frame).T
        torch.testing.assert_close(r,expected+q.new_tensor(chart.origin),rtol=0,atol=0)
        torch.testing.assert_close(chart.material_metric(q),torch.eye(3,dtype=q.dtype).expand(2,-1,-1),rtol=0,atol=0)
    else:
        with pytest.raises(ValueError,match="focal"):
            chart.validate_bounds(((-.2,.2),(-1.,1.),(-.1,.1)))
        with pytest.raises(ValueError,match="wrap"):
            chart.validate_bounds(((-2.,2.),(-.01,.01),(-.1,.1)))
        shifted=q.detach().clone();shifted[:,0]+=2.2
        bounds=((2.,2.5),(-.05,.05),(-.1,.1))
        torch.testing.assert_close(chart.inverse(chart.position(shifted),bounds=bounds),shifted,rtol=3e-14,atol=2e-15)


@pytest.mark.parametrize("curvature",(0.,-3.,3.))
@pytest.mark.parametrize("conductivity",(0.,.02))
@pytest.mark.parametrize("frame",(None,_ROTATED))
def test_transformed_physical_plane_wave_obeys_both_maxwell_curls_with_sigma(curvature,conductivity,frame):
    eps0,mu0=8.8541878128e-12,1.25663706212e-6
    omega=2*math.pi*1e9
    chart=ConstantCurvatureChart(curvature,frame=frame)
    q=torch.tensor(((.11,.03,.04),(-.12,-.02,.05)),dtype=torch.float64,requires_grad=True)
    r=chart.position(q)
    direction=q.new_tensor((1.,2.,3.));direction/=direction.norm()
    polarization=torch.linalg.cross(direction,q.new_tensor((.3,-.4,.7)));polarization/=polarization.norm()
    eps,mu=2.4,1.3
    k=omega*np.sqrt(mu0*mu*(eps0*eps+1j*conductivity/omega))
    phase=torch.exp(1j*k*(r@direction))
    physical_e=phase[:,None]*polarization
    physical_h=phase[:,None]*torch.linalg.cross(direction,polarization)*(k/(omega*mu0*mu))
    e,h=chart.pullback_field(q,physical_e),chart.pullback_field(q,physical_h)
    metric=chart.material_metric(q).to(e.dtype)
    expected_e=1j*omega*mu0*mu*torch.einsum("nij,nj->ni",metric,h)
    expected_h=(conductivity-1j*omega*eps0*eps)*torch.einsum("nij,nj->ni",metric,e)
    torch.testing.assert_close(curl_from_autograd(e,q),expected_e,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(curl_from_autograd(h,q),expected_h,rtol=2e-13,atol=2e-15)


def test_source_work_bulk_heat_and_every_face_flux_are_physical_coordinate_invariants():
    chart=ConstantCurvatureChart(-3.)
    q=torch.tensor(((.11,.03,.04),(-.12,-.02,.05)),dtype=torch.float64)
    e=q.new_tensor(((1.,2.,-3.),(.5,-1.,2.))).to(torch.complex128)*(1+.3j)
    h=q.new_tensor(((-2.,.4,1.),(3.,.2,.8))).to(torch.complex128)*(.7-.2j)
    j=q.new_tensor(((.3,.6,-.2),(-.4,.8,.3)))
    ce,ch=chart.pullback_field(q,e),chart.pullback_field(q,h)
    cj=chart.pullback_current(q,j)
    torch.testing.assert_close((ce.conj()*cj).sum(-1),chart.metric(q)*(e.conj()*j).sum(-1),rtol=3e-14,atol=2e-15)
    metric=chart.material_metric(q).to(e.dtype)
    chart_heat=torch.einsum("ni,nij,nj->n",ce.conj(),metric,ce).real
    torch.testing.assert_close(chart_heat,chart.metric(q)*e.abs().square().sum(-1),rtol=3e-14,atol=2e-15)
    for axis in range(3):
        normal=torch.eye(3,dtype=q.dtype)[axis].expand_as(q)
        chart_flux=(torch.linalg.cross(ce,ch.conj()).real*normal).sum(-1)
        area=chart.physical_area_vector(q,normal)
        physical_flux=(torch.linalg.cross(e,h.conj()).real*area).sum(-1)
        torch.testing.assert_close(chart_flux,physical_flux,rtol=3e-14,atol=2e-15)
    torch.testing.assert_close(chart.physical_field(q,ce),e,rtol=3e-14,atol=2e-15)


def test_covariant_port_voltage_current_and_two_component_copper_power_are_invariant():
    chart=ConstantCurvatureChart(3.)
    abscissa,weights=np.polynomial.legendre.leggauss(16)
    x=torch.tensor(abscissa,dtype=torch.float64)
    w=torch.tensor(weights,dtype=torch.float64)
    def integral(start,end,magnetic=False):
        start,end=x.new_tensor(start),x.new_tensor(end)
        q=(start+end)/2+x[:,None]*(end-start)/2
        r=chart.position(q)
        field=torch.stack((-r[:,1]/2,r[:,0]/2,torch.zeros_like(x)),-1) if magnetic else x.new_tensor((0.,1.,0.)).expand_as(q)
        pulled=chart.pullback_field(q,field)
        return (w*(pulled*(end-start)/2).sum(-1)).sum()
    a,b=(-.1,.02,0.),(.12,.02,0.)
    voltage=integral(a,b)
    displacement=chart.position(x.new_tensor(b))-chart.position(x.new_tensor(a))
    torch.testing.assert_close(voltage,displacement[1],rtol=3e-14,atol=2e-15)
    points=((-0.1,-.03,0.),(.12,-.03,0.),(.12,.04,0.),(-.1,.04,0.))
    current=sum(integral(start,end,True) for start,end in zip(points,points[1:]+points[:1]))
    area=(.12+.1)*(.04+.03)*(1-3*(.04-.03)/2)
    torch.testing.assert_close(current,x.new_tensor(area),rtol=3e-14,atol=2e-15)
    q=x.new_tensor(((.1,-.03,0.),(-.1,.04,0.)))
    scale=chart.metric(q)
    physical_e=x.new_tensor(((.7,1.2),(-.3,.8)))
    covariant_e=physical_e*torch.stack((scale,torch.ones_like(scale)),-1)
    effective_y=torch.stack((1/scale,scale),-1)
    physical_power=scale*physical_e.square().sum(-1)
    chart_power=(effective_y*covariant_e.square()).sum(-1)
    torch.testing.assert_close(chart_power,physical_power,rtol=3e-14,atol=2e-15)
