"""Differential Maxwell, reciprocity and low-frequency conductive DC checks."""
import math
import pytest
import torch
from witwin.maxwell.mom.conductive_static import conductive_dc_green

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_both_source_types_obey_coupled_maxwell_away_from_source():
    point = torch.tensor([.3, -.2, .4], device='cuda', dtype=torch.float64, requires_grad=True)
    j = point.new_tensor([.2, -.7, .4]); k = point.new_tensor([-.3, .1, .8])
    def fields(x):
        ee, ek, hj, hk = conductive_dc_green(x, sigma_e=2.3, sigma_m=.7)
        return torch.cat((ee@j+ek@k, hj@j+hk@k))
    values = fields(point)
    derivative = torch.autograd.functional.jacobian(fields, point)
    def curl(matrix):
        return torch.stack((matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]))
    torch.testing.assert_close(curl(derivative[:3]), -.7*values[3:], rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(curl(derivative[3:]), 2.3*values[:3], rtol=2e-13, atol=2e-14)
    assert abs(float(derivative[:3].trace())) < 1e-13
    assert abs(float(derivative[3:].trace())) < 1e-13


def test_dc_is_limit_of_full_positive_frequency_maxwell_green():
    from witwin.maxwell.constants import EPSILON_0, MU_0
    point = torch.tensor([[.3, -.2, .4], [.15, .25, -.3]], device='cuda', dtype=torch.float64)
    radius = point.norm(dim=-1); unit = point/radius[:, None]
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    dc = conductive_dc_green(point, sigma_e=.02, sigma_m=80.)
    errors = []
    for omega in (1e5, 1e3, 10.):
        epsilon = 3*EPSILON_0+1j*.02/omega
        mu = 2*MU_0+1j*80./omega
        wave = torch.sqrt(point.new_tensor(omega**2*epsilon*mu, dtype=torch.complex128))
        green = torch.exp(1j*wave*radius)/(4*math.pi*radius)
        first_over_r = green*(1j*wave-1/radius)/radius
        second = green*(-wave**2-2j*wave/radius+2/radius**2)
        hessian = first_over_r[:, None, None]*eye+(second-first_over_r)[:, None, None]*unit[:, :, None]*unit[:, None, :]
        dyadic = green[:, None, None]*eye+hessian/wave**2
        actual = (1j*omega*mu*dyadic, 1j*omega*epsilon*dyadic)
        errors.append(max(float((value-target).norm()/target.norm()) for value, target in zip(actual, (dc[0], dc[3]))))
    assert errors[1] < errors[0]/90 and errors[2] < errors[1]/90
    assert errors[-1] < 1e-6


def test_material_gradients_and_reciprocal_displacement():
    x = torch.tensor([.3, .1, .2], device='cuda', dtype=torch.float64)
    a = x.new_tensor(2., requires_grad=True); b = x.new_tensor(3., requires_grad=True)
    assert torch.autograd.gradcheck(lambda e, m: conductive_dc_green(x, sigma_e=e, sigma_m=m), (a,b))
    plus = conductive_dc_green(x, sigma_e=a, sigma_m=b)
    minus = conductive_dc_green(-x, sigma_e=a, sigma_m=b)
    for index in (0,3): torch.testing.assert_close(plus[index], minus[index].T, rtol=0, atol=0)
    torch.testing.assert_close(plus[1], -minus[2].T, rtol=0, atol=0)


@pytest.mark.parametrize('electric,magnetic', [(0.,1.), (-1.,1.), (1.,float('inf')), (1e-300,1e-300), (1e300,1e300)])
def test_singular_or_unresolved_channels_reject(electric, magnetic):
    x = torch.ones(3, device='cuda', dtype=torch.float64)
    with pytest.raises(ValueError): conductive_dc_green(x, sigma_e=electric, sigma_m=magnetic)
    with pytest.raises(ValueError, match='coincident'):
        conductive_dc_green(x*0, sigma_e=1., sigma_m=1.)


def test_volume_self_contact_and_current_gradients():
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    from witwin.maxwell.mom.vie_operator import cell_green_potentials
    from witwin.maxwell.mom.conductive_static import conductive_dc_cell_fields
    vertices = torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]], device='cuda', dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices, torch.tensor([[0,1,2,3]], device='cuda'), region_id=1)
    points = torch.stack((vertices.mean(0), vertices.new_tensor([.2,.15,.3])))
    scalar, gradient, hessian = cell_green_potentials(volume, points, 1j*math.sqrt(6), order=10)
    trace = hessian.diagonal(dim1=-2, dim2=-1).sum(-1)
    target = 6*scalar-torch.tensor([[1.],[0.]],device='cuda')
    torch.testing.assert_close(trace, target, rtol=1e-8, atol=1e-9)
    j = vertices.new_tensor([[.3,.2,-.1]], requires_grad=True)
    k = vertices.new_tensor([[.1,-.2,.4]], requires_grad=True)
    e, h = conductive_dc_cell_fields(volume,j,k,points,sigma_e=2.,sigma_m=3.,order=10)
    assert bool(torch.isfinite(e).all() & torch.isfinite(h).all())
    gj,gk = torch.autograd.grad(e.square().sum()+h.square().sum(), (j,k))
    assert bool(torch.isfinite(gj).all() & torch.isfinite(gk).all()) and float(gj.norm()*gk.norm()) > 0
    # The far observer sees the integrated point-current limit as the volume shrinks.
    small = TetrahedralVolumeMesh(vertices*1e-3, volume.cells, region_id=1)
    far = points[1:]
    a,b = conductive_dc_cell_fields(small,j,k,far,sigma_e=2.,sigma_m=3.,order=10)
    blocks = conductive_dc_green(far-small.vertices.mean(0),sigma_e=2.,sigma_m=3.)
    expected_e = (blocks[0]@j[0]+blocks[1]@k[0])*small.volumes.sum()
    expected_h = (blocks[2]@j[0]+blocks[3]@k[0])*small.volumes.sum()
    torch.testing.assert_close(a,expected_e,rtol=2e-5,atol=1e-16)
    torch.testing.assert_close(b,expected_h,rtol=2e-5,atol=1e-16)
