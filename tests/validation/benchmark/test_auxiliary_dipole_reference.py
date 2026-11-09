"""Independent physical checks for the benchmark auxiliary-source reference."""
import math

import numpy as np
import pytest
import torch

from benchmark.mom_auxiliary_reference import (
    AuxiliaryConfig, ETA0, angular_rule, electric_dipole_kernels, solve_reference,
    weighted_relative,
)
from benchmark.sie_reference import sphere_fields, sphere_far_field

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Reference core requires CUDA")


@pytest.mark.parametrize("epsilon", (2.25+0j, 2.25+0.1j))
def test_electric_dipole_satisfies_both_maxwell_curls(epsilon):
    k = 6.3*np.sqrt(epsilon)
    eta = ETA0/np.sqrt(epsilon)
    source = torch.tensor([[.01,-.02,.03]],device="cuda",dtype=torch.float64)
    moment = torch.tensor([.7+.2j,-.1+.3j,.4-.2j],device="cuda",dtype=torch.complex128)
    point = torch.tensor([.21,.13,-.17],device="cuda",dtype=torch.float64)

    def fields(position):
        e,h = electric_dipole_kernels(position[None],source,k,eta)
        return torch.stack((e[0,0]@moment,h[0,0]@moment))

    value = fields(point)
    jac = torch.autograd.functional.jacobian(lambda p: torch.view_as_real(fields(p)),point)
    derivative = jac[:,:,0,:]+1j*jac[:,:,1,:]
    curl = lambda j: torch.stack((j[2,1]-j[1,2],j[0,2]-j[2,0],j[1,0]-j[0,1]))
    torch.testing.assert_close(curl(derivative[0]),1j*k*eta*value[1],rtol=2e-12,atol=1e-12)
    torch.testing.assert_close(curl(derivative[1]),-1j*k/eta*value[0],rtol=2e-12,atol=1e-12)


@pytest.mark.parametrize("epsilon", (2.25+0j, 2.25+0.1j))
def test_auxiliary_sphere_matches_independent_mie(epsilon):
    reference = solve_reference(axes=(.1,.1,.1),epsilon=epsilon,phase=.23,
        config=AuxiliaryConfig(order=10,inner_factor=.4))
    directions,weights = angular_rule(6,offset=.413)
    for factor in (.3,.7,1.15,1.5,3.):
        points = .1*factor*directions
        for kind in (('total',) if factor<1 else ('total','scattered')):
            actual = reference.fields(points,region='interior' if factor<1 else 'exterior',kind=kind)
            expected = sphere_fields(.1,reference.k0,epsilon,points.cpu().numpy(),eta=ETA0,field_kind=kind,cutoff=24)
            for a,b in zip(actual,expected):
                b = torch.tensor(b*np.exp(-.23j),device='cuda')
                assert weighted_relative(a,b,weights)<5e-4
    far = sphere_far_field(.1,reference.k0,epsilon,directions.cpu().numpy(),cutoff=24)*np.exp(-.23j)
    assert weighted_relative(reference.far_field(directions),torch.tensor(far,device='cuda'),weights)<5e-4
    holdout = reference.holdout()
    assert max(holdout[k] for k in ('tangential_E','tangential_H','normal_D','normal_B'))<5e-4


def test_auxiliary_no_contrast_reproduces_incident_field():
    reference = solve_reference(axes=(.1,.1,.1),epsilon=1,phase=.23,
        config=AuxiliaryConfig(order=10,inner_factor=.4))
    directions,_ = angular_rule(6,offset=.413)
    for factor in (.7,1.15):
        points = factor*.1*directions
        actual = reference.fields(points,region='interior' if factor<1 else 'exterior')
        expected = reference.incident(points)
        for a,b in zip(actual,expected):
            assert float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))<5e-5
    assert float(torch.linalg.vector_norm(reference.far_field(directions)))<1e-7


def test_reference_rejects_nonphysical_queries_and_cpu_solves():
    with pytest.raises(ValueError,match='CUDA'):
        solve_reference(device='cpu')
    reference = solve_reference(axes=(.1,.1,.1),config=AuxiliaryConfig(order=4))
    with pytest.raises(ValueError,match='declared'):
        reference.fields([[0,0,0]],region='exterior')


@pytest.mark.parametrize('config', (AuxiliaryConfig(outer_factor=math.inf),
    AuxiliaryConfig(inner_factor=math.nan), AuxiliaryConfig(collocation_order=0)))
def test_auxiliary_configuration_rejects_invalid_shells_and_collocation(config):
    with pytest.raises(ValueError):
        solve_reference(config=config)


def test_standalone_reference_measures_control_and_far_power_closure():
    from benchmark.mom_auxiliary_acceptance import BUDGET, independent_queries, measure

    row, _ = measure(AuxiliaryConfig(order=10, inner_factor=.4), 2.25+0j,
                     (.1, .1, .1), query_spec=independent_queries())
    assert set(row['independent_power_closure_relative']) == {'scattering_W', 'extinction_W'}
    assert all(value < BUDGET for value in row['independent_power_closure_relative'].values())
    assert row['matched_powers']['scattering_W'] > 0
