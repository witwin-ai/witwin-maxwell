"""Analytic carrier tensor RHS controls, without projecting the incident current."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.carrier import IncidentCarrier
from witwin.maxwell.mom.mesh import sphere_mesh, RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.tensor_medium import TensorMedium, scalar_medium_if_isotropic
from witwin.maxwell.mom.lifting import build_lifted_rhs
from witwin.maxwell.mom.anisotropic_lifting import build_tensor_lifted_rhs, _pair_rhs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def handle_for(epsilon):
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    return SimpleNamespace(basis=basis, exterior=exterior,
        interior=TensorMedium(3e8, epsilon*eye, eye),
        integration=IntegrationConfig(order=3, singular_order=4),
        incident_carrier=IncidentCarrier.compile([mw.PlaneWave()], exterior))


@pytest.mark.parametrize('epsilon', [1., 1.+1e-12, 2.])
def test_tensor_lifted_rhs_scalar_limit_keeps_factored_contrast(epsilon):
    handle = handle_for(epsilon)
    actual, receipt = build_tensor_lifted_rhs(handle)
    scalar_handle = SimpleNamespace(**{**vars(handle), 'interior': scalar_medium_if_isotropic(handle.interior)})
    expected, _ = build_lifted_rhs(scalar_handle)
    # CUDA index_add may change summation order across separate evaluations.
    error = torch.linalg.vector_norm(actual-expected)
    assert error <= 32*torch.finfo(actual.real.dtype).eps*torch.linalg.vector_norm(expected)
    assert receipt['incident_carrier_identity'] == handle.incident_carrier.identity
    assert receipt['tensor_specialization'] == 'exact_isotropic_factored_difference'
    assert not receipt['physical_qualification']
    if epsilon == 1.:
        assert torch.count_nonzero(actual) == 0
    else:
        assert torch.linalg.vector_norm(actual) > 0
