"""Independent modal reference phase, polarization and resource-metric checks."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from benchmark.mom_modal_convergence import radial_scattering_coefficient, reference_modal_far, _norm
from benchmark.sie_reference import sphere_coefficients
from witwin.maxwell.mom.physics import compile_medium


def test_modal_radial_transmission_matches_independent_plane_wave_coefficients():
    exterior=compile_medium(mw.Material(),3e8)
    interior=compile_medium(mw.Material(eps_r=2.5,sigma_e=.003),3e8)
    a,b=sphere_coefficients(exterior.k.real*.1,interior.epsilon_r,cutoff=5)
    for l in range(1,6):
        assert radial_scattering_coefficient(l,.1,exterior,interior,'TE') == pytest.approx(-b[l-1],rel=1e-12,abs=1e-20)
        assert radial_scattering_coefficient(l,.1,exterior,interior,'TM') == pytest.approx(-a[l-1],rel=1e-12,abs=1e-20)
    ordering=[(1,0,'TE'),(1,0,'TM')]
    far=reference_modal_far(ordering,np.array([[1.,0.,0.]]),.1,exterior,interior,[1.,1.])
    prefactor=math.sqrt(3/(8*math.pi))/exterior.k
    np.testing.assert_allclose(far[0,0],np.array([0.,-b[0]*prefactor,0.]),atol=1e-18,rtol=1e-12)
    np.testing.assert_allclose(far[0,1],np.array([0.,0.,-a[0]*prefactor]),atol=1e-18,rtol=1e-12)
    empty=reference_modal_far(ordering,np.array([[1.,0.,0.]]),.1,exterior,exterior,[1.,1.])
    np.testing.assert_array_equal(empty,0.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA metric required')
def test_modal_resource_metric_matches_explicit_surface_mass_weight():
    generator=torch.Generator(device='cuda').manual_seed(192)
    matrix=torch.randn(6,6,device='cuda',dtype=torch.complex128,generator=generator)
    mass=matrix@matrix.mH+torch.eye(6,device='cuda')
    eta=376.73
    handle=SimpleNamespace(basis=SimpleNamespace(n_edges=6),exterior=SimpleNamespace(eta=eta),
                           _riesz_factor=torch.linalg.cholesky(torch.block_diag(mass/eta**2,mass)))
    response=torch.randn(12,5,device='cuda',dtype=torch.complex128,generator=generator)
    explicit=torch.block_diag(eta*torch.linalg.cholesky(mass).mH,torch.linalg.cholesky(mass).mH)@response
    torch.testing.assert_close(_norm(response,handle),explicit,rtol=1e-12,atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA source projection required')
def test_plane_wave_projection_converges_on_independent_control_grid():
    from benchmark.mom_modal_convergence import plane_wave_projection
    report=plane_wave_projection([1,3,5,7,9])
    rows=report['rows']
    for field in ['relative_control_E_L2_error','relative_control_H_L2_error']:
        assert all(b[field]<a[field] for a,b in zip(rows,rows[1:]))
        assert rows[-1][field]<.0005
    assert max(row['coefficient_quadrature_relative_difference'] for row in rows)<1e-11
