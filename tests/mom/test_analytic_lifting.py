"""Independent arithmetic and physical controls for analytic trace lifting."""
import math
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.sie_reference import sphere_far_field
from witwin.maxwell.mom.fields import far_field
from witwin.maxwell.mom.linalg import solve_reference, StoppingRule
from witwin.maxwell.mom.lifting import build_lifted_rhs, difference_kernels
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_difference_kernel_against_high_precision_oracle(dtype):
    exterior = SimpleNamespace(k=6.2+.1j)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    with mp.workdps(100):
        for radius in (1e-12, 1e-8, 1e-4, .1, 1.):
            for contrast in (0., 1e-12, 1e-6, .1):
                interior = SimpleNamespace(k=exterior.k*(1+contrast))
                delta = torch.tensor([[radius, 0., 0.]], device='cuda', dtype=dtype)
                _, _, green, gradient = difference_kernels(delta, exterior, interior)
                r = mp.mpf(float(delta[0, 0]))
                k0 = mp.mpc(exterior.k.real, exterior.k.imag)
                dk = interior.k-exterior.k
                ki = k0+mp.mpc(dk.real, dk.imag)
                expected_g = (mp.exp(1j*ki*r)-mp.exp(1j*k0*r))/(4*mp.pi*r)
                expected_grad = (mp.exp(1j*ki*r)*(1j*ki*r-1)-mp.exp(1j*k0*r)*(1j*k0*r-1))/(4*mp.pi*r*r)
                for actual, reference in ((green[0], expected_g), (gradient[0, 0], expected_grad)):
                    value, target = complex(actual), complex(reference)
                    assert abs(value-target) <= tolerance*max(abs(target), 1e-40)
                assert torch.equal(gradient[0, 1:], torch.zeros_like(gradient[0, 1:]))


@pytest.mark.parametrize('epsilon', [1., 1.001, 1.01])
def test_lifted_rhs_zero_identity_and_nonzero_weak_scattering(epsilon):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=epsilon))],
                     sources=[mw.PlaneWave()])
    op = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(subdivisions=1,
        riesz_storage='compact_rwg_v1', integration=IntegrationConfig(order=4, singular_order=6))).prepare().solver.handle
    rhs, metadata = build_lifted_rhs(op)
    y, log = solve_reference(op, rhs, StoppingRule(method='direct', rtol=1e-10))
    assert log['converged']
    assert metadata['rhs_near_quadrature']['pair_evaluations'] > 0
    assert metadata['rhs_near_quadrature']['maximum_accepted_relative_difference'] <= 1e-5
    directions = torch.tensor([[1., .3, .2], [-.2, 1., .4], [.3, -.5, 1.], [0., 0., 1.], [0., 0., -1.]],
                             device='cuda', dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    field = far_field(op, y, directions)[0]
    if epsilon == 1.:
        assert torch.count_nonzero(rhs) == torch.count_nonzero(y) == torch.count_nonzero(field) == 0
    else:
        reference = sphere_far_field(.1, op.exterior.k.real, complex(epsilon), directions.cpu().numpy())
        error = np.linalg.norm(field.cpu().numpy()-reference)/np.linalg.norm(reference)
        # Coarse polyhedral geometry has a measured 12.3% error here. This is a
        # sign/scaling regression, not the final 0.5% physical qualification.
        assert .11 < error < .14
        assert float(torch.linalg.vector_norm(op.apply(y)-rhs)/torch.linalg.vector_norm(rhs)) < 1e-10


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_material_weighted_difference_has_no_spurious_static_pole(dtype):
    from witwin.maxwell.mom.lifting import medium_differences, weighted_green_difference
    from witwin.maxwell.mom.physics import compile_medium
    exterior = compile_medium(mw.Material(), 3e8, background=True)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    with mp.workdps(100):
        for contrast in (0., 1e-12, 1e-6, .1):
            interior = compile_medium(mw.Material(eps_r=1+contrast), 3e8)
            de, dm, dk = medium_differences(exterior, interior)
            assert dm == 0
            for radius in (1e-12, .1):
                delta = torch.tensor([[radius, 0., 0.]], device='cuda', dtype=dtype)
                g, _, dg, _ = difference_kernels(delta, exterior, interior, delta_k=dk)
                electric, electric_div = weighted_green_difference(g, dg, interior.epsilon, de,
                    interior.mu, exterior.mu, dm, exterior.omega)
                magnetic, magnetic_div = weighted_green_difference(g, dg, interior.mu, dm,
                    interior.epsilon, exterior.epsilon, de, exterior.omega)
                # With equal mu, the divergence kernel is exactly a multiple
                # of delta G, without a material-independent 1/r contribution.
                torch.testing.assert_close(electric_div, dg/(exterior.omega**2*interior.mu), rtol=1e-6, atol=0)
                r = mp.mpf(float(delta[0, 0]))
                k0 = mp.mpc(exterior.k)
                ki = k0+mp.mpc(dk)
                ge, gi = mp.exp(1j*k0*r)/(4*mp.pi*r), mp.exp(1j*ki*r)/(4*mp.pi*r)
                ee = mp.mpc(exterior.epsilon)
                ei = ee+mp.mpc(de)
                mu = mp.mpc(exterior.mu)
                omega = mp.mpf(exterior.omega)
                expected = (ei*gi-ee*ge, (gi-ge)/(omega**2*mu),
                            mu*(gi-ge), gi/(omega**2*ei)-ge/(omega**2*ee))
                for actual, reference in zip((electric, electric_div, magnetic, magnetic_div), expected):
                    target = complex(reference)
                    assert abs(complex(actual[0])-target) <= tolerance*max(abs(target), 1e-40)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_rhs_multiple_directions_phases_and_exact_source_cancellation(dtype):
    from witwin.maxwell.mom.runtime import compile_surface, DecodeHandle
    frequency = 3e8
    a = mw.PlaneWave(source_time=mw.CW(frequency=frequency, phase=.37))
    b = mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1),
                     source_time=mw.CW(frequency=frequency, phase=-.51))
    opposite = mw.PlaneWave(source_time=mw.CW(frequency=frequency, phase=.37, amplitude=-1.))
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=1.01))], sources=[a])
    config = mw.MoMConfig(subdivisions=0, dtype=dtype,
                          integration=IntegrationConfig(order=4, singular_order=6, near_tolerance=2e-5))
    basis, exterior, interior = compile_surface(scene, frequency, config)
    def evaluate(sources):
        return build_lifted_rhs(DecodeHandle(basis, exterior, interior, tuple(sources), config.integration, {}))[0]
    first, second, summed = evaluate([a]), evaluate([b]), evaluate([a, b])
    tolerance = 5e-5 if dtype == torch.complex64 else 5e-7
    assert summed.dtype == dtype and summed.is_cuda
    assert float(torch.linalg.vector_norm(summed-first-second)/torch.linalg.vector_norm(first+second)) < tolerance
    assert torch.count_nonzero(evaluate([a, opposite])) == 0


@pytest.mark.parametrize('epsilon', [1.001, 2.25])
def test_complete_rhs_order_convergence_is_independent_of_pair_estimator(epsilon):
    from witwin.maxwell.mom.runtime import compile_surface, DecodeHandle
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=epsilon))],
                     sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8, phase=.37))])
    config = mw.MoMConfig(subdivisions=0)
    basis, exterior, interior = compile_surface(scene, 3e8, config)
    values = []
    for order, singular in ((4, 6), (6, 8)):
        integration = IntegrationConfig(order=order, singular_order=singular)
        rhs, info = build_lifted_rhs(DecodeHandle(basis, exterior, interior, tuple(scene.sources), integration, {}))
        assert info['global_rhs_error_certified'] is False
        values.append(rhs)
    # This independent whole-vector order control includes accumulation and
    # touching/far terms omitted from the adaptive per-pair near estimator.
    scale = torch.cat((torch.full((basis.n_edges,), abs(exterior.eta), device=basis.mesh.device),
                       torch.ones(basis.n_edges, device=basis.mesh.device))).to(torch.float64)
    error = torch.linalg.vector_norm((values[1]-values[0])*scale)/torch.linalg.vector_norm(values[1]*scale)
    assert float(error) < 1e-4
