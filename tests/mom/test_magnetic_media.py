"""Passive magnetic branches, constitutive accounting and electric/magnetic duality."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.lifting import medium_differences
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.operator import IntegrationConfig


def test_magnetic_conduction_and_all_poles_are_sampled_once():
    poles = dict(mu_debye_poles=(mw.DebyePole(delta_eps=.7, tau=1e-10),),
                 mu_drude_poles=(mw.DrudePole(plasma_frequency=.2e9, gamma=.03e9),),
                 mu_lorentz_poles=(mw.LorentzPole(delta_eps=.3, resonance_frequency=2e9, gamma=.1e9),))
    material = mw.Material(eps_r=2., mu_r=1.5, sigma_e=.01, sigma_m=20., **poles)
    medium = compile_medium(material, 1e9)
    expected = 1.5+1j*20/(medium.omega*MU_0)+sum(p.susceptibility(medium.omega) for family in poles.values() for p in family)
    assert medium.mu_r == pytest.approx(expected, rel=1e-14)
    assert medium.epsilon_r == pytest.approx(2+1j*.01/(medium.omega*EPSILON_0), rel=1e-14)
    assert medium.k.imag > 0 and medium.eta.real > 0
    assert medium.metadata['constitutive_parameters']['sigma_m'] == 20.
    assert len(medium.metadata['constitutive_parameters']['magnetic_poles']) == 3
    with pytest.raises(ValueError, match='lossless'): compile_medium(material, 1e9, background=True)


@pytest.mark.parametrize('electric,magnetic', [(True, False), (False, True), (True, True)])
def test_negative_sampled_parameters_and_opposite_branch_difference(electric, magnetic):
    pole = mw.DrudePole(plasma_frequency=math.sqrt(2)*1e9, gamma=0.)
    material = mw.Material(drude_poles=(pole,) if electric else (), mu_drude_poles=(pole,) if magnetic else ())
    ext, medium = compile_medium(mw.Material(), 1e9), compile_medium(material, 1e9)
    assert medium.k.imag >= 0
    if electric and magnetic:
        assert medium.k.real < 0 and medium.eta.real > 0
        assert medium.k == pytest.approx(-ext.k, rel=2e-15)
    else: assert medium.k.imag > 0
    assert medium.k**2 == pytest.approx(medium.omega**2*medium.epsilon*medium.mu, rel=2e-14)
    de, dm, dk = medium_differences(ext, medium)
    assert dk == pytest.approx(medium.k-ext.k, rel=1e-14)
    assert all(math.isfinite(v) for z in (de, dm, dk) for v in (z.real, z.imag))


def test_magnetic_frequency_sweep_preserves_limiting_absorption_branch():
    pole = mw.DrudePole(plasma_frequency=1.5e9, gamma=.01e9)
    material = mw.Material(drude_poles=(pole,), mu_drude_poles=(pole,))
    samples = [compile_medium(material, (1+.005*i)*1e9) for i in range(201)]
    scale = 2*math.pi*1e9*math.sqrt(EPSILON_0*MU_0)
    for medium in samples:
        assert medium.k.imag >= 0 and medium.eta.real > 0
        assert medium.k/medium.omega == pytest.approx(math.sqrt(EPSILON_0*MU_0)*medium.epsilon_r, rel=1e-13)
    assert max(abs(b.k-a.k)/scale for a, b in zip(samples, samples[1:])) < .017
    assert samples[0].k.real < 0 < samples[-1].k.real


def test_nonzero_near_zero_and_exact_zero_are_distinguished():
    pole = mw.DrudePole(plasma_frequency=1e9, gamma=0.)
    material = mw.Material(mu_drude_poles=(pole,))
    with pytest.raises(ValueError, match='Exactly zero'): compile_medium(material, 1e9)
    for frequency in (1e9*(1-1e-8), 1e9*(1+1e-8)):
        medium = compile_medium(material, frequency)
        assert 0 < abs(medium.mu_r) < 3e-8 and abs(medium.k) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_public_magnetic_duality_of_solved_currents_and_fields(tmp_path):
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=4, singular_order=6),
                         query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    results = []
    for eps, mu, pol in [(2., 1., (1., 0., 0.)), (1., 2., (0., 1., 0.))]:
        scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.07), mw.Material(eps_r=eps, mu_r=mu))],
                         sources=[mw.PlaneWave(polarization=pol)])
        results.append(mw.Simulation.mom(scene, frequency=3e8, config=config).run())
    first, second = (r.surface_currents for r in results)
    eta = first.handle.exterior.eta
    torch.testing.assert_close(second.J_scattered, first.M_scattered/eta, rtol=2e-10, atol=1e-13)
    torch.testing.assert_close(second.M_scattered, -eta*first.J_scattered, rtol=2e-10, atol=1e-11)
    for region, xyz in [(0, (.2, .1, .15)), (1, (.01, 0., 0.))]:
        p = torch.tensor([xyz], device='cuda', dtype=torch.float64)
        e, h = first.query_fields(p, region, 'total')
        ed, hd = second.query_fields(p, region, 'total')
        torch.testing.assert_close(ed, eta*h, rtol=2e-10, atol=1e-11)
        torch.testing.assert_close(hd, -e/eta, rtol=2e-10, atol=1e-13)
    path = tmp_path/'magnetic.pt'
    results[1].save(path)
    restored = mw.Result.load(path, scene=results[1].scene)
    assert restored.surface_currents.handle.interior.mu_r == 2
    assert torch.equal(restored.surface_currents.scattered_coefficients, second.scattered_coefficients)


def test_exact_undamped_magnetic_pole_is_rejected_explicitly():
    pole = mw.LorentzPole(delta_eps=1., resonance_frequency=1e9, gamma=0.)
    with pytest.raises(ValueError, match='material pole'):
        compile_medium(mw.Material(mu_lorentz_poles=(pole,)), 1e9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_opposite_branch_public_solution_and_magnetic_volume_loss():
    pole = mw.DrudePole(plasma_frequency=math.sqrt(2)*3e8, gamma=.001e8)
    material = mw.Material(drude_poles=(pole,), mu_drude_poles=(pole,), sigma_m=.001)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.03), material)], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=4),
                         query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    result = mw.Simulation.mom(scene, frequency=3e8, config=config).run()
    data = result.surface_currents
    assert data.handle.interior.k.real < 0 and data.handle.interior.k.imag > 0
    assert torch.isfinite(data.scattered_coefficients).all()
    points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float64)
    weights = torch.tensor([1e-6], device='cuda', dtype=torch.float64)
    e, h = data.query_fields(points, 1, 'total')
    medium = data.handle.interior
    expected = .5*medium.omega*(medium.epsilon.imag*e.abs().square().sum()
                                +medium.mu.imag*h.abs().square().sum())*weights[0]
    power = data.absorbed_power(points, weights)
    assert power > 0
    torch.testing.assert_close(power, expected, rtol=1e-13, atol=0.)
