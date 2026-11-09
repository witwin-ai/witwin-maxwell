"""Tensor incident modes, Bloch sectors and exact analytic sheet divergences."""
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.tensor_carrier import TensorIncidentCarrier
from witwin.maxwell.mom.dyadic_green import compile_plane_modes
from witwin.maxwell.mom.periodic import LatticeSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def mode():
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon = eye.new_tensor([[2., .2, .1], [.2, 3., .15], [.1, .15, 4.]])+.2j*eye
    mu = eye.new_tensor([[1.1, .1, 0.], [.1, 1.3, .05], [0., .05, 1.6]])+.1j*eye
    medium = TensorMedium(3e8, epsilon, mu)
    modes = compile_plane_modes(eye.real.new_tensor([[.2, .1]]), medium)
    residue = modes.residues[0, 0]
    column = residue.norm(dim=0).argmax()
    fields = residue[:, column]
    fields = fields/fields[:3].norm()
    q = torch.cat((modes.transverse[0].to(eye.dtype), modes.longitudinal[0, :1]))[None]
    return medium, q, fields[:3][None], fields[3:][None]


def test_actual_anisotropic_carrier_bloch_and_exact_surface_divergences():
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.lifting import incident_trace
    medium, q, e, h = mode()
    carrier = TensorIncidentCarrier(medium, q, e, h)
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    faces = torch.tensor([0, 1], device='cuda')
    points = basis.mesh.centroids[faces, None]
    handle = SimpleNamespace(basis=basis, exterior=medium, incident_carrier=carrier)
    j, m, dj, dm = incident_trace(handle, points, faces)
    torch.testing.assert_close(dj, 1j*(j*q).sum(-1), rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(dm, 1j*(m*q).sum(-1), rtol=1e-10, atol=1e-13)
    lattice = LatticeSpec(points.new_tensor([[1., 0., 0.]]), points.new_tensor([float(q[0, 0].real), 0., 0.]))
    carrier.validate_lattice(lattice)
    actual = carrier.evaluate(points+lattice.vectors[0], medium)
    reference = carrier.evaluate(points, medium)
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected*torch.exp(1j*lattice.bloch[0]), rtol=1e-12, atol=1e-13)
    with pytest.raises(ValueError, match='Bloch sector'):
        carrier.validate_lattice(LatticeSpec(points.new_tensor([[0., 0., 1.]]), points.new_zeros(3)))
    with pytest.raises(ValueError, match='Maxwell curl'):
        TensorIncidentCarrier(medium, q, e, .9*h)
    identity = carrier.identity
    q.add_(1.)
    assert carrier.identity == identity
    carrier.electric.mul_(2.)
    with pytest.raises(ValueError, match='changed'):
        carrier.evaluate(points, medium)


def test_tensor_carrier_scalar_limit_and_native_amplitude_gradient():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.carrier import IncidentCarrier
    from witwin.maxwell.mom.physics import compile_medium
    scalar = compile_medium(mw.Material(eps_r=2., mu_r=1.3), 3e8)
    source = mw.PlaneWave()
    expected = IncidentCarrier.compile([source], scalar)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(scalar.frequency, scalar.epsilon_r*eye, scalar.mu_r*eye)
    direction = eye.new_tensor(source.direction)[None]
    e = eye.new_tensor(source.polarization)[None]
    h = torch.linalg.cross(direction, e)/scalar.eta
    amplitude = eye.new_tensor(1., requires_grad=True)
    carrier = TensorIncidentCarrier(medium, scalar.k*direction, amplitude*e, amplitude*h)
    points = eye.real.new_tensor([[.1, .2, .3], [-.4, .2, .1]])
    actual = carrier.evaluate(points, medium)
    reference = expected.evaluate(points, scalar)
    for value, target in zip(actual, reference):
        torch.testing.assert_close(value, target, rtol=1e-12, atol=1e-13)
    loss = sum(value.abs().square().sum() for value in actual)
    gradient, = torch.autograd.grad(loss, amplitude)
    torch.testing.assert_close(gradient, 2*loss.detach().to(gradient.dtype), rtol=1e-12, atol=1e-13)


def test_tensor_exterior_lifted_rhs_preserves_exact_zero_contrast():
    import runpy
    from witwin.maxwell.mom.anisotropic_lifting import build_tensor_lifted_rhs
    medium, q, e, h = mode()
    amplitude = e.new_tensor(1., requires_grad=True)
    carrier = TensorIncidentCarrier(medium, q, amplitude*e, amplitude*h)
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    handle = SimpleNamespace(basis=original.system.bodies[0].basis, exterior=medium,
        interior=TensorMedium(medium.frequency, medium.epsilon_r, medium.mu_r),
        integration=original.integration, incident_carrier=carrier)
    rhs, receipt = build_tensor_lifted_rhs(handle)
    assert torch.count_nonzero(rhs) == 0
    derivative, = torch.autograd.grad(rhs.real.sum(), amplitude)
    assert derivative == 0
    assert receipt['tensor_specialization'] == 'exact_zero_constitutive_contrast'
    assert receipt['incident_carrier_identity'] == carrier.identity
    assert not receipt['physical_qualification']


@pytest.mark.parametrize('regular_kind', ('image', 'spectral_1d', 'spectral_2d'))
def test_periodic_tensor_rhs_checks_sector_and_keeps_regional_zero_contrast(monkeypatch, regular_kind):
    import runpy
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
    from witwin.maxwell.mom.periodic_dyadic_operator import build_periodic_dyadic_rhs
    medium, q, e, h = mode()
    amplitude = e.new_tensor(1., requires_grad=True)
    carrier = TensorIncidentCarrier(medium, q, amplitude*e, amplitude*h)
    original, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    system = SurfaceSystem.create([(body.structure_index, body.name, body.basis, medium)
                                   for body in original.system.bodies])
    lattice = LatticeSpec(q.real.new_tensor([[1., 0., 0.]]), q.real.new_tensor([float(q[0, 0].real), 0., 0.]))
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel
    if regular_kind == 'spectral_2d':
        lattice = LatticeSpec(q.real.new_tensor([[1., 0., 0.], [0., 1., 0.]]),
                             q.real.new_tensor([float(q[0, 0].real), float(q[0, 1].real), 0.]))
    def reference_for(cell):
        if regular_kind == 'image':
            return LossyDyadicImageReference(cell, medium, 1)
        return QuasiperiodicDyadicRegularKernel(QuasiperiodicDyadicGreenKernel(cell, medium),
            8, .12, 3, 1e-4, 16,
            normal=q.real.new_tensor([0., 0., 1.]) if regular_kind == 'spectral_1d' else None)
    reference = reference_for(lattice)
    def forbidden(*args, **kwargs):
        pytest.fail('Analytic lifted RHS must not add exterior periodic image forcing')
    monkeypatch.setattr(type(reference), 'evaluate', forbidden)
    rhs, receipt = build_periodic_dyadic_rhs(system, carrier,
        reference=reference)
    assert rhs.shape == (2*system.total_edges,) and torch.count_nonzero(rhs) == 0
    derivative, = torch.autograd.grad(rhs.real.sum(), amplitude)
    assert derivative == 0
    assert len(receipt['regions']) == len(system.bodies)
    assert receipt['spectral_config']['relative_tolerance'] > 0
    with pytest.raises(TypeError, match='spectral_config'):
        build_periodic_dyadic_rhs(system, carrier, reference=reference,
                                 spectral_config={})
    assert not receipt['image_forcing_added'] and not receipt['interior_cross_body_terms']
    wrong = LatticeSpec(lattice.vectors, lattice.bloch+.1*lattice.vectors[0])
    with pytest.raises(ValueError, match='Bloch sector'):
        build_periodic_dyadic_rhs(system, carrier, reference=reference_for(wrong))


def test_nonzero_tensor_background_carrier_pair_has_small_contrast_linear_limit():
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.anisotropic_lifting import _pair_rhs
    from witwin.maxwell.mom.dyadic_contrast import DyadicContrastKernel
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    medium, q, e, h = mode()
    carrier = TensorIncidentCarrier(medium, q, e, h)
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    rows = torch.tensor([0], device='cuda')
    cols = torch.tensor([10], device='cuda')
    perturbation = medium.epsilon_r.new_tensor([[.3, .1, 0.], [.1, .2, .05], [0., .05, .1]])
    results = []
    for step in (1e-5, 2e-5):
        inside = TensorMedium(medium.frequency, medium.epsilon_r+step*perturbation, medium.mu_r)
        contrast = DyadicContrastKernel(medium, inside,
            DyadicIntegrationConfig(256, 32, 512, 64, relative_tolerance=1e-4, absolute_tolerance=1e-12))
        handle = SimpleNamespace(basis=basis, exterior=medium, interior=inside, incident_carrier=carrier)
        from witwin.maxwell.mom.dyadic_green import DyadicConvergenceError
        try:
            value, receipt = _pair_rhs(handle, rows, cols, 4, 1, contrast)
        except DyadicConvergenceError as error:
            print({'contrast_step': step, 'spectral_failure_receipt': error.receipt})
            raise
        assert torch.isfinite(value).all() and value.norm() > 0
        assert receipt['kernels'][0]['converged']
        results.append(value)
    from witwin.maxwell.constants import ETA_0
    balance = results[0].real.new_tensor([ETA_0, 1.])[None, :, None]
    error = ((results[1]-2*results[0])*balance).norm()/(results[1]*balance).norm()
    assert error < 1e-3
    print({'tensor_rhs_pair_small_contrast_relative_difference': float(error)})
