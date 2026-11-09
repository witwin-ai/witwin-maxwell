"""Eigenpairs and checked fallback for batched small CUDA spectral systems."""
import pytest
import torch
from witwin.maxwell.mom.spectral_linalg import batched_eig4
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype,tolerance', [(torch.complex128, 2e-12), (torch.complex64, 2e-4)])
def test_qr_eigenpairs_match_native_and_preserve_degenerate_fallback(dtype, tolerance):
    generator = torch.Generator(device='cuda').manual_seed(413)
    matrix = torch.randn(64, 4, 4, generator=generator, device='cuda', dtype=dtype)
    eye = torch.eye(4, device='cuda', dtype=dtype)
    matrix[0] = eye
    matrix[1] = torch.diag(eye.new_tensor([1, 1, -1, -1]))
    roots, vectors, receipt = batched_eig4(matrix)
    expected = torch.linalg.eigvals(matrix)
    distance = (roots[:, :, None]-expected[:, None, :]).abs()
    assert float(distance.amin(-1).amax()) < tolerance
    assert float(distance.amin(-2).amax()) < tolerance
    torch.testing.assert_close(matrix@vectors, vectors*roots[:, None], rtol=tolerance, atol=tolerance)
    assert 2 <= receipt['fallback_count'] < len(matrix)//2


def test_differentiable_eigenspaces_retain_native_autograd():
    matrix = torch.diag(torch.tensor([1.+.1j, 2.+.2j, 3.+.3j, 4.+.4j], device='cuda', dtype=torch.complex128))[None].requires_grad_()
    roots, _, receipt = batched_eig4(matrix)
    derivative, = torch.autograd.grad(roots.real.sum(), matrix)
    torch.testing.assert_close(derivative, torch.eye(4, device='cuda', dtype=matrix.dtype)[None])
    assert receipt['native_reason'] == 'autograd' and receipt['fallback_count'] == 1


def test_maxwell_outgoing_residues_match_native_over_transverse_scales(monkeypatch):
    import math
    from witwin.maxwell.constants import EPSILON_0, MU_0
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    from witwin.maxwell.mom.dyadic_green import compile_plane_modes
    from witwin.maxwell.mom import spectral_linalg
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon = eye.new_tensor([[2+.8j, .1, .2], [.1, 2.4+.8j, 0], [.2, 0, 3+.8j]])
    mu = eye.clone()
    mu[0, 1], mu[1, 0] = .2j, -.2j
    medium = TensorMedium(3e8, epsilon, mu, reciprocal=False)
    scales = torch.tensor([0., .1, .5, 1., 3., 10., 100., 1000.], device='cuda', dtype=torch.float64)
    transverse = scales[:, None]*scales.new_tensor([[.8, .6]])
    actual = compile_plane_modes(transverse, medium)
    def native(matrix):
        roots, vectors = torch.linalg.eig(matrix)
        return roots, vectors, {'version': 'native_reference'}
    monkeypatch.setattr(spectral_linalg, 'batched_eig4', native)
    expected = compile_plane_modes(transverse, medium)
    eta0 = math.sqrt(MU_0/EPSILON_0)
    balance = torch.diag(eye.new_tensor([1, 1, 1, eta0, eta0, eta0]))
    error = torch.linalg.matrix_norm(balance@(actual.spectrum(1e-4)-expected.spectrum(1e-4))@balance)
    norm = torch.linalg.matrix_norm(balance@expected.spectrum(1e-4)@balance)
    assert bool((error/norm < 1e-10).all()), error/norm
    assert actual.eigensolver['fallback_count'] < len(scales)


def test_scalar_high_transverse_modes_match_closed_form_weyl_dyadic():
    """Resolve the evanescent tail without a cancellation-induced native fallback."""
    import math
    from witwin.maxwell.constants import EPSILON_0, MU_0
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    from witwin.maxwell.mom.dyadic_green import compile_plane_modes, cross_matrix
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon, mu = 1+2j, 1+2j
    medium = TensorMedium(3e8, epsilon*eye, mu*eye)
    scales = torch.tensor([0., .1, 1., 10., 100., 1000., 3000., 10000.],
                          device='cuda', dtype=torch.float64)
    transverse = scales[:, None]*scales.new_tensor([[.8, .6]])
    actual = compile_plane_modes(transverse, medium, pole_relative_tolerance=1e-10)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    qz = k0*torch.sqrt(epsilon*mu-transverse.square().sum(-1))
    q = torch.cat((k0*transverse.to(eye.dtype), qz[:, None]), -1)
    projector = eye-q[:, :, None]*q[:, None, :]/(k0*k0*epsilon*mu)
    curl = cross_matrix(q)
    # Scalar outgoing Weyl Green function and its analytic spatial derivatives.
    expected = torch.cat((torch.cat((1j*medium.omega*MU_0*mu*projector, -1j*curl), -1),
                          torch.cat((1j*curl, 1j*medium.omega*EPSILON_0*epsilon*projector), -1)), -2)
    distance = 1e-5
    expected = expected*(1j/(2*qz)*torch.exp(1j*qz*distance))[:, None, None]
    eta0 = math.sqrt(MU_0/EPSILON_0)
    units = scales.new_tensor([1, 1, 1, eta0, eta0, eta0])
    error = torch.linalg.matrix_norm(units[:, None]*(actual.spectrum(distance)-expected)*units[None, :])
    norm = torch.linalg.matrix_norm(units[:, None]*expected*units[None, :])
    assert bool((error/norm < 1e-10).all()), error/norm
    assert actual.eigensolver['analytic_fallback_count'] == 0
    assert actual.eigensolver['fallback_count'] == 0


def test_near_degenerate_modes_skip_discarded_vector_repairs(monkeypatch):
    import math
    import witwin.maxwell.mom.dyadic_green as dg
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.float64))
    rotation=metric.new_tensor([[math.cos(.37),-math.sin(.37),0.],[math.sin(.37),math.cos(.37),0.],[0.,0.,1.]])
    medium=TensorMedium(3e7,(rotation@(2*metric)@rotation.T).to(torch.complex128),
        (rotation@(1.3*metric)@rotation.T).to(torch.complex128))
    q=metric.new_tensor([[0.,0.],[.2,.3],[1.8,.7],[4.,-3.]])
    def forbidden(*args,**kwargs):raise AssertionError('Do not repair vectors whose roots mandate fallback')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'svd',forbidden)
        actual=dg.compile_plane_modes(q,medium)
    assert actual.eigensolver['svd_vector_repairs']==0
    assert actual.eigensolver['fallback_count']==len(q)
    assert actual.eigensolver['fallback_causes_nonexclusive']['small_root_separation']==len(q)
    def native(generator,transverse,material):
        roots,vectors=torch.linalg.eig(generator)
        return roots,vectors,dict(version='native_control',fallback_count=len(generator))
    monkeypatch.setattr(dg,'_plane_eigensystem',native)
    expected=dg.compile_plane_modes(q,medium)
    torch.testing.assert_close(actual.spectrum(.01),expected.spectrum(.01),rtol=0,atol=0)
