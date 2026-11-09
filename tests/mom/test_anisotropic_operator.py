"""Weak tensor block conventions against independent polynomial/Coulomb integrals."""
import math
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.anisotropic_operator import weak_dyadic_pairs
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


class ManufacturedRegularKernel:
    """A constant dyadic tests weak assembly algebra, not a physical Green oracle."""
    def __init__(self):
        eye = torch.eye(3, device='cuda', dtype=torch.complex128)
        self.medium = TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye)
        self.value = torch.arange(36, device='cuda', dtype=torch.float64).reshape(6, 6).to(torch.complex128)/1000
        self.value = self.value+1j*self.value.mT

    def evaluate_regular(self, points):
        return self.value.expand(len(points), 6, 6), {'converged': True, 'oracle': 'manufactured_constant_dyadic',
            'static_diagonal_restored': False, 'static_diagonal_singularity_subtracted': True}


def moments(basis, face):
    mesh = basis.mesh
    signs = basis.face_signs[face]
    lengths = basis.edge_lengths[basis.face_edges[face]]
    return (signs*lengths/2)[:, None]*(mesh.centroids[face]-mesh.triangles[face])


def test_weak_tensor_row_signs_and_static_divergences_on_separate_bodies():
    test = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    source = RWGBasis(sphere_mesh(radius=.08, center=(.6, 0., 0.), subdivisions=0, device='cuda'))
    indices = torch.tensor([0], device='cuda')
    kernel = ManufacturedRegularKernel()
    actual, receipt = weak_dyadic_pairs(test, source, indices, indices, kernel, category=4, order=4)
    f, h = moments(test, 0).to(kernel.value.dtype), moments(source, 0).to(kernel.value.dtype)
    expected = torch.empty_like(actual)
    for row, field, sign in ((0, 3, 1), (1, 0, -1)):
        for column, excitation in ((0, 0), (1, 3)):
            expected[0, row, column] = sign*f@kernel.value[field:field+3, excitation:excitation+3]@h.mT
    # A distinct high-order scalar Coulomb reference; no production static kernel.
    x, wx = triangle_rule(test.mesh.triangles[:1], 12)
    y, wy = triangle_rule(source.mesh.triangles[:1], 12)
    integral = (wx[:, :, None]*wy[:, None]/torch.linalg.vector_norm(x[:, :, None]-y[:, None], dim=-1)).sum()/(4*math.pi)
    div_test = test.face_signs[0]*test.edge_lengths[test.face_edges[0]]/test.mesh.areas[0]
    div_source = source.face_signs[0]*source.edge_lengths[source.face_edges[0]]/source.mesh.areas[0]
    product = div_test[:, None]*div_source[None, :]*integral
    expected[0, 0, 1] -= 1j*product/(kernel.medium.omega*MU_0*(1.3+.2j))
    expected[0, 1, 0] += 1j*product/(kernel.medium.omega*EPSILON_0*(2+.1j))
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=1e-12)
    assert receipt['test_source_bases_distinct'] and receipt['static_derivatives_integrated_by_parts']
    assert receipt['panel_error_certified'] is False
    with pytest.raises(ValueError, match='Cross-body'):
        weak_dyadic_pairs(test, source, indices, indices, kernel, category=0, order=3)


def test_self_panel_does_not_inherit_scalar_zero_cross_block_assumption():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    indices = torch.tensor([0], device='cuda')
    kernel = ManufacturedRegularKernel()
    actual, _ = weak_dyadic_pairs(basis, basis, indices, indices, kernel, category=0, order=6)
    f = moments(basis, 0).to(kernel.value.dtype)
    expected = f@kernel.value[3:, :3]@f.mT
    assert torch.linalg.vector_norm(expected) > 0
    torch.testing.assert_close(actual[0, 0, 0], expected, rtol=1e-10, atol=1e-16)


def test_weak_kernel_rejects_double_counted_static_terms():
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    kernel = ManufacturedRegularKernel()
    original = kernel.evaluate_regular
    def mislabeled(points):
        values, receipt = original(points)
        receipt['static_diagonal_restored'] = True
        return values, receipt
    kernel.evaluate_regular = mislabeled
    rows = torch.tensor([0], device='cuda')
    cols = torch.tensor([10], device='cuda')
    with pytest.raises(ValueError, match='counted twice'):
        weak_dyadic_pairs(basis, basis, rows, cols, kernel, category=4, order=2)


def test_isotropic_tensor_region_uses_identical_scalar_weak_operator():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, assemble_region_weak
    from witwin.maxwell.mom.anisotropic_operator import assemble_tensor_region_weak
    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye)
    from witwin.core import PhysicalMaterial
    scalar = compile_medium(PhysicalMaterial(eps_r=2+.1j, mu_r=1.3+.2j), 3e8)
    exterior = compile_medium(mw.Material(), 3e8)
    config = IntegrationConfig(order=3, singular_order=4)
    actual, receipt = assemble_tensor_region_weak(basis, medium, config)
    expected, _ = assemble_region_weak(basis, scalar, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    outside, _ = assemble_region_weak(basis, exterior, config)
    complete, _ = assemble_weak(basis, exterior, scalar, config)
    torch.testing.assert_close(actual+outside, complete, rtol=1e-12, atol=1e-13)
    assert receipt['physical_qualification'] is False
    assert receipt['kernel_specialization'] == 'exact_isotropic_scalar_weak'


def test_multibody_tensor_regions_reduce_to_scalar_global_system():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
    from witwin.maxwell.mom.anisotropic_operator import assemble_generalized_system
    a = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    b = RWGBasis(sphere_mesh(radius=.08, center=(.6, 0., 0.), subdivisions=0, device='cuda'))
    ext, interior = [compile_medium(mw.Material(eps_r=e), 3e8) for e in (1., 2.)]
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    tensor = TensorMedium(3e8, 2*eye, eye)
    mixed = SurfaceSystem.create([(0, 'tensor', a, tensor), (2, 'scalar', b, interior)])
    scalar = SurfaceSystem.create([(0, 'tensor', a, interior), (2, 'scalar', b, interior)])
    config = IntegrationConfig(order=3, singular_order=4)
    actual, receipt = assemble_generalized_system(mixed, ext, config)
    expected, _ = assemble_system(scalar, ext, config)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    assert receipt['body_count'] == 2 and receipt['physical_qualification'] is False
    for record in receipt['blocks']:
        if record['test_region'] != record['source_region']:
            assert record['contribution'] == 'exterior_only'


def test_anisotropic_cross_region_obeys_reciprocal_current_exchange():
    import runpy
    from witwin.maxwell.mom.anisotropic_operator import assemble_tensor_region_weak
    from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel, DyadicIntegrationConfig
    from witwin.maxwell.mom.operator import IntegrationConfig
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    a, b = [body.basis for body in handle.system.bodies]
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    epsilon = eye.new_tensor([[2., .2, .1], [.2, 3., .15], [.1, .15, 4.]])+2j*eye
    mu = eye.new_tensor([[1.1, .1, 0.], [.1, 1.3, .05], [0., .05, 1.6]])+2j*eye
    medium = TensorMedium(handle.exterior.frequency, epsilon, mu)
    kernel = DyadicGreenKernel(medium, DyadicIntegrationConfig(64, 16, 256, 64,
        relative_tolerance=1e-4, absolute_tolerance=1e-9))
    config = IntegrationConfig(order=1, singular_order=1)
    forward, receipt = assemble_tensor_region_weak(a, medium, config, kernel=kernel, source_basis=b)
    reverse, _ = assemble_tensor_region_weak(b, medium, config, kernel=kernel, source_basis=a)
    # Lorentz reciprocity exchanges electric/magnetic current testing rows.
    perm_a = torch.cat((torch.arange(a.n_edges, 2*a.n_edges, device='cuda'),
                        torch.arange(a.n_edges, device='cuda')))
    perm_b = torch.cat((torch.arange(b.n_edges, 2*b.n_edges, device='cuda'),
                        torch.arange(b.n_edges, device='cuda')))
    expected = reverse.mT[perm_a][:, perm_b]
    from witwin.maxwell.constants import ETA_0
    scale_a = forward.real.new_tensor([1.]*a.n_edges+[ETA_0]*a.n_edges)
    scale_b = forward.real.new_tensor([1.]*b.n_edges+[ETA_0]*b.n_edges)
    error = ((forward-expected)*scale_b[None, :]/scale_a[:, None]).norm()
    norm = (forward*scale_b[None, :]/scale_a[:, None]).norm()
    assert error <= 3e-4*norm+1e-14
    assert receipt['kernel_specialization'] == 'general_tensor_weyl'
    assert receipt['contribution'] == 'exterior_only'
    assert not receipt['physical_qualification']
    print({'tensor_cross_reciprocity_relative_difference': float(error/norm)})
