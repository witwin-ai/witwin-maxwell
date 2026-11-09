"""Physical weak-action and independent two-sign RWG carrier controls."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import PreparedRWGGeometry, IntegrationConfig
from witwin.maxwell.mom.quadrature import triangle_rule
from witwin.maxwell.mom.fmm_expansion import multi_indices
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalRWGProjection, DirectionalFFTElectromagneticM2L
from witwin.maxwell.mom.nearfield import scalar_regional_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _basis(center, radius=.02):
    return RWGBasis(sphere_mesh(radius=radius, center=center, subdivisions=0,
                               device='cuda', dtype=torch.float64))


def _independent_edge_moments(basis, edges, center, direction, wave, order, quadrature):
    geometry = PreparedRWGGeometry(basis)
    powers = torch.tensor(multi_indices(order), device='cuda')
    values = []
    for e in edges.tolist():
        total = torch.zeros((len(powers), 4), device='cuda', dtype=torch.complex128)
        for face in (int(basis.plus_faces[e]), int(basis.minus_faces[e])):
            triangle = geometry.triangles[face:face+1]
            points, weights = triangle_rule(triangle, quadrature)
            points, weights = points[0], weights[0]
            position = int((basis.face_edges[face] == e).nonzero()[0])
            slope = geometry.slopes[face, position]
            rwg = slope*(points-triangle[0, position])
            divergence = torch.ones_like(points[:, :1])*2*slope
            poly = (points-center)[:, None].pow(powers).prod(-1)
            phase = torch.exp(1j*wave*((points-center)*direction).sum(-1))
            channels = torch.cat((rwg, divergence), dim=-1).to(torch.complex128)
            total += torch.einsum('qc,qa,q->ac', channels, poly.to(torch.complex128), weights*phase)
        values.append(total)
    return torch.stack(values)


def test_independent_lossy_source_receiver_maps_complete_support_order_and_adjoints(tmp_path):
    basis = _basis((.3, -.2, .1), .1)
    center = basis.mesh.vertices.new_tensor([.29, -.19, .12])
    direction = center.new_tensor([.7, -.2, .3]); direction = direction/direction.norm()
    edges = torch.tensor([7, 2, 11], device='cuda')
    wave, order, q = 5.+.8j, 3, 8
    projection = DirectionalRWGProjection(basis, edges, center, direction, wave, order, quadrature_order=q)
    source = _independent_edge_moments(basis, edges, center, direction, -wave, order, q)
    receiver = _independent_edge_moments(basis, edges, center, direction, wave, order, q)
    actual_source = torch.cat((projection.source_vector, projection.source_divergence[..., None]), -1)
    actual_receiver = torch.cat((projection.receiver_vector, projection.receiver_divergence[..., None]), -1)
    torch.testing.assert_close(actual_source, source, rtol=2e-12, atol=2e-16)
    torch.testing.assert_close(actual_receiver, receiver, rtol=2e-12, atol=2e-16)
    difference = float((receiver-source.conj()).norm()/receiver.norm())
    assert difference > .01
    generator = torch.Generator(device='cuda').manual_seed(1261)
    currents = torch.randn((3, 5, len(edges)), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    coefficients = torch.randn((*currents.shape[:-1], projection.count, 4), device='cuda', dtype=torch.complex128, generator=generator).requires_grad_()
    moments = projection.source_moments(currents)
    evaluated = projection.receiver_evaluate(coefficients)
    torch.testing.assert_close(moments, torch.einsum('...e,eac->...ac', currents, source))
    torch.testing.assert_close(evaluated, torch.einsum('...ac,eac->...e', coefficients, receiver))
    dual_m = torch.randn_like(moments); dual_e = torch.randn_like(evaluated)
    grad_s, = torch.autograd.grad(moments, currents, dual_m)
    grad_r, = torch.autograd.grad(evaluated, coefficients, dual_e)
    torch.testing.assert_close(grad_s, projection.source_adjoint(dual_m), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(grad_r, projection.receiver_adjoint(dual_e), rtol=1e-12, atol=1e-14)
    record = dict(receiver_vs_conjugate_source=difference, selected_edge_order=edges.tolist(),
                  complete_plus_minus_support=True, receipt=projection.receipt(), physical_qualification=False)
    (tmp_path/'projection.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


@pytest.mark.parametrize('lossy', (False, True))
def test_directional_em_actual_original_weak_all_columns_adjoint_and_current_gradient(lossy, tmp_path):
    test = _basis((-.5, .1, 0.)); source = _basis((.5, -.1, .05))
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3, sigma_e=.01 if lossy else 0., sigma_m=.02 if lossy else 0.), 3e8)
    tc, sc = test.mesh.vertices.mean(0), source.mesh.vertices.mean(0)
    displacement = tc-sc
    direction = displacement/displacement.norm()+displacement.new_tensor([.003, -.002, .001])
    direction = direction/direction.norm()
    edges = torch.arange(test.n_edges, device='cuda'); q = 8; order = 6
    target = DirectionalRWGProjection(test, edges, tc, direction, medium.k, order, quadrature_order=q)
    origin = DirectionalRWGProjection(source, edges, sc, direction, medium.k, order, quadrature_order=q)
    op = DirectionalFFTElectromagneticM2L(displacement, direction, medium, displacement.new_tensor(.02), order)
    n = len(edges)
    def action(value, adjoint=False):
        currents = torch.stack((value[..., :n], value[..., n:]), dim=-2)
        if adjoint:
            moment_dual = target.receiver_adjoint(currents)
            projected = origin.source_adjoint(op.adjoint(moment_dual))
        else:
            moments = origin.source_moments(currents)
            projected = target.receiver_evaluate(op.apply(moments))
        return torch.cat((projected[..., 0, :], projected[..., 1, :]), dim=-1)
    eye = torch.eye(2*n, device='cuda', dtype=torch.complex128)
    actual = action(eye).mT
    reference, _ = scalar_regional_block(test, source, edges, edges, (medium,), IntegrationConfig(order=q, singular_order=3))
    eta = abs(medium.eta)
    left = actual.real.new_tensor([eta]*n+[1.]*n); right = actual.real.new_tensor([1.]*n+[eta]*n)
    error = float((left[:, None]*(actual-reference)*right).norm()/(left[:, None]*reference*right).norm())
    generator = torch.Generator(device='cuda').manual_seed(1262)
    currents = torch.randn((3, 5, 2*n), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    dual = torch.randn_like(currents)
    applied, adjoint = action(currents), action(dual, True)
    original_adjoint = dual@reference.conj()
    # Independent original kernel adjoint keeps the physical 1e-8 gate.
    original_adjoint_error = float((adjoint-original_adjoint).norm()/original_adjoint.norm())
    dense_adjoint = dual@actual.conj()
    adjoint_error = float((adjoint-dense_adjoint).norm()/dense_adjoint.norm())
    derivative, = torch.autograd.grad(applied, currents, dual)
    derivative_error = float((derivative-adjoint).norm()/adjoint.norm())
    lhs, rhs = (applied.conj()*dual).sum(), (currents.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach())
    record = dict(lossy=lossy, k=str(medium.k), balanced_original_weak_all_column_error=error,
        original_physical_adjoint_error=original_adjoint_error, explicit_adjoint_error=adjoint_error,
        current_gradient_error=derivative_error, dot=dot, edges=n, quadrature=q,
        projection_receiver_not_source_conjugate=lossy, receipt=op.receipt(),
        physical_qualification=False, high_frequency_directional_fmm=False)
    (tmp_path/'weak.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert error < 1e-8 and original_adjoint_error < 1e-8
    assert adjoint_error < 1e-11 and derivative_error < 1e-11 and dot < 1e-11


def test_directional_projection_owned_inputs_budget_and_mutation_guards(monkeypatch):
    import witwin.maxwell.mom.fmm_directional_rwg as module
    basis = _basis((0., 0., 0.)); center = basis.mesh.vertices.mean(0)
    direction = center.new_tensor([1., 0., 0.]); edges = torch.tensor([7, 2, 11], device='cuda')
    original = module._rwg_weighted_moments
    def forbidden(*args, **kwargs): raise AssertionError('Rejected projection reached integration')
    monkeypatch.setattr(module, '_rwg_weighted_moments', forbidden)
    for changed, exception in [({'direction':direction*2}, ValueError), ({'wave_number':5.-1j}, ValueError),
            ({'quadrature_order':1}, ValueError), ({'retained_budget_bytes':1}, MemoryError),
            ({'tensor_workspace_budget_bytes':1}, MemoryError)]:
        kw = dict(basis=basis, edges=edges, center=center, direction=direction, wave_number=5., order=3, quadrature_order=4)
        kw.update(changed)
        with pytest.raises(exception): DirectionalRWGProjection(**kw)
    monkeypatch.setattr(module, '_rwg_weighted_moments', original)
    objects = [DirectionalRWGProjection(basis, edges, center, direction, 5.+.2j, 1, quadrature_order=3) for _ in range(7)]
    edges.add_(1); center.add_(1); direction.mul_(2)
    for obj in objects:
        obj.assert_current()
        assert all(t._base is None and t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in obj.tensors())
    for i, obj in enumerate(objects):
        obj.tensors()[i].add_(1)
        with pytest.raises(ValueError, match='changed'):obj.source_moments(torch.ones(len(obj.edges), device='cuda', dtype=torch.complex128))
