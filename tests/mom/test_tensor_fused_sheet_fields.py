"""Combined sheet quadrature equivalence, principal values and current gradients."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('trace', [False, True])
def test_combined_sheet_matches_separate_integrals_and_gradient(trace, monkeypatch):
    import witwin.maxwell as mw
    from witwin.maxwell.mom import anisotropic_fields as fields
    from witwin.maxwell.mom.carrier import IncidentCarrier
    from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.tensor_medium import TensorMedium

    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    carrier = IncidentCarrier.compile([mw.PlaneWave()], exterior)
    handle = SimpleNamespace(basis=basis, exterior=exterior, incident_carrier=carrier)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    kernel = DyadicGreenKernel(TensorMedium(3e8, (2+.1j)*eye, (1.3+.2j)*eye))
    generator = torch.Generator(device='cuda').manual_seed(147)
    coefficients = (1e-3*torch.randn(2*basis.n_edges, device='cuda',
        dtype=torch.complex128, generator=generator)).requires_grad_()
    faces = torch.tensor([0, 3], device='cuda') if trace else None
    points = (basis.mesh.triangles[faces].mean(1) if trace else
        basis.mesh.vertices.new_tensor([[.01, .02, 0.], [.4, .3, .5]]))
    calls = []
    original = DyadicGreenKernel.evaluate_regular

    def measured(self, displacements):
        if self is kernel:
            calls.append(len(displacements))
        return original(self, displacements)

    monkeypatch.setattr(DyadicGreenKernel, 'evaluate_regular', measured)
    options = dict(order=3, pair_batch=4, surface_faces=faces)
    raw = fields.tensor_current_potentials(basis, coefficients, points, kernel, **options)
    incident = fields.tensor_incident_potentials(handle, points, kernel, **options)
    separate_nodes = sum(calls)
    separate_calls = len(calls)
    calls.clear()
    electric, magnetic, receipt = fields.tensor_lifted_potentials(
        handle, coefficients, points, kernel, **options)
    expected = (raw[0]+incident[0], raw[1]+incident[1])
    for actual, reference in zip((electric, magnetic), expected):
        torch.testing.assert_close(actual, reference, rtol=2e-11, atol=1e-11)
    actual_loss = electric.abs().square().sum()+magnetic.abs().square().sum()
    expected_loss = sum(field.abs().square().sum() for field in expected)
    actual_gradient, = torch.autograd.grad(actual_loss, coefficients)
    expected_gradient, = torch.autograd.grad(expected_loss, coefficients)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-11, atol=1e-9)
    assert 2*len(calls) == separate_calls
    assert 2*sum(calls) == separate_nodes
    assert receipt['shared_sheet_kernel'] is True
    assert receipt['principal_value'] is trace
    assert receipt['incident_carrier_identity'] == carrier.identity
    assert receipt['physical_qualification'] is False


@pytest.mark.parametrize('trace', [False, True])
@pytest.mark.parametrize('accept', [False, True])
def test_general_tensor_decoder_matches_separate_sheet_oracle(trace, accept, monkeypatch):
    """Manufactured regular kernel isolates composition; this is not physics."""
    import copy
    import witwin.maxwell as mw
    from witwin.maxwell.mom import anisotropic_fields as fields
    from witwin.maxwell.mom.carrier import IncidentCarrier
    from witwin.maxwell.mom.fields import FieldQueryConfig
    from witwin.maxwell.mom.lifting import incident_trace
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.tensor_medium import TensorMedium

    basis = RWGBasis(sphere_mesh(radius=.1, subdivisions=0, device='cuda'))
    exterior = compile_medium(mw.Material(), 3e8)
    carrier = IncidentCarrier.compile([mw.PlaneWave()], exterior)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e8, torch.diag(eye.new_tensor([2+.1j, 3+.1j, 4+.1j])), eye)

    def regular(displacements):
        scale = (1+displacements.square().sum(-1)).to(eye.dtype)
        value = scale[:, None, None]*torch.eye(6, device='cuda', dtype=eye.dtype)
        return value, dict(converged=True, static_diagonal_singularity_subtracted=True,
                           static_diagonal_restored=False)

    kernel = SimpleNamespace(medium=medium, evaluate_regular=regular)
    handle = SimpleNamespace(basis=basis, exterior=exterior, interior=medium,
        incident_carrier=carrier, tensor_kernel=kernel,
        integration=IntegrationConfig(order=1, singular_order=1, pair_batch=4))
    generator = torch.Generator(device='cuda').manual_seed(149)
    current = 1e-3*torch.randn(2*basis.n_edges, generator=generator,
                              device='cuda', dtype=eye.dtype)
    faces = torch.tensor([0], device='cuda') if trace else None
    points = basis.mesh.triangles[faces].mean(1) if trace else basis.mesh.vertices.new_tensor([[.01, .02, 0.]])
    side = 'interior' if trace else None
    policy = FieldQueryConfig(relative_tolerance=0.,
                              absolute_tolerance=1e8 if accept else 1e-30, max_order=2)
    if accept:
        actual = fields.decode_tensor_lifted(handle, current, points, side, query_config=policy)
    else:
        with pytest.raises(fields.TensorFieldConvergenceError):
            fields.decode_tensor_lifted(handle, current, points, side, query_config=policy)
    actual_receipt = copy.deepcopy(handle.last_field_query_log)

    # Reconstruct the previous two-integral equations, including both jumps,
    # independently of the fused helper and decoder's sign/jump composition.
    def separate(order):
        options = dict(order=order, pair_batch=4, surface_faces=faces)
        raw = fields.tensor_current_potentials(basis, current, points, kernel, **options)
        incident = fields.tensor_incident_potentials(handle, points, kernel, **options)
        values = [-raw[0]-incident[0], -raw[1]-incident[1]]
        if trace:
            raw_jump = fields.tensor_rwg_trace_jump(basis, current, points, faces, medium)
            j, m, dj, dm = incident_trace(handle, points[:, None], faces)
            carrier_jump = fields.tensor_sheet_jump(basis.mesh.normals[faces],
                j[:, 0], m[:, 0], dj[:, 0], dm[:, 0], medium)
            values = [v+.5*(r+c) for v, r, c in zip(values, raw_jump, carrier_jump)]
        return *values, dict(surface_order=order)

    if accept:
        expected = fields._adaptive_tensor_fields(separate, points, policy, 1)
        for value, reference in zip(actual, expected[:2]):
            torch.testing.assert_close(value, reference, rtol=2e-11, atol=1e-10)
        oracle_receipt = expected[2]
        assert actual_receipt['evaluations'][-1]['combined_sheet']['shared_sheet_kernel']
    else:
        with pytest.raises(fields.TensorFieldConvergenceError) as error:
            fields._adaptive_tensor_fields(separate, points, policy, 1)
        oracle_receipt = error.value.receipt
    assert actual_receipt['converged'] is oracle_receipt['converged'] is accept
    assert actual_receipt['final_order'] == oracle_receipt['final_order'] == 2
    assert actual_receipt['failure_reason'] == oracle_receipt['failure_reason']
