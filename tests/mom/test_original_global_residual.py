"""Complete original residuals use bounded native blocks and separate RHS gates."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator_error import measure_original_global_residual

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(scope='module')
def original():
    vertices = torch.tensor([[0., 0., 0.], [.03, 0., 0.], [0., .03, 0.],
                             [0., 0., .03]], device='cuda', dtype=torch.float64)
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device='cuda')
    bodies = []
    for i, x in enumerate((-.2, .2)):
        basis = RWGBasis(SurfaceMesh(vertices+vertices.new_tensor([x, 0., 0.]), faces))
        interior = compile_medium(mw.Material(eps_r=2.+i, mu_r=1.4+.2*i,
                                             sigma_e=.01, sigma_m=.02), 3e8)
        bodies.append((i, str(i), basis, interior))
    system = SurfaceSystem.create(bodies)
    exterior = compile_medium(mw.Material(), 3e8)
    config = IntegrationConfig(order=3, singular_order=3, pair_batch=4, far_pair_batch=8)
    provider = ScalarSystemBlocks(system, exterior, config)
    dense, _ = assemble_system(system, exterior, config)
    assert system.total_edges == 12 and dense.shape == (24, 24)
    generator = torch.Generator(device='cuda').manual_seed(1729)
    value = torch.randn((3, 5, 24), generator=generator, device='cuda',
                        dtype=torch.complex128).transpose(0, 1)
    return provider, dense, value


def test_all_rows_noncontiguous_fifteen_rhs_and_bounded_original_blocks(original, monkeypatch, tmp_path):
    provider, dense, value = original
    rhs = value@dense.mT
    calls = []
    primary = ScalarSystemBlocks.__call__
    def bounded(self, rows, columns):
        assert self is provider and 0 < len(rows) <= 2 and 0 < len(columns) <= 3
        calls.append((tuple(rows.cpu().tolist()), tuple(columns.cpu().tolist())))
        return primary(self, rows, columns)
    monkeypatch.setattr(ScalarSystemBlocks, '__call__', bounded)
    variable = value.detach().requires_grad_(True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor,
                                                 lambda tensor: tensor):
        receipt = measure_original_global_residual(provider, variable, rhs, edge_batch=2, source_batch=3)
    assert not value.is_contiguous() and variable.requires_grad and not saved
    assert len(calls) == 24
    expected = [(tuple(range(r, r+2)), tuple(range(c, c+3)))
                for r in range(0, 12, 2) for c in range(0, 12, 3)]
    assert calls == expected
    assert receipt['rows_measured'] == receipt['total_rows'] == 24
    assert receipt['rhs_count'] == 15 and receipt['all_original_rows_measured']
    assert receipt['all_rhs_passed']
    torch.testing.assert_close(torch.tensor(receipt['rhs_norm'], device='cuda', dtype=torch.float64),
                               torch.linalg.vector_norm(rhs, dim=-1), rtol=1e-12, atol=0)
    assert max(v for row in receipt['relative_residual'] for v in row) < 1e-10
    assert not any(receipt[key] for key in ('compressed_action_used', 'dense_global_matrix_constructed',
        'all_operator_coefficients_error_measured', 'quadrature_convergence_certified',
        'directed_rounding_certified', 'physical_qualification', 'complexity_qualification',
        'peak_memory_qualified'))
    (tmp_path/'residual.json').write_text(json.dumps(receipt, indent=2)+'\n')


def test_unsampled_row_and_small_rhs_cannot_hide_behind_large_rhs(original, tmp_path):
    provider, dense, value = original
    current = value[:2, 0].clone()
    current[0] *= 1e120
    current[1] *= 1e-120
    rhs = current@dense.mT
    rhs[1, 10] += torch.linalg.vector_norm(rhs[1])*.02
    receipt = measure_original_global_residual(provider, current, rhs, edge_batch=5, source_batch=7)
    assert receipt['passed_per_rhs'] == [True, False] and not receipt['all_rhs_passed']
    expected = torch.linalg.vector_norm(current@dense.mT-rhs, dim=-1)/torch.linalg.vector_norm(rhs, dim=-1)
    assert receipt['relative_residual'][1] == pytest.approx(float(expected[1]), rel=1e-10)
    assert receipt['relative_residual'][0] < 1e-10 and receipt['relative_residual'][1] > .01
    assert receipt['rows_measured'] == 24
    (tmp_path/'residual.json').write_text(json.dumps(receipt, indent=2)+'\n')


def test_zero_rhs_exact_zero_and_nonzero_original_action(original):
    provider, _, value = original
    current = torch.stack((torch.zeros_like(value[0, 0]), value[0, 0]))
    receipt = measure_original_global_residual(provider, current, torch.zeros_like(current),
                                             edge_batch=6, source_batch=6)
    assert receipt['passed_per_rhs'] == [True, False]
    assert receipt['relative_residual'] == [0., None]
    assert receipt['rhs_norm'] == [0., 0.] and receipt['threshold'] == [0., 0.]
    json.dumps(receipt, allow_nan=False)


def test_invalid_inputs_rejected_before_original_action(original, monkeypatch):
    provider, dense, value = original
    rhs = value@dense.mT
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid inputs reached original action')
    monkeypatch.setattr(provider, 'sampled_action', forbidden)
    invalid = [(value[..., :-1], rhs[..., :-1]), (value.cpu(), rhs.cpu()),
               (value.to(torch.complex64), rhs.to(torch.complex64)),
               (value[:0], rhs[:0]), (value, rhs[0]), (None, rhs)]
    nonfinite = value.clone(); nonfinite[0, 0, 0] = complex(float('nan'), 0)
    invalid.append((nonfinite, rhs))
    for current, target in invalid:
        with pytest.raises(ValueError, match='matching finite'):
            measure_original_global_residual(provider, current, target)
    for kwargs in (dict(edge_batch=True), dict(edge_batch=0), dict(source_batch=1.5),
                   dict(tolerance=True), dict(tolerance=0), dict(tolerance=float('inf')),
                   dict(absolute_tolerance=-1), dict(absolute_tolerance=float('nan'))):
        with pytest.raises(ValueError):
            measure_original_global_residual(provider, value, rhs, **kwargs)
    with pytest.raises(TypeError, match='scalar system blocks'):
        measure_original_global_residual(object(), value, rhs)


@pytest.mark.parametrize('mutation', ['current_version', 'rhs_version', 'current_mode', 'provider',
    'config_field', 'material_field', 'body_offset'])
def test_mid_measurement_binding_changes_rejected(original, monkeypatch, mutation):
    provider, dense, base = original
    value = base[0, 0].clone(); rhs = value@dense.mT
    def changed(rows, current, *, source_batch):
        indices = torch.cat((rows, rows+12))
        result = (current@dense.mT)[..., indices]
        if mutation == 'current_version':
            current.add_(0)
        elif mutation == 'rhs_version':
            rhs.add_(0)
        elif mutation == 'current_mode':
            current.requires_grad_(True)
        elif mutation == 'provider':
            monkeypatch.setattr(provider, 'config', IntegrationConfig())
        elif mutation == 'config_field':
            monkeypatch.setitem(vars(provider.config), 'order', provider.config.order+1)
        elif mutation == 'material_field':
            monkeypatch.setitem(vars(provider.system.bodies[0].interior), 'k',
                                provider.system.bodies[0].interior.k+1)
        else:
            monkeypatch.setitem(vars(provider.system.bodies[0]), 'edge_offset', 1)
        return result
    monkeypatch.setattr(provider, 'sampled_action', changed)
    with pytest.raises(ValueError, match='changed'):
        measure_original_global_residual(provider, value, rhs, edge_batch=2)


def test_native_nonfinite_action_rejected(original, monkeypatch):
    provider, _, base = original
    def nonfinite(rows, value, *, source_batch):
        return torch.full((2*len(rows),), complex(float('inf'), 0), device=value.device, dtype=value.dtype)
    monkeypatch.setattr(provider, 'sampled_action', nonfinite)
    with pytest.raises(ValueError, match='exceeds native precision'):
        measure_original_global_residual(provider, base[0, 0], torch.zeros_like(base[0, 0]))


def test_scalar_receipt_and_explicit_absolute_gate(original, monkeypatch):
    provider, dense, base = original
    value = base[0, 0]
    rhs = value@dense.mT
    rhs = rhs.clone(); rhs[10] += .125
    def cached_original(rows, current, *, source_batch):
        return (current@dense.mT)[..., torch.cat((rows, rows+12))]
    monkeypatch.setattr(provider, 'sampled_action', cached_original)
    receipt = measure_original_global_residual(provider, value, rhs, edge_batch=5,
                                             absolute_tolerance=.25)
    assert receipt['rhs_count'] == 1 and receipt['passed_per_rhs'] is True
    assert receipt['absolute_residual'] == pytest.approx(.125)
    assert receipt['threshold'] == pytest.approx(.25+1e-10*float(rhs.norm()))
    assert isinstance(receipt['relative_residual'], float)
    assert receipt['all_rhs_passed'] and receipt['all_original_rows_measured']
