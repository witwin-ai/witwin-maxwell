"""Repeated homogeneous MoM runs on CUDA are bitwise identical without deterministic-mode switches."""
import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.ordered_reduction import ordered_index_add_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
REPEATS = 3


def _sequential_reference(target, index, source):
    """Host fp64 sum of each destination in original contribution order, added once."""
    total = torch.zeros_like(target)
    for i, row in enumerate(index.tolist()):
        total[row] = total[row]+source[i]
    return target+total


def test_ordered_index_add_matches_sequential_sum_and_replays():
    generator = torch.Generator(device='cuda').manual_seed(3)
    index = torch.randint(0, 37, (5000,), device='cuda', generator=generator)
    index[:7] = 36
    source = torch.randn(len(index), 2, 3, device='cuda', dtype=torch.complex128, generator=generator)
    base = torch.randn(1, 2, 40, 3, device='cuda', dtype=torch.complex128, generator=generator)
    contributions = source.permute(1, 0, 2)[None]
    actual = ordered_index_add_(base.clone(), 2, index, contributions)
    expected = _sequential_reference(base[0].permute(1, 0, 2).cpu(), index.cpu(), source.cpu())
    assert torch.equal(actual[0].permute(1, 0, 2).cpu(), expected)
    for _ in range(9):
        assert torch.equal(ordered_index_add_(base.clone(), 2, index, contributions), actual)
    unchanged = base.clone()
    assert ordered_index_add_(unchanged, 2, index[:0], contributions[:, :, :0]) is unchanged
    assert torch.equal(unchanged, base)


def test_ordered_index_add_gradient_matches_index_add():
    index = torch.tensor([4, 0, 4, 4, 1, 0], device='cuda')
    source = torch.randn(len(index), 3, device='cuda', dtype=torch.float64, requires_grad=True)
    base = torch.randn(6, 3, device='cuda', dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 3, device='cuda', dtype=torch.float64)
    got = torch.autograd.grad((ordered_index_add_(base.clone(), 0, index, source)*weight).sum(), (base, source))
    reference = torch.autograd.grad((base.index_add(0, index, source)*weight).sum(), (base, source))
    for a, b in zip(got, reference):
        assert torch.equal(a, b)
    # Second derivatives, as in create_graph field recovery.
    nonlinear = lambda f: ((f(base.square(), 0, index, source.sin()))*weight).sum()
    got = torch.autograd.grad(nonlinear(lambda t, d, i, s: ordered_index_add_(t.clone(), d, i, s)), (base, source), create_graph=True)
    reference = torch.autograd.grad(nonlinear(lambda t, d, i, s: t.index_add(d, i, s)), (base, source), create_graph=True)
    for a, b in zip(torch.autograd.grad(sum(g.sum() for g in got), (base, source)),
                    torch.autograd.grad(sum(g.sum() for g in reference), (base, source))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def _linear_run():
    scene = mw.Scene(structures=[mw.Structure(geometry=mw.Sphere(radius=.1), material=mw.Material(eps_r=2.25))],
                     sources=[mw.PlaneWave()],
                     monitors=[mw.PointMonitor('probe', (.3, .2, .1), fields=('Ex', 'Hy', 'Ez')),
                               mw.PointMonitor('inside', (.02, 0., .01), fields=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'))])
    prepared = mw.Simulation.mom(scene, frequencies=3e8, config=mw.mom.MoMConfig(
        subdivisions=0, integration=IntegrationConfig(order=3, singular_order=6))).prepare()
    result = prepared.run()
    values = {'matrix': prepared.solver.handle.matrix.clone(), 'rhs': prepared.solver.rhs.clone()}
    for name in ('probe', 'inside'):
        data = result.monitor(name)
        for component in data.coords['component']:
            values[f'{name}/{component}'] = data.sel(component=component).values.clone()
    points = torch.tensor([[.3, .2, .1], [.02, 0., .01]], device='cuda', dtype=torch.float64)
    for row, region in ((0, 0), (1, 1)):
        electric, magnetic = result.query_fields(points[row:row+1], region_id=region, field_kind='total')
        values[f'query{row}'] = torch.cat((electric, magnetic), -1)
    return values


def test_linear_dense_operator_rhs_monitors_and_queries_replay_bitwise():
    first = _linear_run()
    for _ in range(REPEATS-1):
        again = _linear_run()
        for key, value in first.items():
            assert torch.equal(again[key], value), key


def test_harmonic_state_and_monitors_replay_bitwise():
    from tests.mom.test_harmonic_public import setup

    def run():
        scene, config, points = setup()
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PointMonitor('inside', tuple(points[0].tolist()), fields=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')),
        ))
        result = mw.Simulation.mom(scene, frequencies=3e8, config=config).run()
        state = mw.mom.harmonic_state(result)
        values = {name: getattr(state, name).clone() for name in
                  ('cell_electric', 'polarization', 'dissipative_current', 'total_volume_current', 'dc_surface_charge')}
        values.update({f'surface{i}': value.clone() for i, value in enumerate(state.surface_coefficients)})
        data = result.monitor('inside')
        values.update({f'inside/{k}': data.sel(component=k).values.clone() for k in data.coords['component']})
        for harmonic in (1, 2):
            values[f'query{harmonic}'] = torch.cat(result.query_fields(points, region_id=1, field_kind='total',
                                                                       harmonic=harmonic), -1)
        return values

    first = run()
    for _ in range(REPEATS-1):
        again = run()
        for key, value in first.items():
            assert torch.equal(again[key], value), key


@pytest.mark.parametrize('kind', ['near', 'touching'])
def test_refined_near_and_touching_integrals_replay_bitwise(kind):
    from benchmark import mom_near_quadrature_accuracy as near
    from benchmark import mom_touching_quadrature_accuracy as touching
    from witwin.maxwell.mom import operator as op
    medium = SimpleNamespace(k=2*math.pi)
    if kind == 'near':
        basis = near.pair_basis(*near.coplanar_sliver(.1, 30))
        rows, cols = torch.tensor([0], device='cuda'), torch.tensor([1], device='cuda')
        evaluate = lambda: op._fan_near_integrals(basis, rows, cols, medium, 6, 8, 1.)
    else:
        basis = touching.strip(30, True)
        r, c, category = touching.PAIRS['edge_long']
        rows, cols = torch.tensor([r], device='cuda'), torch.tensor([c], device='cuda')
        evaluate = lambda: op._subtracted_touching_integrals(basis, rows, cols, medium, 6, 8, 4., category=category)
    d, magnetic, stats = evaluate()
    assert stats['leaves'] > 2
    for _ in range(REPEATS-1):
        again = evaluate()
        assert torch.equal(again[0], d) and torch.equal(again[1], magnetic)
