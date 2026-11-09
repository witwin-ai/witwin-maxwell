"""CPU checks of static gather topology, clamps, and interpolation weights."""

from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.adjoint.accumulation import (
    build_point_cell_plan, build_tfsf_sample_plan, prepare_tfsf_sample_plans,
)


@pytest.mark.parametrize("axis", range(3))
def test_line_plan_keeps_repeated_samples_in_cell_order(axis):
    shape = (2, 3, 4)
    view = tuple(shape[d] if d == axis else 1 for d in range(3))
    samples = torch.tensor([2, 2, 0, 2][:shape[axis]])
    term = dict(coeff_patch=torch.ones(shape), sample_view=view,
                sample_indices=samples, scalar_sample_index=None)
    offsets, cells, weights = build_tfsf_sample_plan(term, sample_count=4, origin=0, ds=1)
    expected = [[] for _ in range(4)]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                expected[int(samples[(i, j, k)[axis]])].append((i * shape[1] + j) * shape[2] + k)
    for sample in range(4):
        assert cells[offsets[sample]:offsets[sample + 1]].tolist() == expected[sample]
    assert torch.equal(weights, torch.ones(24))


@pytest.mark.parametrize("ds", [0.0, 0.5])
@pytest.mark.parametrize("count", [1, 4])
def test_interpolated_plan_clamps_and_retains_weights(ds, count):
    positions = torch.tensor([-1.0, 0.25, 0.5, 0.75, 1.25, 9.0]).reshape(1, 2, 3)
    term = dict(coeff_patch=torch.ones_like(positions), sample_positions=positions)
    offsets, cells, weights = build_tfsf_sample_plan(term, sample_count=count, origin=0.25, ds=ds)
    expected = [[] for _ in range(count)]
    for cell, position in enumerate(positions.flatten().tolist()):
        coord = min(max((position - 0.25) / ds if ds > 0 else 0.0, 0), count - 1)
        lower = int(coord)
        upper = min(lower + 1, count - 1)
        expected[lower].append((cell, 1.0 - (coord - lower)))
        if upper != lower:
            expected[upper].append((cell, coord - lower))
    for sample, entries in enumerate(expected):
        start, end = int(offsets[sample]), int(offsets[sample + 1])
        assert cells[start:end].tolist() == [cell for cell, _ in entries]
        assert weights[start:end].tolist() == [weight for _, weight in entries]


def test_scalar_plan_is_cached_before_reverse_steps():
    term = dict(coeff_patch=torch.ones((2, 3, 4)), scalar_sample_index=2)
    aux = SimpleNamespace(ds=0.5, s_min=-1, electric=torch.zeros(5), magnetic=torch.zeros(4))
    solver = SimpleNamespace(_tfsf_state=dict(auxiliary_grid=aux, electric_terms=(term,)))
    prepare_tfsf_sample_plans(solver)
    plan = term["adjoint_sample_plan"]
    assert plan[0].tolist() == [0, 0, 0, 24, 24]
    assert plan[1].tolist() == list(range(24))
    prepare_tfsf_sample_plans(solver)
    assert term["adjoint_sample_plan"] is plan


def test_point_plan_deduplicates_cells_with_stable_point_order():
    i = torch.tensor([1, 0, 1, 1, 0])
    j = torch.tensor([0, 1, 0, 0, 1])
    k = torch.tensor([1, 0, 1, 1, 0])
    cells, offsets, points = build_point_cell_plan(i, j, k, (2, 3, 4))
    assert cells.tolist() == [4, 13]
    assert offsets.tolist() == [0, 2, 5]
    assert points.tolist() == [1, 4, 0, 2, 3]


def test_empty_point_plan():
    empty = torch.empty(0, dtype=torch.long)
    cells, offsets, points = build_point_cell_plan(empty, empty, empty, (2, 3, 4))
    assert cells.numel() == points.numel() == 0
    assert offsets.tolist() == [0]
