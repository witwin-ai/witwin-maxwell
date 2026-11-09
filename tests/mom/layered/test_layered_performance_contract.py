"""P1 gates for analytic indexing, symmetric integration and sparse conditioning."""

import math

import pytest
import torch

from witwin.maxwell.mom.layered.assembly import _symmetric_rule, assemble_potentials
from witwin.maxwell.mom.layered.build import build_extension
from witwin.maxwell.mom.layered.lowfreq import build_loop_tree, scale_potentials
from tests.mom.layered.test_layered_lowfreq import loop


pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("nodes,degree", [(1, 1), (3, 2), (6, 4), (7, 5), (12, 6), (13, 7), (25, 10)])
def test_symmetric_rule_polynomial_moments(nodes, degree):
    rule = _symmetric_rule(nodes, "cuda")
    assert len(rule) == nodes
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            actual = (rule[:, 1]**a * rule[:, 2]**b * rule[:, 3]).sum()
            expected = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            assert abs(actual.item() - expected) < 5e-15


@pytest.mark.parametrize("nodes,near,limit", [(5121, 1e-7, 3e-4), (8192, 1e-6, 1e-3),
                                           (513, 1e-12, 1e2)])
def test_radial_index_matches_binary_search_at_nodes_and_ulps(nodes, near, limit):
    ext = build_extension()
    anchor = torch.empty(0, device="cuda", dtype=torch.float64)
    coarse = ext.grid(anchor, nodes, near, limit)
    grid = torch.empty(2 * len(coarse) - 1, device="cuda", dtype=torch.float64)
    grid[::2] = coarse
    grid[1::2] = (coarse[:-1] + coarse[1:]) * .5
    queries = torch.cat((grid, torch.nextafter(grid, torch.zeros_like(grid)),
                         torch.nextafter(grid, torch.full_like(grid, float("inf"))),
                         torch.linspace(0, limit, 100003, device="cuda", dtype=torch.float64)))
    queries.clamp_(0, float(grid[-1]))
    reference = (torch.searchsorted(grid, queries, right=True) - 1).clamp(0, len(grid) - 2)
    for _ in range(3):
        actual = torch.ops.witwin_maxwell_mom_layered.radial_indices(grid, queries)
        assert torch.equal(actual, reference)


def test_sparse_congruences_match_dense_and_repeat():
    prepared = loop((1e6, 1e9))
    lt, op = prepared.loop_tree, torch.ops.witwin_maxwell_mom_layered
    vector, scalar = assemble_potentials(prepared.layout, prepared.table, frequency_index=-1)
    dense_v = lt.q.T @ vector @ lt.q
    dense_p = lt.charge.T @ scalar @ lt.charge
    for _ in range(3):
        v = op.sparse_product(op.sparse_product(vector, lt.sparse_q, True), lt.sparse_q, False)
        p = op.sparse_product(op.sparse_product(scalar, lt.sparse_charge, True), lt.sparse_charge, False)
        assert (v - dense_v).abs().max() <= 1e-12 * dense_v.abs().max()
        assert (p - dense_p).abs().max() <= 1e-12 * dense_p.abs().max()
        if _ == 0:
            first_v, first_p = v, p
        else:
            assert torch.equal(v, first_v)
            assert torch.equal(p, first_p)


def test_sparse_conditioning_against_orthogonal_basis(record_property):
    prepared = loop((1., 1e6, 1e9))
    lt, op = prepared.loop_tree, torch.ops.witwin_maxwell_mom_layered
    vector, scalar = assemble_potentials(prepared.layout, prepared.table, frequency_index=-1)
    matrix, _, _ = scale_potentials(prepared, vector, scalar)
    # Independent small dense oracle: orthogonalize the same complete basis.
    q, r = torch.linalg.qr(lt.transform)
    d = op.divergence(q[:, lt.loops:].contiguous(),
                      prepared.layout.basis.face_edges.contiguous(),
                      prepared.layout.basis.face_signs.contiguous(),
                      prepared.layout.basis.edge_lengths.contiguous()).to(torch.complex128)
    q = q.to(torch.complex128)
    work = op.port_work(lt.transform, prepared.layout.basis.edge_lengths.contiguous(), prepared.excitation)
    rhs = torch.linalg.solve_triangular(r.to(torch.complex128).T, work, upper=False).contiguous()
    reference, _, _ = op.scaled((q.T @ vector @ q).contiguous(),
                                (d.T @ scalar @ d).contiguous(),
                                torch.empty(0, device="cuda", dtype=torch.complex128), rhs,
                                prepared.table.frequencies, lt.loops, lt.omega0)
    sparse_condition = torch.linalg.cond(matrix)
    orthogonal_condition = torch.linalg.cond(reference)
    assert sparse_condition.max() / sparse_condition.min() < 10
    assert sparse_condition.max() < 1e6
    record_property("sparse_condition", str(sparse_condition.tolist()))
    record_property("orthogonal_condition", str(orthogonal_condition.tolist()))
