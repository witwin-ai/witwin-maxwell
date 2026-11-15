"""P1 gates for analytic indexing, symmetric integration and sparse conditioning."""

import json
import math

import pytest
import torch

from witwin.maxwell.mom.layered.assembly import _symmetric_rule, assemble_potentials
from witwin.maxwell.mom.layered.build import build_extension
from witwin.maxwell.mom.layered.lowfreq import scaled_system
from tests.mom.layered.test_layered_lowfreq import loop


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


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


EPS0 = 8.8541878128e-12
MU0 = 1 / (EPS0 * 299792458.0**2)


def scaled_reference(vector, scalar, loss, q, charge, rhs, frequencies, loops, omega0):
    """Torch oracle of the equilibrated loop/tree system for dense Q and D*Q_tree."""
    w = 2 * math.pi * frequencies[:, None, None]
    n = q.shape[1]
    s = torch.ones(len(frequencies), n, device=q.device, dtype=torch.float64)
    s[:, loops:] = w[:, :, 0] / omega0
    ss = s[:, :, None] * s[:, None, :]
    q, charge = q.to(torch.complex128), charge.to(torch.complex128)
    entry = -1j * MU0 * (q.T @ vector @ q) * ss + loss_term(loss, q) * ss / w
    entry[:, loops:, loops:] += 1j / (EPS0 * omega0**2) * (charge.T @ scalar @ charge)
    d = torch.diagonal(entry, dim1=1, dim2=2).abs().rsqrt()
    drive = rhs[None] * (d * s)[:, :, None]
    recovery = d * torch.where(torch.arange(n, device=q.device) < loops, 1 / w[:, :, 0], 1 / omega0)
    return entry * d[:, :, None] * d[:, None, :], drive, recovery


def loss_term(loss, q):
    return 0 if loss is None else q.T @ loss @ q


def dense_loss(prepared):
    if prepared.loss is None:
        return None
    n = prepared.layout.basis.n_edges
    loss = torch.zeros((len(prepared.table.frequencies), n, n), device="cuda", dtype=torch.complex128)
    for f in range(len(loss)):
        prepared.loss.add_to(loss[f], f)
    return loss


def oracle_inputs(prepared):
    from tests.mom.layered.test_layered_lowfreq import dense_columns
    lt = prepared.loop_tree
    n, faces = prepared.layout.basis.n_edges, len(prepared.layout.mesh.faces)
    vector, scalar = assemble_potentials(prepared.layout, prepared.table, frequency_index=-1,
                                         far_dtype=torch.float32, _network_accuracy=True)
    transform = dense_columns(lt.sparse_transform, n)
    # Integer cycle work: entries are exactly +-1 times +-l/l.
    work = (torch.sign(transform).T @ (prepared.excitation.real / prepared.layout.basis.edge_lengths[:, None]))
    return dict(vector=vector, scalar=scalar, loss=dense_loss(prepared), transform=transform,
                q=dense_columns(lt.sparse_q, n), charge=dense_columns(lt.sparse_charge, faces),
                work=work.to(torch.complex128))


@pytest.mark.parametrize("copper", [False, True])
def test_inplace_system_matches_dense_oracle_and_repeats(copper, record_property):
    prepared = loop((1e3, 1e6, 1e9), copper=copper)
    lt = prepared.loop_tree
    first = scaled_system(prepared)
    for _ in range(2):
        again = scaled_system(prepared)
        for name in ("matrix", "rhs", "recovery"):
            assert torch.equal(getattr(first, name), getattr(again, name))
    o = oracle_inputs(prepared)
    rhs = o["work"] / lt.norms[:, None]
    matrix, drive, recovery = scaled_reference(o["vector"], o["scalar"], o["loss"], o["q"], o["charge"], rhs,
                                               prepared.table.frequencies, lt.loops, lt.omega0)
    errors = {}
    for name, ref, new in (("matrix", matrix, first.matrix), ("rhs", drive, first.rhs),
                           ("recovery", recovery, first.recovery)):
        errors[name] = ((new - ref).abs().max() / ref.abs().max()).item()
        # fp64 setup-tensor bound of the numerical-equivalence standard.
        assert errors[name] <= 1e-12, name
    record_property("relative_to_dense_oracle", json.dumps(errors))


def test_sparse_conditioning_against_orthogonal_basis(record_property):
    prepared = loop((1., 1e6, 1e9))
    lt = prepared.loop_tree
    matrix = scaled_system(prepared).matrix
    # Independent small dense oracle: orthogonalize the same complete basis.
    o = oracle_inputs(prepared)
    q, r = torch.linalg.qr(o["transform"])
    d = torch.ops.witwin_maxwell_mom_layered.divergence(
        q[:, lt.loops:].contiguous(), prepared.layout.basis.face_edges.contiguous(),
        prepared.layout.basis.face_signs.contiguous(), prepared.layout.basis.edge_lengths.contiguous())
    rhs = torch.linalg.solve_triangular(r.to(torch.complex128).T, o["work"], upper=False)
    reference, _, _ = scaled_reference(o["vector"], o["scalar"], None, q, d, rhs,
                                       prepared.table.frequencies, lt.loops, lt.omega0)
    sparse_condition = torch.linalg.cond(matrix)
    orthogonal_condition = torch.linalg.cond(reference)
    assert sparse_condition.max() / sparse_condition.min() < 10
    assert sparse_condition.max() < 1e6
    record_property("sparse_condition", str(sparse_condition.tolist()))
    record_property("orthogonal_condition", str(orthogonal_condition.tolist()))
