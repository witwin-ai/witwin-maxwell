"""True split geometry, positive Hodge, and independent local/harmonic controls.

CPU tests do no field time evolution. The optional native time-step parity
control requires an explicit CUDA validation window.
"""
import math
import os

import pytest
import torch

from scripts.dev.fdtd.split_copper_plane import (
    SplitPlaneCUDA, compile_split_plane, compiled_midpoint_algebra,
    frequency_oracle, independent_film_reference,
)
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.copper_foil import CopperFoil, EPSILON_0, MU_0


def material():
    thickness = math.sqrt(2 / (2 * math.pi * 1e9 * MU_0 * 5.8e7))
    return CopperFoil(thickness=thickness, frequency_range=(.5e9, 2e9))


@pytest.mark.parametrize("offset", (0., .3, .7))
def test_cut_geometry_has_independent_traces_and_exact_physical_intervals(offset):
    h = .005
    nodes = torch.arange(-3, 4, dtype=torch.float64) * h
    g = compile_split_plane(nodes, offset * h)
    left, right = map(int, g.traces)
    assert left + 1 == right
    assert g.electric_positions[left] == g.electric_positions[right] == offset*h
    assert g.aligned == (offset == 0)
    assert len(g.electric_positions) == len(nodes) + (1 if g.aligned else 2)
    assert bool((g.electric_mass > 0).all())
    assert bool((g.magnetic_mass > 0).all())
    # Compensated geometry sums avoid testing an arbitrary parallel reduction
    # order instead of the actual physical control-volume partition.
    assert abs(math.fsum(g.electric_lengths.tolist())-float(nodes[-1]-nodes[0])) <= 4e-18
    assert abs(math.fsum(g.magnetic_lengths.tolist())-float(nodes[-1]-nodes[0])) <= 4e-18
    assert not bool(((g.edge_low <= left) & (g.edge_high >= right)).any())
    if offset:
        torch.testing.assert_close(g.magnetic_lengths[left-1], torch.tensor(offset*h, dtype=torch.float64), rtol=1e-15, atol=0)
        torch.testing.assert_close(g.magnetic_lengths[left], torch.tensor((1-offset)*h, dtype=torch.float64), rtol=1e-15, atol=0)
    else:
        # The aligned branch uses the actual adjacent full magnetic intervals.
        torch.testing.assert_close(g.magnetic_lengths, nodes[1:]-nodes[:-1], rtol=0, atol=0)
    for node in nodes:
        assert int((g.electric_positions == node).sum()) == (2 if node == offset*h else 1)


@pytest.mark.parametrize("offset", (0., .3, .7, 1e-12))
def test_incidence_reproduces_each_one_sided_affine_field_and_exact_work(offset):
    g = compile_split_plane(torch.arange(-2, 3, dtype=torch.float64)*.01, offset*.01)
    C = g.incidence()
    left, right = map(int, g.traces)
    e = torch.cat((2*g.electric_positions[:right]+.7, -3*g.electric_positions[right:]-1.1))
    expected = torch.cat((torch.full((left,), 2., dtype=torch.float64),
                          torch.full((len(g.magnetic_lengths)-left,), -3., dtype=torch.float64)))
    # Cancellation in a tiny cut is bounded by the actual affine input ULP.
    torch.testing.assert_close(C@e, expected*g.magnetic_lengths, rtol=0, atol=3e-16)
    generator = torch.Generator().manual_seed(718)
    e = torch.randn(len(e), generator=generator, dtype=torch.float64)
    h = torch.randn(len(g.magnetic_lengths), generator=generator, dtype=torch.float64)
    torch.testing.assert_close(torch.dot(e, C.T@h), torch.dot(C@e, h), rtol=2e-15, atol=2e-15)
    assert torch.count_nonzero(C[:left, right:]) == 0
    assert torch.count_nonzero(C[left:, :right]) == 0


def test_near_node_cut_remains_a_real_small_interval_without_regularization():
    nodes = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    aligned = compile_split_plane(nodes, 0.)
    near = compile_split_plane(nodes, 1e-12)
    assert aligned.aligned and not near.aligned
    assert float(near.magnetic_lengths.min()) == 1e-12
    assert float(near.electric_lengths[near.traces[0]]) == .5e-12
    assert near.stable_dt() / aligned.stable_dt() == pytest.approx(1e-12, rel=1e-15)


@pytest.mark.parametrize("nodes,cut", (([0, 1, 1], .3), ([0, 1, 2], 0),
    ([0, 1, 2], 2), ([0, float('nan'), 2], .3), ([0, 1, 2], float('nan'))))
def test_invalid_geometry_rejects_before_field_placement(nodes, cut):
    with pytest.raises(ValueError):
        compile_split_plane(nodes, cut)


def independent_augmented_system(g, foil, dt):
    C, me, mh = g.incidence(), g.electric_mass, g.magnetic_mass
    ne, nh = len(me), len(mh)
    response = compile_copper_response(foil)
    B, D, rates = response.state_space.B, response.state_space.D, response.rates
    T = torch.zeros((2, ne), dtype=torch.float64)
    T[torch.arange(2), g.traces] = 1
    Qfield = torch.cat((torch.cat((torch.diag(me), -dt*C.T/2), 1),
                       torch.cat((-dt*C/2, torch.diag(mh)), 1)), 0)
    Q = torch.block_diag(Qfield, torch.eye(len(rates), dtype=torch.float64))
    A = torch.zeros_like(Q)
    A[:ne, :ne] = -T.T@D@T
    A[:ne, ne:ne+nh] = C.T
    A[ne:ne+nh, :ne] = -C
    A[:ne, ne+nh:] = -T.T@B.T
    A[ne+nh:, :ne] = B@T
    A[ne+nh:, ne+nh:] = -torch.diag(rates)
    return Q, A, T, response


@pytest.mark.parametrize("offset", (0., .3, .7))
@pytest.mark.parametrize("complex_state", (False, True))
def test_local_split_schur_matches_full_cn_and_exact_passive_work(offset, complex_state):
    g = compile_split_plane(torch.arange(-2, 3, dtype=torch.float64)*.005, offset*.005)
    dt, foil = g.stable_dt(), material()
    a = compiled_midpoint_algebra(g, foil, dt)
    Q, A, T, response = independent_augmented_system(g, foil, dt)
    scale = Q.diag().sqrt()
    normalized_mass = Q / scale[:, None] / scale[None, :]
    assert float(torch.linalg.eigvalsh(normalized_mass).min()) > .3
    L = torch.linalg.cholesky(Q)
    normalized_A = torch.linalg.solve_triangular(L, A, upper=False)
    normalized_A = torch.linalg.solve_triangular(L, normalized_A.T, upper=False).T
    dtype = torch.complex128 if complex_state else torch.float64
    generator = torch.Generator().manual_seed(2017)
    normalized_old = torch.randn(len(Q), generator=generator, dtype=dtype)
    old = torch.linalg.solve_triangular(L.T.to(dtype), normalized_old[:, None], upper=True)[:, 0]
    eye = torch.eye(len(Q), dtype=torch.float64)
    full = torch.linalg.solve((eye-dt*normalized_A/2).to(dtype),
        (eye+dt*normalized_A/2).to(dtype)@normalized_old)
    full = torch.linalg.solve_triangular(L.T.to(dtype), full[:, None], upper=True)[:, 0]
    ne, nh = len(g.electric_mass), len(g.magnetic_mass)
    e, h, z = old[:ne], old[ne:ne+nh], old[ne+nh:]
    C = g.incidence().to(dtype)
    hnew = h-dt*(C@e)/g.magnetic_mass
    force = C.T@hnew
    enew = e+dt*force/g.electric_mass
    history = a['B'].T.to(dtype)@(a['inverse_denominator']*z)
    enew[g.traces] = a['inverse'].to(dtype)@(a['rhs'].to(dtype)@e[g.traces]+force[g.traces]-history)
    v = (e[g.traces]+enew[g.traces])/2
    znew = a['decay']*z+a['drive'].to(dtype)@v
    reduced = torch.cat((enew, hnew, znew))
    error = torch.linalg.vector_norm(L.T.to(dtype)@(reduced-full))/torch.linalg.vector_norm(L.T.to(dtype)@full)
    assert float(error) < 3e-11
    zmid = (z+znew)/2
    heat = dt*((response.rates*zmid.abs().square()).sum()
        + torch.vdot(v, response.state_space.D.to(dtype)@v).real)
    change = (torch.vdot(reduced, Q.to(dtype)@reduced).real-torch.vdot(old, Q.to(dtype)@old).real)/2
    assert float(heat) >= 0
    torch.testing.assert_close(change, -heat, rtol=4e-11, atol=4e-12)
    assert int((response.rates == 0).sum()) == 1


@pytest.mark.parametrize("spacing", (.005, .0025))
@pytest.mark.parametrize("offset", (0., .3, .7))
def test_independent_harmonic_algebra_preserves_small_copper_transmission(spacing, offset):
    half_intervals = round(.2/spacing)
    g = compile_split_plane(torch.arange(-half_intervals, half_intervals+1, dtype=torch.float64)*spacing, offset*spacing)
    foil = material()
    actual = frequency_oracle(g, foil, g.stable_dt(), 1e9)
    r, t, absorption = independent_film_reference(foil.thickness, foil.conductivity, 1e9)
    assert 1e-5 < abs(t) < 1e-4
    assert abs(actual['reflection']-r) < .002
    assert abs(actual['transmission']-t)/abs(t) < .03
    assert abs(actual['absorption']-absorption)/absorption < .03
    assert abs(actual['heat_absorption']-actual['absorption'])/absorption < 2e-8
    assert actual['harmonic_residual'] < 2e-12


def test_actual_field_runtime_rejects_cpu_before_any_cuda_allocation():
    g = compile_split_plane([-1., 0., 1.], 0.)
    with pytest.raises(ValueError, match='requires CUDA'):
        SplitPlaneCUDA(g, material(), g.stable_dt(), 1e9, device='cpu')


@pytest.mark.skipif(os.environ.get('WITWIN_RUN_SPLIT_CUT_CUDA') != '1',
                    reason='requires an explicitly scheduled split-cell CUDA validation window')
@pytest.mark.parametrize("offset", (0., .3, .7))
def test_actual_cuda_step_matches_independent_full_augmented_cn(offset):
    g = compile_split_plane(torch.arange(-2, 3, dtype=torch.float64)*.005, offset*.005)
    dt, foil = g.stable_dt(), material()
    runtime = SplitPlaneCUDA(g, foil, dt, 1e9)
    Q, A, _, _ = independent_augmented_system(g, foil, dt)
    ne, nh = len(runtime.electric), len(runtime.magnetic)
    A[0, 0] -= runtime.boundary_y
    A[ne-1, ne-1] -= runtime.boundary_y
    L = torch.linalg.cholesky(Q)
    normalized_A = torch.linalg.solve_triangular(L, A, upper=False)
    normalized_A = torch.linalg.solve_triangular(L, normalized_A.T, upper=False).T
    generator = torch.Generator().manual_seed(706)
    normalized_old = torch.randn(len(Q), generator=generator, dtype=torch.complex128)
    old = torch.linalg.solve_triangular(L.T.to(torch.complex128), normalized_old[:, None], upper=True)[:, 0]
    runtime.electric.copy_(old[:ne].cuda())
    runtime.magnetic.copy_(old[ne:ne+nh].cuda())
    runtime.state.copy_(old[ne+nh:].cuda())
    drive = torch.tensor(.7-.3j, dtype=torch.complex128)
    force = torch.zeros(len(Q), dtype=torch.complex128)
    force[0] = 2*runtime.boundary_y*drive
    force = torch.linalg.solve_triangular(L.to(torch.complex128), force[:, None], upper=False)[:, 0]
    eye = torch.eye(len(Q), dtype=torch.complex128)
    next_normalized = torch.linalg.solve(eye-dt*normalized_A/2,
        (eye+dt*normalized_A/2)@normalized_old+dt*force)
    expected = torch.linalg.solve_triangular(L.T.to(torch.complex128), next_normalized[:, None], upper=True)[:, 0]
    runtime.step(drive.cuda())
    actual = torch.cat((runtime.electric, runtime.magnetic, runtime.state)).cpu()
    relative = torch.linalg.vector_norm(L.T.to(torch.complex128)@(actual-expected))/torch.linalg.vector_norm(L.T.to(torch.complex128)@expected)
    assert float(relative) < 3e-11
