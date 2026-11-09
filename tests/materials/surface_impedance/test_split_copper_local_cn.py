"""Independent local IMEX algebra; no CPU field time integration.

The transition matrix is assembled once for spectral and energy analysis.
Optional actual field-step controls stay explicitly CUDA-only.
"""
import math
import os
import json
import sys

import mpmath as mp
import pytest
import torch

from scripts.dev.fdtd.split_copper_plane import (
    SplitPlaneCUDA, compile_local_cn_support, compile_split_plane,
    compiled_local_cn_algebra, independent_film_reference,
    local_cn_frequency_oracle,
)
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.copper_foil import CopperFoil, EPSILON_0, MU_0


ALPHAS = (0., 1e-12, 1e-9, 1e-6, .3, .7, 1-1e-12, 1.)


def case(alpha, *, spacing=.005, courant=.9):
    nodes = torch.arange(-4, 5, dtype=torch.float64)*spacing
    geometry = compile_split_plane(nodes, alpha*spacing)
    dt = geometry.bulk_stable_dt(courant)
    delta = math.sqrt(2/(2*math.pi*1e9*MU_0*5.8e7))
    material = CopperFoil(thickness=delta, frequency_range=(.5e9, 2e9))
    bounds = (-spacing, 2*spacing)
    return geometry, material, dt, bounds


def independent_full_system(geometry, material, dt, bounds):
    """Direct physical modified-Q and semidiscrete Maxwell/material matrices."""
    C = geometry.incidence()
    # Independent coordinate test selects implicit physical intervals, not the
    # implementation's compiled support or local field factor.
    xlow = geometry.electric_positions[geometry.edge_low]
    xhigh = geometry.electric_positions[geometry.edge_high]
    explicit = (xlow < bounds[0]) | (xhigh > bounds[1])
    Cexp = C*explicit[:, None]
    me, mh = geometry.electric_mass, geometry.magnetic_mass
    ne, nh = len(me), len(mh)
    foil = compile_copper_response(material)
    B, D, rates = foil.state_space.B, foil.state_space.D, foil.rates
    selector = torch.zeros((2, ne), dtype=torch.float64)
    selector[torch.arange(2), geometry.traces] = 1
    field_mass = torch.cat((torch.cat((torch.diag(me), -dt*Cexp.T/2), 1),
                            torch.cat((-dt*Cexp/2, torch.diag(mh)), 1)), 0)
    Q = torch.block_diag(field_mass, torch.eye(len(rates), dtype=torch.float64))
    A = torch.zeros_like(Q)
    A[:ne, :ne] = -selector.T@D@selector
    A[:ne, ne:ne+nh] = C.T
    A[ne:ne+nh, :ne] = -C
    A[:ne, ne+nh:] = -selector.T@B.T
    A[ne+nh:, :ne] = B@selector
    A[ne+nh:, ne+nh:] = -torch.diag(rates)
    return Q, A, foil


def reduced_midpoint_columns(geometry, material, dt, bounds, old, *, return_midpoints=False):
    """One batched local Schur algebra, not an iterative CPU field solver."""
    p = compiled_local_cn_algebra(geometry, material, dt, bounds)
    a = p['material']
    ne, nh = len(geometry.electric_mass), len(geometry.magnetic_mass)
    e, h, z = old[:ne], old[ne:ne+nh], old[ne+nh:]
    C = geometry.incidence().to(old.dtype)
    explicit, ep, hp = p['explicit'], p['electric'], p['magnetic']
    hnew = h.clone()
    hnew[explicit] = h[explicit]-dt*(C[explicit]@e)/geometry.magnetic_mass[explicit, None]
    force = C[explicit].T@hnew[explicit]
    enew = e+dt*force/geometry.electric_mass[:, None]
    history = a['B'].T.to(old.dtype)@(a['inverse_denominator'][:, None]*z)
    rhs_e = p['electric_diagonal'][:, None]*e[ep]+p['eta']*force[ep]
    rhs_e -= p['eta']*p['trace_selector'].T.to(old.dtype)@history
    rhs_h = p['magnetic_diagonal'][:, None]*p['eta']*h[hp]
    midpoint = torch.linalg.lu_solve(p['factor'], p['pivots'], torch.cat((rhs_e, rhs_h)).to(torch.complex128))
    if not old.is_complex():
        assert float(midpoint.imag.abs().max()) < 1e-20
        midpoint = midpoint.real
    enew[ep] = 2*midpoint[:len(ep)]-e[ep]
    hnew[hp] = 2*midpoint[len(ep):]/p['eta']-h[hp]
    # Use the actually solved midpoint, without subtracting large fast fields.
    v = p['trace_selector'].to(old.dtype)@midpoint[:len(ep)]
    znew = a['decay'][:, None]*z+a['drive'].to(old.dtype)@v
    result = torch.cat((enew, hnew, znew))
    return (result, v, (z+znew)/2) if return_midpoints else result


def spectral_control(alpha, courant=.9):
    geometry, material, dt, bounds = case(alpha, courant=courant)
    Q, A, _ = independent_full_system(geometry, material, dt, bounds)
    diagonal = Q.diag().sqrt()
    minimum_q = float(torch.linalg.eigvalsh(Q/diagonal[:, None]/diagonal[None, :]).min())
    L = torch.linalg.cholesky(Q)
    identity = torch.eye(len(Q), dtype=torch.float64)
    basis = torch.linalg.solve_triangular(L.T, identity, upper=True)
    new = reduced_midpoint_columns(geometry, material, dt, bounds, basis)
    transition = L.T@new
    radius = float(torch.linalg.eigvals(transition).abs().max())
    norm = float(torch.linalg.matrix_norm(transition, ord=2))
    contraction_min = float(torch.linalg.eigvalsh(identity-transition.T@transition).min())
    return dict(alpha=alpha, courant=courant, dt_s=dt, minimum_normalized_Q_eigenvalue=minimum_q,
        spectral_radius=radius, energy_operator_norm=norm,
        minimum_energy_contraction_eigenvalue=contraction_min)


@pytest.mark.parametrize('alpha', ALPHAS)
def test_true_intervals_and_bounded_local_matrix_survive_sliver_limits(alpha):
    geometry, material, dt, bounds = case(alpha)
    p = compiled_local_cn_algebra(geometry, material, dt, bounds)
    support = compile_local_cn_support(geometry, bounds)
    assert p['bounds'] == bounds
    assert len(support['magnetic']) == (3 if geometry.aligned else 4)
    torch.testing.assert_close(geometry.incidence()[support['magnetic']][:, support['electric']],
                               support['local_curl'], rtol=0, atol=0)
    assert bool((geometry.magnetic_lengths[support['magnetic']] > 0).all())
    assert bool((geometry.electric_mass[support['electric']] > 0).all())
    assert float(torch.linalg.cond(p['matrix'])) < 2e4
    assert dt == geometry.bulk_stable_dt(.9)
    if alpha == 1e-12:
        assert float(geometry.magnetic_lengths.min()) == .005e-12
        assert dt/geometry.stable_dt() > 1e12
    # A geometry-independent bound includes the finite shared copper matrix.
    # The initial fixture incorrectly omitted that material coefficient.
    largest_original = float((geometry.original_nodes[1:]-geometry.original_nodes[:-1]).max())
    bulk_diagonal_bound = 2*math.sqrt(MU_0*EPSILON_0)*largest_original/dt
    material_bound = p['eta']*float(p['material']['effective'].abs().max())
    exact_bound = max(1., bulk_diagonal_bound, material_bound+bulk_diagonal_bound)
    assert float(p['matrix'].abs().max()) <= exact_bound*(1+8*torch.finfo(torch.float64).eps)


@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('courant', (.5, .9))
def test_bulk_cfl_positive_mass_and_spectral_energy_contraction(alpha, courant):
    report = spectral_control(alpha, courant)
    assert report['minimum_normalized_Q_eigenvalue'] > 1-courant-2e-14
    assert report['spectral_radius'] <= 1+2e-11
    assert report['energy_operator_norm'] <= 1+2e-11
    assert report['minimum_energy_contraction_eigenvalue'] >= -4e-11


def high_precision_midpoint(Q, A, dt, old, *, return_midpoint=False):
    """Independent complete augmented CN matrix, including all skin states."""
    with mp.workdps(85):
        def scalar(value):
            value = complex(value)
            return mp.mpc(mp.mpf(value.real), mp.mpf(value.imag))
        q = mp.matrix([[scalar(value) for value in row] for row in Q.tolist()])
        a = mp.matrix([[scalar(value) for value in row] for row in A.tolist()])
        u = mp.matrix([scalar(value) for value in old.tolist()])
        half = mp.mpf(float(dt))/2
        midpoint = mp.lu_solve(q-half*a, q*u)
        result = torch.tensor([complex(2*midpoint[i]-u[i]) for i in range(len(old))], dtype=torch.complex128)
        mid = torch.tensor([complex(midpoint[i]) for i in range(len(old))], dtype=torch.complex128)
        return (result, mid) if return_midpoint else result


def precision_control(alpha, complex_state):
    geometry, material, dt, bounds = case(alpha)
    Q, A, response = independent_full_system(geometry, material, dt, bounds)
    L = torch.linalg.cholesky(Q)
    dtype = torch.complex128 if complex_state else torch.float64
    generator = torch.Generator().manual_seed(882)
    normalized_old = torch.randn(len(Q), dtype=dtype, generator=generator)
    old = torch.linalg.solve_triangular(L.T.to(dtype), normalized_old[:, None], upper=True)[:, 0]
    expected, oracle_midpoint = high_precision_midpoint(Q, A, dt, old, return_midpoint=True)
    next_columns, v_columns, z_columns = reduced_midpoint_columns(geometry, material, dt, bounds, old[:, None], return_midpoints=True)
    actual, v, z = next_columns[:, 0], v_columns[:, 0], z_columns[:, 0]
    error = torch.linalg.vector_norm(L.T.to(torch.complex128)@(actual-expected))/torch.linalg.vector_norm(L.T.to(torch.complex128)@expected)
    heat = dt*(torch.vdot(v, response.state_space.D.to(dtype)@v).real+(response.rates*z.abs().square()).sum())
    energy_change = (torch.vdot(actual, Q.to(dtype)@actual).real-torch.vdot(old, Q.to(dtype)@old).real)/2
    v_oracle = oracle_midpoint[geometry.traces]
    midpoint_error = float(torch.linalg.vector_norm(v-v_oracle)/torch.linalg.vector_norm(v_oracle))
    # Report the rejected reconstruction as a diagnostic, without using it in
    # the actual state advance or the work identity.
    reconstructed = (old[geometry.traces]+actual[geometry.traces])/2
    reconstruction_error = float(torch.linalg.vector_norm(reconstructed-v_oracle)/torch.linalg.vector_norm(v_oracle))
    return dict(alpha=alpha, complex_state=complex_state, normalized_full_state_relative_error=float(error),
        solved_trace_midpoint_relative_error=midpoint_error,
        rejected_reconstructed_midpoint_relative_error=reconstruction_error,
        irreversible_heat=float(heat), energy_change=float(energy_change),
        absolute_energy_work_residual=float(abs(energy_change+heat)))


@pytest.mark.parametrize('alpha', (0., 1e-12, .3, .7, 1-1e-12, 1.))
@pytest.mark.parametrize('complex_state', (False, True))
def test_local_schur_matches_independent_85_digit_augmented_cn_and_positive_heat(alpha, complex_state):
    report = precision_control(alpha, complex_state)
    assert report['normalized_full_state_relative_error'] < 3e-11
    assert report['solved_trace_midpoint_relative_error'] < 3e-11
    assert report['irreversible_heat'] >= 0
    assert report['absolute_energy_work_residual'] <= 2e-11+5e-10*report['irreversible_heat']


@pytest.mark.parametrize('spacing', (.005, .0025))
@pytest.mark.parametrize('alpha', ALPHAS)
def test_full_harmonic_algebra_keeps_original_small_transmission_gate_at_bulk_cfl(alpha, spacing):
    half_intervals = round(.2/spacing)
    geometry = compile_split_plane(torch.arange(-half_intervals, half_intervals+1, dtype=torch.float64)*spacing,
                                   alpha*spacing)
    _, material, _, _ = case(alpha)
    actual = local_cn_frequency_oracle(geometry, material, geometry.bulk_stable_dt(.9), 1e9, (-spacing, 2*spacing))
    r, t, absorption = independent_film_reference(material.thickness, material.conductivity, 1e9)
    assert 1e-5 < abs(t) < 1e-4
    assert abs(actual['reflection']-r) < .002
    assert abs(actual['transmission']-t)/abs(t) < .03
    assert abs(actual['absorption']-absorption)/absorption < .03
    assert abs(actual['heat_absorption']-actual['absorption'])/absorption < 2e-8
    assert actual['harmonic_residual'] < 2e-10
    assert actual['harmonic_matrix_condition'] < 1e7


@pytest.mark.parametrize('aligned_alpha,near_alpha', ((0., 1e-12), (1., 1-1e-12)))
def test_fixed_implicit_patch_has_the_actual_aligned_harmonic_limit(aligned_alpha, near_alpha):
    values = []
    for alpha in (aligned_alpha, near_alpha):
        geometry, material, dt, bounds = case(alpha)
        values.append(local_cn_frequency_oracle(geometry, material, dt, 1e9, bounds))
    assert abs(values[0]['reflection']-values[1]['reflection']) < 2e-12
    assert abs(values[0]['transmission']-values[1]['transmission']) < 2e-15


@pytest.mark.parametrize('bounds', ((-.004, .01), (-.005, .004), (.005, .01), (-.02, .01)))
def test_implicit_bounds_cannot_snap_or_discard_actual_intervals(bounds):
    geometry, _, _, _ = case(.3)
    with pytest.raises(ValueError):
        compile_local_cn_support(geometry, bounds)


def test_local_cn_still_rejects_a_timestep_above_bulk_cfl():
    geometry, material, _, bounds = case(1e-12)
    with pytest.raises(ValueError, match='original mesh CFL'):
        compiled_local_cn_algebra(geometry, material, geometry.bulk_stable_dt(.99)*1.001, bounds)


def test_local_cn_actual_field_runtime_cannot_run_on_cpu():
    geometry, material, dt, bounds = case(1e-12)
    with pytest.raises(ValueError, match='requires CUDA'):
        SplitPlaneCUDA(geometry, material, dt, 1e9, scheme='local_cn', implicit_bounds=bounds, device='cpu')


def test_host_driver_keeps_real_sliver_bulk_dt_source_pins_and_no_cuda_claim(tmp_path, monkeypatch):
    from scripts.dev.fdtd.split_copper_plane_validation import main
    output = tmp_path/'local-cn-host.json'
    before = torch.cuda.is_initialized()
    monkeypatch.setattr(sys, 'argv', ['split-copper-validation', '--cpu-audit', '--scheme', 'local-cn',
        '--offsets', '1e-12', '--spacings', '.005', '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    assert torch.cuda.is_initialized() == before
    assert report['source_unchanged']
    assert report['research_time_domain_completed'] is False
    assert report['research_original_plane_gates_pass'] is None
    assert report['imported_packages_before'] == report['imported_packages_after']
    row = report['rows'][0]
    assert 'cuda_research' not in row
    assert row['harmonic_algebra']['original_plane_gates_pass']
    assert row['minimum_magnetic_interval_m'] == .005e-12
    assert row['compatible_local_cn']['bulk_over_cut_dt'] > 9e11
    assert 0. in row['compatible_local_cn']['algorithmic_magnetic_time_offsets_relative_E_s']
    assert -row['dt_s']/2 in row['compatible_local_cn']['algorithmic_magnetic_time_offsets_relative_E_s']


@pytest.mark.skipif(os.environ.get('WITWIN_RUN_SPLIT_CUT_CUDA') != '1',
                    reason='requires an explicitly scheduled split-cell CUDA validation window')
@pytest.mark.parametrize('alpha', (1e-12, .3, 1-1e-12))
def test_actual_cuda_local_cn_step_matches_independent_augmented_matrix(alpha):
    geometry, material, dt, bounds = case(alpha)
    Q, A, _ = independent_full_system(geometry, material, dt, bounds)
    runtime = SplitPlaneCUDA(geometry, material, dt, 1e9, scheme='local_cn', implicit_bounds=bounds)
    ne, nh = len(geometry.electric_mass), len(geometry.magnetic_mass)
    A[0, 0] -= runtime.boundary_y
    A[ne-1, ne-1] -= runtime.boundary_y
    L = torch.linalg.cholesky(Q)
    generator = torch.Generator().manual_seed(818)
    normalized = torch.randn(len(Q), generator=generator, dtype=torch.complex128)
    old = torch.linalg.solve_triangular(L.T.to(normalized.dtype), normalized[:, None], upper=True)[:, 0]
    expected = high_precision_midpoint(Q, A, dt, old)
    runtime.electric.copy_(old[:ne].cuda())
    runtime.magnetic.copy_(old[ne:ne+nh].cuda())
    runtime.state.copy_(old[ne+nh:].cuda())
    runtime.step(torch.zeros((), dtype=torch.complex128, device='cuda'))
    actual = torch.cat((runtime.electric, runtime.magnetic, runtime.state)).cpu()
    error = torch.linalg.vector_norm(L.T.to(actual.dtype)@(actual-expected))/torch.linalg.vector_norm(L.T.to(actual.dtype)@expected)
    assert float(error) < 3e-11
