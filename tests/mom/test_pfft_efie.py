"""Precorrected-FFT open-sheet PEC EFIE against the dense S-1 operator (CUDA)."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, MU_0
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

FREQUENCY = 3e8
WAVELENGTH = C_0/FREQUENCY


def plates(side, gap=None):
    structures = [mw.Structure(mw.Box(size=(side, side, 0)), mw.Material.pec(), name='a')]
    if gap is not None:
        structures.append(mw.Structure(mw.Box(position=(0., 0., gap), size=(0.8*side, side, 0)),
                                       mw.Material.pec(), name='b'))
    return mw.Scene(structures=structures,
                    sources=[mw.PlaneWave(direction=(0.0, 0.6, -0.8), polarization=(1.0, 0.0, 0.0),
                                          source_time=mw.CW(frequency=FREQUENCY))], device='cuda')


def gmres(rtol=1e-6, maxiter=1500):
    return StoppingRule(method='gmres', rtol=rtol, maxiter=maxiter, restart=maxiter, preconditioner='none',
                        stagnation_window=maxiter)


def config(edge, backend, stopping, **kwargs):
    extra = {}
    if backend == 'pfft':
        extra = dict(operator_backend='pfft', riesz_storage='local_rwg_v1',
                     pfft=kwargs.pop('pfft', mw.PFFTConfig(solve_dual_tolerance=1e-4)))
    return mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=edge), stopping=stopping,
                        **extra, **kwargs)


def prepared(scene, edge, backend, stopping=None, **kwargs):
    return mw.Simulation.mom(scene, frequency=FREQUENCY,
                             config=config(edge, backend, stopping or gmres(), **kwargs)).prepare().solver


def _random(n, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return [torch.randn(n, generator=generator, device='cuda', dtype=torch.complex128) for _ in range(2)]


def _check_action(fast, matrix):
    x, y = _random(matrix.shape[0], 3)
    action = fast.apply(x)
    assert float((action-matrix@x).norm()/(matrix@x).norm()) < 3e-3
    # Z is symmetric: <y, Z x> = <Z^H y, x>.
    left, right = torch.vdot(y, action), torch.vdot(fast.adjoint(y), x)
    assert abs(complex(left-right))/abs(complex(left)) < 1e-6


@pytest.mark.parametrize('gap', [None, 0.15*WAVELENGTH])
def test_precorrection_entries_equal_the_dense_single_layer_and_actions_agree(gap):
    side = 0.6*WAVELENGTH
    scene = plates(side, gap)
    dense = prepared(scene, side/10, 'dense').handle
    fast = prepared(scene, side/10, 'pfft').handle
    assert fast.metadata['operator_backend'] == 'pfft' and fast.metadata['pfft']['version']
    omega_mu = 2*math.pi*FREQUENCY*MU_0
    correction = fast.precorrection
    exact = correction.delta_d+fast.region.grid_entries(correction.rows, correction.cols)[0]
    reference = dense.matrix[correction.rows, correction.cols]/(1j*omega_mu)
    # Same S-1 face-pair rules, classification and orders as the dense assembly.
    assert float((exact-reference).abs().max()/reference.abs().max()) < 1e-12
    if gap is not None:
        first = fast.system.J_slices[0].stop
        assert bool(((correction.rows < first) & (correction.cols >= first)).any())
    _check_action(fast, dense.matrix)
    x, _ = _random(dense.shape[0], 4)
    # Gram metric equals the dense S-1 Riesz map.
    torch.testing.assert_close(fast.metric.apply(x), dense.test_riesz.to(x.dtype)@x, rtol=1e-12, atol=1e-18)


def test_public_plate_rcs_and_iterations_match_dense():
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner
    side = WAVELENGTH
    directions = torch.nn.functional.normalize(torch.tensor(
        [[0., 0., 1.], [0., -0.6, 0.8], [0.6, 0., 0.8], [0., 0.6, 0.8], [0.5, 0.5, -0.7]],
        device='cuda', dtype=torch.float64), dim=-1)
    records = {}
    for backend in ('dense', 'pfft'):
        for factory in (None, CalderonPreconditioner()):
            result = mw.Simulation.mom(plates(side), frequency=FREQUENCY,
                                       config=config(side/12, backend, gmres(), right_preconditioner=factory)).run()
            assert result.solver_stats['converged']
            records[backend, factory is not None] = (result.solver_stats['iterations'],
                                                    result.surface_currents.radar_cross_section(directions),
                                                    result.surface_currents.J)
    for calderon in (False, True):
        (dense_it, dense_rcs, dense_j), (fast_it, fast_rcs, fast_j) = records['dense', calderon], records['pfft', calderon]
        assert float(((fast_rcs-dense_rcs).abs()/dense_rcs.abs().max()).max()) < 5e-3
        assert float((fast_j-dense_j).norm()/dense_j.norm()) < 1e-2
        # The accelerator perturbs the ill-conditioned plain EFIE (about +15 % iterations at
        # rtol 1e-6); the Calderon-preconditioned counts agree within a few iterations.
        assert abs(fast_it-dense_it) <= (3 if calderon else 0.25*dense_it)
    assert records['pfft', True][0] < records['pfft', False][0]


def test_pfft_efie_fails_closed():
    from witwin.maxwell.mom.efie_open import validate_efie_config
    from witwin.maxwell.mom.pfft_efie import edge_face_slots
    from witwin.maxwell.mom.open_surface import OpenRWGBasis, rectangular_plate_mesh
    with pytest.raises(ValueError, match='local_rwg_v1'):
        mw.MoMConfig(operator_backend='pfft', stopping=gmres())
    validate_efie_config(config(0.1, 'pfft', gmres()))
    half = OpenRWGBasis(rectangular_plate_mesh((1., 1.), (3, 3)), boundary_half_functions=True)
    with pytest.raises(ValueError, match='two-face'):
        edge_face_slots(half)
    with pytest.raises(ValueError):
        config(0.05*WAVELENGTH, 'pfft', StoppingRule(method='direct'))
