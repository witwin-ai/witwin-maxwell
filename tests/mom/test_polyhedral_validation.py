"""Opt-in CUDA physics checks for sharp-edged and thin dielectric bodies.

Run with ``MOM_POLYHEDRAL_GPU=1``. Every case is a coarse dense PMCHWT solve
(at most about 3500 RWG edges); the full three-level study lives in
``benchmark/mom_polyhedral_convergence.py``. Tolerances are physical bounds
for these coarse meshes, not converged accuracy claims.
"""
import json
import math
import os

import pytest
import torch

import witwin.maxwell as mw
from benchmark import mom_polyhedral_convergence as study


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('MOM_POLYHEDRAL_GPU') != '1',
                                reason='Opt-in CUDA validation: set MOM_POLYHEDRAL_GPU=1')
CUBE = (study.CUBE,)*3


def record(name, value):
    # Calibration evidence for thresholds; visible with ``-s``.
    print(json.dumps({'case': name, **value}, default=str), flush=True)


@pytest.mark.parametrize('material', ['eps2', 'eps6'])
@pytest.mark.parametrize('graded', [False, True])
def test_lossless_cube_satisfies_optical_theorem(material, graded):
    row, _ = study.run_case(mw.Box(size=CUBE), study.MATERIALS[material], study._mesh(8., graded), size=CUBE,
                            near_distances=())
    record(f'optical_{material}_{graded}', {k: v for k, v in row.items() if k != 'solver'})
    assert row['extinction_m2'] > 0
    # Measured 3e-5..1.3e-4 (192 and 768 triangles): far-field angular
    # quadrature plus discretization inconsistency of a lossless body.
    assert row['optical_theorem_relative'] < 1e-3


@pytest.mark.parametrize('graded', [False, True])
def test_lossy_cube_balances_extinction_scattering_and_volume_absorption(graded):
    row, _ = study.run_case(mw.Box(size=CUBE), study.MATERIALS['eps4_lossy'], study._mesh(8., graded),
                            size=CUBE, absorption=True, near_distances=())
    record(f'lossy_cube_{graded}', {k: v for k, v in row.items() if k != 'solver'})
    assert 0 < row['absorption_m2'] < row['extinction_m2']
    # Measured 2.3e-4 (uniform) and 5.4e-4 (graded); volume absorption is an
    # independent interior-field quadrature, not the extinction difference.
    assert row['energy_balance_relative'] < 3e-3


def test_thin_slab_solves_and_balances_energy():
    row, _ = study.run_case(mw.Box(size=study.SLAB), study.MATERIALS['eps4_lossy'], study._mesh(8., False),
                            size=study.SLAB, absorption=True, near_distances=())
    record('slab', {k: v for k, v in row.items() if k != 'solver'})
    # Measured 3.2e-3 with one element across the 0.02 m thickness (h/t = 2.5).
    assert row['energy_balance_relative'] < 1e-2


def test_rotated_lossy_cube_is_reciprocal():
    report = {'cases': []}
    study.reciprocity_study(report, lambda: None, 16, epw=8.)
    record('reciprocity', report['cases'][0])
    # Measured 1.1e-5 on a rotated lossy cube with 192 triangles.
    assert report['cases'][0]['relative_difference'] < 1e-4


def test_uniform_cube_integrated_quantities_converge():
    levels = [study.run_case(mw.Box(size=CUBE), study.MATERIALS['eps6'], study._mesh(epw, False), size=CUBE,
                             near_distances=(study.WAVELENGTH/20,)) for epw in (4., 8., 16.)]
    orders = study.convergence_orders(levels)['integrated']
    record('uniform_convergence', {'rows': [{k: v for k, v in r.items() if k != 'solver'} for r, _ in levels],
                                   'orders': orders})
    # Levels have 2, 4 and 8 elements per cube edge, so these orders are
    # pre-asymptotic. Measured: far field 2.03, scattering 0.87, extinction
    # 0.76; edge singularities slow the scalar cross sections. Require strict
    # contraction here; asymptotic orders belong to the three-level benchmark.
    assert orders['far_E'] is not None and orders['far_E'] > 1.5
    for key in ('scattering_m2', 'extinction_m2'):
        assert orders[key] is not None and orders[key] > .5


def _rounded_case(radius, h, max_arc_angle=math.pi/12):
    surface = study.rounded_box_mesh(CUBE, radius, edge_length=h, max_arc_angle=max_arc_angle)
    geometry = mw.Mesh(surface.vertices, surface.faces, recenter=False, fill_mode='solid')
    return study.run_case(geometry, study.MATERIALS['eps6'], None, size=CUBE, near_distances=())[0]


def test_rounded_cube_extinction_is_continuous_as_radius_vanishes():
    h = study.edge_length(8.)
    sharp = _rounded_case(0., h)['extinction_m2']
    volume = study.CUBE**3
    radii, changes = (.3*study.CUBE, .1*study.CUBE), []
    for radius in radii:
        row = _rounded_case(radius, h)
        removed = 1-study.pm.rounded_box_volume(CUBE, radius)/volume
        changes.append(abs(row['extinction_m2']-sharp)/sharp)
        record(f'rounded_{radius}', {**{k: v for k, v in row.items() if k != 'solver'},
                                     'relative_change_from_sharp': changes[-1], 'removed_volume_fraction': removed})
    # Rounding removes material where the dielectric edge field is singular,
    # E ~ rho^(nu-1) with 1/2 < nu < 1, so the shape perturbation scales as
    # r^(2 nu) rather than with the removed volume ~ r^2. Measured
    # (2026-10-28, 192/768/1200 triangles): 45.5% at r = 0.06 m, 8.5% at
    # r = 0.02 m, exponent 1.53.
    exponent = math.log(changes[1]/changes[0])/math.log(radii[1]/radii[0])
    assert 1 < exponent <= 2


@pytest.mark.parametrize('aspect', [2.5, 5., 10., 20.])
def test_thin_slab_energy_balance_holds_as_panels_outgrow_the_thickness(aspect):
    h = study.edge_length(8.)
    size = (study.SLAB[0], study.SLAB[1], h/aspect)
    row, _ = study.run_case(mw.Box(size=size), study.MATERIALS['eps4_lossy'], study._mesh(8., False), size=size,
                            absorption=True, near_distances=())
    record(f'slab_h_over_t_{aspect}', {k: v for k, v in row.items() if k != 'solver'})
    assert 0 < row['absorption_m2'] < row['extinction_m2']
    assert row['energy_balance_relative'] < 1e-2


@pytest.mark.parametrize('radius', [.002])
def test_rounded_cube_with_sliver_arc_strips_assembles_and_stays_continuous(radius):
    """One arc segment per 45 degrees: r/h = 0.04 gives 25:1 strips.

    r = 0.0006 m (85:1 strips) needs more than one short opt-in GPU window and
    belongs to the rounded study of the convergence benchmark.

    The default 15-degree sampling gives 95:1 and 315:1 strips and about 1200
    triangles, whose dense assembly exceeds a short opt-in GPU window; the
    rounded study of the convergence benchmark covers it.
    """
    h = study.edge_length(8.)
    sharp = _rounded_case(0., h)
    row = _rounded_case(radius, h, max_arc_angle=math.pi/4)
    change = abs(row['extinction_m2']-sharp['extinction_m2'])/sharp['extinction_m2']
    record(f'rounded_sliver_{radius}', {**{k: v for k, v in row.items() if k != 'solver'},
                                        'relative_change_from_sharp': change})
    assert row['near_fallback'] is not None and row['near_fallback']['pairs'] > 0
    # Edge-singular scaling measured above (exponent 1.53 from r = 0.06, 0.02 m):
    # 8.5% at 0.02 m extrapolates to 2.3% and 0.36%; allow a factor of three.
    assert change < 3*.0847*(radius/.02)**1.53
    assert row['touching_anisotropic'] is not None
    # Fixed-order joint Duffy rules on these strips gave 3.7e-3; with the
    # anisotropy-aware touching rule the measured residual is 1.1e-4.
    assert row['optical_theorem_relative'] < 1e-3
