"""Sharp-edged and thin dielectric bodies in the dense PMCHWT path.

Studies, each written incrementally to one JSON report:

* ``cube``: dielectric cube (lossless eps 2 and 6, lossy eps 4 with
  conductivity), three-level uniform and edge-graded refinement. Integrated
  observables (far field, extinction, scattering, volume absorption) are
  reported separately from near fields sampled within lambda/20 of edges and
  corners, with observed self-convergence orders for refinement ratio 2.
* ``slab``: thin closed slab, same refinement and observable split.
* ``rounded``: rounded cube at a fixed mesh size with the rounding radius
  swept to zero, checking continuity toward the sharp cube.
* ``reciprocity``: two plane waves on one cube, p2 . F1(-d2) = p1 . F2(-d1),
  plus the optical-theorem / energy balance recorded in every case.

This is self-convergence evidence only; no independent reference is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, EPSILON_0
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom import primitive_mesh as pm
from witwin.maxwell.mom.primitive_mesh import EdgeGrading, SurfaceMeshConfig, rounded_box_mesh
from witwin.maxwell.mom.quadrature import gauss_rule


FREQUENCY = 3e8
WAVELENGTH = C_0/FREQUENCY
CUBE = .2
SLAB = (.3, .3, .02)
INTEGRATION = IntegrationConfig(order=4, singular_order=6)
DIRECTION = (.3, .2, math.sqrt(1-.13))
POLARIZATION = (math.sqrt(1-.13), 0., -.3)
LOSSY_SIGMA = 2*math.pi*FREQUENCY*EPSILON_0  # Im(eps_r) = 1 at FREQUENCY
MATERIALS = {'eps2': dict(eps_r=2.), 'eps6': dict(eps_r=6.), 'eps4_lossy': dict(eps_r=4., sigma_e=LOSSY_SIGMA)}
# Elements per eps=6 interior wavelength. Every material uses the same mesh
# per level; the edge-graded family keeps the bulk size h with min_size h/4
# and growth ratio 1.5, so both families halve h between levels.
LEVELS = (8., 16., 32.)
NEAR_DISTANCES = (WAVELENGTH/20, WAVELENGTH/80, WAVELENGTH/320)


def _normalize(vector):
    norm = math.sqrt(sum(v*v for v in vector))
    return tuple(v/norm for v in vector)


def sphere_rule(order=8):
    u, weights = gauss_rule(order, device='cuda', dtype=torch.float64)
    z = 2*u-1
    phi = torch.arange(2*order, device='cuda', dtype=torch.float64)*math.pi/order
    ring = (1-z*z).sqrt()
    directions = torch.stack((ring[:, None]*phi.cos(), ring[:, None]*phi.sin(), z[:, None].expand(-1, len(phi))), -1)
    return directions.reshape(-1, 3), (2*weights[:, None].expand(-1, len(phi))*math.pi/order).reshape(-1)


def box_volume_rule(size, cells=4, order=3):
    """Composite tensor Gauss rule [m^3] strictly inside an origin-centred box.

    ``cells`` spans the longest side; shorter sides get proportionally fewer
    cells (at least one), so thin slabs are not sampled densely next to their
    faces, where every interior field query is near-singular.
    """
    u, w = gauss_rule(order, device='cuda', dtype=torch.float64)
    axes, weights = [], []
    longest = max(size)
    for length in size:
        count = max(1, round(cells*length/longest))
        edges = torch.linspace(-length/2, length/2, count+1, device='cuda', dtype=torch.float64)
        axes.append((edges[:-1, None]+(edges[1:]-edges[:-1])[:, None]*u).reshape(-1))
        weights.append(((edges[1:]-edges[:-1])[:, None]*w).reshape(-1))
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    weight = (weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]).reshape(-1)
    return grid, weight


def near_points(size, distance):
    """Exterior and interior samples near an edge midpoint, a corner and a face centre."""
    a = torch.tensor(size, device='cuda', dtype=torch.float64)/2
    edge = torch.tensor([a[0], a[1], 0.], device='cuda', dtype=torch.float64)
    corner = a.clone()
    face = torch.tensor([a[0], 0., 0.], device='cuda', dtype=torch.float64)
    bisector = torch.tensor([1., 1., 0.], device='cuda', dtype=torch.float64)/math.sqrt(2)
    diagonal = torch.ones(3, device='cuda', dtype=torch.float64)/math.sqrt(3)
    normal = torch.tensor([1., 0., 0.], device='cuda', dtype=torch.float64)
    exterior = torch.stack((edge+distance*bisector, corner+distance*diagonal, face+distance*normal))
    # Interior samples stay inside thin bodies: at most half the smallest half-size.
    inner = min(distance, .5*float(a.min()))
    interior = torch.stack((edge-inner*bisector, corner-inner*diagonal, face-inner*normal))
    return exterior, interior, ('edge', 'corner', 'face')


def run_case(geometry, material, mesh, *, size, direction=DIRECTION, polarization=POLARIZATION, absorption=False,
             near_distances=NEAR_DISTANCES, pair_batch=16):
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    scene = mw.Scene(structures=[mw.Structure(geometry, mw.Material(**material))],
                     sources=[mw.PlaneWave(direction=_normalize(direction), polarization=_normalize(polarization))])
    config = mw.MoMConfig(integration=replace(INTEGRATION, pair_batch=pair_batch), surface_mesh=mesh,
                          query_config=mw.FieldQueryConfig(max_order=128))
    result = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).run()
    data = result.surface_currents
    directions, weights = sphere_rule()
    sections = data.plane_wave_cross_sections(directions, weights)
    far, _ = data.far_field(directions)
    near_disjoint = data.handle.metadata.get('quadrature_error_estimate', {}).get('near_disjoint', {})
    row = {'triangles': int(data.handle.system.total_faces), 'rwg_edges': int(data.handle.system.total_edges),
           'assembly_seconds': data.handle.metadata.get('assembly_seconds'), 'near_fallback': near_disjoint.get('fallback'),
           'touching_anisotropic': data.handle.metadata.get('quadrature_error_estimate', {}).get('touching_anisotropic'),
           'scattering_m2': float(sections['scattering_m2']), 'extinction_m2': float(sections['extinction_m2'])}
    if absorption:
        points, volume = box_volume_rule(size)
        intensity = float(sections['incident_intensity_W_per_m2'])
        row['absorption_m2'] = float(data.absorbed_power(points, volume))/intensity
        row['energy_balance_relative'] = abs(row['extinction_m2']-row['scattering_m2']-row['absorption_m2'])/row['extinction_m2']
    else:
        row['optical_theorem_relative'] = abs(row['extinction_m2']-row['scattering_m2'])/row['extinction_m2']
    near = {}
    for distance in near_distances:
        exterior, interior, labels = near_points(size, distance)
        e_out, h_out = data.query_fields(exterior, region_id='exterior', field_kind='total')
        e_in, h_in = data.query_fields(interior, region_id='interior', field_kind='total')
        near[f'{distance:.6g}'] = {'labels': labels, 'E_exterior': e_out, 'H_exterior': h_out,
                                   'E_interior': e_in, 'H_interior': h_in}
    torch.cuda.synchronize()
    row.update(wall_seconds=time.perf_counter()-start, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
               solver=result.stats())
    tensors = {'far_E': far, 'weights': weights, 'near': near}
    del result
    torch.cuda.empty_cache()
    return row, tensors


def _weighted(value, weights):
    return float((weights*value.abs().square().sum(-1)).sum().sqrt())


def _order(d01, d12):
    return math.log2(d01/d12) if d01 > 0 and d12 > 0 else None


def convergence_orders(levels):
    """Observed orders from three successive ratio-2 levels, integrated vs near split."""
    (r0, t0), (r1, t1), (r2, t2) = levels
    out = {'integrated': {}, 'near': {}}
    for key in ('scattering_m2', 'extinction_m2', 'absorption_m2'):
        if key in r0:
            out['integrated'][key] = _order(abs(r0[key]-r1[key]), abs(r1[key]-r2[key]))
    w = t2['weights']
    out['integrated']['far_E'] = _order(_weighted(t0['far_E']-t1['far_E'], w), _weighted(t1['far_E']-t2['far_E'], w))
    for distance, sample in t2['near'].items():
        for field in ('E_exterior', 'H_exterior', 'E_interior', 'H_interior'):
            for i, label in enumerate(sample['labels']):
                a, b, c = (t['near'][distance][field][i] for t in (t0, t1, t2))
                out['near'][f'{field}@{label}@{distance}'] = _order(float((a-b).norm()), float((b-c).norm()))
    return out


def edge_length(epw):
    return WAVELENGTH/math.sqrt(6.)/epw  # shortest interior wavelength over the studied materials


def _mesh(epw, graded):
    h = edge_length(epw)
    return SurfaceMeshConfig(elements_per_wavelength=None, max_edge_length=h,
                             edge_grading=EdgeGrading(h/4, 1.5) if graded else None)


def refinement_study(name, geometry_factory, size, materials, report, write, pair_batch, levels_epw=LEVELS):
    for material_name in materials:
        for graded in (False, True):
            levels = []
            for epw in levels_epw:
                row, tensors = run_case(geometry_factory(), MATERIALS[material_name], _mesh(epw, graded), size=size,
                                        absorption='lossy' in material_name, pair_batch=pair_batch)
                row.update(study=name, material=material_name, graded=graded, elements_per_wavelength=epw)
                report['cases'].append(row)
                levels.append((row, tensors))
                write()
                print(json.dumps({k: row[k] for k in row if k != 'solver'}, default=str), flush=True)
            for start in range(len(levels)-2):
                report['orders'].append({'study': name, 'material': material_name, 'graded': graded,
                                         'levels_epw': list(levels_epw[start:start+3]),
                                         **convergence_orders(levels[start:start+3])})
            write()


def rounded_study(report, write, pair_batch, epw=16.):
    h = edge_length(epw)
    size = (CUBE,)*3
    for radius in (.1*CUBE, .03*CUBE, .01*CUBE, .003*CUBE, 0.):
        surface = rounded_box_mesh(size, radius, edge_length=h)
        geometry = mw.Mesh(surface.vertices, surface.faces, recenter=False, fill_mode='solid')
        quality = pm.mesh_quality(surface.vertices, surface.faces)
        try:
            # Only lambda/20 samples: closer interior samples near a sharp edge
            # can fall outside the rounded body.
            row, _ = run_case(geometry, MATERIALS['eps6'], None, size=size, pair_batch=pair_batch,
                              near_distances=(WAVELENGTH/20,))
        except RuntimeError as error:
            # Sliver arc strips (r << h) exceed the near-panel quadrature limit;
            # the solver fails closed and the sweep records it.
            row = {'status': 'failed_closed', 'error': str(error)}
        row.update(study='rounded', radius_m=radius, edge_length_m=h, max_aspect_ratio=quality['max_aspect_ratio'])
        report['cases'].append(row)
        write()
        print(json.dumps({k: row[k] for k in row if k != 'solver'}, default=str), flush=True)


def reciprocity_study(report, write, pair_batch, epw=16.):
    first = (_normalize(DIRECTION), _normalize(POLARIZATION))
    second = (_normalize((-.5, .7, .2)), _normalize((.7, .5, 0.)))
    results = []
    for direction, polarization in (first, second):
        scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(CUBE,)*3, rotation=(.2, .1, .3)),
                                                  mw.Material(**MATERIALS['eps4_lossy']))],
                         sources=[mw.PlaneWave(direction=direction, polarization=polarization)])
        config = mw.MoMConfig(integration=replace(INTEGRATION, pair_batch=pair_batch), surface_mesh=_mesh(epw, False))
        results.append(mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).run().surface_currents)
    back = lambda data, d: data.far_field(-torch.tensor([d], device='cuda', dtype=torch.float64))[0][0]
    p1 = torch.tensor(first[1], device='cuda', dtype=torch.complex128)
    p2 = torch.tensor(second[1], device='cuda', dtype=torch.complex128)
    left = (p2*back(results[0], second[0])).sum()
    right = (p1*back(results[1], first[0])).sum()
    row = {'study': 'reciprocity', 'left': [left.real.item(), left.imag.item()],
           'right': [right.real.item(), right.imag.item()],
           'relative_difference': float((left-right).abs()/torch.maximum(left.abs(), right.abs()))}
    report['cases'].append(row)
    write()
    print(json.dumps(row), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=('cube', 'slab', 'rounded', 'reciprocity'), nargs='+',
                        default=['cube', 'slab', 'rounded', 'reciprocity'])
    parser.add_argument('--materials', nargs='+', default=list(MATERIALS), choices=list(MATERIALS))
    parser.add_argument('--levels', nargs='+', type=float, default=list(LEVELS),
                        help='Elements per eps=6 interior wavelength; successive levels should differ by 2x')
    parser.add_argument('--pair-batch', type=int, default=16)
    args = parser.parse_args(argv)
    report = {'frequency_hz': FREQUENCY, 'cube_m': CUBE, 'slab_m': SLAB, 'integration': asdict(INTEGRATION),
              'levels_elements_per_interior_wavelength_eps6': args.levels, 'cases': [], 'orders': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Self-convergence on identical physical samples; no independent reference.'}

    def write():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
    if 'cube' in args.study:
        refinement_study('cube', lambda: mw.Box(size=(CUBE,)*3), (CUBE,)*3, args.materials, report, write,
                         args.pair_batch, tuple(args.levels))
    if 'slab' in args.study:
        refinement_study('slab', lambda: mw.Box(size=SLAB), SLAB, [m for m in ('eps6', 'eps4_lossy') if m in args.materials],
                         report, write, args.pair_batch, tuple(args.levels))
    if 'rounded' in args.study:
        rounded_study(report, write, args.pair_batch)
    if 'reciprocity' in args.study:
        reciprocity_study(report, write, args.pair_batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
