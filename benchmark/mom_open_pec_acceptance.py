"""Acceptance studies for the open-surface PEC EFIE (dense, vacuum).

Each study writes one JSON record. References are independent of the solver:

* ``plate``: square-plate monostatic/bistatic RCS and total scattering cross
  section under uniform refinement, with the observed order of the successive
  differences, physical optics as a large-plate sanity bound, and near-field
  samples within lambda/20 of an edge reported separately.
* ``disk``: low-frequency electric and magnetic polarizabilities of a thin
  circular PEC disk, alpha_e = 16/3 eps0 a^3 (field in the disk plane) and
  alpha_m = -8/3 a^3 (field normal to it), and the Rayleigh cross section
  128/(27 pi) k^4 a^6 at normal incidence.
* ``strip``: center-fed strip-dipole input impedance (delta gap) against the
  induced-EMF closed form for the equivalent wire radius w/4.
* ``checks``: bistatic reciprocity and lossless power balance.
* ``iterations``: GMRES iteration counts with and without the open-surface
  preconditioners under h refinement and over electrical size.

Run, for example::

    CUDA_VISIBLE_DEVICES=1 python -m benchmark.mom_open_pec_acceptance plate --output plate.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0 as C0, EPSILON_0, ETA_0, MU_0
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.quadrature import triangle_rule

ETA0 = ETA_0


def sphere_rule(n_theta=32, n_phi=64):
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2*math.pi*np.arange(n_phi)/n_phi
    ct, ph = np.meshgrid(x, phi, indexing='ij')
    st = np.sqrt(1-ct**2)
    directions = np.stack((st*np.cos(ph), st*np.sin(ph), ct), -1).reshape(-1, 3)
    return (torch.tensor(directions, device='cuda'), torch.tensor(np.repeat(w, n_phi)*2*math.pi/n_phi, device='cuda'))


def plane_wave(frequency, direction=(0., 0., -1.), polarization=(1., 0., 0.)):
    return mw.PlaneWave(direction=direction, polarization=polarization, source_time=mw.CW(frequency=frequency))


def observed_order(values):
    """Order from three successive halvings: log2(|q1-q2|/|q2-q3|)."""
    if len(values) < 3:
        return None
    a, b, c = values[-3:]
    if b == c or a == b:
        return None
    return math.log2(abs(a-b)/abs(b-c))


def solve(scene, frequency, edge):
    started = time.perf_counter()
    result = mw.Simulation.mom(scene, frequency=frequency,
                               config=mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=edge))).run()
    return result, time.perf_counter()-started


def plate(args):
    frequency, wavelength = args.frequency, C0/args.frequency
    side = args.side*wavelength
    directions, weights = sphere_rule()
    bistatic = torch.tensor([[math.sin(math.radians(30)), 0., math.cos(math.radians(30))]], device='cuda',
                            dtype=torch.float64)
    backscatter = torch.tensor([[0., 0., 1.]], device='cuda', dtype=torch.float64)
    offset = wavelength/40  # lambda/40 from the edge: inside the lambda/20 edge layer
    near_points = torch.tensor([[side/2-offset, 0., offset], [side/2+offset, 0., offset],
                                [0., 0., wavelength/4]], device='cuda', dtype=torch.float64)
    records = []
    for cells in args.cells:
        scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(side, side, 0)), mw.Material.pec())],
                         sources=[plane_wave(frequency)], device='cuda')
        result, seconds = solve(scene, frequency, side/cells)
        data = result.surface_currents
        sections = data.plane_wave_cross_sections(directions, weights)
        electric, _ = data.query_fields(near_points, field_kind='scattered')
        records.append({'cells_per_side': cells, 'h_over_lambda': 1/cells*args.side, 'edges': len(data.J),
                        'monostatic_rcs_m2': float(data.radar_cross_section(backscatter)[0]),
                        'bistatic30_rcs_m2': float(data.radar_cross_section(bistatic)[0]),
                        'scattering_m2': float(sections['scattering_m2']),
                        'extinction_m2': float(sections['extinction_m2']),
                        'near_edge_inside_Ex': [float(electric[0, 0].real), float(electric[0, 0].imag)],
                        'near_edge_inside_absE': float(electric[0].abs().square().sum().sqrt()),
                        'near_edge_outside_absE': float(electric[1].abs().square().sum().sqrt()),
                        'center_quarter_wave_absE': float(electric[2].abs().square().sum().sqrt()),
                        'seconds': seconds, 'converged': result.solver_stats['converged']})
        print(json.dumps(records[-1]), flush=True)
    summary = {}
    for key in ('monostatic_rcs_m2', 'bistatic30_rcs_m2', 'scattering_m2', 'near_edge_inside_absE',
                'near_edge_outside_absE', 'center_quarter_wave_absE'):
        values = [r[key] for r in records]
        summary[key] = {'values': values, 'observed_order': observed_order(values),
                        'finest_relative_change': abs(values[-1]-values[-2])/abs(values[-1]) if len(values) > 1 else None}
    area = side*side
    return {'study': 'plate', 'side_over_lambda': args.side, 'frequency_hz': frequency, 'records': records,
            'summary': summary, 'physical_optics_monostatic_m2': 4*math.pi*area**2/wavelength**2,
            'near_points_m': near_points.cpu().tolist(), 'edge_offset_over_lambda': 1/40}


def current_moments(data):
    """Electric dipole p = (i/omega) int J and magnetic dipole m = 1/2 int r x J."""
    basis, coefficients = data.basis, data.J
    points, weights = triangle_rule(basis.mesh.triangles, 6)
    current = basis.current(coefficients, points)
    w = weights.to(current.dtype)
    integral = torch.einsum('tqc,tq->c', current, w)
    moment = 0.5*torch.einsum('tqc,tq->c', torch.linalg.cross(points.to(current.dtype), current), w)
    return integral, moment


def disk(args):
    radius = args.radius
    records = []
    for ka in args.ka:
        frequency = ka*C0/(2*math.pi*radius)
        k, omega = 2*math.pi*frequency/C0, 2*math.pi*frequency
        for rings in args.rings:
            edge = radius/rings
            normal = mw.Scene(structures=[mw.Structure(mw.Cylinder(radius=radius, height=0), mw.Material.pec())],
                              sources=[plane_wave(frequency)], device='cuda')
            result, seconds = solve(normal, frequency, edge)
            integral, _ = current_moments(result.surface_currents)
            p = 1j/omega*integral
            alpha_e = complex(p[0])/EPSILON_0  # E0 = 1 V/m along x
            directions, weights = sphere_rule(24, 48)
            scattering = float(result.surface_currents.plane_wave_cross_sections(directions, weights)['scattering_m2'])
            # Edge-on incidence along +x with E along y (in plane) and H = x_hat x E / eta along +z (normal).
            edge_on = mw.Scene(structures=[mw.Structure(mw.Cylinder(radius=radius, height=0), mw.Material.pec())],
                               sources=[plane_wave(frequency, (1., 0., 0.), (0., 1., 0.))], device='cuda')
            second, _ = solve(edge_on, frequency, edge)
            integral_b, moment = current_moments(second.surface_currents)
            h_normal = complex(second.surface_currents.incident_carrier.evaluate(
                torch.zeros((1, 3), device='cuda', dtype=torch.float64), second.solver.handle.exterior)[1][0, 2])
            alpha_m = complex(moment[2])/h_normal
            alpha_e_edge = complex(1j/omega*integral_b[1])/EPSILON_0
            rayleigh = 128/(27*math.pi)*k**4*radius**6
            records.append({'ka': ka, 'rings': rings, 'edges': len(result.surface_currents.J),
                            'alpha_e_over_a3': alpha_e.real/radius**3, 'alpha_e_imag_over_a3': alpha_e.imag/radius**3,
                            'alpha_e_edge_on_over_a3': alpha_e_edge.real/radius**3,
                            'alpha_m_over_a3': alpha_m.real/radius**3,
                            'reference_alpha_e_over_a3': 16/3, 'reference_alpha_m_over_a3': -8/3,
                            'alpha_e_relative_error': abs(alpha_e.real/radius**3-16/3)/(16/3),
                            'alpha_m_relative_error': abs(alpha_m.real/radius**3+8/3)/(8/3),
                            'scattering_m2': scattering, 'rayleigh_scattering_m2': rayleigh,
                            'scattering_relative_deviation': scattering/rayleigh-1, 'seconds': seconds})
            print(json.dumps(records[-1]), flush=True)
    return {'study': 'disk', 'radius_m': radius, 'records': records,
            'reference': 'thin PEC disk static polarizabilities alpha_e=16/3 eps0 a^3, alpha_m=-8/3 a^3; '
                         'Rayleigh cross section 128/(27 pi) k^4 a^6 (leading order in ka)'}


def induced_emf(length, radius, wavelength):
    from scipy.special import sici
    k = 2*math.pi/wavelength
    kl = k*length

    def si(x): return sici(x)[0]
    def ci(x): return sici(x)[1]
    gamma = 0.5772156649015329
    rm = ETA0/(2*math.pi)*(gamma+math.log(kl)-ci(kl)+0.5*math.sin(kl)*(si(2*kl)-2*si(kl))
                           + 0.5*math.cos(kl)*(gamma+math.log(kl/2)+ci(2*kl)-2*ci(kl)))
    xm = ETA0/(4*math.pi)*(2*si(kl)+math.cos(kl)*(2*si(kl)-si(2*kl))
                            - math.sin(kl)*(2*ci(kl)-ci(2*kl)-ci(2*k*radius**2/length)))
    scale = math.sin(kl/2)**2
    return rm/scale, xm/scale


def strip(args):
    from witwin.maxwell.mom.efie_open import OpenPECOperator, compile_open_pec_scene
    frequency, wavelength = args.frequency, C0/args.frequency
    width = args.width*wavelength
    records = []
    for length_ratio in args.lengths:
        length = length_ratio*wavelength
        segments = 2*math.ceil(length/(args.segment*wavelength)/2)
        scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(width, 0, length)), mw.Material.pec())],
                         sources=[plane_wave(frequency, (1., 0., 0.), (0., 0., 1.))], device='cuda')
        config = mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=length/segments))
        system, exterior = compile_open_pec_scene(scene, frequency, config)
        handle = OpenPECOperator(system, exterior, scene.sources, config)
        basis = system.bodies[0].basis
        vertices = basis.mesh.vertices
        midpoint = vertices[basis.edges].mean(1)
        tangent = vertices[basis.edges[:, 1]]-vertices[basis.edges[:, 0]]
        gap = (midpoint[:, 2].abs() < 1e-9*length) & (tangent[:, 2].abs() < 1e-9*length)
        plus_side = basis.mesh.centroids[basis.plus_faces][:, 2] < 0
        sign = torch.where(plus_side, 1.0, -1.0).to(torch.float64)
        voltage = 1.0
        rhs = torch.zeros(basis.n_edges, device='cuda', dtype=torch.complex128)
        rhs[gap] = (-voltage*sign[gap]*basis.edge_lengths[gap]).to(rhs.dtype)
        coefficients, log = solve_reference(handle, rhs, StoppingRule(method='direct'))
        current = (coefficients[gap]*(sign[gap]*basis.edge_lengths[gap]).to(coefficients.dtype)).sum()
        impedance = voltage/complex(current)
        reference = induced_emf(length, width/4, wavelength)
        # Z = V/I under exp(-i omega t); the engineering (exp(+j omega t)) reactance is -Im Z.
        records.append({'length_over_lambda': length_ratio, 'segments': segments, 'edges': basis.n_edges,
                        'gap_edges': int(gap.sum()), 'R_in_ohm': impedance.real, 'X_in_ohm': -impedance.imag,
                        'Im_Z_exp_minus_iwt_ohm': impedance.imag,
                        'induced_emf_R_ohm': reference[0], 'induced_emf_X_ohm': reference[1],
                        'converged': log['converged']})
        print(json.dumps(records[-1]), flush=True)
    def crossing(key):
        for a, b in zip(records, records[1:]):
            if a[key] <= 0 <= b[key] or a[key] >= 0 >= b[key]:
                t = a[key]/(a[key]-b[key])
                return a['length_over_lambda']+t*(b['length_over_lambda']-a['length_over_lambda'])
        return None
    return {'study': 'strip', 'width_over_lambda': args.width, 'equivalent_wire_radius_over_lambda': args.width/4,
            'records': records, 'efie_resonance_length_over_lambda': crossing('X_in_ohm'),
            'induced_emf_resonance_length_over_lambda': crossing('induced_emf_X_ohm'),
            'reactance_convention': 'engineering exp(+j omega t): X = -Im(Z) of the exp(-i omega t) solve',
            'reference': 'induced-EMF sinusoidal-current closed form (thin wire of radius w/4); trend reference'}


def checks(args):
    frequency, wavelength = args.frequency, C0/args.frequency
    side = args.side*wavelength
    def unit(v):
        v = np.asarray(v, float)
        return tuple(v/np.linalg.norm(v))
    a_dir, a_pol = unit((0.2, 0.1, -1.0)), unit((1.0, 0.0, 0.2))
    b_dir, b_pol = unit((-0.5, 0.3, -0.8)), unit((0.0, 0.8, 0.3))
    records = []
    for cells in args.cells:
        def run(direction, polarization):
            scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(side, 0.7*side, 0)), mw.Material.pec())],
                             sources=[plane_wave(frequency, direction, polarization)], device='cuda')
            return solve(scene, frequency, side/cells)[0].surface_currents
        first, second = run(a_dir, a_pol), run(b_dir, b_pol)
        fa, _ = first.far_field(-torch.tensor([b_dir], device='cuda', dtype=torch.float64))
        fb, _ = second.far_field(-torch.tensor([a_dir], device='cuda', dtype=torch.float64))
        left = complex((fa[0]*torch.tensor(b_pol, device='cuda')).sum())
        right = complex((fb[0]*torch.tensor(a_pol, device='cuda')).sum())
        sections = first.plane_wave_cross_sections(*sphere_rule())
        scattering, extinction = float(sections['scattering_m2']), float(sections['extinction_m2'])
        records.append({'cells': cells, 'edges': len(first.J), 'reciprocity_relative_error': abs(left-right)/abs(left),
                        'scattering_m2': scattering, 'extinction_m2': extinction,
                        'power_balance_relative_error': abs(extinction-scattering)/scattering})
        print(json.dumps(records[-1]), flush=True)
    return {'study': 'checks', 'side_over_lambda': args.side, 'records': records}


PRECONDITIONERS = ('none', 'row_norm', 'calderon', 'calderon_zero_flux')


def _solve_iterations(side, cells, frequency, method, rtol, maxiter):
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner, preconditioned_residual
    factory = {'none': None, 'row_norm': None, 'calderon': CalderonPreconditioner(),
               'calderon_zero_flux': CalderonPreconditioner(open_edge_rule='zero_flux')}[method]
    stopping = StoppingRule(method='gmres', rtol=rtol, maxiter=maxiter, restart=maxiter,
                            preconditioner='row_norm' if method == 'row_norm' else 'none',
                            stagnation_window=maxiter)
    scene = mw.Scene(structures=[mw.Structure(mw.Box(size=(side, side, 0)), mw.Material.pec())],
                     sources=[plane_wave(frequency, (0.0, 0.6, -0.8), (1.0, 0.0, 0.0))], device='cuda')
    config = mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=side/cells), stopping=stopping,
                          right_preconditioner=factory)
    started = time.perf_counter()
    prepared = mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()
    setup = time.perf_counter()-started
    runtime = prepared.solver
    result = prepared.run()
    stats = result.solver_stats
    record = {'method': method, 'iterations': stats['iterations'], 'converged': stats['converged'],
              'relative_residual': stats['relative_residual'], 'edges': runtime.handle.shape[0],
              'setup_seconds': setup, 'solve_seconds': stats.get('solve_seconds')}
    if runtime.right_preconditioner is not None:
        record['receipt'] = {k: v for k, v in runtime.right_preconditioner_receipt.items()
                             if isinstance(v, (int, float, str, bool))}
        record.update(preconditioned_residual(runtime.handle, result.surface_currents.J, runtime.rhs,
                                              runtime.right_preconditioner.apply))
    reference = solve_reference(runtime.handle, runtime.rhs, StoppingRule(method='direct'))[0]
    record['solution_relative_error'] = float((result.surface_currents.J-reference).norm()/reference.norm())
    return record


def iterations(args):
    records = []
    frequency = args.frequency
    wavelength = C0/frequency
    for sweep, cases in (('h_refinement', [(args.side*wavelength, cells) for cells in args.cells]),
                         ('electrical_size', [(size*wavelength, args.size_cells) for size in args.sizes])):
        for side, cells in cases:
            for method in args.methods:
                record = _solve_iterations(side, cells, frequency, method, args.rtol, args.maxiter)
                record.update(sweep=sweep, side_over_lambda=side/wavelength, cells=cells)
                records.append(record)
                print(json.dumps(record), flush=True)
    return {'study': 'iterations', 'rtol': args.rtol, 'maxiter': args.maxiter, 'records': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='study', required=True)
    p = sub.add_parser('plate')
    p.add_argument('--side', type=float, default=1.0)
    p.add_argument('--cells', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--frequency', type=float, default=3e8)
    d = sub.add_parser('disk')
    d.add_argument('--radius', type=float, default=0.1)
    d.add_argument('--ka', type=float, nargs='+', default=[0.05, 0.1])
    d.add_argument('--rings', type=int, nargs='+', default=[6, 12, 24])
    s = sub.add_parser('strip')
    s.add_argument('--width', type=float, default=0.01)
    s.add_argument('--segment', type=float, default=1/80)
    s.add_argument('--lengths', type=float, nargs='+', default=[0.40, 0.44, 0.46, 0.48, 0.50, 0.52, 0.56, 0.60])
    s.add_argument('--frequency', type=float, default=3e8)
    c = sub.add_parser('checks')
    c.add_argument('--side', type=float, default=0.8)
    c.add_argument('--cells', type=int, nargs='+', default=[8, 16])
    c.add_argument('--frequency', type=float, default=3e8)
    it = sub.add_parser('iterations')
    it.add_argument('--side', type=float, default=1.0)
    it.add_argument('--cells', type=int, nargs='*', default=[8, 16, 32])
    it.add_argument('--sizes', type=float, nargs='*', default=[0.01, 0.1, 1.0, 3.0])
    it.add_argument('--size-cells', type=int, default=20)
    it.add_argument('--methods', nargs='+', default=list(PRECONDITIONERS))
    it.add_argument('--rtol', type=float, default=1e-6)
    it.add_argument('--maxiter', type=int, default=1500)
    it.add_argument('--frequency', type=float, default=3e8)
    for item in (p, d, s, c, it):
        item.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    record = {'plate': plate, 'disk': disk, 'strip': strip, 'checks': checks,
              'iterations': iterations}[args.study](args)
    record['device'] = torch.cuda.get_device_name()
    with open(args.output, 'w') as stream:
        json.dump(record, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
