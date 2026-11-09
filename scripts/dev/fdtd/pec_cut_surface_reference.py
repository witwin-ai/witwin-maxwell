"""Independent PEC Mie near-field control for true GPU cut surfaces.

Validation-only SciPy special functions are never used by the time marcher.
Coefficient source: https://doi.org/10.3390/electronics10020216 .
Convention: exp(-i omega t), outgoing spherical Hankel function of first kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import spherical_jn, spherical_yn
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import sphere_mesh
from benchmark.mom_fdtd import add_query_stencils


def pec_sphere_fields(radius, k, points, *, eta=376.730313668):
    """Exterior scattered Cartesian E,H for a unit +z, x-polarized plane wave."""
    points = np.asarray(points, dtype=float)
    r = np.linalg.norm(points, axis=-1)
    if np.any(r < radius*(1-1.e-14)):
        raise ValueError("The independent PEC oracle accepts exterior points only.")
    cutoff = int(np.ceil(k*radius+4*(k*radius)**(1/3)+8))
    orders = np.arange(1, cutoff+1)
    size = k*radius
    j = spherical_jn(orders, size)
    h = j+1j*spherical_yn(orders, size)
    dj = j+size*spherical_jn(orders, size, derivative=True)
    dh = dj+1j*(spherical_yn(orders, size)+size*spherical_yn(orders, size, derivative=True))
    a, b = dj/dh, j/h
    mu = points[:, 2]/r
    sine = np.sqrt(np.maximum(0., 1-mu*mu))
    phi = np.arctan2(points[:, 1], points[:, 0])
    cp, sp = np.cos(phi), np.sin(phi)
    er = np.column_stack((sine*cp, sine*sp, mu))
    et = np.column_stack((mu*cp, mu*sp, -sine))
    ep = np.column_stack((-sp, cp, np.zeros_like(r)))
    pi_previous, pi_current = np.zeros_like(r), np.ones_like(r)
    electric = np.zeros_like(points, dtype=complex)
    magnetic = np.zeros_like(points, dtype=complex)
    rho = k*r
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tau = n*mu*pi_current-(n+1)*pi_previous
        radial_function = spherical_jn(n, rho)+1j*spherical_yn(n, rho)
        derivative = spherical_jn(n, rho, derivative=True)+1j*spherical_yn(n, rho, derivative=True)
        zr = radial_function/rho
        dr = zr+derivative
        radial = n*(n+1)*sine*pi_current*zr
        mo = (cp*pi_current*radial_function)[:, None]*et-(sp*tau*radial_function)[:, None]*ep
        me = -(sp*pi_current*radial_function)[:, None]*et-(cp*tau*radial_function)[:, None]*ep
        ne = (cp*radial)[:, None]*er+(cp*tau*dr)[:, None]*et-(sp*pi_current*dr)[:, None]*ep
        no = (sp*radial)[:, None]*er+(sp*tau*dr)[:, None]*et+(cp*pi_current*dr)[:, None]*ep
        en = 1j**n*(2*n+1)/(n*(n+1))
        electric += en*(1j*an*ne-bn*mo)
        magnetic += en*(an*me+1j*bn*no)/eta
        pi_previous, pi_current = pi_current, ((2*n+1)*mu*pi_current-(n+1)*pi_previous)/n
    return electric, magnetic


def campaign(spacings, output, *, cycles=12, subdivisions=2):
    frequency, radius = 3.e9, .02
    k = 2*math.pi*frequency/mw.constants.C_0
    directions = np.asarray(((1., .3, .2), (-.3, 1., -.2), (.2, -.4, 1.), (.1, .2, -1.)))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    points = .032*directions
    expected_e, expected_h = pec_sphere_fields(radius, k, points)
    mesh = sphere_mesh(radius, subdivisions=subdivisions)
    geometry = mw.SurfaceMesh(mesh.vertices, mesh.faces)
    geometry_hash = hashlib.sha256(mesh.vertices.cpu().numpy().tobytes()+mesh.faces.cpu().numpy().tobytes()).hexdigest()
    report = dict(frequency_hz=frequency, radius_m=radius, geometry_hash=geometry_hash,
        triangles=len(mesh.faces), points_m=points.tolist(), cycles_per_window=cycles,
        reference="PEC Mie exterior series; no fitted material or geometry",
        acceptance=dict(aggregate_complex_E_H_relative_error_max=.05, maximum_point_complex_E_H_relative_error_max=.05),
        source_hashes={name:hashlib.sha256(Path(mw.__file__).parent.joinpath(name).read_bytes()).hexdigest()
            for name in ("compiler/cut_surface.py","fdtd/cut_surface.py","geometry/surface_mesh.py","fdtd/runtime/stepping.py")}, rows=[])
    query = torch.as_tensor(points, dtype=torch.float64, device="cuda")
    for spacing in spacings:
        start = time.perf_counter()
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.12, .12),)*3), grid=mw.GridSpec.uniform(spacing),
            boundary=mw.BoundarySpec.pml(num_layers=math.ceil(.06/spacing)), device="cuda",
            structures=(mw.Structure(geometry=geometry, material=mw.Material.pec()),),
            sources=(mw.PlaneWave(direction=(0., 0., 1.), polarization=(1., 0., 0.),
                source_time=mw.CW(frequency), injection=mw.TFSF(bounds=((-0.05,.05),)*3)),))
        run = dict(frequency=frequency, run_time=mw.TimeConfig(transient_cycles=cycles,
            steady_cycles=cycles, dt=.8*spacing/(mw.constants.C_0*math.sqrt(3))),
            spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=True, cuda_graph=True)
        add_query_stencils(scene, points)
        background = mw.Simulation.fdtd(scene.clone(structures=()), **run).run()
        total = mw.Simulation.fdtd(scene, **run).run()
        numerical_e, numerical_h = total.query_fields(query, region_id="exterior", field_kind="scattered", reference=background)
        incident_e, incident_h = background.query_fields(query, region_id="exterior", field_kind="total")
        analytic_incident = np.exp(1j*k*points[:, 2])
        amplitude = np.vdot(analytic_incident, incident_e[:, 0].cpu().numpy())/np.vdot(analytic_incident, analytic_incident)
        actual_e, actual_h = numerical_e.cpu().numpy()/amplitude, numerical_h.cpu().numpy()/amplitude
        runtime = total.solver._cut_surface
        row = dict(spacing_m=spacing, electric_relative_error=float(np.linalg.norm(actual_e-expected_e)/np.linalg.norm(expected_e)),
            magnetic_relative_error=float(np.linalg.norm(actual_h-expected_h)/np.linalg.norm(expected_h)),
            maximum_electric_point_relative_error=float(np.max(np.linalg.norm(actual_e-expected_e,axis=-1)/np.linalg.norm(expected_e,axis=-1))),
            maximum_magnetic_point_relative_error=float(np.max(np.linalg.norm(actual_h-expected_h,axis=-1)/np.linalg.norm(expected_h,axis=-1))),
            incident_fit_relative_error=float(np.linalg.norm(incident_e[:, 0].cpu().numpy()-amplitude*analytic_incident)/np.linalg.norm(analytic_incident)),
            incident_amplitude=[float(amplitude.real), float(amplitude.imag)],
            constraint_rows=runtime["constraint_rows"], rank=runtime["projection"].rank,
            support_count=runtime["support_count"], elapsed_seconds=time.perf_counter()-start,
            electric_real=actual_e.real.tolist(), electric_imag=actual_e.imag.tolist(),
            magnetic_real=actual_h.real.tolist(), magnetic_imag=actual_h.imag.tolist())
        report["rows"].append(row)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacings", nargs="+", type=float, default=(.01, .0075, .005))
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign(args.spacings, args.output, cycles=args.cycles, subdivisions=args.subdivisions)
