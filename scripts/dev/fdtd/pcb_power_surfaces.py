"""Compare independent enclosing vacuum surfaces without changing the PCB model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_application_campaign import power_observations, simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=12e-9)
    parser.add_argument('--sample-spacing', type=float, default=.0012)
    parser.add_argument('--native-planes', action='store_true')
    parser.add_argument('--surface-factors', type=float, nargs='+', default=(.5, 1., 1.5))
    args = parser.parse_args()
    frequency = 2.45e9
    source = package_identity(mw, 'witwin-maxwell')
    driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_bytes = args.input.read_bytes()
    definition = json.loads(input_bytes)
    scene = import_pcb_scene(definition, frequency=frequency)
    default = power_observations(definition, scene, frequency)
    active_names = {s.name for s in scene.structures}
    geometry = np.asarray([r['bounds_m'] for r in definition['dielectrics']+definition['conductors']
                           if r['id'] in active_names]).reshape(-1, 3, 2)
    active = np.stack((geometry[..., 0].min(0), geometry[..., 1].max(0)), -1)
    surfaces = []
    for factor in args.surface_factors:
        if not 0 < factor < 2:
            raise ValueError('Surface factor must remain strictly inside the physical PML interface.')
        bounds = active+(np.asarray(default[0].bounds)-active)*factor
        surfaces.append(mw.ClosedSurfaceMonitor.box(f'radiation_{factor:g}',
            position=bounds.mean(1), size=np.diff(bounds).ravel(), frequencies=(frequency,),
            sample_spacing=None if args.native_planes else args.sample_spacing))
    scene = scene.clone(monitors=surfaces+[default[1]])
    config = simulation(scene, frequency, with_power=True)
    prepared = config.prepare()
    config.config.run_time = mw.TimeConfig(time_steps=math.ceil(args.duration/prepared.solver.dt))
    start = time.perf_counter()
    result = prepared.run()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(args.output/'result.pt')
    bulk = result.power_loss('bulk_loss').conduction
    copper = sum((row['surface_power'].to(bulk.device) for row in result.conductor_losses.values()), torch.zeros_like(bulk))
    records = []
    for monitor in surfaces:
        antenna = result.antenna(surface=monitor.name, driven_port=scene.ports[0].name,
                                 theta_points=91, phi_points=73, radius=10.)
        flux = sum((p*w).sum() for p, w in
                   (plane_normal_poynting(face) for face in result.monitor(monitor.name)['faces'].values()))
        record = {'surface': monitor.name, 'bounds_m': monitor.bounds, 'accepted_w': float(antenna.p_accepted[0]),
                  'radiated_w': float(antenna.p_rad[0]), 'surface_flux_w': float(flux),
                  'bulk_loss_w': float(bulk[0]), 'copper_loss_w': float(copper[0]),
                  'power_closure': float(abs(antenna.p_rad+bulk+copper-antenna.p_accepted)[0]/antenna.p_accepted[0]),
                  'flux_closure': float(abs(flux+bulk+copper-antenna.p_accepted)[0]/antenna.p_accepted[0])}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'schema': 'pcb-enclosing-surfaces/1', 'source': source,
              'source_unchanged': source['source_sha256'] == package_identity(mw, 'witwin-maxwell')['source_sha256'],
              'input': str(args.input.resolve()), 'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
              'input_unchanged': hashlib.sha256(args.input.read_bytes()).hexdigest() == hashlib.sha256(input_bytes).hexdigest(),
              'sample_spacing_m': None if args.native_planes else args.sample_spacing,
              'driver_sha256': driver_sha256,
              'driver_unchanged': driver_sha256 == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'physical_pml_interfaces_m': {axis: [float(getattr(prepared.solver.scene, axis+'_nodes64')[prepared.solver.scene.pml_thickness_for_face(axis, 'low')]),
                    float(getattr(prepared.solver.scene, axis+'_nodes64')[-prepared.solver.scene.pml_thickness_for_face(axis, 'high')-1])]
                    for axis in 'xyz'},
              'steps': config.config.run_time.time_steps, 'dt_s': prepared.solver.dt,
              'actual_duration_s': config.config.run_time.time_steps*prepared.solver.dt,
              'final_live_field_maximum': {name: float(getattr(prepared.solver, name).abs().max())
                                          for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')},
              'run_seconds': seconds, 'records': records}
    port = result.port(scene.ports[0].name)
    report['port'] = {key: {'real': value.real.detach().cpu().tolist(),
                             'imag': value.imag.detach().cpu().tolist()}
                      for key, value in (('voltage', port.voltage), ('current', port.current),
                                         ('reflection', port.reflection_coefficient))}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not (report['source_unchanged'] and report['input_unchanged'] and report['driver_unchanged']):
        raise RuntimeError('The retained run identity changed during execution.')


if __name__ == '__main__':
    main()
