"""One-solve-per-mesh ellipsoid self-convergence; no independent oracle claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch
import witwin.maxwell as mw
from benchmark.mom_fdtd import (FREQUENCY, INTEGRATION, POINTS, PROCESS_PROVENANCE,
                                control_sphere, make_scene, physical_scene_without_stencils,
                                _complex_data, _weighted_norm)
from witwin.maxwell.mom.fields import poynting_flux


def measure(subdivision, integration=INTEGRATION):
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    scene = physical_scene_without_stencils(make_scene('ellipsoid', .01))
    config = mw.MoMConfig(subdivisions=subdivision, integration=integration,
                          query_config=mw.FieldQueryConfig(max_order=128))
    result = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config).run()
    control, normals, areas, angles = control_sphere(order=6, radius=.2)
    exterior = torch.cat((torch.tensor(POINTS[:-1],device='cuda',dtype=torch.float64),control))
    interior = torch.tensor(POINTS[-1:],device='cuda',dtype=torch.float64)
    values = {}
    for kind in ('total','scattered'):
        for name, field in zip(('E','H'), result.query_fields(exterior,region_id='exterior',field_kind=kind)):
            values['exterior_'+kind+'_'+name] = field
    for name,field in zip(('E','H'),result.query_fields(interior,region_id='interior',field_kind='total')):
        values['interior_total_'+name] = field
    far,_ = result.far_field(normals)
    values['far_E'] = far
    e,h = values['exterior_scattered_E'][3:],values['exterior_scattered_H'][3:]
    et,ht = values['exterior_total_E'][3:],values['exterior_total_H'][3:]
    scattered = poynting_flux(e,h,normals,areas)
    extinction = -poynting_flux(et-e,h,normals,areas)-poynting_flux(e,ht-h,normals,areas)
    powers = {'scattering_W':float(scattered),'extinction_W':float(extinction),
              'absorption_W':float(-poynting_flux(et,ht,normals,areas))}
    torch.cuda.synchronize()
    row = {'subdivision':subdivision,'status':'measured','integration':asdict(integration),
           'values':{k:_complex_data(v) for k,v in values.items()},
           'powers':powers,'solver':result.stats(),'wall_seconds':time.perf_counter()-start,
           'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),
           'exterior_points_m':exterior.cpu().tolist(),'interior_points_m':interior.cpu().tolist(),
           'far_directions':normals.cpu().tolist(),'solid_angle_weights':angles.cpu().tolist()}
    del result
    torch.cuda.empty_cache()
    return row,values,angles


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subdivisions',nargs='+',type=int,default=[2,3,4])
    parser.add_argument('--pair-batch',type=int,default=16,
                        help='Bound near-integral workspace without changing integration orders or tolerances')
    args=parser.parse_args(argv)
    integration=replace(INTEGRATION,pair_batch=args.pair_batch)
    report={'frequency_hz':FREQUENCY,'family':'ellipsoid','semiaxes_m':[.1,.08,.065],
            'integration':asdict(integration),'cases':[],'provenance_segments':[PROCESS_PROVENANCE],
            'source_sha256':{str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1])):
                             hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'independent_reference_certified':False,'H2_certified':False,'G0_certified':False,
            'scope':'Three-mesh self-convergence on identical physical samples; no external reference or absolute error certificate.'}
    previous=None
    for subdivision in args.subdivisions:
        row,values,weights=measure(subdivision,integration)
        if previous is not None:
            row['relative_change_from_previous']={key:(_weighted_norm(value-previous[key],weights)/_weighted_norm(value,weights)
                if key=='far_E' else float(torch.linalg.vector_norm(value-previous[key])/torch.linalg.vector_norm(value)))
                for key,value in values.items()}
        previous=values
        report['cases'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:row[k] for k in ('subdivision','wall_seconds','relative_change_from_previous') if k in row}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
