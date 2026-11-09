"""Strict physical charge-continuity diagnostic, separate from query functionality.

The original auto-Bloch / mixed full-field-and-observer clock control failed
monotone convergence and remains unqualified. This driver retains that exact
failure gate for investigation; a successful functional query suite does not
clear this independent physical qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.pcb_acceptance import package_identity
from tests.postprocess.fields.test_copper_query_execution import retain_h_stencils

def run_oblique_charge_control(record_property, *, fixed_wavevector=True, compare_point_clock=True,
                               spacings=(.005,.0025,.00125), require_order=True,
                               transient_cycles=24, steady_cycles=20):
    import math
    errors=[];point_errors=[]
    frequency=1e9;angle=math.radians(35.)
    omega=2*math.pi*frequency;ky=omega/299792458.*math.sin(angle)
    if require_order and len(spacings)!=3:
        raise ValueError("Order acceptance requires three unchanged refinement levels.")
    for spacing in spacings:
        scene=mw.Scene(domain=mw.Domain(bounds=((- .12,.12),(-.005,.005),(-.005,.005))),
            grid=mw.GridSpec.uniform(spacing),device='cuda',
            boundary=mw.BoundarySpec.faces(default='none',x='pml',y='bloch',z='bloch',
                num_layers=12,bloch_wavevector=(0.,ky,0.) if fixed_wavevector else 'auto'),
            structures=(mw.Structure(mw.Box(position=torch.zeros(3,dtype=torch.float64),
                size=torch.tensor((0.,.04,.04),dtype=torch.float64)),mw.CopperFoil(
                    thickness=2e-6,conductivity=1e6,frequency_range=(.5e9,2e9)),name='foil'),),
            sources=(mw.PlaneWave(direction=(math.cos(angle),math.sin(angle),0.),
                polarization=(-math.sin(angle),math.cos(angle),0.),
                injection_axis='x',injection=mw.TFSF.slab(axis='x',bounds=(-.08,.08)),
                source_time=mw.CW(frequency,phase=.27)),))
        points=torch.tensor(((0.,0.,0.),),dtype=torch.float64,device='cuda')
        scene=retain_h_stencils(scene,points,extra_components=('Ex',) if compare_point_clock else ())
        result=mw.Simulation.fdtd(scene,frequency=frequency,full_field_dft=True,cuda_graph=True,
            run_time=mw.TimeConfig.auto(transient_cycles=transient_cycles,steady_cycles=steady_cycles),
            spectral_sampler=mw.SpectralSampler(window='hanning')).run()
        negative=result.query_fields(points,region_id='exterior',field_kind='total',trace_side='negative')
        positive=result.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive')
        charge=8.8541878128e-12*(positive[0][0,0]-negative[0][0,0])
        current=-(positive[1][0,2]-negative[1][0,2])
        errors.append(float(abs((ky*current-omega*charge)/(omega*charge))))
        record_property('spacing_'+str(spacing),repr(dict(charge=complex(charge),current=complex(current),error=errors[-1],dt=result.stats()['dt'],resolved_bloch_wavevector=result.solver.resolved_bloch_wavevector)))
        if compare_point_clock:
            point_result=mw.Result(method='fdtd',scene=scene,prepared_scene=result.prepared_scene,frequency=frequency,
                fields={name:value for name,value in result._fields.items() if name!='EX'},
                monitors=result._monitors,metadata=result._metadata,solver_stats=result.stats())
            negative_point=point_result.query_fields(points,region_id='exterior',field_kind='total',trace_side='negative')
            positive_point=point_result.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive')
            point_charge=8.8541878128e-12*(positive_point[0][0,0]-negative_point[0][0,0])
            point_errors.append(float(abs((ky*current-omega*point_charge)/(omega*point_charge))))
            record_property('point_clock_'+str(spacing),repr(dict(charge=complex(point_charge),error=point_errors[-1],
                full_vs_point_charge_relative=complex((charge-point_charge)/point_charge))))
    record_property('relative_continuity_errors',repr(errors))
    if compare_point_clock:
        record_property('point_clock_relative_continuity_errors',repr(point_errors))
    if not require_order:
        return
    if compare_point_clock:
        assert point_errors[0]>point_errors[1]>point_errors[2],point_errors
        assert point_errors[0]/point_errors[2]>8,point_errors
    assert errors[0]>errors[1]>errors[2],errors
    assert errors[0]/errors[2]>8,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-auto',action='store_true',help='Reproduce the original failed auto-Bloch/mixed-clock control unchanged.')
    parser.add_argument('--spacing',type=float,help='Run one mesh as a clock diagnostic; never claims convergence acceptance.')
    parser.add_argument('--transient-cycles',type=int,default=24)
    parser.add_argument('--steady-cycles',type=int,default=20)
    args=parser.parse_args()
    report={'schema':'copper-query-charge-control/1','status':'running',
        'functional_query_pass_is_not_charge_qualification':True,
        'bloch_policy':'legacy_auto' if args.legacy_auto else 'fixed_physical_ky',
        'source':package_identity(mw,'witwin-maxwell'), 'properties':{},
        'known_control_limits':['legacy-auto mode changes transverse k across meshes',
            'normal E uses full-field DFT while H uses native point observer clock',
            'finite-window/source-settling controls require separate verification']}
    def record(name,value):
        report['properties'][name]=value
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        run_oblique_charge_control(record,fixed_wavevector=not args.legacy_auto,compare_point_clock=not args.legacy_auto,
            spacings=(args.spacing,) if args.spacing else (.005,.0025,.00125),require_order=args.spacing is None,
            transient_cycles=args.transient_cycles,steady_cycles=args.steady_cycles)
    except AssertionError as error:
        report.update(status='failed',failure=str(error))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        raise SystemExit(1) from error
    else:
        report['status']='single_grid_diagnostic_not_order_qualified' if args.spacing else 'passed_original_control_gate'
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
