"""Detached printed-line columns through ordinary public Result Cu queries.

The field solver, source stamps, observers and resume engine remain the primary
Scene/Simulation path. Fixed physical quadrature is declared before execution;
private retained-field builders are never runtime dependencies.
"""
from dataclasses import asdict,replace
from pathlib import Path
import gc,hashlib,json,math,sys,time
import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_broadband import encode,physical_sources,load_helper
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase,make_simulation,runtime_operator_signature,_material_signature,tensor_signature
from scripts.dev.fdtd.printed_line_public_vi import build_public_vi,evaluate_public_vi,validate_plan_support
from scripts.dev.fdtd.printed_line_public_vi_checks import independent_public_audit
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors
from scripts.dev.fdtd.pcb_acceptance import package_identity
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell._fixed_inputs import require_fixed_input
from witwin.maxwell.fdtd.observers import get_observer_results
from scripts.dev.fdtd.printed_line_public_vi_checks import verify_native_oracle_samples
from scripts.dev.fdtd.printed_line_loss_bridge import capture_executed_realization,save_capture,sha256_file


def source_identity():
    import witwin.core
    return dict(maxwell=package_identity(mw,'witwin-maxwell')['source_sha256'],core=package_identity(witwin.core,'witwin')['source_sha256'])


def column_case(length_index):
    if type(length_index)is not int or length_index not in(0,1):
        raise ValueError('The functional experiment has exactly two fixed source separations.')
    length=(.40,.48)[length_index]
    return PilotCase(dx=.00125,longitudinal_dl=.0025,axial_half_length=.30,
        separation=length,positions=(-length/2+.025,0.,length/2-.025),
        probe_reference_dx=.0025,record_legacy_native_vi=True)


def source_basis(channels):
    if channels not in(2,4):
        raise ValueError('A scalar line has two physical source groups; a coupled pair has four.')
    return .75*np.eye(channels)+.25*np.ones((channels,channels))


def actual_clock(dt,case):
    if not math.isfinite(dt)or dt<=0 or not 0<case.checkpoint_time<case.duration:
        raise ValueError('Actual timestep and physical checkpoint/horizon must be positive and ordered.')
    N=math.ceil(case.duration/dt);split=math.ceil(case.checkpoint_time/dt)
    if not 0<split<N:raise ValueError('Actual physical checkpoint must be interior to the complete horizon.')
    return dict(actual_dt_s=dt,total_steps=N,checkpoint_step=split,
        requested_duration_s=case.duration,requested_checkpoint_time_s=case.checkpoint_time,
        actual_duration_s=N*dt,actual_checkpoint_time_s=split*dt)


def _scene(kind,case):
    scene=printed_line_scene(kind,dx=case.dx,longitudinal_dl=case.longitudinal_dl or case.dx,
        axial_half_length=case.axial_half_length,port_separation=case.separation,
        transverse_padding=case.padding,transverse_refinement_radius=.02,
        sigma_e=case.sigma,metal=case.metal,copper_frequency_range=(.1e9,4e9),device='cpu')
    ports,groups=physical_sources(scene,kind,case.longitudinal_dl or case.dx)
    return scene.clone(ports=ports),groups


def build_public_column(kind,case):
    """CPU geometric declaration of original SI rows and typed Cu operations."""
    require_fixed_input(vars(case),name='Detached printed-line benchmark case')
    if kind not in('cpw','differential_pair')or case.metal!='copper':
        raise ValueError('This selected-Cu functional workflow admits CPW and coupled copper pairs.')
    if case.probe_reference_dx!=.0025:
        raise ValueError('Retain the fixed2.5mm physical reference functional.')
    scene,groups=_scene(kind,case);reference,_=_scene(kind,replace(case,dx=case.probe_reference_dx))
    helper,_=load_helper(kind)
    if kind=='cpw':
        monitors,rows,_=helper.native_probe_monitors(reference,kind,.0025,case.longitudinal_dl or case.dx,case.padding,case.positions,case.metal)
    else:
        monitors,nested,_=helper.probes(reference,.0025,case.longitudinal_dl or case.dx,case.padding,case.positions,case.metal)
        rows=[pair for station in nested for pair in station]
    plan=build_public_vi(scene,reference,monitors,rows,case.frequencies)
    validate_plan_support(plan,scene)
    prepared=prepare_scene(scene);oracles=[];oracle_indices={};by_key={};query_oracle={}
    for name,term in plan.queries.items():
        entries=[]
        for index,weight in plan.native_rows[name]:
            key=(term.component,index)
            if key not in by_key:
                label='public_Cu_reference_'+str(len(by_key));by_key[key]=label
                point=tuple(float(a[i])for a,i in zip(component_coords(prepared,term.component),index))
                oracles.append(mw.PointMonitor(label,point,fields=(term.component,)));oracle_indices[label]=(term.component,index)
            entries.append((by_key[key],term.component,weight))
        query_oracle[name]=tuple(entries)
    owned={(q.component,q.points_m[0])for q in plan.queries.values()};original={m.name:m for m in monitors};tails=[]
    for station,(voltage,_)in enumerate(rows):
        ordinary=[(n,c,w)for n,c,w in voltage if(c,tuple(original[n].position))not in owned]
        if not ordinary:raise ValueError('Tail controls require an original ordinary electric Point.')
        name,component,_=ordinary[len(ordinary)//2]
        tails.append(mw.FieldTimeMonitor('full64_tail_'+str(station),components=(component,),position=original[name].position,interval=32))
    bounds=((-0.015,0.015),*scene.domain.bounds[1:]);regional,schema=regional_monitors(scene,bounds,case.frequencies,observation='planes')
    baseline=scene.clone(monitors=[*plan.monitors,*tails,*regional]);observed=scene.clone(monitors=[*baseline.monitors,*oracles])
    return dict(case=case,base=baseline,observed=observed,groups=groups,plan=plan,
        oracle_indices=oracle_indices,query_oracle=query_oracle,regional_schema=schema,regional_bounds=bounds)


def operator_signature(solver):
    return dict(Norton=runtime_operator_signature(solver),material=_material_signature(solver),
        sources=tensor_signature(dict(electric=solver._electric_source_terms,magnetic=solver._magnetic_source_terms,compiled=solver._compiled_sources)),
        copper=tensor_signature([dict(face=e['face'].name,component=e['electric_name'],indices=e['indices'],area=e['area'],B=e['B'],rates=e['rates'],D=e['D'])for e in solver._copper_foils]))


def _run_public_column(kind,case,drive_column,output,*,graph=True,resume=True,length_index=None):
    """ONE full requested physical horizon; no hidden queue or partial Result."""
    if type(graph)is not bool or type(resume)is not bool:
        raise TypeError('Execution modes must be explicit booleans.')
    context=build_public_column(kind,case);basis=source_basis(len(context['groups']))
    if type(drive_column)is not int or not 0<=drive_column<len(basis):
        raise ValueError('Drive column must name one original nonzero independent physical basis column.')
    output=Path(output)
    if output.exists():raise FileExistsError('Every earlier or partial output is preserved.')
    output.mkdir(parents=True);started=time.monotonic();events={};torch.cuda.reset_peak_memory_stats();identity_before=source_identity()
    simulation=make_simulation(context['observed'],context['groups'],basis[:,drive_column],case,graph=graph,steps=1)
    prepared=simulation.prepare();solver=prepared.solver;clock=actual_clock(float(solver.dt),case)
    simulation.config.run_time=mw.TimeConfig(time_steps=clock['total_steps']);operator=operator_signature(solver);events['prepared_s']=time.monotonic()-started
    if resume:
        checkpoint=prepared.run_until(clock['checkpoint_step']);torch.cuda.synchronize()
        if checkpoint.step!=clock['checkpoint_step']or checkpoint.total_steps!=clock['total_steps']:raise ValueError('Real checkpoint lost the complete physical horizon.')
        checkpoint.save(output/'checkpoint.pt');del checkpoint,prepared,solver;gc.collect();torch.cuda.empty_cache()
        checkpoint=mw.FDTDResumeCheckpoint.load(output/'checkpoint.pt',map_location='cpu');prepared=simulation.prepare();solver=prepared.solver
        if actual_clock(float(solver.dt),case)!=clock or operator_signature(solver)!=operator:raise ValueError('Fresh preparation changed physical source/material/time operator.')
        result=prepared.run(resume_from=checkpoint)
    else:result=prepared.run()
    # Resume may rebuild a fresh existing-path solver (automatic setup is off for resume); read the one that ran.
    solver=result.solver
    torch.cuda.synchronize();events['complete_Result_s']=time.monotonic()-started
    if operator_signature(solver)!=operator or solver.dft_enabled or bool(solver._cuda_graph_active)!=graph or bool(solver._observer_cuda_graph_active)!=graph:raise ValueError('Actual operator, dense spectrum or Graph mode changed.')
    if any(float(e['spectral_step'])!=clock['total_steps']for e in solver._copper_foils):raise ValueError('Actual Cu producer did not complete its horizon.')
    native_proof=verify_native_oracle_samples(context,get_observer_results(solver),solver.scene,expected_samples=clock['total_steps'])
    # Direct capture of the executed Cu-loss realization while the producer lives.
    capture=capture_executed_realization(result,solver,operator=operator,clock=clock,context=context,identity=identity_before,kind=kind,length_index=length_index,drive_column=drive_column)
    realization=save_capture(output,capture);del capture;events['realization_captured_s']=time.monotonic()-started
    VI,evidence=evaluate_public_vi(context['plan'],result=result)
    proof=independent_public_audit(context,result,VI,evidence,total=clock['total_steps'])
    reverse,_=evaluate_public_vi(context['plan'],result=result,frequencies=tuple(reversed(case.frequencies)))
    torch.testing.assert_close(reverse,VI.flip(-1),rtol=0,atol=0)
    result.save(output/'result.pt');restored=mw.Result.load(output/'result.pt',scene=simulation.scene,map_location='cuda:0')
    restored_VI,restored_heads=evaluate_public_vi(context['plan'],result=restored)
    torch.testing.assert_close(restored_VI,VI,rtol=0,atol=0)
    for name,head in evidence.items():
        for key in('primary','constitutive_negative','constitutive_positive'):torch.testing.assert_close(restored_heads[name][key],head[key],rtol=0,atol=0)
    restored_proof=independent_public_audit(context,restored,restored_VI,restored_heads,total=clock['total_steps'])
    torch.save(dict(VI=VI.detach().cpu(),queries={n:{k:v.detach().cpu()for k,v in h.items()if torch.is_tensor(v)}for n,h in evidence.items()}),output/'actual-public-VI.pt')
    events['finished_s']=time.monotonic()-started;identity_after=source_identity()
    if identity_after!=identity_before:raise ValueError('Executed numerical source identity changed during the column.')
    saved=('result.pt','actual-public-VI.pt','executed-Cu-realization.pt','executed-Cu-realization.json')+(('checkpoint.pt',)if resume else())
    report=dict(schema='public-printed-line-column/1',functional_passed=True,kind=kind,length_index=length_index,case=asdict(case),drive_column=drive_column,real_basis=basis[:,drive_column],clock=clock,resumed=resume,graph=graph,actual_operator=operator,physical_source_groups=context['groups'],native_oracle_indices=context['oracle_indices'],native_Point_plane_primitive_proof=native_proof,initial_independent_queries=proof,restored_independent_queries=restored_proof,reverse_and_disk_bitwise=True,actual_source=package_identity(mw,'witwin-maxwell'),peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(),events_s=events,other_columns_run=0,physical_mode_attenuation_accuracy_qualified=False,
        source_identity_before=identity_before,source_identity_after=identity_after,
        executed_Cu_realization={k:realization[k]for k in('schema','provenance','binding_sha256','capture_file_sha256','executed_copper_operator_signature','realization_sha256')},
        saved_files_sha256={name:sha256_file(output/name)for name in saved})
    (output/'execution.json').write_text(json.dumps(encode(report),indent=2,allow_nan=False)+'\n')
    del restored,result,solver,prepared,simulation
    gc.collect();torch.cuda.empty_cache()
    return report



def run_public_column(kind,case,drive_column,output,*,graph=True,resume=True,length_index=None):
    """Preserve any failed single-column output; no automatic retry."""
    path=Path(output)
    if path.exists():raise FileExistsError('Every earlier or partial output is preserved.')
    try:return _run_public_column(kind,case,drive_column,path,graph=graph,resume=resume,length_index=length_index)
    except BaseException as error:
        if path.exists()and not(path/'failed.json').exists():
            (path/'failed.json').write_text(json.dumps(dict(error=repr(error),retry=False,other_columns_run=0),indent=2)+'\n')
        raise


CONVERGENCE_OVERRIDES=('dx','longitudinal_dl','padding','sigma')


def convergence_case(length_index,**overrides):
    """Column case with explicit mesh/padding/bulk-loss overrides for convergence
    controls. Lengths, stations, functional, horizon and Cu stay fixed."""
    case=column_case(length_index)
    unknown=set(overrides)-set(CONVERGENCE_OVERRIDES)
    if unknown:raise ValueError(f'Unsupported convergence overrides: {sorted(unknown)}.')
    values={k:float(v)for k,v in overrides.items()if v is not None}
    if any(not math.isfinite(v)or v<0 or(v==0 and k!='sigma')for k,v in values.items()):
        raise ValueError('Convergence overrides must be finite; steps and padding positive, sigma nonnegative.')
    return replace(case,**values)


def command(output,kind,length_index,drive_index,*,execute,overrides=None):
    case=convergence_case(length_index,**(overrides or {}))
    if execute:return run_public_column(kind,case,drive_index,output,length_index=length_index)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    context=build_public_column(kind,case);basis=source_basis(len(context['groups']))
    if type(drive_index)is not int or not 0<=drive_index<len(basis):raise ValueError('Invalid original source basis column.')
    record=dict(status='CPU geometry preparation only; no field/query executed.',kind=kind,case=asdict(case),drive_index=drive_index,real_basis=basis[:,drive_index],groups=context['groups'],physical_rows=context['plan'].rows,expected_native_points=context['plan'].expected_points,Cu_terms={n:asdict(q)for n,q in context['plan'].queries.items()},CUDA_initialized=torch.cuda.is_initialized(),physical_accuracy_qualified=False)
    output.mkdir(parents=True);(output/'preparation.json').write_text(json.dumps(encode(record),indent=2)+'\n')
    return record
