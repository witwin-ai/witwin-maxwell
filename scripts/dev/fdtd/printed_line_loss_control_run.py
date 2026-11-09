"""Root-owned one-case material/mesh control: two real drives at fixed64ns."""
import argparse, gc, hashlib, json, math, shutil, sys
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, DRIVE_BASIS, build_case, make_simulation, run_drive, reduce_pair, execution_identity, tensor_signature, CoherentObservations
from scripts.dev.fdtd.printed_line_broadband import encode
from scripts.dev.fdtd.printed_line_regional_power import regional_plane_power

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--preflight',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
 p.add_argument('--source-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 gate=json.loads(args.gate.read_text())
 if gate.get('all_actual_controls_and_witnesses_passed') is not True or gate.get('tests')!=4 or gate.get('skipped')!=0:
  raise RuntimeError('Root-reviewed actual four-case CUDA/witness prerequisite has not passed.')
 forecast=json.loads(args.preflight.read_text());source=json.loads(args.source_identity.read_text())
 archive=Path(source['archive'])
 def verify():
  for name,value in source['archive_files'].items():
   if sha(archive/name)!=value:raise RuntimeError(f'Frozen source changed: {name}')
  identity=execution_identity()
  if identity!=forecast['source_before']:raise RuntimeError('Pilot actual numerical/core/helper source differs from CPU forecast.')
  return identity
 before=verify()
 if shutil.disk_usage(args.output.parent).free < forecast['observation_and_storage']['minimum_free_disk_bytes']:
  raise RuntimeError('Pilot disk reservation unavailable.')
 memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 if memory<forecast['observation_and_storage']['minimum_available_host_RAM_bytes']:raise RuntimeError('Pilot host RAM reservation unavailable.')
 free,total=torch.cuda.mem_get_info()
 if free<forecast['observation_and_storage']['minimum_free_gpu_bytes']:raise RuntimeError('Pilot free VRAM reservation unavailable; preserve queue and retry with headroom.')
 case_values=dict(forecast['case'])
 for name in ('frequencies','positions'):case_values[name]=tuple(case_values[name])
 case=PilotCase(**case_values);scene,groups,bounds,schema,*_=build_case(case)
 # Authoritative actual CUDA prepare; no field steps advance during this check.
 simulation=make_simulation(scene,groups,DRIVE_BASIS[0],case,graph=True,steps=1)
 actual=simulation.prepare();dt=float(actual.solver.dt)
 if not math.isclose(dt,forecast['CPU_compiled_time_step_forecast_s'],rel_tol=32*np.finfo(float).eps,abs_tol=0):
  raise RuntimeError('Actual CUDA dt differs from independent CPU compiler forecast beyond float64 preparation roundoff.')
 steps=math.ceil(case.duration/dt);split=math.floor(case.checkpoint_time/dt)
 if (steps,split)!=(forecast['planned_steps_forecast'],forecast['checkpoint_step_forecast']):raise RuntimeError('Actual CUDA time horizon differs from CPU forecast.')
 del actual,simulation;gc.collect();torch.cuda.empty_cache()
 args.output.mkdir()
 (args.output/'actual-prepare-contract.json').write_text(json.dumps(dict(dt_s=dt,planned_steps=steps,checkpoint_step=split,
  actual_duration_s=steps*dt,actual_checkpoint_time_s=split*dt,actual_CUDA_prepare=True,field_steps_advanced_by_precheck=0,
  preflight_sha256=sha(args.preflight),root_gate_sha256=sha(args.gate),source_identity_sha256=sha(args.source_identity),
  actual_available_resources_before_prepare=dict(free_GPU_bytes=free,total_GPU_bytes=total,available_host_RAM_bytes=memory),source=before),indent=2)+'\n')
 reports=[];folders=[]
 for index,amplitudes in enumerate(DRIVE_BASIS):
  folder=args.output/f'drive{index}';folders.append(folder)
  report=run_drive(case,amplitudes,folder,graph=True,resume=True)
  if report['planned_steps']!=steps or report['checkpoint']['step']!=split or report['checkpoint']['total_steps']!=steps:
   raise RuntimeError('Actual drive did not retain fixed64ns total /32ns disk checkpoint.')
  reports.append(report)
 # Pure single-end source extraction is a real constant inverse of the known
 # physical source basis. Complex phasors are never interpreted as an i-Gaussian.
 inverse=np.linalg.inv(np.asarray(DRIVE_BASIS).T)
 source_spectra={}
 for port in reports[0]['actual_source_spectra']:
  spectra=np.asarray([report['actual_source_spectra'][port]['thevenin_peak_voltage'].numpy() for report in reports])
  pure=inverse.T@spectra
  group=next(index for index,names in enumerate(groups) if port in names)
  active=pure[group];inactive=pure[1-group]
  if not np.isfinite(pure).all() or not np.all(abs(active)>0):raise RuntimeError('Actual source DFT coverage is not finite/resolved.')
  cancellation=float(np.max(abs(inactive)/abs(active)))
  if cancellation>1e-10:raise RuntimeError('Actual source DFTs do not obey declared real inverse-basis single-end extraction.')
  source_spectra[port]=dict(group=group,basis_thevenin_peak_voltage=spectra,pure_left_right_voltage=pure,
   inactive_to_active_max=cancellation,resistance_ohm=reports[0]['actual_source_spectra'][port]['resistance_ohm'])
 states=np.asarray([report['physical_state'] for report in reports])
 pure_states=np.einsum('dp,dsvf->psvf',inverse,states)
 results=[mw.Result.load(folder/'result.pt',scene=scene,map_location='cpu') for folder in folders]
 bases=[torch.load(folder/'basis.pt',weights_only=False,map_location='cpu') for folder in folders]
 pure_powers={}
 for column,name in enumerate(('pure_left','pure_right')):
  coefficients=np.broadcast_to(inverse[:,column],(len(case.frequencies),2))
  view=CoherentObservations(results,bases,coefficients)
  pure_powers[name]=regional_plane_power(view,scene,schema,bounds)
 del results,bases,view;gc.collect()
 source_record=dict(real_inverse_basis=inverse,actual_source_DFT_extraction=source_spectra,
  pure_left_right_physical_VI=pure_states,pure_left_right_shared_bulk_copper_and_flux=pure_powers,
  definition='Only actual independently accumulated source DFTs and declared real2x2 source inverse; no fitted response amplitude/phase and no constant-i Gaussian assumption.',
  bound_scope='Source extraction interoperability only; finite-window propagation/tail/release gates are assessed separately.')
 (args.output/'actual-real-source-basis-extraction.json').write_text(json.dumps(encode(source_record),indent=2,allow_nan=False)+'\n')
 report=reduce_pair(case,folders)
 (args.output/'physical-loss-report.json').write_text(json.dumps(encode(report),indent=2,allow_nan=False)+'\n')
 after=verify()
 (args.output/'completion-audit.json').write_text(json.dumps(dict(status='Two actual realbasis drives completed; numerical gates remain explicit in physical-loss-report.json.',
  source_before=before,source_after=after,source_unchanged=before==after,actual_drive_count=2,
  single_material_combination=True,single_line_length=True,full_attenuation_qualified=False,
  actual_source_pure_end_extraction_passed=True,root_gate_sha256=sha(args.gate),
  output_sha256={str(path.relative_to(args.output)):sha(path) for path in sorted(args.output.rglob('*')) if path.is_file()}),indent=2)+'\n')
 print(json.dumps(dict(actual_drives=2,physical_pilot_functional_passed=report['pilot_functional_passed'],physical_release_qualified=False)),flush=True)
if __name__=='__main__':main()
