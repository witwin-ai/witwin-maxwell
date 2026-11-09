"""Read-only saved-case product-first diagnostic; see CPU Green/covariance proofs."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase,build_case,CoherentObservations
# This diagnostic module is new MAIN code; all field/source helpers are the executed archive.
spec=__import__('importlib.util',fromlist=['spec_from_file_location'])
module=spec.module_from_spec(s:=spec.spec_from_file_location('product_flux',str(Path(__file__).with_name('printed_line_product_flux.py'))));s.loader.exec_module(module)
root=Path(sys.argv[1]);output=Path(sys.argv[2]);input_audit=Path(sys.argv[3])
validated=json.load(open(input_audit));before={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in validated['input_files_sha256']}
if before!=validated['input_files_sha256']:raise ValueError('Original independent-audit raw signatures no longer match')
report=json.load(open(root/'physical-loss-report.json'))
def decode(v):
 if isinstance(v,dict):return complex(v['real'],v['imag']) if set(v)=={'real','imag'} else {k:decode(x) for k,x in v.items()}
 return [decode(x) for x in v] if isinstance(v,list) else v
values=report['case'].copy()
for key in ('frequencies','positions'):values[key]=tuple(values[key])
case=PilotCase(**values);scene=build_case(case)[0]
folders=[root/f'drive{i}' for i in range(2)];results=[mw.Result.load(p/'result.pt',scene=scene,map_location='cpu') for p in folders];bases=[torch.load(p/'basis.pt',weights_only=False,map_location='cpu') for p in folders]
hasCu=bool(bases[0].realizations)
if hasCu:
 assert all(len(base.realizations)==6 for base in bases)
 for structure in scene.structures:
  if not bool(getattr(structure.material,'is_copper_foil',False)):
   assert float(structure.material.mu_r)==1.,'Only the current collapsed-planar single normal-H DOF with uniform nonmagnetic background is supported'
coefficient=np.array([decode(r['coherent_coefficients']) for r in report['frequency_records']],dtype=np.complex128)
for result,base in zip(results,bases):
 assert not result.stats().get('normalize_source',False)
 assert base.execution_contract.window=='none' and not base.execution_contract.normalize_source
 for realization in base.realizations:
  assert realization.banks==('real',)
  assert bool((realization.spectral_norm==base.execution_contract.planned_steps).all())
 for name,payload in result._monitors.items():
  if name.startswith('line_roi_flux::'):
   assert np.all(np.asarray(payload['samples'])==base.execution_contract.planned_steps)
combined=CoherentObservations(results,bases,coefficient);records=[]
from witwin.maxwell.fdtd.coords import component_coords
axes={name:component_coords(results[0].prepared_scene,name) for name in ('Ex','Ey','Ez')}
for fi,f in enumerate(case.frequencies):
 selected=combined.monitor('line_roi_flux',frequency=f);faces={}
 for label,face in selected['faces'].items():
  raw=copy.copy(results[0]._monitors['line_roi_flux::'+label]);raw['components']={}
  for name,value in results[0]._monitors['line_roi_flux::'+label]['components'].items():
   entry=value.copy();entry['data']=sum(coefficient[fi,j]*results[j]._monitors['line_roi_flux::'+label]['components'][name]['data'][fi].to(torch.complex128) for j in range(2));raw['components'][name]=entry
  traces=[dict(trace,electric_fields=trace['electric_fields'][fi]) for record in combined.conductor_losses.values() for trace in record['bulk_electric_traces']]
  faces[label]=module.native_product_flux(raw,face,side_traces=traces,native_axes=axes,require_sided=hasCu,single_normal_H_DOF_contract=hasCu)
 old=sum(v['old'] for v in faces.values());new=sum(v['product_first'] for v in faces.values())
 original=report['coherent_regional_power']['net_outward_flux_w'][fi];heat=sum(report['coherent_regional_power'][k][fi] for k in ('bulk_conduction_w','copper_film_w'))
 records.append(dict(frequency_hz=f,old_raw_reconstructed=old,old_recorded=original,old_raw_minus_recorded_w=old-original,product_first=new,total_change_w=new-old,native_covariance_change_w=sum(v['covariance_correction'] for v in faces.values()),side_trace_change_w=sum(v.get('side_trace_correction',0.) for v in faces.values()),heat_unchanged=heat,old_heat_residual_relative_recorded=report['coherent_regional_power']['balance_relative_to_heat'][fi],old_heat_residual_relative_raw=abs(old+heat)/heat,new_heat_residual_relative=abs(new+heat)/heat,faces=faces))
 if abs(old-original)>64*np.finfo(np.float32).eps*sum(abs(v['old']) for v in faces.values()):raise ValueError(('Old raw reconstruction mismatch',old,original))
if output.exists():raise FileExistsError(output)
after={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
assert before==after
from scripts.dev.fdtd.printed_line_loss_execution import execution_identity
output.write_text(json.dumps(dict(scope='Diagnostic only; raw normal interpolation/time/crop/width/sign and heat unchanged. Original powers/gates retained. No field replay.',records=records,
 original_raw_files_sha256=before,all_original_raw_files_unchanged=True,original_independent_audit_sha256=hashlib.sha256(input_audit.read_bytes()).hexdigest(),
 executed_field_source=execution_identity(),diagnostic_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 Cu_scope='Bounded CURRENT collapsed planarCu discretecontract: actualleft/rightEt ownadjacentcells and one stored native normalH DOF, uniformnonmagnetic background. Not a general physical finitefilm Bn equality or curved/thickresolved contract. Normal stencil touchingCu rejected.',clock_and_normalization_contract='Realbank; nonewindow; actual Cu spectralnorm and native observer samples equal declaredN. Both endpointE accumulators record after electric/Cu update with phase2pi*f*dt*step and peak2/N. No fitted phase/scale; lefttrace mismatch retained perface.',original_gates_unchanged=report['pilot_functional_gates']),indent=2)+'\n')
print([(r['old_heat_residual_relative_recorded'],r['new_heat_residual_relative']) for r in records])
