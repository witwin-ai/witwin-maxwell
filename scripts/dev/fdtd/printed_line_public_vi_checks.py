"""Independent raw Point/two-sided-head and SI-row audit, never a field solver.

No public query implementation, interpolation/reconstruction helper, trace
normalization factor or synthetic sampling metadata is used. References consume
only actual stored banks, fixed declared rows and actual native Point payloads.
"""
import hashlib
import math
import numpy as np
import torch

U64=2.**-53

def gamma(n):
 if type(n)is not int or n<0 or n*U64>=1:raise ValueError('Invalid finite F64 operation count.')
 return n*U64/(1-n*U64)

def array(value,dtype,shape=None):
 if not isinstance(value,torch.Tensor)or value.dtype!=dtype:raise ValueError('Actual raw primitive precision must be explicit and unchanged.')
 a=value.detach().cpu().numpy()
 if shape is not None and a.shape!=shape:raise ValueError('Actual raw primitive shape differs.')
 if not np.isfinite(a).all():raise ValueError('Actual raw primitive is not finite.')
 return a

def weighted_reference(values,weights,*,public_terms,uncertainty=None):
 """Faithfully summed F64 products and a priori componentwise absolute bounds.

 Each real/imaginary public term requires at most one multiplication and one
 addition. Public interpolation visits eight corners, including exact zeros.
 A general n-term dot reference is bounded by gamma_(2n+2); math.fsum is
 better than that conservative sequential bound. Both paths consume the same
 fixed F64 coefficients. Any previously bounded input uncertainty propagates
 linearly with absolute physical weights; it is never an observed discrepancy.
 """
 a=np.asarray(values);w=np.asarray(weights,dtype=np.float64)
 if a.dtype!=np.complex128 or a.ndim!=2 or w.shape!=(a.shape[1],)or not np.isfinite(a).all()or not np.isfinite(w).all():raise ValueError('Reference requires finite [frequency,term]C128 and fixed F64 coefficients.')
 if public_terms<a.shape[1]:raise ValueError('Operation bound cannot undercount actual terms.')
 m=a.shape[1];ref=np.array([complex(math.fsum(float(w[j])*float(row[j].real)for j in range(m)),math.fsum(float(w[j])*float(row[j].imag)for j in range(m)))for row in a],dtype=np.complex128)
 absolute=np.stack([np.sum(np.abs(a.real)*np.abs(w)[None],axis=1),np.sum(np.abs(a.imag)*np.abs(w)[None],axis=1)],-1)
 # Input absolute-sum evaluation is also rounded: inflate by its gamma_2m
 # denominator, rather than use a data-independent or fitted absolute floor.
 scale=absolute/(1-gamma(2*m))
 bound=(gamma(2*public_terms)+gamma(2*m+2))*scale
 if uncertainty is not None:
  errors=np.asarray(uncertainty,dtype=np.float64)
  if errors.shape!=(a.shape[0],m,2)or not np.isfinite(errors).all()or np.any(errors<0):raise ValueError('Input uncertainty must be predeclared nonnegative component bounds.')
  propagated=np.sum(np.abs(w)[None,:,None]*errors,axis=1)/(1-gamma(2*m))
  bound+=propagated*(1+gamma(2*public_terms))
 return ref,bound

def check_value(value,expected,bound,*,label,nonzero=True):
 a=array(value,torch.complex128,expected.shape)
 if nonzero and not bool(np.any(a!=0)):raise ValueError(label+': zero public functional is not evidence of an active field.')
 delta=np.stack([np.abs(a.real-expected.real),np.abs(a.imag-expected.imag)],-1)
 if bool(np.any(delta>bound)):raise ValueError(label+': exceeds predeclared F64 product/sum bound.')
 return dict(label=label,nonzero=bool(np.any(a!=0)),max_absolute_component_error=float(delta.max()),max_a_priori_absolute_bound=float(bound.max()),operation_model='gamma_(2public_terms)+gamma_(2reference_terms+2), input L1 only; no fitted factor')

def point_spectrum(result,name,component,index,frequencies,total):
 p=result.monitor(name)
 if tuple(p.get('frequencies',()))!=tuple(frequencies)or tuple(p.get('field_indices',{}).get(component,()))!=tuple(index):raise ValueError('Actual Point frequency order or native support changed.')
 samples=torch.as_tensor(p.get('samples',())).reshape(-1)
 if len(samples)!=len(frequencies)or not bool((samples==total).all()):raise ValueError('Actual Point does not share the complete physical horizon.')
 return array(p['components'][component],torch.complex64).reshape(-1).astype(np.complex128)

def raw_trace(result,term,rows,frequencies,total):
 if term.owner not in result.conductor_losses:raise ValueError('Raw physical owner is missing.')
 record=result.conductor_losses[term.owner];rf=torch.as_tensor(record['frequencies']).reshape(-1).tolist()
 if len(set(rf))!=len(rf)or not all(f in rf for f in frequencies):raise ValueError('Raw trace frequency identity is missing or ambiguous.')
 selected=[rf.index(f)for f in frequencies]
 traces=[t for t in record.get('bulk_electric_traces',())if t.get('component')==term.component]
 if len(traces)!=1:raise ValueError('One actual raw owner/component bank is required.')
 trace=traces[0]
 if trace['normal_axis']!=term.normal_axis or trace['normal_coordinate']!=term.normal_coordinate:raise ValueError('Raw physical face normal changed.')
 indices=array(trace['indices'],torch.int64)
 if indices.ndim!=2 or indices.shape[1]!=3:raise ValueError('Raw support must be actual integer global indices.')
 keys=[tuple(map(int,row))for row in indices];lookup={key:j for j,key in enumerate(keys)}
 if len(lookup)!=len(keys):raise ValueError('Duplicate raw face rows cannot establish unique native ownership.')
 if any(k['xyz'.index(term.normal_axis)]!=term.normal_index for k in keys):raise ValueError('Raw rows are not on the declared physical plane.')
 try:offsets=[lookup[tuple(index)]for index,_ in rows]
 except KeyError as error:raise ValueError('A required fixed interpolation corner is absent from the actual face.')from error
 bank=array(trace['electric_fields'],torch.complex128,(len(rf),2,len(keys)))
 sampling=trace.get('sampling',{})
 if sampling.get('owner')!=term.owner or sampling.get('component')!=term.component or sampling.get('planned_steps')!=total or float(sampling.get('completed_steps',-1))!=total:raise ValueError('Actual raw face producer clock/owner differs.')
 if tuple(torch.as_tensor(sampling.get('frequencies',())).reshape(-1).tolist())!=tuple(rf)or not bool((torch.as_tensor(sampling.get('weight_sums',()))==total).all()):raise ValueError('Actual raw face weights or frequency identity differs.')
 return bank[np.asarray(selected)][:,:,offsets],offsets

def independent_public_audit(context,result,VI,evidence,*,total):
 plan=context['plan'];frequencies=tuple(plan.frequencies);nf=len(frequencies)
 def raw_fingerprints():
  mapping={}
  for owner in sorted({term.owner for term in plan.queries.values()}):
   record=result.conductor_losses.get(owner,{})
   for j,trace in enumerate(record.get('bulk_electric_traces',())):
    for key in('indices','electric_fields'):
     value=trace.get(key)
     if isinstance(value,torch.Tensor):
      a=value.detach().cpu().contiguous().numpy();mapping[(owner,j,key)]=(str(a.dtype),a.shape,hashlib.sha256(a.tobytes()).hexdigest())
  return mapping
 raw_before=raw_fingerprints()
 if len(plan.queries)!=6 or set(evidence)!=set(plan.queries):raise ValueError('Exactly six selected physical Cu functionals must be checked.')
 references={};uncertainties={};checks=[];support=[]
 for name,term in plan.queries.items():
  rows=plan.native_rows[name]
  # One corner when the query lands on a node (coarse grids), otherwise two.
  if len(rows) not in(1,2) or len({tuple(index)for index,_ in rows})!=len(rows):raise ValueError('This bounded control requires its actual distinct native corners.')
  entries=context['query_oracle'][name];corners=[];seen=[]
  for point,component,weight in entries:
   actual_component,index=context['oracle_indices'][point]
   if component!=actual_component or component!=term.component:raise ValueError('Independent Point component changed.')
   corners.append(point_spectrum(result,point,component,index,frequencies,total));seen.append((tuple(index),weight))
  if sorted(seen)!=sorted((tuple(i),w)for i,w in rows):raise ValueError('Actual Point corner support/weights differ from the fixed physical rows.')
  reference,bound=weighted_reference(np.stack(corners,axis=1),[w for _,_,w in entries],public_terms=8)
  checks.append(check_value(evidence[name]['primary'][:,0],reference,bound,label=name+'/published_native'))
  references[name]=reference;uncertainties[name]=bound
  bank,offsets=raw_trace(result,term,rows,frequencies,total)
  for side,key in enumerate(('constitutive_negative','constitutive_positive')):
   expected,limit=weighted_reference(bank[:,side,:],[w for _,w in rows],public_terms=8)
   if not bool(np.any(bank[:,side,:]!=0)):raise ValueError('Actual raw constitutive side is zero; no active-head proof.')
   checks.append(check_value(evidence[name][key][:,0],expected,limit,label=name+'/'+key))
  support.append(dict(name=name,owner=term.owner,component=term.component,normal_axis=term.normal_axis,normal_index=term.normal_index,global_native_rows=rows,independent_raw_bank_offsets=offsets))
 for name,(component,index)in plan.expected_points.items():
  references[name]=point_spectrum(result,name,component,index,frequencies,total);uncertainties[name]=np.zeros((nf,2))
 actual=array(VI,torch.complex128,(len(plan.rows),2,nf))
 for station,pair in enumerate(plan.rows):
  for kind,row in enumerate(pair):
   names=[n for n,_,_ in row];coeff=[w for _,_,w in row]
   expected,bound=weighted_reference(np.stack([references[n]for n in names],1),coeff,public_terms=len(row),uncertainty=np.stack([uncertainties[n]for n in names],1))
   checks.append(check_value(VI[station,kind],expected,bound,label=f'VI_station{station}_component{kind}'))
 raw_after=raw_fingerprints()
 if raw_before!=raw_after:raise ValueError('Independent raw head audit mutated its actual input bank.')
 return dict(raw_head_hashes_before={str(k):v for k,v in raw_before.items()},raw_head_hashes_after={str(k):v for k,v in raw_after.items()},independently_verified=True,primary_checks=6,constitutive_side_checks=12,VI_component_checks=2*len(plan.rows),checks=checks,actual_support=support,all_raw_banks_read_only=True,private_query_or_interpolation_helpers_used=False,phase_or_normalization_fit=False,posthoc_tolerance=False,F64_unit_roundoff=U64,public_corner_visits=8,bound_formula='(gamma_16+gamma_(2m+2))*sumabs_input/(1-gamma_2m); SI input bounds propagate with absolute fixed coefficients')


def verify_native_oracle_samples(context,monitors,prepared,*,expected_samples):
    """Compare actual published native primitives, not a private side query."""
    from witwin.maxwell.fdtd.coords import component_coords
    plan=context['plan'];frequencies=tuple(context['case'].frequencies);points={}
    def require_clock(payload):
        if tuple(payload.get('frequencies',()))!=frequencies:
            raise ValueError('An actual native primitive frequency order changed.')
        count=torch.as_tensor(payload.get('samples',())).reshape(-1)
        if len(count)!=len(frequencies)or not bool((count==expected_samples).all()):
            raise ValueError('Actual native primitive counts do not match the prefix stop.')
    for name,(component,index)in context['oracle_indices'].items():
        payload=monitors[name];require_clock(payload)
        if tuple(payload['field_indices'][component])!=tuple(index):
            raise ValueError('An actual Cu oracle Point changed its native component index.')
        value=payload['components'][component]
        if value.dtype!=torch.complex64 or value.numel()!=len(frequencies):
            raise ValueError('The published native primitive must retain its actual complex64 spectrum.')
        points[name]=value.reshape(-1).to(torch.complex128)
    records=[]
    for name,term in plan.queries.items():
        payload=monitors[plan.plane_names[name]];require_clock(payload)
        entry=payload.get('native_components',payload['components'])[term.component]
        if (payload['axis']!=term.normal_axis or entry.get('collocated',False)
            or tuple(entry.get('plane_weights',()))!=(1.,)
            or tuple(entry.get('plane_indices',()))!=(term.normal_index,)):
            raise ValueError('The primitive reference requires one actual uncollocated native plane.')
        data=entry['data'];tangents=[i for i in range(3)if i!='xyz'.index(term.normal_axis)]
        axes=component_coords(prepared,term.component);local_maps=[]
        for axis,coordinate in zip(tangents,entry['coords']):
            actual=np.asarray(coordinate,dtype=np.float64);native=np.asarray(axes[axis],dtype=np.float64)
            location=np.searchsorted(native,actual)
            if actual.ndim!=1 or not len(actual)or np.any(location>=len(native))or not np.array_equal(native[location],actual)or np.any(np.diff(actual)<=0):
                raise ValueError('Actual plane coordinates do not match unique native Yee coordinates.')
            local_maps.append({int(index):j for j,index in enumerate(location)})
        if data.dtype!=torch.complex64 or tuple(data.shape)!=(len(frequencies),len(local_maps[0]),len(local_maps[1])):
            raise ValueError('The actual native plane spectrum shape or precision changed.')
        zero=torch.zeros(len(frequencies),device=data.device,dtype=torch.complex128)
        point_value=sum((points[n]*weight for n,_,weight in context['query_oracle'][name]),zero)
        terms=[]
        for index,weight in plan.native_rows[name]:
            try:local=(local_maps[0][index[tangents[0]]],local_maps[1][index[tangents[1]]])
            except KeyError as error:raise ValueError('The actual native plane lacks a required physical stencil sample.')from error
            terms.append(data[:,local[0],local[1]].to(torch.complex128)*weight)
        plane_value=sum(terms,zero)
        if not torch.equal(point_value,plane_value):
            raise ValueError('Actual Cu native Point and published plane primitives differ; no fitting or tolerance change is allowed.')
        records.append(dict(term=name,owner=term.owner,native_rows=plan.native_rows[name],actual_samples=expected_samples,bitwise=True,maximum_absolute_difference=0.))
    return dict(selected_terms=len(records),native_Points=len(points),actual_Point_plane_weighted_primitive_parity=True,records=records,public_query_called=False,scope='Native primitive evidence at the real prefix clock; not a complete spectrum or constitutive/public positive query.')
