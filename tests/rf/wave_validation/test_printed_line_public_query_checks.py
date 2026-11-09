"""Self-contained fixed-bank CPU fixtures, not simulated field receipts."""
from types import SimpleNamespace as NS
import copy
import numpy as np,pytest,torch
from scripts.dev.fdtd.printed_line_public_vi_checks import independent_public_audit,weighted_reference,check_value


def fixture(scale=1.):
 F=5;N=23025;freq=(.8e9,.9e9,1e9,1.1e9,1.2e9);queries={};rows={};points={};oracle_indices={};oracle={};monitors={};evidence={};primitive={}
 indices=[(j+2,7+j%2,3)for j in range(12)];order=[11,0,9,2,7,4,5,6,3,8,1,10];stored=[indices[j]for j in order]
 left=np.array([[complex(2+.3*f+.2*j,.4+.1*j-.07*f)*scale for j in range(12)]for f in range(F)],np.complex128);right=left*(1.2+.45j)+(.3-.1j)*scale
 banks=np.stack([left[:,order],right[:,order]],1)
 for j in range(6):
  name='q'+str(j);queries[name]=NS(owner='foil',component='Ey',normal_axis='z',normal_index=3,normal_coordinate=.003)
  rows[name]=tuple((indices[k],w)for k,w in((2*j,.25),(2*j+1,.75)));entries=[];native=[]
  for k,(_,w)in zip((2*j,2*j+1),rows[name]):
   monitor='corner'+str(k);a=torch.tensor(left[:,k]*(.91+.13j),dtype=torch.complex64);oracle_indices[monitor]=('Ey',indices[k]);entries.append((monitor,'Ey',w));native.append(a.to(torch.complex128))
   monitors[monitor]=dict(kind='point',frequencies=freq,field_indices={'Ey':indices[k]},samples=torch.full((F,),N,dtype=torch.float64),components={'Ey':a})
  oracle[name]=tuple(entries);primary=sum((v*w for v,(_,_,w)in zip(native,entries)),torch.zeros(F,dtype=torch.complex128));primitive[name]=primary
  offsets=[order.index(2*j),order.index(2*j+1)];weight=torch.tensor([.25,.75],dtype=torch.float64)
  evidence[name]=dict(primary=primary[:,None],constitutive_negative=(torch.tensor(banks[:,0,offsets])*weight).sum(1)[:,None],constitutive_positive=(torch.tensor(banks[:,1,offsets])*weight).sum(1)[:,None])
 for j in range(2):
  name='H'+str(j);component='Hz';index=(5+j,2,1);points[name]=(component,index);a=torch.tensor(np.arange(F)*.1+complex(1+j,.3+j)*scale,dtype=torch.complex64);primitive[name]=a.to(torch.complex128);monitors[name]=dict(frequencies=freq,field_indices={component:index},samples=torch.full((F,),N,dtype=torch.float64),components={component:a})
 physical_rows=tuple((((f'q{2*j}','Ey',.03),(f'q{2*j+1}','Ey',-.01)),(('H0','Hz',.02),('H1','Hz',-.003)))for j in range(3))
 VI=torch.stack([torch.stack([sum((primitive[n]*w for n,_,w in row),torch.zeros(F,dtype=torch.complex128))for row in pair])for pair in physical_rows])
 trace=dict(component='Ey',normal_axis='z',normal_coordinate=.003,indices=torch.tensor(stored,dtype=torch.int64),electric_fields=torch.tensor(banks),sampling=dict(owner='foil',component='Ey',planned_steps=N,completed_steps=torch.tensor(float(N),dtype=torch.float64),frequencies=torch.tensor(freq,dtype=torch.float64),weight_sums=torch.full((F,),N,dtype=torch.float64)))
 result=NS(conductor_losses={'foil':dict(frequencies=torch.tensor(freq,dtype=torch.float64),bulk_electric_traces=[trace])},monitor=lambda name:monitors[name])
 context=dict(plan=NS(frequencies=freq,queries=queries,native_rows=rows,expected_points=points,rows=physical_rows),oracle_indices=oracle_indices,query_oracle=oracle)
 return context,result,VI,evidence,trace,monitors

@pytest.mark.parametrize('scale',[1.,1e-20,1e20])
def test_independent_six_twelve_SI_checks_with_permuted_actual_raw_rows(scale):
 c,r,v,e,_,_=fixture(scale);proof=independent_public_audit(c,r,v,e,total=23025)
 assert proof['primary_checks']==6 and proof['constitutive_side_checks']==12 and proof['VI_component_checks']==6
 assert not proof['phase_or_normalization_fit']and not proof['posthoc_tolerance']

@pytest.mark.parametrize('failure',['primary_phase','positive_side_swap','head_row_swap','native_dtype','head_dtype','wrong_owner','wrong_normal','duplicate_row','missing_row','raw_frequency_order','point_native_index','point_clock','trace_clock','context_weight','zero_primary','zero_positive','zero_VI','VI_phase'])
def test_wrong_phase_side_support_dtype_owner_clock_and_zero_reject(failure):
 c,r,v,e,t,m=fixture()
 if failure=='primary_phase':e['q0']['primary']*=complex(1,1e-6)
 elif failure=='positive_side_swap':e['q0']['constitutive_positive']=e['q0']['constitutive_negative'].clone()
 elif failure=='head_row_swap':t['electric_fields'][:,:,0],t['electric_fields'][:,:,1]=t['electric_fields'][:,:,1].clone(),t['electric_fields'][:,:,0].clone()
 elif failure=='native_dtype':m['corner0']['components']['Ey']=m['corner0']['components']['Ey'].to(torch.complex128)
 elif failure=='head_dtype':t['electric_fields']=t['electric_fields'].to(torch.complex64)
 elif failure=='wrong_owner':c['plan'].queries['q0'].owner='wrong'
 elif failure=='wrong_normal':t['normal_coordinate']+=.001
 elif failure=='duplicate_row':t['indices'][0]=t['indices'][1]
 elif failure=='missing_row':t['indices'][1,0]=999
 elif failure=='raw_frequency_order':r.conductor_losses['foil']['frequencies']=r.conductor_losses['foil']['frequencies'].flip(0)
 elif failure=='point_native_index':m['corner0']['field_indices']['Ey']=(2,7,4)
 elif failure=='point_clock':m['corner0']['samples']-=1
 elif failure=='trace_clock':t['sampling']['completed_steps']-=1
 elif failure=='context_weight':c['query_oracle']['q0']=(('corner0','Ey',.26),('corner1','Ey',.74))
 elif failure=='zero_primary':e['q0']['primary']=torch.zeros_like(e['q0']['primary'])
 elif failure=='zero_positive':t['electric_fields'][:,1,:]=0;e['q0']['constitutive_positive']*=0
 elif failure=='zero_VI':v*=0
 elif failure=='VI_phase':v[0,1]*=complex(1,1e-6)
 with pytest.raises(ValueError):independent_public_audit(c,r,v,e,total=23025)


def test_componentwise_cancellation_bound_uses_only_input_products():
 values=np.array([[3.+2j,-1.-2j/3]],dtype=np.complex128);ref,bound=weighted_reference(values,[.25,.75],public_terms=8)
 assert bound.max()>0 and abs(ref[0])<1e-15
 perturbed=torch.tensor(ref+1e-6,dtype=torch.complex128)
 with pytest.raises(ValueError):check_value(perturbed,ref,bound,label='wrong finite cancellation',nonzero=False)
 good=torch.tensor(ref,dtype=torch.complex128);check_value(good,ref,bound,label='exact reference',nonzero=False)
