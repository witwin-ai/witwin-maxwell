"""Genuine bounded public component queries; no mode/tail/attenuation claims."""
from pathlib import Path
import gc,json,os,hashlib
import numpy as np
import pytest,torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from tests.gradients.test_fdtd_copper_foil_adjoint import FoilDesign

FREQUENCIES=(2.5e9,3e9,3.5e9)


def digest(value):
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous();return dict(dtype=str(x.dtype),shape=tuple(x.shape),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [digest(v) for v in value]
    if value is None or isinstance(value,(int,float,bool,str)):return value
    return repr(value)


def operators(solver):
    from scripts.dev.fdtd.printed_line_loss_execution import runtime_operator_signature,_material_signature
    return dict(Norton=runtime_operator_signature(solver),materials=_material_signature(solver),
        Cu=digest([dict(name=e['face'].name,component=e['electric_name'],indices=e['indices'],area=e['area'],B=e['B'],rates=e['rates'],D=e['D']) for e in solver._copper_foils]),
        sources=digest(dict(electric=solver._electric_source_terms,magnetic=solver._magnetic_source_terms,compiled=solver._compiled_sources)),dt=float(solver.dt))


def scene_and_points(normal,*,dielectric=False,model=None):
    if model is None:
        model=FoilDesign(normal_axis=normal);model.requires_grad_(False)
    base=model.to_scene()
    if dielectric:
        base=base.clone(structures=(*base.structures,mw.Structure(mw.Box(size=(.05,)*3),mw.Material(eps_r=4.,sigma_e=.01),name='dielectric')))
    prepared=prepare_scene(base);component='E'+'xyz'[(normal+2)%3]
    face=next(f for f in compile_copper_faces(prepared,device='cpu') if f.name=='foil' and f.component==(normal+2)%3)
    axes=[np.asarray(x) for x in component_coords(prepared,component)]
    score=sum(np.square(axes[a][face.indices[:,a].cpu().numpy()]) for a in range(3));index=face.indices[int(np.argmin(score))].cpu()
    position=tuple(float(axes[a][index[a]]) for a in range(3));offgrid=list(position)
    for axis in range(3):
        if axis!=normal:offgrid[axis]+=.25*(float(axes[axis][index[axis]+1])-position[axis])
    points=torch.tensor((position,tuple(offgrid)),device='cuda',dtype=torch.float64)
    observed=base.clone(monitors=(*base.monitors,mw.PointMonitor('native_exact',position,fields=(component,)),
        mw.PlaneMonitor('native_plane',axis='xyz'[normal],position=position[normal],fields=(component,))))
    return base,observed,points,component,index,('interior' if dielectric else 'exterior')


def simulation(scene,graph,*,steps=96,frequencies=FREQUENCIES):
    return mw.Simulation.fdtd(scene,frequencies=frequencies,cuda_graph=graph,full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=steps,transient_cycles=0,steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False),
        excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9,phase=.37)))


def owned_head_oracle(result,points,component,side,frequencies):
    """Independent 2D four-corner gather of actual raw heads, no query stencil."""
    trace=next(t for t in result.conductor_losses['foil']['bulk_electric_traces'] if t['component']==component)
    sampling=trace['sampling'];normal='xyz'.index(sampling['normal_axis']);axes=component_coords(result.prepared_scene,component)
    support={tuple(int(x) for x in row):i for i,row in enumerate(trace['indices'].cpu())};fields=trace['electric_fields'];tangents=[a for a in range(3) if a!=normal]
    result_values=[];absolute=[]
    for frequency in frequencies:
        fi=list(result.frequencies).index(frequency);values=[];magnitudes=[]
        for point in points.detach().cpu().numpy():
            lows=[];fractions=[]
            for a in tangents:
                low=int(np.searchsorted(axes[a],point[a],side='right')-1);lows.append(low);fractions.append((point[a]-axes[a][low])/(axes[a][low+1]-axes[a][low]))
            terms=[]
            for i in (0,1):
                for j in (0,1):
                    weight=(fractions[0] if i else 1-fractions[0])*(fractions[1] if j else 1-fractions[1])
                    if weight==0:continue
                    index=[0]*3;index[normal]=sampling['normal_index'];index[tangents[0]]=lows[0]+i;index[tangents[1]]=lows[1]+j
                    terms.append(fields[fi,side,support[tuple(index)]]*weight)
            values.append(sum(terms));magnitudes.append(sum(term.abs() for term in terms))
        result_values.append(torch.stack(values));absolute.append(torch.stack(magnitudes))
    return torch.stack(result_values).to(points.device),torch.stack(absolute).to(points.device)


def exact_physics(a,b):
    assert a.schema==b.schema and a.tensors.keys()==b.tensors.keys()
    for name,value in a.tensors.items():torch.testing.assert_close(value,b.tensors[name],rtol=0,atol=0)


@pytest.mark.parametrize('normal',range(3))
@pytest.mark.parametrize('graph',(False,True))
def test_genuine_multifrequency_selected_plane_and_side_fields(normal,graph,tmp_path):
    exercise(normal,graph,tmp_path,dielectric=False,frequencies=FREQUENCIES)


def test_genuine_singlefrequency_real_snapshot_plus_spectral_monitors(tmp_path):
    exercise(0,True,tmp_path,dielectric=False,frequencies=(3e9,))


def test_genuine_buried_lossy_dielectric_uses_existing_interior_classification(tmp_path):
    exercise(2,True,tmp_path,dielectric=True,frequencies=FREQUENCIES)


def exercise(normal,graph,folder,*,dielectric,frequencies):
    if not torch.cuda.is_available():raise RuntimeError('Genuine GPU query qualification cannot skip.')
    base,scene,points,component,index,region=scene_and_points(normal,dielectric=dielectric)
    old=simulation(base,graph,frequencies=frequencies).run();before=capture_checkpoint_state(old.solver,step=96);operator=operators(old.solver)
    result=simulation(scene,graph,frequencies=frequencies).run();after=capture_checkpoint_state(result.solver,step=96)
    exact_physics(before,after);assert operators(result.solver)==operator
    assert result.stats()['cuda_graph_active'] is graph
    assert result.stats()['observer_cuda_graph_active'] is graph
    torch.testing.assert_close(old.monitor('probe')['data'],result.monitor('probe')['data'],rtol=0,atol=0)
    for owner in old.conductor_losses:
        for a,b in zip(old.conductor_losses[owner]['bulk_electric_traces'],result.conductor_losses[owner]['bulk_electric_traces']):
            torch.testing.assert_close(a['electric_fields'],b['electric_fields'],rtol=0,atol=0)
    chosen=(frequencies[-1],) if len(frequencies)==1 else (frequencies[-1],frequencies[0])
    public={};contrasts=[]
    for side,bank in (('negative',0),('positive',1)):
        values=result.query_fields(points,region_id=region,field_kind='total',components=(component,),frequencies=chosen,trace_side=side,owner='foil',representation='constitutive')[component]
        oracle,scale=owned_head_oracle(result,points,component,bank,chosen)
        assert values.shape==(len(chosen),2) and bool((values.abs()>0).any())
        assert bool(((values-oracle).abs()<=16*torch.finfo(torch.float64).eps*scale.clamp_min(torch.finfo(torch.float64).tiny)).all())
        torch.testing.assert_close(values[:,0],oracle[:,0],rtol=0,atol=0);public[side]=values.detach().cpu()
    contrasts.append(float((public['positive']-public['negative']).abs().max()))
    published=result.query_fields(points[:1],region_id=region,field_kind='total',components=(component,),frequencies=chosen,trace_side='negative',owner='foil',representation='published_native')[component]
    point=result.monitor('native_exact')['components'][component].reshape(-1);selected=[list(frequencies).index(f) for f in chosen]
    assert point.dtype==torch.complex64;torch.testing.assert_close(published[:,0],point[selected].to(torch.complex128),rtol=0,atol=0)
    if len(frequencies)==1:
        assert result._fields and not result._fields[component.upper()].is_complex()
    with pytest.raises(ValueError,match='negative trace_side'):
        result.query_fields(points,region_id=region,field_kind='total',components=(component,),trace_side='positive',owner='foil',representation='published_native')
    with pytest.raises(NotImplementedError,match='coordinate or shape gradients'):
        result.query_fields(points.clone().requires_grad_(),region_id=region,field_kind='total',components=(component,),trace_side='negative',owner='foil')
    with pytest.raises(ValueError,match='trace_side'):
        result.query_fields(points,region_id=region,field_kind='total',components=(component,),owner='foil')
    path=folder/'result.pt';result.save(path);restored=mw.Result.load(path,scene=scene,prepared_scene=result.prepared_scene,map_location='cuda')
    for side in public:
        actual=restored.query_fields(points,region_id=region,field_kind='total',components=(component,),frequencies=chosen,trace_side=side,owner='foil')[component]
        torch.testing.assert_close(actual.cpu(),public[side],rtol=0,atol=0)
    # Corrupt only an independently loaded result; original raw/runtime is intact.
    for trace in restored._metadata['conductor_losses']['foil']['bulk_electric_traces']:
        trace['sampling']['completed_steps']=torch.tensor(95.,dtype=torch.float64)
    with pytest.raises(ValueError,match='count/norm'):
        restored.query_fields(points,region_id=region,field_kind='total',components=(component,),owner='foil',trace_side='negative')
    checkpoint=simulation(scene,graph,frequencies=frequencies).prepare().run_until(48);checkpoint.save(folder/'checkpoint48.pt')
    resumed=simulation(scene,graph,frequencies=frequencies).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(folder/'checkpoint48.pt',map_location='cpu'))
    exact_physics(after,capture_checkpoint_state(resumed.solver,step=96));assert operators(resumed.solver)==operator
    for side in public:
        actual=resumed.query_fields(points,region_id=region,field_kind='total',components=(component,),frequencies=chosen,trace_side=side,owner='foil')[component]
        torch.testing.assert_close(actual.cpu(),public[side],rtol=0,atol=0)
    torch.save(dict(original=before,observed=after,resumed=capture_checkpoint_state(resumed.solver,step=96),public=public,published=published.cpu(),Point=point.cpu(),operator=operator,normal=normal,graph=graph,dielectric=dielectric,frequencies=frequencies,contrast=contrasts),folder/'actual-state-and-query.pt')
    (folder/'actual-query.json').write_text(json.dumps(dict(normal=normal,graph=graph,dielectric=dielectric,region=region,frequencies=frequencies,source_operator_unchanged=True,all_physics_and_original_probe_bitwise=True,actual_constitutive_positive_negative_difference_max=contrasts[0],actual_field_steps=288,actual_checkpoint_step=48,actual_schedule=result.conductor_losses['foil']['bulk_electric_traces'][0]['sampling']['schema'],no_mode_tail_or_attenuation_claim=True),indent=2)+'\n')


def test_actual_native_Cu_clock_uses_positive_DFT_exponent(tmp_path):
    if not torch.cuda.is_available():raise RuntimeError('Actual native clock oracle cannot skip CUDA.')
    _,scene,points,component,index,region=scene_and_points(0)
    histories=[];last=None;runtime_index=None;N=17
    for n in range(1,N):
        execution=simulation(scene,True,steps=N).prepare();checkpoint=execution.run_until(n);solver=execution.solver
        current=next((i,e) for i,e in enumerate(solver._copper_foils) if not e['imaginary'] and e['electric_name']==component and e['face'].name=='foil')
        k,entry=current;matches=torch.nonzero((entry['indices']==index.to(entry['indices'].device)).all(-1)).flatten();assert len(matches)==1
        edge=int(matches[0]);left=getattr(solver,component)[tuple(index.to(solver.device))].double();right=entry['right_e'][edge]
        histories.append(torch.stack((left,right)).detach().cpu());assert float(entry['spectral_step'])==n
        if n==N-1:
            last={key:entry[key].detach().cpu().clone() for key in ('spectral_real','spectral_imag','spectral_norm','spectral_frequencies')};dt=float(solver.dt)
            captured=(torch.complex(last['spectral_real'],last['spectral_imag'])[:,:,edge]);runtime_index=k
        del execution,checkpoint,solver,entry;gc.collect();torch.cuda.empty_cache()
    result=simulation(scene,True,steps=N).run();solver=result.solver
    entry=next(e for e in solver._copper_foils if not e['imaginary'] and e['electric_name']==component and e['face'].name=='foil')
    edge=int(torch.nonzero((entry['indices']==index.to(entry['indices'].device)).all(-1)).flatten()[0])
    histories.append(torch.stack((getattr(solver,component)[tuple(index.to(solver.device))].double(),entry['right_e'][edge])).detach().cpu())
    prefix_captured=captured.clone();captured=torch.complex(entry['spectral_real'],entry['spectral_imag'])[:,:,edge].detach().cpu()
    assert float(entry['spectral_step'])==N and torch.equal(entry['spectral_norm'].cpu(),torch.full_like(last['spectral_norm'],float(N)))
    trace=torch.stack(histories);frequency=last['spectral_frequencies'].numpy();times=(np.arange(N)+1)*dt
    # Independent NumPy clock oracle follows the actual accumulator sign; the sample after update n is labelled (n+1)*dt.
    weights=np.exp(2j*np.pi*frequency[:,None]*times[None,:]);manual=np.einsum('ft,ts->fs',weights,trace.numpy())
    scale=np.einsum('ft,ts->fs',np.abs(weights),np.abs(trace.numpy()));error=np.abs(captured.numpy()-manual)
    assert np.all(error<=64*np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny))
    wrong=np.einsum('ft,ts->fs',np.conj(weights),trace.numpy());wrong_relative=np.linalg.norm(wrong-manual)/np.linalg.norm(manual)
    assert np.isfinite(wrong_relative) and wrong_relative>1e-5
    prefix_manual=np.einsum('ft,ts->fs',weights[:,:-1],trace[:-1].numpy())
    assert np.all(np.abs(prefix_captured.numpy()-prefix_manual)<=64*np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny))
    for side,bank in (('negative',0),('positive',1)):
        queried=result.query_fields(points[:1],region_id=region,field_kind='total',components=(component,),owner='foil',trace_side=side)[component][:,0].cpu()
        oracle=torch.from_numpy(manual[:,bank]*(2./N))
        torch.testing.assert_close(queried,oracle,rtol=64*torch.finfo(torch.float64).eps,atol=64*torch.finfo(torch.float64).eps*float(np.max(scale[:,bank]))*2/N)
    torch.save(dict(actual_native_step_Eminus_Eplus=trace,actual_raw_accumulator=captured,independent_positive_exponent=manual,opposite_exponent_negative=wrong,actual_norm=entry['spectral_norm'].detach().cpu(),actual_total_steps=N,actual_checkpoint_steps=list(range(1,N)),frequencies_Hz=frequency,dt_s=dt,physical_phasor_convention='E(t)=Re(a exp(-i omega t))',DFT_accumulator_convention='exp(+i omega n dt)',wrong_sign_relative=wrong_relative),tmp_path/'actual-clock-oracle.pt')


def test_actual_selected_side_parameter_VJPs_match_independent_ordinary_FD(tmp_path):
    if not torch.cuda.is_available():raise RuntimeError('Actual live side VJP qualification cannot skip CUDA.')
    class Design(FoilDesign):
        def to_scene(self):
            base=super().to_scene();prepared=prepare_scene(base);normal=self.normal_axis;component='E'+'xyz'[(normal+2)%3]
            face=next(f for f in compile_copper_faces(prepared,device='cpu') if f.component==(normal+2)%3)
            axes=component_coords(prepared,component);index=face.indices[len(face.indices)//2];position=tuple(float(axes[a][index[a]]) for a in range(3))
            self.query_point=position;self.query_component=component
            return base.clone(monitors=(*base.monitors,mw.PlaneMonitor('native_plane',axis='xyz'[normal],position=position[normal],fields=(component,))))
    model=Design(normal_axis=0,graph=True,stride=7)
    def objective():
        config=mw.Simulation.fdtd(model,frequencies=FREQUENCIES,cuda_graph=True,full_field_dft=False,
            run_time=mw.TimeConfig(time_steps=96,transient_cycles=0,steady_cycles=1),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False),
            excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9,phase=.37)))
        config.config.adjoint_checkpoint_stride=7;result=config.run();points=torch.tensor((model.query_point,),device='cuda',dtype=torch.float64)
        negative=result.query_fields(points,region_id='exterior',field_kind='total',components=(model.query_component,),owner='foil',trace_side='negative')[model.query_component]
        positive=result.query_fields(points,region_id='exterior',field_kind='total',components=(model.query_component,),owner='foil',trace_side='positive')[model.query_component]
        assert bool((negative.abs()>0).any()) and bool((positive.abs()>0).any())
        return negative.abs().square().sum()+2*positive.abs().square().sum(),result
    value,result=objective();assert value.requires_grad;value.backward();gradients={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.requires_grad_(False);baseline,_=objective();torch.testing.assert_close(baseline,value.detach(),rtol=3e-4,atol=1e-25)
    records=[]
    for name,parameter in model.named_parameters():
        origin=parameter.detach().clone();errors=[]
        for delta in (.08,.04,.02,.01):
            with torch.no_grad():parameter.copy_(origin+delta)
            plus,_=objective()
            with torch.no_grad():parameter.copy_(origin-delta)
            minus,_=objective()
            with torch.no_grad():parameter.copy_(origin)
            finite=(plus-minus)/(2*delta);error=float(abs((finite-gradients[name])/finite));errors.append(error)
            records.append(dict(parameter=name,delta=delta,adjoint=float(gradients[name]),ordinary_FD=float(finite),relative_error=error))
        assert abs(float(gradients[name]))>1e-30 and sum(error<.01 for error in errors)>=2
    torch.save(dict(objective=value.detach().cpu(),gradients={k:v.cpu() for k,v in gradients.items()},FD_records=records),tmp_path/'actual-parameter-VJP.pt')
