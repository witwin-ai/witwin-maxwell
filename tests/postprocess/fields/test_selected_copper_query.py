"""CPU algebra on real compiled sheet geometry; no GPU propagation claim."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tests.postprocess.fields.test_copper_query_interpolation import fixture
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.copper_foil import copper_sampling_provenance
from witwin.maxwell.postprocess.selected_query import query_selected, validate_sampling, selections
from witwin.maxwell.postprocess.yee_interpolation import copper_query_faces

FREQUENCIES=(1e9,1.5e9,2e9)
AMPLITUDES=torch.tensor((1.,.3+.7j,-.8+.4j),dtype=torch.complex128)


def prescribed(normal=0, *, planes=True):
    result,expected,_=fixture(normal,graded=True)
    result.frequencies=FREQUENCIES
    result._solver_stats.update(normalize_source=False,dft_window='none',time_steps=16,steps_run=16,
        dt=1e-12,observer_sample_counts=(16,16,16),time_plan={'source_representation':'real_projected'})
    record=result._metadata['conductor_losses']['foil'];record['frequencies']=torch.tensor(FREQUENCIES,dtype=torch.float64)
    record['spectral_valid']=torch.ones(3,dtype=torch.bool)
    entry=dict(spectral_frequencies=torch.tensor(FREQUENCIES,dtype=torch.float64),spectral_starts=torch.zeros(3,dtype=torch.float64),
        spectral_end=16,spectral_step=torch.tensor(16.,dtype=torch.float64),spectral_window='none',spectral_norm=torch.full((3,),16.,dtype=torch.float64))
    solver=SimpleNamespace(dt=1e-12,bloch_k=(0,0,0),_spectral_integration_dt=None,_spectral_reference_frequency=None,_time_plan={'source_representation':'real_projected'},_compiled_sources=())
    # This is prescribed metadata/field algebra, not an executed solver fixture.
    # Spectral attributes follow the solver rules: per-frequency reference period,
    # and no Fourier-integral dt without compiled pulsed sources.
    for trace in record['bulk_electric_traces']:
        trace['electric_fields']=trace['electric_fields']*AMPLITUDES[:,None,None]
        entry['face']=SimpleNamespace(name='foil',axis=normal,normal_index=int(trace['indices'][0,normal]));entry['electric_name']=trace['component']
        trace['sampling']=copper_sampling_provenance(solver,entry)
    result._fields.clear();result._monitors.clear()
    if planes:
        for trace in record['bulk_electric_traces']:
            component=trace['component'];axes=[torch.tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,component)]
            normal_index=int(trace['indices'][0,normal]);tangents=[a for a in range(3) if a!=normal]
            indices=[torch.arange(2,len(axes[a])-2) for a in tangents]
            coords=[axes[a][index] for a,index in zip(tangents,indices)]
            mesh=torch.meshgrid(*coords,indexing='ij');points=torch.zeros((*mesh[0].shape,3),dtype=torch.float64)
            points[...,normal]=axes[normal][normal_index]
            for a,coordinate in zip(tangents,mesh):points[...,a]=coordinate
            # Deliberately retain the published C64 primitive independently.
            values=(AMPLITUDES[:,None,None]*expected(points,component,False)).to(torch.complex64)
            result._monitors[component]=dict(kind='plane',axis='xyz'[normal],frequencies=FREQUENCIES,samples=(16,16,16),
                components={component:dict(data=values,coords=coords,plane_indices=(normal_index,),plane_weights=(1.,),collocated=False)})
    points=torch.tensor(((.017,.027,.033),(-.019,.025,-.031)),dtype=torch.float64);points[:,normal]=0
    return result,points,expected


def sample(result,points,component,side='negative',representation='constitutive',frequencies=(2e9,1e9)):
    return query_selected(result,points,components=(component,),frequencies=frequencies,
        trace_side=side,owner='foil',representation=representation)[component]


@pytest.mark.parametrize('normal',range(3))
@pytest.mark.parametrize('side',('negative','positive'))
def test_affine_both_real_compiled_tangential_sides_and_selected_frequency_order(normal,side):
    result,points,expected=prescribed(normal)
    for component in ('E'+'xyz'[a] for a in range(3) if a!=normal):
        actual=sample(result,points,component,side)
        oracle=AMPLITUDES[[2,0],None]*expected(points,component,side=='positive')
        assert actual.shape==(2,2) and actual.dtype==torch.complex128
        torch.testing.assert_close(actual,oracle,rtol=3e-15,atol=3e-15)


def test_constitutive_head_only_does_not_invent_unrequested_H_or_normal_E():
    result,points,expected=prescribed(planes=False)
    assert not result._monitors and not result._fields
    torch.testing.assert_close(sample(result,points,'Ey','positive'),AMPLITUDES[[2,0],None]*expected(points,'Ey',True),rtol=3e-15,atol=3e-15)
    with pytest.raises(ValueError,match='tangential trace|owner'):
        sample(result,points,'Hx')


def independent_plane_value(result,points,component,frequency_indices=(2,0),absolute_sum=False):
    entry=result._monitors[component]['components'][component];normal='xyz'.index(result._monitors[component]['axis']);tangents=[a for a in range(3) if a!=normal]
    values=[]
    for fi in frequency_indices:
        fields=[]
        for point in points:
            lows=[];fractions=[]
            for axis,coordinates in zip(tangents,entry['coords']):
                low=int(torch.searchsorted(coordinates,point[axis],right=True))-1
                lows.append(low);fractions.append((point[axis]-coordinates[low])/(coordinates[low+1]-coordinates[low]))
            terms=[entry['data'][fi,lows[0]+i,lows[1]+j].to(torch.complex128)*(fractions[0] if i else 1-fractions[0])*(fractions[1] if j else 1-fractions[1]) for i in (0,1) for j in (0,1)]
            value=sum(term.abs() if absolute_sum else term for term in terms)
            fields.append(value)
        values.append(torch.stack(fields))
    return torch.stack(values)


def test_published_native_primitive_uses_real_C64_plane_with_high_precision_weights():
    result,points,_=prescribed()
    component='Ey';actual=sample(result,points,component,representation='published_native')
    assert result._monitors[component]['components'][component]['data'].dtype==torch.complex64
    oracle=independent_plane_value(result,points,component)
    # Different legal associations of FP64 weighted sums have a forward
    # roundoff bound. Exact native primitive equality is tested separately.
    error=(actual-oracle).abs()
    scale=independent_plane_value(result,points,component,absolute_sum=True).clamp_min(torch.finfo(torch.float64).tiny)
    assert bool((error<=16*torch.finfo(torch.float64).eps*scale).all())
    # A cast constitutive head is a different primitive and cannot substitute.
    constituent=sample(result,points,component)
    assert float((actual-constituent).abs().max())>1e-9
    with pytest.raises(ValueError,match='negative trace_side'):
        sample(result,points,component,'positive','published_native')


def test_exact_native_point_and_plane_primitive_agree_without_fitted_scale():
    result,_,_=prescribed();entry=result._monitors['Ey']['components']['Ey'];coords=entry['coords']
    points=torch.tensor(((0.,float(coords[0][2]),float(coords[1][2])),),dtype=torch.float64)
    face=next(f for f in copper_query_faces(result,device='cpu') if f['component']==1)
    axes=component_coords(result.prepared_scene,'Ey');index=(face['normal'],int(np.nonzero(axes[1]==float(points[0,1]))[0][0]),int(np.nonzero(axes[2]==float(points[0,2]))[0][0]))
    result._monitors['original_point']=dict(kind='point',frequencies=FREQUENCIES,samples=(16,16,16),field_indices={'Ey':index},components={'Ey':entry['data'][:,2,2]})
    actual=sample(result,points,'Ey',representation='published_native')
    torch.testing.assert_close(actual[:,0],entry['data'][[2,0],2,2].to(torch.complex128),rtol=0,atol=0)
    result._monitors['original_point']['components']['Ey']=entry['data'][:,2,2]*1.001
    with pytest.raises(ValueError,match='conflict'):sample(result,points,'Ey',representation='published_native')


@pytest.mark.parametrize('key,wrong',[('dt',2e-12),('planned_steps',17),('completed_steps',15.),('weight_sums',torch.tensor((16.,15.,16.))),('start_steps',torch.tensor((0.,1.,0.))),('window','hanning'),('phasor_factor',1.),('normalize_source',True),('electric_phase_origin_step',0),('electric_time_offset_steps',.5),('phasor_convention','exp(+iwt)'),('frequencies',torch.tensor((1e9,2e9,1.5e9))),('owner','foreign'),('component','Ez'),('normal_axis','z'),('normal_index',5),('energy_representation','combined_real_imaginary_algorithmic_state')])
def test_schedule_mismatch_rejected_without_complex_scale_estimation(key,wrong):
    result,points,_=prescribed()
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:trace['sampling'][key]=wrong
    with pytest.raises(ValueError,match='clock|count/norm|sampling'):sample(result,points,'Ey')


def test_missing_snapshot_or_owner_is_not_inferred_from_native_left():
    result,points,_=prescribed()
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:trace.pop('sampling')
    with pytest.raises(ValueError,match='actual per-face sampling provenance'):sample(result,points,'Ey')
    result,points,_=prescribed()
    for owner in (None,'foreign'):
        with pytest.raises(ValueError,match='owner'):
            query_selected(result,points,components=('Ey',),frequencies=None,trace_side='negative',owner=owner,representation='constitutive')
    with pytest.raises(ValueError,match='requires trace_side'):
        sample(result,points,'Ey',side=None)


@pytest.mark.parametrize('mutation',('collocated','normal_weights','offgrid','missing_frequency','missing_corner','missing_sample_count'))
def test_sparse_plane_strict_support_negatives(mutation):
    result,points,_=prescribed();payload=result._monitors['Ey'];entry=payload['components']['Ey']
    if mutation=='collocated':entry['collocated']=True
    if mutation=='normal_weights':entry['plane_weights']=(.5,.5)
    if mutation=='offgrid':entry['coords'][0]=entry['coords'][0]+1e-8
    if mutation=='missing_frequency':payload['frequencies']=(1e9,1.4e9,1.8e9)
    if mutation=='missing_corner':entry['coords'][0]=entry['coords'][0][:1];entry['data']=entry['data'][:,:1]
    if mutation=='missing_sample_count':payload.pop('samples')
    with pytest.raises(ValueError):sample(result,points,'Ey',representation='published_native')


@pytest.mark.parametrize('side',('negative','positive'))
def test_constitutive_side_fixed_topology_VJP_matches_independent_affine_field_FD(side):
    result,points,expected=prescribed(planes=False);trace=next(t for t in result._metadata['conductor_losses']['foil']['bulk_electric_traces'] if t['component']=='Ey');base=trace['electric_fields'].clone();parameter=torch.tensor(.17,dtype=torch.float64,requires_grad=True)
    def objective(value):
        trace['electric_fields']=torch.stack((base[:,0]*torch.exp(value),base[:,1]*(1+value.square())),1)
        field=sample(result,points,'Ey',side)
        return (field.abs().square()+.2*field.real-.3*field.imag).sum()
    actual=objective(parameter);gradient=torch.autograd.grad(actual,parameter)[0]
    def oracle(value):
        field=AMPLITUDES[[2,0],None]*expected(points,'Ey',side=='positive')*(torch.exp(value) if side=='negative' else 1+value.square())
        return (field.abs().square()+.2*field.real-.3*field.imag).sum()
    exact=torch.autograd.grad(oracle(parameter),parameter)[0]
    finite=(objective(parameter.detach()+1e-5)-objective(parameter.detach()-1e-5))/(2e-5)
    torch.testing.assert_close(gradient,exact,rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(gradient,finite,rtol=1e-9,atol=1e-9)


def test_published_plane_fixed_topology_VJP_matches_independent_native_weights():
    result,points,_=prescribed();entry=result._monitors['Ey']['components']['Ey'];leaf=entry['data'].clone().requires_grad_();entry['data']=leaf
    value=sample(result,points,'Ey',representation='published_native');actual=torch.autograd.grad((value.real+.7*value.imag).sum(),leaf)[0]
    oracle=independent_plane_value(result,points,'Ey');expected=torch.autograd.grad((oracle.real+.7*oracle.imag).sum(),leaf)[0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_default_all_six_single_frequency_guard_stays_unchanged():
    result,points,_=prescribed()
    with pytest.raises(NotImplementedError,match='one recorded frequency'):
        result.query_fields(points,region_id='exterior',field_kind='total')
    with pytest.raises(ValueError,match='explicit components'):
        result.query_fields(points,region_id='exterior',field_kind='total',frequencies=(1e9,))


@pytest.mark.parametrize('mutation',('one_side','edge_count','support_normal','foreign_edge','duplicate_owner'))
def test_actual_head_ownership_schema_cannot_be_relabelled(mutation):
    result,points,_=prescribed();record=result._metadata['conductor_losses']['foil']
    trace=next(t for t in record['bulk_electric_traces'] if t['component']=='Ey')
    if mutation=='one_side':trace['electric_fields']=trace['electric_fields'][:,:1]
    if mutation=='edge_count':trace['electric_fields']=trace['electric_fields'][:,:,:-1]
    if mutation=='support_normal':trace['indices']=trace['indices'].clone();trace['indices'][:,0]+=1
    if mutation=='foreign_edge':trace['indices']=trace['indices'].clone();trace['indices'][0,1]=0
    if mutation=='duplicate_owner':record['bulk_electric_traces'].append(copy.deepcopy(trace))
    with pytest.raises(ValueError,match='heads|owner|ownership|normal|support'):
        sample(result,points,'Ey')


def test_partial_frequency_selection_rejects_absent_and_duplicate_frequency():
    result,points,_=prescribed()
    for frequencies in ((1.2e9,), (1e9,1e9), (float('nan'),), ()):
        with pytest.raises(ValueError):sample(result,points,'Ey',frequencies=frequencies)
    for components in ((), ('Ey','Ey'), ('ey',)):
        with pytest.raises(ValueError):selections(result,components,None)


def test_producer_provenance_is_an_owned_snapshot_not_a_clock_guess():
    entry=dict(spectral_frequencies=torch.tensor(FREQUENCIES,dtype=torch.float64),spectral_starts=torch.zeros(3,dtype=torch.float64),spectral_end=16,
        spectral_step=torch.tensor(13.,dtype=torch.float64),spectral_window='none',spectral_norm=torch.full((3,),13.,dtype=torch.float64))
    entry['face']=SimpleNamespace(name='foil',axis=0,normal_index=4);entry['electric_name']='Ey'
    solver=SimpleNamespace(dt=2e-12,_spectral_integration_dt=None,_spectral_reference_frequency=None,_compiled_sources=(),_time_plan={'source_representation':'real_projected'})
    record=copper_sampling_provenance(solver,entry)
    entry['spectral_step'].fill_(99);entry['spectral_norm'].zero_()
    assert float(record['completed_steps'])==13 and torch.equal(record['weight_sums'],torch.full((3,),13.,dtype=torch.float64))
    assert record['planned_steps']==16 and record['dt']==2e-12


@pytest.mark.parametrize('key',('frequencies','start_steps','completed_steps','weight_sums'))
def test_clock_record_dtype_cannot_be_silently_promoted_into_provenance(key):
    result,points,_=prescribed()
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
        trace['sampling'][key]=trace['sampling'][key].float()
    with pytest.raises(ValueError,match='actual typed float64 clock'):
        sample(result,points,'Ey')


def test_lossy_dielectric_is_not_mistaken_for_nonunit_permeability():
    import witwin.maxwell as mw
    result,points,_=prescribed()
    result.scene=result.scene.clone(structures=(*result.scene.structures,mw.Structure(mw.Box(size=(.5,.5,.5)),mw.Material(eps_r=4.,sigma_e=.01),name='dielectric')))
    sample(result,points,'Ey','positive')


def test_selected_singleton_axes_and_actual_metadata_save_load(tmp_path):
    import witwin.maxwell as mw
    result,points,_=prescribed()
    before=sample(result,points[:1],'Ey','positive',frequencies=(1.5e9,))
    assert before.shape==(1,1)
    result.save(tmp_path/'prescribed-algebra.pt')
    loaded=mw.Result.load(tmp_path/'prescribed-algebra.pt',scene=result.scene,prepared_scene=result.prepared_scene,map_location='cpu')
    after=sample(loaded,points[:1],'Ey','positive',frequencies=(1.5e9,))
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    metadata=loaded.conductor_losses['foil']['bulk_electric_traces'][0]['sampling']
    assert metadata['schema']=='native-copper-electric-sampling/1' and float(metadata['completed_steps'])==16


def test_selected_ordinary_components_reuse_native_dense_stencils_without_heads():
    result,_,expected=prescribed(planes=False)
    result._solver_stats['dft_sample_counts']=(16,16,16)
    for component in ('Ey','Hz'):
        axes=[torch.tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,component)]
        points=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
        result._fields[component.upper()]=AMPLITUDES[:,None,None,None]*expected(points,component)
    points=torch.tensor(((.43,.44,.45),),dtype=torch.float64)
    output=query_selected(result,points,components=('Hz','Ey'),frequencies=(1.5e9,2e9),trace_side=None,owner=None,representation='constitutive')
    assert list(output)==['Hz','Ey']
    for component in output:
        torch.testing.assert_close(output[component],AMPLITUDES[[1,2],None]*expected(points,component),rtol=3e-15,atol=3e-15)



def test_selected_sparse_query_does_not_extend_actual_Yee_component_domain():
    result,_,_=prescribed()
    points=torch.tensor(((.51,.52,.53),),dtype=torch.float64)
    with pytest.raises(ValueError,match='outside the stored Yee component domain'):
        query_selected(result,points,components=('Ey',),frequencies=(1e9,),trace_side=None,owner=None,representation='constitutive')



def test_constitutive_query_never_gathers_an_unrequested_native_plane():
    result,points,expected=prescribed()
    result._monitors['Ey']['samples']=(0,0,0)
    # This operation requests the actual side head, not the native primitive.
    torch.testing.assert_close(sample(result,points,'Ey','positive'),AMPLITUDES[[2,0],None]*expected(points,'Ey',True),rtol=3e-15,atol=3e-15)
    with pytest.raises(ValueError,match='complete spectral horizon'):
        sample(result,points,'Ey',representation='published_native')


@pytest.mark.parametrize('record_order',(('foil','other'),('other','foil')))
@pytest.mark.parametrize('subset',(False,True))
def test_shared_face_join_keeps_requested_frequency_separate_from_each_record_index(record_order,subset):
    import witwin.maxwell as mw
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    result,_,_=prescribed()
    other=mw.Structure(mw.Box(position=(.19,0.,0.),size=torch.tensor((0.,.8,.8),dtype=torch.float64)),mw.CopperFoil(thickness=1e-6,conductivity=5.8e7,frequency_range=(1e9,2e9)),name='other')
    result.scene=result.scene.clone(structures=(*result.scene.structures,other));result._prepared_scene=prepare_scene(result.scene)
    all_faces=compile_copper_faces(result.prepared_scene,device='cpu')
    orders={'foil':(1.5e9,2e9) if subset else (2e9,1e9,1.5e9),
            'other':(1e9,1.5e9) if subset else (1.5e9,2e9,1e9)}
    records={}
    for owner in record_order:
        frequencies=orders[owner];traces=[]
        for face in [f for f in all_faces if f.name==owner]:
            value=torch.tensor([10*(1 if owner=='foil' else 2)+f/1e9+1j*(face.component+1) for f in frequencies],dtype=torch.complex128)
            field=value[:,None,None].expand(-1,2,len(face.indices)).clone()
            traces.append(dict(component='E'+'xyz'[face.component],indices=face.indices,electric_fields=field))
        records[owner]=dict(frequencies=torch.tensor(frequencies,dtype=torch.float64),spectral_valid=torch.ones(len(frequencies),dtype=torch.bool),bulk_electric_traces=traces)
    result._metadata['conductor_losses']=records
    joined=copper_query_faces(result,device='cpu',frequency_index=1)
    assert len(joined)==4
    for face in joined:
        expected=10*(1 if face['name']=='foil' else 2)+1.5+1j*(face['component']+1)
        torch.testing.assert_close(face['electric_fields'],torch.full_like(face['electric_fields'],expected),rtol=0,atol=0)


@pytest.mark.parametrize('corruption',('real','complex64','nan','inf'))
@pytest.mark.parametrize('representation',('constitutive','published_native'))
def test_strict_head_precision_and_finiteness_cannot_be_repaired_by_native_samples(corruption,representation):
    result,points,_=prescribed()
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
        if corruption=='real':trace['electric_fields']=trace['electric_fields'].real
        elif corruption=='complex64':trace['electric_fields']=trace['electric_fields'].to(torch.complex64)
        else:
            trace['electric_fields']=trace['electric_fields'].clone();trace['electric_fields'][0,0,0]=float('nan') if corruption=='nan' else float('inf')
    with pytest.raises(ValueError,match='actual finite complex128 constitutive side heads'):
        sample(result,points,'Ey',frequencies=(1e9,),representation=representation)



def test_real_instantaneous_dense_field_uses_retained_complex_native_spectrum():
    result,points,_=prescribed()
    shape=tuple(len(axis) for axis in component_coords(result.prepared_scene,'Ey'))
    result._fields['EY']=torch.full(shape,123.,dtype=torch.float32)
    actual=sample(result,points,'Ey',representation='published_native')
    oracle=independent_plane_value(result,points,'Ey');scale=independent_plane_value(result,points,'Ey',absolute_sum=True)
    assert bool(((actual-oracle).abs()<=16*torch.finfo(torch.float64).eps*scale).all())
    assert float(actual.abs().max())<10


def test_instantaneous_only_field_never_becomes_a_complex_spectral_fallback():
    result,points,_=prescribed(planes=False)
    shape=tuple(len(axis) for axis in component_coords(result.prepared_scene,'Ey'))
    result._fields['EY']=torch.full(shape,123.,dtype=torch.float32)
    with pytest.raises(ValueError,match='No retained spectral native Ey'):
        sample(result,points,'Ey',representation='published_native')



def test_selected_point_coordinate_gradient_is_explicitly_rejected_without_detaching_field_heads():
    result,points,_=prescribed(planes=False)
    trace=next(t for t in result._metadata['conductor_losses']['foil']['bulk_electric_traces'] if t['component']=='Ey')
    trace['electric_fields']=trace['electric_fields'].clone().requires_grad_()
    differentiable_points=points.clone().requires_grad_()
    with pytest.raises(NotImplementedError,match='coordinate or shape gradients'):
        sample(result,differentiable_points,'Ey','positive')
    field=sample(result,points,'Ey','positive')
    assert field.requires_grad
    gradient=torch.autograd.grad(field.abs().square().sum(),trace['electric_fields'])[0]
    assert bool((gradient[:,1].abs()>0).any()) and not bool((gradient[:,0]!=0).any())



@pytest.mark.parametrize('presentation',('collocated','normal_interpolation'))
def test_complete_native_points_remain_usable_with_unrelated_derived_plane(presentation):
    result,points,_=prescribed();payload=result._monitors['Ey'];entry=payload['components']['Ey'];axes=component_coords(result.prepared_scene,'Ey');normal_index=entry['plane_indices'][0]
    oracle=independent_plane_value(result,points,'Ey');scale=independent_plane_value(result,points,'Ey',absolute_sum=True)
    for i,y in enumerate(entry['coords'][0]):
        for j,z in enumerate(entry['coords'][1]):
            index=(normal_index,int(np.nonzero(axes[1]==float(y))[0][0]),int(np.nonzero(axes[2]==float(z))[0][0]))
            result._monitors['point_'+str(i)+'_'+str(j)]=dict(kind='point',frequencies=FREQUENCIES,samples=(16,16,16),field_indices={'Ey':index},components={'Ey':entry['data'][:,i,j]})
    if presentation=='collocated':entry['collocated']=True
    else:entry['plane_indices']=(normal_index,normal_index+1);entry['plane_weights']=(.5,.5)
    # This derived presentation has a separate clock and cannot be a native
    # sample. It must not invalidate independently retained complete Points.
    payload['samples']=(7,7,7)
    actual=sample(result,points,'Ey',representation='published_native')
    assert bool(((actual-oracle).abs()<=16*torch.finfo(torch.float64).eps*scale).all())
    for name in [name for name in result._monitors if name.startswith('point_')]:del result._monitors[name]
    with pytest.raises(ValueError,match='No retained spectral native Ey'):
        sample(result,points,'Ey',representation='published_native')


@pytest.mark.parametrize('kind',('mapped','periodic','bloch_real_projected','analytic','normalized','hanning','ramp'))
def test_selected_unsupported_scopes_are_explicit_capability_boundaries_for_generic_fields(kind):
    import witwin.maxwell as mw
    result,points,_=prescribed()
    if kind=='mapped':
        result,points,_=prescribed(normal=1)
        result.scene=result.scene.clone(coordinates=mw.BendCoordinates(curvature=.3))
    elif kind=='periodic':result.scene=result.scene.clone(boundary=mw.BoundarySpec.periodic())
    elif kind=='bloch_real_projected':result.scene=result.scene.clone(boundary=mw.BoundarySpec.bloch((1.,0.,0.)))
    elif kind=='analytic':
        result.scene=result.scene.clone(boundary=mw.BoundarySpec.bloch((1.,0.,0.)))
        result._solver_stats['time_plan']['source_representation']='analytic_cw'
    elif kind=='normalized':result._solver_stats['normalize_source']=True
    else:result._solver_stats['dft_window']=kind
    waveform=mw.GaussianPulse(1.5e9,fwidth=.1e9) if kind=='bloch_real_projected' else mw.CW(1.5e9)
    result.scene=result.scene.clone(sources=(mw.UniformCurrentSource(name='declared_drive',center=(-.5,-.5,0.),size=(.05,)*3,polarization='Ez',source_time=waveform),))
    if kind in ('hanning','ramp'):
        for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
            trace['sampling']['window']=kind
            if kind=='hanning':trace['sampling']['weight_sums']=torch.full((3,),8.,dtype=torch.float64)
    expected={'mapped':'mapped coordinates','periodic':'periodic or Bloch','bloch_real_projected':'periodic or Bloch','analytic':'analytic complex time-bank','normalized':'source normalization','hanning':'weighted-window','ramp':'weighted-window'}[kind]
    with pytest.raises(NotImplementedError,match=expected):
        query_selected(result,points,components=('Ey',),frequencies=(1e9,),trace_side=None,owner=None,representation='constitutive')


@pytest.mark.parametrize('representation',(None,'unknown','combined_real_imaginary_algorithmic_state'))
def test_missing_or_unknown_time_representation_is_invalid_retained_data(representation):
    result,points,_=prescribed();result._solver_stats['time_plan']['source_representation']=representation
    with pytest.raises(ValueError,match='explicit retained real_projected'):
        query_selected(result,points,components=('Ey',),frequencies=(1e9,),trace_side=None,owner=None,representation='constitutive')


@pytest.mark.parametrize('kind',('incident','scattered'))
def test_selected_matched_reference_fields_remain_explicit_pending_capability(kind):
    import witwin.maxwell as mw
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.result import _validate_fdtd_query_reference
    result,points,_=prescribed();empty=result.scene.clone(structures=())
    reference=mw.Result(method='fdtd',scene=empty,prepared_scene=prepare_scene(empty),frequencies=FREQUENCIES,
        monitors=copy.deepcopy(result._monitors),solver_stats=copy.deepcopy(result._solver_stats))
    _validate_fdtd_query_reference(result,reference)
    with pytest.raises(NotImplementedError,match='matched-reference frequency and sampling records'):
        result.query_fields(points,region_id='exterior',field_kind=kind,components=('Ey',),frequencies=(1e9,),reference=reference)


def test_geometry_only_face_planning_needs_no_field_frequency():
    # Pre-execution stencil planning (e.g. choosing retained point monitors) has
    # compiled faces but no recorded traces, and therefore no field frequency.
    from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil
    result,points,_=prescribed(planes=False)
    stub=SimpleNamespace(prepared_scene=result.prepared_scene,conductor_losses={})
    faces=copper_query_faces(stub,device='cpu')
    assert faces and all(face['electric_fields'] is None for face in faces)
    axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,'Hy')]
    for side in ('negative','positive'):
        assert len(interpolation_stencil(stub,points,'Hy',axes,trace_side=side))==8
