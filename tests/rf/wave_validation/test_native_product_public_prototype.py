"""Private prototype CPU algebra only; actual solver/gradient qualification pending."""
import copy
import numpy as np
import pytest
import torch
from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
from scripts.dev.fdtd.printed_line_product_flux import native_product_flux
from tests.rf.wave_validation.test_product_first_flux_diagnostic import payload


@pytest.mark.parametrize('axis',['x','y','z'])
@pytest.mark.parametrize('pair',[0,1])
@pytest.mark.parametrize('direction',['+','-'])
@pytest.mark.parametrize('kind',['constant','one_affine','correlated'])
def test_public_shared_native_product_matches_independent_oracle(axis,pair,direction,kind):
    raw,selected,_=payload(axis,pair,direction,kind)
    public=dict(selected,components=raw['components'])
    density,area=plane_normal_poynting(public)
    expected=native_product_flux(raw,selected)['product_first']
    assert float((density*area).sum())==pytest.approx(expected,rel=2e-14,abs=3e-15)


def test_native_tensor_cotangents_match_complex_finite_difference():
    raw,selected,_=payload('x',0,'+','correlated')
    E=raw['components']['Ey']['data'].clone().requires_grad_();H=raw['components']['Hz']['data'].clone().requires_grad_()
    def value(a,b):
        components={key:dict(entry) for key,entry in raw['components'].items()}
        components['Ey']['data']=a;components['Hz']['data']=b
        d,w=plane_normal_poynting(dict(selected,components=components))
        return (d*w).sum()
    assert torch.autograd.gradcheck(value,(E,H),eps=1e-6,atol=1e-7,rtol=1e-5)


def _synthetic_copper_result(*, normalized=False):
    """Actual CPU geometry compiler, prescribed spectra; no field propagation."""
    import witwin.maxwell as mw
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    from witwin.maxwell.fdtd.coords import component_coords
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),boundary=mw.BoundarySpec.none(),device='cpu',
        structures=(mw.Structure(name='foil',geometry=mw.Box(position=torch.tensor([0.,0.,0.],dtype=torch.float64),
            size=torch.tensor([.02,.02,0.],dtype=torch.float64)),material=mw.CopperFoil(thickness=2e-6,frequency_range=(.5e9,2e9))),))
    prepared=prepare_scene(scene)
    # Use compiled float64 axes, never round-trip geometry through field dtype.
    nodes=[np.asarray(prepared.y_nodes64),np.asarray(prepared.z_nodes64)]
    centers=[(x[:-1]+x[1:])/2 for x in nodes]
    raw=dict(kind='plane',axis='x',position=0.,normal_direction='+',compute_flux=True,
        fields=('Ey','Hz','Ez','Hy'),frequency=1e9,frequencies=(1e9,),
        y=centers[0],z=centers[1],cell_widths={'y':np.diff(nodes[0]),'z':np.diff(nodes[1])},components={})
    for component in raw['fields']:
        axes=component_coords(prepared,component)
        # H is already normally interpolated onto the E plane by observer gathering.
        paired='Ey' if component in ('Ey','Hz') else 'Ez'
        transverse=component_coords(prepared,paired)[1:]
        shape=tuple(map(len,transverse));data=torch.zeros(shape,dtype=torch.complex128)
        if component=='Hz':data.fill_(2.)
        if component=='Ey':
            for j,z in enumerate(transverse[1]):data[:,j]=1. if z<=0. else 3.
        plane=int(np.argmin(abs(component_coords(prepared,paired)[0])))
        raw['components'][component]=dict(data=data[None],coords=transverse,plane_indices=(plane,),plane_index=plane,plane_weights=(1.,),collocated=False)
    faces=compile_copper_faces(prepared,device='cpu');traces=[]
    for face in faces:
        count=len(face.indices);values=torch.ones((1,2,count),dtype=torch.complex128);values[:,1]=3.
        traces.append(dict(component='E'+'xyz'[face.component],indices=face.indices,electric_fields=values))
    records={'foil':dict(frequencies=(1e9,),bulk_electric_traces=traces)}
    result=mw.Result(method='fdtd',scene=scene,frequencies=(1e9,),monitors={'surface':raw},
        metadata={'conductor_losses':records},solver_stats={'normalize_source':normalized})
    return result


def test_public_result_late_copper_uses_real_compiler_without_volume_fields(tmp_path):
    import witwin.maxwell as mw
    result=_synthetic_copper_result();payload=result.monitor('surface')
    assert payload['requires_native_flux_heads']
    # Independently replace each head's two adjacent E endpoint contributions.
    base=dict(payload);base.pop('native_flux_heads');base.pop('requires_native_flux_heads')
    d,a=plane_normal_poynting(base);expected=(d*a).sum()
    for head in payload['native_flux_heads']:
        assert head['component']=='Ey'
        for k,(iy,iz) in enumerate(head['indices'].tolist()):
            ywidth=float(payload['cell_widths']['y'][iy]);z=np.asarray(payload['components']['Ey']['coords'][1]);native=payload['components']['Ey']['data'][0,iy,iz]
            # RF factor .5, H=2, trapezoid factor .5 at either adjacent cell.
            expected=expected+.5*ywidth*((z[iz]-z[iz-1])*(head['negative'][0,k]-native).real+(z[iz+1]-z[iz])*(head['positive'][0,k]-native).real)
    torch.testing.assert_close(payload['flux'][0],expected,rtol=2e-14,atol=1e-18)
    path=tmp_path/'prescribed.pt';result.save(path)
    restored=mw.Result.load(path,scene=result.scene,map_location='cpu')
    torch.testing.assert_close(restored.monitor('surface')['flux'],payload['flux'],rtol=0,atol=0)


def test_public_copper_side_output_cotangents_pass_complex_gradcheck():
    result=_synthetic_copper_result()
    trace=next(t for t in result._metadata['conductor_losses']['foil']['bulk_electric_traces'] if t['component']=='Ey')
    values=trace['electric_fields'].clone().requires_grad_()
    def functional(v):
        trace['electric_fields']=v
        return result.monitor('surface')['flux'].sum()
    assert torch.autograd.gradcheck(functional,(values,),eps=1e-6,atol=1e-9,rtol=1e-6)
    functional(values).backward()
    # The right face contributes independently: removing its graph is a failure.
    assert bool((values.grad[:,1].abs()>0).any())


def test_normalized_legacy_copper_requires_actual_denominator_and_new_metadata_works():
    from witwin.maxwell.fdtd.observers import _normalize_monitor_result_inplace
    result=_synthetic_copper_result(normalized=True)
    with pytest.raises(NotImplementedError,match='exact applied source normalization'):
        result.monitor('surface')
    raw=result._monitors['surface'];spectrum=torch.tensor([2+3j],dtype=torch.complex128)
    _normalize_monitor_result_inplace(raw,spectrum)
    actual=result.monitor('surface')['flux']
    expected=_synthetic_copper_result().monitor('surface')['flux']/13
    torch.testing.assert_close(actual,expected,rtol=2e-14,atol=1e-18)


def test_normalization_retains_full_band_clamp_and_component_dtype():
    from witwin.maxwell.fdtd.observers import _normalize_monitor_result_inplace
    source=torch.tensor([10+2j,1e-20+0j],dtype=torch.complex128)
    entry={'data':torch.ones((2,2,3),dtype=torch.complex64)}
    payload=dict(kind='plane',frequencies=(1.,2.),components={'Ey':entry})
    _normalize_monitor_result_inplace(payload,source)
    factor=payload['source_normalization_spectra']['Ey']
    assert factor.dtype==torch.complex64
    assert factor[1].real>1e-9
    torch.testing.assert_close(entry['data'][:,0,0]*factor,torch.ones(2,dtype=torch.complex64),rtol=1e-7,atol=0)
    assert payload['source_normalization_frequencies']==(1.,2.)

@pytest.mark.parametrize('scope',['background','background_material','material_region','anisotropic','magnetic_pole'])
def test_copper_native_H_contract_rejects_nonunit_or_unqualified_permeability(scope):
    from types import SimpleNamespace
    from witwin.maxwell.fdtd.plane_product import _require_uniform_nonmagnetic
    scene=SimpleNamespace(structures=[],material_regions=[])
    material=SimpleNamespace(mu_r=2.)
    if scope in ('background','background_material'):setattr(scene,scope,material)
    elif scope=='material_region':scene.material_regions=[SimpleNamespace(mu_bounds=(1.,2.))]
    else:
        material.mu_r=torch.tensor([1.,1.,2.]) if scope=='anisotropic' else 1.
        if scope=='magnetic_pole':material.mu_lorentz_poles=(object(),)
        scene.structures=[SimpleNamespace(material=material)]
    with pytest.raises(NotImplementedError,match='permeability|nonmagnetic|dispersion'):
        _require_uniform_nonmagnetic(scene)

@pytest.mark.parametrize('one_plane_only',[False,True])
@pytest.mark.parametrize('direction',['+','-'])
def test_normal_interpolated_copper_jump_matches_independent_per_plane_oracle(one_plane_only,direction):
    result=_synthetic_copper_result();raw=result._monitors['surface'];raw['normal_direction']=direction
    from witwin.maxwell.scene import prepare_scene
    nodes=prepare_scene(result.scene).x_nodes64
    choose=(.01,.015) if one_plane_only else (0.,.005)
    indices=tuple(int(np.argmin(abs(nodes-x))) for x in choose);weights=(.3,.7)
    negative_cu=-2.+1j;positive_cu=7.-3j;exterior=4.+2j
    interpolated_left=weights[0]*negative_cu+weights[1]*(exterior if one_plane_only else negative_cu)
    interpolated_right=weights[0]*positive_cu+weights[1]*(exterior if one_plane_only else positive_cu)
    for component,entry in raw['components'].items():
        entry.update(plane_index=indices[-1],plane_indices=indices,plane_weights=weights)
        if component=='Ey':
            z=np.asarray(entry['coords'][1]);entry['data'][...,int(np.argmin(abs(z)))]=interpolated_left
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
        trace['electric_fields'][:,0]=negative_cu;trace['electric_fields'][:,1]=positive_cu
    payload=result.monitor('surface');heads=payload['native_flux_heads']
    assert len(heads)==1 and len(torch.unique(heads[0]['indices'],dim=0))==len(heads[0]['indices'])
    torch.testing.assert_close(heads[0]['negative'],torch.full_like(heads[0]['negative'],interpolated_left),rtol=2e-15,atol=1e-15)
    torch.testing.assert_close(heads[0]['positive'],torch.full_like(heads[0]['positive'],interpolated_right),rtol=2e-15,atol=1e-15)
    # RF .5, Hnormal2, half-cell product endpoint .5, side jump only.
    unsided=dict(payload);unsided.pop('native_flux_heads');unsided.pop('requires_native_flux_heads')
    d,a=plane_normal_poynting(unsided);base=(d*a).sum()
    extent=0.
    for iy,iz in heads[0]['indices'].tolist():
        z=np.asarray(raw['components']['Ey']['coords'][1]);extent+=float(raw['cell_widths']['y'][iy])*(z[iz+1]-z[iz])
    expected=base+(.5 if direction=='+' else -.5)*extent*(interpolated_right-interpolated_left).real
    torch.testing.assert_close(payload['flux'][0],expected,rtol=2e-14,atol=1e-18)


def test_normal_interpolation_both_actual_cu_jump_cotangents_are_retained():
    result=_synthetic_copper_result();raw=result._monitors['surface']
    from witwin.maxwell.scene import prepare_scene
    nodes=prepare_scene(result.scene).x_nodes64;planes=tuple(int(np.argmin(abs(nodes-x))) for x in (0.,.005))
    for entry in raw['components'].values():entry.update(plane_indices=planes,plane_weights=(.3,.7))
    trace=next(t for t in result._metadata['conductor_losses']['foil']['bulk_electric_traces'] if t['component']=='Ey')
    values=trace['electric_fields'].clone().requires_grad_()
    def functional(v):
        trace['electric_fields']=v
        return result.monitor('surface')['flux'].sum()
    assert torch.autograd.gradcheck(functional,(values,),eps=1e-6,atol=1e-9,rtol=1e-6)
    functional(values).backward()
    for plane in planes:
        chosen=trace['indices'][:,0]==plane
        assert bool((values.grad[:,:,chosen].abs()>0).any())
    # Eright−Eleft must seed opposite signed derivatives; adding powers or
    # omitting either trace destroys this actual linear interpolation identity.
    torch.testing.assert_close(values.grad[:,0],-values.grad[:,1],rtol=0,atol=0)


def test_coherent_native_components_and_two_sided_heads_keep_phase_cross_terms():
    from scripts.dev.fdtd.printed_line_loss_execution import CoherentObservations
    first=_synthetic_copper_result();second=_synthetic_copper_result()
    e_scale=2.+1j;h_scale=-.3+.7j
    for name,entry in second._monitors['surface']['components'].items():entry['data']*=e_scale if name.startswith('E') else h_scale
    for trace in second._metadata['conductor_losses']['foil']['bulk_electric_traces']:trace['electric_fields']*=e_scale
    a=first.monitor('surface');b=second.monitor('surface')
    # Retained native presentation can coexist with an already collocated one;
    # it must be combined independently rather than copied from basis column0.
    for value in (a,b):value['native_components']=copy.deepcopy(value['components'])
    weights=torch.tensor([[.4+.3j,-.2+.8j]],dtype=torch.complex128)
    combined=copy.deepcopy(a)
    CoherentObservations._components([a,b],combined,weights)
    CoherentObservations._native_products([a,b],combined,weights)
    expected=_synthetic_copper_result();ec=complex(weights[0,0])+complex(weights[0,1])*e_scale;hc=complex(weights[0,0])+complex(weights[0,1])*h_scale
    for name,entry in expected._monitors['surface']['components'].items():entry['data']*=ec if name.startswith('E') else hc
    for trace in expected._metadata['conductor_losses']['foil']['bulk_electric_traces']:trace['electric_fields']*=ec
    d,area=plane_normal_poynting(combined);flux=(d*area).sum()
    torch.testing.assert_close(flux,expected.monitor('surface')['flux'][0],rtol=1e-14,atol=1e-18)
    stale=copy.deepcopy(combined);stale['native_flux_heads']=copy.deepcopy(a['native_flux_heads'])
    d,area=plane_normal_poynting(stale)
    assert abs(float((d*area).sum()-flux))>1e-5*abs(float(flux))
    stale=copy.deepcopy(combined);stale['native_components']=copy.deepcopy(a['native_components'])
    d,area=plane_normal_poynting(stale)
    assert abs(float((d*area).sum()-flux))>1e-5*abs(float(flux))


def _normal_sheet_result(position,*,actual_position=None):
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    from witwin.maxwell.fdtd.coords import component_coords
    from witwin.maxwell.fdtd.observers import _resolve_plane_sample_positions
    result=_synthetic_copper_result();prepared=prepare_scene(result.scene)
    x,y=prepared.x_nodes64,prepared.y_nodes64;xc,yc=prepared.x_half64,prepared.y_half64
    sample=position if actual_position is None else actual_position
    raw=dict(kind='plane',axis='z',position=position,normal_direction='+',compute_flux=False,
        flux=torch.zeros(1),fields=('Ex','Ey','Hx','Hy'),frequency=1e9,frequencies=(1e9,),x=xc,y=yc,
        cell_widths={'x':np.diff(x),'y':np.diff(y)},components={})
    if actual_position is not None:raw['yee_plane_position']=actual_position
    def L(a,b):return -2+1j+(.7-.2j)*(a+2*b)/.02
    def R(a,b):return 7-3j+(-.4+.6j)*(a-b)/.02
    def exterior(a,b):return 4+2j+.8*(a+b)/.02
    Eplanes=_resolve_plane_sample_positions(prepared.z_nodes64,sample)
    weight=sum(v['weight'] for v in Eplanes if v['plane_index']==int(np.argmin(abs(prepared.z_nodes64))))
    desired={}
    for component in raw['fields']:
        axes=component_coords(prepared,component);coords=axes[:2];a,b=np.meshgrid(*coords,indexing='ij')
        value=weight*L(a,b)+(1-weight)*exterior(a,b) if component.startswith('E') else (2+.3j)+(a-2*b)/.03
        want=value.copy()
        # The explicit finite foil occupies only the central rectangle.
        if component.startswith('E') and sample>0:
            mask=(abs(a)<=.010000000000000002)&(abs(b)<=.010000000000000002)
            # Oracle support is the actual independent compiled Et indices,
            # avoiding a rectangle endpoint convention hidden in this formula.
            mask[:]=False
            for face in compile_copper_faces(prepared,device='cpu'):
                if component=='E'+'xyz'[face.component]:
                    mask[face.indices[:,0],face.indices[:,1]]=True
            want[mask]=weight*R(a,b)[mask]+(1-weight)*exterior(a,b)[mask]
        samples=_resolve_plane_sample_positions(axes[2],sample)
        raw['components'][component]=dict(data=torch.tensor(value[None]),coords=coords,collocated=False,
            plane_indices=tuple(v['plane_index'] for v in samples),plane_weights=tuple(v['weight'] for v in samples),
            plane_positions=tuple(v['plane_position'] for v in samples),plane_index=samples[-1]['plane_index'])
        averaging=1 if component in ('Ex','Hy') else 0
        from scripts.dev.fdtd.printed_line_product_flux import adjacent
        raw[component]=adjacent(torch.tensor(value[None]),averaging+1)
        desired[component]=torch.tensor(want[None])
    traces=result._metadata['conductor_losses']['foil']['bulk_electric_traces']
    for trace in traces:
        axes=component_coords(prepared,trace['component']);idx=trace['indices'].numpy();a,b=axes[0][idx[:,0]],axes[1][idx[:,1]]
        trace['electric_fields']=torch.tensor(np.stack([L(a,b),R(a,b)])[None])
    result._monitors={'normal_sheet':raw}
    return result,desired


@pytest.mark.parametrize('position',[-.003,.003])
def test_public_normal_copper_plane_chooses_actual_side_and_matches_correlated_oracle(position):
    from scripts.dev.fdtd.printed_line_product_flux import adjacent
    from witwin.maxwell.fdtd.plane_product import enrich_plane_flux
    result,desired=_normal_sheet_result(position);raw=copy.deepcopy(result._monitors['normal_sheet'])
    payload=result.monitor('normal_sheet');area=torch.tensor(payload['cell_widths']['x'])[:,None]*torch.tensor(payload['cell_widths']['y'])[None,:]
    density=.5*(adjacent((desired['Ex']*desired['Hy'].conj()).real,2)-adjacent((desired['Ey']*desired['Hx'].conj()).real,1))
    expected=(density*area).sum((-2,-1))
    torch.testing.assert_close(payload['flux'],expected,rtol=2e-14,atol=1e-18)
    for component,averaging in [('Ex',1),('Ey',0)]:
        native=payload.get('native_components',payload['components'])[component]['data']
        torch.testing.assert_close(native,desired[component],rtol=2e-14,atol=1e-14)
        torch.testing.assert_close(payload[component],adjacent(desired[component],averaging+1),rtol=2e-14,atol=1e-14)
    assert payload['native_normal_copper_corrections']
    repeated=enrich_plane_flux(result,payload)
    torch.testing.assert_close(repeated['flux'],payload['flux'],rtol=0,atol=0)
    for name,entry in raw['components'].items():
        torch.testing.assert_close(result._monitors['normal_sheet']['components'][name]['data'],entry['data'],rtol=0,atol=0)


@pytest.mark.parametrize('position,actual,message',[(0.,None,'exactly on'),(.001,None,'same exterior'),(.003,0.,'exactly on')])
def test_normal_sheet_coplanar_H_crossing_and_actual_snapped_coordinate_are_explicit(position,actual,message):
    result,_=_normal_sheet_result(position,actual_position=actual)
    with pytest.raises(NotImplementedError,match=message):result.monitor('normal_sheet')


def test_normal_sheet_actual_jump_cotangents_survive_late_presentation_projection():
    result,_=_normal_sheet_result(.003)
    traces=result._metadata['conductor_losses']['foil']['bulk_electric_traces']
    trace=next(t for t in traces if t['component']=='Ex');values=trace['electric_fields'].clone().requires_grad_()
    def functional(v):
        trace['electric_fields']=v
        return result.monitor('normal_sheet')['flux'].sum()
    assert torch.autograd.gradcheck(functional,(values,),eps=1e-6,atol=1e-9,rtol=1e-6)
    functional(values).backward()
    assert bool((values.grad[:,1].abs()>0).any())
    torch.testing.assert_close(values.grad[:,0],-values.grad[:,1],rtol=0,atol=0)


def test_legacy_numpy_copper_native_spectra_are_explicitly_outside_live_head_context():
    result=_synthetic_copper_result()
    result._monitors['surface']['components']['Ey']['data']=result._monitors['surface']['components']['Ey']['data'].numpy()
    with pytest.raises(NotImplementedError,match='native Torch observer spectra'):
        result.monitor('surface')
