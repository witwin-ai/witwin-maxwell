"""Independent piecewise smooth oracle for native copper-side reconstruction."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.result import _interpolate_fdtd_component


def fixture(normal=0, *, graded=False, quadratic=False, device='cpu', normalized=False, scale=1.):
    nodes=np.array((-.6,-.45,-.25,-.1,0.,.07,.19,.35,.6)) if graded else np.linspace(-.6,.6,13)
    nodes=nodes*scale
    size=[.8*scale]*3;size[normal]=0.
    scene=mw.Scene(domain=mw.Domain(bounds=((- .6*scale,.6*scale),)*3),
        grid=mw.GridSpec.custom(x_coords=nodes,y_coords=nodes,z_coords=nodes),device=device,
        boundary=mw.BoundarySpec.pec(), structures=(mw.Structure(
            mw.Box(size=torch.tensor(size,dtype=torch.float64)),mw.CopperFoil(thickness=1e-6,
                conductivity=5.8e7,frequency_range=(1e9,2e9)),name='foil'),))
    prepared=prepare_scene(scene)
    fields={}
    def expected(points,component,positive=None):
        own='xyz'.index(component[1].lower())
        if positive is None:positive=points[...,normal]>0
        jump=(1.+2j)*(own+1) if (component.startswith('E') or own!=normal) else 0.
        return 1.2+(.8+.2j)*points[...,0]+(.4-.1j)*points[...,1]-.7*points[...,2]+jump*positive+(points[...,normal].square() if quadratic else 0)
    factor=(.3+.7j) if normalized else 1.
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        axes=[torch.as_tensor(a,dtype=torch.float64,device=device) for a in component_coords(prepared,component)]
        q=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
        fields[component.upper()]=expected(q,component).to(torch.complex128)*factor
    traces=[]
    for face in compile_copper_faces(prepared,device=device):
        component='E'+'xyz'[face.component]
        axes=[torch.as_tensor(a,dtype=torch.float64,device=device) for a in component_coords(prepared,component)]
        q=torch.stack([axis[index] for axis,index in zip(axes,face.indices.T)],-1)
        traces.append(dict(component=component,indices=face.indices,electric_fields=torch.stack((expected(q,component,False),expected(q,component,True)))[None]))
    result=mw.Result(method='fdtd',scene=scene,prepared_scene=prepared,frequency=1e9,fields=fields,
        metadata={'conductor_losses':{'foil':{'frequencies':torch.tensor((1e9,),dtype=torch.float64),
            'spectral_valid':torch.tensor((True,)),'bulk_electric_traces':traces}}},solver_stats={'normalize_source':normalized})
    return result,expected,factor


@pytest.mark.host_validation
@pytest.mark.parametrize('normal',range(3))
@pytest.mark.parametrize('graded',(False,True))
@pytest.mark.parametrize('normalized',(False,True))
def test_piecewise_affine_actual_native_fields_all_components_sides(normal,graded,normalized):
    result,expected,factor=fixture(normal,graded=graded,normalized=normalized)
    points=torch.tensor(((.017,.027,.033),(-.019,.025,-.031)),dtype=torch.float64)
    points[:,normal]=torch.tensor((-.025,.025))
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        actual=_interpolate_fdtd_component(result,points,component)
        torch.testing.assert_close(actual,expected(points,component)*factor,rtol=2e-13,atol=2e-13)


@pytest.mark.host_validation
@pytest.mark.parametrize('normal',range(3))
def test_exact_interface_requires_declared_side_and_reconstructs_true_jump(normal):
    result,expected,factor=fixture(normal,graded=True)
    points=torch.tensor(((.017,.027,.033),),dtype=torch.float64);points[:,normal]=0
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        if component=='H'+'xyz'[normal]:
            continue
        with pytest.raises(ValueError,match='trace_side'):
            _interpolate_fdtd_component(result,points,component)
        for side,positive in (('negative',False),('positive',True)):
            actual=_interpolate_fdtd_component(result,points,component,trace_side=side)
            torch.testing.assert_close(actual,expected(points,component,positive),rtol=2e-13,atol=2e-13)


@pytest.mark.host_validation
def test_side_extrapolation_has_second_order_error_on_smooth_quadratic_oracle():
    errors=[]
    for refinement in (1,2,4):
        result,expected,_=fixture(quadratic=True,scale=1/refinement)
        points=torch.tensor(((0.,.01,.01),),dtype=torch.float64)
        value=_interpolate_fdtd_component(result,points,'Hy',trace_side='positive')
        errors.append(float((value-expected(points,'Hy',True)).abs()))
    assert errors[0]/errors[1]==pytest.approx(4.,rel=1e-10)
    assert errors[1]/errors[2]==pytest.approx(4.,rel=1e-10)


@pytest.mark.host_validation
def test_sparse_native_stencil_and_both_side_spectra_are_used_without_dense_fields():
    from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil
    result,expected,_=fixture(graded=True)
    points=torch.tensor(((.025,.027,.033),(-.025,.027,.033)),dtype=torch.float64)
    fields=result._fields.copy()
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,component)]
        indices=torch.unique(torch.cat([index for index,_,_ in interpolation_stencil(result,points,component,axes)]),dim=0)
        for offset,index in enumerate(indices):
            result._monitors[component+str(offset)]={'kind':'point','field_indices':{component:tuple(index.tolist())},
                'components':{component:fields[component.upper()][tuple(index)]}}
    result._fields.clear()
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(_interpolate_fdtd_component(result,points,component),expected(points,component))
    # Remove a true same-side sample. Cross-interface values are never a fallback.
    for key,payload in list(result._monitors.items()):
        if 'Hy' in payload['components']:
            del result._monitors[key]
    with pytest.raises(ValueError,match='No spectral Hy'):
        _interpolate_fdtd_component(result,points,'Hy')


@pytest.mark.host_validation
def test_right_trace_is_not_invented_when_snapshot_is_missing_or_schedules_disagree():
    result,_,_=fixture()
    points=torch.tensor(((.025,.027,.033),),dtype=torch.float64)
    result._metadata.clear()
    with pytest.raises(ValueError,match='two-sided electric spectra'):
        _interpolate_fdtd_component(result,points,'Ey')
    result,_,_=fixture()
    trace=result._metadata['conductor_losses']['foil']['bulk_electric_traces'][0]
    trace['electric_fields'][0,0,0]+=1.
    with pytest.raises(ValueError,match='spectral schedule'):
        _interpolate_fdtd_component(result,points,trace['component'])


@pytest.mark.host_validation
def test_finite_footprint_does_not_change_an_unrelated_interpolation_stencil():
    from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil
    result,_,_=fixture()
    points=torch.tensor(((.025,.525,.525),),dtype=torch.float64)
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,component)]
        corners=interpolation_stencil(result,points,component,axes)
        for indices,weight,overrides in corners:
            assert not overrides
            if weight.any():
                for dim,axis in enumerate(axes):
                    assert bool((abs(axis[indices[:,dim]]-points[:,dim])<=.100001).all())


@pytest.mark.host_validation
def test_live_side_spectrum_gradient_matches_independent_finite_difference():
    result,_,_=fixture(graded=True,normalized=True)
    trace=result._metadata['conductor_losses']['foil']['bulk_electric_traces'][0]
    base=trace['electric_fields'].detach().clone()
    parameter=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    points=torch.tensor(((.023,.027,.033),),dtype=torch.float64)
    def objective(value):
        trace['electric_fields']=torch.stack((base[:,0],base[:,1]*torch.exp(value)),1)
        field=_interpolate_fdtd_component(result,points,trace['component'])
        return (field.abs().square()+.3*field.real-.1*field.imag).sum()
    value=objective(parameter);value.backward()
    finite=(objective(parameter.detach()+1e-5)-objective(parameter.detach()-1e-5))/(2e-5)
    torch.testing.assert_close(parameter.grad,finite,rtol=1e-8,atol=1e-9)


@pytest.mark.host_validation
def test_normal_flux_charge_and_surface_continuity_converge_for_independent_maxwell_sheet(record_property):
    from numpy.polynomial.legendre import leggauss
    eps,mu=8.8541878128e-12,1.25663706212e-6
    omega=2*np.pi*1e9;k=80.;decay=np.sqrt(k*k-omega*omega*eps*mu)
    amplitudes=(1.+.3j,2.-.2j)
    length=.0004
    quadrature,weights=leggauss(8)
    y=torch.tensor(quadrature*length/2,dtype=torch.float64)
    weights=torch.tensor(weights*length/2,dtype=torch.float64)
    true_charge=eps*(amplitudes[1]-amplitudes[0])*2*np.sin(k*length/2)/k
    flux_errors=[];continuity_errors=[]
    for refinement in (1,2,4):
        result,_,_=fixture(scale=.01/refinement)
        def exact(q,component,positive=None):
            positive=(q[...,0]>0) if positive is None else torch.full(q.shape[:-1],positive,dtype=torch.bool)
            amplitude=torch.where(positive,torch.tensor(amplitudes[1]),torch.tensor(amplitudes[0]))
            normal=amplitude*torch.exp(-decay*q[...,0].abs()+1j*k*q[...,1])
            if component=='Ex':return normal
            if component=='Ey':return torch.where(positive,-1j,1j)*decay/k*normal
            if component=='Hz':return -omega*eps/k*normal
            return torch.zeros_like(normal)
        for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,component)]
            q=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
            result._fields[component.upper()]=exact(q,component)
        for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
            axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(result.prepared_scene,trace['component'])]
            q=torch.stack([axis[index] for axis,index in zip(axes,trace['indices'].T)],-1)
            trace['electric_fields']=torch.stack((exact(q,trace['component'],False),exact(q,trace['component'],True)))[None]
        half_height=.0001/refinement
        q=torch.zeros((len(y),3),dtype=torch.float64);q[:,1]=y
        normal_flux=0j
        for sign in (-1,1):
            q[:,0]=sign*half_height
            normal_flux+=sign*eps*(_interpolate_fdtd_component(result,q,'Ex')*weights).sum()
        for sign in (-1,1):
            for side in (-1,1):
                x=(torch.tensor(quadrature,dtype=torch.float64)+side)*half_height/2
                q[:,0]=x;q[:,1]=sign*length/2
                normal_flux+=sign*eps*(_interpolate_fdtd_component(result,q,'Ey')*torch.tensor(weights.numpy()/length*half_height)).sum()
        flux_errors.append(float(abs((normal_flux-true_charge)/true_charge)))
        q[:,0]=0.;q[:,1]=y
        current=-(_interpolate_fdtd_component(result,q,'Hz',trace_side='positive')-
                  _interpolate_fdtd_component(result,q,'Hz',trace_side='negative'))
        residual=-1j*omega*true_charge+1j*k*(current*weights).sum()
        continuity_errors.append(float(abs(residual/(omega*true_charge))))
    record_property("gauss_flux_relative_errors",repr(flux_errors))
    record_property("continuity_relative_residuals",repr(continuity_errors))
    assert flux_errors[0]>flux_errors[1]>flux_errors[2]
    assert continuity_errors[0]>continuity_errors[1]>continuity_errors[2]
    assert flux_errors[0]/flux_errors[2]>8
    assert continuity_errors[0]/continuity_errors[2]>8


def multiple_planes(planes):
    result,_,_=fixture()
    original=result.scene.structures[0]
    structures=[]
    for number,plane in enumerate(planes):
        structures.append(mw.Structure(mw.Box(position=torch.tensor((plane,0.,0.),dtype=torch.float64),
            size=torch.tensor((0.,.8,.8),dtype=torch.float64)),original.material,name='foil'+str(number)))
    scene=result.scene.clone(structures=structures);prepared=prepare_scene(scene)
    faces=compile_copper_faces(prepared,device='cpu')
    resolved={face.name:float(prepared.x_nodes64[face.normal_index]) for face in faces}
    def expected(q,component):
        value=1.+(.4+.2j)*q[...,0]+.2*q[...,1]-.3*q[...,2]
        if component!='Hx':
            for number,plane in enumerate(resolved.values()):
                value=value+(number+1)*(1.+2j)*(q[...,0]>plane)
        return value
    fields={};records={}
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(prepared,component)]
        q=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
        fields[component.upper()]=expected(q,component)
    for face in faces:
        component='E'+'xyz'[face.component]
        axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(prepared,component)]
        q=torch.stack([axis[index] for axis,index in zip(axes,face.indices.T)],-1)
        left=expected(q,component)
        jump=(int(face.name[-1])+1)*(1.+2j)
        record=records.setdefault(face.name,{'frequencies':torch.tensor((1e9,),dtype=torch.float64),'bulk_electric_traces':[]})
        record['bulk_electric_traces'].append(dict(component=component,indices=face.indices,
            electric_fields=torch.stack((left,left+jump))[None]))
    return mw.Result(method='fdtd',scene=scene,prepared_scene=prepared,frequency=1e9,fields=fields,
        metadata={'conductor_losses':records}),expected


@pytest.mark.host_validation
def test_separated_foils_each_use_their_own_side_samples():
    result,expected=multiple_planes((-.2,.2))
    points=torch.tensor(((-.225,.017,.023),(-.175,.017,.023),(.175,.017,.023),(.225,.017,.023)),dtype=torch.float64)
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(_interpolate_fdtd_component(result,points,component),expected(points,component))


@pytest.mark.host_validation
def test_a_neighboring_sheet_cannot_supply_a_claimed_same_side_stencil():
    result,_=multiple_planes((0.,.1))
    points=torch.tensor(((.025,.017,.023),),dtype=torch.float64)
    with pytest.raises(ValueError,match='neighboring copper interfaces'):
        _interpolate_fdtd_component(result,points,'Hy')


@pytest.mark.host_validation
def test_half_node_sufficiency_is_checked_only_on_the_actual_requested_side():
    result,expected=multiple_planes((-.5,))
    positive=torch.tensor(((-.475,.017,.023),),dtype=torch.float64)
    torch.testing.assert_close(_interpolate_fdtd_component(result,positive,'Hy'),expected(positive,'Hy'))
    negative=positive.clone();negative[:,0]=-.525
    with pytest.raises(ValueError,match='two retained half nodes on the requested side'):
        _interpolate_fdtd_component(result,negative,'Hy')


@pytest.mark.host_validation
def test_zero_polarization_has_zero_both_side_fields_without_a_normalization_singularity():
    result,_,_=fixture(normalized=True)
    result._fields['EY'].zero_()
    for trace in result._metadata['conductor_losses']['foil']['bulk_electric_traces']:
        if trace['component']=='Ey':trace['electric_fields'].zero_()
    points=torch.tensor(((.025,.027,.033),),dtype=torch.float64)
    torch.testing.assert_close(_interpolate_fdtd_component(result,points,'Ey'),torch.zeros(1,dtype=torch.complex128))


@pytest.mark.host_validation
def test_intersecting_copper_faces_remain_an_explicit_unsupported_topology():
    result,_=multiple_planes((0.,))
    original=result.scene.structures[0]
    cross=mw.Structure(mw.Box(position=torch.zeros(3,dtype=torch.float64),
        size=torch.tensor((.8,0.,.8),dtype=torch.float64)),original.material,name='cross')
    scene=result.scene.clone(structures=(*result.scene.structures,cross))
    result=mw.Result(method='fdtd',scene=scene,prepared_scene=prepare_scene(scene),frequency=1e9,
        fields=result._fields,metadata=result._metadata)
    records=result._metadata['conductor_losses']
    records['cross']={'frequencies':torch.tensor((1e9,),dtype=torch.float64),'bulk_electric_traces':[]}
    for face in compile_copper_faces(result.prepared_scene,device='cpu'):
        if face.name=='cross':
            records['cross']['bulk_electric_traces'].append(dict(component='E'+'xyz'[face.component],
                indices=face.indices,electric_fields=torch.zeros((1,2,len(face.indices)),dtype=torch.complex128)))
    point=torch.tensor(((.025,.025,.017),),dtype=torch.float64)
    with pytest.raises(NotImplementedError,match='ambiguous intersecting interfaces'):
        _interpolate_fdtd_component(result,point,'Hz')
