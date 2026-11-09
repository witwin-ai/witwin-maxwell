"""Explicit finite-copper modal approximation and applicability contract."""
import pytest
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.waveport_sweep import _mode_context, _mode_source


def _foil_scene(*,thickness=35e-6,model='good_conductor',conductivity=5.8e7,mode_count=None):
    scene=printed_line_scene('stripline',device='cpu')
    structures=[]
    for item in scene.structures:
        if isinstance(item.geometry,mw.PlanarPEC):
            geometry=item.geometry
            lower,upper=geometry.vertices.min(0).values,geometry.vertices.max(0).values
            position=torch.tensor((float((lower[0]+upper[0])/2),float((lower[1]+upper[1])/2),geometry.coordinate),dtype=torch.float64)
            size=torch.tensor((float(upper[0]-lower[0]),float(upper[1]-lower[1]),0.),dtype=torch.float64)
            item=mw.Structure(mw.Box(position=position,size=size),mw.CopperFoil(thickness=thickness,
                conductivity=conductivity,frequency_range=(.8e9,1.2e9),mode_count=mode_count),name=item.name)
        structures.append(item)
    ports=[]
    for port in scene.ports:
        original=port.modes[0]
        mode=mw.WaveModeSpec(original.family,polarization=original.polarization,voltage_path=original.voltage_path,current_contour=original.current_contour,conductor_model=model)
        ports.append(mw.WavePort(port.name,position=port.position,size=port.size,direction=port.direction,
            reference_plane=port.reference_plane,modes=(mode,)))
    return scene.clone(structures=structures,ports=ports)


def _solve(scene,frequency=1e9):
    context=_mode_context(scene)
    context.dt=2e-12
    port=scene.ports[0]
    compiled=context.scene.compile_waveports()[0]
    source=_compile_mode_source(_mode_source(port,compiled.modes[0],frequency=frequency),default_frequency=frequency)
    return context,solve_mode_source_profile(context,source)


def test_finite_copper_requires_explicit_modal_optin():
    with pytest.raises(NotImplementedError,match='explicit WaveModeSpec'):
        _solve(_foil_scene(model=None))


def test_good_conductor_profiles_have_passive_finite_film_correction_and_no_bulk_mutation():
    context,data=_solve(_foil_scene())
    assert context._compiled_material_model['pec_occupancy'] is None
    assert data['beta_complex'].real>data['conductor_approximation']['base_beta'].real
    assert data['beta_complex'].imag>0
    assert data['conductor_approximation']['launch']=='PEC-limit tangential Et=0'
    assert data['conductor_approximation']['relative_propagation_correction']<.01
    assert len(data['conductor_approximation']['foils'])==3
    # Tangential E on the foil plane is exactly zero, so the physical source
    # support guard remains meaningful rather than being bypassed by tolerance.
    profile=data['component_profiles']['Ey']
    z=data['coords_v']
    for coordinate in (-.005,0.,.005):
        index=int(torch.argmin(torch.abs(z-coordinate)))
        if coordinate==0.:
            y=.5*(data['coords_u'][:-1]+data['coords_u'][1:])
            selected=y.abs()<=.005
        else:
            selected=torch.ones(profile.shape[0],dtype=torch.bool)
        assert torch.count_nonzero(profile[selected,index])==0


def test_modal_surface_work_matches_actual_split_ampere_forcing_at_foil_edges():
    """Use the forward operator as an independent native surface-current oracle."""
    from types import SimpleNamespace
    from witwin.maxwell.fdtd.copper_foil import (
        advance_copper_electric, prepare_copper_magnetic_support,
    )
    from witwin.maxwell.fdtd.excitation.modes import sample_mode_source_component
    from witwin.maxwell.fdtd.excitation.modes.conductors import (
        apply_good_conductor_perturbation, prepare_good_conductor_context,
    )
    scene=_foil_scene()
    context=_mode_context(scene)
    context.dt=2e-12
    compiled=context.scene.compile_waveports()[0]
    source=_compile_mode_source(_mode_source(scene.ports[0],compiled.modes[0],frequency=1e9),default_frequency=1e9)
    proxy=prepare_good_conductor_context(context,source)
    base=solve_mode_source_profile(proxy,source)
    actual=apply_good_conductor_perturbation(proxy,source,base)
    native=SimpleNamespace(scene=context.scene,device=torch.device('cpu'),dt=context.dt)
    # Populate native staggered H samples from the unperturbed modal field.
    coordinates={axis:torch.as_tensor(getattr(context.scene,axis+'_nodes64')) for axis in 'xyz'}
    halves={axis:torch.as_tensor(getattr(context.scene,axis+'_half64')) for axis in 'xyz'}
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        dims=[]
        for axis in 'xyz':
            centered=(axis==component[-1].lower())==(component[0]=='E')
            dims.append(halves[axis] if centered else coordinates[axis])
        points=torch.stack(torch.meshgrid(*dims,indexing='ij'),dim=-1)
        field=sample_mode_source_component(base,points,component).double() if component[0]=='H' else torch.zeros(points.shape[:-1],dtype=torch.float64)
        setattr(native,component,field)
        if component[0]=='H':
            setattr(native,'c'+component.lower()+'_curl',torch.ones_like(field))
    for axis in 'xyz':
        setattr(native,'inv_d'+axis+'_h',torch.ones(len(coordinates[axis]),dtype=torch.float64))
    for (face,transverse),record in zip(proxy._good_conductor_faces,actual['conductor_approximation']['foils']):
        indices=torch.empty((len(transverse),3),dtype=torch.int64)
        indices[:,0]=base['plane_index']
        indices[:,1:]=transverse
        support=prepare_copper_magnetic_support(native,face,indices)
        count=len(indices)
        entry=dict(**support,material_ade=False,left_width=.5*context.scene.dz_primal64[face.normal_index-1],
            right_width=.5*context.scene.dz_primal64[face.normal_index],old_e=torch.zeros(2,count,dtype=torch.float64),
            right_e=torch.zeros(count,dtype=torch.float64),inverse=torch.eye(2,dtype=torch.float64).expand(count,2,2),
            rhs_matrix=torch.zeros(count,2,2,dtype=torch.float64),B=torch.empty(0,2,dtype=torch.float64),
            D=torch.zeros(2,2,dtype=torch.float64),rates=torch.empty(0,dtype=torch.float64),
            decay=torch.empty(0,dtype=torch.float64),drive=torch.empty(0,2,dtype=torch.float64),
            state=torch.empty(0,count,dtype=torch.float64),cumulative_loss=torch.zeros((),dtype=torch.float64),
            admittance_scale=torch.ones(count,dtype=torch.float64))
        native._copper_foils=[entry]
        advance_copper_electric(native)
        current=torch.stack((native.Ex[entry['index']],entry['right_e']),dim=-1).to(torch.complex128)
        impedance=face.material.surface_impedance((1e9,))[0]
        density=.5*torch.einsum('mi,ij,mj->m',current.conj(),impedance,current)
        longitudinal=context.scene.dx_primal64[base['plane_index']]
        # Free strip rims (+-5 mm on the signal; grounds run into the PML) carry
        # the forward finite-thickness edge impedance factor.
        from witwin.maxwell.compiler.copper_foil import copper_edge_impedance_factor
        tangent_y=torch.as_tensor(context.scene.y_nodes64)[transverse[:,0]]
        rim=(face.name=='signal')&(torch.isclose(tangent_y.abs(),torch.tensor(.005,dtype=torch.float64)))
        h=float(context.scene.dy_dual64[int(transverse[0,0])])
        aspect=(context.scene.dz_primal64[face.normal_index-1]+context.scene.dz_primal64[face.normal_index])/2/h
        kappa=float(copper_edge_impedance_factor(35e-6,torch.tensor([h]),torch.tensor([aspect]),torch.ones(1)))
        factor=torch.where(rim,kappa,1.).double()
        torch.testing.assert_close(torch.tensor(record['edge_impedance_factor'],dtype=torch.float64),factor,rtol=1e-6,atol=0)
        assert int(rim.sum())==(2 if face.name=='signal' else 0)
        expected=torch.sum(density*entry['area']*factor)/longitudinal
        assert record['complex_surface_work_per_m']==pytest.approx(complex(expected),rel=2e-7)
    # A bare exterior-H estimate misses native half-cell curl work at strip edges.
    assert actual['conductor_approximation']['propagation_correction'].imag>.0065


@pytest.mark.parametrize('thickness,conductivity,error',[(.1e-6,5.8e7,'thickness/skin_depth'),(35e-6,1e5,'thickness/skin_depth')])
def test_good_conductor_rejects_resistive_or_ultrathin_sheets(thickness,conductivity,error):
    with pytest.raises(ValueError,match=error):
        _solve(_foil_scene(thickness=thickness,conductivity=conductivity))


def test_good_conductor_rejects_trainable_modal_foil_material():
    with pytest.raises(NotImplementedError,match='fixed foil material'):
        _solve(_foil_scene(thickness=torch.tensor(35e-6,dtype=torch.float64,requires_grad=True),mode_count=32))


def test_good_conductor_public_api_rejects_invalid_family_and_model():
    with pytest.raises(ValueError,match='requires family'):
        mw.WaveModeSpec('te',conductor_model='good_conductor')
    with pytest.raises(ValueError,match='conductor_model must'):
        mw.WaveModeSpec('te',conductor_model='silent_pec')


@pytest.mark.parametrize('dtype',(torch.float32,torch.complex64))
def test_native_float32_profile_nodes_sample_exactly_without_foil_boundary_leak(dtype):
    from witwin.maxwell.fdtd.excitation.modes.common import _bilinear_sample
    coordinate=torch.arange(-24,25,dtype=torch.float32)*.0025
    a,b=torch.meshgrid(coordinate,coordinate,indexing='ij')
    torch.manual_seed(412)
    profile=torch.randn(a.shape).to(dtype)
    profile[:,24]=0
    actual=_bilinear_sample(profile,coordinate,coordinate,torch.stack((a,b),dim=-1))
    torch.testing.assert_close(actual,profile,rtol=0,atol=0)
    assert torch.count_nonzero(actual[:,24])==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native finite-Cu FDTD requires CUDA')
def test_good_conductor_native_launch_graph_parity_and_persistence(tmp_path):
    scene=_foil_scene().clone(device='cuda:0')
    scene=scene.clone(ports=(scene.ports[0],))
    results=[]
    for graph in (False,True):
        result=mw.Simulation.fdtd(scene,frequencies=(1e9,),excitations=mw.PortExcitation('in'),
            run_time=mw.TimeConfig.auto(transient_cycles=2,steady_cycles=1),
            spectral_sampler=mw.SpectralSampler(window='hanning'),full_field_dft=False,cuda_graph=graph).run()
        results.append(result)
        assert result.port('in').beta.imag.item()>0
        assert sum(float(value['surface_power'].sum()) for value in result.conductor_losses.values())>0
        approximation=result.stats()['waveport_excitation']['conductor_approximations']['in'][0][0]
        assert approximation['launch']=='PEC-limit tangential Et=0'
        assert approximation['relative_propagation_correction']<.01
    torch.testing.assert_close(results[0].port('in').voltage,results[1].port('in').voltage,rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(results[0].port('in').current,results[1].port('in').current,rtol=2e-5,atol=1e-7)
    path=tmp_path/'finite-cu-port.pt'
    results[-1].save(path)
    restored=mw.Result.load(path,scene=scene,map_location='cuda:0')
    assert restored.stats()['waveport_excitation']['conductor_approximations']==results[-1].stats()['waveport_excitation']['conductor_approximations']
    torch.testing.assert_close(restored.port('in').voltage,results[-1].port('in').voltage)


def _cyclic_scene(scene, permutation):
    def vector(value):
        return tuple(value[index] for index in permutation)
    def box(geometry):
        return mw.Box(position=torch.tensor(vector(geometry.position.detach().cpu().tolist()),dtype=torch.float64),
            size=torch.tensor(vector(geometry.size.detach().cpu().tolist()),dtype=torch.float64))
    structures=[mw.Structure(box(item.geometry),item.material,name=item.name) for item in scene.structures]
    ports=[]
    for port in scene.ports:
        modes=[]
        for original in port.modes:
            modes.append(mw.WaveModeSpec(original.family,polarization=('E'+'xyz'[permutation.index('xyz'.index(original.polarization[-1].lower()))] if isinstance(original.polarization,str) else vector(original.polarization)),
                voltage_path=tuple(vector(point) for point in original.voltage_path),
                current_contour=box(original.current_contour),conductor_model=original.conductor_model))
        axis=vector(port.size).index(0.)
        ports.append(mw.WavePort(port.name,position=vector(port.position),size=vector(port.size),
            direction=port.direction,reference_plane=vector(port.position)[axis],modes=modes))
    grids=(scene.grid.x_coords,scene.grid.y_coords,scene.grid.z_coords)
    return scene.clone(domain=mw.Domain(bounds=vector(scene.domain.bounds)),grid=mw.GridSpec.custom(*vector(grids)),structures=structures,ports=ports)


@pytest.mark.parametrize('permutation',[(0,1,2),(2,0,1),(1,2,0)])
@pytest.mark.parametrize('direction',['+','-'])
def test_good_conductor_mode_cyclic_rotation_and_direction_invariance(permutation,direction):
    base=_foil_scene()
    if direction=='-':
        base=base.clone(ports=tuple(reversed(base.ports)))
    _,reference=_solve(base)
    _,rotated=_solve(_cyclic_scene(base,permutation))
    assert rotated['beta_complex']==pytest.approx(reference['beta_complex'],rel=2e-7)
    for old_index,axis in enumerate('xyz'):
        new_axis='xyz'[permutation.index(old_index)]
        for family in 'EH':
            torch.testing.assert_close(rotated['component_profiles'][family+new_axis],reference['component_profiles'][family+axis],rtol=2e-6,atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native finite-Cu FDTD requires CUDA')
@pytest.mark.parametrize('direction',['+','-'])
def test_native_good_conductor_launch_cyclic_field_rotation(direction):
    scene=_foil_scene().clone(device='cuda:0')
    selected=scene.ports[0] if direction=='+' else scene.ports[1]
    scene=scene.clone(ports=(selected,))
    reference=None
    for permutation in ((0,1,2),(2,0,1),(1,2,0)):
        rotated=_cyclic_scene(scene,permutation)
        result=mw.Simulation.fdtd(rotated,frequencies=(1e9,),excitations=mw.PortExcitation(selected.name),
            run_time=mw.TimeConfig(time_steps=64),spectral_sampler=mw.SpectralSampler(window='none'),
            full_field_dft=False,cuda_graph=True).run()
        if reference is None:
            reference=result
            continue
        for old_index,axis in enumerate('xyz'):
            new_axis='xyz'[permutation.index(old_index)]
            for family in 'EH':
                actual=getattr(result.solver,family+new_axis)
                expected=getattr(reference.solver,family+axis).permute(permutation)
                scale=expected.abs().max().clamp_min(1e-7)
                assert (actual-expected).abs().max()/scale<2e-4


def test_good_conductor_rejects_frequency_outside_declared_band():
    with pytest.raises(ValueError,match='outside the declared CopperFoil band'):
        _solve(_foil_scene(),frequency=1.3e9)


def test_good_conductor_rejects_foil_end_at_aperture_stencil():
    scene=_foil_scene()
    structures=list(scene.structures)
    foil=next(index for index,item in enumerate(structures) if item.name=='signal')
    original=structures[foil]
    position=original.geometry.position.detach().clone()
    size=original.geometry.size.detach().clone()
    size[0]=.02
    position[0]=scene.ports[0].position[0]-.01
    structures[foil]=mw.Structure(mw.Box(position=position,size=size),original.material,name=original.name)
    with pytest.raises(NotImplementedError,match='foil end or axial discontinuity'):
        _solve(scene.clone(structures=structures))
