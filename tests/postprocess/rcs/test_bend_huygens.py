"""Physical world-kernel qualification for curved chart Huygens export."""
import math
import numpy as np

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_monitor,SurfaceEquivalentCurrents
from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
from scripts.dev.fdtd.bend_huygens_reference import analytic_result,dipole_far

ROTATION=((0.,0.,1.),(1.,0.,0.),(0.,1.,0.))


@pytest.fixture(autouse=True)
def _small_reference_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _directions():
    directions=torch.tensor(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1.,2.,3.),(-2.,1.,3.),(2.,-3.,1.),(-3.,-2.,1.)),dtype=torch.float64)
    return directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)


@pytest.mark.parametrize('curvature',(0.,-8.,8.))
@pytest.mark.parametrize('magnetic',(False,True))
def test_world_huygens_dipole_phase_polarization_flux_and_quadrature_converge(curvature,magnetic):
    errors=[];flux_errors=[]
    for count in (8,16,32):
        result,direction,physical_flux=analytic_result(curvature=curvature,count=count,frame=ROTATION,magnetic=magnetic)
        currents=equivalent_surface_currents_from_monitor(result,'huygens')
        assert isinstance(currents,SurfaceEquivalentCurrents)
        far=NearFieldFarFieldTransformer(currents,device='cpu',c=mw.constants.C_0,
            eps0=mw.constants.EPSILON_0,mu0=mw.constants.MU_0).transform_directions(_directions(),radius=12.)
        exact=dipole_far(_directions(),frequency=1e9,position=result.scene.coordinates.origin,direction=direction,radius=12.,magnetic=magnetic)
        relative=max(float(torch.linalg.vector_norm(far[key]-value)/torch.linalg.vector_norm(value))
                     for key,value in zip(('E','H'),exact))
        errors.append(relative)
        payload=result._native_raw_monitor('huygens')
        native_flux=sum(face['flux'] for face in payload['faces'].values())
        torch.testing.assert_close(native_flux,physical_flux,rtol=3e-14,atol=1e-12)
        exact_power=mw.constants.ETA_0*(2*math.pi*1e9/mw.constants.C_0)**2/(12*math.pi)
        flux_errors.append(abs(float(native_flux)-exact_power)/exact_power)
        framed=result.monitor('huygens')
        assert framed['field_frame'].kind=='covariant_chart'
        for face in framed['faces'].values():
            assert face['coordinate_kind']=='chart'
            assert face['flux_frame'].startswith('physical integrated flux')
            assert all(component in ('E_s','E_u','E_z','H_s','H_u','H_z') for component in face['components'])
    assert errors[-1]<.002,errors
    assert flux_errors[-1]<.002,flux_errors
    assert errors[1]<.35*errors[0] and errors[2]<.35*errors[1],errors
    assert flux_errors[1]<.35*flux_errors[0] and flux_errors[2]<.35*flux_errors[1],flux_errors


def test_standalone_mapped_face_uses_physical_homogeneous_background():
    result,_,_=analytic_result(curvature=8.,count=8)
    scene=result.scene.clone(structures=(mw.Structure(geometry=mw.Box(size=(1.,1.,1.)),
        material=mw.Material(eps_r=2.4,mu_r=1.3)),))
    result=mw.Result(method='fdtd',scene=scene,frequency=1e9,monitors=result._monitors)
    face=scene.monitors[0].faces[0]
    currents=equivalent_surface_currents_from_monitor(result,face.name)
    assert currents.background_eps_r==pytest.approx(2.4,rel=3e-7)
    assert currents.background_mu_r==pytest.approx(1.3,rel=3e-7)


def test_mapped_closed_huygens_rejects_external_copper_even_without_face_intersection():
    result,_,_=analytic_result(curvature=8.,count=8)
    scene=result.scene.clone(structures=(mw.Structure(geometry=mw.Box(position=(0.,.05,0.),size=(.02,0.,.02)),
        material=mw.CopperFoil(thickness=35e-6,frequency_range=(.5e9,2e9))),))
    result=mw.Result(method='fdtd',scene=scene,frequency=1e9,monitors=result._monitors)
    with pytest.raises(ValueError,match='all copper inside.*homogeneous physical'):
        equivalent_surface_currents_from_monitor(result,'huygens')


@pytest.mark.parametrize('kind',('point','finite_u','finite_s','closed'))
def test_mapped_copper_nearby_ordinary_observations_reject_before_material_lowering(kind):
    from witwin.maxwell.scene import prepare_scene
    foil=mw.Structure(name='foil',geometry=mw.Box(size=(.04,0.,.04)),
        material=mw.CopperFoil(thickness=35e-6,frequency_range=(.5e9,2e9)))
    if kind=='point':monitor=mw.PointMonitor('near',(0.,.002,0.))
    elif kind=='finite_u':monitor=mw.FinitePlaneMonitor('near',position=(0.,.002,0.),size=(.02,0.,.02))
    elif kind=='finite_s':monitor=mw.FinitePlaneMonitor('near',position=(.005,.002,0.),size=(0.,.02,.02))
    else:monitor=mw.ClosedSurfaceMonitor.box('near',position=(0.,.002,0.),size=(.02,.008,.02))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.05,.05),)*3),grid=mw.GridSpec.uniform(.01),
        coordinates=mw.BendCoordinates(curvature=3.),structures=(foil,),monitors=(monitor,),device='cpu')
    with pytest.raises(NotImplementedError,match='Yee interpolation stencil.*one-sided'):
        prepare_scene(scene).compile_materials()
    safe=mw.ClosedSurfaceMonitor.box('safe',size=(.09,.06,.09))
    prepared=prepare_scene(scene.clone(monitors=(safe,)))
    prepared.compile_materials()
    from witwin.maxwell.compiler.coordinates import validate_coordinate_observation_bounds
    with pytest.raises(NotImplementedError,match='Yee interpolation stencil'):
        validate_coordinate_observation_bounds(prepared,points=torch.tensor(((0.,.002,0.),(.04,.04,.04))))
    validate_coordinate_observation_bounds(prepared,points=torch.tensor(((0.,-.03,0.),(0.,.03,0.))))


@pytest.mark.parametrize('component',('Ex','Ez','Hx','Hz','Ey','Hy'))
def test_mapped_copper_stencil_boundaries_use_actual_component_coordinates(component):
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.coordinates import validate_coordinate_observation_bounds
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.05,.05),)*3),grid=mw.GridSpec.uniform(.01),
        coordinates=mw.BendCoordinates(curvature=3.),device='cpu',structures=(mw.Structure(
            geometry=mw.Box(size=(.04,0.,.04)),material=mw.CopperFoil(thickness=35e-6,frequency_range=(.5e9,2e9))),))
    prepared=prepare_scene(scene)
    index=compile_copper_faces(prepared,device='cpu')[0].normal_index
    node=float(prepared.y_nodes64[index]);right=float(prepared.y_nodes64[index+1])
    left_half=float(prepared.y_half64[index-1]);right_half=float(prepared.y_half64[index])
    def check(u,reject):
        arguments=dict(points=torch.tensor(((0.,u,0.),),dtype=torch.float64),components=(component,))
        if reject:
            with pytest.raises(NotImplementedError,match='Yee interpolation stencil'):
                validate_coordinate_observation_bounds(prepared,**arguments)
        else:validate_coordinate_observation_bounds(prepared,**arguments)
    if component=='Hy':
        for u in (node,left_half,right_half,right):check(u,False)
    elif component.startswith('E') and component!='Ey':
        check(np.nextafter(node,-np.inf),False);check(node,True)
        check(np.nextafter(right,node),True);check(right,False)
    else:
        check(left_half,False);check(np.nextafter(left_half,node),True)
        check(np.nextafter(right_half,node),True);check(right_half,False)
