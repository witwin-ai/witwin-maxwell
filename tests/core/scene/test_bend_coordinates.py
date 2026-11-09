"""Public fixed chart geometry, native material placement and field-frame contracts."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.result import Result
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.coords import component_coords
from scripts.dev.fdtd.bend_coordinates_reference import ConstantCurvatureChart


BOUNDS=((-0.02,.02),(-.02,.02),(-.02,.02))
ROTATION=((0.,0.,1.),(1.,0.,0.),(0.,1.,0.))


def _scene(curvature=3., *, device='cpu', **kwargs):
    return mw.Scene(domain=mw.Domain(bounds=BOUNDS),grid=mw.GridSpec.uniform(.005),
        coordinates=mw.BendCoordinates(curvature=curvature,origin=(.2,-.1,.3),frame=ROTATION),
        device=device,**kwargs)


@pytest.mark.parametrize('curvature',(0.,-3.,3.))
def test_public_chart_matches_independent_metric_geometry_oracle(curvature):
    chart=mw.BendCoordinates(curvature=curvature,origin=(.2,-.1,.3),frame=ROTATION)
    oracle=ConstantCurvatureChart(curvature,origin=(.2,-.1,.3),frame=ROTATION)
    points=torch.tensor(((.011,.003,-.007),(-.014,-.008,.009)),dtype=torch.float64)
    vectors=torch.tensor(((1.,-2.,3.),(.5,.8,-.3)),dtype=torch.complex128)*(1+.3j)
    torch.testing.assert_close(chart.position(points),oracle.position(points),rtol=2e-14,atol=1e-16)
    torch.testing.assert_close(chart.jacobian(points),oracle.jacobian(points),rtol=2e-14,atol=1e-16)
    torch.testing.assert_close(chart.inverse(chart.position(points),bounds=BOUNDS),points,rtol=1e-13,atol=3e-17)
    covariant=chart.pullback_field(points,vectors)
    torch.testing.assert_close(chart.physical_field(points,covariant),vectors,rtol=1e-14,atol=1e-15)
    if curvature:
        shifted=points.clone();shifted[:,0]+=2.2
        bounds=((2.,2.5),(-.02,.02),(-.02,.02))
        torch.testing.assert_close(chart.inverse(chart.position(shifted),bounds=bounds),shifted,rtol=1e-13,atol=3e-17)


@pytest.mark.parametrize('kwargs',({'curvature':float('nan')},{'curvature':1.,'frame':np.ones((3,3))},
    {'curvature':1.,'frame':np.diag((-1.,1.,1.))},{'curvature':1.,'origin':(0.,0.)}))
def test_invalid_chart_geometry_rejected(kwargs):
    with pytest.raises(ValueError):mw.BendCoordinates(**kwargs)


def test_fixed_geometry_and_nonintersecting_chart_bounds():
    with pytest.raises(NotImplementedError,match='fixed chart'):
        mw.BendCoordinates(curvature=torch.tensor(1.,requires_grad=True))
    with pytest.raises(ValueError,match='focal'):_scene(50.)
    chart=mw.BendCoordinates(curvature=3.)
    with pytest.raises(ValueError,match='wrap'):chart.validate_bounds(((-2.,2.),(-.02,.02),(-.02,.02)))
    with pytest.raises(NotImplementedError,match='fixed chart'):
        mw.BendCoordinates(curvature=1.,origin=(torch.tensor(0.,requires_grad=True),0.,0.))
    before=chart.snapshot()
    chart.origin.fill_(12.)
    chart.frame.fill_(0.)
    assert chart.snapshot()==before
    with pytest.raises(AttributeError):chart.curvature=4.


def test_mapped_trainable_material_requires_explicit_fixed_time_discretization():
    material=mw.Material(eps_r=torch.tensor(2.,requires_grad=True))
    scene=_scene(structures=(mw.Structure(geometry=mw.Box(size=(.02,.02,.02)),material=material),))
    with pytest.raises(NotImplementedError,match='explicit fixed dt and time_steps'):
        mw.Simulation.fdtd(scene,frequency=1e9)


@pytest.mark.parametrize('curvature',(0.,-3.,3.))
@pytest.mark.parametrize('component',(0,2))
@pytest.mark.parametrize('side',(-1,1))
def test_split_half_metric_integral_matches_independent_quadrature(curvature,component,side):
    from scipy.integrate import quad
    from witwin.maxwell.compiler.coordinates import copper_half_metric_average
    chart=mw.BendCoordinates(curvature=curvature)
    surface,width=.007,.0025
    lower=surface-width if side<0 else surface
    integrand=lambda u:1/(1-curvature*u) if component==0 else 1-curvature*u
    expected=quad(integrand,lower,lower+width,epsabs=1e-14)[0]/width
    assert copper_half_metric_average(chart,surface,width,side,component)==pytest.approx(expected,rel=2e-14)


@pytest.mark.parametrize('curvature',(0.,-3.,3.))
def test_native_component_material_metric_uses_actual_staggered_positions(curvature):
    material=mw.Material(eps_r=2.4,mu_r=1.3,sigma_e=.02,sigma_m=.03)
    structure=mw.Structure(geometry=mw.Box(position=(0.,0.,0.),size=(1.,1.,1.)),material=material)
    scene=_scene(curvature,structures=(structure,),subpixel_samples=mw.SubpixelSpec(samples=1))
    prepared=prepare_scene(scene)
    actual=prepared.compile_materials()
    cartesian=prepare_scene(scene.clone(coordinates=None)).compile_materials()
    for key,magnetic in (('eps',False),('sigma_e',False),('mu',True),('sigma_m',True)):
        for component,values in actual['edge_components'][key].items():
            u=torch.as_tensor(component_coords(prepared,component)[1],dtype=torch.float64)
            h=1-curvature*u
            factor=(1/h if component[1].lower()=='x' else h).to(values.dtype)
            expected=cartesian['edge_components'][key][component]*factor[None,:,None]
            torch.testing.assert_close(values,expected,rtol=0,atol=0)
    assert actual['coordinate_frame']==scene.coordinates.snapshot()
    clone=scene.clone()
    assert clone.coordinates.snapshot()==scene.coordinates.snapshot()
    points=torch.tensor(((.001,.002,.003),),dtype=torch.float64)
    torch.testing.assert_close(clone.physical_positions(points),scene.coordinates.position(points))


def test_padded_cpml_bounds_are_checked_before_metric_lowering():
    scene=_scene(20.,boundary=mw.BoundarySpec.pml(num_layers=8))
    with pytest.raises(ValueError,match='padded.*focal|focal singularity'):
        prepare_scene(scene).compile_materials()


def test_mapped_result_requires_explicit_field_frame_and_matching_snapshot(tmp_path):
    scene=_scene()
    prepared=prepare_scene(scene)
    shape=tuple(len(axis) for axis in component_coords(prepared,'Ex'))
    field=torch.ones(shape,dtype=torch.complex64)
    result=Result(method='fdtd',scene=scene,prepared_scene=prepared,frequency=1e9,fields={'EX':field},
        monitors={'point':dict(kind='point',fields=('Ex',),components={'Ex':torch.tensor(1.+2j)},position=(0.,0.,0.))})
    assert result.chart_fields.components==('E_s',)
    assert result.chart_fields.frame.kind=='covariant_chart'
    torch.testing.assert_close(result.chart_fields.tensor('E_s'),field)
    torch.testing.assert_close(result.chart_fields.positions('E_s',coordinates='world'),
        scene.coordinates.position(result.chart_fields.positions('E_s')))
    assert tuple(result.raw_monitor('point')['components'])==('E_s',)
    assert result.raw_monitor('point')['coordinate_kind']=='chart'
    for access in (lambda:result.tensor('Ex'),lambda:result.fields,lambda:result.material()):
        with pytest.raises(ValueError,match='[Cc]ovariant|frame'):access()
    path=tmp_path/'mapped.pt';result.save(path)
    restored=Result.load(path,scene=scene)
    torch.testing.assert_close(restored.chart_fields.tensor('E_s'),field)
    with pytest.raises(ValueError,match='coordinate.*frame|field frame'):
        Result.load(path,scene=_scene(4.))
    with pytest.raises(ValueError,match='coordinate Scene'):
        Result.load(path,scene=scene.clone(coordinates=None))
    ordinary=Result(method='fdtd',scene=scene.clone(coordinates=None),frequency=1e9,fields={'EX':field})
    torch.testing.assert_close(ordinary.tensor('Ex'),field)


@pytest.mark.parametrize('feature',('source','dispersion','nonlinear','modal','flux','fdfd','normal_periodic'))
def test_unregistered_mapped_combinations_fail_before_advancing(feature):
    kwargs={}
    if feature=='source':kwargs['sources']=(mw.PointDipole(position=(0.,0.,0.)),)
    if feature in ('dispersion','nonlinear'):
        material=(mw.Material(debye_poles=(mw.DebyePole(delta_eps=1.,tau=1e-10),))
                  if feature=='dispersion' else mw.Material(kerr_chi3=1e-20))
        kwargs['structures']=(mw.Structure(geometry=mw.Box(size=(.01,.01,.01)),material=material),)
    if feature=='modal':kwargs['ports']=(mw.ModePort('port',size=(0.,.01,.01),polarization='Ez'),)
    if feature=='flux':kwargs['monitors']=(mw.FluxMonitor('flux'),)
    if feature=='normal_periodic':kwargs['boundary']=mw.BoundarySpec.faces(y='periodic',default='none')
    from witwin.maxwell.compiler.coordinates import validate_coordinate_scene
    with pytest.raises(NotImplementedError):validate_coordinate_scene(_scene(**kwargs),method='fdfd' if feature=='fdfd' else 'fdtd')


def test_automatic_mapped_conduction_is_physical_density_times_physical_volume():
    monitor=mw.PowerLossMonitor('heat',position=(0.,0.,0.),size=(.04,.04,.04))
    material=mw.Material(eps_r=2.,sigma_e=.02)
    scene=_scene(monitors=(monitor,),structures=(mw.Structure(geometry=mw.Box(size=(1.,1.,1.)),material=material),))
    prepared=prepare_scene(scene)
    physical=torch.tensor((1.+.2j,-.4+.8j,.3-.5j),dtype=torch.complex128)
    fields={}
    for index,component in enumerate(('Ex','Ey','Ez')):
        axes=[torch.as_tensor(axis,dtype=torch.float64) for axis in component_coords(prepared,component)]
        q=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
        # Constant components in the orthonormal physical moving frame have
        # constant |E|; their covariant s component includes h_s.
        scale=scene.coordinates.metric_scale(q) if index==0 else torch.ones(q.shape[:-1],dtype=q.dtype)
        fields[component.upper()]=physical[index]*scale
    result=Result(method='fdtd',scene=scene,prepared_scene=prepared,frequency=1e9,fields=fields)
    loss=result.power_loss('heat')
    # Existing point selection retains whole native dual cells. With boundary
    # none, the two transverse endpoint nodes each own a full dual width;
    # the represented control volume is .04*.045**2, not a clipped continuum
    # Domain integral. Test the actual density/volume contract explicitly.
    expected=.5*.02*physical.abs().square().sum()*(.04*.045**2)
    torch.testing.assert_close(loss.total,expected.reshape(1),rtol=2e-6,atol=0)
    assert loss.coordinate_frame['kind']=='covariant_chart'
    with pytest.raises(NotImplementedError,match='explicit density inputs need an explicit frame'):
        result.power_loss('heat',electric_fields={'Ex':fields['EX']})
    reconstructed=sum((loss.volume_density['conduction'][key]*volume).sum(-1)
                      for key,volume in loss.component_volumes.items())
    torch.testing.assert_close(reconstructed,loss.total,rtol=2e-6,atol=0)
    for index,component in enumerate(('Ex','Ey','Ez')):
        expected_density=.5*.02*physical[index].abs().square()
        torch.testing.assert_close(loss.volume_density['conduction'][component],
            expected_density.expand_as(loss.volume_density['conduction'][component]),rtol=2e-6,atol=0)


def test_physical_aperture_area_average_and_native_source_work_are_dual():
    from scipy.integrate import quad
    from witwin.maxwell.compiler.ports import compile_port_geometry
    axes=((-0.02,-.01,0.,.01,.02),(-.02,-.01,0.,.01,.02,.03),(-.01,-.005,0.,.005,.01))
    port=mw.LumpedPort('feed',positive=(0.,0.,.005),negative=(0.,0.,-.005),voltage_path=mw.AxisPath('z'),
        current_surface=mw.Box(position=torch.tensor((0.,.005,-.0025),dtype=torch.float64),
            size=torch.tensor((.03,.04,0.),dtype=torch.float64)),
        voltage_aperture=mw.Box(position=torch.tensor((0.,.005,0.),dtype=torch.float64),
            size=torch.tensor((.02,.03,.01),dtype=torch.float64)))
    scene=mw.Scene(domain=mw.Domain(bounds=tuple((a[0],a[-1]) for a in axes)),grid=mw.GridSpec.custom(*axes),
        coordinates=mw.BendCoordinates(curvature=3.),ports=(port,),device='cpu')
    geometry=compile_port_geometry(scene,port)
    field=torch.zeros((5,6,4),dtype=torch.float64)
    field[:]=(.3+1000*torch.tensor(axes[1],dtype=torch.float64).square())[None,:,None]
    intervals=((-0.01,-.005),(-.005,.005),(.005,.015),(.015,.02))
    values=(.4,.3,.4,.7)
    numerator=sum(value*quad(lambda u:1-3*u,*interval)[0] for value,interval in zip(values,intervals))
    denominator=quad(lambda u:1-3*u,-.01,.02)[0]
    voltage=geometry.integrate_voltage({'Ez':field})
    assert float(voltage)==pytest.approx(.01*numerator/denominator,rel=2e-14)
    selected=field[tuple(geometry.voltage_indices.T)]
    volumes=torch.linspace(.01,.1,len(selected),dtype=torch.float64)
    current=.7
    deposited=geometry.voltage_weights*current/volumes
    torch.testing.assert_close((selected*deposited*volumes).sum(),voltage*current,rtol=1e-14,atol=1e-16)
    zero=compile_port_geometry(scene.clone(coordinates=mw.BendCoordinates(curvature=0.)),port)
    ordinary=compile_port_geometry(scene.clone(coordinates=None),port)
    torch.testing.assert_close(zero.voltage_weights,ordinary.voltage_weights,rtol=0,atol=0)


@pytest.mark.parametrize('group',('thin_wires','material_regions','electrostatic_terminals','charge_densities'))
def test_mapped_preflight_rejects_unsupported_object_groups(group):
    from types import SimpleNamespace
    from witwin.maxwell.compiler.coordinates import validate_coordinate_scene
    scene=SimpleNamespace(**vars(_scene()),thin_wires=[])
    # The preflight consumes the Scene collection protocol, before lowering
    # any member; a nonempty collection must fail regardless of member data.
    setattr(scene,group,[object()])
    with pytest.raises(NotImplementedError,match='wires, density regions and electrostatic'):
        validate_coordinate_scene(scene)


@pytest.mark.parametrize('feature',('is_anisotropic','is_dispersive','is_nonlinear','is_gyromagnetic',
    'is_medium2d','is_lossy_metal','is_surface_impedance'))
def test_mapped_material_capability_preflight_checks_every_flag(feature):
    from types import SimpleNamespace
    from witwin.maxwell.compiler.coordinates import validate_coordinate_scene
    scene=_scene()
    # This is a compiler capability-protocol unit test, not a forward material
    # model. It independently exercises every predicate of the shared guard.
    scene.structures=[SimpleNamespace(enabled=True,geometry=SimpleNamespace(kind='box'),
        material=SimpleNamespace(**{feature:True}))]
    with pytest.raises(NotImplementedError,match=feature):validate_coordinate_scene(scene)


@pytest.mark.parametrize('feature',('breakdown','modulation'))
def test_mapped_dynamic_material_preflight_requires_joint_state(feature):
    from types import SimpleNamespace
    from witwin.maxwell.compiler.coordinates import validate_coordinate_scene
    scene=_scene();scene.structures=[SimpleNamespace(enabled=True,geometry=SimpleNamespace(kind='box'),
        material=SimpleNamespace(**{feature:object()}))]
    with pytest.raises(NotImplementedError,match='dynamic conductivity/modulation'):validate_coordinate_scene(scene)


@pytest.mark.parametrize('feature',('symmetry','surface','copper_normal','copper_curvature','loss_channel','sampling','cache'))
def test_remaining_mapped_preflight_boundaries_are_explicit(feature):
    from witwin.maxwell.compiler.coordinates import validate_coordinate_scene
    scene=_scene()
    if feature=='symmetry':scene=scene.clone(symmetry=('PMC',None,None))
    if feature=='surface':
        mesh=mw.SurfaceMesh(torch.tensor(((0.,0.,0.),(.01,0.,0.),(0.,.01,0.)),dtype=torch.float64),((0,1,2),))
        scene=scene.clone(structures=(mw.Structure(geometry=mesh,material=mw.Material.pec()),))
    if feature.startswith('copper_'):
        geometry=mw.Box(size=(0.,.02,.02) if feature=='copper_normal' else (.02,0.,.02))
        material=mw.CopperFoil(thickness=.001 if feature=='copper_curvature' else 2e-6,frequency_range=(.5e9,2e9))
        scene=scene.clone(structures=(mw.Structure(geometry=geometry,material=material),))
    if feature=='loss_channel':
        scene=scene.clone(monitors=(mw.PowerLossMonitor('heat',(0.,0.,0.),(.02,.02,.02),channels=('electric_dispersion',)),))
    with pytest.raises(NotImplementedError):
        if feature=='sampling':
            from witwin.maxwell.compiler.materials import compile_material_model
            compile_material_model(prepare_scene(scene),include_edges=False)
        elif feature=='cache':mw.Simulation.fdtd(scene,frequency=1e9,material_cache=mw.FDTDMaterialCache())
        else:validate_coordinate_scene(scene)
