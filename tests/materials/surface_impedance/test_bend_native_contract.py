"""GPU chart field identity, RF execution and passive two-trace contracts."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native mapped FDTD requires CUDA.')


def _simulation(*,curvature=3.,graph=False,origin=(0.,0.,0.),steps=96,foil=True,sigma=0.):
    port=mw.LumpedPort('feed',positive=(-.01,-.01,.005),negative=(-.01,-.01,-.005),
        voltage_path=mw.AxisPath('z'),current_surface=mw.Box(position=(-.01,-.01,-.0025),size=(.015,.015,0.)))
    structures=[]
    if foil:
        structures.append(mw.Structure(name='film',geometry=mw.Box(position=torch.tensor((0.,.005,0.),dtype=torch.float64),
            size=torch.tensor((.04,0.,.04),dtype=torch.float64)),
            material=mw.CopperFoil(thickness=2e-6,frequency_range=(.5e9,2e9))))
    if sigma:
        structures.append(mw.Structure(name='substrate',geometry=mw.Box(size=(.1,.1,.1)),material=mw.Material(eps_r=2.,sigma_e=sigma)))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),
        coordinates=mw.BendCoordinates(curvature=curvature,origin=origin) if curvature is not None else None,
        boundary=mw.BoundarySpec.none(),ports=(port,),structures=structures,device='cuda')
    return mw.Simulation.fdtd(scene,frequency=1e9,excitations=mw.PortExcitation('feed'),cuda_graph=graph,
        full_field_dft=True,run_time=mw.TimeConfig(time_steps=steps),spectral_sampler=mw.SpectralSampler(window='none'))


def test_zero_curvature_exactly_reduces_to_existing_cartesian_rf_and_foil():
    cartesian=_simulation(curvature=None).run()
    mapped=_simulation(curvature=0.).run()
    assert mapped.solver.dt==cartesian.solver.dt
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(getattr(mapped.solver,component),getattr(cartesian.solver,component),rtol=0,atol=0)
    for key in ('voltage','current','a','b'):
        torch.testing.assert_close(getattr(mapped.port('feed'),key),getattr(cartesian.port('feed'),key),rtol=0,atol=0)
    for left,right in zip(mapped.solver._copper_foils,cartesian.solver._copper_foils):
        for key in ('B','D','state','right_e','cumulative_loss'):
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def test_mapped_graph_resume_and_changed_world_frame_preflight(tmp_path):
    eager=_simulation().run()
    graph=_simulation(graph=True).run()
    assert graph.solver._cuda_graph_active
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(getattr(graph.solver,component),getattr(eager.solver,component),rtol=2e-6,atol=2e-8)
    checkpoint=_simulation().prepare().run_until(48)
    path=tmp_path/'checkpoint.pt';checkpoint.save(path)
    loaded=mw.FDTDResumeCheckpoint.load(path,map_location='cpu')
    resumed=_simulation().prepare().run(resume_from=loaded)
    for left,right in zip(resumed.solver._copper_foils,eager.solver._copper_foils):
        for key in ('state','right_e','cumulative_loss','spectral_real','spectral_imag'):
            torch.testing.assert_close(left[key],right[key],rtol=2e-6,atol=2e-10)
    changed=_simulation(origin=(.1,0.,0.)).prepare()
    before=changed.solver.Ez.clone()
    with pytest.raises(ValueError,match='does not match|layout|configuration'):
        changed.run(resume_from=loaded)
    torch.testing.assert_close(changed.solver.Ez,before,rtol=0,atol=0)
    assert eager.conductor_losses['film']['dissipated_energy']>0
    assert eager.solver.dt<_simulation(curvature=None).prepare().solver.dt


def test_nonneutral_copper_transforms_both_passive_ports_and_local_work_exactly():
    mapped=_simulation(sigma=.02).prepare().solver
    cartesian=_simulation(curvature=None,sigma=.02).prepare().solver
    generator=torch.Generator(device='cuda').manual_seed(1274)
    for entry,original in zip(mapped._copper_foils,cartesian._copper_foils):
        h=1-3*.005
        alpha=1/h if entry['electric_name']=='Ex' else h
        torch.testing.assert_close(entry['B'],original['B']*math.sqrt(alpha),rtol=1e-14,atol=0)
        torch.testing.assert_close(entry['D'],original['D']*alpha,rtol=1e-14,atol=0)
        from scipy.integrate import quad
        integral=lambda a,b:quad(lambda u:1/(1-3*u) if entry['electric_name']=='Ex' else 1-3*u,a,b,epsabs=1e-14)[0]
        widths=(entry['left_width'],entry['right_width'])
        integrals=torch.tensor((integral(.005-widths[0],.005),integral(.005,.005+widths[1])),
            device='cuda',dtype=torch.float64)[:,None]
        # The side eps/sigma come from the float32 compiled material grid. Each cell value
        # accumulates one rounding per material sub-sample, plus one for storage, so the bound
        # is (prod(samples) + 1) float32 eps: 2.4e-7 for one sample, 7.7e-6 for 4^3.
        # Measured under the 4^3 kottke_edge default: conductance 5.7e-7, capacity 9.8e-8.
        rtol=(math.prod(mapped.scene.subpixel.samples)+1)*torch.finfo(torch.float32).eps
        torch.testing.assert_close(entry['capacity'],(2*mw.constants.EPSILON_0*integrals).expand_as(entry['capacity']),rtol=rtol,atol=0)
        torch.testing.assert_close(entry['conductance'],(.02*integrals).expand_as(entry['conductance']),rtol=rtol,atol=0)
        count=entry['state'].shape[1]
        old=torch.randn((2,count),device='cuda',dtype=torch.float64,generator=generator)
        state=torch.randn(entry['state'].shape,device='cuda',dtype=torch.float64,generator=generator)*1e-7
        forcing=torch.randn((2,count),device='cuda',dtype=torch.float64,generator=generator)*.001
        history=(entry['B'].T*((1+entry['decay'])/2)[None,:])@state
        rhs=torch.einsum('mij,jm->im',entry['rhs_matrix'],old)+forcing-history
        new=torch.einsum('mij,jm->im',entry['inverse'],rhs)
        next_state=entry['decay'][:,None]*state+entry['drive']@(new+old)
        midpoint=(new+old)/2;midstate=(next_state+state)/2
        stored=lambda e,x:(.5*(entry['capacity']*e.square()).sum()+.5*x.square().sum())
        copper_loss=(entry['rates'][:,None]*midstate.square()).sum()+(midpoint*(entry['D']@midpoint)).sum()
        bulk_loss=(entry['conductance']*midpoint.square()).sum()
        initial=stored(old,state)
        residual=stored(new,next_state)-initial+mapped.dt*(copper_loss+bulk_loss-(midpoint*forcing).sum())
        assert abs(float(residual))/float(initial)<2e-12
        assert float(copper_loss)>0


def test_world_queries_transform_interpolated_covariant_fields_and_keep_frame_on_disk(tmp_path):
    chart=mw.BendCoordinates(curvature=3.,origin=(.2,-.1,.3),frame=((0.,0.,1.),(1.,0.,0.),(0.,1.,0.)))
    scene=_simulation(foil=False).scene.clone(ports=[],coordinates=chart)
    prepared=prepare_scene(scene)
    fields={}
    base=torch.tensor((1.+.2j,-.4+.8j,.3-.5j),dtype=torch.complex128,device='cuda')
    slope=torch.tensor((.7-.3j,-.2+.4j,.5+.1j),dtype=torch.complex128,device='cuda')
    for index,component in enumerate(('Ex','Ey','Ez','Hx','Hy','Hz')):
        axes=[torch.as_tensor(axis,device='cuda',dtype=torch.float64) for axis in component_coords(prepared,component)]
        q=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
        fields[component.upper()]=(base[index%3]+(q*slope).sum(-1))*(1 if index<3 else .002)
    result=mw.Result(method='fdtd',scene=scene,prepared_scene=prepared,frequency=1e9,fields=fields)
    q=torch.tensor(((.001,.002,.003),(-.007,-.004,.009)),dtype=torch.float64,device='cuda')
    covariant=base[None,:]+(q*slope).sum(-1)[:,None]
    expected=scene.coordinates.physical_field(q,covariant)
    world=scene.physical_positions(q)
    for positions,kind in ((q,'chart'),(world,'world')):
        electric,magnetic=result.query_fields(positions,region_id='exterior',field_kind='total',coordinates=kind)
        torch.testing.assert_close(electric,expected,rtol=2e-13,atol=2e-14)
        torch.testing.assert_close(magnetic,expected*.002,rtol=2e-13,atol=2e-14)
    path=tmp_path/'world.pt';result.save(path)
    restored=mw.Result.load(path,map_location='cuda',scene=scene)
    actual=restored.query_fields(world,region_id='exterior',field_kind='total')
    torch.testing.assert_close(actual[0],expected,rtol=2e-13,atol=2e-14)


def test_unqualified_mapped_batch_and_distributed_paths_fail_before_advancing():
    from witwin.maxwell.fdtd.runtime.batch import validate_native_solver
    from witwin.maxwell.fdtd.distributed.solver import DistributedFDTD
    from types import SimpleNamespace
    prepared=_simulation().prepare()
    with pytest.raises(NotImplementedError,match='coordinate-aware'):
        validate_native_solver(prepared.solver)
    with pytest.raises(NotImplementedError,match='chart ownership'):
        DistributedFDTD._validate_static_capabilities(SimpleNamespace(logical_scene=prepared.solver.scene))


def _aperture_simulation(kind='rc', *, graph=False):
    import copy
    base=_simulation(graph=graph,sigma=.02)
    feed=copy.deepcopy(base.scene.ports[0])
    object.__setattr__(feed,'voltage_aperture',mw.Box(
        position=torch.tensor((-.01,-.01,0.),dtype=torch.float64),
        size=torch.tensor((.01,.01,.01),dtype=torch.float64)))
    load=mw.LumpedPort('load',positive=(.005,-.01,.005),negative=(.005,-.01,-.005),
        voltage_path=mw.AxisPath('z'),current_surface=mw.Box(position=(.005,-.01,-.0025),size=(.015,.015,0.)),
        termination=mw.SeriesRLC(r=25.,c=1e-12) if kind=='rc' else None)
    circuits=();networks=()
    if kind=='mna':
        circuit=mw.Circuit('load_circuit');a,b=circuit.node('a'),circuit.node('b')
        circuit.add(mw.Resistor('r',a,b,25.));circuit.add(mw.Capacitor('c',b,circuit.ground,1e-12))
        circuit.bind_port('load',positive=a,negative=circuit.ground);circuits=(circuit,)
    if kind=='network':
        model=mw.StateSpaceNetwork(A=torch.tensor([[-1/(25e-12)]],dtype=torch.float64),
            B=torch.tensor([[1/(25e-12)]],dtype=torch.float64),C=torch.tensor([[-1/25]],dtype=torch.float64),
            D=torch.tensor([[1/25]],dtype=torch.float64),representation='Y',port_order=('internal',))
        f=torch.tensor((.5e9,1e9,2e9),dtype=torch.float64)
        data=mw.NetworkData.from_y(frequencies=f,y=model.evaluate(f),z0=50.,port_names=model.port_order)
        networks=(mw.NetworkBlock('load_network',network=data,connections={'internal':'load'},model=model,fit=False),)
    scene=base.scene.clone(ports=(feed,load),circuits=circuits,networks=networks)
    return mw.Simulation.fdtd(scene,frequency=1e9,excitations=mw.PortExcitation('feed'),
        cuda_graph=graph,full_field_dft=True,run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window='none'))


def test_mapped_finite_aperture_actual_field_and_source_work_close():
    from witwin.maxwell.fdtd.ports import _edge_control_volume
    from witwin.maxwell.fdtd.lumped import apply_lumped_runtime
    solver=_aperture_simulation().prepare().solver
    port=solver._port_runtimes[0];runtime=port.lumped
    runtime.diagnostics_enabled=True
    field=getattr(solver,port.field_name);indices=runtime.linear_indices
    generator=torch.Generator(device='cuda').manual_seed(477)
    old=torch.randn(indices.shape,device='cuda',dtype=field.dtype,generator=generator)
    eps=getattr(solver,'eps_'+port.field_name).flatten()[indices].double()
    sigma=getattr(solver,'sigma_e_'+port.field_name).flatten()[indices].double()
    volume=_edge_control_volume(solver,port.field_name).flatten()[indices].double()
    mass=eps*volume
    half=sigma*solver.dt/(2*eps)
    free=((1-half)/(1+half)*old).to(field.dtype)
    field.flatten()[indices]=free
    runtime.last_voltage_after.copy_(runtime.voltage_weights.dot(old))
    drive=torch.tensor(.4,device='cuda',dtype=field.dtype)
    branch=apply_lumped_runtime(runtime,field,thevenin_voltage=drive)
    updated=field.flatten()[indices].double();mid=(old.double()+updated)/2
    field_energy=.5*(mass*(updated.square()-old.double().square())).sum()
    conductive=solver.dt*(sigma*volume*mid.square()).sum()
    resistor=solver.dt*runtime.resistance.double()*branch.double().square()
    source=-solver.dt*drive.double()*branch.double()
    residual=field_energy+conductive+resistor-source
    throughput=abs(field_energy)+conductive+resistor+abs(source)
    assert abs(float(residual))/float(throughput)<2e-6
    measured=port.geometry.integrate_voltage({port.field_name:field})
    torch.testing.assert_close(runtime.last_voltage_after.double(),measured.double(),rtol=2e-6,atol=2e-9)


@pytest.mark.parametrize('kind',('rc','mna','network'))
def test_mapped_aperture_rc_mna_and_embedded_network_equivalence_graph_resume(kind):
    reference=_aperture_simulation('rc').run()
    actual=_aperture_simulation(kind,graph=True).run()
    assert actual.solver._cuda_graph_active
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(getattr(actual.solver,component),getattr(reference.solver,component),rtol=3e-4,atol=1e-6)
    for name in ('feed','load'):
        for quantity in ('voltage','current','a','b'):
            torch.testing.assert_close(getattr(actual.port(name),quantity),getattr(reference.port(name),quantity),rtol=3e-4,atol=2e-7)
    checkpoint=_aperture_simulation(kind).prepare().run_until(48)
    resumed=_aperture_simulation(kind).prepare().run(resume_from=checkpoint)
    torch.testing.assert_close(resumed.chart_fields.tensor('E_z'),reference.chart_fields.tensor('E_z'),rtol=3e-4,atol=1e-6)


def test_runtime_defensive_guard_rejects_a_forged_wrong_normal_chart():
    from witwin.maxwell.fdtd.copper_foil import initialize_copper_foils
    base=_simulation(curvature=None)
    material=base.scene.structures[0].material
    scene=base.scene.clone(structures=(mw.Structure(name='film',material=material,
        geometry=mw.Box(position=torch.tensor((.01,0.,0.),dtype=torch.float64),
                        size=torch.tensor((0.,.02,.02),dtype=torch.float64))),))
    solver=mw.Simulation.fdtd(scene,frequency=1e9,run_time=mw.TimeConfig(time_steps=1)).prepare().solver
    # Public coordinate preflight already rejects this orientation. Exercise
    # the independent runtime defense if a private caller forges the layout.
    solver.scene.coordinates=mw.BendCoordinates(curvature=3.,device='cuda')
    before=solver.Ez.clone()
    with pytest.raises(NotImplementedError,match='chart-normal u with tangential'):
        initialize_copper_foils(solver)
    torch.testing.assert_close(solver.Ez,before,rtol=0,atol=0)
