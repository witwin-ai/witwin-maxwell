"""Actual forward, checkpoint and live-adjoint queries on both foil sides."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native copper queries require CUDA.')


def retain_h_stencils(scene, points, *, extra_components=()):
    prepared=prepare_scene(scene)
    stub=SimpleNamespace(prepared_scene=prepared,conductor_losses={})
    monitors=list(scene.monitors)
    for component in ('Hx','Hy','Hz',*extra_components):
        axes=[torch.as_tensor(a,dtype=torch.float64) for a in component_coords(prepared,component)]
        corners=interpolation_stencil(stub,points.cpu(),component,axes,trace_side='positive')
        corners+=interpolation_stencil(stub,points.cpu(),component,axes,trace_side='negative')
        needed=torch.unique(torch.cat([indices[weight!=0] for indices,weight,_ in corners]),dim=0)
        for number,index in enumerate(needed):
            position=tuple(float(axes[axis][index[axis]]) for axis in range(3))
            monitors.append(mw.PointMonitor('query_'+component+str(number),position,fields=(component,)))
    return scene.clone(monitors=monitors)


def query_points(normal=0,plane=0.):
    points=torch.tensor(((.0011,.0013,.0017),)*3,dtype=torch.float64)
    points[:,normal]=torch.tensor((plane-.001,plane+.001,plane),dtype=torch.float64)
    return points


@pytest.mark.parametrize('normal',range(3))
def test_actual_cartesian_query_graph_and_result_snapshot(normal,tmp_path):
    from tests.gradients.test_fdtd_copper_foil_adjoint import FoilDesign
    model=FoilDesign(normal_axis=normal);model.requires_grad_(False)
    points=query_points(normal).cuda()
    scene=retain_h_stencils(model.to_scene(),points)
    def simulation(graph):
        return mw.Simulation.fdtd(scene,frequency=3e9,cuda_graph=graph,full_field_dft=True,
            run_time=mw.TimeConfig(time_steps=96,transient_cycles=0,steady_cycles=1),spectral_sampler=mw.SpectralSampler(window='none'),
            excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9,phase=.37)))
    eager=simulation(False).run();graph=simulation(True).run()
    expected=eager.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive')
    actual=graph.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive')
    assert max(float(value.abs().max()) for value in expected)>0
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-10)
    minus=eager.query_fields(points[-1:],region_id='exterior',field_kind='total',trace_side='negative')
    assert float((minus[0]-expected[0][-1:]).abs().max())>1e-10
    path=tmp_path/'query.pt';graph.save(path)
    restored=mw.Result.load(path,scene=scene,map_location='cuda')
    for a,b in zip(restored.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive'),actual):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    if normal==0:
        checkpoint=simulation(False).prepare().run_until(43)
        checkpoint_path=tmp_path/'checkpoint.pt';checkpoint.save(checkpoint_path)
        resumed=simulation(False).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(checkpoint_path,map_location='cpu'))
        for a,b in zip(resumed.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive'),expected):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-10)


def test_mapped_query_reconstructs_chart_before_world_pullback_and_persistence(tmp_path):
    from tests.materials.surface_impedance.test_bend_native_contract import _simulation
    base=_simulation(graph=True,origin=(.2,-.1,.3))
    points=query_points(1,.005).cuda()
    scene=retain_h_stencils(base.scene,points)
    result=mw.Simulation.fdtd(scene,frequency=1e9,full_field_dft=True,cuda_graph=True,
        excitations=mw.PortExcitation('feed'),run_time=mw.TimeConfig(time_steps=96,transient_cycles=0,steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none')).run()
    from witwin.maxwell.result import _interpolate_fdtd_component
    covariant=torch.stack([_interpolate_fdtd_component(result,points,name,trace_side='positive')
                          for name in ('Ex','Ey','Ez','Hx','Hy','Hz')],-1)
    expected=tuple(scene.coordinates.physical_field(points,part) for part in (covariant[:,:3],covariant[:,3:]))
    world=scene.physical_positions(points)
    for positions,coordinates in ((points,'chart'),(world,'world')):
        actual=result.query_fields(positions,region_id='exterior',field_kind='total',coordinates=coordinates,trace_side='positive')
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-10,atol=1e-12)
    with pytest.raises(ValueError,match='trace_side'):
        result.query_fields(world[-1:],region_id='exterior',field_kind='total')
    path=tmp_path/'mapped.pt';result.save(path)
    restored=mw.Result.load(path,scene=scene,map_location='cuda')
    for a,b in zip(restored.query_fields(world,region_id='exterior',field_kind='total',trace_side='positive'),expected):
        torch.testing.assert_close(a,b,rtol=2e-10,atol=1e-12)


@pytest.mark.parametrize('normal',range(3))
def test_actual_query_objective_foil_gradients_against_independent_ordinary_forward(normal,record_property):
    from tests.gradients.test_fdtd_copper_foil_adjoint import FoilDesign
    points=query_points(normal).cuda()
    class QueryDesign(FoilDesign):
        def to_scene(self):
            scene=super().to_scene()
            def rotate(values):return tuple(values[(axis-normal)%3] for axis in range(3))
            scene=scene.clone(ports=(),sources=(mw.UniformCurrentSource(name='drive',
                center=rotate((-.015,0.,-.0025)),size=(.005,)*3,polarization='E'+'xyz'[(normal+2)%3],
                source_time=mw.CW(3e9,phase=.43,amplitude=1e-5)),))
            return retain_h_stencils(scene,points)
        def simulation(self):
            simulation=mw.Simulation.fdtd(self,frequency=3e9,full_field_dft=True,cuda_graph=self.graph,
                run_time=mw.TimeConfig(time_steps=512 if normal==0 else 96,dt=5e-12 if normal==0 else None,transient_cycles=0,steady_cycles=1),
                spectral_sampler=mw.SpectralSampler(window='hanning' if normal==0 else 'none',normalize_source=True))
            simulation.config.adjoint_checkpoint_stride=self.stride
            return simulation
    model=QueryDesign(normal_axis=normal,graph=True,stride=7)
    def objective():
        result=model.simulation().run()
        electric,magnetic=result.query_fields(points,region_id='exterior',field_kind='total',trace_side='positive')
        return electric.abs().square().sum()+1e5*magnetic.abs().square().sum()
    value=objective();value.backward()
    gradients={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.requires_grad_(False)
    base=objective();torch.testing.assert_close(base,value.detach(),rtol=3e-4,atol=1e-25)
    for name,parameter in model.named_parameters():
        initial=parameter.detach().clone();errors=[]
        for step in (.08,.04,.02,.01):
            with torch.no_grad():
                parameter.copy_(initial+step);plus=objective()
                parameter.copy_(initial-step);minus=objective()
                parameter.copy_(initial)
            finite=(plus-minus)/(2*step)
            errors.append(float(abs((finite-gradients[name])/finite)))
        record_property(name+'_relative_errors',repr(errors))
        assert abs(float(gradients[name]))>0
        assert sum(error<.01 for error in errors)>=2,(name,gradients[name],errors)



def test_oblique_copper_charge_continuity_converges_with_actual_native_grid(record_property):
    from scripts.dev.fdtd.copper_query_charge_control import run_oblique_charge_control
    run_oblique_charge_control(record_property, fixed_wavevector=True, compare_point_clock=True)
