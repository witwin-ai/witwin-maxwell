"""Independent physical quadrature for finite observation planes."""
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords

FIELDS = ('Ex','Ey','Ez','Hx','Hy','Hz')


def test_sample_spacing_contract_and_surface_propagation():
    for value in (0., -1., math.inf, math.nan):
        with pytest.raises(ValueError,match='sample_spacing'):
            mw.FinitePlaneMonitor('invalid',sample_spacing=value)
    box=mw.ClosedSurfaceMonitor.box('surface',sample_spacing=.002)
    assert all(face.sample_spacing == .002 for face in box.faces)
    assert all(face.sample_spacing is None for face in mw.ClosedSurfaceMonitor.box('native').faces)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
@pytest.mark.parametrize('axis',range(3))
@pytest.mark.parametrize('complex_fields',[False,True])
def test_sampled_quadrature_interpolates_affine_yee_fields_and_exact_area(axis,complex_fields):
    center=np.array([.0003,-.0007,.0009]);size=np.array([.012,.014,.016]);size[axis]=0
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.custom(
        np.linspace(-.02,.02,24),np.linspace(-.02,.02,28),np.linspace(-.02,.02,32)),
        boundary=mw.BoundarySpec.bloch(wavevector=(1.,2.,3.)) if complex_fields else mw.BoundarySpec.none(),device='cuda',
        monitors=[mw.FinitePlaneMonitor('face',position=center,size=size,fields=FIELDS,sample_spacing=.004)])
    solver=mw.Simulation.fdtd(scene,frequency=1e9,full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=1),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False)).prepare().solver
    solver._prepare_observers((1e9,),'none',1)
    for entry in solver._observer_spectral_entries:
        entry.update(start_step=0,end_step=1,phase_cos=1.,phase_sin=0.,h_offset_cos=1.,h_offset_sin=0.)
    for number,field in enumerate(FIELDS):
        coordinates=component_coords(solver.scene,field)
        x,y,z=torch.meshgrid(*(torch.tensor(c,device='cuda') for c in coordinates),indexing='ij')
        getattr(solver,field).copy_((number+1)*(1+2*x+3*y+4*z))
        if complex_fields:
            getattr(solver,field+'_imag').copy_(.3*getattr(solver,field))
    solver.accumulate_observers(0)
    actual=solver.get_observer_results()['face']
    tangent=[i for i in range(3) if i!=axis]
    u,v=np.meshgrid(*actual['coords'],indexing='ij')
    points=[np.full_like(u,center[i]) for i in range(3)]
    points[tangent[0]],points[tangent[1]]=u,v
    base=1+2*points[0]+3*points[1]+4*points[2]
    for number,field in enumerate(FIELDS):
        np.testing.assert_allclose(actual[field].cpu().numpy(),2*(number+1)*base*(1+.3j if complex_fields else 1),rtol=4e-7,atol=3e-6)
    widths=actual['cell_widths']
    assert np.prod([np.asarray(widths['xyz'[i]]).sum() for i in tangent]) == pytest.approx(np.prod(size[tangent]),rel=1e-14)


def simulation(graph=True,spacing=.006,amplitude=1.):
    frequency=2.45e9
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.04,.04),)*3),grid=mw.GridSpec.uniform(.004),
        boundary=mw.BoundarySpec.pml(num_layers=4),device='cuda',
        sources=[mw.PointDipole(name='drive',position=(0,0,0),polarization='Ez',source_time=mw.GaussianPulse(frequency=frequency,fwidth=frequency,amplitude=amplitude))],
        monitors=[mw.ClosedSurfaceMonitor.box('surface',size=(.048,.048,.048),sample_spacing=spacing)])
    return mw.Simulation.fdtd(scene,frequencies=(2e9,2.45e9,3e9),cuda_graph=graph,full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=512,transient_cycles=0,steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_sampled_planes_graph_resume_and_result_roundtrip(tmp_path):
    reference=simulation(False).run()
    captured=simulation(True).run()
    assert captured.solver_stats['observer_cuda_graph_active']
    checkpoint=simulation().prepare().run_until(253)
    checkpoint.save(tmp_path/'state.pt')
    resumed=simulation().prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path/'state.pt'))
    resumed.save(tmp_path/'result.pt')
    restored=mw.Result.load(tmp_path/'result.pt',scene=resumed.scene,map_location='cuda')
    for name in reference.scene.monitors[0].face_monitor_names:
        for field in reference.monitor(name)['fields']:
            expected=reference.monitor(name)[field]
            for result in (captured,resumed,restored):
                torch.testing.assert_close(torch.as_tensor(result.monitor(name)[field],device=expected.device),expected,rtol=3e-5,atol=1e-9)
    with pytest.raises(ValueError,match='configuration|fingerprint|compatible|signature'):
        simulation(spacing=.008).prepare().run(resume_from=checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_sampled_surface_native_batch_reuses_graphs_and_matches_independent():
    models=[simulation(False),simulation(False,amplitude=2.)]
    references=[model.run() for model in models]
    batch=mw.Simulation.batch(models,variation=mw.BatchVariation(sources=('drive',)),execution=mw.BatchExecution(devices=('cuda:0',),batch_size=2,
                                                                  graph='required',graph_steps=7))
    with batch.prepare() as prepared:
        first=prepared.run().results()
        graphs=tuple(prepared._graphs.values())
        repeated=prepared.run().results()
        assert graphs==tuple(prepared._graphs.values())
    for lane,reference in enumerate(references):
        for name in reference.scene.monitors[0].face_monitor_names:
            for field in reference.monitor(name)['fields']:
                for result in (first[lane],repeated[lane]):
                    torch.testing.assert_close(result.monitor(name)[field],reference.monitor(name)[field],rtol=4e-5,atol=1e-9)


class MaterialDesign(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.log_eps=torch.nn.Parameter(torch.tensor(math.log(2.2),device='cuda',dtype=torch.float64))

    def to_scene(self):
        return mw.Scene(domain=mw.Domain(bounds=((-0.04,.04),)*3),grid=mw.GridSpec.uniform(.004),
            boundary=mw.BoundarySpec.pml(num_layers=4),device='cuda',
            structures=[mw.Structure(mw.Box(size=(.016,.016,.016)),mw.Material(eps_r=self.log_eps.exp()),name='body')],
            sources=[mw.PointDipole(position=(0,0,0),polarization='Ez',width=.003,source_time=mw.CW(frequency=2.45e9))],
            monitors=[mw.FinitePlaneMonitor('probe',position=(.014,0,0),size=(0,.024,.024),
                                            fields=('Ez',),sample_spacing=.006)])


def material_objective(model,graph,stride):
    config=mw.Simulation.fdtd(model,frequency=2.45e9,cuda_graph=graph,full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=96,transient_cycles=0,steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False))
    config.config.adjoint_checkpoint_stride=stride
    return config.run().monitor('probe')['data'].abs().square().sum()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_sampled_plane_material_adjoint_matches_independent_finite_difference(record_property):
    model=MaterialDesign()
    value=material_objective(model,False,7)
    value.backward()
    gradient=float(model.log_eps.grad)
    assert abs(gradient)>1e-20
    saved=model.log_eps.detach().clone()
    errors=[]
    model.requires_grad_(False)
    with torch.no_grad():
        for step in (.004,.002,.001):
            model.log_eps.copy_(saved+step);plus=float(material_objective(model,False,7))
            model.log_eps.copy_(saved-step);minus=float(material_objective(model,False,7))
            finite=(plus-minus)/(2*step)
            errors.append(abs(finite-gradient)/abs(gradient))
        model.log_eps.copy_(saved)
    model.requires_grad_(True)
    record_property('finite_difference_relative_errors',str(errors))
    assert sum(error<.01 for error in errors)>=2
    model.log_eps.grad=None
    material_objective(model,True,13).backward()
    assert float(model.log_eps.grad)==pytest.approx(gradient,rel=2e-4,abs=1e-20)
