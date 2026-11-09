"""Public Cartesian no-Cu material-loss gradients versus ordinary FDTD.

This same file is intended to run unchanged on baseline and candidate sources.
It contains no mapped-adjoint override or private capability hook.
"""
import math
import pytest
import torch
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA field adjoint control.')


class CartesianLossDesign(mw.SceneModule):
    def __init__(self,sigma=.15):
        super().__init__()
        self.log_sigma=torch.nn.Parameter(torch.tensor(math.log(sigma),device='cuda',dtype=torch.float64))

    def to_scene(self):
        return mw.Scene(domain=mw.Domain(bounds=((- .03,.03),)*3),grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.pml(num_layers=4),device='cuda',
            structures=(mw.Structure(name='substrate',geometry=mw.Box(
                position=torch.tensor((0.,.005,0.),dtype=torch.float64),size=torch.tensor((.04,.02,.04),dtype=torch.float64)),
                material=mw.Material(eps_r=2.,sigma_e=self.log_sigma.exp())),),
            ports=(mw.LumpedPort('feed',positive=(-.015,-.005,.005),negative=(-.015,-.005,-.005),
                voltage_path=mw.AxisPath('z'),current_surface=mw.Box(
                    position=torch.tensor((-.015,-.005,-.0025),dtype=torch.float64),size=torch.tensor((.015,.015,0.),dtype=torch.float64)),
                voltage_aperture=mw.Box(position=torch.tensor((-.015,-.005,0.),dtype=torch.float64),
                    size=torch.tensor((.01,.01,.01),dtype=torch.float64))),),
            subpixel_samples=mw.SubpixelSpec(samples=(2,4,2),averaging='arithmetic'),
            monitors=(mw.PowerLossMonitor('bulk',position=(0.,.005,0.),size=(.035,.025,.035)),))

    def simulation(self,*,graph=False,stride=7):
        simulation=mw.Simulation.fdtd(self,frequency=3e9,full_field_dft=True,cuda_graph=graph,
            excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9,phase=.37)),
            run_time=mw.TimeConfig(time_steps=96,dt=5e-12,transient_cycles=0,steady_cycles=1),
            spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False))
        simulation.config.adjoint_checkpoint_stride=stride
        return simulation


def loss(result):return result.power_loss('bulk').total.sum()


def measured_run(model,record_property,label,**options):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    result=model.simulation(**options).run()
    torch.cuda.synchronize()
    record_property(label+'_peak_allocated_bytes',torch.cuda.max_memory_allocated())
    record_property(label+'_peak_reserved_bytes',torch.cuda.max_memory_reserved())
    return result


@pytest.mark.parametrize('sigma',(.015,.15))
def test_direct_bulk_sigma_gradient_matches_detached_ordinary_forward(sigma,record_property):
    model=CartesianLossDesign(sigma)
    result=measured_run(model,record_property,'adjoint_eager')
    value=loss(result);value.backward();gradient=model.log_sigma.grad.detach().clone()
    model.requires_grad_(False);base=model.log_sigma.detach().clone()
    ordinary=loss(model.simulation().run())
    torch.testing.assert_close(ordinary.to(value.device),value.detach(),rtol=3e-4,atol=1e-25)
    errors=[];differences=[]
    for step in (.08,.04,.02,.01):
        with torch.no_grad():
            model.log_sigma.copy_(base+step);plus=loss(model.simulation().run())
            model.log_sigma.copy_(base-step);minus=loss(model.simulation().run())
            model.log_sigma.copy_(base)
        finite=((plus-minus)/(2*step)).to(gradient.device)
        differences.append(float(finite));errors.append(float(abs((gradient-finite)/finite)))
    record_property('autograd_log_sigma',float(gradient))
    record_property('ordinary_forward_fd_steps',repr((.08,.04,.02,.01)))
    record_property('ordinary_forward_fd_values',repr(differences))
    record_property('ordinary_forward_relative_errors',repr(errors))
    assert gradient.isfinite()and abs(float(gradient))>1e-25
    assert sum(error<.01 for error in errors)>=2,(sigma,gradient,differences,errors)


def test_direct_sigma_heads_survive_graph_checkpoint_and_disk_resume(tmp_path,record_property):
    model=CartesianLossDesign();reference=loss(measured_run(model,record_property,'eager'))
    reference.backward();gradient=model.log_sigma.grad.detach().clone()
    for stride in (7,13):
        model.zero_grad(set_to_none=True)
        result=measured_run(model,record_property,'graph_stride_'+str(stride),graph=True,stride=stride)
        captured=loss(result);captured.backward()
        assert result.solver._cuda_graph_active
        torch.testing.assert_close(captured,reference,rtol=3e-4,atol=1e-25)
        torch.testing.assert_close(model.log_sigma.grad,gradient,rtol=.001,atol=1e-25)
    model.requires_grad_(False);expected=model.simulation().run()
    checkpoint=model.simulation().prepare().run_until(47);path=tmp_path/'cartesian-loss.pt';checkpoint.save(path)
    resumed=model.simulation(graph=True).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path,map_location='cpu'))
    torch.testing.assert_close(loss(resumed),loss(expected),rtol=3e-4,atol=1e-25)


def test_direct_sigma_heads_survive_native_batch_cache_reuse_and_close():
    models=[CartesianLossDesign(.15),CartesianLossDesign(.15*math.exp(.2))]
    simulations=[model.simulation()for model in models]
    parameters=tuple(model.log_sigma for model in models)
    batch=mw.Simulation.batch(simulations,outputs=mw.BatchOutput(fields=('Ex','Ey','Ez')),
        execution=mw.BatchExecution(devices=('cuda:0',),batch_size=2,graph='required',graph_steps=7))
    with batch.prepare()as prepared:
        first=prepared.run().results();second=prepared.run().results()
    def evaluate(results):return sum(loss(result)for result in results)
    expected_value=evaluate([simulation.run()for simulation in simulations])
    expected=torch.autograd.grad(expected_value,parameters)
    for results in (first,second):
        value=evaluate(results)
        torch.testing.assert_close(value,expected_value,rtol=3e-4,atol=1e-25)
        actual=torch.autograd.grad(value,parameters)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=.002,atol=1e-25)
        assert all(result.stats()['batch_adjoint']=='native_checkpoint_per_lane'for result in results)
