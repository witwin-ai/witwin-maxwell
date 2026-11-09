"""Ordinary-forward finite differences of analytic-CW source and foil spectra."""
import cmath
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.temporal import injection_time

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA FDTD.')


class AnalyticFoil(mw.SceneModule):
    def __init__(self,graph=False,stride=7):
        super().__init__()
        self.log_eps=torch.nn.Parameter(torch.tensor(math.log(2.),dtype=torch.float64,device='cuda'))
        self.log_sigma=torch.nn.Parameter(torch.tensor(math.log(1e6),dtype=torch.float64,device='cuda'))
        self.log_thickness=torch.nn.Parameter(torch.tensor(math.log(2e-6),dtype=torch.float64,device='cuda'))
        self.graph,self.stride=graph,stride

    def to_scene(self):
        return mw.Scene(domain=mw.Domain(bounds=((- .03,.03),)*3),grid=mw.GridSpec.uniform(.005),device='cuda',
            boundary=mw.BoundarySpec.faces(default='none',x='pml',y='bloch',z='bloch',
                bloch_wavevector=(0.,5.,-7.),num_layers=6),
            structures=(mw.Structure(name='foil',geometry=mw.Box(position=(0.,0.,0.),size=(0.,.02,.02)),
                material=mw.CopperFoil(thickness=self.log_thickness.exp(),conductivity=self.log_sigma.exp(),
                    frequency_range=(2e9,4e9),mode_count=16)),
                mw.Structure(geometry=mw.Box(position=(-.01,0.,0.),size=(.015,.02,.02)),
                    material=mw.Material(eps_r=self.log_eps.exp()))),
            sources=(mw.UniformCurrentSource(center=(-.01,0.,-.0025),size=(.005,)*3,polarization='Ez',
                source_time=mw.CW(3e9,amplitude=.01,phase=.47)),),
            monitors=(mw.PointMonitor('probe',(.01,0.,0.),fields=('Ez','Hy')),))

    def objective(self,kind):
        simulation=mw.Simulation.fdtd(self,frequency=3e9,full_field_dft=True,cuda_graph=self.graph,
            run_time=mw.TimeConfig(time_steps=96),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=True))
        simulation.config.adjoint_checkpoint_stride=self.stride
        result=simulation.run()
        if kind=='field':
            return result.monitor('probe')['Ez'].abs().square().sum(),result
        return result.conductor_losses['foil']['surface_power'].sum(),result


@pytest.mark.parametrize('kind',('field','surface_power'))
@pytest.mark.parametrize('graph,stride',((False,1),(True,7),(True,13)))
def test_complex_cw_spectral_source_and_foil_vjp_matches_ordinary_forward(kind,graph,stride,record_property):
    model=AnalyticFoil(graph,stride)
    value,result=model.objective(kind)
    assert result.stats()['time_plan']['source_representation']=='analytic_cw'
    value.backward()
    gradients={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    baseline=value.detach().clone()
    model.requires_grad_(False)
    ordinary,_=model.objective(kind)
    assert not ordinary.requires_grad
    torch.testing.assert_close(ordinary.to(baseline),baseline,rtol=3e-5,atol=1e-24)
    for name,parameter in model.named_parameters():
        actual=gradients[name]
        base=parameter.detach().clone()
        errors=[]
        for step in (.08,.04,.02,.01,.005,.0025):
            parameter.copy_(base+step)
            plus,_=model.objective(kind)
            parameter.copy_(base-step)
            minus,_=model.objective(kind)
            parameter.copy_(base)
            expected=(plus.to(actual)-minus.to(actual))/(2*step)
            errors.append(float((actual-expected).abs()/expected.abs().clamp_min(1e-30)))
        record_property(name+'_ordinary_fd_relative_errors',repr(errors))
        assert actual.isfinite() and abs(float(actual))>1e-30
        assert sum(error<.01 for error in errors)>=2,(kind,name,actual,errors)


class IncidentDielectric(mw.SceneModule):
    def __init__(self,*,bloch,pulse):
        super().__init__()
        self.log_eps=torch.nn.Parameter(torch.tensor(math.log(2.),dtype=torch.float64,device='cuda'))
        self.bloch,self.pulse=bloch,pulse

    def to_scene(self):
        transverse='bloch' if self.bloch else 'periodic'
        waveform=(mw.GaussianPulse(1e9,2e9,delay=1e-9,amplitude=3.,phase=.47)
                  if self.pulse else mw.CW(1e9,amplitude=3.,phase=.47))
        bounds=((- .12,.12),(-.01,.01),(-.01,.01)) if self.bloch else ((- .12,.12),(-.04,.04),(-.04,.04))
        boundary=(mw.BoundarySpec.faces(default='none',x='pml',y=transverse,z=transverse,
                    bloch_wavevector=(0.,0.,0.),num_layers=8) if self.bloch else mw.BoundarySpec.pml(num_layers=8))
        injection=(mw.TFSF.slab(axis='x',bounds=(-.08,.08)) if self.bloch else
                   mw.TFSF(bounds=((- .08,.08),(-.025,.025),(-.025,.025))))
        return mw.Scene(domain=mw.Domain(bounds=bounds),
            grid=mw.GridSpec.uniform(.005),device='cuda',
            boundary=boundary,
            structures=(mw.Structure(geometry=mw.Box(position=(0.,0.,0.),size=(.02,.02,.02)),
                material=mw.Material(eps_r=self.log_eps.exp())),),
            sources=(mw.PlaneWave(direction=(1.,0.,0.),polarization=(0.,0.,1.),source_time=waveform,
                injection=injection),),
            monitors=(mw.PointMonitor('probe',(.025,0.,0.),fields=('Ez',)),))

    def objective(self):
        simulation=mw.Simulation.fdtd(self,frequency=1e9,cuda_graph=True,
            run_time=mw.TimeConfig(time_steps=256),spectral_sampler=mw.SpectralSampler(window='none'))
        simulation.config.adjoint_checkpoint_stride=13
        value=simulation.run().monitor('probe')['Ez'].sum()
        # A phase-sensitive linear objective cannot hide a common source sign.
        return value.real+.37*value.imag


@pytest.mark.parametrize('bloch,pulse',((False,False),(True,False),(False,True)))
def test_tfsf_incident_phase_and_enclosure_polarity_replay_match_ordinary_fd(bloch,pulse,record_property):
    model=IncidentDielectric(bloch=bloch,pulse=pulse)
    value=model.objective()
    value.backward()
    actual=model.log_eps.grad.detach().clone()
    model.requires_grad_(False)
    ordinary=model.objective()
    torch.testing.assert_close(ordinary.to(value),value.detach(),rtol=3e-5,atol=1e-12)
    base=model.log_eps.detach().clone()
    errors=[]
    for step in (.08,.04,.02,.01,.005,.0025):
        model.log_eps.copy_(base+step)
        plus=model.objective()
        model.log_eps.copy_(base-step)
        minus=model.objective()
        model.log_eps.copy_(base)
        expected=(plus-minus).to(actual)/(2*step)
        errors.append(float((actual-expected).abs()/expected.abs().clamp_min(1e-30)))
    record_property('log_eps_phase_sensitive_ordinary_fd_relative_errors',repr(errors))
    assert actual.isfinite() and abs(float(actual))>1e-30
    assert sum(error<.01 for error in errors)>=2,(actual,errors)


@pytest.mark.parametrize('graph',(False,True))
def test_analytic_cw_foil_graph_disk_resume_preserves_window_and_absolute_spectra(graph,tmp_path):
    model=AnalyticFoil(graph=graph)
    model.requires_grad_(False)
    def simulation(capture):
        return mw.Simulation.fdtd(model,frequency=3e9,full_field_dft=True,cuda_graph=capture,
            run_time=mw.TimeConfig.auto(transient_cycles=0,steady_cycles=6),
            spectral_sampler=mw.SpectralSampler(window='hanning',normalize_source=True))
    reference=simulation(False).run()
    complete=simulation(graph).run()
    if graph:
        assert complete.stats()['cuda_graph_active']
    prepared=simulation(graph).prepare()
    plan=prepared.solver._time_plan
    checkpoint=prepared.run_until((plan['spectral_start_step']+plan['total_steps'])//2)
    path=tmp_path/'analytic-copper-mid-window.pt'
    checkpoint.save(path)
    restored=simulation(graph).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    if graph:
        assert restored.stats()['cuda_graph_active']
    for result in (complete,restored):
        assert result.stats()['time_plan']['source_representation']=='analytic_cw'
        assert result.stats()['time_plan']['spectral_phasor_factor']==1
        for component in ('Ex','Ey','Ez'):
            torch.testing.assert_close(result.tensor(component),reference.tensor(component),rtol=2e-5,atol=1e-10)
        for component in ('Ez','Hy'):
            torch.testing.assert_close(torch.as_tensor(result.monitor('probe')[component]),
                torch.as_tensor(reference.monitor('probe')[component]),rtol=2e-5,atol=1e-11)
        losses=result.conductor_losses['foil']
        assert losses['energy_representation']=='combined_real_imaginary_algorithmic_state'
        torch.testing.assert_close(losses['surface_power'],reference.conductor_losses['foil']['surface_power'],rtol=2e-5,atol=1e-22)
        # The analytic CW reference is sampled at its additive-J injection instant (n+1/2)dt and
        # Fourier-labelled at the physical post-step time (n+1)dt (observer origin, beta=1), so the
        # normalised source DFT is exp(-i omega dt (1/2 - 1)) = exp(+i omega dt/2).
        label=injection_time("electric_current",0.,1.)-1.
        expected=cmath.exp(-2j*math.pi*3e9*float(result.solver.dt)*label)
        for entry in (*result.solver._dft_entries,*result.solver._observer_spectral_entries):
            assert entry['phasor_factor']==1
            source=(entry['source_dft_real']+1j*entry['source_dft_imag'])/entry['window_normalization']
            assert abs(complex(source)-expected)<1e-10
