"""Actual GPU public spectral flux, sided Cu replay, Graph and persistence."""
import pytest
import torch
import witwin.maxwell as mw
from tests.gradients.test_fdtd_copper_foil_adjoint import FoilDesign

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA execution required')

class FluxDesign(FoilDesign):
    def __init__(self,*,graph=False,normalize=False):
        super().__init__(graph=graph)
        self.normalize=normalize
    def to_scene(self):
        scene=super().to_scene()
        scene.monitors=(mw.PlaneMonitor('cross_cu',axis='y',position=.005,
            fields=('Ex','Ez','Hx','Hz'),compute_flux=True),)
        return scene
    def simulation(self):
        sim=mw.Simulation.fdtd(self,frequency=3e9,run_time=mw.TimeConfig(time_steps=48),
            cuda_graph=self.graph,spectral_sampler=mw.SpectralSampler(window='none',normalize_source=self.normalize),
            excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9)))
        sim.config.adjoint_checkpoint_stride=self.stride
        return sim

@pytest.mark.parametrize('normalize',[False,True])
@pytest.mark.parametrize('graph',[False,True])
def test_actual_copper_product_flux_graph_disk_resume_and_load(normalize,graph,tmp_path):
    model=FluxDesign(graph=graph,normalize=normalize);model.requires_grad_(False)
    expected=model.simulation().run();payload=expected.monitor('cross_cu')
    assert payload['native_flux_heads']
    assert torch.isfinite(payload['flux']).all() and float(payload['flux'].abs().max())>1e-30
    assert payload['flux_reduction_provenance']=='native_Yee_product_then_cell_projection_v1'
    if graph:assert expected.stats()['cuda_graph_active']
    checkpoint=model.simulation().prepare().run_until(23)
    checkpoint_path=tmp_path/'checkpoint.pt';checkpoint.save(checkpoint_path)
    actual=model.simulation().prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(checkpoint_path))
    torch.testing.assert_close(actual.monitor('cross_cu')['flux'],payload['flux'],rtol=1e-6,atol=1e-25)
    path=tmp_path/'result.pt';actual.save(path)
    loaded=mw.Result.load(path,scene=actual.scene,map_location='cuda:0')
    torch.testing.assert_close(loaded.monitor('cross_cu')['flux'],payload['flux'],rtol=1e-6,atol=1e-25)
    other=FluxDesign(graph=not graph,normalize=normalize);other.requires_grad_(False)
    torch.testing.assert_close(other.simulation().run().monitor('cross_cu')['flux'],payload['flux'],rtol=1e-6,atol=1e-25)

@pytest.mark.parametrize('normalize',[False,True])
def test_actual_copper_product_flux_parameter_adjoint_matches_fd(normalize,record_property):
    model=FluxDesign(normalize=normalize)
    def objective():
        result=model.simulation().run();payload=result.monitor('cross_cu')
        assert payload['native_flux_heads']
        return payload['flux'].square().sum()
    loss=objective();assert loss.requires_grad;loss.backward()
    gradients={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.requires_grad_(False)
    torch.testing.assert_close(objective().detach(),loss.detach(),rtol=2e-4,atol=1e-30)
    for name,actual in gradients.items():
        parameter=getattr(model,name);base=parameter.detach().clone();errors=[];scan=[]
        for delta in (.08,.04,.02,.01,.005):
            with torch.no_grad():parameter.copy_(base+delta)
            plus=objective()
            with torch.no_grad():parameter.copy_(base-delta)
            minus=objective()
            with torch.no_grad():parameter.copy_(base)
            expected=(plus-minus)/(2*delta)
            errors.append(float((actual-expected).abs()/expected.abs().clamp_min(1e-30)))
            scan.append(dict(delta=delta,plus=float(plus),minus=float(minus),finite_difference=float(expected)))
        record_property(name+'_relative_errors',repr(errors))
        record_property(name+'_adjoint_gradient',float(actual))
        record_property(name+'_fd_values',repr(scan))
        assert actual.isfinite() and float(actual.abs())>1e-30
        assert sum(error<.01 for error in errors)>=2,(name,actual,errors)

class NormalEndpointFluxDesign(FluxDesign):
    def to_scene(self):
        scene=super().to_scene()
        # Declared coordinate stays .003 m (compute_flux=False avoids the
        # existing E-node snap). Both Ht samples are above the x=0 sheet.
        scene.monitors=(mw.PlaneMonitor('normal_endpoint',axis='x',position=.003,
            fields=('Ey','Ez','Hy','Hz'),compute_flux=False),)
        return scene

@pytest.mark.parametrize('normalize',[False,True])
def test_actual_off_sheet_normal_E_endpoint_adjoint_matches_fd(normalize,record_property):
    from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
    model=NormalEndpointFluxDesign(normalize=normalize)
    def objective():
        result=model.simulation().run()
        if not model.log_thickness.requires_grad:
            # Exercise the precise ordinary-report CPU / observer CUDA mix.
            for record in result._metadata['conductor_losses'].values():
                for trace in record['bulk_electric_traces']:
                    trace['electric_fields']=trace['electric_fields'].cpu()
                    assert trace['electric_fields'].device.type=='cpu'
        payload=result.monitor('normal_endpoint')
        assert all(entry['data'].device.type=='cuda' for entry in payload.get('native_components',payload['components']).values())
        assert payload['native_normal_copper_corrections']
        assert all(row['side']==1 and row['sampling_position']==.003 for row in payload['native_normal_copper_corrections'])
        density,area=plane_normal_poynting(payload)
        return (density*area).sum().square()
    loss=objective();assert loss.requires_grad;loss.backward()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    model.requires_grad_(False)
    torch.testing.assert_close(objective(),loss.detach(),rtol=2e-4,atol=1e-30)
    for name,actual in gradients.items():
        parameter=getattr(model,name);base=parameter.detach().clone();errors=[];scan=[]
        for delta in (.08,.04,.02,.01,.005):
            with torch.no_grad():parameter.copy_(base+delta)
            plus=objective()
            with torch.no_grad():parameter.copy_(base-delta)
            minus=objective()
            with torch.no_grad():parameter.copy_(base)
            expected=(plus-minus)/(2*delta)
            errors.append(float((actual-expected).abs()/expected.abs().clamp_min(1e-30)))
            scan.append(dict(delta=delta,plus=float(plus),minus=float(minus),finite_difference=float(expected)))
        record_property(name+'_relative_errors',repr(errors))
        record_property(name+'_adjoint_gradient',float(actual))
        record_property(name+'_fd_values',repr(scan))
        assert actual.isfinite() and float(actual.abs())>1e-30
        assert sum(error<.01 for error in errors)>=2,(name,actual,errors)
