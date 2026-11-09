"""Native mapped surface observers retain frames through Graph and resume."""
import pytest
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.bend_rf_radiation_control import build_simulation
from witwin.maxwell.postprocess.stratton_chu import equivalent_surface_currents_from_monitor

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native curved observation contracts require CUDA.')


def _simulation(graph=False):
    scene=build_simulation().scene
    return mw.Simulation.fdtd(scene,frequency=1e9,excitations=mw.PortExcitation('feed'),cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=64),spectral_sampler=mw.SpectralSampler(window='none'))


def test_native_mapped_closed_surface_graph_disk_resume_and_world_current_identity(tmp_path):
    eager=_simulation().run();graph=_simulation(True).run()
    assert graph.solver._cuda_graph_active
    checkpoint=_simulation().prepare().run_until(32)
    path=tmp_path/'surface-execution.pt';checkpoint.save(path)
    loaded=mw.FDTDResumeCheckpoint.load(path,map_location='cpu')
    resumed=_simulation().prepare().run(resume_from=loaded)
    expected=equivalent_surface_currents_from_monitor(eager,'huygens')
    for actual in (graph,resumed):
        surface=actual.raw_monitor('huygens')
        assert surface['field_frame'].kind=='covariant_chart'
        assert surface['power_unit']=='W'
        native=actual._native_raw_monitor('huygens')
        torch.testing.assert_close(surface['flux'],native['flux'],rtol=0,atol=0)
        fields=equivalent_surface_currents_from_monitor(actual,'huygens')
        for key in ('points','weights','J','M'):
            torch.testing.assert_close(getattr(fields,key),getattr(expected,key),rtol=2e-5,atol=2e-7)
    saved=tmp_path/'surface-result.pt';resumed.save(saved)
    restored=mw.Result.load(saved,map_location='cuda',scene=resumed.scene)
    actual=equivalent_surface_currents_from_monitor(restored,'huygens')
    for key in ('points','weights','J','M'):
        torch.testing.assert_close(getattr(actual,key),getattr(expected,key),rtol=2e-5,atol=2e-7)


def test_mapped_world_query_requires_retained_native_stencil_samples():
    scene=build_simulation(pcb=True).scene
    result=mw.Simulation.fdtd(scene,frequency=1e9,excitations=mw.PortExcitation('feed'),
        run_time=mw.TimeConfig(time_steps=8),spectral_sampler=mw.SpectralSampler(window='none')).run()
    points=torch.tensor(((0.,.001,0.),),device='cuda',dtype=torch.float64)
    for positions,coordinates in ((points,'chart'),(scene.physical_positions(points),'world')):
        with pytest.raises(ValueError,match='No spectral|Missing .* interpolation stencil'):
            result.query_fields(positions,region_id='interior',field_kind='total',coordinates=coordinates)
