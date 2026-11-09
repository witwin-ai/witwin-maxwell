"""Actual CUDA native-plane/point/full-field observation parity."""
import pytest
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_broadband import physical_sources,port_copy
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors,regional_plane_power,regional_power

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native plane execution requires CUDA.')

@pytest.mark.parametrize('graph',[False,True])
@pytest.mark.parametrize('metal,sigma',[('pec',.01),('copper',0.),('copper',.01)])
def test_actual_native_planes_match_points_and_full_electric_loss(metal,sigma,graph,tmp_path):
    frequencies=(.8e9,1e9,1.2e9);bounds=((-0.015,.015),(-.012,.012),(-.007,.007))
    cpu=printed_line_scene('stripline',dx=.0025,longitudinal_dl=.0025,metal=metal,sigma_e=sigma,
        axial_half_length=.06,port_separation=.06,transverse_padding=.02,
        transverse_refinement_radius=.02,device='cpu')
    planes,plane_schema=regional_monitors(cpu,bounds,frequencies,observation='planes')
    points,point_schema=regional_monitors(cpu,bounds,frequencies)
    monitors=[*planes,*(monitor for monitor in points if isinstance(monitor,mw.PointMonitor))]
    cpu=cpu.clone(monitors=monitors)
    gpu=printed_line_scene('stripline',dx=.0025,longitudinal_dl=.0025,metal=metal,sigma_e=sigma,
        axial_half_length=.06,port_separation=.06,transverse_padding=.02,
        transverse_refinement_radius=.02,device='cuda')
    ports,groups=physical_sources(gpu,'stripline',.0025)
    active=groups[0]
    gpu=gpu.clone(monitors=monitors,ports=[port_copy(port,None if port.name in active else mw.SeriesRLC(r=50.))
        for port in ports])
    execution=mw.Simulation.fdtd(gpu,frequencies=frequencies,cuda_graph=graph,full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=1024),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False),
        excitations=[mw.PortExcitation(name,source_time=mw.GaussianPulse(1e9,fwidth=.8e9)) for name in active]).prepare()
    result=execution.run()
    assert execution.solver._cuda_graph_active==graph
    assert execution.solver._observer_cuda_graph_active==graph
    plane=regional_plane_power(result,cpu,plane_schema,bounds)
    point=regional_power(result,cpu,point_schema,bounds)
    full=result.power_loss('line_roi_bulk').conduction.detach().cpu()
    torch.testing.assert_close(torch.tensor(plane['bulk_conduction_w'],dtype=torch.float64),full,rtol=3e-6,atol=1e-18)
    for channel in ('bulk_conduction_w','copper_film_w','net_outward_flux_w'):
        torch.testing.assert_close(torch.as_tensor(plane[channel],dtype=torch.float64),
            torch.as_tensor(point[channel],dtype=torch.float64),rtol=3e-6,atol=1e-18)
    path=tmp_path/'native-planes.pt';result.save(path)
    loaded=mw.Result.load(path,scene=cpu,map_location='cpu')
    restored=regional_plane_power(loaded,cpu,plane_schema,bounds)
    assert restored['bulk_conduction_w']==plane['bulk_conduction_w']
    torch.testing.assert_close(restored['copper_film_w'],plane['copper_film_w'],rtol=0,atol=0)
