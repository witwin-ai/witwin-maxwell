"""Native gap-port spectra do not require an unrequested volume field DFT."""
import pytest
import torch
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native FDTD requires CUDA')


def scene_for(binding):
    if binding=='named_planar':
        structures=[mw.Structure(mw.PlanarPEC.rectangle(((-.003,.003),(-.003,.003)),
            coordinate=z),mw.Material.pec(),name=name) for name,z in (('signal',.002),('ground',-.002))]
        port=mw.TerminalPort('feed',mw.TerminalRef('signal'),mw.TerminalRef('ground'),
            mw.AxisPath('z'),.0005,
            current_surface=mw.Box(position=(0.,0.,.0005),size=(.005,.005,0.)),
            voltage_aperture=mw.Box(position=(0.,0.,0.),size=(.002,.002,.004)))
        return mw.Scene(domain=mw.Domain(bounds=((- .008,.008),)*3),
            grid=mw.GridSpec.uniform(.001),boundary=mw.BoundarySpec.none(),
            structures=structures,ports=(port,),device='cuda:0')
    if binding=='wire_nodes':
        wires=[mw.ThinWire(name,points,.002,mw.WireConductor.pec(),
            endpoints=(mw.WireEnd.open(),mw.WireEnd.open())) for name,points in (
                ('negative',((-.08,-.08,0.),(-.04,-.04,0.))),
                ('positive',((.04,.04,0.),(.08,.08,0.))))]
        port=mw.TerminalPort('feed',wire_binding=mw.WirePortBinding.nodes(
            negative=mw.WireNodeRef('negative',1),positive=mw.WireNodeRef('positive',0)))
    else:
        wires=[mw.ThinWire('dipole',((-.08,-.08,0.),(-.04,-.04,0.),(.04,.04,0.),(.08,.08,0.)),
            .002,mw.WireConductor.pec(),endpoints=(mw.WireEnd.open(),mw.WireEnd.open()))]
        port=mw.LumpedPort('feed',wire_binding=mw.WirePortBinding.gap(
            negative=mw.WireNodeRef('dipole',1),positive=mw.WireNodeRef('dipole',2)))
    return mw.Scene(domain=mw.Domain(bounds=((- .12,.12),)*3),grid=mw.GridSpec.uniform(.04),
        boundary=mw.BoundarySpec.none(),thin_wires=wires,ports=(port,),device='cuda:0')


@pytest.mark.parametrize('binding',['named_planar','wire_nodes','wire_gap'])
@pytest.mark.parametrize('graph',[False,True])
def test_two_frequency_native_port_and_final_fields_equal_with_optional_volume_dft(binding,graph):
    scene=scene_for(binding);results=[]
    for full in (False,True):
        result=mw.Simulation.fdtd(scene,frequencies=(1e9,1.2e9),
            excitations=mw.PortExcitation('feed',source_time=mw.CW(1e9)),
            run_time=mw.TimeConfig(time_steps=256),
            spectral_sampler=mw.SpectralSampler(window='none'),full_field_dft=full,cuda_graph=graph).run()
        assert result.stats()['full_field_dft'] is full
        assert bool(result._fields) is full
        assert torch.count_nonzero(result.port('feed').voltage)>0
        assert len(result.port('feed').voltage)==2
        results.append(result)
    a,b=results
    for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        torch.testing.assert_close(getattr(a.solver,component),getattr(b.solver,component),rtol=0,atol=0)
    for quantity in ('voltage','current','reflection_coefficient'):
        torch.testing.assert_close(getattr(a.port('feed'),quantity),getattr(b.port('feed'),quantity),rtol=0,atol=0)
