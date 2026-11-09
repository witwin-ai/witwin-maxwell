"""Broadband physics uses real finite apertures and balanced grounded launches."""
import numpy as np
import pytest
import torch
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_broadband import physical_sources,port_copy
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('kind',['microstrip','stripline','cpw','differential_pair'])
@pytest.mark.parametrize('dx',[.00125,.000625,.0003125,.00015625])
def test_real_broadband_source_apertures_compile_on_all_refinements(kind,dx):
    scene=printed_line_scene(kind,dx=dx,longitudinal_dl=.0025,transverse_padding=.02,
        axial_half_length=.3,port_separation=.4,transverse_refinement_radius=.02,device='cpu')
    ports,groups=physical_sources(scene,kind,.0025)
    assert len(groups)==(4 if kind=='differential_pair' else 2)
    assert all(len(group)==(2 if kind in ('stripline','cpw') else 1) for group in groups)
    prepared=prepare_scene(scene.clone(ports=ports))
    axes=[getattr(prepared,a+'_half64') for a in 'xyz']
    for port in prepared.ports:
        compiled=compile_port_geometry(prepared,port,device='cpu')
        assert len(compiled.voltage_weights)>0
        assert np.isfinite(compiled.voltage_weights.numpy()).all()
        normal='xyz'.index(port.voltage_path.axis)
        assert np.min(abs(np.asarray(axes[normal])-float(port.current_surface.position[normal])))<1e-14
        # The physical feed has fixed axial width even when local refinement changes.
        assert float(port.voltage_aperture.size[0])==pytest.approx(.005)
        copied=port_copy(port,None)
        torch.testing.assert_close(copied.voltage_aperture.size,port.voltage_aperture.size,rtol=0,atol=0)
    if kind in ('stripline','cpw'):
        for group in groups:
            a,b=[next(p for p in prepared.ports if p.name==name) for name in group]
            normal='xyz'.index(a.voltage_path.axis)
            va=np.asarray(a.positive)-np.asarray(a.negative)
            vb=np.asarray(b.positive)-np.asarray(b.negative)
            np.testing.assert_allclose(va,-vb,rtol=0,atol=0)
            assert float(a.voltage_aperture.position[normal]) == -float(b.voltage_aperture.position[normal])
