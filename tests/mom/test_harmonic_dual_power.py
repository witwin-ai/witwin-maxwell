"""Dual DC power includes both conductivities with real, unhalved work."""
from dataclasses import replace
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.quadrature import triangle_rule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_dual_power_real_flux_and_both_ohmic_channels():
    scene,config,_,_=runpy.run_path('tests/mom/test_harmonic_dual_public.py')['dual_setup']()
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    surface=sphere_mesh(radius=.2,subdivisions=0,device='cuda')
    report=result.harmonic_power_balance(surface,surface_order=2,volume_order=2,
        query_config=replace(config.query_config,max_order=192))
    volume=config.harmonic_balance.volumes[0]
    points,weights=volume.quadrature(2)
    fields=result.query_dc_dual(points.reshape(-1,3),region_id=1)
    electric=fields['electric_V_per_m'].reshape(*weights.shape,3)
    magnetic=fields['magnetic_A_per_m'].reshape(*weights.shape,3)
    e_work=(.001*electric.square().sum(-1)*weights).sum()
    h_work=(.002*magnetic.square().sum(-1)*weights).sum()
    assert e_work>0 and h_work>0
    torch.testing.assert_close(report['linear_absorption_W'][0],e_work+h_work,rtol=1e-13,atol=0)
    nonlinear=result.harmonic_state.dissipative_current[0].real
    expected=((electric*nonlinear[:,None]).sum(-1)*weights).sum()
    torch.testing.assert_close(report['nonlinear_dissipation_W'][0],expected,rtol=1e-13,atol=0)
    points,weights=triangle_rule(surface.triangles,2)
    fields=result.query_dc_dual(points.reshape(-1,3),region_id=0)
    poynting=torch.linalg.cross(fields['electric_V_per_m'],fields['magnetic_A_per_m'])
    expected=((poynting.reshape(*weights.shape,3)*surface.normals[:,None]).sum(-1)*weights).sum()
    torch.testing.assert_close(report['outward_flux_W'][0],expected,rtol=1e-13,atol=0)
    assert torch.isfinite(report['balance_residual_W']).all()
    assert not report['physical_qualification'] and not report['global_power_balance_verified']
    assert not report['quadrature_convergence_verified']
    print('dual_power_relative_residual',report['relative_balance_residual'].detach().cpu().tolist())
