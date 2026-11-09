import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.quadrature import gauss_rule
from witwin.maxwell.mom.observables import plane_wave_cross_sections, volume_absorption
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def test_actual_lossy_solution_independent_volume_dissipation():
    scene=mw.Scene(domain=mw.Domain(bounds=((-1.,1.),)*3),grid=mw.GridSpec.uniform(.1),
                   structures=[mw.Structure(mw.Sphere(radius=.1),mw.Material(eps_r=2.25,sigma_e=.003))],sources=[mw.PlaneWave()])
    result=mw.Simulation.mom(scene,frequency=3e8,config=mw.MoMConfig(subdivisions=1)).run()
    nodes,weights=gauss_rule(8,device="cuda")
    z,phi=torch.meshgrid(2*nodes-1,torch.arange(16,device="cuda",dtype=torch.float64)*torch.pi/8,indexing="ij")
    d=torch.stack((torch.sqrt(1-z*z)*torch.cos(phi),torch.sqrt(1-z*z)*torch.sin(phi),z),-1).reshape(-1,3)
    w=(weights[:,None].expand(-1,16)*torch.pi/4).flatten()
    sections=result.surface_currents.plane_wave_cross_sections(d,w)
    assert sections["scattering_m2"] > 0
    assert sections["extinction_m2"] > sections["scattering_m2"]
    # Centroid tetrahedral volume rule is independent of the surface flux identity.
    tri=result.surface_currents.handle.basis.mesh.triangles
    points=tri.sum(1)/4
    weights=torch.abs((tri[:,0]*torch.linalg.cross(tri[:,1],tri[:,2])).sum(-1))/6
    loss=result.surface_currents.absorbed_power(points,weights,
                           query_config=FieldQueryConfig(relative_tolerance=1e-4,absolute_tolerance=1e-9,max_order=64))
    cross=loss/sections["incident_intensity_W_per_m2"]
    assert cross > 0
    # Low-order polyhedral volume sampling is explicitly only a power sign/scale test.
    assert abs(cross/sections["absorption_by_balance_m2"]-1) < .2
