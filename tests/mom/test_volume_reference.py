"""Independent physical volume quadrature checks, without a surface solve."""
import math

import pytest
import torch
import witwin.maxwell as mw
from benchmark.sie_reference import sphere_cross_sections
from benchmark.mom_volume_reference import radial_face_volume_rule, mie_volume
from witwin.maxwell.mom.physics import compile_medium


def test_independent_mie_volume_integral_refines_and_matches_absorption():
    medium=compile_medium(mw.Material(eps_r=2.25,sigma_e=.004),3e8)
    exterior=compile_medium(mw.Material(),3e8)
    a=mie_volume(.1,exterior.k.real,medium.epsilon_r,exterior.eta.real,.004,6,6)
    b=mie_volume(.1,exterior.k.real,medium.epsilon_r,exterior.eta.real,.004,10,10)
    expected=sphere_cross_sections(.1,exterior.k.real,medium.epsilon_r)['absorption']/(2*exterior.eta.real)
    assert b['volume_m3']==pytest.approx(4*math.pi*.1**3/3,rel=1e-13)
    assert a['independent_absorption_W']==pytest.approx(b['independent_absorption_W'],rel=1e-8)
    assert b['independent_absorption_W']==pytest.approx(expected,rel=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical quadrature')
def test_positive_radial_face_rule_integrates_tetrahedral_quadratic_moments():
    from witwin.maxwell.mom.mesh import SurfaceMesh
    vertices=torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],device='cuda',dtype=torch.float64)
    faces=torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
    mesh=SurfaceMesh(vertices,faces)
    for order in (3,4):
        points,weights=radial_face_volume_rule(mesh,order)
        assert bool((weights>0).all())
        assert float(weights.sum())==pytest.approx(1/6,rel=1e-12)
        assert float((weights*points[:,0]).sum())==pytest.approx(1/24,rel=1e-12)
        assert float((weights*points[:,0].square()).sum())==pytest.approx(1/60,rel=1e-12)
        assert float((weights*points[:,0]*points[:,1]).sum())==pytest.approx(1/120,rel=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA solved volume integration')
def test_larger_far_batches_preserve_actual_volume_dissipation():
    from dataclasses import replace
    from benchmark.mom_fdtd import INTEGRATION, tetrahedral_volume_rule
    from witwin.maxwell.mom.observables import volume_absorption
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1),mw.Material(eps_r=2.25,sigma_e=.004))],
                   sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8,phase=.23))])
    currents=[];powers=[]
    for batch in (1024,4096):
        config=mw.MoMConfig(subdivisions=1,integration=replace(INTEGRATION,far_pair_batch=batch))
        result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
        data=result.surface_currents;points,weights=tetrahedral_volume_rule(data.mesh,degree=2)
        powers.append(data.absorbed_power(points,weights))
        currents.append(data.scattered_coefficients)
    torch.testing.assert_close(currents[0],currents[1],rtol=1e-11,atol=1e-14)
    torch.testing.assert_close(powers[0],powers[1],rtol=1e-11,atol=1e-15)
