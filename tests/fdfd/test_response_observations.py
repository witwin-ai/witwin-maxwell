"""Independent analytic field and work-normalization observation checks."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdfd.observations import compile_projection, gap_layout
from witwin.maxwell.scene import prepare_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def scene_at(step=.04):
    return mw.Scene(domain=mw.Domain(bounds=((-0.2,.2),)*3),grid=mw.GridSpec.uniform(step),
                    boundary=mw.BoundarySpec.none(),device='cuda')


def test_magnetic_projection_of_manufactured_field():
    scene = prepare_scene(scene_at())
    electric = []
    for component in ('Ex','Ey','Ez'):
        coords = [getattr(scene, a+('_half' if a==component[1].lower() else '')).double() for a in 'xyz']
        x,y,z = torch.meshgrid(*coords,indexing='ij')
        electric.append((x if component=='Ez' else torch.zeros_like(x)).flatten())
    electric = torch.cat(electric).to(torch.complex128)
    L = compile_projection((mw.FieldProjection('h',component='Hy',position=(.01,.02,.03)),),scene,torch.complex128,1e9)
    actual = torch.sparse.mm(L,electric[:,None])[0,0]
    expected = 1j/(2*math.pi*1e9*(4*math.pi*1e-7))
    torch.testing.assert_close(actual,torch.tensor(expected,device='cuda'),rtol=1e-6,atol=1e-12,check_dtype=False)


def test_lumped_current_voltage_work_and_full_observable_api():
    scene = scene_at()
    start,end = (0.,0.,-.04),(0.,0.,.04)
    scene.add_source(mw.LumpedCurrentSource(start,end,current=2-1j))
    observation = mw.FieldProjection.voltage('feed',start,end)
    sim = mw.Simulation.fdfd(scene,frequency=1e9,solver=mw.GMRES(solver_type='direct',precision='double'),
                             observations=(observation,))
    prepared = sim.prepare(); solver = prepared.solver
    solver._ensure_system_matrix()
    d = torch.from_dlpack(solver._ensure_symmetrization_scale())
    density = torch.from_dlpack(solver._build_source_vector_yee())/d/(-1j*solver.omega*solver.mu0)
    indices,lengths,area = gap_layout(solver.scene,start,end)
    torch.testing.assert_close(density[indices]*area,torch.full_like(density[indices],2-1j),rtol=1e-6,atol=1e-8)
    result = prepared.run()
    electric = torch.cat([result.tensor(c).flatten() for c in ('Ex','Ey','Ez')])
    voltage = -(electric[indices]*lengths).sum()
    torch.testing.assert_close(result.monitor('feed')['value'],voltage,rtol=1e-6,atol=1e-8,check_dtype=False)
    solver._release_direct_solver()


def test_vacuum_huygens_projection_converges_to_analytic_dipole():
    errors = []
    for step in (.04,.02):
        scene = prepare_scene(scene_at(step))
        k = 2*math.pi*1e9/299792458.
        electric = []
        for component in ('Ex','Ey','Ez'):
            coords = [getattr(scene,a+('_half' if a==component[1].lower() else '')).double() for a in 'xyz']
            xyz = torch.stack(torch.meshgrid(*coords,indexing='ij'),dim=-1)
            r = torch.linalg.vector_norm(xyz,dim=-1).clamp_min(step/10)
            direction = xyz/r[...,None]
            p = torch.tensor((0.,0.,1.),dtype=torch.float64,device='cuda')
            dot = direction[...,2]
            field = torch.exp(1j*k*r)[...,None]*(k*k/r[...,None]*(p-direction*dot[...,None])
                     +(1/r**3-1j*k/r**2)[...,None]*(3*direction*dot[...,None]-p))
            electric.append(field[...,'xyz'.index(component[1].lower())].flatten())
        spec = mw.FarFieldProjection('far',bounds=((-0.12,.12),)*3,direction=(1.,0.,0.),polarization=(0.,0.,1.))
        L = compile_projection((spec,),scene,torch.complex128,1e9)
        actual = torch.sparse.mm(L,torch.cat(electric)[:,None])[0,0]
        errors.append(float(abs(actual-k*k)/(k*k)))
    assert errors[1] < .06, errors
    assert errors[1] < errors[0]*.5, errors


def test_full_farfield_rejects_source_outside_huygens_box():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.12,.12),)*3),grid=mw.GridSpec.uniform(.04),
                     boundary=mw.BoundarySpec.pml(num_layers=2),device='cuda')
    scene.add_source(mw.PointDipole(position=(.08,0.,0.),polarization='Ez',width=.01,
                                    source_time=mw.CW(frequency=1e9)))
    far = mw.FarFieldProjection('far',bounds=((-0.04,.04),)*3,
                               direction=(0.,0.,1.),polarization=(1.,0.,0.))
    with pytest.raises(ValueError,match='complete discretized source'):
        mw.Simulation.fdfd(scene,frequency=1e9,solver=mw.GMRES(solver_type='direct',precision='double'),
                            observations=(far,)).run()
