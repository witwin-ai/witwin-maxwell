"""Transverse mode eigenvalues against independent rectangular-waveguide spectra."""
import math
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('axis',('x','z'))
def test_vacuum_mode_matches_discrete_rectangular_cutoff(axis):
    scene = mw.Scene(domain=mw.Domain(bounds=((0.,.2),(0.,.12),(0.,.16))),
                     grid=mw.GridSpec.uniform(.02),boundary=mw.BoundarySpec.none(),device='cuda')
    prepared = mw.Simulation.fdfd(scene,frequency=2e9,
                 solver=mw.GMRES(solver_type='direct',precision='double')).prepare()
    result = prepared.solve_modes(axis=axis,position=.04,num_modes=2)
    transverse = [i for i in range(3) if i!='xyz'.index(axis)]
    lengths = [.2,.12,.16]
    width = max(lengths[i] for i in transverse)
    cutoff = 2/.02*math.sin(math.pi/(2*round(width/.02)))
    expected = math.sqrt(prepared.solver.k0**2-cutoff**2)
    beta = result.monitor('modes')['beta']
    torch.testing.assert_close(beta[0].real,torch.tensor(expected,device='cuda',dtype=torch.float64),rtol=1e-10,atol=1e-10)
    assert beta.is_cuda and float(beta.imag.abs().max())<1e-9
    assert all(v.is_cuda for v in result.raw_output['profiles'].values())


def test_graded_dielectric_mode_converges_to_continuum():
    errors = []
    for cells in (8,16):
        nodes = torch.linspace(0,1,cells+1,dtype=torch.float64).pow(1.2).numpy()
        scene = mw.Scene(domain=mw.Domain(bounds=((0.,.2),(0.,.12),(0.,.04))),
                         grid=mw.GridSpec.custom(nodes*.2,nodes*.12,[0.,.02,.04]),
                         boundary=mw.BoundarySpec.none(),device='cuda')
        scene.add_structure(mw.Structure(name='fill',geometry=mw.Box(position=(.1,.06,.02),size=(.4,.3,.2)),
                                         material=mw.Material(eps_r=2.25)))
        prepared = mw.Simulation.fdfd(scene,frequency=2e9,
                      solver=mw.GMRES(solver_type='direct',precision='double')).prepare()
        result = prepared.solve_modes(position=.02)
        actual = float(result.monitor('modes')['beta'][0].real)
        expected = math.sqrt(prepared.solver.k0**2*2.25-(math.pi/.2)**2)
        errors.append(abs(actual-expected)/expected)
    assert errors[1]<errors[0]*.4 and errors[1]<5e-4
