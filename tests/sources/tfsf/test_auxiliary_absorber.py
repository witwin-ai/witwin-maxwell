"""Independent incident-line reflection regression for matched source damping."""
import math
import pytest
import torch
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.fdtd.excitation.spatial import AuxiliaryGrid1D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('spacing', [.03, .02])
def test_incident_auxiliary_reflection_is_small_across_grid_refinement(spacing):
    frequency = 3e8
    dt = .45 * spacing / C_0
    eta = math.sqrt(MU_0/EPSILON_0)
    grid = AuxiliaryGrid1D(s_min=0, s_max=2.4+20*spacing, ds=spacing, dt=dt,
        wave_speed=C_0, impedance=eta, source_time={'kind':'cw', 'frequency':frequency,
        'amplitude':1., 'phase':.23}, device='cuda', dtype=torch.float32, absorber_cells=20)
    indices = torch.tensor([round(.45/spacing), round(.7/spacing), round(.95/spacing)], device='cuda')
    phasor = torch.zeros(3, dtype=torch.complex128, device='cuda')
    count = 0
    total = math.ceil(30/frequency/dt)
    for step in range(total):
        grid.advance()
        if step*dt > 20/frequency:
            phase = complex(math.cos(2*math.pi*frequency*grid.electric_time), math.sin(2*math.pi*frequency*grid.electric_time))
            phasor += grid.electric[indices] * phase
            count += 1
    phasor *= 2/count
    k = 2/spacing * math.asin(math.sin(math.pi*frequency*dt)/(C_0*dt/spacing))
    coordinates = indices.to(torch.float64)*spacing
    plane_basis = torch.stack((torch.exp(1j*k*coordinates), torch.exp(-1j*k*coordinates)), -1)
    forward, backward = torch.linalg.lstsq(plane_basis, phasor).solution
    reflection = float(abs(backward/forward))
    assert reflection < .005, f'Incident auxiliary reflection {reflection:.6g} at ds={spacing}'
    assert phasor.is_cuda
