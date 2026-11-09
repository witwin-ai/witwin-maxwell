"""Exact body-owned zero-induced state, without a second material contrast."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.linear_volume_state import LinearCoupledConfig, LinearVolumeState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def system_and_volumes(dtype):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                            device='cuda', dtype=dtype)*.07
    cells = torch.tensor([[0, 1, 2, 3]], device='cuda')
    volumes = tuple(TetrahedralVolumeMesh(vertices+shift, cells, region_id=i+1)
                    for i, shift in enumerate((0., .3)))
    entries = [(i, f'body{i}', RWGBasis(SurfaceMesh(v.boundary.vertices, v.boundary.faces)),
                compile_medium(mw.Material(eps_r=2.+i), 3e8)) for i, v in enumerate(volumes)]
    return SurfaceSystem.create(entries), volumes


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_exact_state_restore_and_reject_changed_physical_support(dtype):
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    system, volumes = system_and_volumes(real)
    state = LinearVolumeState(system, LinearCoupledConfig(volumes[::-1]), dtype=dtype)
    assert tuple(b.volume.region_id for b in state.bindings) == (1, 2)
    assert all(not bool(v.any()) for v in (*state.polarization, *state.currents))
    payload = state.snapshot()
    restored = LinearVolumeState.restore(payload, system, dtype=dtype)
    for old, new in zip(state.bindings, restored.bindings):
        assert torch.equal(old.volume.vertices, new.volume.vertices)
        assert torch.equal(old.volume.cells, new.volume.cells)
        assert old.volume is not new.volume
    for mutation in ('nonzero', 'nan', 'precision', 'region', 'geometry', 'partition', 'qualification'):
        bad = copy.deepcopy(payload)
        if mutation == 'nonzero': bad['regions'][1]['additional_polarization'][0, 1] = 1
        elif mutation == 'nan': bad['regions'][0]['additional_current'][0, 0] = float('nan')
        elif mutation == 'precision': bad['regions'][0]['additional_current'] = bad['regions'][0]['additional_current'].to(torch.float64)
        elif mutation == 'region': bad['regions'].reverse()
        elif mutation == 'geometry': bad['regions'][1]['vertices'] += .01
        elif mutation == 'partition': bad['material_partition'] = 'duplicate_linear_contrast'
        else: bad['physical_qualification'] = True
        with pytest.raises(ValueError): LinearVolumeState.restore(bad, system, dtype=dtype)
    # Detached serialization must not expose writable aliases of live arrays.
    payload['regions'][0]['vertices'] += 1
    state.assert_current()
    restored.currents[0].add_(1)
    with pytest.raises(ValueError, match='changed'): restored.snapshot()
    with pytest.raises(ValueError, match='one conforming volume'):
        LinearVolumeState(system, LinearCoupledConfig(volumes[:1]), dtype=dtype)
