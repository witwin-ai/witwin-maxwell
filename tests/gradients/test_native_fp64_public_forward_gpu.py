"""Actual public detached native FP64 forward: no admission bypass fixture."""
import pytest
import torch
import witwin.maxwell as mw
from tests.core.scene.fp64_public_scene import scene,simulation

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA execution.')


def power(result):return result.antenna(surface='radiation',driven_port='feed',theta_points=9,phi_points=13,radius=10.).p_rad


def test_direct_public_fp64_forward_and_precision_persistence(tmp_path):
    value=scene(device='cuda')
    low=simulation(value,dtype=torch.float32).run()
    high=simulation(value).run()
    assert high.solver.Ex.dtype==torch.float64
    assert high.stats()['field_dtype']=='torch.float64'
    assert high.stats()['observer_dtype']=='torch.float64'
    assert high.solver.eps_Ex.dtype==high.solver.cpml_b_e_x.dtype==torch.float64
    assert high.prepared_scene.x.dtype==torch.float64
    for face in high._native_raw_monitor('radiation')['faces'].values():
        assert all(face[name].dtype==torch.complex128 for name in face['fields'])
    torch.testing.assert_close(power(high),power(low),rtol=1e-5,atol=0)
    path=tmp_path/'public64.pt';high.save(path)
    restored=mw.Result.load(path,scene=value,map_location='cuda:0')
    assert restored.prepared_scene.x.dtype==torch.float64
    torch.testing.assert_close(power(restored),power(high),rtol=1e-12,atol=1e-25)
    with pytest.raises(ValueError,match='frequency-domain volume fields'):high.power_loss('bulk')


def test_public_detached_material_change_uses_actual_native_double_coefficients():
    first=simulation(scene(device='cuda',bulk_sigma=.15)).run()
    second=simulation(scene(device='cuda',bulk_sigma=.2)).run()
    assert first.solver.sigma_e_Ex.dtype==second.solver.sigma_e_Ex.dtype==torch.float64
    assert float((power(first)-power(second)).abs().max())>1e-10


def test_actual_public_trainable_fp64_fails_closed_before_field_update():
    parameter=torch.tensor(2.,device='cuda',requires_grad=True)
    value=scene(device='cuda',curvature=None,material=mw.Material(eps_r=parameter))
    with pytest.raises(ValueError,match='detached'):
        simulation(value).prepare()
