"""FFT native admission precedes support traversal even with no far pairs."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage
from witwin.maxwell.mom.fmm_order import plan_multipole_order
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_fft_native_order_and_float32_fail_before_support_tree_or_pair_walk(monkeypatch):
    import witwin.maxwell.mom.fmm_capacity as capacity
    import witwin.maxwell.mom.fmm_order as planner
    import witwin.maxwell.mom.fmm_system as system_module
    def forbidden(*args,**kwargs):raise AssertionError('Invalid FFT request reached tree traversal')
    monkeypatch.setattr(capacity,'sphere_block_partition',forbidden)
    monkeypatch.setattr(planner,'_kernel_order_chunk',forbidden)
    monkeypatch.setattr(system_module,'build_support_octree',forbidden)
    for dtype,order in ((torch.float32,6),(torch.float64,13)):
        tree=SimpleNamespace(lower=torch.zeros((1,3),device='cuda',dtype=dtype))
        with pytest.raises(ValueError):plan_multipole_storage((tree,),order=order,separation_ratio=.5,
            budget_bytes=1024**3,table_storage='fft_v1')
        with pytest.raises(ValueError):plan_multipole_order(((tree,None),),mw.KernelErrorBudget(1e-6,1e-5),
            min_order=order,max_order=order,table_storage='fft_v1')
    for dtype,order,max_order in ((torch.float32,6,None),(torch.float64,13,None),(torch.float64,6,13)):
        medium=compile_medium(mw.Material(),3e8)
        basis=RWGBasis(sphere_mesh(radius=.01,subdivisions=0,device='cuda',dtype=dtype))
        system=SurfaceSystem.create([(0,'body',basis,compile_medium(mw.Material(eps_r=2.),3e8))])
        provider=ScalarSystemBlocks(system,medium,IntegrationConfig(order=3,singular_order=3))
        with pytest.raises(ValueError):MultipoleSystemActions(provider,order=order,max_order=max_order,
            table_storage='fft_v1',kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5))
