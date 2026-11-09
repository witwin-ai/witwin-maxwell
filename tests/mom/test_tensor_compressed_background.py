"""Actual compressed tensor-source background in its scalar limit."""
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_volume_source import setup
from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.compressed_background import prepare_compressed_background

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def inverse_for(tensor):
    handle,binding=setup(tensor)
    config=IntegrationConfig(order=3,singular_order=3)
    provider=(GeneralizedSystemBlocks(handle.system,handle.exterior,config,
        interior_kernels={1:handle.body_handles[0].tensor_kernel}) if tensor else
        ScalarSystemBlocks(handle.system,handle.exterior,config))
    operator=HierarchicalOperator(provider,HMatrixConfig(leaf_size=64))
    inverse=FixedSurfaceInverse(operator,StoppingRule(method='block_gmres',preconditioner='row_column_norm',
        rtol=1e-11,atol=1e-13,stagnation_window=48))
    return inverse,binding


def test_tensor_compressed_background_scalar_limit_and_scale_gradient(monkeypatch):
    sources=(mw.PlaneWave(),)
    inverse,binding=inverse_for(True)
    bank=prepare_compressed_background(inverse,(binding,),sources,edge_batch=2,query_batch=1)
    scalar,sbinding=inverse_for(False)
    reference=prepare_compressed_background(scalar,(sbinding,),sources,edge_batch=2,query_batch=1)
    for name in ('rhs','surface','electric','magnetic'):
        torch.testing.assert_close(getattr(bank,name),getattr(reference,name),rtol=2e-8,atol=1e-10)
    scale=torch.tensor([.7+.2j],device='cuda',dtype=inverse.dtype,requires_grad=True)
    field=bank.combine(scale)['background_electric']
    gradient,=torch.autograd.grad(field.abs().square().sum(),scale)
    expected=2*scale*bank.electric.abs().square().sum()
    torch.testing.assert_close(gradient,expected,rtol=2e-12,atol=1e-12)
    assert bank.receipt['converged'] and bank.receipt['tensor_regions']==[1]
    assert bank.receipt['tensor_rhs_cache_max_bytes_per_handle']==0
    def forbidden(*args,**kwargs):raise AssertionError('Empty tensor background must not solve')
    monkeypatch.setattr(inverse,'apply',forbidden)
    empty=prepare_compressed_background(inverse,(binding,),())
    assert empty.electric.shape==(0,1,3)
    kernel=inverse.operator.provider.interior_kernels[0]
    kernel.mode_cache.max_bytes+=1
    with pytest.raises(ValueError,match='binding changed'):
        bank.combine(scale)
    with pytest.raises(ValueError,match='binding changed'):
        prepare_compressed_background(inverse,(binding,),sources)
    with pytest.raises(ValueError,match='cache policy'):
        GeneralizedSystemBlocks(inverse.operator.provider.system,inverse.operator.provider.exterior,
            inverse.operator.provider.config,interior_kernels={1:kernel})
    kernel.mode_cache.max_bytes-=1
    bank.assert_current()
    kernel.mode_cache.medium=setup(True)[0].body_handles[0].interior
    with pytest.raises(ValueError,match='binding changed'):bank.combine(scale)
