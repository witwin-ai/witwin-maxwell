"""Exercise the changed Sparse builder preflight without constructing maps."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan
from witwin.maxwell.mom.fmm_capacity import plan_directional_reciprocal_near_storage
from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_sparse_reciprocal_near_capacity_rejects_before_maps(tmp_path,monkeypatch):
    exterior=compile_medium(mw.Material(),3e8)
    interior=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01,sigma_m=.02),3e8)
    basis=RWGBasis(sphere_mesh(radius=.02,subdivisions=0,device='cuda',dtype=torch.float64))
    provider=ScalarSystemBlocks(SurfaceSystem.create([(0,'body',basis,interior)]),exterior,
        IntegrationConfig(order=3,singular_order=3))
    root=AutomaticDirectionalRegionPlan(provider,exterior,
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(exterior.eta),leaf_size=4,separation_ratio=.15,cone_subdivisions=(2**20,))
    capacity=plan_directional_reciprocal_near_storage(root.tree,root.near_nodes,budget_bytes=1024**3)
    assert capacity['stored_blocks']<capacity['logical_blocks']
    def forbidden(*args,**kwargs):raise AssertionError('Rejected Sparse reciprocal capacity reached maps')
    monkeypatch.setattr(SparseDirectionalRWGProjection,'__init__',forbidden)
    with pytest.raises(MemoryError,match='before maps'):
        root.build_region(integration=provider.config,quadrature_order=12,
            near_storage='scalar_reciprocal_dk_v1',near_retained_budget_bytes=capacity['retained_bytes']-1)
    (tmp_path/'sparse_preflight.json').write_text(json.dumps(dict(capacity=capacity,
        original_near_integration_not_called=True,Sparse_maps_not_constructed=True,
        only_changed_builder_preflight_collected=True,positive_sparse_FAR_qualification=False,
        qualification=False),indent=2)+'\n')
