"""Adaptive compressed preparation shares solves and matches the dense response."""
import json
import runpy
import pytest
import torch
from witwin.maxwell.sources import PlaneWave
from witwin.maxwell.mom.compressed_volume_preparation import prepare_compressed_cell_response
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.h2 import H2Config
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeHConfig
from witwin.maxwell.mom.volume_response_matrix import prepare_cell_field_response,VolumeMatrixConvergenceError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup():
    dense,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    h=HierarchicalOperator(ScalarSystemBlocks(dense.system,dense.exterior,dense.integration),HMatrixConfig(leaf_size=64))
    inverse=FixedSurfaceInverse(HierarchicalH2Operator(h),StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,stagnation_window=48))
    return dense,bindings[::-1],inverse


def test_actual_default_preparation_shared_eh_solves_and_dense_control(monkeypatch,tmp_path):
    dense,bindings,inverse=setup()
    settings=dict(source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),
        volume_config=VolumeHConfig(leaf_size=64),volume_h2_config=H2Config(),column_batch=2)
    calls=[];old=inverse.apply
    def counted(value):calls.append(value.shape);return old(value)
    monkeypatch.setattr(inverse,'apply',counted)
    result=prepare_compressed_cell_response(inverse,bindings,incident_sources=(PlaneWave(),),**settings)
    receipt=result['receipt']
    assert receipt['converged'] and len(calls)==3*(len(receipt['field_checks'])+1)+1
    assert all(r['passed'] for r in receipt['original_checks'])
    assert not receipt['physical_qualification'] and receipt['incident_background_included']
    (tmp_path/'receipt.json').write_text(json.dumps(receipt,indent=2))
    reference=prepare_cell_field_response(dense,bindings)
    background=result['background_bank'].combine(torch.ones(1,device='cuda',dtype=inverse.dtype))
    for name in ('background_electric','background_magnetic','background_surface'):
        torch.testing.assert_close(background[name],reference[name],rtol=2e-8,atol=1e-10)
    eye=torch.eye(6,device='cuda',dtype=inverse.dtype)
    for field in ('electric','magnetic'):
        value=result[field+'_action'].apply(eye)
        torch.testing.assert_close(value,reference[field+'_response'].mT,rtol=1e-7,atol=1e-10)


def test_initial_field_budget_fails_before_source_preparation(monkeypatch):
    _,bindings,inverse=setup()
    import witwin.maxwell.mom.compressed_volume_preparation as preparation
    def forbidden(*args,**kwargs):raise AssertionError('Source assembly should not run')
    monkeypatch.setattr(preparation,'prepare_source_action',forbidden)
    with pytest.raises(VolumeMatrixConvergenceError) as info:
        prepare_compressed_cell_response(inverse,bindings,query_config=FieldQueryConfig(max_order=3))
    assert info.value.receipt['failure_reason']=='initial_field_order_leaves_no_refinement'
    assert not info.value.receipt['converged']
