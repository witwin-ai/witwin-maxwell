"""A source corrupted after local admission must fail whole preparation."""
import runpy
import pytest
import torch
import witwin.maxwell.mom.compressed_volume_preparation as preparation
from witwin.maxwell.mom.volume_response_matrix import VolumeMatrixConvergenceError
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeHConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_admitted_then_corrupted_source_is_rejected_by_original_equation(monkeypatch):
    _,bindings,inverse=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']()
    original=preparation.prepare_source_action
    def corrupted(binding,*args,**kwargs):
        action,receipt=original(binding,*args,**kwargs)
        assert receipt['converged']
        if binding.volume.region_id==2:
            apply=action.apply
            monkeypatch.setattr(action,'apply',lambda value:1.01*apply(value))
        return action,receipt
    monkeypatch.setattr(preparation,'prepare_source_action',corrupted)
    with pytest.raises(VolumeMatrixConvergenceError) as info:
        preparation.prepare_compressed_cell_response(inverse,bindings,
            source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),
            volume_config=VolumeHConfig(leaf_size=64),column_batch=6)
    receipt=info.value.receipt
    assert not receipt['converged']
    assert receipt['failure_reason']=='original_equation_or_field_check_failed'
    assert all(source['converged'] for source in receipt['sources'])
    assert all(channel['passed'] for channel in receipt['field_checks'][-1]['channels'])
    assert any(not c['original_equation_pass'] for r in receipt['original_checks'] for c in r['checks'])
