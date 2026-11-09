"""Actual adaptive tensor compressed preparation and exact factor persistence."""
import json
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_compressed_background import inverse_for
from witwin.maxwell.mom.compressed_volume_preparation import prepare_compressed_cell_response
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.h2 import H2Config
from witwin.maxwell.mom import coupled_response_snapshot as codec
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_adaptive_tensor_compressed_preparation(monkeypatch,tmp_path):
    inverse,binding=inverse_for(True)
    print('START adaptive tensor compressed response',flush=True)
    prepared=prepare_compressed_cell_response(inverse,(binding,),
        query_config=FieldQueryConfig(relative_tolerance=1e-4,absolute_tolerance=1e-10,max_order=48),
        source_settings=dict(max_order=64,max_inner_order=64),column_batch=3,
        tensor_face_rule='projected_duffy',tensor_kernel_batch=8192,
        volume_h2_config=H2Config(),incident_sources=(mw.PlaneWave(),))
    receipt=prepared['receipt']
    (tmp_path/'preparation.json').write_text(json.dumps(receipt,indent=2))
    assert receipt['converged'] and receipt['incident_background_included']
    assert all(check['passed'] for check in receipt['original_checks'])
    assert all(check['passed'] for check in receipt['field_checks'][-1]['channels'])
    assert not receipt['physical_qualification']
    print('COMPLETE adaptive preparation',flush=True)
    actions=prepared['electric_action'],prepared['magnetic_action']
    payload=codec.encode(*actions)
    path=tmp_path/'tensor-compressed.pt';torch.save(payload,path)
    fresh,fresh_binding=inverse_for(True)
    def forbidden(*args,**kwargs):raise AssertionError('Replay must not sample or solve')
    with monkeypatch.context() as patch:
        for cls in (GeneralizedSystemBlocks,SurfaceVolumeBlocks,InteriorVolumeBlocks,VolumeCellBlocks):
            patch.setattr(cls,'__call__',forbidden)
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        restored=codec.restore(torch.load(path,weights_only=True),fresh.operator.provider,(fresh_binding.volume,))
    assert codec.encode(*restored)['sha256']==payload['sha256']
    x=torch.tensor([.3+.1j,-.2j,.4],device='cuda',dtype=inverse.dtype)
    for before,after in zip(actions,restored):
        torch.testing.assert_close(before.apply(x),after.apply(x),rtol=2e-10,atol=1e-12)
    print('COMPLETE exact factors and numerical response replay',flush=True)
