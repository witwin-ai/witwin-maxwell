"""Physical multi-source preparation restores and verifies saved currents without solves."""
import copy
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.harmonic_preparation import _prepare_compressed_harmonic
from witwin.maxwell.mom.compressed_harmonic_snapshot import encode,restore,validate_surface_state
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_admitted_physical_preparation_and_saved_surface(monkeypatch,tmp_path):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        build=runpy.run_path('tests/mom/test_surface_volume.py')['problem']
        dense,bindings=build((2.,3.));fresh,fresh_bindings=build((2.,3.))
        sources=(mw.PlaneWave(source_time=mw.CW(3e8,amplitude=1.2,phase=.2)),
            mw.PlaneWave(direction=(1,0,0),polarization=(0,1,0),source_time=mw.CW(3e8,amplitude=.8,phase=-.3)))
        config=mw.MoMConfig(operator_backend='h2',integration=dense.integration,
            hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
            stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
        op=HierarchicalH2SystemOperator(dense.system,dense.exterior,sources,config)
        options=dict(source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),volume_config=VolumeHConfig(leaf_size=64))
        part=_prepare_compressed_harmonic(op,bindings[::-1],None,options)
        x=torch.randn(2,6,device='cuda',dtype=op.dtype)*1e-5
        surface=part['electric_action'].surface_response(x)
        expected=tuple(part[k].apply(x) for k in ('electric_action','magnetic_action'))
        payload=encode(part,sources=sources);torch.save(payload,tmp_path/'preparation.pt')
        payload=torch.load(tmp_path/'preparation.pt',weights_only=True)
        provider=ScalarSystemBlocks(fresh.system,fresh.exterior,fresh.integration)
        def forbidden(*args,**kwargs):raise AssertionError('Restore and state verification must not solve or sample')
        with monkeypatch.context() as patch:
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            for cls in (ScalarSystemBlocks,SurfaceVolumeBlocks,InteriorVolumeBlocks,VolumeCellBlocks):patch.setattr(cls,'__call__',forbidden)
            restored=restore(payload,provider,[b.volume for b in fresh_bindings[::-1]],sources=iter(sources))
            actual=validate_surface_state(restored,x,surface)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-8,atol=1e-12)
            with pytest.raises(ValueError,match='tolerance'):validate_surface_state(restored,x,surface+1e-3)
        assert encode(restored,sources=sources)['sha256']==payload['sha256']
        with pytest.raises(ValueError,match='carrier'):restore(payload,provider,[b.volume for b in fresh_bindings[::-1]],sources=sources[::-1])
        bad=copy.deepcopy(payload);bad['receipt']['physical_qualification']=True
        bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError,match='receipt'):restore(bad,provider,[b.volume for b in fresh_bindings[::-1]],sources=sources)
        with torch.no_grad():restored['background_electric'][0,0]+=1
        with pytest.raises(ValueError,match='derived'):encode(restored,sources=sources)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
