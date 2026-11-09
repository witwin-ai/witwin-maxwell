"""Complete compressed dielectric HB state restores without electromagnetic solves."""
import copy
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import _compile_basis,_signature
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material,material_signature
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance
from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
from witwin.maxwell.mom.harmonic_runtime import HarmonicData
from witwin.maxwell.mom.harmonic_snapshot import snapshot,restore
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeHConfig
from witwin.maxwell.mom.volume_h_snapshot import _digest,restore as restore_h

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_complete_compressed_hb_state_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        scene,config,inside=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
        materials=tuple(compile_harmonic_material(s.material,3e8,2) for s in scene.structures)
        exterior=compile_harmonic_material(mw.Material(),3e8,2)
        bases=[_compile_basis(scene,s.geometry,config) for s in scene.structures]
        linear=replace(config,harmonic_balance=None,operator_backend='h2',hierarchical=HMatrixConfig(leaf_size=64),
            riesz_storage='local_rwg_v1',stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
        operators=tuple(HierarchicalH2SystemOperator(SurfaceSystem.create([(i,s.name,b,m.linear_samples[n])
            for i,(s,b,m) in enumerate(zip(scene.structures,bases,materials))]),exterior.linear_samples[n],scene.sources if n==0 else (),linear) for n in range(2))
        options=dict(source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),volume_config=VolumeHConfig(leaf_size=64))
        model,prepared=prepare_harmonic_balance(operators,config.harmonic_balance.volumes,materials,exterior,
            query_config=config.query_config,compressed_settings=options)
        hb=config.harmonic_balance
        solution=model.solve_ramp(hb.pump_scales,branch_id=hb.branch_id,**hb.solve_settings())[-1]
        state=recover_harmonic_state(model,prepared,solution)
        signature=(_signature(scene,exterior.linear_samples[0],tuple(m.linear_samples[0] for m in materials)),
            tuple(material_signature(s.material) for s in scene.structures),3e8)
        data=HarmonicData(prepared,state,config,signature)
        payload=snapshot(data);torch.save(payload,tmp_path/'hb.pt');payload=torch.load(tmp_path/'hb.pt',weights_only=True)
        def forbidden(*args,**kwargs):raise AssertionError('Compressed HB restore must not solve')
        with monkeypatch.context() as patch:
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=restore(payload,scene)
            assert _digest(snapshot(restored))==_digest(payload)
            for n in (1,2):
                a=restored.query_fields(inside,1,'total',harmonic=n)
                b=data.query_fields(inside,1,'total',harmonic=n)
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for parts in (payload['prepared']['harmonics'][:-1],payload['prepared']['harmonics']+payload['prepared']['harmonics'][:1]):
                bad=copy.deepcopy(payload);bad['prepared']['harmonics']=parts
                with pytest.raises(ValueError,match='response count'):restore(bad,scene)
            action=restored.decoder.prepared['harmonics'][0]['electric_action'].sources[0]
            raw=payload['prepared']['harmonics'][0]['actions']['regions'][0]['source']
            assert action.restored_retained_bytes>raw['receipt']['retained_bytes']
            bad=copy.deepcopy(raw);budget=raw['receipt']['retained_bytes']
            bad['configuration']['retained_budget_bytes']=budget
            bad['receipt']['configuration']['retained_budget_bytes']=budget
            bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
            with pytest.raises(MemoryError):restore_h(bad,action.provider)
            for name in ('surface_coefficients','induced_surface_coefficients'):
                bad=copy.deepcopy(payload);bad['state'][name][0].add_(1e-3)
                with pytest.raises(ValueError,match='surface|tolerance'):restore(bad,scene)
        with torch.no_grad():restored.state.induced_surface_coefficients[0].add_(1)
        with pytest.raises(RuntimeError,match='mutated'):restored.decoder._assert_current()
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
