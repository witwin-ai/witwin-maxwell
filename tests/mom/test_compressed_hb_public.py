"""Public compressed current, mixed and dual HB results preserve native state."""
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(kind,backend):
    original,config,_=runpy.run_path('tests/mom/test_harmonic_public.py')['setup']()
    volumes=config.harmonic_balance.volumes
    material=mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    structures=[mw.Structure(geometry=original.structures[0].geometry,material=material)]
    if kind=='dual':
        scene,config,_,_=runpy.run_path('tests/mom/test_harmonic_dual_public.py')['dual_setup']()
        # Separate query verification below avoids duplicate monitor work.
        scene=mw.Scene(structures=scene.structures,sources=scene.sources)
    else:
        if kind=='mixed':
            first=volumes[0]
            second=mw.TetrahedralVolumeMesh(first.vertices+first.vertices.new_tensor([.18,.02,.01]),first.cells,region_id=2)
            structures.append(mw.Structure(geometry=mw.Mesh(second.boundary.vertices,second.boundary.faces,recenter=False,fill_mode='solid'),
                material=mw.Material(eps_r=3.,mu_r=2.,nonlinearity=mw.NonlinearSusceptibility(chi2=.004,chi3=.001))))
            config=replace(config,harmonic_balance=replace(config.harmonic_balance,volumes=(second,first)))
        scene=mw.Scene(structures=structures,sources=original.sources)
    config=replace(config,operator_backend=backend,hierarchical=HMatrixConfig(leaf_size=64),
        riesz_storage='local_rwg_v1',stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm',
            rtol=1e-11,atol=1e-20,stagnation_window=48),
        harmonic_balance=replace(config.harmonic_balance,source_max_order=64,source_max_inner_order=96))
    return scene,config


@pytest.mark.parametrize('kind,backend',[('current','hmatrix'),('mixed','h2'),('dual','h2')])
def test_public_compressed_state_restore(kind,backend,tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        scene,config=setup(kind,backend)
        prepared=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
        assert prepared.solver.prepared['response_backend']=='compressed'
        assert not hasattr(prepared.solver.model,'current_response')
        result=prepared.run();path=tmp_path/'result.pt';result.save(path)
        payload=torch.load(path,weights_only=True)['harmonic_data']
        assert payload['schema_version']==15 and payload['numerical']['operator_backend']==backend
        state=result.harmonic_state
        check_newton=runpy.run_path('tests/mom/test_harmonic_newton_krylov.py')['check_compressed_newton']
        check_newton(prepared.solver.model,state.cell_electric,config.harmonic_balance,implicit_solver='streamed_krylov')
        assert len(state.induced_surface_coefficients)==2
        assert not state.receipt['physical_qualification']
        if kind=='mixed':assert state.dc_current_regions==(1,)
        if kind=='dual':assert state.dc_dual_state is not None
        point=config.harmonic_balance.volumes[0].vertices.new_tensor([[.16,.1,.09]])
        ac=result.query_fields(point,region_id=0,field_kind='total',harmonic=2)
        dc=result.query_dc_dual(point,region_id=0) if kind=='dual' else result.query_dc(point,region_id=0)
        def forbidden(*args,**kwargs):raise AssertionError('Public compressed restore/query cannot solve')
        with monkeypatch.context() as patch:
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(path,scene=scene)
            from witwin.maxwell.mom.harmonic_convergence import _validate_dc_prefix
            _validate_dc_prefix(result._harmonic_data.decoder.prepared,restored._harmonic_data.decoder.prepared)
            from witwin.maxwell.mom.runtime import _numerical_identity
            assert _numerical_identity(restored._harmonic_data.config)==_numerical_identity(config)
            for a,b in zip(restored.query_fields(point,region_id=0,field_kind='total',harmonic=2),ac):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            actual=restored.query_dc_dual(point,region_id=0) if kind=='dual' else restored.query_dc(point,region_id=0)
            pairs=((actual[k],v) for k,v in dc.items() if isinstance(v,torch.Tensor)) if kind=='dual' else zip(actual,dc)
            for a,b in pairs:torch.testing.assert_close(a,b,rtol=0,atol=0)
        restored.save(tmp_path/'restored.pt')
        from witwin.maxwell.mom.volume_h_snapshot import _digest
        assert _digest(torch.load(tmp_path/'restored.pt',weights_only=True)['harmonic_data'])==_digest(payload)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
