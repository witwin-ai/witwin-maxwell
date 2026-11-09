"""Public tensor susceptibility sources with scalar regional AC backgrounds."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('backend',['dense','h2'])
def test_public_tensor_sources_exact_result_and_scene_binding(tmp_path,monkeypatch,backend):
    scene,config,point=setup()
    if backend=='h2':
        from witwin.maxwell.mom.hmatrix import HMatrixConfig
        from witwin.maxwell.mom.linalg import StoppingRule
        config=replace(config,operator_backend='h2',riesz_storage='local_rwg_v1',
            hierarchical=HMatrixConfig(leaf_size=64),stopping=StoppingRule(method='block_gmres',
                preconditioner='row_column_norm',rtol=1e-11,atol=1e-20,stagnation_window=48))
    axis=point.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
    q=.002*torch.einsum('i,j,k->ijk',axis,axis,axis)
    c=.001*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis)
    def material(scale=1.):
        return mw.Material(eps_r=2.,nonlinearity=mw.TensorNonlinearSusceptibility(chi2=q*scale,chi3=c))
    scene.structures[1]=replace(scene.structures[1],material=material())
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    state=result.harmonic_state
    assert not state.receipt['physical_qualification']
    assert bool(state.cell_electric[0].norm()>0) and bool(state.cell_electric[2].norm()>0)
    prepared=result._harmonic_data.decoder.prepared
    assert prepared['response_backend']==('compressed' if backend=='h2' else 'dense')
    for name,expected in (('chi2_tensor',q),('chi3_tensor',c)):
        torch.testing.assert_close(prepared['nonlinear_coefficients'][name][0],expected,rtol=0,atol=0)
    fields=[result.query_fields(point,region_id=2,field_kind='total',harmonic=n) for n in (1,2)]
    dc=result.query_dc(point,region_id=2)
    path=tmp_path/'tensor-sources.pt';result.save(path)
    from witwin.maxwell.mom.runtime import SystemOperator
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    def forbidden(*args,**kwargs):raise AssertionError('Public restore must not assemble or solve')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        if backend=='h2':
            from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
            from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(GeneralizedSystemBlocks,'__call__',forbidden)
        restored=mw.Result.load(path,scene=scene)
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(restored.harmonic_state,name),getattr(state,name),rtol=0,atol=0)
        for actual,expected in zip(restored.query_dc(point,region_id=2),dc):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for n,expected_fields in enumerate(fields,1):
            for actual,expected in zip(restored.query_fields(point,region_id=2,field_kind='total',harmonic=n),expected_fields):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        original=scene.structures[1]
        scene.structures[1]=replace(original,material=material(2.))
        with pytest.raises(ValueError,match='Scene identity'):
            mw.Result.load(path,scene=scene)
        scene.structures[1]=original
    print({'public_tensor_sources':True,'exact_state_and_queries':True,'physical_qualification':False})
