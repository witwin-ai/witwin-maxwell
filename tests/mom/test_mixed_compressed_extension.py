"""Extend an independently restored compressed mixed harmonic prefix."""
import os
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from witwin.maxwell.mom.tensor_harmonic_bundle import (
    snapshot_mixed_tensor_harmonic_bundle,restore_mixed_tensor_harmonic_bundle)
from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def material(count):
    return compile_harmonic_material(mw.Material(eps_r=2.,
        nonlinearity=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)),3e8,count)


def prefix_bundle():
    # Acceptance collectors may reuse an independently captured bundle instead
    # of repeating its covered preparation. Default pytest remains standalone.
    path=os.environ.get('GSP_MIXED_COMPRESSED_PREFIX')
    if path:return torch.load(path,weights_only=True)
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.hmatrix import HMatrixConfig
    from witwin.maxwell.mom.linalg import StoppingRule
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    sample=material(1);exterior=compile_harmonic_material(mw.Material(),3e8,1)
    config=mw.MoMConfig(operator_backend='h2',integration=IntegrationConfig(order=3,singular_order=3),
        hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
    operator=HierarchicalH2SystemOperator(SurfaceSystem.create([(0,'body',RWGBasis(volume.boundary),sample.linear_samples[0])]),
        exterior.linear_samples[0],(mw.PlaneWave(),),config)
    model,prepared=prepare_mixed_tensor_harmonic_balance((operator,),(volume,),(sample,),exterior,compressed_settings={})
    solution=model.solve(pump_scale=.5)
    state=recover_harmonic_state(model,prepared,solution)
    payload,owner=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
    return dict(payload=payload,owner=owner)


def test_restored_compressed_prefix_only_prepares_new_frequency(monkeypatch,tmp_path):
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    import witwin.maxwell.mom.harmonic_preparation as shared
    from witwin.maxwell.mom.compressed_harmonic_snapshot import encode
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    saved=prefix_bundle()
    _,prefix,old_state=restore_mixed_tensor_harmonic_bundle(saved['payload'],expected_binding=saved['owner'],device='cuda')
    old=prefix['operators'][0]
    volumes=tuple(b.volume for b in prefix['bindings'][0])
    sample=material(2);exterior=compile_harmonic_material(mw.Material(),3e8,2)
    system=SurfaceSystem.create([(b.structure_index,b.name,b.basis,sample.linear_samples[1]) for b in old.system.bodies])
    new=HierarchicalH2SystemOperator(system,exterior.linear_samples[1],(),old._public_config)
    original=shared._prepare_compressed_harmonic;calls=[]
    def appended(operator,*args,**kwargs):
        assert operator is new
        calls.append(operator.exterior.frequency)
        return original(operator,*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Restored prefix must not be prepared again')
    before=_digest(encode(prefix['harmonics'][0],sources=old.sources))
    with monkeypatch.context() as patch:
        patch.setattr(shared,'_prepare_compressed_harmonic',appended)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        with pytest.raises(ValueError,match='response backend'):
            prepare_mixed_tensor_harmonic_balance((old,new),volumes,(sample,),exterior,reuse_prepared=prefix)
        model,extended=prepare_mixed_tensor_harmonic_balance((old,new),volumes,(sample,),exterior,
            reuse_prepared=prefix,compressed_settings={})
        assert calls==[6e8]
        assert extended['static'] is prefix['static']
        for name in ('operators','harmonics','bindings'):assert extended[name][0] is prefix[name][0]
        assert _digest(encode(prefix['harmonics'][0],sources=old.sources))==before
        initial=torch.cat((old_state.cell_electric,torch.zeros_like(old_state.cell_electric[:1])))
        solution=model.solve(initial,pump_scale=.5,relative_tolerance=1e-9,absolute_tolerance=1e-10)
        state=recover_harmonic_state(model,extended,solution)
    assert state.receipt['solver']['converged'] and len(state.frequencies)==3
    assert state.cell_electric[2].norm()>0
    payload,owner=snapshot_mixed_tensor_harmonic_bundle(state,extended)
    path=tmp_path/'extended.pt';torch.save(dict(payload=payload,owner=owner),path)
    saved=torch.load(path,weights_only=True)
    _,restored,again=restore_mixed_tensor_harmonic_bundle(saved['payload'],expected_binding=saved['owner'],device='cuda')
    torch.testing.assert_close(again.cell_electric,state.cell_electric,rtol=0,atol=0)
    assert snapshot_mixed_tensor_harmonic_bundle(again,restored)[1]==owner
    assert not state.receipt['physical_qualification']
