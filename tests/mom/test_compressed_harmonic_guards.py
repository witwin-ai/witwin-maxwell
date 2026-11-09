"""Compressed preparation prefix reuse and decode-only mutation guards."""
import copy
import runpy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.surface_volume import VolumeFieldResponse
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalH2SystemOperator
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance
from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeHConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def config(integration):
    return mw.MoMConfig(operator_backend='h2',integration=integration,
        hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))


def test_pump_scaled_decode_view_and_dense_matrix_mutation_guards():
    dense,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    physical=HierarchicalH2SystemOperator(dense.system,dense.exterior,dense.sources,config(dense.integration))
    handle=copy.copy(physical)
    handle.body_handles=tuple(copy.copy(h) for h in physical.body_handles)
    for h in (handle,*handle.body_handles):
        h.incident_carrier=replace(h.incident_carrier,waves=tuple(replace(w,
            amplitude=tuple(.5*x for x in w.amplitude)) for w in h.incident_carrier.waves))
    currents=[torch.zeros(len(b.volume.cells),3,device='cuda',dtype=dense.dtype) for b in bindings]
    coefficients=torch.zeros(dense.shape[0],device='cuda',dtype=dense.dtype)
    view=VolumeFieldResponse(handle,bindings,currents,coefficients)
    points=bindings[0].volume.vertices[:1]
    for name,value in [('integration',replace(handle.integration,order=4)),('incident_carrier',None)]:
        local=handle.body_handles[0];previous=getattr(local,name);setattr(local,name,value)
        with pytest.raises(RuntimeError,match='mutated'):view.fields(points,0)
        setattr(local,name,previous)
    view=VolumeFieldResponse(dense,bindings,currents,coefficients)
    with torch.no_grad():dense.matrix.mul_(1.01)
    with pytest.raises(RuntimeError,match='mutated'):view.fields(points,0)


def test_actual_compressed_prefix_only_prepares_appended_harmonic(monkeypatch):
    dense,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    volumes=[b.volume for b in bindings]
    public=[mw.Material(eps_r=e,nonlinearity=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)) for e in (2.,3.)]
    def materials(n):return [compile_harmonic_material(m,3e8,n) for m in public],compile_harmonic_material(mw.Material(),3e8,n)
    def operator(n,ms,ex):
        system=SurfaceSystem.create([(b.structure_index,b.name,b.basis,m.linear_samples[n]) for b,m in zip(dense.system.bodies,ms)])
        return HierarchicalH2SystemOperator(system,ex.linear_samples[n],dense.sources if n==0 else (),config(dense.integration))
    options=dict(source_config=SurfaceVolumeHConfig(leaf_size=64),field_config=InteriorVolumeHConfig(leaf_size=64),volume_config=VolumeHConfig(leaf_size=64))
    ms,ex=materials(1);first=operator(0,ms,ex)
    model,prepared=prepare_harmonic_balance([first],volumes,ms,ex,compressed_settings=options)
    ms,ex=materials(2);second=operator(1,ms,ex)
    import witwin.maxwell.mom.harmonic_preparation as module
    original=module._prepare_compressed_harmonic;calls=[]
    def only_new(op,*args):
        assert op is second
        calls.append(op)
        return original(op,*args)
    def forbidden(*args,**kwargs):raise AssertionError('Existing DC response must be reused')
    with monkeypatch.context() as patch:
        patch.setattr(module,'_prepare_compressed_harmonic',only_new)
        patch.setattr(module,'prepare_static_cell_response',forbidden)
        extended,result=prepare_harmonic_balance([first,second],volumes,ms,ex,
            compressed_settings=options,reuse_prepared=prepared)
    assert calls==[second] and result['static'] is prepared['static']
    assert result['harmonics'][0] is prepared['harmonics'][0]
    assert not bool(extended.background[2].any())
    solution=extended.solve(pump_scale=.5)
    state=recover_harmonic_state(extended,result,solution)
    assert solution.receipt['converged'] and bool(state.cell_electric[2].abs().max()>0)
    with pytest.raises(ValueError,match='response backend'):
        prepare_harmonic_balance([first,second],volumes,ms,ex,reuse_prepared=prepared)
