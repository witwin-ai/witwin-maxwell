"""Analytic backgrounds are added once and retain native source-scale gradients."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.compressed_background import prepare_compressed_background
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.lifting_fields import decode_system
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_source_bank_dense_background_coherence_and_native_scales():
    dense,bindings,inverse=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']()
    sources=[mw.PlaneWave(source_time=mw.CW(3e8,amplitude=1.2,phase=.2)),
        mw.PlaneWave(direction=(1,0,0),polarization=(0,1,0),source_time=mw.CW(3e8,amplitude=.8,phase=-.3))]
    bank=prepare_compressed_background(inverse,bindings,sources,edge_batch=4,query_batch=1)
    references=[];surfaces=[]
    for source in sources:
        op=SystemOperator(dense.system,dense.exterior,[source],dense._public_config)
        surface=torch.linalg.solve(op.matrix,op.build_rhs())
        ee=[];hh=[]
        for binding in bindings:
            points=binding.volume.vertices[binding.volume.cells].mean(1)
            e,h=decode_system(op,surface,points,binding.volume.region_id,'total',query_config=FieldQueryConfig())
            ee.append(e);hh.append(h)
        references.append((torch.cat(ee),torch.cat(hh)));surfaces.append(surface)
    e=torch.stack([x[0] for x in references]);h=torch.stack([x[1] for x in references]);s=torch.stack(surfaces)
    for actual,expected in ((bank.electric,e),(bank.magnetic,h),(bank.surface,s)):
        torch.testing.assert_close(actual,expected,rtol=2e-8,atol=1e-10)
    scales=torch.tensor([[.7+.1j,-.4j],[0.,1.]],device='cuda',dtype=inverse.dtype,requires_grad=True)
    value=bank.combine(scales)
    expected=torch.einsum('bs,sqc->bqc',scales,e)
    torch.testing.assert_close(value['background_electric'],expected,rtol=2e-8,atol=1e-10)
    actual_gradient,=torch.autograd.grad(value['background_electric'].abs().square().sum(),scales)
    expected_gradient,=torch.autograd.grad(expected.abs().square().sum(),scales)
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=3e-8,atol=1e-10)
    assert bank.receipt['converged'] and bank.receipt['incident_background_included']
    assert not bank.receipt['physical_qualification']
    assert all(r['converged'] for r in bank.receipt['field_queries'])
    with torch.no_grad():bank.electric[0,0,0]+=1
    with pytest.raises(ValueError,match='changed'):bank.combine(scales)


def test_empty_harmonic_bank_has_no_surface_solve(monkeypatch):
    _,bindings,inverse=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']()
    def forbidden(*args,**kwargs):raise AssertionError('Unpumped source bank must not solve')
    monkeypatch.setattr(inverse,'apply',forbidden)
    bank=prepare_compressed_background(inverse,bindings,())
    result=bank.combine(torch.empty(0,device='cuda',dtype=inverse.dtype))
    assert result['background_electric'].shape==(2,3)
    assert not bool(result['background_electric'].any())
    assert not bool(result['background_surface'].any())
    for scales in (torch.ones(1,device='cuda',dtype=inverse.dtype),torch.empty(0,dtype=inverse.dtype)):
        with pytest.raises(ValueError,match='scales'):bank.combine(scales)
