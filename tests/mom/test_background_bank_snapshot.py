"""Stored background arrays retain coherent scale gradients without a solve."""
import copy
import io
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.compressed_background import prepare_compressed_background
from witwin.maxwell.mom.background_bank_snapshot import encode,restore
from witwin.maxwell.mom.surface_h_snapshot import encode_inverse,restore_inverse
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('pumped',[False,True])
def test_stored_background_sources_scales_and_bindings(pumped,monkeypatch):
    setup=runpy.run_path('tests/mom/test_compressed_volume_preparation.py')['setup']
    _,bindings,inverse=setup();_,fresh_bindings,fresh_inverse=setup()
    sources=(mw.PlaneWave(source_time=mw.CW(3e8,amplitude=1.2,phase=.2)),) if pumped else ()
    bank=prepare_compressed_background(inverse,bindings,sources)
    payload=encode(bank,sources=sources);saved_inverse=encode_inverse(inverse)
    data=io.BytesIO();torch.save(payload,data);data.seek(0);payload=torch.load(data,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Background restore cannot solve')
    with monkeypatch.context() as patch:
        patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
        loaded_inverse=restore_inverse(saved_inverse,fresh_inverse.operator.provider)
        loaded=restore(payload,loaded_inverse,fresh_bindings,sources=sources)
    assert encode(loaded,sources=sources)['sha256']==payload['sha256']
    scales=torch.ones(2,len(sources),device='cuda',dtype=inverse.dtype,requires_grad=True)
    for name,a in bank.combine(scales).items():
        b=loaded.combine(scales)[name]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga,=torch.autograd.grad(a.abs().square().sum(),scales)
        gb,=torch.autograd.grad(b.abs().square().sum(),scales)
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    # Solver observations can evolve without changing the fixed response binding.
    loaded_inverse.receipt['last_forward']={'historical_test':True}
    restore(payload,loaded_inverse,fresh_bindings,sources=sources)
    with pytest.raises(ValueError,match='binding'):restore(payload,loaded_inverse,fresh_bindings[::-1],sources=sources)
    if pumped:
        with pytest.raises(ValueError,match='carrier'):restore(payload,loaded_inverse,fresh_bindings,sources=(mw.PlaneWave(),))
    bad=copy.deepcopy(payload);bad['receipt']['physical_qualification']=True
    bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='receipt'):restore(bad,loaded_inverse,fresh_bindings,sources=sources)
    if pumped:
        with torch.no_grad():loaded.electric[0,0,0]+=1
        with pytest.raises(ValueError,match='changed'):loaded.combine(scales)
