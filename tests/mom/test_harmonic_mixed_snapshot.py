"""Mixed numerical records restore operators without assembling or solving."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.static_mixed import MixedStaticOperator
from witwin.maxwell.mom.harmonic_mixed import MixedHarmonicClosure
from witwin.maxwell.mom.harmonic_mixed_snapshot import encode,restore
from witwin.maxwell.mom.harmonic_snapshot import tensors

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_mixed_records_detached_restore_and_corruption_guards(monkeypatch):
    laplace,bindings=runpy.run_path('tests/mom/test_static_mixed.py')['setup']()
    bindings=bindings[::-1]
    materials=tuple(compile_harmonic_material(m,3e8,2) for m in (
        mw.Material(eps_r=2.,sigma_e=.001),mw.Material(eps_r=3.)))
    operator=MixedStaticOperator(laplace,EPSILON_0,[None,3*EPSILON_0])
    closure=MixedHarmonicClosure(operator,bindings,{1:.001})
    channels=tuple((m.electric_dc.kind,m.electric_dc.finite_relative_part,m.electric_dc.conductivity,
                    m.electric_dc.plasma_angular_frequency_squared) for m in materials)
    static={'current_closure':closure,'mixed_operator':operator,'floating_operator':laplace,
            'material_channels':channels,'magnetic_permeabilities':(MU_0,(MU_0,MU_0)),
            'points':torch.cat([b.volume.vertices[b.volume.cells].mean(1) for b in bindings]),
            'receipt':{'physical_qualification':False}}
    encoded=encode(static)
    def primitive(value):
        if isinstance(value,dict):
            return all(primitive(item) for item in value.values())
        if isinstance(value,(tuple,list)):
            return all(primitive(item) for item in value)
        return isinstance(value,(torch.Tensor,bool,int,float,complex,str)) or value is None
    assert primitive(encoded)
    sample=laplace.triangles
    source=sample.new_tensor([[.1,-.2,.3]])*EPSILON_0
    boundary=sample.new_tensor([.2,.1,-.1,.3])
    original=operator.solve(closure.dielectric_bindings,[source],[boundary])
    def forbidden(*args,**kwargs):
        raise AssertionError('Mixed record restoration must not assemble, factor or solve')
    with monkeypatch.context() as patch:
        for name in ('solve','lu_factor','lu_solve'):
            patch.setattr(torch.linalg,name,forbidden)
        patch.setattr(MixedStaticOperator,'__init__',forbidden)
        patch.setattr(MixedHarmonicClosure,'__init__',forbidden)
        restored=restore(tensors(encoded,sample.device),bindings,materials,4)
        restored_closure=restored['current_closure']
        restored_closure.assert_current()
        for name in ('electric_response','surface_response','offset_response'):
            torch.testing.assert_close(restored_closure.maps[name],closure.maps[name],rtol=0,atol=0)
        for kind in ('indices','lu','surface_map'):
            bad=tensors(encoded,sample.device)
            if kind=='indices':
                bad['mixed_closure']['current_indices'][0]=0
            elif kind=='lu':
                bad['mixed_operator']['lu'][0,0]+=1
            else:
                bad['mixed_closure']['maps']['surface_response'][0,0]+=1
            with pytest.raises(ValueError,match='Mixed snapshot cell index|Inconsistent mixed snapshot|violates neutral current charge'):
                restore(bad,bindings,materials,4)
    response=restored['mixed_operator'].solve(restored_closure.dielectric_bindings,[source],[boundary])
    torch.testing.assert_close(response.density,original.density,rtol=0,atol=0)
    torch.testing.assert_close(response.offsets,original.offsets,rtol=0,atol=0)
