"""Complete native material admission reports before any AC interface work."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material,compile_tensor_harmonic_material
from witwin.maxwell.mom.tensor_harmonic_preparation import admit_mixed_dc_channels,MixedHarmonicDCAdmissionError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_all_material_channel_failures_preserved_without_scalarization():
    tensor=mw.Material(epsilon_tensor=mw.ComplexTensor3x3([[2+.1j,0,0],[0,3+.2j,0],[0,0,4+.3j]]),
                       mu_tensor=mw.ComplexTensor3x3([[1+.01j,0,0],[0,2+.02j,0],[0,0,3+.03j]]))
    materials=(compile_tensor_harmonic_material(tensor,3e8,2,device='cuda'),
               compile_harmonic_material(mw.Material(eps_r=2.,sigma_e=.001,sigma_m=.01,
                   nonlinearity=mw.TwoPhotonAbsorption(beta=5.)),3e8,2))
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    before=materials[0].electric_dc.finite_relative_part.clone()
    with pytest.raises(MixedHarmonicDCAdmissionError) as caught:
        admit_mixed_dc_channels(materials,exterior,(2,),device='cuda',dtype=torch.float64)
    rows=caught.value.failures
    assert [(r['region'],r['channel'],r['kind']) for r in rows]==[
        (1,'electric','unresolved_complex_dc'),(1,'magnetic','unresolved_complex_dc'),(2,'magnetic','conductive_limit')]
    assert all(r['reason'] in str(caught.value) for r in rows)
    torch.testing.assert_close(materials[0].electric_dc.finite_relative_part,before,rtol=0,atol=0)


def test_tensor_dc_current_opt_in_preserves_physical_sample():
    material=compile_tensor_harmonic_material(mw.Material(sigma_e_tensor=mw.DiagonalTensor3(.001,.002,.0015)),3e8,2,device='cuda')
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    with pytest.raises(MixedHarmonicDCAdmissionError,match='explicit internal'):
        admit_mixed_dc_channels((material,),exterior,(1,),device='cuda',dtype=torch.float64)
    electric,current=admit_mixed_dc_channels((material,),exterior,(1,),device='cuda',dtype=torch.float64,tensor_current=True)
    assert electric[1] is None
    torch.testing.assert_close(current[1],material.electric_dc.conductivity,rtol=0,atol=0)
    with pytest.raises(ValueError,match='explicit volume'):
        admit_mixed_dc_channels((material,),exterior,(),device='cuda',dtype=torch.float64,tensor_current=True)

    with pytest.raises(TypeError,match='explicit boolean'):
        admit_mixed_dc_channels((material,),exterior,(1,),device='cuda',dtype=torch.float64,tensor_current=1)
