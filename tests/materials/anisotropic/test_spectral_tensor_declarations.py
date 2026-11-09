"""Frequency-only declarations must not enter a scalar or static Yee compiler."""
import pytest
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('rows', [((1, 0), (0, 1)), ((float('nan'), 0, 0), (0, 1, 0), (0, 0, 1))])
def test_complex_descriptor_rejects_malformed_or_nonfinite(rows):
    with pytest.raises(ValueError):
        mw.ComplexTensor3x3(rows)


def test_complex_descriptor_preserves_values_and_explicit_reciprocity():
    rows = ((2+.3j, .1j, 0), (-.1j, 3+.4j, 0), (0, 0, 4+.5j))
    value = mw.ComplexTensor3x3(rows, reciprocal=False)
    assert value.rows == rows and value.reciprocal is False
    with pytest.raises(TypeError, match='boolean'):
        mw.ComplexTensor3x3(rows, reciprocal=0)
    with pytest.raises(TypeError, match='conductivity'):
        mw.Material(sigma_e_tensor=value)


@pytest.mark.parametrize('name,descriptor', [
    ('epsilon_tensor', mw.ComplexTensor3x3(((2+.1j, 0, 0), (0, 3+.2j, 0), (0, 0, 4+.3j)))),
    ('mu_tensor', mw.ComplexTensor3x3(((2+.1j, 0, 0), (0, 3+.2j, 0), (0, 0, 4+.3j)))),
    ('mu_tensor', mw.Tensor3x3(((2, .1, 0), (.1, 3, 0), (0, 0, 4)))),
    ('sigma_e_tensor', mw.Tensor3x3(((.002, .001, 0), (.001, .002, 0), (0, 0, 0))))])
def test_spectral_tensor_rejected_by_static_and_scalar_frequency_compilers(name, descriptor):
    material = mw.Material(**{name: descriptor})
    assert material.is_anisotropic and material.requires_tensor_spectral_compiler
    with pytest.raises(NotImplementedError, match='tensor-aware spectral'):
        material.evaluate_static()
    with pytest.raises(NotImplementedError, match='tensor-aware spectral'):
        material.evaluate_at_frequency(3e8)
    scene = mw.Scene(domain=mw.Domain(bounds=((-1, 1),)*3), grid=mw.GridSpec.uniform(.2),
        structures=[mw.Structure(mw.Box(size=(.4, .4, .4)), material)], device='cpu')
    with pytest.raises(NotImplementedError, match='tensor-aware spectral'):
        prepare_scene(scene).compile_materials()
