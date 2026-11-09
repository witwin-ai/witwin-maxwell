"""Tidy3D export of the opt-in conductor surface-roughness models on LossyMetalMedium."""

from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest

import witwin.maxwell as mw
from witwin.maxwell.adapters.tidy3d import _convert_material

_LENGTH_SCALE = 1.0e6  # m -> um


class _Recorder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.kwargs = kwargs


def _mock_td():
    td = types.SimpleNamespace()
    for name in ("LossyMetalMedium", "HammerstadSurfaceRoughness", "HuraySurfaceRoughness"):
        setattr(td, name, type(name, (_Recorder,), {}))
    return td


def test_smooth_lossy_metal_export_carries_no_roughness_argument():
    td = _mock_td()
    result = _convert_material(mw.LossyMetalMedium(conductivity=1.0e7), td, _LENGTH_SCALE, frequencies=(1e10,))
    assert "roughness" not in result.kwargs


def test_hammerstad_exports_rq_in_micrometres():
    td = _mock_td()
    medium = mw.LossyMetalMedium(
        conductivity=5.8e7, roughness=mw.HammerstadRoughness(rms_height=1.2e-6, roughness_factor=2.5)
    )
    result = _convert_material(medium, td, _LENGTH_SCALE, frequencies=(1e9, 2e10))
    assert isinstance(result.roughness, td.HammerstadSurfaceRoughness)
    assert result.roughness.rq == pytest.approx(1.2)
    assert result.roughness.roughness_factor == pytest.approx(2.5)


def test_huray_exports_surface_ratio_and_radius_pairs():
    td = _mock_td()
    roughness = mw.HurayRoughness(sphere_radius=0.5e-6, sphere_density=2.0e11, relative_area=1.2)
    medium = mw.LossyMetalMedium(conductivity=5.8e7, roughness=roughness)
    result = _convert_material(medium, td, _LENGTH_SCALE, frequencies=(1e9, 2e10))
    assert isinstance(result.roughness, td.HuraySurfaceRoughness)
    assert result.roughness.relative_area == pytest.approx(1.2)
    ((ratio, radius),) = result.roughness.coeffs
    assert ratio == pytest.approx(roughness.surface_ratio)
    assert radius == pytest.approx(0.5)


def test_groisse_export_is_rejected():
    td = _mock_td()
    medium = mw.LossyMetalMedium(conductivity=5.8e7, roughness=mw.GroisseRoughness(rms_height=1e-6))
    with pytest.raises(ValueError, match="Groisse"):
        _convert_material(medium, td, _LENGTH_SCALE, frequencies=(1e9, 2e10))


@pytest.mark.skipif(importlib.util.find_spec("tidy3d") is None, reason="requires the real tidy3d package")
@pytest.mark.parametrize(
    "roughness",
    [
        mw.HammerstadRoughness(rms_height=1.0e-6),
        mw.HammerstadRoughness(rms_height=0.4e-6, roughness_factor=3.0),
        mw.HurayRoughness.cannonball(0.5e-6),
        mw.HurayRoughness(sphere_radius=0.4e-6, sphere_density=1.5e11, relative_area=1.3),
    ],
)
def test_rough_surface_impedance_matches_real_tidy3d(roughness):
    import tidy3d as td

    medium = mw.LossyMetalMedium(conductivity=5.8e7, roughness=roughness)
    result = _convert_material(medium, td, frequencies=(1e9, 4e10))
    assert isinstance(result, td.LossyMetalMedium)
    frequencies = np.logspace(9, np.log10(4e10), 9)
    z_reference = np.asarray(result.surface_impedance(frequencies), dtype=complex)
    z_maxwell = np.array([medium.surface_impedance_at_freq(float(f)) for f in frequencies])
    # The reference uses the full conductor index sqrt(1 + i sigma/(omega eps0)) instead of
    # the Leontovich good-conductor limit; the difference is O(omega eps0 / sigma) ~ 1e-8.
    np.testing.assert_allclose(z_maxwell, z_reference, rtol=1e-6)
