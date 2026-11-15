"""Non-trainable conductor-loss inputs construct exactly as before the layered gradients.

LossyMetalMedium, the roughness descriptors and PlanarPEC keep a tensor only when
it requires grad (layered MoM loss gradients). Every other input is stored as a
Python float with the value float(input), and the validation messages are unchanged.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.host_validation

INPUTS = {
    "float": lambda v: float(v),
    "int": lambda v: int(v),
    "numpy": lambda v: np.float64(v),
    "tensor64": lambda v: torch.tensor(v, dtype=torch.float64),
    "tensor32": lambda v: torch.tensor(v, dtype=torch.float32),
    "tensor64_1d": lambda v: torch.tensor([v], dtype=torch.float64),
}

FIELDS = {
    "LossyMetalMedium.conductivity": (58000000, lambda v: mw.media.LossyMetalMedium(conductivity=v).conductivity),
    "HammerstadRoughness.rms_height": (1e-6, lambda v: mw.media.HammerstadRoughness(rms_height=v).rms_height),
    "HammerstadRoughness.roughness_factor": (
        3, lambda v: mw.media.HammerstadRoughness(rms_height=1e-6, roughness_factor=v).roughness_factor),
    "HurayRoughness.sphere_radius": (
        5e-7, lambda v: mw.media.HurayRoughness(sphere_radius=v, sphere_density=1e11).sphere_radius),
    "HurayRoughness.sphere_density": (
        1e11, lambda v: mw.media.HurayRoughness(sphere_radius=5e-7, sphere_density=v).sphere_density),
    "HurayRoughness.relative_area": (
        2, lambda v: mw.media.HurayRoughness(sphere_radius=5e-7, sphere_density=1e11, relative_area=v).relative_area),
    "PlanarPEC.physical_thickness": (
        18e-6, lambda v: mw.geometry.PlanarPEC.rectangle(((0., 1.), (0., 1.)), physical_thickness=v).physical_thickness),
    "PlanarPEC.coordinate": (
        2e-3, lambda v: mw.geometry.PlanarPEC.rectangle(((0., 1.), (0., 1.)), coordinate=v).coordinate),
}


@pytest.mark.parametrize("kind", tuple(INPUTS))
@pytest.mark.parametrize("field", tuple(FIELDS))
def test_fixed_inputs_are_stored_as_python_floats(field, kind):
    nominal, construct = FIELDS[field]
    if kind == "int" and nominal != int(nominal):
        pytest.skip("integer input needs an integral nominal value")
    value = INPUTS[kind](nominal)
    stored = construct(value)
    assert type(stored) is float
    assert stored == float(value)


def test_fixed_inputs_keep_roughness_hash_and_messages():
    assert hash(mw.media.HammerstadRoughness(rms_height=1e-6)) == hash(
        mw.media.HammerstadRoughness(rms_height=torch.tensor(1e-6, dtype=torch.float64)))
    with pytest.raises(ValueError, match=r"^conductivity must be > 0\.$"):
        mw.media.LossyMetalMedium(conductivity=torch.tensor(-1.0))
    with pytest.raises(ValueError, match=r"^rms_height must be finite and > 0\.$"):
        mw.media.HammerstadRoughness(rms_height=float("nan"))
    with pytest.raises(ValueError, match=r"^roughness_factor must be finite and > 1\.$"):
        mw.media.HammerstadRoughness(rms_height=1e-6, roughness_factor=1.0)
    with pytest.raises(ValueError, match=r"^sphere_density must be finite and > 0\.$"):
        mw.media.HurayRoughness(sphere_radius=5e-7, sphere_density=0.0)
    with pytest.raises(ValueError, match=r"^PlanarPEC.physical_thickness must be finite and nonnegative\.$"):
        mw.geometry.PlanarPEC.rectangle(((0., 1.), (0., 1.)), physical_thickness=torch.tensor(-1.0))
