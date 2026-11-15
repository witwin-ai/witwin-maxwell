"""Conductor-loss gradient fields are a layered MoM capability only.

LossyMetalMedium.conductivity, every SurfaceRoughness field and
PlanarPEC.physical_thickness keep a requires_grad tensor for the layered MoM
native adjoint. Every other Simulation path reads them as fixed values, so the
capability registry row ``structure.trainable_conductor_loss`` refuses a
trainable value there by field name at construction. Host-only: the refusal
precedes any device work.
"""

import pytest
import torch

import witwin.maxwell as mw
from tests.support.layered_gradient_fixtures import FREQUENCIES, loop_config, loop_scene

pytestmark = pytest.mark.host_validation

FIELDS = (
    "LossyMetalMedium.conductivity",
    "HammerstadRoughness.rms_height",
    "HammerstadRoughness.roughness_factor",
    "HurayRoughness.sphere_radius",
    "HurayRoughness.sphere_density",
    "HurayRoughness.relative_area",
    "PlanarPEC.physical_thickness",
)

NOMINAL = {
    "LossyMetalMedium.conductivity": 5.8e7,
    "HammerstadRoughness.rms_height": 0.5e-6,
    "HammerstadRoughness.roughness_factor": 2.0,
    "HurayRoughness.sphere_radius": 0.5e-6,
    "HurayRoughness.sphere_density": 1e11,
    "HurayRoughness.relative_area": 1.0,
    "PlanarPEC.physical_thickness": 18e-6,
}


def _value(field, trained):
    value = NOMINAL[field]
    return torch.tensor(value, dtype=torch.float64, requires_grad=True) if field == trained else value


def _structure(trained):
    """One lossy PlanarPEC sheet whose ``trained`` field is a requires_grad tensor."""
    if trained.startswith("HurayRoughness"):
        roughness = mw.media.HurayRoughness(
            sphere_radius=_value("HurayRoughness.sphere_radius", trained),
            sphere_density=_value("HurayRoughness.sphere_density", trained),
            relative_area=_value("HurayRoughness.relative_area", trained))
    else:
        roughness = mw.media.HammerstadRoughness(
            rms_height=_value("HammerstadRoughness.rms_height", trained),
            roughness_factor=_value("HammerstadRoughness.roughness_factor", trained))
    material = mw.media.LossyMetalMedium(
        conductivity=_value("LossyMetalMedium.conductivity", trained), roughness=roughness)
    geometry = mw.geometry.PlanarPEC.rectangle(
        ((-1e-3, 1e-3), (-1e-3, 1e-3)), normal_axis="z", coordinate=0.0,
        physical_thickness=_value("PlanarPEC.physical_thickness", trained))
    return mw.Structure(geometry, material)


def _scene(trained):
    return mw.Scene(
        domain=mw.Domain(bounds=((-2e-3, 2e-3),) * 3), grid=mw.GridSpec.uniform(0.5e-3),
        boundary=mw.BoundarySpec.none(), structures=[_structure(trained)])


METHODS = {
    "fdtd": lambda scene: mw.Simulation.fdtd(scene, frequencies=[1e9]),
    "fdfd": lambda scene: mw.Simulation.fdfd(scene, frequencies=1e9),
    "mom": lambda scene: mw.Simulation.mom(scene, frequencies=1e9),
}


def _refusal(field):
    return rf"Gradients for {field} are supported on layered MoM only"


@pytest.mark.parametrize("method", tuple(METHODS))
@pytest.mark.parametrize("field", FIELDS)
def test_non_layered_paths_refuse_trainable_conductor_loss_fields(field, method):
    with pytest.raises(mw.UnsupportedFeatureError, match=_refusal(field)):
        METHODS[method](_scene(field))


@pytest.mark.parametrize("field", FIELDS)
def test_fdtd_refuses_trainable_conductor_loss_fields_from_a_scene_module(field):
    class Module(mw.SceneModule):
        def to_scene(self):
            return _scene(field)

    with pytest.raises(mw.UnsupportedFeatureError, match=_refusal(field)):
        mw.Simulation.fdtd(Module(), frequencies=[1e9])


def test_fixed_conductor_loss_fields_are_not_refused():
    """Floats and detached tensors are fixed values: FDTD constructs, the others keep their scene guard."""
    sheet = _structure("none")
    detached = _scene("none").updated(structures=[mw.Structure(
        sheet.geometry, mw.media.LossyMetalMedium(conductivity=torch.tensor(5.8e7)))])
    for scene in (_scene("none"), detached):
        assert not mw.Simulation.fdtd(scene, frequencies=[1e9]).has_trainable_parameters
        for method in ("fdfd", "mom"):
            with pytest.raises(NotImplementedError, match="PlanarPEC requires the FDTD sheet runtime"):
                METHODS[method](scene)


def test_layered_mom_admits_trainable_conductor_loss_fields():
    sigma = torch.tensor(5.8e7, dtype=torch.float64, requires_grad=True)
    scene = loop_scene(sheet_sigma=sigma, via_sigma=sigma * 0.5)
    simulation = mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config("sheet"))
    assert simulation.has_trainable_parameters
