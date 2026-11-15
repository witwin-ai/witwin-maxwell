"""Simulation and the adjoint input census read their scene strictly.

``Simulation.scene`` and the scene the adjoint input collectors take are the declarative ``Scene``. A wrong
object, such as a ``PreparedScene`` (its declarations are ``prepared.scene``) or any stand-in, must raise
``AttributeError`` naming the field instead of silently reading as "no networks", "no ports" or "no wires".
A ``Scene`` whose optional fields are ``None`` (no background, no coordinate chart) keeps working.
"""
from types import SimpleNamespace

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell import adjoint_inputs
from witwin.maxwell.fdtd.plane_product import _require_uniform_nonmagnetic
from witwin.maxwell.scene import prepare_scene


def _scene(**fields):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=1),
        **fields,
    )


def _wrong_scenes():
    return {
        "prepared_scene": prepare_scene(_scene(), device="cpu"),
        "plain_object": SimpleNamespace(),
    }


class _Module(mw.SceneModule):
    def __init__(self, scene):
        super().__init__()
        self._scene = scene

    def to_scene(self):
        return self._scene


@pytest.mark.parametrize("kind", ["prepared_scene", "plain_object"])
def test_simulation_construction_rejects_a_wrong_scene_from_a_scene_module(kind):
    with pytest.raises(AttributeError, match="networks"):
        mw.Simulation.fdtd(_Module(_wrong_scenes()[kind]), frequencies=1.0e9, config=mw.FDTDConfig(device="cpu"))


@pytest.mark.parametrize("kind", ["prepared_scene", "plain_object"])
@pytest.mark.parametrize(
    ("collector", "field"),
    [
        (adjoint_inputs.scene_trainable_rf_tensors, "ports"),
        (adjoint_inputs.scene_trainable_wire_tensors, "thin_wires"),
        (adjoint_inputs.named_scene_tensors, "structures"),
        (adjoint_inputs.ungradable_material_channels, "structures"),
    ],
)
def test_adjoint_input_collection_rejects_a_wrong_scene(kind, collector, field):
    scene = _wrong_scenes()[kind]
    with pytest.raises(AttributeError, match=field):
        collector(scene)


def test_native_copper_flux_admission_reads_the_scene_background():
    with pytest.raises(AttributeError, match="background"):
        _require_uniform_nonmagnetic(SimpleNamespace(structures=(), material_regions=()))


def test_a_scene_without_background_or_coordinates_keeps_working():
    scene = _scene()
    assert scene.background is None and scene.coordinates is None
    simulation = mw.Simulation.fdtd(scene, frequencies=1.0e9, config=mw.FDTDConfig(device="cpu"))
    assert not simulation.has_trainable_parameters
    assert simulation._collect_fdtd_requested_frequencies() == (1.0e9,)
    assert adjoint_inputs.scene_trainable_rf_tensors(scene) == ()
    assert adjoint_inputs.scene_trainable_wire_tensors(scene) == ()
    assert adjoint_inputs.named_scene_tensors(scene) == ()
    assert adjoint_inputs.ungradable_material_channels(scene) == ()
    _require_uniform_nonmagnetic(scene)


def test_a_trainable_scene_without_background_is_still_censused():
    wire = mw.rf.ThinWire(name="wire", points=((0.0, 0.0, -0.01), (0.0, 0.0, 0.01)),
                          radius=torch.tensor(0.002, requires_grad=True), conductor=mw.rf.WireConductor.pec())
    scene = _scene(thin_wires=(wire,))
    assert wire.radius.requires_grad
    assert adjoint_inputs.scene_trainable_wire_tensors(scene) == (wire.radius,)
    assert adjoint_inputs.ungradable_material_channels(scene) == ()
