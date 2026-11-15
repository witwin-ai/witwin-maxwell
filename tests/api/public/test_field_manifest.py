"""Field-consumption manifest: every public scene field states how each method treats it.

Default deny: a public class that is neither a manifest scene class nor a
classified non-scene class fails, and so does a constructor field without a
manifest entry. Every refused cell is backed by a registry row, every other
non-read cell carries a reason, and the open gaps are pinned here.
"""
from __future__ import annotations

import inspect

import pytest

import witwin.core as core
import witwin.maxwell as mw
from witwin.maxwell.capability import Support, check_scene
from witwin.maxwell.capability.fields import (
    ENTRIES, MANIFEST, METHODS, NOT_SCENE, consumption, declared_fields, gaps, row,
)
from witwin.maxwell.capability.keys import SCENE_KEYS

from tests.support.public_api import public_objects

pytestmark = pytest.mark.host_validation

F = 1.0e9
_BY_NAME = {entry.cls.__name__: entry for entry in ENTRIES}

# Open silent ignores, each with its owner. Removing one means the method now reads or refuses the field.
# None is open: a new gap must be pinned here with its owner.
PINNED_GAPS: set[tuple[str, str, str]] = set()


def _public_classes():
    # The analysis subpackages hold no scene objects; every other namespace is covered.
    objects = public_objects(exclude=("capability", "invdes", "postprocess"))
    return {name for name, value in objects.items() if inspect.isclass(value)}


def test_every_public_class_is_classified():
    scene = {cls.__name__ for cls in MANIFEST}
    other = [name for names in NOT_SCENE.values() for name in names]
    assert len(other) == len(set(other))
    assert scene.isdisjoint(other)
    assert _public_classes() - scene - set(other) == set()
    assert (scene | set(other)) - _public_classes() == {"PhysicalMaterial"}
    assert MANIFEST[core.PhysicalMaterial].cls is core.PhysicalMaterial


@pytest.mark.parametrize("cls", tuple(MANIFEST), ids=[cls.__name__ for cls in MANIFEST])
def test_manifest_declares_every_field(cls):
    declared = MANIFEST[cls].fields
    fields = declared_fields(cls)
    assert [name for name in fields if name not in declared] == [], "undeclared public field(s)"
    assert [name for name in declared if name not in fields] == [], "stale manifest field(s)"


def test_manifest_names_registry_vocabulary():
    for entry in ENTRIES:
        label = entry.cls.__name__
        assert entry.owner is None or entry.owner in SCENE_KEYS, label
        for holder in entry.within:
            name, _, field = holder.partition(".")
            assert name in _BY_NAME and (not field or field in _BY_NAME[name].fields), (label, holder)
        for name, use in entry.fields.items():
            assert set(use.keys) | set(use.when) <= set(SCENE_KEYS), (label, name)
            assert set(use.not_read) | set(use.gaps) <= set(METHODS), (label, name)
            assert set(use.not_read).isdisjoint(use.gaps), (label, name)
            assert all(use.not_read.values()) and all(use.gaps.values()), (label, name)
            assert not use.neutral or not (use.keys or use.when or use.not_read or use.gaps), (label, name)


def test_every_scene_key_is_declared_by_a_field():
    declared = {entry.owner for entry in ENTRIES} | {
        key for entry in ENTRIES for use in entry.fields.values() for key in (*use.keys, *use.when)}
    assert [key for key in SCENE_KEYS if key not in declared] == []


@pytest.mark.parametrize("cls", tuple(MANIFEST), ids=[cls.__name__ for cls in MANIFEST])
def test_cells_follow_the_registry(cls):
    entry = MANIFEST[cls]
    for name, cells in consumption(cls).items():
        use = entry.fields[name]
        assert set(cells) == set(METHODS)
        for method, cell in cells.items():
            where = (cls.__name__, name, method)
            if use.neutral:
                assert cell.support is None, where
                continue
            assert cell.support in (Support.SUPPORTED, Support.UNSUPPORTED, Support.INHERENT), where
            if cell.key is not None:
                expected = row(method, cell.key).support
                assert cell.support is expected or (cell.support is Support.SUPPORTED and use.when), where
            else:
                # Not read or a gap: always explained, never a silent refusal.
                assert cell.support is Support.SUPPORTED or cell.reason, where
            assert all(row(method, key).support is not Support.SUPPORTED for key in cell.conditional), where


def test_not_read_and_gaps_are_declared_only_where_the_method_sees_the_class():
    for entry in ENTRIES:
        for name, use in entry.fields.items():
            for method in (*use.not_read, *use.gaps):
                owner = None if entry.owner is None else row(method, entry.owner)
                assert owner is None or owner.support is Support.SUPPORTED, (entry.cls.__name__, name, method)


def test_open_gaps_are_pinned():
    assert set(gaps()) == PINNED_GAPS
    for cls_name, name, method in PINNED_GAPS:
        cell = consumption(_BY_NAME[cls_name].cls)[name][method]
        assert cell.gap and cell.support is Support.UNSUPPORTED


@pytest.mark.parametrize("cls, name, key", [
    (mw.Material, "kerr_chi3", "material.nonlinear"), (mw.Material, "nonlinearity", "material.nonlinear"),
    (mw.Material, "breakdown", "material.breakdown"),
    *((mw.media.PerturbationMedium, name, "material.perturbation") for name in (
        "perturbation", "eps_sensitivity", "debye_sensitivities", "drude_sensitivities", "lorentz_sensitivities")),
], ids=lambda value: getattr(value, "__name__", value))
def test_static_methods_refuse_the_fields_they_would_drop(cls, name, key):
    cells = consumption(cls)[name]
    for method in ("electrostatic", "capacitance"):
        cell = cells[method]
        assert not cell.gap and cell.key == key and cell.reason, (method, cell)
        assert cell.support is not Support.SUPPORTED, (method, cell)
    assert cells["fdtd"].support is Support.SUPPORTED


def test_consumption_reads_the_registry_through_containers():
    plane_wave = consumption(mw.PlaneWave)
    assert plane_wave["direction"]["fdtd"].support is Support.SUPPORTED
    assert (plane_wave["direction"]["electrostatic"].support, plane_wave["direction"]["electrostatic"].key) == (
        Support.INHERENT, "source")
    assert plane_wave["injection"]["mom"].support is Support.INHERENT
    assert plane_wave["injection"]["mom"].conditional == ("source.injection_tfsf",)
    assert plane_wave["name"]["fdtd"].support is None
    material = consumption(mw.Material)
    assert material["epsilon_tensor"]["fdfd"].support is Support.SUPPORTED
    assert material["epsilon_tensor"]["fdfd"].conditional == ("material.full_tensor",)
    assert material["modulation"]["fdfd"].support is Support.INHERENT
    assert material["modulation"]["fdfd"].key == "material.modulation"
    assert material["mu_r"]["electrostatic"].support is Support.INHERENT
    assert material["mu_r"]["electrostatic"].key is None
    # A nested value is seen only where its container is read.
    assert consumption(mw.media.ModulationSpec)["amplitude"]["fdfd"].key == "material.modulation"
    assert consumption(mw.GridSpec)["dx"]["mom"].support is Support.INHERENT
    assert consumption(mw.TFSF)["bounds"]["electrostatic"].key == "source"
    # The electrostatic tables admit a LayerStack and grid preparation refuses every background at runtime.
    layers = consumption(mw.mom.LayerStack)["layers"]
    assert {method for method, cell in layers.items() if cell.support is Support.SUPPORTED} == {
        "mom.layered", "electrostatic", "capacitance"}
    assert layers["mom"].key == "background.layer_stack"


def _core_material_scene(**fields):
    material = core.PhysicalMaterial(eps_r=4.0, **fields)
    return mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.1),
                    structures=[mw.Structure(name="body", geometry=mw.Box(position=(0, 0, 0), size=(0.2, 0.2, 0.2)),
                                             material=material)])


@pytest.mark.parametrize("fields", ({"conductor_model": "perfect"}, {"thickness_m": 1e-3},
                                    {"scattering_coefficient": 0.3}))
def test_grid_solvers_refuse_core_surface_fields(fields):
    scene = _core_material_scene(**fields)
    (name,) = fields
    for method, solver in (("fdtd", "FDTD"), ("fdfd", "FDFD")):
        (item,) = check_scene(scene, method, frequencies=(F,)).unsupported
        assert item.key == "material.core_surface_fields"
        assert str(item).startswith(f"Structure 'body': core material field(s) {name} are not modelled by {solver}")
    with pytest.raises(mw.UnsupportedFeatureError, match=f"field\\(s\\) {name} are not modelled by FDFD"):
        mw.Simulation.fdfd(scene, frequencies=F)
    assert check_scene(_core_material_scene(), "fdtd").unsupported == ()
