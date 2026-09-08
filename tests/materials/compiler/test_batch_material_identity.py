"""Published core identity tokens must not invalidate declared design reuse."""
from dataclasses import replace

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _material_definition, _same_material_definition


def test_declared_design_identity_changes_preserve_fixed_material_guards():
    first = mw.Structure(name="design", geometry=mw.Sphere(radius=.02, device="cpu"),
                         material=mw.Material(eps_r=2.))
    second = mw.Structure(name="design", geometry=mw.Sphere(radius=.03, device="cpu"),
                          material=mw.Material(eps_r=3.))
    assert first.structure_id != second.structure_id
    definition = _material_definition(first, structures={"design"}, snapshot=True)
    assert _same_material_definition(definition, _material_definition(second, structures={"design"}))
    assert not _same_material_definition(definition,
        _material_definition(replace(second, priority=2), structures={"design"}))
    assert not _same_material_definition(_material_definition(first), _material_definition(second))
