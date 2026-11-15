"""The public names of ``witwin.maxwell``: the top level and every namespace's ``__all__``."""
from __future__ import annotations

import types

import witwin.maxwell as mw


def public_objects(*, exclude: tuple[str, ...] = ()) -> dict[str, object]:
    """Public object by name, from the top level and the namespaces (``mw.rf``, ``mw.rf.rational``, ...).

    ``exclude`` names top-level namespaces to skip. Each name has one home, so the mapping is
    unambiguous; sub-namespace modules themselves are not listed.
    """
    found = {name: getattr(mw, name) for name in mw.__all__}

    def collect(module: types.ModuleType) -> None:
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                collect(value)
            else:
                assert name not in found, f"{name} has two public homes"
                found[name] = value

    for namespace in mw._NAMESPACES:
        if namespace not in exclude:
            collect(getattr(mw, namespace))
    return found
