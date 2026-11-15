"""Names of a public namespace that are imported on first access (PEP 562).

A namespace's ``__all__`` lists its public contract. The names it collects from other
implementation modules are imported when first read, because those modules import the
namespace module themselves (``array`` imports ``antenna``, ``simulation`` imports ``fdtd``);
the top level loads its domain namespaces the same way. Anything not listed raises
``AttributeError``.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping

_PACKAGE = "witwin.maxwell"


def lazy_attributes(module_name: str, *, names: Mapping[str, Iterable[str]] = {},
                    modules: Mapping[str, str] = {}):
    """``(__getattr__, __dir__)`` for ``module_name``.

    ``names`` maps a module path relative to ``witwin.maxwell`` (``".simulation"``) to the
    names read from it; ``modules`` maps an attribute name to the module path bound to it
    (``{"fit": ".dispersion_fit"}``). A resolved attribute is stored on the module.
    """
    module = sys.modules[module_name]
    origin = {name: path for path, group in names.items() for name in group}
    submodules = dict(modules)

    def __getattr__(name: str):
        if name in origin:
            value = getattr(importlib.import_module(origin[name], _PACKAGE), name)
        elif name in submodules:
            value = importlib.import_module(submodules[name], _PACKAGE)
        else:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        setattr(module, name, value)
        return value

    def __dir__():
        return sorted({*vars(module), *origin, *submodules})

    return __getattr__, __dir__
