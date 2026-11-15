"""Default-deny admission of a scene against one method table."""
from __future__ import annotations

from typing import Mapping

from .keys import scene_features
from .model import Capability, CapabilityReport, Support, Unsupported
from .tables import ENTRY_POINTS, TABLES


def _method_name(method) -> str:
    name = str(getattr(method, "value", method))
    if name not in TABLES:
        raise ValueError(f"Unknown simulation method {name!r}; expected one of {tuple(TABLES)}.")
    return name


def capabilities(method) -> Mapping[str, Capability]:
    """The read-only capability table of ``method``.

    ``method`` is a ``SimulationMethod`` or one of ``"fdtd"``, ``"fdfd"``,
    ``"mom"``, ``"mom.layered"``, ``"electrostatic"`` and ``"capacitance"``.
    A feature key without a row is refused by that method.
    """
    return TABLES[_method_name(method)]


def _render(capability: Capability, feature, method: str) -> Unsupported:
    template = capability.reason
    reason = template.format(owner=feature.owner, **feature.details)
    if "{owner}" not in template:
        reason = f"{feature.owner}: {reason}"
    return Unsupported(feature.key, method, feature.owner, reason, capability.alternative, capability.support)


def check_scene(scene, method, *, frequencies=(), config=None) -> CapabilityReport:
    """Classify every feature key of ``scene`` for ``method``; never raises for a refusal.

    A key is looked up with its class lineage; a key without a row is refused
    as not declared for the method. A key whose dotted prefix was itself
    refused (a refused collection, a refused ``Scene.symmetry``) is not
    reported again.
    """
    name = _method_name(method)
    rows = TABLES[name]
    refused_keys: set[str] = set()
    supported: dict[str, None] = {}
    unsupported: list[Unsupported] = []
    for feature in scene_features(scene, frequencies=tuple(frequencies), config=config):
        parts = feature.key.split(".")
        if any(".".join(parts[:end]) in refused_keys for end in range(1, len(parts))):
            continue
        row = next((rows[key] for key in feature.lookup if key in rows), None)
        if row is None:
            row = Capability(feature.key, Support.UNSUPPORTED, f"{feature.key} is not declared for {name}")
        if row.support is Support.SUPPORTED:
            supported[feature.key] = None
            continue
        refused_keys.update(feature.lookup)
        unsupported.append(_render(row, feature, name))
    return CapabilityReport(name, ENTRY_POINTS[name], tuple(supported), tuple(unsupported))
