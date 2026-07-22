"""Shared scene-structure filters used across compiler and runtime modules."""

from __future__ import annotations

from typing import Any


def pec_structures(scene) -> tuple[Any, ...]:
    """Enabled scene structures whose material is PEC, in scene order.

    Disabled structures never reach the compiled material model, so they cannot
    suppress Yee edges and must not count as conductors for terminal-contact or
    occupancy checks either.
    """

    return tuple(
        structure
        for structure in getattr(scene, "structures", ())
        if bool(getattr(structure, "enabled", True))
        and bool(getattr(getattr(structure, "material", None), "is_pec", False))
    )
