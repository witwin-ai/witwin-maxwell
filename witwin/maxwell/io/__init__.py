"""Serialized payloads of witwin-maxwell (``mw.io``).

``Scene.to_dict`` / ``Scene.from_dict`` write and read ``witwin.scene/2`` scene dictionaries;
:func:`migrate_scene_dict` upgrades an older ``witwin.scene/1`` dictionary (``from_dict`` calls it);
:func:`scene_json_schema` returns the committed JSON Schema of ``witwin.scene/2``.
"""

from .migrate import migrate_scene_dict
from .schema import scene_json_schema

__all__ = ["migrate_scene_dict", "scene_json_schema"]
