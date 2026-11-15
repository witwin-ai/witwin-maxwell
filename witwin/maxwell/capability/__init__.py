"""Capability registry: which scene features each simulation method admits.

``keys`` extracts the feature keys of a scene, ``tables`` holds one row per key
and method, and ``check`` admits a scene default-deny: a key without a row is
refused. ``Simulation.<method>(scene, ...)`` raises one
``UnsupportedFeatureError`` listing every refused item at construction.
Runtime-deep guards stay at their sites; a ``SUPPORTED`` row means the method
admits the feature at construction.
"""
from .check import capabilities, check_scene
from .keys import MOM_CONFIG_KEYS as CONFIG_KEYS
from .keys import SCENE_KEYS
from .model import Capability, CapabilityReport, Support, Unsupported
from .tables import TABLES

# The registry vocabulary: the method names ``capabilities()`` accepts and every
# feature key a table decides (scene keys, then configuration keys). Namespace
# facades and documentation generators read these instead of the tables' modules.
METHODS: tuple[str, ...] = tuple(TABLES)
FEATURE_KEYS: tuple[str, ...] = (*SCENE_KEYS, *CONFIG_KEYS)

# ``capabilities`` is public at the top level (``mw.capabilities``).
__all__ = ["CONFIG_KEYS", "Capability", "CapabilityReport", "FEATURE_KEYS", "METHODS", "SCENE_KEYS", "Support",
           "Unsupported", "check_scene"]
