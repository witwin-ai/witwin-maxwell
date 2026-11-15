"""Base of the MoM option Enums and the payload writer that keeps their records plain strings.

MoM configuration objects parse each string option once at construction (``parse_enum``) and
store the member. Persisted payloads, receipts and identities must stay byte-identical to the
string records they replace (``_same_metadata`` compares types, snapshots are pickled and
restore checks read ``type(value) is str``), so every writer stores ``member.value``.
"""
from __future__ import annotations

from dataclasses import asdict

from .._enums import StrEnum


class MoMOption(StrEnum):
    """Memberless base of the MoM option Enums; ``option_asdict`` writes its members as values."""


def plain_option(value):
    """``value.value`` for a MoM option member, ``value`` otherwise."""
    return value.value if isinstance(value, MoMOption) else value


def _plain_items(items):
    return {key: plain_option(value) for key, value in items}


def option_asdict(instance) -> dict:
    """``dataclasses.asdict(instance)`` with every MoM option member written as its value string."""
    return asdict(instance, dict_factory=_plain_items)
