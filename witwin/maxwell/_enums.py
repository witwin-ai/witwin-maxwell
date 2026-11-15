"""String-valued option Enums shared by several public objects, and their boundary parser.

Every public option that takes one of a fixed set of strings is a ``str`` Enum.
A constructor accepts a member or exactly its value string, parses it once with
``parse_enum`` and stores the member; any other value raises ``SetupError``.
Members compare equal to, hash like, and format and repr as their value strings.
"""
from __future__ import annotations

from enum import Enum

from .errors import SetupError


class StrEnum(str, Enum):
    """Base of the option Enums: a member formats, prints and reprs as its value string.

    ``repr`` follows the value so that messages, reprs of the configuration records
    and repr-based identities read exactly as they did when the options were strings.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)


def parse_enum(enum_type, value, *, name: str):
    """``value`` as a member of ``enum_type``: a member, or exactly a member's value string."""
    try:
        return enum_type(value)
    except ValueError:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise SetupError(f"{name} must be one of {choices}, got {value!r}.") from None


class FieldComponent(StrEnum):
    """A Cartesian electric or magnetic field component."""

    EX = "Ex"
    EY = "Ey"
    EZ = "Ez"
    HX = "Hx"
    HY = "Hy"
    HZ = "Hz"


class Direction(StrEnum):
    """Propagation or normal sense along an axis: ``PLUS`` is ``"+"``, ``MINUS`` is ``"-"``."""

    PLUS = "+"
    MINUS = "-"


class ModePolarization(StrEnum):
    """The polarization family that indexes the modes of a mode plane.

    ``AUTO`` picks the larger tangential extent of the plane; ``EX``/``EY``/``EZ``
    pick the electric component, which must be tangential to the plane.
    """

    AUTO = "auto"
    EX = "Ex"
    EY = "Ey"
    EZ = "Ez"
