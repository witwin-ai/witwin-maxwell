"""Domain exception and warning hierarchy of witwin-maxwell.

Domain exceptions derive from ``MaxwellError``. Each family also keeps the
built-in base of the errors it classifies, so existing
``except NotImplementedError`` / ``ValueError`` / ``RuntimeError`` /
``MemoryError`` handlers keep working.
"""

from __future__ import annotations

__all__ = [
    "MaxwellError",
    "UnsupportedFeatureError",
    "SetupError",
    "ConvergenceError",
    "SchemaError",
    "CapacityError",
    "MaxwellWarning",
    "ExperimentalWarning",
]


class MaxwellError(Exception):
    """Base class of every witwin-maxwell exception."""


class UnsupportedFeatureError(MaxwellError, NotImplementedError):
    """A declared feature is not supported by the selected method.

    ``feature`` is the capability key (``"guard:<census id>"`` for runtime-deep
    guards), ``method`` the simulation method or ``None`` when the refusal is
    method independent, ``alternative`` a hint for a supported route and
    ``items`` every collected item when the refusal aggregates several.
    """

    def __init__(
        self,
        message: str,
        *,
        feature: str,
        method: str | None,
        alternative: str | None = None,
        items: tuple = (),
    ):
        super().__init__(message)
        self.feature = feature
        self.method = method
        self.alternative = alternative
        self.items = tuple(items)


class SetupError(MaxwellError, ValueError):
    """An invalid scene or simulation declaration."""


class ConvergenceError(MaxwellError, RuntimeError):
    """An iterative solve, fit or stability check failed to converge."""


class SchemaError(MaxwellError, ValueError):
    """A serialized payload or file does not match its schema or version."""


class CapacityError(MaxwellError, MemoryError):
    """A requested solve exceeds the admitted memory or problem-size capacity."""


class MaxwellWarning(UserWarning):
    """Base class of every witwin-maxwell warning."""


class ExperimentalWarning(MaxwellWarning):
    """The used feature is experimental and its API may change."""
