"""Capability rows, report items and the aggregated report."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Support(str, Enum):
    """How a method treats one feature key.

    ``SUPPORTED`` means the method admits the feature at construction; a
    runtime-deep guard may still refuse a narrower configuration of it.
    ``UNSUPPORTED`` is a capability the method lacks and could gain;
    ``INHERENT`` is a feature with no meaning for the method.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INHERENT = "inherent"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Capability:
    """One row of a method table.

    ``reason`` is a ``str.format`` template over the feature's owner label
    (``{owner}``) and the details the key extractor records. Without an
    ``{owner}`` field the rendered item reads ``"<owner>: <reason>"``.
    ``negative_tests`` name the tests that observe a refusal row.
    """

    key: str
    support: Support
    reason: str = ""
    alternative: str | None = None
    negative_tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unsupported:
    """One refused feature of one scene: the rendered row for its owner."""

    key: str
    method: str
    owner: str
    reason: str
    alternative: str | None
    support: Support = Support.UNSUPPORTED

    def __str__(self) -> str:
        return self.reason if self.alternative is None else f"{self.reason}; {self.alternative}"


@dataclass(frozen=True)
class CapabilityReport:
    """Every feature key of a scene, split by the method table."""

    method: str
    entry: str
    supported: tuple[str, ...]
    unsupported: tuple[Unsupported, ...]

    def raise_if_unsupported(self) -> None:
        """Raise one ``UnsupportedFeatureError`` that lists every refused item."""
        if not self.unsupported:
            return
        from ..errors import UnsupportedFeatureError

        items = self.unsupported
        raise UnsupportedFeatureError(
            f"Simulation.{self.entry}(...) cannot represent {len(items)} scene feature(s):\n"
            + "\n".join(f"- {item}" for item in items),
            feature=items[0].key,
            method=self.method,
            alternative=items[0].alternative if len(items) == 1 else None,
            items=items,
        )


def supported(*keys: str) -> tuple[Capability, ...]:
    return tuple(Capability(key, Support.SUPPORTED) for key in keys)


def unsupported(key: str, reason: str, alternative: str | None = None, *, tests: tuple[str, ...]) -> Capability:
    return Capability(key, Support.UNSUPPORTED, reason, alternative, tuple(tests))


def inherent(key: str, reason: str, alternative: str | None = None, *, tests: tuple[str, ...]) -> Capability:
    return Capability(key, Support.INHERENT, reason, alternative, tuple(tests))


def table(*rows) -> Mapping[str, Capability]:
    """A read-only key -> row mapping; nested row tuples are flattened."""
    flat: dict[str, Capability] = {}
    for row in rows:
        for capability in (row,) if isinstance(row, Capability) else row:
            if capability.key in flat:
                raise ValueError(f"duplicate capability row {capability.key!r}")
            flat[capability.key] = capability
    return MappingProxyType(flat)


def override(base: Mapping[str, Capability], *rows) -> Mapping[str, Capability]:
    """``base`` with the given rows replacing (or adding to) its entries."""
    return MappingProxyType({**base, **table(*rows)})
