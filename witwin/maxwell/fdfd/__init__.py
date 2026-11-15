"""Public namespace ``mw.fdfd``: FDFD solver configuration and local-response contracts.

The runtime modules of this package are internal.
"""
from .._lazy import lazy_attributes
from ..fdfd_response import (
    FarFieldProjection,
    FieldProjection,
    LocalDesignRegion,
    LocalResponseConfig,
    LumpedCurrentSource,
)

__all__ = [
    "FDFDPreconditioner",
    "FDFDSolverConfig",
    "FDFDSolverType",
    "FarFieldProjection",
    "FieldProjection",
    "LocalDesignRegion",
    "LocalResponseConfig",
    "LumpedCurrentSource",
    "Precision",
]

# ``simulation`` imports this package, so its names are read on first access.
__getattr__, __dir__ = lazy_attributes(__name__, names={
    ".simulation": ("FDFDPreconditioner", "FDFDSolverConfig", "FDFDSolverType", "Precision"),
})
del lazy_attributes
