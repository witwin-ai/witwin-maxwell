"""Public namespace ``mw.fdtd``: FDTD configuration, absorbers, checkpoints, errors and run records.

The runtime modules of this package are internal.
"""
from .._lazy import lazy_attributes
from ..absorbers import InternalAbsorber, ModeABC, MurABC
from ..fdtd_cpml import CPMLConfig, CPMLMemoryMode
from ..fdtd_parallel import FDTDParallelConfig
from ..material_cache import FDTDMaterialCache
from .resume import FDTDResumeCheckpoint
from .runtime.passivity import FDTDPassivityError

__all__ = [
    "AbsorberKind",
    "CPMLConfig",
    "CPMLMemoryMode",
    "FDTDMaterialCache",
    "FDTDParallelConfig",
    "FDTDPassivityError",
    "FDTDResumeCheckpoint",
    "InternalAbsorber",
    "ModeABC",
    "MurABC",
    "SpectralSampler",
    "SpectralWindowKind",
    "tfsf_h_mask",
]

# ``simulation`` imports this package, so its names are read on first access.
__getattr__, __dir__ = lazy_attributes(__name__, names={
    ".simulation": ("AbsorberKind", "SpectralSampler", "SpectralWindowKind"),
    ".fdtd.results": ("tfsf_h_mask",),
})
del lazy_attributes
