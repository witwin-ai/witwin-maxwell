from __future__ import annotations

from benchmark.scenes.media.anisotropic_slab import SCENARIO as ANISOTROPIC_SLAB
from benchmark.scenes.media.debye_slab import SCENARIO as DEBYE_SLAB
from benchmark.scenes.media.graphene_sheet import SCENARIO as GRAPHENE_SHEET
from benchmark.scenes.media.kerr_slab import SCENARIO as KERR_SLAB
from benchmark.scenes.media.modulated_slab import SCENARIO as MODULATED_SLAB
from benchmark.scenes.media.sigma_e_drude_slab import SCENARIO as SIGMA_E_DRUDE_SLAB


# P3-media benchmark scenarios that carry a real Tidy3D export equivalent. Each
# is registered in the top-level ``SCENARIOS`` map so ``python -m benchmark``
# cross-validates it, and each is named as a Tidy3D validation path by the
# coverage gate (``benchmark/media_coverage.py``).
MEDIA_EXPORT_SCENARIOS = (
    DEBYE_SLAB,
    SIGMA_E_DRUDE_SLAB,
    ANISOTROPIC_SLAB,
    KERR_SLAB,
    MODULATED_SLAB,
    GRAPHENE_SHEET,
)
