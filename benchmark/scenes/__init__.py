from __future__ import annotations

from benchmark.scenes.dipole import (
    DIPOLE_DIELECTRIC_BOX,
    DIPOLE_DIELECTRIC_SPHERE,
    DIPOLE_EY,
    DIPOLE_OFFCENTER,
    DIPOLE_TWO_FREQ,
    DIPOLE_VACUUM,
    HIGH_EPS_BOX,
    LORENTZ_RESONATOR,
    MULTI_DIELECTRIC,
)
from benchmark.scenes.media import MEDIA_EXPORT_SCENARIOS
from benchmark.scenes.planewave import (
    DIELECTRIC_SLAB,
    DIELECTRIC_SPHERE,
    METAL_SPHERE,
    PLANEWAVE_DIELECTRIC_SPHERE,
    PLANEWAVE_VACUUM,
)
from benchmark.scenes.planned import PLANNED_SCENARIOS


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        DIPOLE_VACUUM,
        PLANEWAVE_VACUUM,
        DIPOLE_DIELECTRIC_BOX,
        PLANEWAVE_DIELECTRIC_SPHERE,
        DIELECTRIC_SLAB,
        METAL_SPHERE,
        MULTI_DIELECTRIC,
        LORENTZ_RESONATOR,
        DIPOLE_EY,
        DIPOLE_OFFCENTER,
        HIGH_EPS_BOX,
        DIELECTRIC_SPHERE,
        DIPOLE_DIELECTRIC_SPHERE,
        DIPOLE_TWO_FREQ,
        *MEDIA_EXPORT_SCENARIOS,
        *PLANNED_SCENARIOS,
    )
}


def build_scene(name: str):
    if name not in SCENARIOS:
        raise KeyError(f"Unknown benchmark scenario '{name}'. Available: {list(SCENARIOS)}")
    return SCENARIOS[name].builder()
