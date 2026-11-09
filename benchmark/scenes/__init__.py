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
from benchmark.scenes.coverage import COVERAGE_SCENARIOS
from benchmark.scenes.media import MEDIA_EXPORT_SCENARIOS
from benchmark.scenes.planewave import (
    DIELECTRIC_SLAB,
    DIELECTRIC_SPHERE,
    METAL_SPHERE,
    PLANEWAVE_DIELECTRIC_SPHERE,
    PLANEWAVE_PERIODIC_VACUUM,
    PLANEWAVE_VACUUM,
)
from benchmark.scenes.planned import PLANNED_SCENARIOS


_REGISTERED = (
    DIPOLE_VACUUM,
    PLANEWAVE_VACUUM,
    PLANEWAVE_PERIODIC_VACUUM,
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
    *COVERAGE_SCENARIOS,
)

# Real-field closed-transverse PML rows whose reference comparisons were validated on
# CPML. The solver default absorber="auto" selects the graded absorber (>= 40 layers)
# for this topology; these rows pin CPML, which keeps their results bit-unchanged and
# runs them under the fail-closed passivity guard.
CPML_PINNED_SCENARIOS = frozenset({
    "planewave_periodic_vacuum", "periodic_slab", "periodic_grating", "grating_diffraction",
    "diffraction_normal_orders", "mixed_faces", "modulated_slab", "modulated_slab_phase",
    "kerr_slab", "kerr_slab_strong", "dispersive_kerr_slab", "tpa_slab", "graphene_sheet",
})

_DUPLICATES = sorted(
    {
        scenario.name
        for index, scenario in enumerate(_REGISTERED)
        if any(other.name == scenario.name for other in _REGISTERED[:index])
    }
)
if _DUPLICATES:
    raise RuntimeError(f"Duplicate benchmark scenario names: {_DUPLICATES}")

SCENARIOS = {scenario.name: scenario for scenario in _REGISTERED}
_UNKNOWN_PINS = sorted(CPML_PINNED_SCENARIOS - SCENARIOS.keys())
if _UNKNOWN_PINS:
    raise RuntimeError(f"CPML-pinned benchmark scenarios are not registered: {_UNKNOWN_PINS}")


def build_scene(name: str):
    if name not in SCENARIOS:
        raise KeyError(f"Unknown benchmark scenario '{name}'. Available: {list(SCENARIOS)}")
    return SCENARIOS[name].builder()
