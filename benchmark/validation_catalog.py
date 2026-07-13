"""Validation campaign inventory from docs/dev/validation-vs-tidy3d-plan.md."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.cache import has_cache
from benchmark.scenes import SCENARIOS


@dataclass(frozen=True)
class ValidationCase:
    family: str
    name: str
    observable: str
    solver: str = "fdtd"

    @property
    def registered(self) -> bool:
        return self.name in SCENARIOS

    @property
    def cached(self) -> bool:
        return has_cache(self.name)


def _cases(family: str, rows: tuple[tuple[str, str], ...], *, solver: str = "fdtd"):
    return tuple(ValidationCase(family, name, observable, solver) for name, observable in rows)


VALIDATION_CASES = (
    *_cases("sources", (("astigmatic_beam", "waist profile"), ("uniform_current", "radiated power"),
        ("custom_field_source", "reconstructed plane-wave flux"), ("custom_current_source", "field correlation"),
        ("mode_source_wg", "transmitted mode power"))),
    *_cases("media", (("debye_slab", "reflection/transmission spectrum"),
        ("sigma_e_drude_slab", "absorption"), ("anisotropic_slab", "polarization-resolved T/R"),
        ("kerr_slab", "intensity-dependent transmission"), ("modulated_slab", "sideband power"),
        ("graphene_sheet", "reflection/transmission"), ("pec_box", "total reflection"),
        ("full_tensor_slab", "polarization-resolved T/R"), ("sigma_e_slab", "absorption"),
        ("tpa_slab", "intensity-dependent transmission"), ("custom_pole_uniform_slab", "T/R spectrum"),
        ("perturbation_uniform_slab", "T/R"), ("lossy_metal_slab", "reflection"),
        ("sellmeier_slab", "dispersive transmission phase"))),
    *_cases("boundaries", (("pml_only", "steady-state field correlation"),
        ("periodic_grating", "transmission"), ("bloch_oblique", "complex transmission"),
        ("pec_cavity", "resonance frequency"), ("pmc_cavity", "resonance frequency"),
        ("symmetry_center", "field versus full-domain run"))),
    *_cases("grid_geometry", (("custom_grid_slab", "field L2"), ("autogrid_ring", "field L2 and resonance"),
        ("polyslab_wg", "transmitted power"), ("mesh_primitive_scatter", "scattered-field correlation"))),
    *_cases("postprocess", (("ring_resonator_s21", "S21 magnitude and phase"),
        ("waveguide_s_matrix", "S11/S21"), ("grating_diffraction", "per-order efficiency"),
        ("sphere_rcs", "bistatic RCS"), ("antenna_directivity", "directivity, gain, beamwidth"))),
    *_cases("fdfd", (("fdfd_dielectric_slab", "field correlation and T/R"),
        ("fdfd_drude_sphere", "field correlation"), ("fdfd_sigma_e_slab", "absorption"),
        ("fdfd_diag_aniso_slab", "polarization-resolved transmission")), solver="fdfd"),
)


def inventory_markdown() -> str:
    lines = ["| Family | Case | Solver | Observable | Registered | Tidy3D cache |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in VALIDATION_CASES:
        lines.append(
            f"| {case.family} | `{case.name}` | {case.solver} | {case.observable} | "
            f"{'yes' if case.registered else 'no'} | {'hit' if case.cached else 'missing'} |"
        )
    return "\n".join(lines)
