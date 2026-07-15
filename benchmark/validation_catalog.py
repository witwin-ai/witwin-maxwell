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
        scenario = SCENARIOS.get(self.name)
        return bool(scenario and scenario.reference_solver != "tidy3d") or has_cache(self.name)


def _cases(family: str, rows: tuple[tuple[str, str], ...], *, solver: str = "fdtd"):
    return tuple(ValidationCase(family, name, observable, solver) for name, observable in rows)


VALIDATION_CASES = (
    *_cases("sources", (("astigmatic_beam", "waist profile"), ("uniform_current", "radiated power"),
        ("custom_field_source", "reconstructed plane-wave flux"), ("custom_current_source", "field correlation"),
        ("mode_source_wg", "transmitted mode power"),
        ("gaussian_beam_normal", "on-axis waist profile"),
        ("gaussian_beam_defocused", "downstream-focus beam profile"),
        ("planewave_cw", "continuous-wave steady-state field"),
        ("dipole_cw_vacuum", "continuous-wave dipole field"),
        ("tfsf_vacuum", "scattered-field leakage outside the TFSF box"),
        ("tfsf_dielectric_sphere", "scattered-field flux"),
        ("mode_source_higher_order", "mode_index=1 transmitted amplitude"),
        ("magnetic_current_vacuum", "magnetic-dipole radiation pattern"),
        ("ricker_axis_x_anisotropic", "negative-x Ricker-driven eps_zz transmission"))),
    *_cases("media", (("debye_slab", "reflection/transmission spectrum"),
        ("sigma_e_drude_slab", "absorption"), ("anisotropic_slab", "polarization-resolved T/R"),
        ("kerr_slab", "intensity-dependent transmission"), ("modulated_slab", "sideband power"),
        ("graphene_sheet", "reflection/transmission"), ("pec_box", "total reflection"),
        ("full_tensor_slab", "polarization-resolved T/R"), ("sigma_e_slab", "absorption"),
        ("tpa_slab", "intensity-dependent transmission"), ("custom_pole_uniform_slab", "T/R spectrum"),
        ("perturbation_uniform_slab", "T/R"), ("lossy_metal_slab", "reflection"),
        ("sellmeier_slab", "dispersive transmission phase"),
        ("drude_slab", "Drude T/R spectrum"), ("lorentz_slab", "single-pole Lorentz T/R"),
        ("lorentz_two_pole_slab", "two-pole Lorentz T/R"),
        ("drude_lorentz_slab", "folded PoleResidue T/R"),
        ("debye_sphere", "Debye scattering across a curved interface"),
        ("sellmeier_sphere", "lossless dispersive scattering"),
        ("diag_aniso_sphere", "polarization-resolved curved-interface scattering"),
        ("pec_sphere", "PEC curved-surface scattering"),
        ("lossy_metal_slab_high_sigma", "surface-impedance reflection at high sigma"),
        ("kerr_slab_strong", "strong-drive intensity-dependent transmission"),
        ("modulated_slab_phase", "phase-shifted sideband power"),
        ("static_medium2d_sheet", "nondispersive sheet reflection/transmission"),
        ("material_region_slab", "uniform design-region transmission"),
        ("dispersive_kerr_slab", "dispersive nonlinear transmission"))),
    *_cases("boundaries", (("pml_only", "steady-state field correlation"),
        ("periodic_grating", "transmission"), ("bloch_oblique", "complex transmission"),
        ("pec_cavity", "resonance frequency"), ("pmc_cavity", "resonance frequency"),
        ("symmetry_center", "field versus full-domain run"),
        ("pml_thin", "6-layer PML residual reflection"),
        ("pml_slab_through", "material-loaded PML transmission"),
        ("periodic_slab", "normal-incidence periodic transmission"),
        ("bloch_oblique_te", "in-plane-polarized complex transmission"),
        ("symmetry_pec_center", "translation-invariant PEC symmetry export"),
        ("symmetry_pmc_center", "translation-invariant PMC symmetry export"),
        ("mixed_faces", "per-face periodic/PEC/PML field correlation"),
        ("asymmetric_boundary_faces", "different low/high face conditions"))),
    *_cases("grid_geometry", (("custom_grid_slab", "field L2"), ("autogrid_ring", "field L2 and resonance"),
        ("polyslab_wg", "transmitted power"), ("mesh_primitive_scatter", "scattered-field correlation"),
        ("cylinder_scatter", "scattered-field correlation"),
        ("cone_scatter", "sidewall-angle scattered-field correlation"),
        ("ellipsoid_scatter", "scattered-field correlation"),
        ("pyramid_scatter", "mesh-export scattered-field correlation"),
        ("prism_scatter", "mesh-export scattered-field correlation"),
        ("hollow_box_scatter", "shell scattered-field correlation"),
        ("polyslab_pentagon", "non-rectangular PolySlab scattering"),
        ("autogrid_slab", "auto-mesher field L2"),
        ("nonuniform_custom_grid", "graded-grid field L2"),
        ("anisotropic_uniform_grid", "per-axis grid field L2"),
        ("explicit_mesh_scatter", "explicit triangle-mesh field correlation"),
        ("autogrid_override_refinement", "override/refinement resolved-grid field L2"))),
    *_cases("postprocess", (("ring_resonator_s21", "S21 magnitude and phase"),
        ("waveguide_s_matrix", "S11/S21"), ("grating_diffraction", "per-order efficiency"),
        ("sphere_rcs", "bistatic RCS"), ("antenna_directivity", "directivity, gain, beamwidth"),
        ("rcs_pec_sphere", "bistatic RCS"), ("rcs_dielectric_box", "bistatic RCS"),
        ("directivity_two_dipoles", "array-factor directivity and nulls"),
        ("mode_monitor_straight_wg", "n_eff and S21"),
        ("mode_monitor_two_planes", "modal amplitude ratio and propagation phase"),
        ("diffraction_normal_orders", "0th and +-1st order efficiency"),
        ("point_monitor_probe", "point-probe complex field values"),
        ("mode_port_straight_wg", "ModePort complex amplitude ratio"),
        ("permittivity_monitor_slab", "permittivity component statistics"),
        ("time_monitor_vacuum", "normalized field and flux time traces"))),
    *_cases("fdfd", (("fdfd_dielectric_slab", "field correlation and T/R"),
        ("fdfd_drude_sphere", "field correlation"), ("fdfd_sigma_e_slab", "absorption"),
        ("fdfd_diag_aniso_slab", "polarization-resolved transmission")), solver="fdfd"),
)


def inventory_markdown() -> str:
    lines = ["| Family | Case | Solver | Observable | Registered | Reference |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in VALIDATION_CASES:
        lines.append(
            f"| {case.family} | `{case.name}` | {case.solver} | {case.observable} | "
            f"{'yes' if case.registered else 'no'} | "
            f"{'local' if case.registered and SCENARIOS[case.name].reference_solver != 'tidy3d' else ('hit' if case.cached else 'missing')} |"
        )
    return "\n".join(lines)
