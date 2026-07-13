"""Validation-coverage registry for every public Maxwell medium capability.

P5.6 acceptance criterion (plan §5.4, P5.6): *every* public medium must have a
validation path -- a Tidy3D benchmark scenario, an FDFD cross-check, or a
documented analytic-reference test -- and the benchmark suite must fail loudly
if a shipped medium has none.

This module is the single source of truth for that mapping. The coverage gate
(``tests/validation/benchmark/test_media_validation_coverage.py``) discovers the
public capability *flags* on ``media.py`` dynamically and fails if any flag is
missing a ``MEDIA_VALIDATION`` entry, so a newly-shipped capability cannot slip
through without a declared, verified validation path. The gate additionally
verifies each entry's claim: a ``tidy3d`` entry must actually export through the
adapter, and a non-Tidy3D fallback entry whose ``tidy3d_equivalent`` is ``False``
must genuinely raise on export (proving the fallback is justified, not lazy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

import witwin.maxwell as mw
from witwin.maxwell.media import (
    Graphene,
    LossyMetalMedium,
    Material,
    Medium2D,
    PerturbationMedium,
)


# Validation-path categories.
TIDY3D = "tidy3d"
FDTD_ANALYTIC = "fdtd-analytic"
FDFD = "fdfd"

# Frequency (Hz) at which convention checks compare a Tidy3D-exported medium's
# ``eps_model`` against maxwell's ``Material.relative_permittivity``.
CONVENTION_FREQUENCY = 2.0e14

# Public medium classes whose capability flags the gate must cover.
MEDIA_CLASSES = (Material, Medium2D, Graphene, LossyMetalMedium, PerturbationMedium)


@dataclass(frozen=True)
class MediumValidation:
    """One capability's declared and verifiable validation path."""

    capability: str
    path: str  # TIDY3D | FDTD_ANALYTIC | FDFD
    reference: str  # benchmark scenario name, or a tests/ file path
    probe: Callable[[], Material]  # a medium that turns this capability on
    tidy3d_equivalent: bool  # whether Tidy3D has a construct for this probe
    note: str
    export_frequencies: tuple[float, ...] | None = None  # needed by LossyMetal export
    convention_check: bool = False  # assert eps_model ~= relative_permittivity


def discover_capability_flags() -> set[str]:
    """Public ``is_*`` / ``has_*`` capability properties across the medium classes.

    Introspection is by capability *flag*, not a hardcoded class list, so a new
    medium capability added to ``media.py`` is discovered automatically and the
    coverage gate fails until it is given a validation entry below.
    """
    flags: set[str] = set()
    for cls in MEDIA_CLASSES:
        for name, obj in vars(cls).items():
            if isinstance(obj, property) and (name.startswith("is_") or name.startswith("has_")):
                flags.add(name)
    return flags


# --- probe builders ---------------------------------------------------------
# Each probe returns a medium that turns exactly the target capability on. The
# tensors are uniform where the export homogenizes them and graded where the
# fallback path must reject a spatially-varying profile.
_GRADED = torch.linspace(1.0, 2.0, 8).reshape(2, 2, 2)


def _drude() -> Material:
    return Material.drude(eps_inf=1.0, plasma_frequency=3.0e14, gamma=1.0e13)


def _debye() -> Material:
    return Material.debye(eps_inf=2.0, delta_eps=1.5, tau=3.0e-15)


def _mu_lorentz() -> Material:
    return Material(mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2.0e14, gamma=1.0e13),))


def _custom_electric_uniform() -> Material:
    return Material(lorentz_poles=(mw.CustomLorentzPole(delta_eps=torch.ones(2, 2, 2), resonance_frequency=2.0e14, gamma=1.0e13),))


def _custom_magnetic_uniform() -> Material:
    return Material(mu_lorentz_poles=(mw.CustomLorentzPole(delta_eps=torch.ones(2, 2, 2), resonance_frequency=2.0e14, gamma=1.0e13),))


def _custom_electric_spatial() -> Material:
    return Material(lorentz_poles=(mw.CustomLorentzPole(delta_eps=_GRADED, resonance_frequency=2.0e14, gamma=1.0e13),))


def _diagonal_anisotropic() -> Material:
    return Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))


def _full_tensor() -> Material:
    return Material(epsilon_tensor=mw.Tensor3x3(((4.0, 0.1, 0.0), (0.1, 3.0, 0.0), (0.0, 0.0, 2.0))))


def _full_tensor_dispersive() -> Material:
    return Material(
        epsilon_tensor=mw.Tensor3x3(((4.0, 0.1, 0.0), (0.1, 3.0, 0.0), (0.0, 0.0, 2.0))),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2.0e14, gamma=1.0e13),),
    )


def _kerr() -> Material:
    return Material(eps_r=2.25, kerr_chi3=1.0e-19)


def _chi2() -> Material:
    return Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-12))


def _tpa() -> Material:
    return Material(eps_r=2.25, nonlinearity=mw.TwoPhotonAbsorption(beta=1.0e-11, n0=1.5))


def _modulated() -> Material:
    return Material(eps_r=4.0, modulation=mw.ModulationSpec(frequency=1.0e12, amplitude=0.1, phase=0.0))


def _modulated_dispersive() -> Material:
    return Material(
        eps_r=4.0,
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2.0e14, gamma=1.0e13),),
        modulation=mw.ModulationSpec(frequency=1.0e12, amplitude=0.1, phase=0.0),
    )


def _medium2d() -> Material:
    return Medium2D(sigma_s=1.0e-3)


def _graphene_interband() -> Material:
    return Graphene(chemical_potential=0.4, scattering_time=1.0e-13, include_interband=True)


def _lossy_metal() -> Material:
    return LossyMetalMedium(conductivity=6.0e7)


def _sigma_e() -> Material:
    return Material(eps_r=2.0, sigma_e=50.0)


def _sigma_e_dispersive() -> Material:
    return Material(eps_r=2.0, sigma_e=50.0, drude_poles=(mw.DrudePole(plasma_frequency=3.0e14, gamma=1.0e13),))


def _sigma_m() -> Material:
    return Material(sigma_m=100.0)


def _magnetic_mu_r() -> Material:
    return Material(mu_r=2.0)


def _pec() -> Material:
    return Material.pec()


def _perturbation_uniform() -> Material:
    return PerturbationMedium(Material(eps_r=2.0), perturbation=torch.ones(2, 2, 2), eps_sensitivity=0.5)


def _perturbation_spatial() -> Material:
    return PerturbationMedium(Material(eps_r=2.0), perturbation=_GRADED, eps_sensitivity=0.5)


_ADAPTER_TEST = "tests/api/adapters/tidy3d/test_tidy3d_adapter.py"


# --- the registry -----------------------------------------------------------
# Keyed by capability. Keys prefixed ``is_`` / ``has_`` are the discovered flags
# and MUST cover every flag ``discover_capability_flags`` finds; the remaining
# keys pin the physically-distinct sub-capabilities the plan enumerates
# (conductive, magnetic, the nonlinear sub-orders, the spatially-varying custom
# / perturbation profiles, and the composed no-Tidy3D-equivalent combinations).
MEDIA_VALIDATION: dict[str, MediumValidation] = {
    "is_dispersive": MediumValidation(
        "is_dispersive", TIDY3D, "debye_slab", _debye, True,
        "Debye slab benchmark; exports as td.Debye.",
    ),
    "is_electric_dispersive": MediumValidation(
        "is_electric_dispersive", TIDY3D, "metal_sphere", _drude, True,
        "Drude metal_sphere benchmark; eps_model matches relative_permittivity.",
        convention_check=True,
    ),
    "is_magnetic_dispersive": MediumValidation(
        "is_magnetic_dispersive", FDTD_ANALYTIC,
        "tests/materials/dispersive/test_fdtd_dispersive.py", _mu_lorentz, False,
        "Magnetic-Lorentz slab reflection near resonance; Tidy3D has no magnetic model.",
    ),
    "has_custom_electric_poles": MediumValidation(
        "has_custom_electric_poles", TIDY3D, _ADAPTER_TEST, _custom_electric_uniform, True,
        "Uniform custom electric pole lowers to its scalar reference and exports.",
    ),
    "has_custom_magnetic_poles": MediumValidation(
        "has_custom_magnetic_poles", FDTD_ANALYTIC,
        "tests/materials/dispersive/test_custom_dispersive.py", _custom_magnetic_uniform, False,
        "Custom magnetic pole is magnetic-dispersive; Tidy3D has no magnetic model.",
    ),
    "has_custom_poles": MediumValidation(
        "has_custom_poles", TIDY3D, _ADAPTER_TEST, _custom_electric_uniform, True,
        "Aggregate custom-pole flag; the uniform electric case exports.",
    ),
    "is_anisotropic": MediumValidation(
        "is_anisotropic", TIDY3D, "anisotropic_slab", _diagonal_anisotropic, True,
        "Diagonal-anisotropic slab benchmark; exports as AnisotropicMedium.",
    ),
    "has_full_epsilon_tensor": MediumValidation(
        "has_full_epsilon_tensor", TIDY3D, _ADAPTER_TEST, _full_tensor, True,
        "Non-dispersive full tensor exports as FullyAnisotropicMedium.",
    ),
    "is_nonlinear": MediumValidation(
        "is_nonlinear", TIDY3D, "kerr_slab", _kerr, True,
        "Kerr (chi3) slab benchmark; exports onto td NonlinearSpec.",
    ),
    "is_modulated": MediumValidation(
        "is_modulated", TIDY3D, "modulated_slab", _modulated, True,
        "Time-modulated slab benchmark; exports as Medium + ModulationSpec.",
    ),
    "is_pec": MediumValidation(
        "is_pec", TIDY3D, _ADAPTER_TEST, _pec, True,
        "Material.pec() exports as td.PECMedium.",
    ),
    "is_medium2d": MediumValidation(
        "is_medium2d", TIDY3D, "graphene_sheet", _medium2d, True,
        "Zero-thickness conductive sheet exports as td.Medium2D.",
    ),
    "is_lossy_metal": MediumValidation(
        "is_lossy_metal", TIDY3D, "tests/validation/physics/test_lossy_metal_sibc.py",
        _lossy_metal, True,
        "SIBC metal exports as td.LossyMetalMedium (needs export frequencies).",
        export_frequencies=(2.0e14,),
    ),
    # --- physically-distinct sub-capabilities (explicit, non-flag keys) ------
    "conductive_sigma_e": MediumValidation(
        "conductive_sigma_e", TIDY3D, _ADAPTER_TEST, _sigma_e, True,
        "Static electric conductivity exports via Medium.conductivity [S/um].",
    ),
    "conductive_sigma_e_dispersive": MediumValidation(
        "conductive_sigma_e_dispersive", TIDY3D, "sigma_e_drude_slab", _sigma_e_dispersive, True,
        "sigma_e + dispersion fold into one PoleResidue (zero-frequency conductivity pole).",
    ),
    "magnetic_sigma_m": MediumValidation(
        "magnetic_sigma_m", FDTD_ANALYTIC,
        "tests/materials/conductive/test_fdtd_magnetic_conductive.py", _sigma_m, False,
        "sigma_m slab absorption matches analytic; Tidy3D has no magnetic loss.",
    ),
    "magnetic_mu_r": MediumValidation(
        "magnetic_mu_r", FDTD_ANALYTIC,
        "tests/fdtd/cuda/test_cuda_magnetic_parity.py", _magnetic_mu_r, False,
        "Static magnetic (mu_r != 1) forward parity; Tidy3D fixes mu_r = 1.",
    ),
    "nonlinear_chi2": MediumValidation(
        "nonlinear_chi2", FDTD_ANALYTIC,
        "tests/materials/combinations/test_shg_dispersive_phase_matching.py", _chi2, False,
        "chi2 SHG analytic conversion; Tidy3D's nonlinear API is chi3-family only.",
    ),
    "nonlinear_tpa": MediumValidation(
        "nonlinear_tpa", TIDY3D, _ADAPTER_TEST, _tpa, True,
        "Two-photon absorption exports as td.TwoPhotonAbsorption.",
    ),
    "graphene_interband": MediumValidation(
        "graphene_interband", TIDY3D, "tests/materials/sheet/test_fdtd_graphene.py",
        _graphene_interband, True,
        "Graphene interband Lorentz sheet fit exports as td.Medium2D (PoleResidue).",
    ),
    "custom_poles_spatial": MediumValidation(
        "custom_poles_spatial", FDTD_ANALYTIC,
        "tests/materials/dispersive/test_custom_dispersive.py", _custom_electric_spatial, False,
        "Spatially-varying custom pole; no homogeneous Tidy3D CustomPoleResidue at conversion.",
    ),
    "perturbation_uniform": MediumValidation(
        "perturbation_uniform", TIDY3D, _ADAPTER_TEST, _perturbation_uniform, True,
        "Uniform perturbation lowers to a homogeneous permittivity shift and exports.",
    ),
    "perturbation_spatial": MediumValidation(
        "perturbation_spatial", FDTD_ANALYTIC,
        "tests/materials/perturbation/test_perturbation_medium.py", _perturbation_spatial, False,
        "Spatially-varying perturbation; no homogeneous Tidy3D CustomMedium at conversion.",
    ),
    "full_aniso_dispersive": MediumValidation(
        "full_aniso_dispersive", FDTD_ANALYTIC,
        "tests/materials/combinations/test_rotated_birefringent_dispersive_slab.py",
        _full_tensor_dispersive, False,
        "Full off-diagonal tensor + dispersion; Tidy3D FullyAnisotropicMedium is non-dispersive.",
    ),
    "modulated_dispersive": MediumValidation(
        "modulated_dispersive", FDTD_ANALYTIC,
        "tests/materials/combinations/test_modulated_dispersive_nonlinear.py",
        _modulated_dispersive, False,
        "Modulation + dispersion; Tidy3D modulates only non-dispersive eps_inf.",
    ),
}


def validation_coverage_markdown_lines() -> list[str]:
    """The ``Validation coverage`` section appended to ``benchmark/RESULTS.md``."""
    lines = [
        "## Validation coverage",
        "",
        "Per-medium validation path enforced by "
        "`tests/validation/benchmark/test_media_validation_coverage.py`. The gate "
        "discovers every public `is_*`/`has_*` capability flag on `media.py` and fails "
        "if one lacks an entry below; it also verifies each claim (Tidy3D rows must "
        "export through the adapter, and non-equivalent fallback rows must genuinely "
        "raise on export).",
        "",
        "| Capability | Validation path | Reference | Tidy3D equivalent |",
        "| --- | --- | --- | --- |",
    ]
    for key in sorted(MEDIA_VALIDATION):
        entry = MEDIA_VALIDATION[key]
        equivalent = "yes" if entry.tidy3d_equivalent else "no (Tidy3D lacks the construct)"
        lines.append(f"| `{entry.capability}` | {entry.path} | `{entry.reference}` | {equivalent} |")
    lines.extend(
        [
            "",
            "**Metric staleness.** Group 1 (`planewave_vacuum`, `pml_only`, and the "
            "duplicate `symmetry_center` baseline) was regenerated on 2026-07-12 after "
            "the external-PML, grid, CPML, source-power, and flux-monitor repairs. Other "
            "historical plane-wave/material rows remain STALE until their groups are rerun. "
            "The P3-media scenarios "
            "(`debye_slab`, `sigma_e_drude_slab`, `anisotropic_slab`, `kerr_slab`, "
            "`modulated_slab`, `graphene_sheet`) build and export but have not been "
            "cloud-run (Tidy3D runs are cache-gated), so they carry no metric row yet.",
            "",
        ]
    )
    return lines
