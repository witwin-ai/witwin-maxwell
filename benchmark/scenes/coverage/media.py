"""Second-instance media coverage scenarios.

Every material model already validated by a single benchmark case gets an
independent partner here: a different geometry (curved instead of planar), a
different pole structure, or a different drive amplitude. The mixed
Drude + Lorentz slab additionally forces the ``td.PoleResidue`` folding branch of
the Tidy3D adapter, which the single-pole-family scenarios never reach.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw
from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import DX, HALF_SPAN, base_scene
from benchmark.scenes.planned import FREQUENCIES, PULSE, _make, _observed, _plane_scene


SPHERE_RADIUS = 0.15
RUN_TIME_FACTOR = 20.0

# Sideband spacing of the phase-shifted modulation case. Wide enough to separate
# the sidebands from the 2 GHz carrier on the benchmark's DFT grid, narrow enough
# that both stay inside the source spectrum the coarse grid resolves.
MODULATION_FREQUENCY = 0.35e9
MODULATED_FREQUENCIES = (
    FREQUENCIES[0] - MODULATION_FREQUENCY,
    FREQUENCIES[0],
    FREQUENCIES[0] + MODULATION_FREQUENCY,
)

PERIODIC_BOUNDARY = mw.BoundarySpec.pml(num_layers=12).with_faces(
    x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"
)


def _sphere_scene(material: mw.Material) -> mw.Scene:
    """Plane wave scattering off a sphere of ``material``.

    The curved interface is the point: it exercises each constitutive model
    through subpixel-averaged staircased cells rather than the flat, grid-aligned
    face the slab scenarios present.
    """
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=SPHERE_RADIUS),
            material=material,
            name="sample",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _lossy_metal_slab(conductivity: float) -> mw.Scene:
    """Lossy-metal half space with its face pinned to a Yee node plane.

    Same grid-alignment care as ``planned._lossy_metal_half_space``: the SIBC
    surface update lives on a node plane, so the physical metal face must land on
    that plane in both solvers or the comparison picks up a round-trip reflection
    phase unrelated to the surface-impedance model. Only the conductivity (hence
    the skin depth and Z_s) differs from the existing case.
    """
    physical_cell_count = int(math.ceil(2.0 * HALF_SPAN / DX))
    physical_step = 2.0 * HALF_SPAN / physical_cell_count
    surface = HALF_SPAN - 2.0 * physical_step
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, surface + 0.25), size=(3.0, 3.0, 0.50)),
            material=mw.LossyMetalMedium(conductivity=conductivity),
            name="lossy_metal_boundary",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene, flux_positions=(-0.3, 0.45))


def _modulated_phase_scene() -> mw.Scene:
    """Time-modulated slab with a nonzero modulation phase.

    Modulation may not be combined with dispersion or a nonlinearity, so the host
    is a plain dielectric. Periodic transverse faces keep the sideband generation
    one-dimensional, and a CW drive gives the sidebands a clean carrier to be
    measured against.
    """
    reference = base_scene()
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=PERIODIC_BOUNDARY,
        subpixel_samples=reference.subpixel,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.1)),
            material=mw.Material(
                eps_r=3.0,
                modulation=mw.ModulationSpec(
                    frequency=MODULATION_FREQUENCY, amplitude=0.2, phase=math.pi / 3.0
                ),
                name="phase_modulated_dielectric",
            ),
            name="sample",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0, 0, 1),
            polarization="Ex",
            source_time=mw.CW(frequency=FREQUENCIES[0], amplitude=1.0),
        )
    )
    return _observed(scene, frequencies=MODULATED_FREQUENCIES)


def _static_medium2d_scene() -> mw.Scene:
    """Nondispersive conductive sheet, distinct from the Graphene model."""
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.0),
            ),
            material=mw.Medium2D(sigma_s=0.02, name="static_sheet"),
            name="static_medium2d_sheet",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _material_region_scene() -> mw.Scene:
    """Uniform design density lowered to the same homogeneous slab in both solvers."""
    scene = base_scene().add_material_region(
        mw.MaterialRegion(
            name="uniform_design_region",
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.12),
            ),
            density=torch.full((2, 2, 2), 0.5),
            eps_bounds=(1.0, 5.0),
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _dispersive_kerr_scene() -> mw.Scene:
    """Lorentz dispersion composed with the adapter-supported chi3 nonlinearity."""
    material = mw.Material(
        eps_r=1.5,
        lorentz_poles=(
            mw.LorentzPole(
                delta_eps=0.8,
                resonance_frequency=3.5e9,
                gamma=0.25e9,
            ),
        ),
        kerr_chi3=1.0e-18,
        name="dispersive_kerr",
    )
    return _plane_scene(
        material=material,
        boundary=PERIODIC_BOUNDARY,
        source=mw.PlaneWave(
            direction=(0, 0, 1),
            polarization="Ex",
            source_time=mw.GaussianPulse(
                frequency=FREQUENCIES[0],
                fwidth=0.5e9,
                amplitude=1.0e8,
            ),
        ),
    )


MEDIA_COVERAGE_SCENARIOS = (
    _make(
        "drude_slab",
        "media",
        "pure Drude dispersion",
        lambda: _plane_scene(
            material=mw.Material.drude(eps_inf=3.0, plasma_frequency=2.5e9, gamma=0.2e9)
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "lorentz_slab",
        "media",
        "single-pole Lorentz dispersion",
        lambda: _plane_scene(
            material=mw.Material.lorentz(
                eps_inf=1.5, delta_eps=1.0, resonance_frequency=3.5e9, gamma=0.25e9
            )
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "lorentz_two_pole_slab",
        "media",
        "two-pole Lorentz dispersion",
        lambda: _plane_scene(
            material=mw.Material(
                eps_r=1.5,
                lorentz_poles=(
                    mw.LorentzPole(delta_eps=0.8, resonance_frequency=3.0e9, gamma=0.2e9),
                    mw.LorentzPole(delta_eps=0.5, resonance_frequency=5.0e9, gamma=0.3e9),
                ),
                name="two_pole_lorentz",
            )
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "drude_lorentz_slab",
        "media",
        "mixed Drude+Lorentz poles folded to one PoleResidue",
        lambda: _plane_scene(
            material=mw.Material(
                eps_r=2.0,
                drude_poles=(mw.DrudePole(plasma_frequency=1.5e9, gamma=0.2e9),),
                lorentz_poles=(
                    mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.5e9, gamma=0.25e9),
                ),
                name="drude_lorentz",
            )
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "debye_sphere",
        "media",
        "Debye relaxation across a curved interface",
        lambda: _sphere_scene(mw.Material.debye(eps_inf=2.0, delta_eps=2.0, tau=5.0e-11)),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "sellmeier_sphere",
        "media",
        "lossless Sellmeier dispersion across a curved interface",
        lambda: _sphere_scene(
            mw.Material.sellmeier(
                eps_inf=1.5, b_coefficients=(1.5,), c_coefficients=(2.5e-3,)
            )
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "diag_aniso_sphere",
        "media",
        "diagonal-anisotropic permittivity across a curved interface",
        lambda: _sphere_scene(
            mw.Material(
                epsilon_tensor=mw.DiagonalTensor3(3.0, 2.0, 2.5),
                name="uniaxial_sphere",
            )
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "pec_sphere",
        "media",
        "PEC scattering off a curved surface",
        lambda: _sphere_scene(mw.Material.pec()),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "lossy_metal_slab_high_sigma",
        "media",
        "surface-impedance reflection at a 100x higher conductivity",
        lambda: _lossy_metal_slab(1.0e6),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "kerr_slab_strong",
        "media",
        "Kerr chi3 slab at a strong drive amplitude",
        lambda: _plane_scene(
            material=mw.Material(eps_r=4.0, kerr_chi3=1.0e-18, name="kerr_strong"),
            boundary=PERIODIC_BOUNDARY,
            source=mw.PlaneWave(
                direction=(0, 0, 1),
                polarization="Ex",
                source_time=mw.GaussianPulse(
                    frequency=FREQUENCIES[0], fwidth=0.5e9, amplitude=1.0e8
                ),
            ),
        ),
        run_time_factor=RUN_TIME_FACTOR,
    ),
    ScenarioDefinition(
        name="modulated_slab_phase",
        description="media: phase-shifted time modulation at a wider sideband spacing",
        builder=_modulated_phase_scene,
        frequencies=MODULATED_FREQUENCIES,
        display_monitor="field",
        display_component="Ex",
        run_time_factor=RUN_TIME_FACTOR,
        # Sidebands carry no incident source spectrum to divide by; compare their
        # complex fields against the carrier instead of the raw source normalization.
        normalize_source=False,
        spectral_reference_index=1,
        compare_flux=False,
    ),
    _make(
        "static_medium2d_sheet",
        "media",
        "nondispersive Medium2D sheet reflection and transmission",
        _static_medium2d_scene,
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "material_region_slab",
        "media",
        "uniform MaterialRegion density lowered to a homogeneous slab",
        _material_region_scene,
        run_time_factor=RUN_TIME_FACTOR,
    ),
    _make(
        "dispersive_kerr_slab",
        "media",
        "Lorentz dispersion composed with strong-drive Kerr response",
        _dispersive_kerr_scene,
        run_time_factor=RUN_TIME_FACTOR,
    ),
)
