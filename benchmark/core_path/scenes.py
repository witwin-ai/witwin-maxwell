"""Declarative core correctness scenes and the computational-grid size ladder."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import witwin.maxwell as mw

from benchmark.scenes import CPML_PINNED_SCENARIOS, SCENARIOS, build_scene as build_benchmark_scene

from .pinned_scenes import PINNED_SCENES


G2_SCENES = (
    "planewave_vacuum", "pml_only",
    "dipole_vacuum", "dipole_ey", "dipole_offcenter", "dielectric_slab",
    "dielectric_sphere", "pec_sphere", "sigma_e_slab", "anisotropic_uniform_grid",
)
EXTRA_SCENES = (
    "mesh_dielectric", "boolean_dielectric", "multi_structure_dielectric",
    "cpml_dense", "cpml_compressed", "odd_grid",
    "cw_dielectric", "gaussian_beam_dielectric",
    "subpixel_kottke_edge", "subpixel_kottke_smooth", "subpixel_polarized",
    "closed_surface_box", "closed_surface_sampled",
)
O2A_SCENES = (
    "o2_short", "o2_remainder", "o2_zero", "o2_nonzero",
    "o2_pulse_overlap", "o2_cw_overlap", "o2_soft_wave", "o2_tfsf_pulse",
    "o2_tfsf_cw", "o2_beam", "o2_early_shutoff", "o2_capture_failure",
    "o2_delayed_inline", "o2_boundaries",
)
ABSORBER_SCENES = (
    "absorber_stablepml", "absorber_pml", "absorber_graded",
    "closed_transverse_graded", "closed_transverse_cpml", "boundary_none", "boundary_pec",
)
W1_SCENES = (
    "w1_copper_patch", "w1_thin_sheet_cavity", "w1_lumped_port", "w1_coax_port",
    "w1_thin_wire", "w1_cut_surface", "w1_sheet_feed", "w1_sibc",
    "f3_coarse_11f", "f4_dipole_32", "w1_circuit", "w1_network",
    "w1_terminal_port", "w1_wave_port",
)
W2_SCENES = (
    "copper_patch", "copper_board_60k", "f3_coarse_11f",
    "w1_coax_port", "w1_lumped_port", "w1_thin_wire",
)
W3_SCENES = (
    "kerr_slab", "kerr_slab_strong", "tpa_slab", "modulated_slab", "modulated_slab_phase",
    "breakdown_slab",
    "w3_modulation_two_omega", "w3_modulation_cpml_dense", "w3_modulation_cpml_slab",
    "w3_modulation_no_cpml", "w3_modulation_kerr", "w3_modulation_tpa",
    "w3_debye_kerr", "w3_debye_modulation", "w3_debye_modulation_tpa",
)
SRCMON_SCENES = (
    "astigmatic_beam", "uniform_current", "custom_field_source", "custom_current_source",
    "mode_source_wg", "mode_source_higher_order", "magnetic_current_vacuum",
    "mode_monitor_straight_wg", "mode_monitor_two_planes", "ring_resonator_s21",
    "waveguide_s_matrix", "grating_diffraction", "diffraction_normal_orders",
    "permittivity_monitor_slab", "time_monitor_vacuum", "pmc_cavity",
)
DISPERSIVE_SCENES = (
    "debye_slab", "drude_slab", "lorentz_slab", "lorentz_two_pole_slab",
    "drude_lorentz_slab", "debye_sphere", "sellmeier_slab", "sellmeier_sphere",
    "metal_sphere", "lorentz_resonator", "sigma_e_drude_slab", "custom_pole_uniform_slab",
    "graphene_sheet", "dispersive_lumped_port", "magnetic_lorentz_slab",
)
# W7 row 4 admissions: PMC faces (declared and symmetry planes) on every axis and side, with
# no, slab and dense CPML; permeability, magnetic ADE and magnetic conductivity; first- and
# second-order Mur, mode-matched and broadband one-way faces, also with separate physics;
# internal absorber planes; medium, projection and Gaussian-overlap monitors.
W7R4_SCENES = (
    "pmc_cavity", "symmetry_pmc_center", "pmc_faces_slab", "pmc_faces_dense",
    "mu_slab", "sigma_m_slab", "mur_faces", "mur2_faces", "mur_lumped_port",
    "mode_abc_guide", "mode_abc_broadband", "internal_absorber_guide",
    "medium_monitor_slab", "projection_monitors", "overlap_monitors",
)
W7R4_BENCHMARK_SCENES = ("symmetry_pmc_center",)
# Full (off-diagonal) anisotropy: the census tensor slab (TFSF, CPML overlap), a conductive
# crystal around a dipole (all off-diagonals, conduction closure, tensor current source, CPML
# without overlap) and a periodic Debye crystal (graded absorber, periodic wrap, tensor ADE).
ANISO_SCENES = ("full_tensor_slab", "aniso_conductive_dipole", "aniso_debye_periodic")
# W7 row 5: real-field closed-transverse CPML (periodic/PEC-closed transverse faces with an explicit
# CPML absorber), where the passivity guard samples at K-chunk boundaries. The absorber cell and
# every CPML-pinned benchmark scenario; the guard is skipped (uncounted storage) in graphene_sheet
# and dispersive_kerr_slab.
W7R5_SCENES = ("closed_transverse_cpml", *sorted(CPML_PINNED_SCENARIOS))
# Gyromagnetic ferrites: an axis-aligned (-y) bias around a vector dipole (fast path, slab CPML),
# an oblique conductive ferrite under a TFSF plane wave (general path, dense CPML), opposed +z/-z
# latching regions in an unbounded cell (mixed-bias general path) and a ferrite beside a lumped
# port (the separate-physics closure).
FERRITE_SCENES = ("ferrite_axis_dipole", "ferrite_oblique_tfsf", "ferrite_mixed_bias", "ferrite_lumped_port")
# W7 row 12: float64 fields inside the bounded FP64 forward admission (direct Norton lumped ports, Box
# scalar media and two-sided copper, all-face dense CPML, one frequency): a vacuum port with one point
# monitor (native point observers), a lossy slab on a graded grid (indexed coefficients) with finite-plane
# and closed-surface spectra (grouped observer gather), copper foil over a lossy board, and a driven
# plus an undriven port without monitors.
FP64_SCENES = ("fp64_vacuum_port", "fp64_slab_port", "fp64_copper_port", "fp64_two_ports")
# W7 row 13: complex (Bloch split) fields on the core path. Fixed-k TFSF (the census Bloch rows and a
# Debye slab), a CW TFSF slab (discrete CW provider: Bloch CW patch rows), a zero-k pulsed slab
# (time-shifted Bloch patches: per-step core loop), Bloch dipoles (scalar Bloch rows, complex point,
# plane and flux spectra) on every Bloch/CPML electric layout (x/y Bloch + z PML slab and dense,
# single PML axis x, Bloch x with PML y/z, Bloch x/y with PEC z and no CPML).
BLOCH_SCENES = (
    "bloch_oblique", "bloch_oblique_te", "bloch_debye_tfsf", "bloch_cw_tfsf", "bloch_pulse_k0",
    "bloch_dipole_slab", "bloch_dipole_dense", "bloch_pml_x", "bloch_multi_pml", "bloch_pec_z",
)
# W7 row 10: bent coordinate charts (metric folded into the constitutive tables at compile time): a
# concave chart around a lossy board (slab CPML, point and closed-surface spectra), the public split
# copper fixture (chart copper metric factors and response scaling, dense CPML), a convex chart on a
# graded u grid with PEC u faces (closed-transverse CPML under the passivity guard), a driven and a
# loaded port and a finite chart-face plane, and the float64 public fixture.
CHART_SCENES = ("chart_board_port", "chart_copper_port", "chart_convex_two_ports", "chart_fp64_copper")
# W7 row 11: runs with an FDTDMaterialCache bound, on the core path. Each scene is a gate scene (the
# value) prepared through a cache in the outcome its name gives: miss (first bind), hit (the second
# bind of the same scene) or local (a Box recompile from a warm-up with the slab moved). Its payloads
# must equal the base scene's frozen payloads: a Mesh dielectric, a slab, a Debye slab, a copper foil,
# copper terminals with a port and a fixed-k Bloch slab.
CACHE_SCENES = {
    "cache_miss_mesh_dielectric": "mesh_dielectric",
    "cache_hit_mesh_dielectric": "mesh_dielectric",
    "cache_local_dielectric_slab": "dielectric_slab",
    "cache_hit_debye_slab": "debye_slab",
    "cache_hit_w1_copper_patch": "w1_copper_patch",
    "cache_hit_w1_terminal_port": "w1_terminal_port",
    "cache_hit_bloch_oblique": "bloch_oblique",
}
# W7 row 16a: the last monitor types on the core path. Power loss (conduction and native electric
# dispersion over a conductive Debye slab with both quadratures, beside a copper foil with its side
# spectra, and on a bent chart), dipole emission (a point observer group), component stress over a
# driven port's field time series, surface and auxiliary-state (ADE) time traces over a Debye slab,
# and wire monitors (host-driven step-loop observers) on a thin wire.
MONITOR_SCENES = (
    "power_loss_slab", "power_loss_copper", "power_loss_chart", "dipole_emission",
    "component_stress_port", "time_surface_aux_debye", "wire_monitor",
)
# W7 row 16b: complex (Bloch split) fields beside the separate-physics pieces their runtimes admit
# (passive LTI lumped elements, undriven lumped ports, copper foil). Bloch x/y with a z PML: R, L and C
# elements beside a dipole (whole-step separate-physics graph); undriven ports with series and parallel RLC
# terminations and the reference load, one inside a Debye slab (imaginary pole-history bank); a
# Bloch-periodic copper foil on a conductive Debye substrate under a CW current drive with a PML x axis,
# and on an all-Bloch cell under a pulse beside a resistor; a fixed-k TFSF slab with a resistor (per-step
# core loop: TFSF with separate physics).
BLOCH_SEPARATE_SCENES = (
    "bloch_lumped_elements", "bloch_loaded_ports", "bloch_copper_foil", "bloch_copper_resistor",
    "bloch_tfsf_resistor",
)
CORE_SCENES = tuple(dict.fromkeys(
    G2_SCENES + EXTRA_SCENES + ABSORBER_SCENES + SRCMON_SCENES + W1_SCENES + W2_SCENES
    + W3_SCENES + DISPERSIVE_SCENES + W7R4_SCENES + ANISO_SCENES + W7R5_SCENES + FERRITE_SCENES
    + FP64_SCENES + BLOCH_SCENES + CHART_SCENES + tuple(CACHE_SCENES) + MONITOR_SCENES
    + BLOCH_SEPARATE_SCENES))
GATE_SCENES = CORE_SCENES + O2A_SCENES
SIZES = (64, 128, 256, 384)
FREQUENCY = 2.0e9
PML_LAYERS = 8


@dataclass(frozen=True)
class CoreCase:
    name: str
    scene: mw.Scene
    frequencies: tuple[float, ...] = (FREQUENCY,)
    cpml_memory: str = "slab"
    shape: tuple[int, int, int] | None = None
    steps: int | None = None
    initial_fields: bool = False
    capture_failure: bool = False
    shutoff: float = 0.0
    boundaries: bool = False
    absorber: str | None = None
    excitations: tuple = ()
    cpml_kappa_max: float = 1.0
    field_dtype: torch.dtype = torch.float32
    # Cache scenes keep their base scene's name; the cache outcome and the warm-up scene of a local bind.
    material_cache: str | None = None
    cache_warmup: mw.Scene | None = None

    def plan(self) -> dict:
        return {
            "scene": self.name,
            "computational_primal_shape": self.shape,
            "structures": len(self.scene.structures),
            "sources": [type(item).__name__ for item in self.scene.sources],
            "monitors": [type(item).__name__ for item in self.scene.monitors],
            "ports": [type(item).__name__ for item in self.scene.ports],
            "thin_wires": len(self.scene.thin_wires),
            "frequencies": self.frequencies, "cpml_memory": self.cpml_memory,
            "cpml_kappa_max": self.cpml_kappa_max,
        }


def _grid_scene(
    shape: tuple[int, int, int], *,
    source=mw.PointDipole(
        position=(0.0, 0.0, -0.28), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
    ),
    subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
) -> mw.Scene:
    # CPML extends the physical grid by eight nodes on each face. Custom nodes
    # make the requested computational shape exact, without ceil-rounding drift.
    nodes = tuple(np.linspace(-0.64, 0.64, n - 2 * PML_LAYERS) for n in shape)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64),) * 3),
        grid=mw.GridSpec.custom(*nodes),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        subpixel=subpixel,
        sources=(source,),
    )


def _sphere(scene: mw.Scene) -> mw.Scene:
    return scene.updated(structures=(
        *scene.structures,
        mw.Structure(
            geometry=mw.Sphere(position=(0, 0, 0), radius=0.12, device="cpu"),
            material=mw.Material(eps_r=4.0), name="dielectric_sphere",
        ),
    ))


def _absorber_case(name: str) -> CoreCase:
    # Open-boundary sphere scenes for the non-CPML absorber families, and a
    # closed-transverse (periodic x/y, absorbing z) cell for the topology rule.
    if name.startswith("absorber_"):
        scene = _sphere(_grid_scene((64, 64, 64)))
        absorber = {"absorber_stablepml": "stablepml", "absorber_pml": "pml",
                    "absorber_graded": "absorber"}[name]
        shape = (64, 64, 64)
    elif name.startswith("boundary_"):
        # The same sphere scene without absorbing faces: unbounded (none) and closed PEC.
        boundary = mw.BoundarySpec.none() if name == "boundary_none" else mw.BoundarySpec.pec()
        scene = _sphere(_grid_scene((64, 64, 64))).updated(boundary=boundary)
        absorber, shape = None, None
    else:
        layers = 40
        nodes = np.linspace(-0.48, 0.48, 48)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.48, 0.48),) * 3),
            grid=mw.GridSpec.custom(nodes, nodes, nodes),
            boundary=mw.BoundarySpec.faces(default="periodic", num_layers=layers, strength=1e6, z="pml"),
            subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
            sources=(
                mw.PointDipole(
                    position=(0.0, 0.0, -0.2), polarization="Ex",
                    source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
                ),
            ),
        )
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(
                geometry=mw.Box(position=(0, 0, 0), size=(0.96, 0.96, 0.12), device="cpu"),
                material=mw.Material(eps_r=4.0), name="slab",
            ),
        ))
        absorber = "absorber" if name == "closed_transverse_graded" else "cpml"
        shape = None
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.PointMonitor("point", position=(0.1, 0.1, 0.25), fields=("Ex", "Ez")),
        mw.PlaneMonitor(
            "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
        ),
    ))
    return CoreCase(name, scene, absorber=absorber, shape=shape)


def build_core_case(name: str) -> CoreCase:
    if name in CACHE_SCENES:
        return _cache_case(name)
    if name in W3_SCENES and name.startswith(("w3_modulation_", "w3_debye_")):
        frequency = 1e9
        no_cpml = name == "w3_modulation_no_cpml"
        nonlinear = name.endswith(("_kerr", "_tpa"))
        nodes = np.linspace(-0.32, 0.32, 48 if no_cpml else 48 - 2 * PML_LAYERS)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
            grid=mw.GridSpec.custom(nodes, nodes, nodes),
            boundary=(mw.BoundarySpec.pec() if no_cpml
                      else mw.BoundarySpec.pml(num_layers=PML_LAYERS)),
            subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
            sources=(
                mw.PlaneWave(
                    direction=(1., 0., 0.), polarization=(0., 0., 1.),
                    source_time=mw.CW(frequency=frequency, amplitude=1e6 if nonlinear else 50.),
                ),
            ),
        )
        if name == "w3_modulation_two_omega":
            scene = scene.updated(structures=(
                *scene.structures,
                *(mw.Structure(
                      geometry=mw.Box(position=(x, 0., 0.), size=(0.12, 0.5, 0.5), device="cpu"),
                      material=mw.Material(eps_r=4., modulation=mw.media.ModulationSpec(
                      frequency=modulation_frequency, amplitude=0.2, phase=phase)),
                  ) for x, modulation_frequency, phase in ((-0.16, 2.5e8, 0.31), (0.16, 4e8, -0.27))),
            ))
        else:
            # The modulation/nonlinearity/Debye composition fixtures, without time observers.
            material_options = {}
            if name.endswith("_kerr"):
                material_options["kerr_chi3"] = 1e-16
            elif name.endswith("_tpa"):
                material_options["nonlinearity"] = mw.media.TwoPhotonAbsorption(beta=1e-10)
            if "modulation" in name:
                material_options["modulation"] = mw.media.ModulationSpec(frequency=2.5e8, amplitude=0.2, phase=0.31)
            if name.startswith("w3_debye_"):
                material_options["debye_poles"] = (mw.DebyePole(delta_eps=1., tau=8e-11),)
            scene = scene.updated(structures=(
                *scene.structures,
                mw.Structure(
                    geometry=mw.Box(position=(0., 0., 0.), size=(0.24, 0.64, 0.64), device="cpu"),
                    material=mw.Material(eps_r=4., **material_options),
                ),
            ))
        frequencies = (0.75e9, frequency, 1.25e9)
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PointMonitor("point", position=(0.28, 0., 0.), fields=("Ez",)),
            mw.PlaneMonitor(
                "plane", axis="x", position=0.28, fields=("Ez",), frequencies=frequencies,
            ),
        ))
        return CoreCase(
            name, scene, frequencies, shape=(48, 48, 48), steps=256,
            cpml_memory="dense" if name == "w3_modulation_cpml_dense" else "slab",
            cpml_kappa_max=5.,
        )
    if name == "breakdown_slab":
        scene = _grid_scene((40, 40, 40), source=mw.PointDipole(
            position=(0., 0., 0.), polarization="Ez",
            source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9, amplitude=1e3),
        ))
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(
                geometry=mw.Box(position=(0., 0., 0.), size=(0.6,) * 3, device="cpu"),
                material=mw.Material(eps_r=2., breakdown=mw.breakdown.DielectricBreakdown(
                critical_field=1e-8, minimum_duration=2e-10,
                post_breakdown_conductivity=0.1, ramp_time=1e-9)),
            ),
        ), monitors=(
            *scene.monitors,
            mw.PointMonitor("point", position=(0., 0., 0.), fields=("Ez",)),
            mw.esd.BreakdownMonitor(
                "stress", position=(0., 0., 0.), size=(0.3,) * 3, critical_field=1e-8,
            ),
        ))
        return CoreCase(name, scene, shape=(40, 40, 40), initial_fields=True)
    if name == "magnetic_lorentz_slab":
        scene = build_benchmark_scene("lorentz_slab")
        scene = scene.updated(structures=[mw.Structure(
            geometry=structure.geometry,
            material=mw.Material(mu_lorentz_poles=(mw.LorentzPole(
                delta_eps=1., resonance_frequency=3.5e9, gamma=.25e9),)),
            name=structure.name,
        ) for structure in scene.structures])
        return CoreCase(name, scene, SCENARIOS["lorentz_slab"].frequencies)
    if name == "dispersive_lumped_port":
        case = _w1_case("w1_lumped_port")
        scene = case.scene.updated(structures=(
            *case.scene.structures,
            mw.Structure(
                mw.Box(position=(0, 0, 0), size=(.03, .03, .03), device="cpu"),
                mw.Material.debye(eps_inf=2., delta_eps=1., tau=8e-11), name="dispersive_host"),
        ))
        return CoreCase(name, scene, case.frequencies, excitations=case.excitations)
    if name in ("copper_patch", "copper_board_60k"):
        from scripts.dev.fdtd.thin_sheet_overhead import SCENARIOS as SHEET_SCENARIOS, _scene
        # Match the D3b wrapper's frequency, geometry, source and default time-step settings.
        return CoreCase(name, _scene(SHEET_SCENARIOS[name], True), (2.4e9,))
    if name in ("aniso_conductive_dipole", "aniso_debye_periodic"):
        return _aniso_case(name)
    if name in FERRITE_SCENES:
        return _ferrite_case(name)
    if name in FP64_SCENES:
        return _fp64_case(name)
    if name in BLOCH_SCENES and name not in ("bloch_oblique", "bloch_oblique_te"):
        return _bloch_case(name)
    if name in CHART_SCENES:
        return _chart_case(name)
    if name in MONITOR_SCENES:
        return _monitor_case(name)
    if name in BLOCH_SEPARATE_SCENES:
        return _bloch_separate_case(name)
    if name in W1_SCENES:
        return _w1_case(name)
    if name in W7R4_SCENES and name not in SRCMON_SCENES + W7R4_BENCHMARK_SCENES:
        return _w7r4_case(name)
    if name in ABSORBER_SCENES:
        return _absorber_case(name)
    if name in O2A_SCENES:
        return _o2a_case(name)
    if name == "symmetry_pmc_center":
        # The admission gate keeps the standard bounds with the PMC plane on the y-low face
        # (its frozen payloads); the benchmark row is now the kept half of that scene.
        from benchmark.scenes.coverage.boundaries import _symmetry_scene

        return CoreCase(name, _symmetry_scene((None, "PMC", None), folded=False), SCENARIOS[name].frequencies)
    if name in (G2_SCENES + SRCMON_SCENES + W3_SCENES + DISPERSIVE_SCENES + W7R4_BENCHMARK_SCENES
                + ("full_tensor_slab", "bloch_oblique", "bloch_oblique_te") + tuple(CPML_PINNED_SCENARIOS)):
        if name in PINNED_SCENES:
            # Re-specified in the benchmark registry: the gate keeps its frozen-store definition.
            scene, frequencies, absorber = PINNED_SCENES[name]()
            return CoreCase(name, scene, frequencies, absorber=absorber)
        pinned = SCENARIOS[name].absorber
        return CoreCase(name, build_benchmark_scene(name), SCENARIOS[name].frequencies,
                        absorber=None if pinned == "auto" else pinned)
    shape = (61, 47, 53) if name == "odd_grid" else (64, 64, 64)
    if name == "cw_dielectric":
        scene = _grid_scene(shape, source=mw.PointDipole(
            position=(0.0, 0.0, -0.28), polarization="Ez",
            source_time=mw.CW(frequency=FREQUENCY),
        ))
    elif name == "gaussian_beam_dielectric":
        scene = _grid_scene(shape, source=mw.GaussianBeam(
            direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
            beam_waist=0.2, focus=(0.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
        ))
    elif name.startswith("subpixel_"):
        scene = _grid_scene(shape, subpixel=mw.SubpixelSpec(
            samples=4, averaging=name.removeprefix("subpixel_"), pec="staircase",
        ))
    else:
        scene = _grid_scene(shape)
    if name == "mesh_dielectric":
        vertices, faces = mw.geometry.Pyramid(
            position=(0, 0, -0.12), base_size=0.3, height=0.24, device="cpu",
        ).to_mesh()
        geometry = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid", device="cpu")
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0)),
        ))
    elif name == "boolean_dielectric":
        geometry = mw.geometry.Difference(
            mw.Box(position=(0, 0, 0), size=(0.32, 0.3, 0.28), device="cpu"),
            mw.Sphere(position=(0.12, 0, 0), radius=0.15, device="cpu"), device="cpu",
        )
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(geometry=geometry, material=mw.Material(eps_r=3.0)),
        ))
    elif name == "multi_structure_dielectric":
        scene = _sphere(scene)
        scene = scene.updated(structures=(
            *scene.structures,
            *(mw.Structure(
                  geometry=mw.Box(position=(x, 0, 0), size=(0.12, 0.24, 0.2), device="cpu"),
                  material=mw.Material(eps_r=eps),
              ) for x, eps in ((-0.24, 2.0), (0.24, 6.0))),
        ))
    elif name.startswith("subpixel_"):
        scene = _sphere(scene)
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(
                geometry=mw.Box(position=(0.24, 0, 0), size=(0.12, 0.24, 0.2), device="cpu"),
                material=mw.Material(eps_r=3.0),
            ),
        ))
    elif name in ("cpml_dense", "cpml_compressed", "odd_grid", "cw_dielectric", "gaussian_beam_dielectric",
                  "closed_surface_box", "closed_surface_sampled"):
        scene = _sphere(scene)
    else:
        raise KeyError(name)
    if name.startswith("closed_surface_"):
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.monitors.ClosedSurfaceMonitor.box(
                "surface", size=(0.6, 0.6, 0.6), frequencies=(FREQUENCY,),
                sample_spacing=0.02 if name == "closed_surface_sampled" else None,
            ),
        ))
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")),
        mw.PlaneMonitor(
            "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
        ),
        mw.FluxMonitor("flux", axis="z", position=0.3, frequencies=(FREQUENCY,)),
    ))
    return CoreCase(name, scene, cpml_memory="dense" if name == "cpml_dense" else "slab", shape=shape)


def _o2a_case(name):
    frequencies = (1.5e9, FREQUENCY, 2.5e9)
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    waveform = mw.CW(frequency=FREQUENCY, phase=0.31) if name in ("o2_cw_overlap", "o2_tfsf_cw", "o2_soft_wave") else pulse
    source = mw.PointDipole(position=(0, 0, -0.28), polarization="Ez", source_time=waveform)
    if name in ("o2_soft_wave", "o2_tfsf_pulse", "o2_tfsf_cw", "o2_capture_failure"):
        injection = mw.TFSF(bounds=((-0.4, 0.4),) * 3) if name != "o2_soft_wave" else "soft"
        direction, polarization = ((0.2, 0, 1), (0, 1, 0)) if name == "o2_soft_wave" else ((0, 0, 1), (1, 0, 0))
        source = mw.PlaneWave(direction=direction, polarization=polarization,
                              source_time=waveform, injection=injection)
    if name in ("o2_beam", "o2_delayed_inline"):
        source = mw.GaussianBeam(direction=(0, 0, 1), polarization=(1, 0, 0),
            beam_waist=0.2, focus=(0, 0, 0), source_time=pulse)
    scene = _grid_scene((64, 64, 64), source=source)
    if name in ("o2_pulse_overlap", "o2_cw_overlap"):
        scene = scene.updated(sources=(
            *scene.sources,
            mw.PointDipole(position=(0, 0, -0.28), polarization="Ez",
                source_time=(mw.CW(frequency=FREQUENCY, amplitude=-0.3, phase=-0.2) if name == "o2_cw_overlap"
            else mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9, amplitude=-0.3, phase=-0.2))),
        ))
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")),
        mw.PlaneMonitor("plane", axis="y", position=0, fields=("Ex", "Ey", "Ez"), frequencies=frequencies),
        mw.FluxMonitor("flux", axis="z", position=0.3, frequencies=frequencies),
    ))
    steps = {"o2_short": 3, "o2_remainder": 35, "o2_zero": 0,
             "o2_early_shutoff": 2048, "o2_boundaries": 67}.get(name)
    return CoreCase(name, scene, frequencies, shape=(64, 64, 64), steps=steps,
        initial_fields=name in ("o2_nonzero", "o2_capture_failure"),
        capture_failure=name == "o2_capture_failure", shutoff=0.99 if name == "o2_early_shutoff" else 0.0,
        boundaries=name == "o2_boundaries")


def _w1_case(name: str) -> CoreCase:
    if name.startswith(("f3_", "f4_")):
        from benchmark.datagen.scenes import scene_builders
        builder = next(builder for builder in scene_builders() if builder.name == name)
        sample = builder.build()
        return CoreCase(name, sample.scene, tuple(sample.fdtd_kwargs["frequencies"]),
                        excitations=tuple(sample.fdtd_kwargs["excitations"]))

    frequency = 1e9
    excitations = ()
    if name == "w1_terminal_port":
        from tests.rf.terminal.test_planar_terminal_targets import _scene
        scene = _scene(metal="copper")
        excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name == "w1_wave_port":
        # Exercise the ModeSource/ModeMonitor carrier used by lowered wave-port columns.
        from benchmark.scenes.rf.rectangular_waveguide import rectangular_waveguide_reference_scene
        frequency = .4e9
        scene = rectangular_waveguide_reference_scene(frequencies=(frequency,))
    elif name == "w1_thin_sheet_cavity":
        from tests.materials.thin_sheet.test_thin_sheet_runtime import _cavity, _plate
        scene = _cavity(_plate(mw.Material.pec(), sheet_model="conformal_two_sided"))
        frequency = 2e9
    elif name == "w1_sheet_feed":
        from tests.materials.thin_sheet.test_thin_sheet_feeds import _runtime_scene
        scene = _runtime_scene(tilted=True)
        excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name == "w1_coax_port":
        # The shielded coax with feed, probe and load of the port-step bit-identity fixture, built on CPU.
        h, a, b, z0, z1, half, length = 0.25e-3, 1.0e-3, 2.3e-3, 1.0e-3, 6.0e-3, 3.0e-3, 7.0e-3
        circle = [(b * np.cos(2 * np.pi * k / 96), b * np.sin(2 * np.pi * k / 96)) for k in range(96)]
        square = [(-half - h, -half - h), (half + h, -half - h), (half + h, half + h), (-half - h, half + h)]
        scene = mw.Scene(subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase"),
                         domain=mw.Domain(bounds=((-half, half), (-half, half), (0.0, length))),
                         grid=mw.GridSpec.uniform(h), boundary=mw.BoundarySpec.pec(),
                         structures=(
                             mw.Structure(mw.geometry.ComplexPolySlab([square, circle], bounds=(0.0, length), axis="z"),
                                          mw.Material.pec(), name="shield"),
                             mw.Structure(mw.Cylinder(position=(0, 0, 0.5 * (z0 + z1)), radius=a, height=z1 - z0,
                                                      axis="z"), mw.Material.pec(), name="inner"),
                         ),
                         ports=(
                             mw.rf.CoaxialLumpedPort("feed", center=(0.0, 0.0, z0), inner_diameter=2 * a, outer_diameter=2 * b,
                                                  normal_axis="z", reference_impedance=50.0),
                             mw.rf.CoaxialLumpedPort("probe", center=(0.0, 0.0, 3.5e-3), inner_diameter=2 * a,
                                                  outer_diameter=2 * b, normal_axis="z"),
                         ),
                         lumped_elements=(
                             mw.rf.CoaxialLumpedResistor("load", center=(0, 0, z1), inner_diameter=2 * a,
                                                      outer_diameter=2 * b, normal_axis="z", resistance=75.0),
                         ))
        excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name in ("w1_copper_patch", "w1_lumped_port", "w1_circuit", "w1_network"):
        # The small gap/foil fixture in test_copper_foil_execution._simulation.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),)*3),
                         grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none())
        if name == "w1_copper_patch":
            from benchmark.datagen.scenes import _box
            scene = scene.updated(structures=(
                *scene.structures,
                mw.Structure(
                    _box((.01, 0, 0), (0, .02, .02)),
                    mw.media.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)), name="foil"),
            ))
        if name in ("w1_lumped_port", "w1_circuit", "w1_network"):
            scene = scene.updated(ports=(
                *scene.ports,
                mw.LumpedPort(
                    "feed", positive=(0, 0, .005), negative=(0, 0, -.005), voltage_path=mw.AxisPath("z"),
                    current_surface=mw.Box(position=(0, 0, -.0025), size=(.015, .015, 0), device="cpu")),
            ))
            if name == "w1_lumped_port":
                scene = scene.updated(lumped_elements=(
                    *scene.lumped_elements,
                    mw.rf.Resistor(
                        "load", positive=(.01, 0, .005), negative=(.01, 0, -.005), resistance=100.),
                ))
                excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
            elif name == "w1_circuit":
                # The resistive, field-bound circuit in the port graph fixture.
                circuit = mw.circuits.Circuit("driven")
                node = circuit.node("input")
                circuit.add(mw.rf.Resistor("R1", node, circuit.ground, 50.))
                circuit.add(mw.circuits.CurrentSource("I1", circuit.ground, node,
                                             waveform=mw.circuits.SineWaveform(0., .01, 1e9)))
                circuit.bind_port("feed", positive=node, negative=circuit.ground)
                scene = scene.updated(circuits=(circuit,))
            else:
                # One-port passive specialization of the embedded-network fixture.
                model = mw.rf.rational.StateSpaceNetwork(
                    A=torch.tensor([[-4e9]], dtype=torch.float64),
                    B=torch.tensor([[4e9]], dtype=torch.float64),
                    C=torch.tensor([[-.02]], dtype=torch.float64),
                    D=torch.tensor([[.022]], dtype=torch.float64),
                    representation="Y", port_order=("load",))
                # Cover the pulse's effective band, which starts at DC.
                band = torch.linspace(0., 3e9, 13, dtype=torch.float64)
                scene = scene.updated(networks=(mw.rf.NetworkBlock(
                    "load", model=model, fit=False,
                    network=mw.rf.NetworkData.from_y(frequencies=band, y=model.evaluate(band),
                                                  z0=50., port_names=model.port_order),
                    connections={"load": "feed"}),))
        # The circuit's CW current source drives that scene alone; mixing it with a pulse is refused.
        if not excitations and name != "w1_circuit":
            scene = scene.updated(sources=(
                *scene.sources,
                mw.PointDipole(position=(-.01, 0, -.01), polarization="Ez", profile="ideal",
                              source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),
            ))
    elif name == "w1_thin_wire":
        # test_thin_wire_forward._scene, with an admitted point readout.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.12, .12),)*3),
                         grid=mw.GridSpec.uniform(.04), boundary=mw.BoundarySpec.none(),
                         thin_wires=(
                             mw.rf.ThinWire("wire", ((0, 0, -.08), (0, 0, .08)), .002,
                                        mw.rf.WireConductor.pec(), snap="strict"),
                         ),
                         sources=(
                             mw.PointDipole(position=(0, 0, .02), polarization="Ez", width=.04,
                                           source_time=mw.CW(2e9, amplitude=10.)),
                         ))
        frequency = 2e9
    elif name == "w1_cut_surface":
        # The oblique two-triangle patch in test_cut_surface._mesh.
        vertices = np.array(((-.008, -.008, 0), (.008, -.008, 0),
                             (.008, .008, 0), (-.008, .008, 0)))
        vertices[:, 2] = .3*vertices[:, 0] + .001
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),)*3),
                         grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(),
                         structures=(
                             mw.Structure(
                                 mw.geometry.SurfaceMesh(vertices, ((0, 1, 2), (0, 2, 3)), device="cpu"), mw.Material.pec()),
                         ),
                         sources=(
                             mw.PointDipole(position=(-.025, 0, -.02), polarization="Ez", profile="ideal",
                                           source_time=mw.CW(1e9)),
                         ))
    elif name == "w1_sibc":
        # Six finite exposed faces from test_surface_impedance_runtime.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.03, .03),)*3),
                         grid=mw.GridSpec.uniform(.004), boundary=mw.BoundarySpec.pml(num_layers=8),
                         structures=(
                             mw.Structure(
                                 mw.Box(position=(.008, 0, 0), size=(.012, .02, .02), device="cpu"),
                                 mw.media.LossyMetalMedium(conductivity=50.)),
                         ),
                         sources=(
                             mw.PointDipole(position=(-.012, 0, 0), polarization="Ez", width=.008,
                                           source_time=mw.CW(2e9, amplitude=40.)),
                         ))
        frequency = 2e9
    else:
        raise KeyError(name)
    if not scene.monitors:
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PointMonitor("point", position=(0, .01, 0), fields=("Ex", "Ez")),
        ))
    return CoreCase(name, scene, (frequency,), excitations=excitations)


def _point_plane_flux(scene: mw.Scene) -> mw.Scene:
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")),
        mw.PlaneMonitor(
            "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
        ),
    ))
    return scene.updated(monitors=(*scene.monitors, mw.FluxMonitor("flux", axis="z", position=0.3, frequencies=(FREQUENCY,))))


def _slab(scene: mw.Scene, x: float, material: mw.Material) -> mw.Scene:
    return scene.updated(structures=(
        *scene.structures,
        mw.Structure(
            geometry=mw.Box(position=(x, 0.0, 0.0), size=(0.2, 0.6, 0.6), device="cpu"), material=material,
        ),
    ))


def _guide_case(name: str) -> CoreCase:
    # The eps_r 4 guide of tests/boundaries/absorbers/test_one_way_absorbers.py, shortened.
    frequencies = (1.6e9, 2.0e9, 2.4e9)
    plane = (0.0, float("inf"), float("inf"))
    single = mw.fdtd.ModeABC(polarization="Ez", frequency=2.0e9)
    absorbers, x_high = (), 0.205
    if name == "internal_absorber_guide":
        # A mode-matched plane (native Mur kernel) downstream and a second-order Mur
        # (Higdon) plane behind the source, inside a PML-terminated domain.
        x_high = 0.405
        absorbers = (mw.fdtd.InternalAbsorber((0.205, 0, 0), plane, "+", boundary=single, name="sink"),
                     mw.fdtd.InternalAbsorber((-0.29, 0, 0), plane, "-", boundary=mw.fdtd.MurABC(order=2),
                                              name="back"))
        boundary = mw.BoundarySpec.pml(num_layers=PML_LAYERS)
    else:
        spec = single if name == "mode_abc_guide" else mw.fdtd.ModeABC(
            polarization="Ez", frequency_range=(1.5e9, 2.5e9), order=2)
        boundary = mw.BoundarySpec.faces(default="pml", num_layers=PML_LAYERS, x_high="mode_abc", mode_abc=spec)
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.32, x_high), (-0.32, 0.32), (-0.32, 0.32))),
                grid=mw.GridSpec.uniform(0.0125), boundary=boundary,
                subpixel=mw.SubpixelSpec(samples=3, averaging="polarized"),
                internal_absorbers=absorbers,
                structures=(
            mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(20.0, 0.20, 0.30), device="cpu"),
                         material=mw.Material(eps_r=4.0)),
            ),
                sources=(
            mw.ModeSource(position=(-0.25, 0, 0), size=(0, 0.5, 0.5), polarization="Ez",
                          source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=1.0e9)),
            ),
                monitors=(
            mw.ModeMonitor("mode", position=(0.0, 0, 0), size=(0, 0.5, 0.5), polarization="Ez",
                           frequencies=frequencies),
            mw.PointMonitor("point", position=(0.15, 0.0, 0.0), fields=("Ez",)),
            mw.PlaneMonitor("plane", axis="y", position=0.0, fields=("Ez",), frequencies=frequencies),
            ),
            )
    return CoreCase(name, scene, frequencies)


def _w7r4_case(name: str) -> CoreCase:
    if name in ("mode_abc_guide", "mode_abc_broadband", "internal_absorber_guide"):
        return _guide_case(name)
    if name == "mur_lumped_port":
        case = _w1_case("w1_lumped_port")
        scene = case.scene.updated(boundary=mw.BoundarySpec.faces(default="none", x_high="mur", z_low="mur"))
        return CoreCase(name, scene, case.frequencies, excitations=case.excitations)
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    shape = (64, 64, 64)
    if name.startswith("pmc_faces_"):
        # Each axis carries one PMC face per scene; the two scenes cover both sides of every axis.
        dense = name == "pmc_faces_dense"
        sign = 1.0 if dense else -1.0
        faces = (dict(x_high="pmc", y_low="pmc", z_high="pmc") if dense
                 else dict(x_low="pmc", y_high="pmc", z_low="pmc"))
        scene = _sphere(_grid_scene(shape, source=mw.PointDipole(
            position=(0.5 * sign, -0.5 * sign, 0.5 * sign), polarization="Ex" if dense else "Ez",
            source_time=pulse)))
        scene = scene.updated(boundary=mw.BoundarySpec.faces(default="pml", num_layers=PML_LAYERS, **faces))
        return CoreCase(name, _point_plane_flux(scene), cpml_memory="dense" if dense else "slab")
    if name in ("mur_faces", "mur2_faces"):
        scene = _sphere(_grid_scene((48, 48, 48)))
        boundary = (mw.BoundarySpec.mur() if name == "mur_faces" else mw.BoundarySpec.faces(
            default="pml", num_layers=PML_LAYERS, x_high="mur", y_low="mur", z_high="mur",
            mur_abc=mw.fdtd.MurABC(order=2)))
        return CoreCase(name, _point_plane_flux(scene.updated(boundary=boundary)))
    scene = _grid_scene(shape)
    if name == "mu_slab":
        scene = _slab(scene, -0.3, mw.Material(eps_r=2.0, mu_r=2.0))
        scene = _slab(scene, 0.0, mw.Material(mu_tensor=mw.DiagonalTensor3(2.0, 3.0, 1.5)))
        scene = _slab(scene, 0.3, mw.Material(mu_r=1.5, mu_lorentz_poles=(mw.LorentzPole(
            delta_eps=1.0, resonance_frequency=3.5e9, gamma=0.25e9),)))
    elif name == "sigma_m_slab":
        scene = _slab(scene, -0.2, mw.Material(eps_r=2.0, sigma_m=400.0))
        scene = _slab(scene, 0.2, mw.Material(eps_r=1.5, mu_r=1.5, sigma_e=0.01, sigma_m=200.0))
    elif name == "medium_monitor_slab":
        scene = _sphere(scene)
        scene = _slab(scene, 0.3, mw.Material(eps_r=2.0, mu_r=1.5, sigma_e=0.01))
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.monitors.MediumMonitor("medium", position=(0.1, 0, 0), size=(0.6, 0.4, 0.4),
                                      frequencies=(FREQUENCY,)),
        ))
    elif name == "projection_monitors":
        scene = _sphere(scene)
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.monitors.AngleProjectionMonitor(
                "angle", size=(0.6, 0.6, 0.6), theta=np.linspace(0.2, 2.9, 5), phi=np.linspace(0.0, 3.0, 4),
                frequencies=(FREQUENCY,), radius=10.0),
            mw.monitors.CartesianProjectionMonitor(
                "cartesian", position=(0, 0, 0.35), size=(0.6, 0.6, 0.0), x=np.linspace(-0.5, 0.5, 4),
                y=np.linspace(-0.5, 0.5, 3), distance=5.0, frequencies=(FREQUENCY,)),
            mw.monitors.KSpaceProjectionMonitor(
                "kspace", position=(0, 0, -0.35), size=(0.6, 0.6, 0.0), ux=np.linspace(-0.6, 0.6, 4),
                uy=np.linspace(-0.6, 0.6, 3), normal_direction="-", frequencies=(FREQUENCY,)),
        ))
    elif name == "overlap_monitors":
        scene = _grid_scene(shape, source=mw.GaussianBeam(
            direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), beam_waist=0.2, focus=(0.0, 0.0, 0.0),
            source_time=pulse))
        scene = _sphere(scene)
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.monitors.GaussianOverlapMonitor(
                "gaussian", position=(0, 0, 0.3), size=(0.8, 0.8, 0.0), beam_waist=0.2, frequencies=(FREQUENCY,)),
            mw.monitors.AstigmaticGaussianOverlapMonitor(
                "astigmatic", position=(0, 0, 0.35), size=(0.8, 0.8, 0.0), beam_waist=(0.2, 0.25),
                frequencies=(FREQUENCY,), focus_u=0.05),
        ))
    else:
        raise KeyError(name)
    return CoreCase(name, _point_plane_flux(scene), shape=shape)


def _aniso_case(name: str) -> CoreCase:
    tensor = mw.media.Tensor3x3(((3.0, 0.3, 0.2), (0.3, 2.5, 0.25), (0.2, 0.25, 2.0)))
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    # Polarized subpixel averaging refuses full tensors.
    subpixel = mw.SubpixelSpec(samples=3, averaging="arithmetic", pec="staircase")
    if name == "aniso_conductive_dipole":
        scene = _grid_scene((40, 40, 40), subpixel=subpixel, source=mw.PointDipole(
            position=(0.05, -0.03, 0.04), polarization="Ez", source_time=pulse))
        material = mw.Material(epsilon_tensor=tensor, sigma_e=0.02)
        geometry = mw.Box(position=(0, 0, 0), size=(0.6, 0.5, 0.56), device="cpu")
        monitor = mw.PointMonitor("point", position=(-0.2, 0.15, -0.2), fields=("Ex", "Ey", "Ez"))
    else:
        nodes = np.linspace(-0.32, 0.32, 33)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
            grid=mw.GridSpec.custom(nodes, nodes, nodes),
            # Automatic selection takes the graded absorber here, which needs 40 layers.
            boundary=mw.BoundarySpec.pml(num_layers=40).with_faces(
                x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"),
            subpixel=subpixel,
            sources=(
                mw.PointDipole(position=(0.02, -0.04, -0.22), polarization=(1.0, 0.5, 0.2),
                               source_time=pulse),
            ),
        )
        material = mw.Material(epsilon_tensor=tensor, debye_poles=(mw.DebyePole(delta_eps=1.0, tau=8e-11),))
        # Spans the periodic axes, so the off-diagonal stencil wraps; clear of the z absorber.
        geometry = mw.Box(position=(0, 0, 0.02), size=(1.0, 1.0, 0.24), device="cpu")
        monitor = mw.PlaneMonitor("plane", axis="z", position=0.2, fields=("Ex", "Ey", "Ez"),
                                  frequencies=(FREQUENCY,))
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(geometry=geometry, material=material, name="crystal"),
    ), monitors=(*scene.monitors, monitor))
    return CoreCase(name, scene)


def _ferrite(bias, **options) -> mw.media.GyromagneticFerrite:
    return mw.media.GyromagneticFerrite(eps_r=4.0, saturation_magnetization=1.4e5, bias_field=bias,
                                        gilbert_damping=options.pop("gilbert_damping", 2e-3), **options)


def _ferrite_case(name: str) -> CoreCase:
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    if name == "ferrite_lumped_port":
        case = _w1_case("w1_lumped_port")
        scene = case.scene.updated(structures=(
            *case.scene.structures,
            mw.Structure(
                mw.Box(position=(0.01, 0.01, 0.0), size=(0.015, 0.015, 0.03), device="cpu"),
                _ferrite((0.0, 0.0, 1.75e5)), name="ferrite"),
        ))
        return CoreCase(name, scene, case.frequencies, excitations=case.excitations)
    nodes = np.linspace(-0.32, 0.32, 33)
    if name == "ferrite_axis_dipole":
        boundary = mw.BoundarySpec.pml(num_layers=PML_LAYERS)
        source = mw.PointDipole(position=(0.03, -0.02, 0.04), polarization=(1.0, 0.5, 0.2), source_time=pulse)
        structures = [(mw.Box(position=(0, 0, 0), size=(0.36, 0.3, 0.28), device="cpu"),
                       _ferrite((0.0, -1.75e5, 0.0)))]
        monitors = [mw.PointMonitor("point", position=(-0.2, 0.15, -0.2), fields=("Ex", "Ey", "Ez")),
                    mw.PlaneMonitor("plane", axis="z", position=0.2, fields=("Ex", "Ey", "Ez"),
                                    frequencies=(FREQUENCY,))]
    elif name == "ferrite_oblique_tfsf":
        boundary = mw.BoundarySpec.pml(num_layers=PML_LAYERS)
        source = mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0), source_time=pulse)
        structures = [(mw.Box(position=(0, 0, 0.04), size=(0.3, 0.3, 0.2), device="cpu"),
                       _ferrite((1.0e5, 0.6e5, 1.2e5), gilbert_damping=1e-2, sigma_e=0.02))]
        monitors = [mw.PlaneMonitor("plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"),
                                    frequencies=(FREQUENCY,)),
                    mw.FluxMonitor("flux", axis="z", position=0.25, frequencies=(FREQUENCY,))]
    elif name == "ferrite_mixed_bias":
        boundary = mw.BoundarySpec.none()
        source = mw.PointDipole(position=(0.0, 0.0, -0.2), polarization="Ex", source_time=pulse)
        structures = [(mw.Box(position=(-0.1, 0, 0), size=(0.18, 0.4, 0.24), device="cpu"),
                       _ferrite((0.0, 0.0, 1.75e5))),
                      (mw.Box(position=(0.1, 0, 0), size=(0.18, 0.4, 0.24), device="cpu"),
                       _ferrite((0.0, 0.0, -1.75e5), gilbert_damping=0.0))]
        monitors = [mw.PointMonitor("point", position=(0.2, 0.1, 0.18), fields=("Ex", "Ey", "Ez")),
                    mw.PlaneMonitor("plane", axis="z", position=0.0, fields=("Ex", "Ey", "Ez"),
                                    frequencies=(FREQUENCY,))]
    else:
        raise KeyError(name)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.32, 0.32),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
                     boundary=boundary,
                     sources=(source,))
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(geometry=geometry, material=material, name=f"ferrite_{index}") for index, (geometry, material) in enumerate(structures)),
    ))
    scene = scene.updated(monitors=(*scene.monitors, *monitors))
    return CoreCase(name, scene, cpml_memory="dense" if name == "ferrite_oblique_tfsf" else "slab")


def _fp64_box(position, size) -> mw.Box:
    return mw.Box(position=torch.tensor(position, dtype=torch.float64),
                  size=torch.tensor(size, dtype=torch.float64))


def _fp64_port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(name, positive=(x, -0.005, 0.005), negative=(x, -0.005, -0.005),
                         voltage_path=mw.AxisPath("z"),
                         current_surface=_fp64_box((x, -0.005, -0.0025), (0.015, 0.015, 0.0)),
                         voltage_aperture=_fp64_box((x, -0.005, 0.0), (0.01, 0.01, 0.01)))


def _fp64_case(name: str) -> CoreCase:
    frequency = 3e9
    grid = mw.GridSpec.uniform(0.005)
    if name == "fp64_slab_port":
        # Graded x spacing (5 mm on the port side, 6.7 mm beyond x = 0): nonuniform inverse spacing and CPML profiles.
        x = np.concatenate([np.linspace(-0.04, 0.0, 9), np.linspace(0.0, 0.04, 7)[1:]])
        y = np.linspace(-0.04, 0.04, 17)
        grid = mw.GridSpec.custom(x, y, y)
    ports = [_fp64_port("feed", -0.015)]
    structures, monitors = [], []
    if name == "fp64_vacuum_port":
        monitors = [mw.PointMonitor("point", position=(0.02, 0.01, 0.0), fields=("Ex", "Ey", "Ez"))]
    elif name == "fp64_slab_port":
        structures = [mw.Structure(_fp64_box((0.01, 0.0, 0.0), (0.02, 0.05, 0.05)),
                                   mw.Material(eps_r=4.0, sigma_e=0.05), name="slab")]
        monitors = [mw.monitors.FinitePlaneMonitor("plane", position=(0.025, 0.0, 0.0), size=(0.0, 0.04, 0.04),
                                                   fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                                                   frequencies=(frequency,)),
                    mw.monitors.ClosedSurfaceMonitor.box("radiation", size=(0.06, 0.06, 0.06), sample_spacing=0.007)]
    elif name == "fp64_copper_port":
        structures = [mw.Structure(_fp64_box((0.0, 0.005, 0.0), (0.04, 0.02, 0.04)),
                                   mw.Material(eps_r=2.0, sigma_e=0.15), name="board"),
                      mw.Structure(_fp64_box((0.0, 0.005, 0.0), (0.02, 0.0, 0.02)),
                                   mw.media.CopperFoil(thickness=1e-6, conductivity=1e6, frequency_range=(2e9, 4e9),
                                                       mode_count=16), name="foil")]
        monitors = [mw.monitors.ClosedSurfaceMonitor.box("radiation", size=(0.06, 0.06, 0.06), sample_spacing=0.007)]
    elif name == "fp64_two_ports":
        structures = [mw.Structure(_fp64_box((0.0, 0.0, 0.0), (0.01, 0.03, 0.03)),
                                   mw.Material(eps_r=3.0), name="block")]
        ports.append(_fp64_port("load", 0.015))
    else:
        raise KeyError(name)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, 0.04),) * 3), grid=grid,
                     boundary=mw.BoundarySpec.pml(num_layers=4), ports=tuple(ports),
                     structures=tuple(structures), monitors=tuple(monitors),
                     subpixel=mw.SubpixelSpec(samples=(2, 4, 2), averaging="arithmetic"))
    excitation = mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(frequency, fwidth=1.5e9))
    # 1024 steps put the Rule 2 half horizon after the pulse, so the growth check compares settled spectra
    # rather than the rise of the accumulating DFT sums.
    return CoreCase(name, scene, (frequency,), cpml_memory="dense", steps=1024, excitations=(excitation,),
                    field_dtype=torch.float64)


def _bloch_case(name: str) -> CoreCase:
    theta, phi = np.deg2rad(20.0), np.deg2rad(30.0)
    direction = (float(np.sin(theta) * np.cos(phi)), float(np.sin(theta) * np.sin(phi)), float(np.cos(theta)))
    k0 = 2.0 * np.pi * FREQUENCY / 299_792_458.0
    k = (0.0, 0.0, 0.0) if name == "bloch_pulse_k0" else tuple(k0 * value for value in direction)
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    faces = {"bloch_pml_x": dict(x="pml", y="bloch", z="bloch"),
             "bloch_multi_pml": dict(x="bloch", y="pml", z="pml"),
             "bloch_pec_z": dict(x="bloch", y="bloch", z="pec")}.get(name, dict(x="bloch", y="bloch", z="pml"))
    wavevector = tuple(k[a] if faces["xyz"[a]] == "bloch" else 0.0 for a in range(3))
    nodes = np.linspace(-0.32, 0.32, 33)
    slab = mw.Material.debye(eps_inf=2.0, delta_eps=1.5, tau=8e-11) if name == "bloch_debye_tfsf" else mw.Material(eps_r=3.0)
    if name in ("bloch_debye_tfsf", "bloch_cw_tfsf", "bloch_pulse_k0"):
        polarization = (-float(np.sin(phi)), float(np.cos(phi)), 0.0)
        source = mw.PlaneWave(
            direction=direction if name != "bloch_pulse_k0" else (0.0, 0.0, 1.0),
            polarization=polarization if name != "bloch_pulse_k0" else (0.0, 1.0, 0.0),
            source_time=mw.CW(frequency=FREQUENCY) if name == "bloch_cw_tfsf" else pulse,
            injection=mw.TFSF.slab(axis="z", bounds=(-0.2, 0.2)))
    else:
        source = mw.PointDipole(
            position=(0.03, -0.02, -0.1), polarization=(0.3, 1.0, 0.2),
            source_time=mw.CW(frequency=FREQUENCY) if name == "bloch_dipole_dense" else pulse)
    normal = "x" if name == "bloch_pml_x" else ("y" if name == "bloch_multi_pml" else "z")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=PML_LAYERS, bloch_wavevector=wavevector, **faces),
        structures=(mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 2.0, 0.12), device="cpu"),
                                 slab, name="bloch_slab"),),
        sources=(source,),
        monitors=(
            mw.PointMonitor("point", position=(-0.06, 0.05, 0.1), fields=("Ex", "Ey", "Ez", "Hx")),
            mw.PlaneMonitor("plane", axis=normal, position=0.15, fields=("Ex", "Ey", "Hz"),
                            frequencies=(FREQUENCY,)),
            mw.FluxMonitor("flux", axis=normal, position=0.2, frequencies=(FREQUENCY,)),
        ),
    )
    return CoreCase(name, scene, cpml_memory="dense" if name == "bloch_dipole_dense" else "slab")


def _chart_case(name: str, extra_monitors: tuple = ()) -> CoreCase:
    frequency = 3e9
    fp64 = name == "chart_fp64_copper"
    curvature = 4.0 if name == "chart_convex_two_ports" else -3.0
    grid, boundary = mw.GridSpec.uniform(0.005), mw.BoundarySpec.pml(num_layers=4)
    ports = [_fp64_port("feed", -0.015)]
    board = mw.Structure(_fp64_box((0.0, 0.005, 0.0), (0.04, 0.02, 0.04)), mw.Material(eps_r=2.0, sigma_e=0.15),
                         name="board")
    radiation = mw.monitors.ClosedSurfaceMonitor.box("radiation", size=(0.05, 0.05, 0.05), sample_spacing=0.007)
    foil = mw.Structure(_fp64_box((0.0, 0.005, 0.0), (0.02, 0.0, 0.02)),
                        mw.media.CopperFoil(thickness=1e-6, conductivity=1e6, frequency_range=(2e9, 4e9),
                                            mode_count=16), name="foil")
    if name == "chart_board_port":
        structures = [board]
        monitors = [radiation, mw.PointMonitor("point", position=(0.02, 0.015, -0.01), fields=("Ex", "Ey", "Ez"))]
    elif name in ("chart_copper_port", "chart_fp64_copper"):
        structures, monitors = [board, foil], [radiation]
    elif name == "chart_convex_two_ports":
        # Graded u spacing (5 mm up to u = 5 mm around the ports, 6.25 mm above) with PEC u faces:
        # nonuniform metric and spacing.
        # The closed u axis needs the explicit CPML absorber, sampled by the passivity guard.
        x = np.linspace(-0.03, 0.03, 13)
        y = np.concatenate([np.linspace(-0.03, 0.005, 8), np.linspace(0.005, 0.03, 5)[1:]])
        grid = mw.GridSpec.custom(x, y, x)
        boundary = mw.BoundarySpec.pml(num_layers=4).with_faces(y_low="pec", y_high="pec")
        ports.append(_fp64_port("load", 0.015))
        structures = [mw.Structure(_fp64_box((0.0, 0.0, 0.0), (0.01, 0.03, 0.03)), mw.Material(eps_r=3.0),
                                   name="block")]
        monitors = [mw.monitors.FinitePlaneMonitor("plane", position=(0.02, 0.0, 0.0), size=(0.0, 0.03, 0.03),
                                                   fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                                                   frequencies=(frequency,))]
    else:
        raise KeyError(name)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.03, 0.03),) * 3), grid=grid, boundary=boundary,
                     coordinates=mw.coordinates.BendCoordinates(curvature=curvature, origin=(0.2, -0.1, 0.3)),
                     ports=tuple(ports), structures=tuple(structures), monitors=(*monitors, *extra_monitors),
                     subpixel=mw.SubpixelSpec(samples=(2, 4, 2), averaging="arithmetic"))
    excitation = mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(frequency, fwidth=1.5e9))
    return CoreCase(name, scene, (frequency,), cpml_memory="slab" if name == "chart_board_port" else "dense",
                    steps=1024 if fp64 else None, excitations=(excitation,),
                    absorber="cpml" if name == "chart_convex_two_ports" else None,
                    field_dtype=torch.float64 if fp64 else torch.float32)


def _monitor_case(name: str) -> CoreCase:
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    excitations = ()
    if name == "power_loss_chart":
        # chart_board_port with a conduction loss volume over the lossy board (charts admit conduction only).
        loss = mw.monitors.PowerLossMonitor("loss", position=(0.0, 0.005, 0.0), size=(0.04, 0.02, 0.04))
        return replace(_chart_case("chart_board_port", (loss,)), name=name)
    if name == "power_loss_copper":
        # A copper foil with a one-sided conductive Debye substrate, fed by a port (split bulk loss with
        # retained side spectra); the full set and a one-frequency subset of both automatic channels.
        frequencies = (2.5e9, 3.0e9)
        foil = mw.media.CopperFoil(thickness=1e-6, conductivity=1e6, frequency_range=(2e9, 4e9), mode_count=16)
        substrate = mw.Material(eps_r=2.0, sigma_e=0.1, debye_poles=(mw.DebyePole(delta_eps=1.5, tau=1e-10),))
        channels = ("conduction", "electric_dispersion")
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.03, 0.03),) * 3), grid=mw.GridSpec.uniform(0.005),
            boundary=mw.BoundarySpec.pec(),
            structures=(mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=(0.0, 0.02, 0.02), device="cpu"), foil,
                                     name="foil"),
                        mw.Structure(mw.Box(position=(0.01, 0.0, 0.0), size=(0.02, 0.04, 0.04), device="cpu"),
                                     substrate, name="substrate")),
            ports=(mw.LumpedPort("feed", positive=(-0.01, 0.0, 0.005), negative=(-0.01, 0.0, -0.005),
                                 voltage_path=mw.AxisPath("z"),
                                 current_surface=mw.Box(position=(-0.01, 0.0, -0.0025), size=(0.015, 0.015, 0.0),
                                                        device="cpu")),),
            monitors=(mw.PointMonitor("probe", (0.01, 0.0, 0.0), fields=("Ez",)),
                      mw.monitors.PowerLossMonitor("bulk", position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04),
                                                   channels=channels),
                      mw.monitors.PowerLossMonitor("subset", position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04),
                                                   frequencies=(3.0e9,), channels=channels)),
        )
        excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(3.0e9, fwidth=1.0e9)),)
        return CoreCase(name, scene, frequencies, excitations=excitations)
    if name == "component_stress_port":
        # The w1_lumped_port gap (driven port, resistive load) with gap field time series bound to the port.
        rating = mw.esd.ComponentRating(voltage=1e-2, current=1e-2, energy=1e-15, model="gate-rating")
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.02, 0.02),) * 3), grid=mw.GridSpec.uniform(0.005),
            boundary=mw.BoundarySpec.none(),
            ports=(mw.LumpedPort("feed", positive=(0, 0, 0.005), negative=(0, 0, -0.005), voltage_path=mw.AxisPath("z"),
                                 current_surface=mw.Box(position=(0, 0, -0.0025), size=(0.015, 0.015, 0),
                                                        device="cpu")),),
            lumped_elements=(mw.rf.Resistor("load", positive=(0.01, 0, 0.005), negative=(0.01, 0, -0.005),
                                            resistance=100.0),),
            monitors=(mw.PointMonitor("point", position=(0, 0.01, 0), fields=("Ex", "Ez")),
                      mw.monitors.FieldTimeMonitor("vprobe", components=("Ez",), position=(0.0, 0.0, 0.0)),
                      mw.monitors.FieldTimeMonitor("iprobe", components=("Ez",), position=(0.005, 0.0, 0.0)),
                      mw.esd.ComponentStressMonitor("part", port="feed", rating=rating, voltage_series="vprobe",
                                                    current_series="iprobe")),
        )
        excitations = (mw.rf.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=0.5e9)),)
        return CoreCase(name, scene, (1e9,), excitations=excitations)
    if name == "wire_monitor":
        # w1_thin_wire with a wire monitor (current and charge spectra along the wire).
        frequency = 2e9
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.12, 0.12),) * 3), grid=mw.GridSpec.uniform(0.04),
            boundary=mw.BoundarySpec.none(),
            thin_wires=(mw.rf.ThinWire("wire", ((0, 0, -0.08), (0, 0, 0.08)), 0.002, mw.rf.WireConductor.pec(),
                                       snap="strict"),),
            sources=(mw.PointDipole(position=(0, 0, 0.02), polarization="Ez", width=0.04,
                                    source_time=mw.CW(frequency, amplitude=10.0)),),
            monitors=(mw.PointMonitor("point", position=(0, 0.01, 0), fields=("Ex", "Ez")),
                      mw.rf.WireMonitor("wire_spectra", "wire", frequencies=(frequency,),
                                        quantities=("current", "charge"))),
        )
        return CoreCase(name, scene, (frequency,))
    # Open 48^3 cells (32^3 interior plus slab CPML) around a dipole.
    nodes = np.linspace(-0.32, 0.32, 48 - 2 * PML_LAYERS)
    frequencies = (1.8e9, FREQUENCY)
    debye = mw.Material(eps_r=2.0, sigma_e=0.05, debye_poles=(mw.DebyePole(delta_eps=1.5, tau=1e-10),))
    if name == "power_loss_slab":
        structures = (mw.Structure(mw.Box(position=(0.06, 0.0, 0.0), size=(0.16, 0.4, 0.4), device="cpu"), debye,
                                   name="lossy_slab"),)
        sources = (mw.PointDipole(position=(-0.12, 0.0, 0.0), polarization="Ez", source_time=pulse),)
        monitors = (
            mw.monitors.PowerLossMonitor("loss", position=(0.06, 0.0, 0.0), size=(0.2, 0.44, 0.44),
                                         channels=("conduction", "electric_dispersion")),
            # A box clipping the slab: physical intersection quadrature, one frequency.
            mw.monitors.PowerLossMonitor("clipped", position=(0.05, 0.03, -0.02), size=(0.13, 0.21, 0.17),
                                         frequencies=(FREQUENCY,), quadrature="physical_intersection"),
            mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")),
        )
    elif name == "dipole_emission":
        # A named oblique dipole inside a dielectric sphere: its emitted power, beside a plane spectrum.
        structures = (mw.Structure(mw.Sphere(position=(0, 0, 0), radius=0.12, device="cpu"), mw.Material(eps_r=4.0),
                                   name="dielectric_sphere"),)
        sources = (mw.PointDipole(position=(0.02, -0.01, 0.03), polarization=(0.3, 0.0, 1.0), source_time=pulse,
                                  name="emitter"),)
        monitors = (
            mw.monitors.DipoleEmissionMonitor("emission", "emitter"),
            mw.monitors.DipoleEmissionMonitor("emission_subset", "emitter", frequencies=(FREQUENCY,)),
            mw.PlaneMonitor("plane", axis="y", position=0.0, fields=("Ex", "Ez"), frequencies=(FREQUENCY,)),
        )
    elif name == "time_surface_aux_debye":
        # Detached surface and ADE-state time traces: a box surface around the dipole, a polarization patch
        # and a point inside the Debye slab (ADE current too), on interval and start/stop schedules.
        structures = (mw.Structure(mw.Box(position=(0.06, 0.0, 0.0), size=(0.16, 0.4, 0.4), device="cpu"), debye,
                                   name="debye_slab"),)
        sources = (mw.PointDipole(position=(-0.12, 0.0, 0.0), polarization="Ez", source_time=pulse),)
        monitors = (
            mw.monitors.SurfaceFieldTimeMonitor("surface", position=(-0.12, 0.0, 0.0), size=(0.12, 0.12, 0.12),
                                                components=("Ex", "Ez", "Hy"), interval=3),
            mw.monitors.SurfaceFieldTimeMonitor("patch", position=(0.2, 0.0, 0.0), size=(0.0, 0.2, 0.2),
                                                components=("Ez",), start=40, stop=200, interval=5),
            mw.monitors.AuxFieldTimeMonitor("polarization", components=("Px", "Pz"), position=(0.06, 0.0, 0.0),
                                            size=(0.0, 0.16, 0.16), interval=2),
            mw.monitors.AuxFieldTimeMonitor("point_state", components=("Pz", "Jpz", "Mz"), position=(0.06, 0.02, 0.0)),
            mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")),
        )
    else:
        raise KeyError(name)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
        structures=structures, sources=sources, monitors=monitors,
    )
    return CoreCase(name, scene, frequencies, excitations=excitations)


def _bloch_separate_case(name: str) -> CoreCase:
    pulse = mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9)
    debye = mw.Material(eps_r=2.0, sigma_e=0.05, debye_poles=(mw.DebyePole(delta_eps=1.5, tau=1e-10),))
    if name in ("bloch_copper_foil", "bloch_copper_resistor"):
        # The Bloch-periodic foil of the copper-foil adjoint fixture (a 6 cm cell, 5 mm grid, foil across
        # the y/z cell at x = 0), driven from one side.
        frequency = 3e9
        foil = mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=(0.0, 0.06, 0.06), device="cpu"),
                            mw.media.CopperFoil(thickness=1e-6, conductivity=1e6, frequency_range=(2e9, 4e9),
                                                mode_count=16), name="foil")
        if name == "bloch_copper_foil":
            boundary = mw.BoundarySpec.faces(default="bloch", x="pml", num_layers=2, bloch_wavevector=(0.0, 7.0, 11.0))
            structures = (foil, mw.Structure(mw.Box(position=(0.01, 0.0, 0.0), size=(0.02, 0.06, 0.06), device="cpu"),
                                             debye, name="substrate"))
            source_time, elements = mw.CW(frequency, amplitude=1e4), ()
            monitors = (mw.monitors.PowerLossMonitor("loss", position=(0.0, 0.0, 0.0), size=(0.06, 0.06, 0.06),
                                                     channels=("conduction", "electric_dispersion")),)
        else:
            boundary = mw.BoundarySpec.bloch(wavevector=(0.0, 7.0, 11.0))
            structures, source_time = (foil,), mw.GaussianPulse(frequency, fwidth=1.5e9)
            elements = (mw.rf.Resistor("load", positive=(-0.02, 0.01, 0.005), negative=(-0.02, 0.01, -0.005),
                                       resistance=50.0),)
            monitors = ()
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.03, 0.03),) * 3), grid=mw.GridSpec.uniform(0.005), boundary=boundary,
            structures=structures, lumped_elements=elements,
            sources=(mw.UniformCurrentSource(name="drive", center=(-0.015, 0.0, -0.0025), size=(0.005, 0.005, 0.005),
                                             polarization="Ez", source_time=source_time),),
            monitors=(mw.PointMonitor("probe", (0.01, 0.0, 0.0), fields=("Ey", "Ez")),
                      mw.PlaneMonitor("plane", axis="x", position=-0.01, fields=("Ey", "Ez"), frequencies=(frequency,)),
                      *monitors),
        )
        return CoreCase(name, scene, (frequency,))
    # The bloch_dipole_slab cell (Bloch x/y with a z PML, 2 cm grid, eps 3 slab in z).
    theta, phi = np.deg2rad(20.0), np.deg2rad(30.0)
    direction = (float(np.sin(theta) * np.cos(phi)), float(np.sin(theta) * np.sin(phi)), float(np.cos(theta)))
    k0 = 2.0 * np.pi * FREQUENCY / 299_792_458.0
    nodes = np.linspace(-0.32, 0.32, 33)
    boundary = mw.BoundarySpec.faces(default="pml", num_layers=PML_LAYERS, x="bloch", y="bloch", z="pml",
                                     bloch_wavevector=(k0 * direction[0], k0 * direction[1], 0.0))
    structures = [mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 2.0, 0.12), device="cpu"),
                               mw.Material(eps_r=3.0), name="bloch_slab")]
    elements, ports = (), ()
    dipole = mw.PointDipole(position=(0.03, -0.02, -0.1), polarization=(0.3, 1.0, 0.2), source_time=pulse)
    sources = (dipole,)
    if name == "bloch_lumped_elements":
        elements = (
            mw.rf.Resistor("r", positive=(0.1, 0.0, 0.02), negative=(0.1, 0.0, -0.02), resistance=50.0),
            mw.rf.Capacitor("c", positive=(-0.1, 0.06, 0.02), negative=(-0.1, 0.06, -0.02), capacitance=1e-12),
            mw.rf.Inductor("l", positive=(0.04, 0.12, 0.2), negative=(0.0, 0.12, 0.2), inductance=5e-9),
        )
    elif name == "bloch_loaded_ports":
        def port(port_name, x, y, termination=None):
            return mw.LumpedPort(port_name, positive=(x, y, 0.02), negative=(x, y, -0.02), voltage_path=mw.AxisPath("z"),
                                 current_surface=mw.Box(position=(x, y, -0.01), size=(0.06, 0.06, 0.0), device="cpu"),
                                 **({"termination": termination} if termination is not None else {}))
        structures.append(mw.Structure(mw.Box(position=(-0.12, 0.0, 0.0), size=(0.08, 0.08, 0.08), device="cpu"),
                                       debye, name="debye_block"))
        ports = (port("series", 0.12, 0.0, mw.rf.SeriesRLC(r=50.0, l=2e-9, c=2e-12)),
                 port("parallel", -0.12, 0.0, mw.rf.ParallelRLC(r=100.0, l=5e-9, c=1e-12)),
                 port("matched", 0.0, 0.14))
    elif name == "bloch_tfsf_resistor":
        sources = (mw.PlaneWave(direction=direction, polarization=(-float(np.sin(phi)), float(np.cos(phi)), 0.0),
                                source_time=pulse, injection=mw.TFSF.slab(axis="z", bounds=(-0.2, 0.2))),)
        elements = (mw.rf.Resistor("r", positive=(0.1, 0.0, 0.1), negative=(0.1, 0.0, 0.14), resistance=50.0),)
    else:
        raise KeyError(name)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3), grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=boundary, structures=tuple(structures), sources=sources, ports=ports,
        lumped_elements=elements,
        monitors=(mw.PointMonitor("point", position=(-0.06, 0.05, 0.1), fields=("Ex", "Ey", "Ez", "Hx")),
                  mw.PlaneMonitor("plane", axis="z", position=0.15, fields=("Ex", "Ey", "Hz"), frequencies=(FREQUENCY,)),
                  mw.FluxMonitor("flux", axis="z", position=0.2, frequencies=(FREQUENCY,))),
    )
    return CoreCase(name, scene)


def _cache_case(name: str) -> CoreCase:
    case = build_core_case(CACHE_SCENES[name])
    outcome = name.split("_")[1]
    warmup = None
    if outcome == "local":
        # Same grid frame, slab moved along its normal: the gated bind patches the warm-up compilation.
        slab, = case.scene.structures
        moved = mw.Box(position=(0.0, 0.0, 0.05), size=tuple(slab.geometry.size.tolist()), device="cpu")
        warmup = case.scene.updated(structures=(mw.Structure(moved, slab.material, name=slab.name),))
    return replace(case, material_cache=outcome, cache_warmup=warmup)


def ladder_names() -> tuple[str, ...]:
    return tuple(
        f"{medium}_{n}_{monitor}" for medium in ("vacuum", "dielectric")
        for n in SIZES for monitor in ("bare", "plane_dft")
    )


def build_ladder_case(name: str) -> CoreCase:
    medium, size, monitor = name.split("_", 2)
    shape = (int(size),) * 3
    scene = _grid_scene(shape)
    if medium == "dielectric":
        scene = _sphere(scene)
    if monitor == "bare":
        # Keep one output so the solver can finalize a bare ladder run.
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ez",)),
        ))
    if monitor == "plane_dft":
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PlaneMonitor(
                "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
            ),
        ))
    return CoreCase(name, scene, shape=shape)
