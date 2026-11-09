"""Declarative core correctness scenes and the computational-grid size ladder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import witwin.maxwell as mw

from benchmark.scenes import CPML_PINNED_SCENARIOS, SCENARIOS, build_scene as build_benchmark_scene


# planewave_periodic_vacuum (periodic x/y + PML z) is real-field closed-transverse CPML,
# which the core path refuses until it carries the passivity guard.
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
# Only absorber classes the core path admits are gated by default; the others stay buildable for refusals.
# closed_transverse_cpml is real-field closed-transverse CPML, refused until the core loop carries the
# passivity guard.
ADMITTED_ABSORBER_SCENES = tuple(n for n in ABSORBER_SCENES if n != "closed_transverse_cpml")
# pmc_cavity stays buildable for its PMC refusal. grating_diffraction and diffraction_normal_orders
# (periodic x/y + PML z, pinned to CPML) stay buildable for the closed-transverse CPML refusal.
SRCMON_REFUSED_SCENES = ("pmc_cavity", "grating_diffraction", "diffraction_normal_orders")
# Keep master's live-coefficient and ADE fixtures buildable. Their pinned
# closed-transverse CPML cases require the main loop's passivity guard.
CORE_REFUSED_BENCHMARK_SCENES = CPML_PINNED_SCENARIOS.intersection(W3_SCENES + DISPERSIVE_SCENES)
CORE_SCENES = (G2_SCENES + EXTRA_SCENES + ADMITTED_ABSORBER_SCENES
               + tuple(n for n in SRCMON_SCENES if n not in SRCMON_REFUSED_SCENES)
               + tuple(dict.fromkeys(W1_SCENES + W2_SCENES))
               + tuple(n for n in W3_SCENES + DISPERSIVE_SCENES if n not in CORE_REFUSED_BENCHMARK_SCENES))
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

    def plan(self) -> dict:
        return {
            "scene": self.name, "device": str(self.scene.device),
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
        subpixel_samples=subpixel,
        device="cpu",
    ).add_source(source)


def _sphere(scene: mw.Scene) -> mw.Scene:
    return scene.add_structure(mw.Structure(
        geometry=mw.Sphere(position=(0, 0, 0), radius=0.12, device="cpu"),
        material=mw.Material(eps_r=4.0), name="dielectric_sphere",
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
        scene = _sphere(_grid_scene((64, 64, 64))).clone(boundary=boundary)
        absorber, shape = None, None
    else:
        layers = 40
        nodes = np.linspace(-0.48, 0.48, 48)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.48, 0.48),) * 3),
            grid=mw.GridSpec.custom(nodes, nodes, nodes),
            boundary=mw.BoundarySpec.faces(default="periodic", num_layers=layers, strength=1e6, z="pml"),
            subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
            device="cpu",
        ).add_source(mw.PointDipole(
            position=(0.0, 0.0, -0.2), polarization="Ex",
            source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9),
        ))
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0, 0, 0), size=(0.96, 0.96, 0.12), device="cpu"),
            material=mw.Material(eps_r=4.0), name="slab",
        ))
        absorber = "absorber" if name == "closed_transverse_graded" else "cpml"
        shape = None
    scene.add_monitor(mw.PointMonitor("point", position=(0.1, 0.1, 0.25), fields=("Ex", "Ez")))
    scene.add_monitor(mw.PlaneMonitor(
        "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
    ))
    return CoreCase(name, scene, absorber=absorber, shape=shape)


def build_core_case(name: str) -> CoreCase:
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
            subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase"),
            device="cpu",
        ).add_source(mw.PlaneWave(
            direction=(1., 0., 0.), polarization=(0., 0., 1.),
            source_time=mw.CW(frequency=frequency, amplitude=1e6 if nonlinear else 50.),
        ))
        if name == "w3_modulation_two_omega":
            for x, modulation_frequency, phase in ((-0.16, 2.5e8, 0.31), (0.16, 4e8, -0.27)):
                scene.add_structure(mw.Structure(
                    geometry=mw.Box(position=(x, 0., 0.), size=(0.12, 0.5, 0.5), device="cpu"),
                    material=mw.Material(eps_r=4., modulation=mw.ModulationSpec(
                        frequency=modulation_frequency, amplitude=0.2, phase=phase)),
                ))
        else:
            # The modulation/nonlinearity/Debye composition fixtures, without time observers.
            material_options = {}
            if name.endswith("_kerr"):
                material_options["kerr_chi3"] = 1e-16
            elif name.endswith("_tpa"):
                material_options["nonlinearity"] = mw.TwoPhotonAbsorption(beta=1e-10)
            if "modulation" in name:
                material_options["modulation"] = mw.ModulationSpec(frequency=2.5e8, amplitude=0.2, phase=0.31)
            if name.startswith("w3_debye_"):
                material_options["debye_poles"] = (mw.DebyePole(delta_eps=1., tau=8e-11),)
            scene.add_structure(mw.Structure(
                geometry=mw.Box(position=(0., 0., 0.), size=(0.24, 0.64, 0.64), device="cpu"),
                material=mw.Material(eps_r=4., **material_options),
            ))
        frequencies = (0.75e9, frequency, 1.25e9)
        scene.add_monitor(mw.PointMonitor("point", position=(0.28, 0., 0.), fields=("Ez",)))
        scene.add_monitor(mw.PlaneMonitor(
            "plane", axis="x", position=0.28, fields=("Ez",), frequencies=frequencies,
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
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0., 0., 0.), size=(0.6,) * 3, device="cpu"),
            material=mw.Material(eps_r=2., breakdown=mw.DielectricBreakdown(
                critical_field=1e-8, minimum_duration=2e-10,
                post_breakdown_conductivity=0.1, ramp_time=1e-9)),
        ))
        scene.add_monitor(mw.PointMonitor("point", position=(0., 0., 0.), fields=("Ez",)))
        scene.add_monitor(mw.BreakdownMonitor(
            "stress", position=(0., 0., 0.), size=(0.3,) * 3, critical_field=1e-8,
        ))
        return CoreCase(name, scene, shape=(40, 40, 40), initial_fields=True)
    if name == "magnetic_lorentz_slab":
        scene = build_benchmark_scene("lorentz_slab")
        scene = scene.clone(structures=[mw.Structure(
            geometry=structure.geometry,
            material=mw.Material(mu_lorentz_poles=(mw.LorentzPole(
                delta_eps=1., resonance_frequency=3.5e9, gamma=.25e9),)),
            name=structure.name,
        ) for structure in scene.structures])
        return CoreCase(name, scene, SCENARIOS["lorentz_slab"].frequencies)
    if name == "dispersive_lumped_port":
        case = _w1_case("w1_lumped_port")
        case.scene.add_structure(mw.Structure(
            mw.Box(position=(0, 0, 0), size=(.03, .03, .03), device="cpu"),
            mw.Material.debye(eps_inf=2., delta_eps=1., tau=8e-11), name="dispersive_host"))
        return CoreCase(name, case.scene, case.frequencies, excitations=case.excitations)
    if name in ("copper_patch", "copper_board_60k"):
        from scripts.dev.fdtd.thin_sheet_overhead import SCENARIOS as SHEET_SCENARIOS, _scene
        # Match the D3b wrapper's frequency, geometry, source and default time-step settings.
        return CoreCase(name, _scene(SHEET_SCENARIOS[name], True, device="cpu"), (2.4e9,))
    if name in W1_SCENES:
        return _w1_case(name)
    if name in ABSORBER_SCENES:
        return _absorber_case(name)
    if name in O2A_SCENES:
        return _o2a_case(name)
    if name in G2_SCENES + SRCMON_SCENES + W3_SCENES + DISPERSIVE_SCENES + ("planewave_periodic_vacuum",):
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
        vertices, faces = mw.Pyramid(
            position=(0, 0, -0.12), base_size=0.3, height=0.24, device="cpu",
        ).to_mesh()
        geometry = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid", device="cpu")
        scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0)))
    elif name == "boolean_dielectric":
        geometry = mw.Difference(
            mw.Box(position=(0, 0, 0), size=(0.32, 0.3, 0.28), device="cpu"),
            mw.Sphere(position=(0.12, 0, 0), radius=0.15, device="cpu"), device="cpu",
        )
        scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=3.0)))
    elif name == "multi_structure_dielectric":
        _sphere(scene)
        for x, eps in ((-0.24, 2.0), (0.24, 6.0)):
            scene.add_structure(mw.Structure(
                geometry=mw.Box(position=(x, 0, 0), size=(0.12, 0.24, 0.2), device="cpu"),
                material=mw.Material(eps_r=eps),
            ))
    elif name.startswith("subpixel_"):
        _sphere(scene)
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0.24, 0, 0), size=(0.12, 0.24, 0.2), device="cpu"),
            material=mw.Material(eps_r=3.0),
        ))
    elif name in ("cpml_dense", "cpml_compressed", "odd_grid", "cw_dielectric", "gaussian_beam_dielectric",
                  "closed_surface_box", "closed_surface_sampled"):
        _sphere(scene)
    else:
        raise KeyError(name)
    if name.startswith("closed_surface_"):
        scene.add_monitor(mw.ClosedSurfaceMonitor.box(
            "surface", size=(0.6, 0.6, 0.6), frequencies=(FREQUENCY,),
            sample_spacing=0.02 if name == "closed_surface_sampled" else None,
        ))
    scene.add_monitor(mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")))
    scene.add_monitor(mw.PlaneMonitor(
        "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
    ))
    scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.3, frequencies=(FREQUENCY,)))
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
        scene.add_source(mw.PointDipole(position=(0, 0, -0.28), polarization="Ez",
            source_time=(mw.CW(frequency=FREQUENCY, amplitude=-0.3, phase=-0.2) if name == "o2_cw_overlap"
                         else mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5e9, amplitude=-0.3, phase=-0.2))))
    scene.add_monitor(mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ex", "Ez")))
    scene.add_monitor(mw.PlaneMonitor("plane", axis="y", position=0, fields=("Ex", "Ey", "Ez"), frequencies=frequencies))
    scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.3, frequencies=frequencies))
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
        sample = builder.build(device="cpu")
        return CoreCase(name, sample.scene, tuple(sample.fdtd_kwargs["frequencies"]),
                        excitations=tuple(sample.fdtd_kwargs["excitations"]))

    frequency = 1e9
    excitations = ()
    if name == "w1_terminal_port":
        from tests.rf.terminal.test_planar_terminal_targets import _scene
        scene = _scene(metal="copper", device="cpu")
        excitations = (mw.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name == "w1_wave_port":
        # Exercise the ModeSource/ModeMonitor carrier used by lowered wave-port columns.
        from benchmark.scenes.rf.rectangular_waveguide import rectangular_waveguide_reference_scene
        frequency = .4e9
        scene = rectangular_waveguide_reference_scene(frequencies=(frequency,), device="cpu")
    elif name == "w1_thin_sheet_cavity":
        from tests.materials.thin_sheet.test_thin_sheet_runtime import _cavity, _plate
        scene = _cavity(_plate(mw.Material.pec(), sheet_model="conformal_two_sided"))
        frequency = 2e9
    elif name == "w1_sheet_feed":
        from tests.materials.thin_sheet.test_thin_sheet_feeds import _runtime_scene
        scene = _runtime_scene(tilted=True, device="cpu")
        excitations = (mw.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name == "w1_coax_port":
        # The shielded coax with feed, probe and load of the port-step bit-identity fixture, built on CPU.
        h, a, b, z0, z1, half, length = 0.25e-3, 1.0e-3, 2.3e-3, 1.0e-3, 6.0e-3, 3.0e-3, 7.0e-3
        scene = mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase"),
                         domain=mw.Domain(bounds=((-half, half), (-half, half), (0.0, length))),
                         grid=mw.GridSpec.uniform(h), boundary=mw.BoundarySpec.pec(), device="cpu")
        circle = [(b * np.cos(2 * np.pi * k / 96), b * np.sin(2 * np.pi * k / 96)) for k in range(96)]
        square = [(-half - h, -half - h), (half + h, -half - h), (half + h, half + h), (-half - h, half + h)]
        scene.add_structure(mw.Structure(mw.ComplexPolySlab([square, circle], bounds=(0.0, length), axis="z"),
                                         mw.Material.pec(), name="shield"))
        scene.add_structure(mw.Structure(mw.Cylinder(position=(0, 0, 0.5 * (z0 + z1)), radius=a, height=z1 - z0,
                                                     axis="z"), mw.Material.pec(), name="inner"))
        scene.add_port(mw.CoaxialLumpedPort("feed", center=(0.0, 0.0, z0), inner_diameter=2 * a, outer_diameter=2 * b,
                                            normal_axis="z", reference_impedance=50.0))
        scene.add_port(mw.CoaxialLumpedPort("probe", center=(0.0, 0.0, 3.5e-3), inner_diameter=2 * a,
                                            outer_diameter=2 * b, normal_axis="z"))
        scene.add_lumped_element(mw.CoaxialLumpedResistor("load", center=(0, 0, z1), inner_diameter=2 * a,
                                                          outer_diameter=2 * b, normal_axis="z", resistance=75.0))
        excitations = (mw.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    elif name in ("w1_copper_patch", "w1_lumped_port", "w1_circuit", "w1_network"):
        # The small gap/foil fixture in test_copper_foil_execution._simulation.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),)*3),
                         grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(), device="cpu")
        if name == "w1_copper_patch":
            from benchmark.datagen.scenes import _box
            scene.add_structure(mw.Structure(
                _box((.01, 0, 0), (0, .02, .02)),
                mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)), name="foil"))
        if name in ("w1_lumped_port", "w1_circuit", "w1_network"):
            scene.add_port(mw.LumpedPort(
                "feed", positive=(0, 0, .005), negative=(0, 0, -.005), voltage_path=mw.AxisPath("z"),
                current_surface=mw.Box(position=(0, 0, -.0025), size=(.015, .015, 0), device="cpu")))
            if name == "w1_lumped_port":
                scene.add_lumped_element(mw.Resistor(
                    "load", positive=(.01, 0, .005), negative=(.01, 0, -.005), resistance=100.))
                excitations = (mw.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
            elif name == "w1_circuit":
                # The resistive, field-bound circuit in the port graph fixture.
                circuit = mw.Circuit("driven")
                node = circuit.node("input")
                circuit.add(mw.Resistor("R1", node, circuit.ground, 50.))
                circuit.add(mw.CurrentSource("I1", circuit.ground, node,
                                             waveform=mw.SineWaveform(0., .01, 1e9)))
                circuit.bind_port("feed", positive=node, negative=circuit.ground)
                scene = scene.clone(circuits=(circuit,))
            else:
                # One-port passive specialization of the embedded-network fixture.
                model = mw.StateSpaceNetwork(
                    A=torch.tensor([[-4e9]], dtype=torch.float64),
                    B=torch.tensor([[4e9]], dtype=torch.float64),
                    C=torch.tensor([[-.02]], dtype=torch.float64),
                    D=torch.tensor([[.022]], dtype=torch.float64),
                    representation="Y", port_order=("load",))
                # Cover the pulse's effective band, which starts at DC.
                band = torch.linspace(0., 3e9, 13, dtype=torch.float64)
                scene = scene.clone(networks=(mw.NetworkBlock(
                    "load", model=model, fit=False,
                    network=mw.NetworkData.from_y(frequencies=band, y=model.evaluate(band),
                                                  z0=50., port_names=model.port_order),
                    connections={"load": "feed"}),))
        # The circuit's CW current source drives that scene alone; mixing it with a pulse is refused.
        if not excitations and name != "w1_circuit":
            scene.add_source(mw.PointDipole(position=(-.01, 0, -.01), polarization="Ez", profile="ideal",
                                           source_time=mw.GaussianPulse(1e9, fwidth=.5e9)))
    elif name == "w1_thin_wire":
        # test_thin_wire_forward._scene, with an admitted point readout.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.12, .12),)*3),
                         grid=mw.GridSpec.uniform(.04), boundary=mw.BoundarySpec.none(), device="cpu")
        scene.add_thin_wire(mw.ThinWire("wire", ((0, 0, -.08), (0, 0, .08)), .002,
                                       mw.WireConductor.pec(), snap="strict"))
        scene.add_source(mw.PointDipole(position=(0, 0, .02), polarization="Ez", width=.04,
                                       source_time=mw.CW(2e9, amplitude=10.)))
        frequency = 2e9
    elif name == "w1_cut_surface":
        # The oblique two-triangle patch in test_cut_surface._mesh.
        vertices = np.array(((-.008, -.008, 0), (.008, -.008, 0),
                             (.008, .008, 0), (-.008, .008, 0)))
        vertices[:, 2] = .3*vertices[:, 0] + .001
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),)*3),
                         grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(), device="cpu")
        scene.add_structure(mw.Structure(
            mw.SurfaceMesh(vertices, ((0, 1, 2), (0, 2, 3)), device="cpu"), mw.Material.pec()))
        scene.add_source(mw.PointDipole(position=(-.025, 0, -.02), polarization="Ez", profile="ideal",
                                       source_time=mw.CW(1e9)))
    elif name == "w1_sibc":
        # Six finite exposed faces from test_surface_impedance_runtime.
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.03, .03),)*3),
                         grid=mw.GridSpec.uniform(.004), boundary=mw.BoundarySpec.pml(num_layers=8), device="cpu")
        scene.add_structure(mw.Structure(
            mw.Box(position=(.008, 0, 0), size=(.012, .02, .02), device="cpu"),
            mw.LossyMetalMedium(conductivity=50.)))
        scene.add_source(mw.PointDipole(position=(-.012, 0, 0), polarization="Ez", width=.008,
                                       source_time=mw.CW(2e9, amplitude=40.)))
        frequency = 2e9
    else:
        raise KeyError(name)
    if not scene.monitors:
        scene.add_monitor(mw.PointMonitor("point", position=(0, .01, 0), fields=("Ex", "Ez")))
    return CoreCase(name, scene, (frequency,), excitations=excitations)


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
        _sphere(scene)
    if monitor == "bare":
        # Keep one output so the solver can finalize a bare ladder run.
        scene.add_monitor(mw.PointMonitor("point", position=(0.2, 0.1, 0.1), fields=("Ez",)))
    if monitor == "plane_dft":
        scene.add_monitor(mw.PlaneMonitor(
            "plane", axis="y", position=0.0, fields=("Ex", "Ey", "Ez"), frequencies=(FREQUENCY,),
        ))
    return CoreCase(name, scene, shape=shape)
