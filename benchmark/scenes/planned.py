"""Runnable scenarios for the S1-S6 numerical-validation campaign."""

from __future__ import annotations

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import DX, HALF_SPAN, base_scene, with_plot_monitors

FREQUENCIES = (2.0e9,)
GRATING_FREQUENCIES = (1.0e9,)
FAR_FIELD_FREQUENCIES = (0.5e9,)
RING_FREQUENCIES = tuple(float(value) for value in np.linspace(1.8e9, 2.2e9, 9))
CAVITY_FREQUENCIES = tuple(float(value) for value in np.linspace(1.1e8, 2.5e8, 29))
PULSE = mw.GaussianPulse(frequency=FREQUENCIES[0], fwidth=0.5e9)
GRATING_PULSE = mw.GaussianPulse(frequency=GRATING_FREQUENCIES[0], fwidth=0.25e9)
FAR_FIELD_PULSE = mw.CW(frequency=FAR_FIELD_FREQUENCIES[0])
CAVITY_PULSE = mw.GaussianPulse(frequency=1.9e8, fwidth=1.0e8)
_C0 = 299_792_458.0


def _observed(
    scene: mw.Scene,
    *,
    axis: str = "y",
    position: float = 0.0,
    component: str = "Ex",
    flux_axis: str = "z",
    flux_positions: tuple[float, float] = (-0.3, 0.3),
    frequencies: tuple[float, ...] = FREQUENCIES,
) -> mw.Scene:
    scene.add_monitor(mw.PlaneMonitor("field", axis=axis, position=position, fields=(component,), frequencies=frequencies))
    scene.add_monitor(mw.FluxMonitor("reflected", axis=flux_axis, position=flux_positions[0], frequencies=frequencies, normal_direction="-"))
    scene.add_monitor(mw.FluxMonitor("transmitted", axis=flux_axis, position=flux_positions[1], frequencies=frequencies))
    return with_plot_monitors(scene, frequencies=frequencies)


def _plane_scene(*, material=None, boundary=None, grid=None, source=None, subpixel=None) -> mw.Scene:
    scene = base_scene()
    if boundary is not None or grid is not None or subpixel is not None:
        scene = mw.Scene(
            domain=scene.domain,
            grid=grid or scene.grid,
            boundary=boundary or scene.boundary,
            subpixel_samples=subpixel or scene.subpixel,
            device="cpu",
        )
    if material is not None:
        absorbing_kinds = {"pml", "absorber", "stablepml"}

        def _transverse_size(axis):
            has_external_absorber = any(
                kind in absorbing_kinds for kind in scene.boundary.axis_face_kinds(axis)
            )
            return (4 if has_external_absorber else 2) * HALF_SPAN

        scene.add_structure(mw.Structure(
            # Continue a nominally infinite slab through external transverse
            # absorber cells. Ending it at Domain.bounds creates a finite plate
            # edge exactly at the PML entrance and contaminates the intended 1D
            # grid/material comparison with transverse diffraction.
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(_transverse_size("x"), _transverse_size("y"), 0.1),
            ),
            material=material, name="sample",
        ))
    scene.add_source(source or mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE))
    return _observed(scene)


def _waveguide_scene(*, diffraction=False) -> mw.Scene:
    scene = base_scene()
    # The guided wavelength at 2 GHz (n_eff ~ 1.96) is c/(f*n_eff) ~ 76 mm, so
    # the shared 25 mm grid leaves ~3 cells per guided wavelength and the mode
    # is beyond the Yee propagation cutoff (sin(beta*dx/2) > 1 at the benchmark
    # time step). Refine this scene to 12.5 mm (~6.1 cells per guided
    # wavelength) so the scenario validates modal launch, not grid cutoff junk.
    scene = mw.Scene(
        domain=scene.domain,
        grid=mw.GridSpec.uniform(DX / 2),
        boundary=scene.boundary,
        subpixel_samples=scene.subpixel,
        device="cpu",
    )
    if diffraction:
        boundary = mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic")
        scene = mw.Scene(domain=scene.domain, grid=scene.grid, boundary=boundary, device="cpu")
    scene.add_structure(mw.Structure(
        # A rectangular 0.20 x 0.30 core keeps the fundamental Ey/Ez pair
        # non-degenerate (a square core makes them exactly degenerate, and
        # eigensolvers return an arbitrary rotation of that subspace, so the
        # two solvers launch different polarizations). Continue the guide
        # through the external PML: ending it at the physical-domain boundary
        # creates a dielectric facet before the PML and turns the case into a
        # Fabry-Perot cavity.
        geometry=mw.Box(position=(0, 0, 0), size=(4 * HALF_SPAN, 0.20, 0.30)),
        material=mw.Material(eps_r=4.0), name="waveguide",
    ))
    scene.add_source(mw.ModeSource(position=(-0.35, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", source_time=PULSE, name="mode"))
    scene.add_monitor(mw.ModeMonitor("mode_out", position=(0.35, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", frequencies=FREQUENCIES))
    if diffraction:
        scene.add_monitor(mw.DiffractionMonitor("orders", position=(0, 0, 0.3), size=(1.0, 1.0, 0), frequencies=FREQUENCIES, orders=1))
    return _observed(
        scene,
        axis="y",
        component="Ez",
        flux_axis="x",
        flux_positions=(-0.5, 0.5),
    )


def _s_parameter_waveguide_scene(frequencies: tuple[float, ...]) -> mw.Scene:
    """Reciprocal rectangular guide with a calibrated input reference plane."""
    base = base_scene()
    # Match the mode-source validation grid: the shared 25 mm grid is beyond
    # the Yee propagation cutoff near 2 GHz for this n_eff ~ 1.96 guide.
    scene = mw.Scene(
        domain=base.domain,
        grid=mw.GridSpec.uniform(DX / 2),
        boundary=base.boundary,
        subpixel_samples=base.subpixel,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            # Continue the guide through the external PML. Ending it at the
            # physical-domain boundary creates a dielectric facet before the PML
            # and turns the validation case into a Fabry-Perot cavity.
            geometry=mw.Box(position=(0, 0, 0), size=(4 * HALF_SPAN, 0.20, 0.30)),
            material=mw.Material(eps_r=4.0),
            name="waveguide",
        )
    )
    center_frequency = 0.5 * (frequencies[0] + frequencies[-1])
    bandwidth = max(frequencies[-1] - frequencies[0], 0.5e9)
    scene.add_source(
        mw.ModeSource(
            position=(-0.35, 0, 0),
            size=(0, 0.5, 0.5),
            polarization="Ez",
            source_time=mw.GaussianPulse(
                frequency=center_frequency,
                fwidth=bandwidth,
            ),
            name="mode",
        )
    )
    scene.add_monitor(
        mw.ModeMonitor(
            "mode_out",
            position=(0.35, 0, 0),
            size=(0, 0.5, 0.5),
            polarization="Ez",
            direction="+",
            frequencies=frequencies,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="x",
            position=0.35,
            fields=("Ez",),
            frequencies=frequencies,
        )
    )
    return with_plot_monitors(scene, frequencies=frequencies)


def _custom_field_source():
    coords = np.linspace(-0.4, 0.4, 17)
    z = np.array([-0.25])
    shape = (coords.size, coords.size, 1)
    dataset = mw.FieldDataset((coords, coords, z), {"Ex": np.ones(shape), "Hy": np.ones(shape) / 376.730313668})
    return mw.CustomFieldSource(dataset, source_time=PULSE, name="custom_field")


def _custom_current_source():
    coords = np.linspace(-0.08, 0.08, 5)
    values = np.ones((5, 5, 5))
    return mw.CustomCurrentSource(mw.CurrentDataset((coords, coords, coords - 0.25), {"Jx": values}), source_time=PULSE, name="custom_current")


def _source_scene(source, *, monitor_position: float = 0.0) -> mw.Scene:
    scene = base_scene().add_source(source)
    scene.add_monitor(mw.FluxMonitor("radiated", axis="z", position=0.3, frequencies=FREQUENCIES))
    return _observed(scene, position=monitor_position)


def _scatter_scene(geometry=None) -> mw.Scene:
    geometry = geometry or mw.Sphere(position=(0, 0, 0), radius=0.16)
    scene = base_scene().add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=4), name="scatterer"))
    scene.add_source(mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE))
    return _observed(scene)


def _fdfd_scene(material: mw.Material, *, geometry=None) -> mw.Scene:
    half_span = 0.40
    resolution = 0.04
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-half_span, half_span),) * 3,
        ),
        grid=mw.GridSpec.uniform(resolution),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device="cpu",
    )
    geometry = geometry or mw.Box(
        position=(0.0, 0.0, 0.0),
        size=(2 * half_span, 2 * half_span, 0.12),
    )
    scene.add_structure(
        mw.Structure(geometry=geometry, material=material, name="sample")
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, -0.20),
            polarization="Ex",
            width=resolution,
            source_time=mw.CW(frequency=FREQUENCIES[0]),
            name="dipole",
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ex",),
            frequencies=FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field_output",
            axis="z",
            position=0.20,
            fields=("Ex",),
            frequencies=FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


def _full_tensor_scene() -> mw.Scene:
    tensor = mw.Tensor3x3(((3.0, 0.2, 0.0), (0.2, 2.5, 0.0), (0.0, 0.0, 2.0)))
    # Launch one transverse tensor eigenmode. This still exercises the coupled
    # Ex/Ey update while avoiding a grid-dispersion-sensitive beat between two
    # eigenmodes that obscures the constitutive validation.
    polarization = (0.9436283192, 0.3310069414, 0.0)
    source = mw.PlaneWave(
        direction=(0.0, 0.0, 1.0),
        polarization=polarization,
        source_time=PULSE,
    )
    return _plane_scene(
        material=mw.Material(epsilon_tensor=tensor),
        source=source,
        subpixel=mw.SubpixelSpec(samples=3, averaging="arithmetic"),
    )


def _lossy_metal_half_space() -> mw.Scene:
    scene = base_scene()
    # SIBC lives on a Yee node plane. Put the physical metal face on that
    # plane in both solvers; otherwise Maxwell voxelizes the requested face
    # while Tidy3D retains the continuous coordinate, adding a round-trip
    # reflection phase that is unrelated to the surface-impedance model.
    physical_cell_count = int(np.ceil(2.0 * HALF_SPAN / DX))
    physical_step = 2.0 * HALF_SPAN / physical_cell_count
    surface = HALF_SPAN - 2.0 * physical_step
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, surface + 0.25),
                size=(3.0, 3.0, 0.50),
            ),
            material=mw.LossyMetalMedium(conductivity=1.0e4),
            name="lossy_metal_boundary",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene, flux_positions=(-0.3, 0.45))


def _periodic_grating_scene(*, oblique: bool = False, diffraction: bool = False) -> mw.Scene:
    if oblique:
        theta = np.deg2rad(20.0)
        phi = np.deg2rad(30.0)
        direction = (
            float(np.sin(theta) * np.cos(phi)),
            float(np.sin(theta) * np.sin(phi)),
            float(np.cos(theta)),
        )
        bloch_kx = 2.0 * np.pi * GRATING_FREQUENCIES[0] * direction[0] / _C0
        bloch_ky = 2.0 * np.pi * GRATING_FREQUENCIES[0] * direction[1] / _C0
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=12,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(bloch_kx, bloch_ky, 0.0),
        )
        polarization = (-float(np.sin(phi)), float(np.cos(phi)), 0.0)
    else:
        direction = (0.0, 0.0, 1.0)
        polarization = (0.0, 1.0, 0.0)
        boundary = mw.BoundarySpec.pml(num_layers=12).with_faces(
            x_low="periodic",
            x_high="periodic",
            y_low="periodic",
            y_high="periodic",
        )

    reference = base_scene()
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.52, 2 * HALF_SPAN, 0.12)),
            material=mw.Material(eps_r=4.0),
            name="periodic_bar",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=GRATING_PULSE,
            injection=mw.TFSF.slab(axis="z", bounds=(-0.45, 0.45)),
            name="grating_tfsf",
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ey",),
            frequencies=GRATING_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "reflected",
            axis="z",
            position=-0.50,
            frequencies=GRATING_FREQUENCIES,
            normal_direction="-",
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "transmitted",
            axis="z",
            position=0.30,
            frequencies=GRATING_FREQUENCIES,
        )
    )
    if diffraction:
        scene.add_monitor(
            mw.DiffractionMonitor(
                "orders",
                position=(0.0, 0.0, 0.30),
                size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.0),
                frequencies=GRATING_FREQUENCIES,
                orders=3,
            )
        )
    return with_plot_monitors(scene, frequencies=GRATING_FREQUENCIES)


def _cavity(boundary, *, magnetic: bool = False) -> mw.Scene:
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=boundary,
        device="cpu",
    )
    source_position = (0.07, 0.03, -0.02)
    field_component = "Hz" if magnetic else "Ez"
    if magnetic:
        half_width = 0.02
        coords = tuple(
            (coordinate - half_width, coordinate + half_width)
            for coordinate in source_position
        )
        scene.add_source(
            mw.CustomCurrentSource(
                mw.CurrentDataset(coords, {"Mz": np.ones((2, 2, 2))}),
                source_time=CAVITY_PULSE,
                name="magnetic_cavity_source",
            )
        )
    else:
        scene.add_source(
            mw.PointDipole(
                position=source_position,
                polarization="Ez",
                source_time=CAVITY_PULSE,
            )
        )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.08,
            fields=(field_component,),
            frequencies=CAVITY_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            "resonance_probe",
            position=(0.13, 0.09, 0.04),
            fields=(field_component,),
        )
    )
    return with_plot_monitors(scene, frequencies=CAVITY_FREQUENCIES)


def _sphere_rcs_scene() -> mw.Scene:
    scene = base_scene()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12),
            material=mw.Material(eps_r=4.0),
            name="rcs_sphere",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=FAR_FIELD_PULSE,
            injection=mw.TFSF(bounds=((-0.22, 0.22),) * 3),
            name="rcs_tfsf",
        )
    )
    scene.add_monitor(
        mw.ClosedSurfaceMonitor.box(
            "huygens",
            position=(0.0, 0.0, 0.0),
            size=(0.60, 0.60, 0.60),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ex",),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=FAR_FIELD_FREQUENCIES)


def _antenna_directivity_scene() -> mw.Scene:
    scene = base_scene()
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=FAR_FIELD_PULSE,
            name="dipole",
        )
    )
    scene.add_monitor(
        mw.ClosedSurfaceMonitor.box(
            "huygens",
            position=(0.0, 0.0, 0.0),
            size=(0.30, 0.30, 0.30),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ez",),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=FAR_FIELD_FREQUENCIES)


def _make(
    name,
    family,
    observable,
    builder,
    *,
    component="Ex",
    solver="fdtd",
    reference_solver="tidy3d",
    run_time_factor=8.0,
    frequencies=FREQUENCIES,
    scalar_observable=None,
    compare_flux=True,
    compare_magnitude=False,
    normalize_source=None,
    spectral_reference_index=None,
    comparison_components=None,
):
    return ScenarioDefinition(name=name, description=f"{family}: {observable}", builder=builder,
        frequencies=frequencies, display_monitor="field", display_component=component,
        solver=solver, reference_solver=reference_solver, run_time_factor=run_time_factor,
        scalar_observable=scalar_observable, normalize_source=normalize_source,
        compare_flux=compare_flux, compare_magnitude=compare_magnitude,
        spectral_reference_index=spectral_reference_index,
        comparison_components=comparison_components)


PLANNED_SCENARIOS = (
    _make("astigmatic_beam", "sources", "astigmatic waist profile", lambda: _source_scene(mw.AstigmaticGaussianBeam(direction=(0,0,1), polarization="Ex", beam_waist=(0.22,0.35), focus=(0,0,0), focus_u=0.05, focus_v=-0.04, source_time=PULSE))),
    _make("uniform_current", "sources", "radiated power", lambda: _source_scene(mw.UniformCurrentSource(size=(0.12,0.12,0.12), polarization="Ex", center=(0,0,-0.2), source_time=PULSE), monitor_position=0.2)),
    _make("custom_field_source", "sources", "plane replay", lambda: _source_scene(_custom_field_source(), monitor_position=0.5)),
    _make("custom_current_source", "sources", "custom current radiation", lambda: _source_scene(_custom_current_source(), monitor_position=0.2)),
    _make("mode_source_wg", "sources", "transmitted mode power", _waveguide_scene, component="Ez"),
    _make("pec_box", "media", "PEC reflection", lambda: _plane_scene(material=mw.Material.pec())),
    _make("full_tensor_slab", "media", "full-tensor eigenpolarization", _full_tensor_scene),
    _make("sigma_e_slab", "media", "conductive absorption", lambda: _plane_scene(material=mw.Material(eps_r=3.0, sigma_e=0.05))),
    _make(
        "tpa_slab",
        "media",
        "two-photon absorption",
        lambda: _plane_scene(
            material=mw.Material(eps_r=3.0, nonlinearity=mw.TwoPhotonAbsorption(beta=1e-10)),
            boundary=mw.BoundarySpec.pml(num_layers=12).with_faces(
                x_low="periodic", x_high="periodic", y_low="periodic", y_high="periodic"
            ),
            source=mw.PlaneWave(
                direction=(0, 0, 1),
                polarization="Ex",
                source_time=mw.GaussianPulse(
                    frequency=FREQUENCIES[0],
                    fwidth=0.5e9,
                    amplitude=1.0e5,
                ),
            ),
        ),
    ),
    _make(
        "custom_pole_uniform_slab",
        "media",
        "uniform custom Lorentz pole",
        lambda: _plane_scene(
            material=mw.Material(
                eps_r=2.0,
                lorentz_poles=(
                    mw.CustomLorentzPole(
                        delta_eps=torch.ones((2, 2, 2)),
                        resonance_frequency=3e9,
                        gamma=0.2e9,
                    ),
                ),
            )
        ),
    ),
    _make(
        "perturbation_uniform_slab",
        "media",
        "uniform perturbation",
        lambda: _plane_scene(
            material=mw.PerturbationMedium(
                mw.Material(eps_r=2.0),
                perturbation=torch.ones((2, 2, 2)),
                eps_sensitivity=0.2,
            )
        ),
    ),
    _make("lossy_metal_slab", "media", "boundary-flush lossy-metal reflection", _lossy_metal_half_space),
    _make("sellmeier_slab", "media", "Sellmeier phase", lambda: _plane_scene(material=mw.Material.sellmeier(b_coefficients=(1.0,), c_coefficients=(1e-4,)))),
    _make("pml_only", "boundaries", "PML vacuum", lambda: _plane_scene(boundary=mw.BoundarySpec.pml(num_layers=12))),
    _make(
        "periodic_grating",
        "boundaries",
        "periodic transmission",
        _periodic_grating_scene,
        component="Ey",
        frequencies=GRATING_FREQUENCIES,
    ),
    _make(
        "bloch_oblique",
        "boundaries",
        "Bloch transmission",
        lambda: _periodic_grating_scene(oblique=True),
        component="Ey",
        frequencies=GRATING_FREQUENCIES,
    ),
    _make(
        "pec_cavity",
        "boundaries",
        "PEC resonance",
        lambda: _cavity(mw.BoundarySpec.pec()),
        component="Ez",
        run_time_factor=80,
        frequencies=CAVITY_FREQUENCIES,
        scalar_observable="cavity_resonance",
        compare_flux=False,
    ),
    _make(
        "pmc_cavity",
        "boundaries",
        "PMC resonance",
        lambda: _cavity(mw.BoundarySpec.pmc(), magnetic=True),
        component="Hz",
        run_time_factor=80,
        frequencies=CAVITY_FREQUENCIES,
        scalar_observable="cavity_resonance",
        compare_flux=False,
    ),
    _make("symmetry_center", "boundaries", "center symmetry", lambda: _plane_scene()),
    _make("custom_grid_slab", "grid_geometry", "custom-grid field", lambda: _plane_scene(material=mw.Material(eps_r=3), grid=mw.GridSpec.custom(np.linspace(-0.64,0.64,54), np.linspace(-0.64,0.64,54), np.linspace(-0.64,0.64,54)))),
    _make("autogrid_ring", "grid_geometry", "auto-grid resonance", lambda: _scatter_scene(mw.Torus(position=(0,0,0), major_radius=0.2, minor_radius=0.06))),
    _make("polyslab_wg", "grid_geometry", "PolySlab transmission", lambda: _scatter_scene(mw.PolySlab(vertices=((-0.2,-0.15),(0.2,-0.15),(0.2,0.15),(-0.2,0.15)), bounds=(-0.1,0.1)))),
    _make("mesh_primitive_scatter", "grid_geometry", "mesh primitive scattering", lambda: _scatter_scene(mw.Torus(position=(0,0,0), major_radius=0.18, minor_radius=0.06))),
    _make(
        "ring_resonator_s21",
        "postprocess",
        "all-pass ring S21 from solver-specific guided dispersion",
        lambda: _s_parameter_waveguide_scene(RING_FREQUENCIES),
        component="Ez",
        run_time_factor=30,
        frequencies=RING_FREQUENCIES,
        scalar_observable="ring_s21",
        compare_flux=False,
    ),
    _make(
        "waveguide_s_matrix",
        "postprocess",
        "reciprocal straight-waveguide S matrix",
        lambda: _s_parameter_waveguide_scene(FREQUENCIES),
        component="Ez",
        run_time_factor=20,
        scalar_observable="waveguide_s_matrix",
        compare_flux=False,
    ),
    _make(
        "grating_diffraction",
        "postprocess",
        "per-order diffraction efficiency",
        lambda: _periodic_grating_scene(diffraction=True),
        component="Ey",
        frequencies=GRATING_FREQUENCIES,
        scalar_observable="diffraction_orders",
    ),
    _make(
        "sphere_rcs",
        "postprocess",
        "bistatic sphere RCS",
        _sphere_rcs_scene,
        frequencies=FAR_FIELD_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="sphere_rcs",
        compare_flux=False,
    ),
    _make(
        "antenna_directivity",
        "postprocess",
        "dipole directivity and beamwidth",
        _antenna_directivity_scene,
        component="Ez",
        frequencies=FAR_FIELD_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="dipole_directivity",
        compare_flux=False,
    ),
    _make("fdfd_dielectric_slab", "fdfd", "dielectric-slab Ex-magnitude pattern", lambda: _fdfd_scene(mw.Material(eps_r=4)), component="Ex", solver="fdfd", reference_solver="fdtd", run_time_factor=20.0, compare_flux=False, compare_magnitude=True),
    _make("fdfd_drude_sphere", "fdfd", "Drude-sphere Ex-magnitude pattern", lambda: _fdfd_scene(mw.Material.drude(eps_inf=1.0, plasma_frequency=5.0e9, gamma=0.1e9), geometry=mw.Sphere(position=(0,0,0), radius=0.12)), component="Ex", solver="fdfd", reference_solver="fdtd", run_time_factor=20.0, compare_flux=False, compare_magnitude=True),
    _make("fdfd_sigma_e_slab", "fdfd", "conductive-slab Ex-magnitude pattern", lambda: _fdfd_scene(mw.Material(eps_r=3, sigma_e=0.05)), component="Ex", solver="fdfd", reference_solver="fdtd", run_time_factor=20.0, compare_flux=False, compare_magnitude=True),
    _make("fdfd_diag_aniso_slab", "fdfd", "diagonal-anisotropic-slab Ex-magnitude pattern", lambda: _fdfd_scene(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2,3,4))), component="Ex", solver="fdfd", reference_solver="fdtd", run_time_factor=20.0, compare_flux=False, compare_magnitude=True),
)
