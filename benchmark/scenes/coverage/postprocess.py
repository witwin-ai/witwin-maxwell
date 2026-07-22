"""Independent postprocess coverage: far field, modal projection, diffraction, probes.

These are deliberately reduced cases that overlap the existing postprocess family
(sphere_rcs, antenna_directivity, waveguide_s_matrix, grating_diffraction) so a
failure can be localized to one mechanism instead of a compound scene.
"""

from __future__ import annotations

import witwin.maxwell as mw
from benchmark.scenes._common import HALF_SPAN, base_scene, with_plot_monitors
from benchmark.scenes.planned import (
    FAR_FIELD_FREQUENCIES,
    FAR_FIELD_PULSE,
    PULSE,
    _C0,
    _make,
    _observed,
)

# Far-field cases reuse the CW convention of the existing Huygens scenarios.
RCS_TFSF_BOUNDS = ((-0.22, 0.22),) * 3
RCS_SURFACE_SIZE = (0.60, 0.60, 0.60)

# Two in-phase Ez elements separated by half a free-space wavelength. The array
# factor nulls along the element axis, so the pattern is strongly shaped and the
# directivity comparison is sensitive to the near-to-far-field transform.
DIPOLE_SEPARATION = 0.5 * _C0 / FAR_FIELD_FREQUENCIES[0]
ARRAY_SURFACE_SIZE = (0.80, 0.50, 0.50)

# The modal cases run at 0.5 GHz because a guide that is single-mode at 2 GHz is
# unresolvable on this grid. At 0.5 GHz the 0.25 x 0.175 m eps=4 core carries only
# the fundamental pair, which removes the mode-ordering ambiguity of the existing
# multimode S-matrix scene.
MODE_FREQUENCIES = (0.5e9,)
MODE_PULSE = mw.GaussianPulse(frequency=MODE_FREQUENCIES[0], fwidth=0.15e9)
MODE_CORE_SIZE = (4 * HALF_SPAN, 0.25, 0.175)
MODE_PLANE_SIZE = (0.0, 0.7, 0.7)
MODE_SOURCE_X = -0.30

# Period 2*HALF_SPAN = 1.28 m with lambda = 0.75 m leaves exactly the 0th and
# +-1st orders propagating, so the order decomposition is unambiguous.
DIFFRACTION_FREQUENCIES = (0.4e9,)
DIFFRACTION_PULSE = mw.GaussianPulse(frequency=DIFFRACTION_FREQUENCIES[0], fwidth=0.15e9)

PROBE_POSITIONS = (
    (0.10, 0.0, 0.0),
    (0.25, 0.0, 0.0),
    (0.45, 0.0, 0.0),
    (0.0, 0.0, 0.25),
    (0.20, 0.20, 0.10),
)


def _tfsf_rcs_scene(geometry, material, *, name: str) -> mw.Scene:
    scene = base_scene()
    scene.add_structure(mw.Structure(geometry=geometry, material=material, name=name))
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=FAR_FIELD_PULSE,
            injection=mw.TFSF(bounds=RCS_TFSF_BOUNDS),
            name="rcs_tfsf",
        )
    )
    scene.add_monitor(
        mw.ClosedSurfaceMonitor.box(
            "huygens",
            position=(0.0, 0.0, 0.0),
            size=RCS_SURFACE_SIZE,
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


def _two_dipole_array_scene() -> mw.Scene:
    scene = base_scene()
    for index, offset in enumerate((-0.5 * DIPOLE_SEPARATION, 0.5 * DIPOLE_SEPARATION)):
        scene.add_source(
            mw.PointDipole(
                position=(offset, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=FAR_FIELD_PULSE,
                name=f"dipole_{index}",
            )
        )
    scene.add_monitor(
        mw.ClosedSurfaceMonitor.box(
            "huygens",
            position=(0.0, 0.0, 0.0),
            size=ARRAY_SURFACE_SIZE,
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    # Offset the display plane from y = 0 so neither singular element cell enters
    # the field metric; the array main beam points along +y.
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.30,
            fields=("Ez",),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=FAR_FIELD_FREQUENCIES)


def _single_mode_guide_scene(*, mode_monitors: tuple[tuple[str, float], ...]) -> mw.Scene:
    scene = base_scene()
    scene.add_structure(
        mw.Structure(
            # Run the core through the external PML on both ends so the case is a
            # pure launch/propagate/project check with no dielectric facet.
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=MODE_CORE_SIZE),
            material=mw.Material(eps_r=4.0),
            name="waveguide",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(MODE_SOURCE_X, 0.0, 0.0),
            size=MODE_PLANE_SIZE,
            polarization="Ey",
            source_time=MODE_PULSE,
            name="mode",
        )
    )
    for name, position_x in mode_monitors:
        scene.add_monitor(
            mw.ModeMonitor(
                name,
                position=(position_x, 0.0, 0.0),
                size=MODE_PLANE_SIZE,
                polarization="Ey",
                direction="+",
                frequencies=MODE_FREQUENCIES,
            )
        )
    # A longitudinal cut through the core shows the accumulated guided phase, which
    # is the quantity the modal amplitudes are supposed to reproduce.
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="z",
            position=0.0,
            fields=("Ey",),
            frequencies=MODE_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=MODE_FREQUENCIES)


def _normal_incidence_grating_scene() -> mw.Scene:
    reference = base_scene()
    boundary = mw.BoundarySpec.pml(num_layers=12).with_faces(
        x_low="periodic",
        x_high="periodic",
        y_low="periodic",
        y_high="periodic",
    )
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            # Half-period fill with a ~0.7 pi single-pass phase delay puts a large
            # fraction of the transmitted power into the +-1st orders.
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(HALF_SPAN, 2 * HALF_SPAN, 0.25)),
            material=mw.Material(eps_r=4.0),
            name="grating_bar",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(0.0, 1.0, 0.0),
            source_time=DIFFRACTION_PULSE,
            name="grating_plane_wave",
        )
    )
    # Keep the order plane far enough above the bar that the +-2nd evanescent
    # orders have decayed before the transverse Fourier decomposition.
    scene.add_monitor(
        mw.DiffractionMonitor(
            "orders",
            position=(0.0, 0.0, 0.45),
            size=(2 * HALF_SPAN, 2 * HALF_SPAN, 0.0),
            frequencies=DIFFRACTION_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ey",),
            frequencies=DIFFRACTION_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "reflected",
            axis="z",
            position=-0.45,
            frequencies=DIFFRACTION_FREQUENCIES,
            normal_direction="-",
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "transmitted",
            axis="z",
            position=0.50,
            frequencies=DIFFRACTION_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=DIFFRACTION_FREQUENCIES)


def _point_probe_scene() -> mw.Scene:
    scene = base_scene()
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=PULSE,
            name="dipole",
        )
    )
    for index, position in enumerate(PROBE_POSITIONS):
        scene.add_monitor(
            mw.PointMonitor(f"probe_{index}", position=position, fields=("Ex", "Ey", "Ez"))
        )
    return _observed(scene, axis="y", position=0.0, component="Ez")


def _mode_port_scene() -> mw.Scene:
    """Straight guide assembled through the public ModePort abstraction."""
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=MODE_CORE_SIZE),
            material=mw.Material(eps_r=4.0),
            name="waveguide",
        )
    )
    scene.add_port(
        mw.ModePort(
            "mode_in",
            position=(MODE_SOURCE_X, 0.0, 0.0),
            size=MODE_PLANE_SIZE,
            polarization="Ey",
            frequencies=MODE_FREQUENCIES,
            source_time=MODE_PULSE,
            monitor_offset=0.05,
        )
    )
    scene.add_port(
        mw.ModePort(
            "mode_out",
            position=(0.30, 0.0, 0.0),
            size=MODE_PLANE_SIZE,
            polarization="Ey",
            frequencies=MODE_FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="z",
            position=0.0,
            fields=("Ey",),
            frequencies=MODE_FREQUENCIES,
        )
    )
    return with_plot_monitors(scene, frequencies=MODE_FREQUENCIES)


def _permittivity_monitor_scene() -> mw.Scene:
    """Volume permittivity monitor over a dielectric slab and vacuum."""
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(0.50, 0.42, 0.20),
            ),
            material=mw.Material(eps_r=4.0),
            name="permittivity_box",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0, 0, 1),
            polarization="Ex",
            source_time=MODE_PULSE,
        )
    )
    scene.add_monitor(
        mw.PermittivityMonitor(
            "permittivity",
            position=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.8),
            frequencies=FAR_FIELD_FREQUENCIES,
        )
    )
    return _observed(
        scene,
        frequencies=FAR_FIELD_FREQUENCIES,
        component="Ex",
    )


def _time_monitor_scene() -> mw.Scene:
    """Point-field and plane-flux time traces sampled on the shared FDTD step."""
    scene = base_scene().add_source(
        mw.PlaneWave(
            direction=(0, 0, 1),
            polarization="Ex",
            source_time=MODE_PULSE,
        )
    )
    scene.add_monitor(
        mw.FieldTimeMonitor(
            "field_time",
            components=("Ex",),
            position=(0.0, 0.0, 0.25),
            interval=2,
        )
    )
    scene.add_monitor(
        mw.FluxTimeMonitor(
            "flux_time",
            axis="z",
            position=0.30,
            interval=2,
        )
    )
    return _observed(
        scene,
        frequencies=MODE_FREQUENCIES,
        component="Ex",
    )


POSTPROCESS_COVERAGE_SCENARIOS = (
    _make(
        "rcs_pec_sphere",
        "postprocess",
        "bistatic PEC-sphere RCS",
        lambda: _tfsf_rcs_scene(
            mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.12),
            mw.Material.pec(),
            name="rcs_pec_sphere",
        ),
        frequencies=FAR_FIELD_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="sphere_rcs",
        compare_flux=False,
    ),
    _make(
        "rcs_dielectric_box",
        "postprocess",
        "bistatic flat-facet dielectric-box RCS",
        lambda: _tfsf_rcs_scene(
            mw.Box(position=(0.0, 0.0, 0.0), size=(0.20, 0.20, 0.20)),
            mw.Material(eps_r=4.0),
            name="rcs_box",
        ),
        frequencies=FAR_FIELD_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="sphere_rcs",
        compare_flux=False,
    ),
    _make(
        "directivity_two_dipoles",
        "postprocess",
        "half-wavelength two-element array directivity",
        _two_dipole_array_scene,
        component="Ez",
        frequencies=FAR_FIELD_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="dipole_directivity",
        compare_flux=False,
        # The array carries two sources, which the FDTD spectral sampler cannot
        # source-normalize. Disabling it keeps Maxwell and Tidy3D on the same raw
        # spectra; directivity is a scale-invariant ratio, so nothing is lost.
        normalize_source=False,
        # With source normalization off, the Maxwell fields are raw DFT sums
        # (O(1e5)) while the reference stores physical phasors (O(1e-5)), so the
        # amplitude ratio alone inflates field_l2 by ~1e7. Reference both spectra
        # to the display-monitor RMS at the carrier so the metric measures field
        # shape; the directivity scalar is a ratio and is unaffected.
        spectral_reference_index=0,
    ),
    _make(
        "mode_monitor_straight_wg",
        "postprocess",
        "single-mode straight-guide modal projection",
        lambda: _single_mode_guide_scene(mode_monitors=(("mode_out", 0.30),)),
        component="Ey",
        frequencies=MODE_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="waveguide_s_matrix",
        compare_flux=False,
    ),
    _make(
        "mode_monitor_two_planes",
        "postprocess",
        "modal amplitude ratio across two reference planes",
        lambda: _single_mode_guide_scene(
            mode_monitors=(("mode_out", 0.30), ("mode_mid", 0.0)),
        ),
        component="Ey",
        frequencies=MODE_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="mode_plane_ratio",
        compare_flux=False,
    ),
    _make(
        "diffraction_normal_orders",
        "postprocess",
        "normal-incidence 0th and +-1st order efficiency",
        _normal_incidence_grating_scene,
        component="Ey",
        frequencies=DIFFRACTION_FREQUENCIES,
        run_time_factor=12,
        scalar_observable="diffraction_orders",
    ),
    _make(
        "point_monitor_probe",
        "postprocess",
        "point-probe field values around an Ez dipole",
        _point_probe_scene,
        component="Ez",
        scalar_observable="point_probe_values",
    ),
    _make(
        "mode_port_straight_wg",
        "postprocess",
        "ModePort launch and complex through-port amplitude",
        _mode_port_scene,
        component="Ey",
        frequencies=MODE_FREQUENCIES,
        run_time_factor=20,
        scalar_observable="mode_port_transmission",
        compare_flux=False,
    ),
    _make(
        "permittivity_monitor_slab",
        "postprocess",
        "volume PermittivityMonitor material statistics",
        _permittivity_monitor_scene,
        component="Ex",
        frequencies=FAR_FIELD_FREQUENCIES,
        scalar_observable="permittivity_stats",
    ),
    _make(
        "time_monitor_vacuum",
        "postprocess",
        "FieldTimeMonitor and FluxTimeMonitor normalized traces",
        _time_monitor_scene,
        component="Ex",
        frequencies=MODE_FREQUENCIES,
        scalar_observable="time_monitor_traces",
        normalize_source=False,
        spectral_reference_index=0,
        compare_flux=False,
    ),
)
