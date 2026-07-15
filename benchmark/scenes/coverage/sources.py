"""Additional source-family coverage: Gaussian beams, CW drive, TFSF, higher-order modes."""

from __future__ import annotations

import numpy as np

import witwin.maxwell as mw
from benchmark.scenes._common import HALF_SPAN, base_scene, with_plot_monitors
from benchmark.scenes.planned import FREQUENCIES, PULSE, _make, _observed, _source_scene

CW_DRIVE = mw.CW(frequency=FREQUENCIES[0])
CW_RUN_TIME_FACTOR = 24.0


def _tfsf_scene(*, bounds, geometry=None, flux_positions) -> mw.Scene:
    """Plane wave injected through a TFSF box, optionally around a scatterer."""
    scene = base_scene()
    if geometry is not None:
        scene.add_structure(
            mw.Structure(geometry=geometry, material=mw.Material(eps_r=4.0), name="scatterer")
        )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=PULSE,
            injection=mw.TFSF(bounds=bounds),
            name="tfsf",
        )
    )
    return _observed(scene, flux_positions=flux_positions)


def _higher_order_waveguide_scene() -> mw.Scene:
    """Straight guide launched and probed on the mode_index=1 order."""
    scene = base_scene()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0, 0, 0), size=(2 * HALF_SPAN, 0.30, 0.15)),
            material=mw.Material(eps_r=4.0),
            name="waveguide",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.35, 0, 0),
            size=(0, 0.525, 0.375),
            mode_index=1,
            polarization="Ez",
            source_time=PULSE,
            name="mode",
        )
    )
    scene.add_monitor(
        mw.FinitePlaneMonitor(
            "field",
            position=(0.20, 0, 0),
            size=(0, 0.525, 0.375),
            fields=("Ex", "Ey", "Ez"),
            frequencies=FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.ModeMonitor(
            "mode_out",
            position=(0.35, 0, 0),
            size=(0, 0.525, 0.375),
            mode_index=1,
            polarization="Ez",
            frequencies=FREQUENCIES,
        )
    )
    scene.add_monitor(
        mw.FluxMonitor(
            "reflected", axis="x", position=-0.5, frequencies=FREQUENCIES, normal_direction="-"
        )
    )
    scene.add_monitor(
        mw.FluxMonitor("transmitted", axis="x", position=0.5, frequencies=FREQUENCIES)
    )
    return with_plot_monitors(scene, frequencies=FREQUENCIES)


def _magnetic_current_scene() -> mw.Scene:
    """Uniform Mz volume current in vacuum (adapter maps M* to Tidy3D H* components)."""
    coords = np.linspace(-0.06, 0.06, 5)
    scene = base_scene().add_source(
        mw.CustomCurrentSource(
            mw.CurrentDataset(
                (coords, coords, coords - 0.2), {"Mz": np.ones((5, 5, 5))}
            ),
            source_time=PULSE,
            name="magnetic_current",
        )
    )
    scene.add_monitor(mw.FluxMonitor("radiated", axis="z", position=0.3, frequencies=FREQUENCIES))
    # A z-directed magnetic dipole radiates a phi-directed E field, i.e. Ey on the y=0 cut.
    return _observed(scene, axis="y", position=0.0, component="Ey")


def _ricker_axis_x_anisotropic_scene() -> mw.Scene:
    """Negative-x Ricker launch through a slab whose Ez response uses eps_zz."""
    scene = base_scene().add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(0.12, 2.0 * HALF_SPAN, 2.0 * HALF_SPAN),
            ),
            material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0)),
            name="axis_x_anisotropic_slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(-1.0, 0.0, 0.0),
            polarization="Ez",
            source_time=mw.RickerWavelet(frequency=FREQUENCIES[0]),
            name="ricker_axis_x",
        )
    )
    return _observed(
        scene,
        axis="y",
        component="Ez",
        flux_axis="x",
        flux_positions=(-0.50, 0.50),
    )


SOURCE_COVERAGE_SCENARIOS = (
    _make(
        "gaussian_beam_normal",
        "sources",
        "on-axis Gaussian beam waist",
        lambda: _source_scene(
            mw.GaussianBeam(
                direction=(0, 0, 1),
                polarization="Ex",
                beam_waist=0.25,
                focus=(0, 0, 0),
                source_time=PULSE,
                name="gaussian_beam",
            )
        ),
    ),
    _make(
        "gaussian_beam_defocused",
        "sources",
        "Gaussian beam focused downstream of the injection plane",
        lambda: _source_scene(
            mw.GaussianBeam(
                direction=(0, 0, 1),
                polarization="Ex",
                beam_waist=0.18,
                focus=(0, 0, 0.40),
                source_time=PULSE,
                name="gaussian_beam_defocused",
            ),
            monitor_position=0.0,
        ),
    ),
    _make(
        "planewave_cw",
        "sources",
        "continuous-wave plane wave in vacuum",
        lambda: _observed(
            base_scene().add_source(
                mw.PlaneWave(
                    direction=(0, 0, 1),
                    polarization="Ex",
                    source_time=CW_DRIVE,
                    name="cw_plane_wave",
                )
            )
        ),
        run_time_factor=CW_RUN_TIME_FACTOR,
    ),
    _make(
        "dipole_cw_vacuum",
        "sources",
        "continuous-wave point-dipole radiation",
        lambda: _source_scene(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ex",
                profile="ideal",
                source_time=CW_DRIVE,
                name="cw_dipole",
            )
        ),
        run_time_factor=CW_RUN_TIME_FACTOR,
    ),
    _make(
        "tfsf_vacuum",
        "sources",
        "TFSF box in vacuum (zero scattered field outside the box)",
        lambda: _tfsf_scene(bounds=((-0.30, 0.30),) * 3, flux_positions=(-0.40, 0.40)),
    ),
    _make(
        "tfsf_dielectric_sphere",
        "sources",
        "TFSF box around a dielectric sphere (scattered-field flux)",
        lambda: _tfsf_scene(
            bounds=((-0.28, 0.28),) * 3,
            geometry=mw.Sphere(position=(0, 0, 0), radius=0.16),
            flux_positions=(-0.40, 0.40),
        ),
    ),
    _make(
        "mode_source_higher_order",
        "sources",
        "mode_index=1 launch and modal transmission",
        _higher_order_waveguide_scene,
        component="Ez",
        scalar_observable="mode_effective_index",
        comparison_components=("Ex", "Ey", "Ez"),
    ),
    _make(
        "magnetic_current_vacuum",
        "sources",
        "uniform magnetic volume current radiation",
        _magnetic_current_scene,
        component="Ey",
    ),
    _make(
        "ricker_axis_x_anisotropic",
        "sources",
        "negative-x Ricker launch through an eps_zz anisotropic slab",
        _ricker_axis_x_anisotropic_scene,
        component="Ez",
    ),
)


__all__ = ["SOURCE_COVERAGE_SCENARIOS"]
