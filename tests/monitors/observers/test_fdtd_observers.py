import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import calculate_required_steps
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.postprocess import NearFieldFarFieldTransformer, equivalent_surface_currents_from_monitor


def _build_scene(*, bounds, spacing):
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="box",
            geometry=mw.Box(position=(0.0, 0.2, 0.0), size=(0.2, 0.2, 0.2)),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )
    return scene


def _build_broadband_scene(*, bounds, spacing):
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="box",
            geometry=mw.Box(position=(0.0, 0.1, 0.0), size=(0.18, 0.18, 0.18)),
            material=mw.Material(eps_r=3.5),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=100.0),
            name="src",
        )
    )
    return scene


def _build_pulse_scene(*, source_time):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=source_time,
            name="src",
        )
    )
    return scene


def _domain_size(scene):
    return max(float(high - low) for low, high in scene.domain.bounds)


def _fixed_runtime(scene, frequencies, *, steady_cycles=4, transient_cycles=15):
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[min(frequencies)],
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).prepare()
    fdtd = prepared.solver
    domain_size = _domain_size(scene)
    steps = calculate_required_steps(
        frequency=min(frequencies),
        dt=fdtd.dt,
        c=fdtd.c,
        num_cycles=steady_cycles,
        transient_cycles=transient_cycles,
        domain_size=domain_size,
        source_time=getattr(fdtd, "_source_time", None),
    )
    return mw.TimeConfig(time_steps=steps)


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _manual_flux(monitor):
    axis = monitor["axis"]
    if axis != "z":
        raise ValueError("This test helper currently expects a z-normal plane.")
    x = np.asarray(monitor["x"], dtype=float)
    y = np.asarray(monitor["y"], dtype=float)
    wx = np.empty_like(x)
    wy = np.empty_like(y)
    dx = np.diff(x)
    dy = np.diff(y)
    wx[0] = dx[0]
    wx[-1] = dx[-1]
    wy[0] = dy[0]
    wy[-1] = dy[-1]
    if x.size > 2:
        wx[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    if y.size > 2:
        wy[1:-1] = (dy[:-1] + dy[1:]) / 2.0
    weights = wx[:, None] * wy[None, :]
    ex = _to_cpu_numpy(monitor["Ex"])
    ey = _to_cpu_numpy(monitor["Ey"])
    hx = _to_cpu_numpy(monitor["Hx"])
    hy = _to_cpu_numpy(monitor["Hy"])
    sz = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
    return float(np.sum(sz * weights))


def _interpolate_axis(field, coords, position, axis):
    values = np.asarray(field)
    axis_coords = np.asarray(coords, dtype=float)
    target = float(position)
    upper = int(np.searchsorted(axis_coords, target, side="left"))
    if upper <= 0:
        return np.take(values, 0, axis=axis)
    if upper >= axis_coords.size:
        return np.take(values, axis_coords.size - 1, axis=axis)

    lower = upper - 1
    lower_coord = float(axis_coords[lower])
    upper_coord = float(axis_coords[upper])
    tolerance = 1e-12 * max(abs(target), abs(lower_coord), abs(upper_coord), 1.0)
    if abs(target - lower_coord) <= tolerance:
        return np.take(values, lower, axis=axis)
    if abs(target - upper_coord) <= tolerance:
        return np.take(values, upper, axis=axis)

    upper_weight = (target - lower_coord) / (upper_coord - lower_coord)
    lower_weight = 1.0 - upper_weight
    return (
        lower_weight * np.take(values, lower, axis=axis)
        + upper_weight * np.take(values, upper, axis=axis)
    )


def _plane_reference(full_result, scene, component, *, axis, position):
    field = full_result.tensor(component).detach().cpu().numpy()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    normal_coords = component_coords(scene, component)[axis_index]
    return _interpolate_axis(field, normal_coords, position, axis_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_pulsed_sources_start_spectral_accumulation_immediately():
    pulsed_scene = _build_pulse_scene(
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=100.0)
    )
    pulsed_solver = mw.Simulation.fdtd(
        pulsed_scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).prepare().solver

    cw_scene = _build_pulse_scene(
        source_time=mw.CW(frequency=1e9, amplitude=100.0)
    )
    cw_solver = mw.Simulation.fdtd(
        cw_scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).prepare().solver

    assert pulsed_solver._compute_spectral_start_step(1e9) == 0
    assert cw_solver._compute_spectral_start_step(1e9) > 0
    assert pulsed_solver._resolve_spectral_window_type("hanning") == "none"
    assert cw_solver._resolve_spectral_window_type("hanning") == "hanning"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_selective_observers_and_precomputed_coefficients():
    scene = _build_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    scene.add_monitor(mw.PointMonitor("center_ez", (0.0, 0.0, 0.0), fields=("Ez",)))
    scene.add_monitor(mw.PlaneMonitor("midplane_ez", axis="z", position=0.0, fields=("Ez",)))

    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    )
    prepared = sim.prepare()
    fdtd = prepared.solver

    assert fdtd.cex_decay.shape == fdtd.Ex.shape
    assert fdtd.cex_curl.shape == fdtd.Ex.shape
    assert fdtd.cey_decay.shape == fdtd.Ey.shape
    assert fdtd.cey_curl.shape == fdtd.Ey.shape
    assert fdtd.cez_decay.shape == fdtd.Ez.shape
    assert fdtd.cez_curl.shape == fdtd.Ez.shape
    assert fdtd.chx_decay.shape == fdtd.Hx.shape
    assert fdtd.chx_curl.shape == fdtd.Hx.shape
    assert fdtd.chy_decay.shape == fdtd.Hy.shape
    assert fdtd.chy_curl.shape == fdtd.Hy.shape
    assert fdtd.chz_decay.shape == fdtd.Hz.shape
    assert fdtd.chz_curl.shape == fdtd.Hz.shape

    domain_size = _domain_size(scene)
    steps = calculate_required_steps(
        frequency=1e9,
        dt=fdtd.dt,
        c=fdtd.c,
        num_cycles=4,
        transient_cycles=15,
        domain_size=domain_size,
    )
    prepared.simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = prepared.run()
    observers = result.monitors
    assert set(observers) == {"center_ez", "midplane_ez"}
    assert torch.is_tensor(observers["center_ez"]["data"])
    assert _to_cpu_numpy(observers["center_ez"]["data"]).shape == ()
    assert observers["midplane_ez"]["data"].shape == fdtd.Ez[:, :, fdtd.Ez.shape[2] // 2].shape
    assert torch.abs(observers["center_ez"]["data"]).item() > 0.0
    assert torch.max(torch.abs(observers["midplane_ez"]["data"])).item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_observers_match_full_field_dft_slices():
    scene = _build_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    scene.add_monitor(mw.PointMonitor("center_ez", (0.0, 0.0, 0.0), fields=("Ez",)))
    scene.add_monitor(mw.PlaneMonitor("midplane_ez", axis="z", position=0.0, fields=("Ez",)))

    observer_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    full_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    point_observer = observer_result.monitor("center_ez")
    plane_observer = observer_result.monitor("midplane_ez")
    point_reference = full_result.tensor("Ez").detach().cpu().numpy()[point_observer["field_index"]]
    plane_reference = _plane_reference(full_result, scene, "Ez", axis="z", position=0.0)

    np.testing.assert_allclose(_to_cpu_numpy(point_observer["data"]), point_reference, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_to_cpu_numpy(plane_observer["data"]), plane_reference, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_plane_monitor_supports_multi_component_eh_and_postprocess_bridge():
    scene = _build_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "nf_box",
            axis="z",
            position=0.0,
            fields=("Ex", "Ey", "Hx", "Hy"),
        )
    )

    observer_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    full_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    monitor = observer_result.monitor("nf_box")
    prepared_scene = observer_result.prepared_scene
    assert monitor["fields"] == ("Ex", "Ey", "Hx", "Hy")
    assert set(monitor["components"]) == {"Ex", "Ey", "Hx", "Hy"}
    assert monitor["x"].shape == (prepared_scene.Nx - 1,)
    assert monitor["y"].shape == (prepared_scene.Ny - 1,)
    assert monitor["Ex"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    assert monitor["Ey"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    assert monitor["Hx"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    assert monitor["Hy"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)

    ex_reference = _plane_reference(full_result, scene, "Ex", axis="z", position=0.0)
    ey_reference = _plane_reference(full_result, scene, "Ey", axis="z", position=0.0)
    ex_reference = 0.5 * (ex_reference[:, :-1] + ex_reference[:, 1:])
    ey_reference = 0.5 * (ey_reference[:-1, :] + ey_reference[1:, :])

    np.testing.assert_allclose(_to_cpu_numpy(monitor["Ex"]), ex_reference, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_to_cpu_numpy(monitor["Ey"]), ey_reference, rtol=1e-2, atol=1e-3)
    assert torch.max(torch.abs(monitor["Hx"])).item() > 0.0
    assert torch.max(torch.abs(monitor["Hy"])).item() > 0.0

    currents = equivalent_surface_currents_from_monitor(observer_result, "nf_box")
    assert currents.J.shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1, 3)
    assert currents.M.shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1, 3)

    far_field = NearFieldFarFieldTransformer(currents, solver=observer_result.solver).transform(
        theta=np.array([0.0, 0.2]),
        phi=np.array([0.0, 0.0]),
        radius=5.0,
    )
    assert far_field["Ex"].shape == (2,)
    assert far_field["power_density"].shape == (2,)
    assert np.max(np.abs(_to_cpu_numpy(far_field["E"]))) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_plane_monitor_aligns_normal_electric_component_to_in_plane_centers():
    scene = _build_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field_xy",
            axis="z",
            position=0.0,
            fields=("Ex", "Ey", "Ez"),
        )
    )

    observer_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    full_result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    monitor = observer_result.monitor("field_xy")
    prepared_scene = observer_result.prepared_scene
    ex_reference = _plane_reference(full_result, scene, "Ex", axis="z", position=0.0)
    ey_reference = _plane_reference(full_result, scene, "Ey", axis="z", position=0.0)
    ez_reference = _plane_reference(full_result, scene, "Ez", axis="z", position=0.0)
    ex_reference = 0.5 * (ex_reference[:, :-1] + ex_reference[:, 1:])
    ey_reference = 0.5 * (ey_reference[:-1, :] + ey_reference[1:, :])
    ez_reference = 0.25 * (
        ez_reference[:-1, :-1]
        + ez_reference[1:, :-1]
        + ez_reference[:-1, 1:]
        + ez_reference[1:, 1:]
    )

    assert monitor["Ex"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    assert monitor["Ey"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    assert monitor["Ez"].shape == (prepared_scene.Nx - 1, prepared_scene.Ny - 1)
    np.testing.assert_allclose(_to_cpu_numpy(monitor["Ex"]), ex_reference, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_to_cpu_numpy(monitor["Ey"]), ey_reference, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_to_cpu_numpy(monitor["Ez"]), ez_reference, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_odd_grid_dimensions_cover_tail_cells():
    scene = _build_scene(
        bounds=((-0.455, 0.455), (-0.385, 0.385), (-0.315, 0.315)),
        spacing=0.07,
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()
    prepared_scene = result.prepared_scene

    # The odd physical counts (13, 11, 9) remain intact and four PML cells are
    # appended outside each face.
    assert prepared_scene.Nx == 21
    assert prepared_scene.Ny == 19
    assert prepared_scene.Nz == 17
    assert result.tensor("Ex").shape == (prepared_scene.Nx - 1, prepared_scene.Ny, prepared_scene.Nz)
    assert result.tensor("Ey").shape == (prepared_scene.Nx, prepared_scene.Ny - 1, prepared_scene.Nz)
    assert result.tensor("Ez").shape == (prepared_scene.Nx, prepared_scene.Ny, prepared_scene.Nz - 1)
    assert torch.max(torch.abs(result.tensor("Ez"))).item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_multi_frequency_full_field_matches_single_frequency_runs():
    frequencies = (0.9e9, 1.1e9)
    scene = _build_broadband_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    runtime = _fixed_runtime(scene, frequencies)

    multi_result = mw.Simulation.fdtd(
        scene,
        frequencies=frequencies,
        run_time=runtime,
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()

    for frequency in frequencies:
        single_result = mw.Simulation.fdtd(
            scene,
            frequencies=[frequency],
            run_time=runtime,
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=True,
        ).run()
        np.testing.assert_allclose(
            multi_result.tensor("Ex", frequency=frequency).detach().cpu().numpy(),
            single_result.tensor("Ex").detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            multi_result.tensor("Ey", frequency=frequency).detach().cpu().numpy(),
            single_result.tensor("Ey").detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            multi_result.tensor("Ez", frequency=frequency).detach().cpu().numpy(),
            single_result.tensor("Ez").detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_multi_frequency_flux_monitor_matches_single_frequency_runs():
    frequencies = (0.9e9, 1.1e9)
    position = 0.16
    scene = _build_broadband_scene(
        bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
        spacing=0.08,
    )
    scene.add_monitor(mw.FluxMonitor("port_flux", axis="z", position=position, frequencies=frequencies))
    scene.add_monitor(
        mw.PlaneMonitor(
            "port_plane",
            axis="z",
            position=position,
            fields=("Ex", "Ey", "Hx", "Hy"),
            frequencies=frequencies,
            compute_flux=True,
        )
    )
    runtime = _fixed_runtime(scene, frequencies)

    multi_result = mw.Simulation.fdtd(
        scene,
        frequencies=frequencies,
        run_time=runtime,
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    flux_monitor = multi_result.monitor("port_flux")
    plane_monitor = multi_result.monitor("port_plane")
    assert flux_monitor["flux"].shape == (2,)
    np.testing.assert_allclose(_to_cpu_numpy(flux_monitor["flux"]), _to_cpu_numpy(plane_monitor["flux"]), rtol=1e-4, atol=1e-5)

    for frequency in frequencies:
        selected_plane = multi_result.monitor("port_plane", frequency=frequency)
        manual_flux = _manual_flux(selected_plane)
        np.testing.assert_allclose(_to_cpu_numpy(selected_plane["flux"]), manual_flux, rtol=1e-5, atol=1e-6)

        single_scene = _build_broadband_scene(
            bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64)),
            spacing=0.08,
        )
        single_scene.add_monitor(mw.FluxMonitor("port_flux", axis="z", position=position, frequencies=[frequency]))
        single_result = mw.Simulation.fdtd(
            single_scene,
            frequencies=[frequency],
            run_time=runtime,
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        np.testing.assert_allclose(
            _to_cpu_numpy(multi_result.monitor("port_flux", frequency=frequency)["flux"]),
            _to_cpu_numpy(single_result.monitor("port_flux")["flux"]),
            rtol=1e-2,
            atol=1e-3,
        )
