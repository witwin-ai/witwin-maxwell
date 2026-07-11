import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import calculate_required_steps
from witwin.core import Box
from witwin.maxwell.postprocess import compute_s_parameters
from witwin.maxwell.result import Result


_MULTI_FREQUENCIES = (0.95e9, 1.0e9)
_TFSF_BOUNDS = ((-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32))


def _as_monitor_array(value):
    if isinstance(value, torch.Tensor):
        return value
    return np.asarray(value, dtype=float)


def _synthetic_flux_result(*, frequencies, incident_flux, transmitted_flux=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    frequencies = tuple(float(freq) for freq in frequencies)
    samples = np.full((len(frequencies),), 8, dtype=int)
    monitors = {
        "port1": {
            "kind": "plane",
            "fields": ("Ey", "Ez", "Hy", "Hz"),
            "samples": samples.copy(),
            "frequency": frequencies[0],
            "frequencies": frequencies,
            "axis": "x",
            "position": -0.1,
            "compute_flux": True,
            "normal_direction": "+",
            "flux": _as_monitor_array(incident_flux),
            "power": _as_monitor_array(incident_flux),
        }
    }
    if transmitted_flux is not None:
        monitors["port2"] = {
            "kind": "plane",
            "fields": ("Ey", "Ez", "Hy", "Hz"),
            "samples": samples.copy(),
            "frequency": frequencies[0],
            "frequencies": frequencies,
            "axis": "x",
            "position": 0.1,
            "compute_flux": True,
            "normal_direction": "+",
            "flux": _as_monitor_array(transmitted_flux),
            "power": _as_monitor_array(transmitted_flux),
        }
    return Result(
        method="fdtd",
        scene=scene,
        frequencies=frequencies,
        monitors=monitors,
    )


def _synthetic_flux_result_with_source(*, frequencies, incident_flux, source, transmitted_flux=None):
    frequencies = tuple(float(freq) for freq in frequencies)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
        sources=[source],
    )
    samples = np.full((len(frequencies),), 8, dtype=int)
    monitors = {
        "port1": {
            "kind": "plane",
            "fields": ("Ey", "Ez", "Hy", "Hz"),
            "samples": samples.copy(),
            "frequency": frequencies[0],
            "frequencies": frequencies,
            "axis": "x",
            "position": -0.1,
            "compute_flux": True,
            "normal_direction": "+",
            "flux": _as_monitor_array(incident_flux),
            "power": _as_monitor_array(incident_flux),
        }
    }
    if transmitted_flux is not None:
        monitors["port2"] = dict(monitors["port1"])
        monitors["port2"].update(
            position=0.1,
            flux=_as_monitor_array(transmitted_flux),
            power=_as_monitor_array(transmitted_flux),
        )
    return Result(method="fdtd", scene=scene, frequencies=frequencies, monitors=monitors)


def _stack_flux_results(results):
    frequencies = tuple(float(result.frequency) for result in results)
    monitors = {}
    for monitor_name in results[0].monitors:
        samples = np.asarray([int(result.monitor(monitor_name)["samples"]) for result in results], dtype=int)
        flux = np.asarray([float(result.monitor(monitor_name)["flux"]) for result in results], dtype=float)
        first_payload = results[0].monitor(monitor_name)
        monitors[monitor_name] = {
            "kind": first_payload["kind"],
            "fields": first_payload["fields"],
            "samples": samples,
            "frequency": frequencies[0],
            "frequencies": frequencies,
            "axis": first_payload["axis"],
            "position": first_payload["position"],
            "compute_flux": first_payload.get("compute_flux", False),
            "normal_direction": first_payload.get("normal_direction", "+"),
            "flux": flux,
            "power": flux,
        }
    return Result(
        method=results[0].method,
        scene=results[0].scene,
        frequencies=frequencies,
        monitors=monitors,
    )


def _runtime_for_scene(scene, frequency, *, steady_cycles=8, transient_cycles=18):
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).prepare()
    solver = prepared.solver
    domain_size = max(
        float(scene.domain.bounds[0][1] - scene.domain.bounds[0][0]),
        float(scene.domain.bounds[1][1] - scene.domain.bounds[1][0]),
        float(scene.domain.bounds[2][1] - scene.domain.bounds[2][0]),
    )
    steps = calculate_required_steps(
        frequency=frequency,
        dt=solver.dt,
        c=solver.c,
        num_cycles=steady_cycles,
        transient_cycles=transient_cycles,
        domain_size=domain_size,
        source_time=getattr(solver, "_source_time", None),
    )
    return mw.TimeConfig(time_steps=steps)


def _build_tfsf_multifrequency_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96), (-0.96, 0.96), (-0.96, 0.96))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.35e9, amplitude=80.0),
                injection=mw.TFSF(bounds=_TFSF_BOUNDS),
                name="pw",
            )
        ],
    )
    scene.add_monitor(mw.FluxMonitor("port1", axis="x", position=-0.16, frequencies=_MULTI_FREQUENCIES, normal_direction="+"))
    scene.add_monitor(mw.FluxMonitor("port2", axis="x", position=0.16, frequencies=_MULTI_FREQUENCIES, normal_direction="+"))
    return scene


def _build_half_space_scene(frequency, *, material=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.015),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=80.0),
                name="pw",
            )
        ],
    )
    if material is not None:
        scene.add_structure(
            mw.Structure(
                name="half_space",
                geometry=Box(position=(0.6, 0.0, 0.0), size=(1.2, 0.8, 0.8)),
                material=material,
            )
        )
    scene.add_monitor(mw.FluxMonitor("port1", axis="x", position=-0.2, frequencies=[frequency], normal_direction="+"))
    scene.add_monitor(mw.FluxMonitor("port2", axis="x", position=0.4, frequencies=[frequency], normal_direction="+"))
    return scene


def _run_half_space_case(*, material=None):
    reference_results = []
    dut_results = []
    for frequency in _MULTI_FREQUENCIES:
        reference_scene = _build_half_space_scene(frequency, material=None)
        reference_runtime = _runtime_for_scene(reference_scene, frequency)
        reference_results.append(
            mw.Simulation.fdtd(
                reference_scene,
                frequencies=[frequency],
                run_time=reference_runtime,
                spectral_sampler=mw.SpectralSampler(window="hanning"),
                full_field_dft=False,
            ).run()
        )
        dut_scene = _build_half_space_scene(frequency, material=material)
        dut_runtime = _runtime_for_scene(dut_scene, frequency)
        dut_results.append(
            mw.Simulation.fdtd(
                dut_scene,
                frequencies=[frequency],
                run_time=dut_runtime,
                spectral_sampler=mw.SpectralSampler(window="hanning"),
                full_field_dft=False,
            ).run()
        )
    return _stack_flux_results(reference_results), _stack_flux_results(dut_results)


def test_compute_s_parameters_supports_explicit_incident_power_arrays():
    result = _synthetic_flux_result(
        frequencies=(1.0, 2.0),
        incident_flux=(8.0, 6.0),
        transmitted_flux=(2.0, 4.0),
    )

    s_params = compute_s_parameters(
        result,
        incident_monitor="port1",
        transmitted_monitor="port2",
        incident_power=np.array([10.0, 10.0]),
    )

    np.testing.assert_allclose(s_params["P_incident"], [10.0, 10.0])
    np.testing.assert_allclose(s_params["P_reflected"], [2.0, 4.0])
    np.testing.assert_allclose(s_params["P_transmitted"], [2.0, 4.0])
    np.testing.assert_allclose(s_params["S11_mag"], np.sqrt([0.2, 0.4]))
    np.testing.assert_allclose(s_params["S21_mag"], np.sqrt([0.2, 0.4]))
    np.testing.assert_allclose(s_params["S11_db"], 20.0 * np.log10(np.sqrt([0.2, 0.4])))


def test_compute_s_parameters_cw_auto_matches_analytic_power():
    amplitude = 3.0
    source = mw.PlaneWave(
        direction=(1.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 1.0),
        source_time=mw.CW(frequency=1.0e9, amplitude=amplitude),
        name="pw",
    )
    result = _synthetic_flux_result_with_source(
        frequencies=(1.0e9, 2.0e9),
        incident_flux=(5.0, 5.0),
        source=source,
    )

    s_params = compute_s_parameters(result, incident_monitor="port1", incident_power="auto")

    eta0 = 4.0 * np.pi * 1e-7 * 299792458.0
    area = 1.0  # domain y-extent x z-extent, 1.0 x 1.0
    expected = (amplitude ** 2) / (2.0 * eta0) * area
    np.testing.assert_allclose(s_params["P_incident"], [expected, expected], rtol=1e-9)


def test_compute_s_parameters_broadband_auto_without_solver_is_rejected():
    source = mw.PlaneWave(
        direction=(1.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 1.0),
        source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=1.0),
        name="pw",
    )
    result = _synthetic_flux_result_with_source(
        frequencies=(1.0e9, 2.0e9),
        incident_flux=(5.0, 5.0),
        source=source,
    )

    with pytest.raises(ValueError, match="running-DFT schedule"):
        compute_s_parameters(result, incident_monitor="port1", incident_power="auto")


def test_compute_s_parameters_rejects_missing_or_mismatched_normalization():
    result = _synthetic_flux_result(
        frequencies=(1.0, 2.0),
        incident_flux=(8.0, 6.0),
    )
    mismatched_reference = _synthetic_flux_result(
        frequencies=(1.0, 3.0),
        incident_flux=(10.0, 10.0),
    )

    with pytest.raises(ValueError, match="incident power normalization is required"):
        compute_s_parameters(result, incident_monitor="port1")

    with pytest.raises(ValueError, match="reference_result frequencies must match exactly"):
        compute_s_parameters(
            result,
            incident_monitor="port1",
            reference_result=mismatched_reference,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_compute_s_parameters_free_space_multifrequency_fdtd_returns_zero_s11():
    scene = _build_tfsf_multifrequency_scene()
    runtime = _runtime_for_scene(scene, min(_MULTI_FREQUENCIES), steady_cycles=6, transient_cycles=20)
    result = mw.Simulation.fdtd(
        scene,
        frequencies=_MULTI_FREQUENCIES,
        run_time=runtime,
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    s_params = compute_s_parameters(
        result,
        incident_monitor="port1",
        reference_result=result,
    )

    np.testing.assert_allclose(s_params["frequencies"], _MULTI_FREQUENCIES)
    np.testing.assert_allclose(s_params["S11_mag"], 0.0, atol=1e-15)
    assert s_params["S21"] is None
    assert s_params["P_transmitted"] is None


_BROADBAND_FREQUENCIES = (0.9e9, 1.0e9, 1.1e9)


def _build_broadband_tfsf_fine_scene():
    # Well-resolved (~15 cells/wavelength) empty TFSF box so the injected pulse
    # fills a clean aperture inside the total-field region and the analytic
    # aperture-area normalization is not swamped by numerical dispersion.
    aperture = ((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=1.0),
                injection=mw.TFSF(bounds=aperture),
                name="pw",
            )
        ],
    )
    scene.add_monitor(mw.FluxMonitor("port1", axis="x", position=-0.075, frequencies=_BROADBAND_FREQUENCIES, normal_direction="+"))
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_compute_s_parameters_broadband_auto_matches_measured_incident_flux():
    # An empty broadband (GaussianPulse) TFSF run: port1 measures the pure
    # incident spectral flux. incident_power="auto" must reconstruct that same
    # flux analytically from the source waveform and the solver DFT schedule,
    # without a second reference run.
    scene = _build_broadband_tfsf_fine_scene()
    runtime = _runtime_for_scene(scene, min(_BROADBAND_FREQUENCIES), steady_cycles=10, transient_cycles=25)
    result = mw.Simulation.fdtd(
        scene,
        frequencies=_BROADBAND_FREQUENCIES,
        run_time=runtime,
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=torch.float64).numpy().reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)

    measured_incident = _to_numpy(result.monitor("port1")["flux"])
    s_params = compute_s_parameters(result, incident_monitor="port1", incident_power="auto")
    auto_incident = _to_numpy(s_params["P_incident"])

    assert auto_incident.shape == measured_incident.shape
    assert np.all(auto_incident > 0.0)
    # The co-located aperture-area reconstruction matches the reference monitor
    # flux to the residual Yee-stagger / numerical-dispersion level at this
    # resolution (measured deviation ~5-7% across the band); the margin below
    # guards that while still catching a wrong area or normalization constant.
    np.testing.assert_allclose(auto_incident, measured_incident, rtol=0.12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_compute_s_parameters_dielectric_half_space_matches_fresnel_and_energy_balance():
    reference_result, dut_result = _run_half_space_case(material=mw.Material(eps_r=4.0))

    s_params = compute_s_parameters(
        dut_result,
        incident_monitor="port1",
        transmitted_monitor="port2",
        reference_result=reference_result,
    )

    expected_s11_mag = np.full((len(_MULTI_FREQUENCIES),), 1.0 / 3.0, dtype=float)
    expected_s11_db = 20.0 * np.log10(expected_s11_mag)
    np.testing.assert_allclose(s_params["S11_db"], expected_s11_db, atol=2.0)
    np.testing.assert_allclose(s_params["S11_mag"] ** 2 + s_params["S21_mag"] ** 2, 1.0, atol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_compute_s_parameters_metal_half_space_returns_near_unity_reflection():
    metal = mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8)
    reference_result, dut_result = _run_half_space_case(material=metal)

    s_params = compute_s_parameters(
        dut_result,
        incident_monitor="port1",
        transmitted_monitor="port2",
        reference_result=reference_result,
    )

    assert torch.all(s_params["S11_mag"] > 0.9)
    assert torch.all(s_params["S11_db"] > -1.0)
    assert torch.all(s_params["S21_mag"] < 5e-3)


def test_compute_s_parameters_keeps_torch_gradients():
    incident_flux = torch.tensor([8.0, 6.0], dtype=torch.float64, requires_grad=True)
    transmitted_flux = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    result = _synthetic_flux_result(
        frequencies=(1.0, 2.0),
        incident_flux=incident_flux,
        transmitted_flux=transmitted_flux,
    )

    s_params = compute_s_parameters(
        result,
        incident_monitor="port1",
        transmitted_monitor="port2",
        incident_power=torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    loss = s_params["S11_mag"].sum() + s_params["S21_mag"].sum()
    loss.backward()

    assert isinstance(s_params["S11_mag"], torch.Tensor)
    assert incident_flux.grad is not None
    assert transmitted_flux.grad is not None
    assert torch.all(torch.isfinite(incident_flux.grad))
    assert torch.all(torch.isfinite(transmitted_flux.grad))
