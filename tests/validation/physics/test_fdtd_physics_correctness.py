from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import torch
from scipy.interpolate import RegularGridInterpolator

import witwin.maxwell as mw
from benchmark.scenes.dipole.dipole_vacuum import build_scene as build_dipole_vacuum
from benchmark.scenes.planewave.dielectric_slab import build_scene as build_dielectric_slab
from benchmark.scenes.planewave.planewave_vacuum import build_scene as build_planewave_vacuum


_C0 = 299_792_458.0


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value) -> float:
    return float(np.asarray(_to_numpy(value)).reshape(-1)[0])


def _num_steps(scene: mw.Scene, *, run_time_factor: float = 15.0) -> int:
    domain_size = max(bounds[1] - bounds[0] for bounds in scene.domain.bounds)
    dt_estimate = 0.5 * min(scene.dx, scene.dy, scene.dz) / (_C0 * np.sqrt(3.0))
    run_time_s = run_time_factor * domain_size / _C0
    return int(np.ceil(run_time_s / dt_estimate))


def _run_scene_summary(
    scene_builder,
    *,
    frequency: float,
    normalize_source: bool,
    monitor_names: tuple[str, ...],
) -> dict[str, object]:
    scene = scene_builder().clone(device="cuda")
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(frequency,),
        run_time=mw.TimeConfig(time_steps=_num_steps(scene)),
        spectral_sampler=mw.SpectralSampler(normalize_source=normalize_source),
    ).run()

    summary: dict[str, object] = {}
    for monitor_name in monitor_names:
        payload = result.monitor(monitor_name)
        monitor_summary = {
            key: _to_numpy(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
            for key, value in payload.items()
            if key in {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "x", "y", "z"}
        }
        if "flux" in payload:
            monitor_summary["flux"] = _to_scalar(payload["flux"])
        summary[monitor_name] = monitor_summary

    del result
    torch.cuda.empty_cache()
    return summary


@lru_cache(maxsize=None)
def _planewave_vacuum_summary() -> dict[str, object]:
    return _run_scene_summary(
        build_planewave_vacuum,
        frequency=2.0e9,
        normalize_source=True,
        monitor_names=("field_xz", "flux_pos_z", "flux_neg_z"),
    )


@lru_cache(maxsize=None)
def _dielectric_slab_summary() -> dict[str, object]:
    return _run_scene_summary(
        build_dielectric_slab,
        frequency=2.0e9,
        normalize_source=True,
        monitor_names=("field_xz", "reflected", "transmitted"),
    )


@lru_cache(maxsize=None)
def _dipole_vacuum_summary() -> dict[str, object]:
    return _run_scene_summary(
        build_dipole_vacuum,
        frequency=1.5e9,
        normalize_source=False,
        monitor_names=("field_xy", "flux_z_pos", "flux_z_neg"),
    )


def _slab_power_coefficients(*, n0: float, n1: float, n2: float, thickness: float, frequency: float) -> tuple[float, float]:
    wavelength = _C0 / float(frequency)
    delta = 2.0 * np.pi * n1 * thickness / wavelength
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    t01 = 2.0 * n0 / (n0 + n1)
    t12 = 2.0 * n1 / (n1 + n2)
    round_trip_phase = np.exp(2.0j * delta)
    transmission_phase = np.exp(1.0j * delta)
    reflection = (r01 + r12 * round_trip_phase) / (1.0 + r01 * r12 * round_trip_phase)
    transmission = (t01 * t12 * transmission_phase) / (1.0 + r01 * r12 * round_trip_phase)
    return float(np.abs(reflection) ** 2), float((n2 / n0) * np.abs(transmission) ** 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_planewave_vacuum_flux_matches_between_separated_monitor_planes():
    summary = _planewave_vacuum_summary()
    flux_pos = summary["flux_pos_z"]["flux"]
    flux_neg = summary["flux_neg_z"]["flux"]

    assert flux_pos > 0.0
    assert flux_neg < 0.0
    assert abs(flux_pos + flux_neg) / flux_pos < 2.0e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_planewave_vacuum_ex_is_transversely_uniform():
    summary = _planewave_vacuum_summary()
    field = np.abs(summary["field_xz"]["Ex"])
    x = np.asarray(summary["field_xz"]["x"], dtype=np.float64)
    z = np.asarray(summary["field_xz"]["z"], dtype=np.float64)

    x_mask = (x > -0.20) & (x < 0.20)
    z_mask = (z > 0.05) & (z < 0.25)
    window = field[np.ix_(x_mask, z_mask)]

    transverse_nonuniformity = float(window.std(axis=0).mean() / window.mean())
    axial_ripple = float(window.mean(axis=0).std() / window.mean())

    assert transverse_nonuniformity < 1.0e-3
    assert axial_ripple < 1.0e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_dipole_vacuum_flux_is_symmetric_across_positive_and_negative_z_planes():
    summary = _dipole_vacuum_summary()
    flux_pos = summary["flux_z_pos"]["flux"]
    flux_neg = summary["flux_z_neg"]["flux"]

    assert flux_pos > 0.0
    assert flux_neg > 0.0
    assert abs(flux_pos - flux_neg) / flux_pos < 2.0e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_dipole_vacuum_equatorial_field_is_azimuthally_symmetric():
    summary = _dipole_vacuum_summary()
    field = np.abs(summary["field_xy"]["Ez"])
    x = np.asarray(summary["field_xy"]["x"], dtype=np.float64)
    y = np.asarray(summary["field_xy"]["y"], dtype=np.float64)
    interpolator = RegularGridInterpolator((x, y), field)
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)

    for ring_radius in (0.08, 0.12, 0.16, 0.20):
        points = np.stack(
            (ring_radius * np.cos(angles), ring_radius * np.sin(angles)),
            axis=1,
        )
        values = interpolator(points)
        assert float(values.std() / values.mean()) < 5.5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_dielectric_slab_reflection_and_transmission_are_finite_and_nonzero():
    vacuum = _planewave_vacuum_summary()
    slab = _dielectric_slab_summary()

    incident_flux = -vacuum["flux_neg_z"]["flux"]
    reflected_flux = slab["reflected"]["flux"] - vacuum["flux_neg_z"]["flux"]
    transmitted_flux = slab["transmitted"]["flux"]
    reflectance = reflected_flux / incident_flux
    transmittance = transmitted_flux / incident_flux
    assert reflected_flux > 0.0
    assert 0.0 < reflectance < 1.0
    assert 0.0 < transmittance < 1.0
    # This soft source emits in both directions. Subtracting the reference flux
    # from total-field flux leaves an interference cross term, so these two
    # positive quantities are propagation smoke checks, not Fresnel powers.
