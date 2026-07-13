from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


_C0 = 299_792_458.0

_FREQUENCY = 2.0e9
_WAVELENGTH = _C0 / _FREQUENCY
_SLAB_EPS = 4.0
_SLAB_THICKNESS = 0.1

# Transverse axes: uniform (expressed as custom coords; GridSpec.custom takes
# all three axes). Normal (z) axis: fine inside the slab, geometric growth
# (ratio 1.2) outside. PreparedScene appends constant-step PML externally.
_DX = 0.02
_HALF_SPAN = 0.48
_PML_LAYERS = 12
_FINE_DZ = 0.005
_FINE_EXTENT = 0.08
_GRADING_RATIO = 1.2
_COARSE_DZ = 0.01
_PHYSICAL_EXTENT = 0.30

# The single-plane DFT flux of superposed incident + reflected waves carries a
# residual standing-wave term (Yee half-cell E/H staggering) oscillating as
# cos(2kz); averaging two planes a quarter wavelength apart cancels it.
_REFLECTED_PLANES = (-0.25, -0.25 + 0.25 * _WAVELENGTH)
_TRANSMITTED_PLANE = 0.25


def _graded_half_axis():
    """Physical half z-axis from 0: fine slab region then geometric growth."""
    nodes = [0.0]
    dz = _FINE_DZ
    while nodes[-1] < _FINE_EXTENT - 1e-12:
        nodes.append(nodes[-1] + dz)
    while nodes[-1] < _PHYSICAL_EXTENT - 1e-12:
        dz = min(dz * _GRADING_RATIO, _COARSE_DZ)
        nodes.append(nodes[-1] + dz)
    return np.asarray(nodes, dtype=np.float64)


def _z_nodes():
    half = _graded_half_axis()
    return np.concatenate([-half[:0:-1], half])


def _build_scene(*, with_slab):
    z_nodes = _z_nodes()
    count = int(round(2.0 * _HALF_SPAN / _DX))
    xy_nodes = -_HALF_SPAN + np.arange(count, dtype=np.float64) * _DX
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(xy_nodes[0]), float(xy_nodes[-1])),
                (float(xy_nodes[0]), float(xy_nodes[-1])),
                (float(z_nodes[0]), float(z_nodes[-1])),
            )
        ),
        grid=mw.GridSpec.custom(xy_nodes, xy_nodes, z_nodes),
        boundary=mw.BoundarySpec.pml(num_layers=_PML_LAYERS),
        device="cuda",
    )
    if with_slab:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.0),
                    size=(2.0 * _HALF_SPAN, 2.0 * _HALF_SPAN, _SLAB_THICKNESS),
                ),
                material=mw.Material(eps_r=_SLAB_EPS),
                name="slab",
            )
        )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.5e9),
        )
    )
    for index, position in enumerate(_REFLECTED_PLANES):
        scene.add_monitor(
            mw.FluxMonitor(
                name=f"upstream_{index}", axis="z", position=position, frequencies=(_FREQUENCY,)
            )
        )
    scene.add_monitor(
        mw.FluxMonitor(
            name="transmitted", axis="z", position=_TRANSMITTED_PLANE, frequencies=(_FREQUENCY,)
        )
    )
    return scene


def _run_fluxes(*, with_slab):
    scene = _build_scene(with_slab=with_slab)
    z_extent = float(_z_nodes()[-1] - _z_nodes()[0])
    dt_estimate = 0.5 * _FINE_DZ / (_C0 * np.sqrt(3.0))
    time_steps = int(np.ceil(12.0 * z_extent / _C0 / dt_estimate))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
    ).run()
    names = tuple(f"upstream_{index}" for index in range(len(_REFLECTED_PLANES)))
    names += ("transmitted",)
    fluxes = {
        name: float(np.asarray(result.monitor(name)["flux"].detach().cpu()).reshape(-1)[0])
        for name in names
    }
    del result
    torch.cuda.empty_cache()
    return fluxes


@lru_cache(maxsize=None)
def _slab_and_vacuum_fluxes():
    return _run_fluxes(with_slab=True), _run_fluxes(with_slab=False)


def _slab_power_coefficients():
    """Analytic normal-incidence Fresnel R/T for a single dielectric slab."""
    n1 = float(np.sqrt(_SLAB_EPS))
    delta = 2.0 * np.pi * n1 * _SLAB_THICKNESS / _WAVELENGTH
    r01 = (1.0 - n1) / (1.0 + n1)
    r12 = -r01
    t01 = 2.0 / (1.0 + n1)
    t12 = 2.0 * n1 / (1.0 + n1)
    round_trip = np.exp(2.0j * delta)
    reflection = (r01 + r12 * round_trip) / (1.0 + r01 * r12 * round_trip)
    transmission = (t01 * t12 * np.exp(1.0j * delta)) / (1.0 + r01 * r12 * round_trip)
    return float(np.abs(reflection) ** 2), float(np.abs(transmission) ** 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_graded_mesh_dielectric_slab_energy_balance_and_fresnel():
    # A vacuum run on the identical graded grid provides the reference fluxes:
    # the upstream net flux with the slab is incident - reflected, so
    # R = 1 - slab/vacuum plane-averaged over the quarter-wave plane pair, and
    # T = slab/vacuum at the matched downstream plane (standard with/without
    # sample normalization, which also cancels transverse spreading loss).
    slab, vacuum = _slab_and_vacuum_fluxes()

    for name, flux in vacuum.items():
        assert flux > 0.0, name

    reflectance = float(
        np.mean(
            [
                1.0 - slab[f"upstream_{index}"] / vacuum[f"upstream_{index}"]
                for index in range(len(_REFLECTED_PLANES))
            ]
        )
    )
    transmittance = slab["transmitted"] / vacuum["transmitted"]
    reflectance_exact, transmittance_exact = _slab_power_coefficients()

    assert 0.0 < reflectance < 1.0
    assert 0.0 < transmittance < 1.0
    # Energy balance across the graded mesh (independent upstream/downstream
    # measurements on either side of the refinement region).
    # With external PML the two upstream flux planes snap to physical Yee planes
    # instead of the old internal-PML indices; the residual two-plane standing-wave
    # cancellation is about 5%, while the independent Fresnel checks below remain
    # within their original 8% accuracy tier.
    assert abs((reflectance + transmittance) - 1.0) < 6.0e-2
    # Fresnel agreement is limited by the half-cell effective-thickness
    # ambiguity of the staircased slab faces (the Fabry-Perot response is
    # steep in thickness); tolerance matches the uniform-grid slab test tier.
    assert abs(reflectance - reflectance_exact) < 8.0e-2
    assert abs(transmittance - transmittance_exact) < 8.0e-2
