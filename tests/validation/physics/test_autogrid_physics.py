from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene

_C0 = 299_792_458.0

_FREQUENCY = 2.0e9
_WAVELENGTH = _C0 / _FREQUENCY
_SLAB_EPS = 4.0
_MSW = 10

# Fine-uniform reference step: the slab target lambda / (n * msw). The slab
# thickness is a whole number of reference cells so both grids resolve the
# slab faces exactly and the comparison isolates the graded background mesh.
_FINE_DL = _WAVELENGTH / (2.0 * _MSW)
_SLAB_THICKNESS = 12.0 * _FINE_DL
_HALF_SPAN = 24.0 * _FINE_DL
_Z_HALF_SPAN = 0.5
_PML_LAYERS = 12

_REFLECTED_PLANES = (-0.25, -0.25 + 0.25 * _WAVELENGTH)
_TRANSMITTED_PLANE = 0.25


def _domain():
    return mw.Domain(
        bounds=(
            (-_HALF_SPAN, _HALF_SPAN),
            (-_HALF_SPAN, _HALF_SPAN),
            (-_Z_HALF_SPAN, _Z_HALF_SPAN),
        )
    )


def _populate(scene, *, with_slab):
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


def _build_scene(grid, *, with_slab):
    scene = mw.Scene(
        domain=_domain(),
        grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=_PML_LAYERS),
        device="cuda",
    )
    return _populate(scene, with_slab=with_slab)


def _run_fluxes(scene):
    dt_estimate = 0.5 * _FINE_DL / (_C0 * np.sqrt(3.0))
    time_steps = int(np.ceil(12.0 * (2.0 * _Z_HALF_SPAN) / _C0 / dt_estimate))
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


def _coefficients(slab, vacuum):
    reflectance = float(
        np.mean(
            [
                1.0 - slab[f"upstream_{index}"] / vacuum[f"upstream_{index}"]
                for index in range(len(_REFLECTED_PLANES))
            ]
        )
    )
    transmittance = slab["transmitted"] / vacuum["transmitted"]
    return reflectance, transmittance


@lru_cache(maxsize=None)
def _auto_and_uniform_coefficients():
    # AutoGrid slab scene; the vacuum normalization run reuses the resolved
    # nodes through GridSpec.custom so both runs share the exact same mesh.
    auto_slab_scene = _build_scene(
        mw.GridSpec.auto(min_steps_per_wavelength=_MSW), with_slab=True
    )
    prepared = prepare_scene(auto_slab_scene)
    same_mesh = mw.GridSpec.custom(
        prepared.x_nodes64, prepared.y_nodes64, prepared.z_nodes64
    )
    auto = _coefficients(
        _run_fluxes(auto_slab_scene),
        _run_fluxes(_build_scene(same_mesh, with_slab=False)),
    )
    uniform = _coefficients(
        _run_fluxes(_build_scene(mw.GridSpec.uniform(_FINE_DL), with_slab=True)),
        _run_fluxes(_build_scene(mw.GridSpec.uniform(_FINE_DL), with_slab=False)),
    )
    return auto, uniform, prepared.N_total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD physics validation")
def test_autogrid_slab_transmission_matches_fine_uniform_reference():
    (r_auto, t_auto), (r_uniform, t_uniform), auto_cells = _auto_and_uniform_coefficients()

    for value in (r_auto, t_auto, r_uniform, t_uniform):
        assert 0.0 < value < 1.0

    # Energy balance on both grids.
    assert abs((r_auto + t_auto) - 1.0) < 2.0e-2
    assert abs((r_uniform + t_uniform) - 1.0) < 2.0e-2

    # AutoGrid agrees with the fine-uniform reference within a small tolerance.
    assert abs(t_auto - t_uniform) < 5.0e-2
    assert abs(r_auto - r_uniform) < 5.0e-2

    # The adaptive mesh is meaningfully smaller than the uniform-fine grid.
    uniform_cells = (
        int(np.ceil(2.0 * _HALF_SPAN / _FINE_DL)) ** 2
        * int(np.ceil(2.0 * _Z_HALF_SPAN / _FINE_DL))
    )
    assert auto_cells < 0.8 * uniform_cells
