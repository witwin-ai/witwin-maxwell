"""Generalized Bloch + CPML face layouts.

Beyond the single-PML-axis / two-Bloch-axes split, a Bloch axis can be combined
with two PML axes or with a PEC axis and a PML axis. The Bloch field from a real
source is checked against real periodic supercell runs:

* k = 0 must reproduce the periodic run;
* k = 2 pi / (3 L) must equal sum_m exp(j k m L) G3(x - m L), where G3 is the
  response of a 3L-periodic supercell to the single real source shifted by m L.

Measured relative errors are 2e-7 to 1e-6 (float32 round-off).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

# Equivalence against real periodic or supercell runs needs the same absorber on both
# sides: complex fixed-k runs default to CFS-CPML, so these runs pin alpha_max = 0.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_F = 1e9
_DX = 0.03125
_LX = 0.375
_FREQS = (0.8 * _F, _F, 1.2 * _F)


def _scene(x_kind, bloch_k, others, *, cells=1, dipoles=((0.0, 1.0),)):
    faces = dict(default="pml", num_layers=8, x=x_kind)
    if x_kind == "bloch":
        faces["bloch_wavevector"] = (bloch_k, 0.0, 0.0)
    faces.update(others)
    scene = mw.Scene(
        domain=mw.Domain(((-_LX / 2, _LX / 2 + (cells - 1) * _LX), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.faces(**faces),
        device="cuda",
    )
    for x0, amplitude in dipoles:
        scene.add_source(
            mw.PointDipole(
                position=(x0, 0.03125, 0.0),
                polarization="Ez",
                width=_DX,
                source_time=mw.GaussianPulse(frequency=_F, fwidth=0.5 * _F, amplitude=amplitude),
            )
        )
    scene.add_monitor(mw.PointMonitor("p", (0.09375, 0.15625, 0.0625), fields=("Ez",)))
    scene.add_monitor(mw.PointMonitor("q", (-0.125, -0.1875, -0.03125), fields=("Ez",)))
    return scene


def _spectra(scene):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=list(_FREQS),
        run_time=mw.TimeConfig(time_steps=700),
        spectral_sampler=mw.SpectralSampler(window="none"),
        # CPML physics comparison: pin CPML (absorber="auto" picks the graded
        # absorber for real closed-transverse PML).
        full_field_dft=False, absorber="cpml", cpml_config={"alpha_max": 0.0}).run()
    values = [
        complex(np.asarray(result.monitor(name, frequency=frequency)["Ez"].detach().cpu()).ravel()[0])
        for name in ("p", "q")
        for frequency in _FREQS
    ]
    return np.asarray(values), result.solver


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


_LAYOUTS = {
    "bloch_x_pml_yz": {},
    "bloch_x_pec_y_pml_z": {"y": "pec"},
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_generalized_bloch_layout_matches_supercell_superposition(layout):
    others = _LAYOUTS[layout]
    wavenumber = 2.0 * math.pi / (3.0 * _LX)
    bloch, solver = _spectra(_scene("bloch", wavenumber, others))
    assert solver.complex_fields_enabled and solver.uses_cpml
    total = 0.0
    for shift in range(3):
        part, _ = _spectra(_scene("periodic", 0.0, others, cells=3, dipoles=((shift * _LX, 1.0),)))
        total = total + np.exp(1j * wavenumber * shift * _LX) * part
    assert _relative(bloch, total) < 1e-4


def test_zero_phase_single_bloch_axis_matches_periodic_run():
    bloch, _ = _spectra(_scene("bloch", 0.0, {}))
    periodic, solver = _spectra(_scene("periodic", 0.0, {}))
    assert not solver.complex_fields_enabled
    assert _relative(bloch, periodic) < 1e-4


def test_generalized_bloch_layout_rejects_adjoint():
    density = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float64, requires_grad=True)
    scene = _scene("bloch", 1.0, {})
    scene.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
    scene.add_material_region(
        mw.MaterialRegion(name="design", geometry=mw.Box(position=(0, 0, 0), size=(_DX,) * 3),
                          density=density, eps_bounds=(1.0, 4.0))
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[_F], run_time=mw.TimeConfig(time_steps=20),
                                    spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False, cpml_config={"alpha_max": 0.0})
    with pytest.raises(NotImplementedError, match="generalized layouts"):
        result = simulation.run()
        result.monitor("p")["Ez"].abs().square().sum().backward()
