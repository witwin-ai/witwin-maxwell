"""Bloch + PEC faces against an independent real periodic supercell superposition.

A Bloch run with wavevector ``k`` along x and a real source in the unit cell is the
cell-0 restriction of the real periodic supercell (``M`` cells, ``M k L = 0 mod 2 pi``)
driven by the replicated sources ``exp(i k n L) * s``. By linearity the supercell
splits into two real runs weighted by ``cos(k n L)`` and ``sin(k n L)``, so

    E_bloch + i E_bloch_imag == E_cos + i E_sin   (on the unit cell)

holds to float32 roundoff at every time step. The y/z faces are PEC in both runs,
so the imaginary Bloch field must obey the same tangential clamp as the real one.
The real phases 0 and pi/L keep the imaginary field identically zero and act as
controls of the superposition itself.
"""
import math

import pytest
import torch

import witwin.maxwell as mw


PERIOD = 0.5
STEP = 0.03125  # exact binary fraction: 16 cells per period
TRANSVERSE = ((0.0, 0.25), (0.0, 0.25))
SOURCE_OFFSET = (0.185, 0.11, 0.135)
TIME_STEPS = 400
FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _final_fields(boundary, cells, weights):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, cells * PERIOD), *TRANSVERSE)),
        grid=mw.GridSpec.uniform(STEP),
        boundary=boundary,
        device="cuda",
    )
    for index, weight in enumerate(weights):
        if weight == 0.0:
            continue
        x, y, z = SOURCE_OFFSET
        scene.add_source(
            mw.PointDipole(
                position=(x + index * PERIOD, y, z),
                polarization="Ez",
                width=0.03,
                source_time=mw.GaussianPulse(1.0e9, fwidth=0.6e9, amplitude=float(weight)),
                name=f"src{index}",
            )
        )
    scene.add_monitor(mw.PointMonitor("probe", (0.4, 0.125, 0.125), fields=("Ez",)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=TIME_STEPS),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    solver = result.solver
    assert bool(solver.has_pec_faces)
    fields = {name: getattr(solver, name).detach().double() for name in FIELD_NAMES}
    if solver.complex_fields_enabled:
        fields.update({name + "_imag": getattr(solver, name + "_imag").detach().double() for name in FIELD_NAMES})
    return fields, solver


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize(("phase", "cells"), ((0.0, 1), (math.pi, 2), (2.0 * math.pi / 3.0, 3)))
def test_bloch_pec_matches_real_periodic_supercell_superposition(phase, cells, record_property):
    wavevector = phase / PERIOD
    bloch, solver = _final_fields(
        mw.BoundarySpec.faces(default="pec", x="bloch", bloch_wavevector=(wavevector, 0.0, 0.0)),
        1,
        (1.0,),
    )
    assert solver.complex_fields_enabled
    supercell = mw.BoundarySpec.faces(default="pec", x="periodic")
    cosine, _ = _final_fields(supercell, cells, tuple(math.cos(phase * n) for n in range(cells)))
    sine_weights = tuple(math.sin(phase * n) for n in range(cells))
    sine = _final_fields(supercell, cells, sine_weights)[0] if any(abs(w) > 1e-12 for w in sine_weights) else None

    error_sq = 0.0
    reference_sq = 0.0
    for name in FIELD_NAMES:
        actual = torch.complex(bloch[name], bloch[name + "_imag"])
        length = actual.shape[0]
        real_part = cosine[name][:length]
        imag_part = sine[name][:length] if sine is not None else torch.zeros_like(real_part)
        expected = torch.complex(real_part, imag_part)
        error_sq += float((actual - expected).abs().square().sum())
        reference_sq += float(expected.abs().square().sum())
    relative_error = math.sqrt(error_sq / reference_sq)
    record_property("relative_error", repr(relative_error))
    print(f"phase*L={phase:.6f} cells={cells} relative_error={relative_error:.3e}")
    assert reference_sq > 0.0
    assert relative_error < 2e-5

    # The tangential imaginary E on every PEC face is exactly zero.
    for name, axes in (("Ex", (1, 2)), ("Ey", (2,)), ("Ez", (1,))):
        imag = bloch[name + "_imag"]
        for axis in axes:
            assert float(imag.select(axis, 0).abs().max()) == 0.0
            assert float(imag.select(axis, imag.shape[axis] - 1).abs().max()) == 0.0
