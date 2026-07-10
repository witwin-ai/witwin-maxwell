"""Combination: a time-modulated (traveling-index) slab absorbed by compressed
(slab-memory) CPML.

Removing the dense-memory force means a modulated scene may now allocate the
compact per-side psi layout instead of a full field-shaped psi tensor. The
compressed CPML update is the *same* arithmetic as the dense one, only reading
and writing psi from a compacted buffer, so the physical result must be
identical bit-for-bit. The load-bearing risk (per the P5.2 ledger) is the
narrow()/offset bookkeeping threaded into the new modulated compressed kernels:
an off-by-one in the slab-region start corrupts only the boundary cells and can
slip past a coarse energy check. We therefore assert exact parity against the
dense modulated run over a multi-step propagation that fills every PML face, and
independently confirm the slab path still produces the omega +/- Omega sidebands.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_modulated_slab_scene(*, carrier_frequency, modulation_frequency, amplitude=80.0):
    # PML on every face with a thickness (6) below half of each transverse axis
    # (16 cells), so the compact psi layout carries distinct low/high segments on
    # all three axes and the narrow()/offset bookkeeping is genuinely exercised.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=carrier_frequency, amplitude=amplitude),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.64, 0.64)),
                material=mw.Material(
                    eps_r=4.0,
                    modulation=mw.ModulationSpec(frequency=modulation_frequency, amplitude=0.25),
                ),
            )
        ],
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ez",), position=(0.42, 0.0, 0.0), interval=1)
    )
    return scene


def _run(scene, *, memory_mode, time_steps):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        absorber="cpml",
        cpml_config={"memory_mode": memory_mode},
        run_time=mw.TimeConfig(time_steps=time_steps),
        full_field_dft=False,
    ).run()
    trace = _to_numpy(result.monitor("probe")["field"]).astype(np.float64)
    memory = result.solver._cpml_memory_mode
    dt = float(result.solver.dt)
    del result
    torch.cuda.empty_cache()
    return trace, memory, dt


def _sideband_powers(trace, dt, carrier_frequency, modulation_frequency):
    """Windowed-FFT peak power at the carrier and the two first-order sidebands."""
    tail = trace[len(trace) // 4 :]
    window = np.hanning(tail.size)
    spectrum = np.fft.rfft(tail * window)
    freqs = np.fft.rfftfreq(tail.size, d=dt)
    power = np.abs(spectrum) ** 2

    def peak_power(target):
        index = int(np.argmin(np.abs(freqs - target)))
        low = max(index - 2, 0)
        high = min(index + 3, power.size)
        return float(power[low:high].max())

    return (
        peak_power(carrier_frequency),
        peak_power(carrier_frequency - modulation_frequency),
        peak_power(carrier_frequency + modulation_frequency),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_slab_cpml_bit_parity_with_dense():
    carrier_frequency = 1.0e9
    modulation_frequency = 2.5e8
    # Enough steps for the plane wave to cross the 1.2 m domain (~100 steps at the
    # Courant limit) and load psi on every PML face, so a slab-region off-by-one
    # would perturb the recorded trace.
    time_steps = 600

    scene = _build_modulated_slab_scene(
        carrier_frequency=carrier_frequency, modulation_frequency=modulation_frequency
    )
    dense_trace, dense_mode, dense_dt = _run(scene, memory_mode="dense", time_steps=time_steps)

    scene = _build_modulated_slab_scene(
        carrier_frequency=carrier_frequency, modulation_frequency=modulation_frequency
    )
    slab_trace, slab_mode, slab_dt = _run(scene, memory_mode="slab", time_steps=time_steps)

    assert dense_mode == "dense"
    assert slab_mode == "slab"
    assert dense_dt == slab_dt
    assert np.all(np.isfinite(slab_trace))
    assert float(np.abs(slab_trace).max()) > 0.0

    # The compressed layout stores the identical psi values in a compact buffer,
    # so the modulated update is bit-for-bit the dense result. Any nonzero
    # difference signals corrupted narrow()/offset bookkeeping.
    max_abs_diff = float(np.max(np.abs(slab_trace - dense_trace)))
    assert max_abs_diff == 0.0, f"slab vs dense modulated CPML mismatch: {max_abs_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_slab_cpml_generates_sidebands():
    carrier_frequency = 1.0e9
    modulation_frequency = 2.5e8
    time_steps = 4096

    scene = _build_modulated_slab_scene(
        carrier_frequency=carrier_frequency, modulation_frequency=modulation_frequency
    )
    slab_trace, slab_mode, dt = _run(scene, memory_mode="slab", time_steps=time_steps)
    assert slab_mode == "slab"

    carrier, lower, upper = _sideband_powers(
        slab_trace, dt, carrier_frequency, modulation_frequency
    )
    assert carrier > 0.0
    # A traveling-index slab converts carrier power into the first-order
    # omega +/- Omega sidebands; the compressed-CPML modulated path must show them.
    assert lower > 1.0e-3 * carrier
    assert upper > 1.0e-3 * carrier
