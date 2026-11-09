"""A pulsed point dipole under fixed-k Bloch boundaries is exact off the carrier.

The fixed-k source guard admits point and current sources because the Bloch
boundary supplies the fixed-k phase at every frequency. One pulsed run must
therefore match separate CW runs at off-carrier frequencies.

The pulse spectrum must stay clear of the fixed-k cutoff f_c = c*k_parallel/(2*pi)
(0.4 f0 here). Content below f_c is evanescent along z and cannot leave a lossless
cell, and content just above it propagates slowly. Either keeps the run from
decaying, which a finite-record DFT then smears. The decay rule rejects such runs.
"""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

F0 = 1.0e9
C0 = 299792458.0
OFF_CARRIER = (0.85 * F0, 1.15 * F0)


def _scene(source_time):
    k_parallel = 0.4 * 2 * math.pi * F0 / C0
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.03, 0.03), (-0.03, 0.03), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, x="bloch", y="bloch", z="pml",
                                       bloch_wavevector=(k_parallel, 0.0, 0.0)),
        device="cuda",
    )
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, -0.1), polarization="Ez", source_time=source_time))
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.15), fields=("Ex", "Ez")))
    return scene


def _phasor(result, frequency):
    payload = result.monitor("probe", frequency=frequency)
    return torch.stack([torch.as_tensor(payload[name]).reshape(-1)[0] for name in ("Ex", "Ez")]).to(torch.complex128)


def test_pulsed_point_dipole_matches_cw_off_carrier_under_fixed_k():
    pulsed = mw.Simulation.fdtd(
        # fwidth 0.12 f0 leaves exp(-12.5) of the spectrum at the 0.4 f0 cutoff.
        _scene(mw.GaussianPulse(frequency=F0, fwidth=0.12 * F0)), frequencies=OFF_CARRIER,
        run_time=mw.TimeConfig(time_steps=12000), shutoff=0.0,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True)).run()
    assert pulsed.solver.complex_fields_enabled
    for frequency in OFF_CARRIER:
        cw = mw.Simulation.fdtd(
            _scene(mw.CW(frequency=frequency)), frequencies=(frequency,),
            run_time=mw.TimeConfig.auto(transient_cycles=60, steady_cycles=20),
            spectral_sampler=mw.SpectralSampler(window="hanning", normalize_source=True)).run()
        expected = _phasor(cw, frequency)
        actual = _phasor(pulsed, frequency)
        assert expected.abs().max() > 0
        relative = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
        assert relative < 5e-4, (frequency, relative)
