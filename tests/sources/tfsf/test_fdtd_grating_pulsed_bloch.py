"""Broadband (pulsed) plane-wave injection under Bloch transverse boundaries.

P5.4 lifts the "Bloch-boundary source patches require CW phased source terms"
guard (excitation/temporal.py) and its adjoint-replay twin (adjoint/core.py) so a
``GaussianPulse`` grating slab injects its time-delayed surface currents into the
split real/imag Bloch field. The forward and adjoint paths both evaluate the same
per-cell delayed waveform and scatter it with the Bloch wrap phase.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.sources import evaluate_source_time
from witwin.maxwell.fdtd.excitation.temporal import (
    apply_compiled_source_terms,
    apply_generic_source_terms,
)
from witwin.maxwell.fdtd.adjoint.core import _apply_source_term_list


def _bloch_slab_boundary(*, wavevector, num_layers=4):
    return mw.BoundarySpec.faces(
        default="pml",
        num_layers=num_layers,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector=wavevector,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_pulsed_bloch_grating_slab_runs_and_is_finite():
    # Oblique broadband plane wave under Bloch transverse boundaries. The delayed
    # (pulsed) TFSF surface-current patches are scattered into the split real/imag
    # Bloch field with the boundary wrap phase, a path that previously raised
    # NotImplementedError for any non-CW source_time. The run must stay finite and
    # keep the grating slab provider (i.e. the pulse is not silently downgraded).
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=_bloch_slab_boundary(wavevector=(math.pi, 0.5 * math.pi, 0.0)),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="grating_pulse",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[0.85e9, 1.0e9, 1.15e9],
        run_time=mw.TimeConfig(time_steps=256),
        absorber="cpml",
    ).run()

    assert result.solver.tfsf_enabled is True
    assert result.solver.complex_fields_enabled is True
    assert result.solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"
    for frequency in (0.85e9, 1.0e9, 1.15e9):
        payload = result.monitor("center", frequency=frequency)
        assert torch.isfinite(torch.as_tensor(payload["Ex"]).abs()).all()
        assert torch.isfinite(torch.as_tensor(payload["Ez"]).abs()).all()


# --- Deterministic injection checks (no CUDA required) ------------------------
# A capture module records the kernel call so the exact per-cell signal and Bloch
# wrap phases injected for a delayed (pulsed) patch can be verified directly,
# independent of a full solve. A steady-state CW comparison is intentionally
# avoided here: an oblique Bloch grating driven by CW excites long-lived
# transverse Bloch modes that do not settle within a practical run, so the CW
# reference itself is unreliable -- exactly why broadband pulses are used for
# gratings and the reason this capability exists.


class _CaptureLaunch:
    def __init__(self, calls, name, kwargs):
        self._calls = calls
        self._name = name
        self._kwargs = kwargs

    def launchRaw(self, **launch_kwargs):
        self._calls.append((self._name, self._kwargs, launch_kwargs))


class _CaptureModule:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def kernel(**kwargs):
            return _CaptureLaunch(self.calls, name, kwargs)

        return kernel


def _bloch_capture_solver():
    fdtd_module = _CaptureModule()
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="periodic",
        y="bloch",
        z="pml",
        bloch_wavevector=(0.0, math.pi, 0.0),
    )
    return SimpleNamespace(
        scene=SimpleNamespace(boundary=boundary),
        fdtd_module=fdtd_module,
        Ex=object(),
        Ex_imag=object(),
        Ey=object(),
        Ey_imag=object(),
        Ez=object(),
        Ez_imag=object(),
        Hx=object(),
        Hx_imag=object(),
        Hy=object(),
        Hy_imag=object(),
        Hz=object(),
        Hz_imag=object(),
        boundary_phase_cos=(1.0, 0.25, 1.0),
        boundary_phase_sin=(0.0, 0.75, 0.0),
        _clamp_pec_boundaries=lambda: None,
    )


_PULSE_SOURCE_TIME = {
    "kind": "gaussian_pulse",
    "kind_code": 1,
    "frequency": 1.2e9,
    "fwidth": 0.4e9,
    "amplitude": 2.5,
    "phase": 0.3,
    "delay": 1.0e-9,
}


def _delayed_bloch_source_term():
    return {
        "field_name": "Ez",
        "offsets": (0, 0, 0),
        "patch": torch.tensor([[[1.5]], [[-0.5]]], dtype=torch.float32),
        "grid": (1, 1, 1),
        "phase_real": 1.0,
        "phase_imag": 0.0,
        "delay_patch": torch.tensor([[[2.0e-10]], [[5.0e-10]]], dtype=torch.float32),
        "activation_delay_patch": None,
        "cw_cos_patch": None,
        "cw_sin_patch": None,
        "source_index": None,
        "source_time": _PULSE_SOURCE_TIME,
        "omega": 2.0 * math.pi * _PULSE_SOURCE_TIME["frequency"],
    }


@pytest.mark.parametrize(
    "dispatcher",
    (apply_generic_source_terms, apply_compiled_source_terms),
    ids=("generic", "compiled"),
)
def test_delayed_bloch_source_injects_per_cell_pulse_via_bloch_kernel(dispatcher):
    solver = _bloch_capture_solver()
    term = _delayed_bloch_source_term()
    time_value = 2.5e-9

    dispatcher(
        solver,
        [term],
        source_time=_PULSE_SOURCE_TIME,
        omega=term["omega"],
        time_value=time_value,
    )

    assert len(solver.fdtd_module.calls) == 1
    kernel_name, kwargs, _ = solver.fdtd_module.calls[0]
    # The real current is scattered into the split real/imag Bloch field, so the
    # signal is carried entirely in the per-cell patch (signalReal=1, signalImag=0)
    # rather than a scalar amplitude, and the boundary wrap phases match the CW path.
    assert kernel_name == "addSourcePatchBloch3D"
    assert kwargs["signalReal"] == pytest.approx(1.0)
    assert kwargs["signalImag"] == pytest.approx(0.0)
    assert kwargs["wrapAxisA"] == 1
    assert kwargs["wrapAxisB"] == 1
    assert kwargs["phaseCosA"] == pytest.approx(1.0)
    assert kwargs["phaseSinA"] == pytest.approx(0.0)
    assert kwargs["phaseCosB"] == pytest.approx(0.25)
    assert kwargs["phaseSinB"] == pytest.approx(0.75)

    # The injected patch is the scalar reference waveform delayed per cell, matching
    # the non-Bloch time-shifted kernel branch-for-branch.
    delay_patch = term["delay_patch"]
    patch = term["patch"]
    expected = torch.empty_like(patch)
    for i in range(patch.shape[0]):
        sampled = evaluate_source_time(_PULSE_SOURCE_TIME, time_value - float(delay_patch[i, 0, 0]))
        expected[i, 0, 0] = sampled * float(patch[i, 0, 0])
    torch.testing.assert_close(kwargs["sourcePatch"], expected, rtol=1e-5, atol=1e-6)


def _complex_field_mapping(shape):
    mapping = {}
    for name in ("Ex", "Ey", "Ez"):
        mapping[name] = torch.zeros(shape, dtype=torch.float32)
        mapping[f"{name}_imag"] = torch.zeros(shape, dtype=torch.float32)
    return mapping


def test_adjoint_replay_reconstructs_delayed_bloch_injection():
    # The reverse-replay reconstruction (adjoint/core._apply_source_term_list) must
    # rebuild the same delayed-pulse Bloch injection the forward solver applied,
    # rather than raising. An interior patch (no wrap) makes the expected value an
    # exact real-only injection.
    solver = _bloch_capture_solver()
    solver.complex_fields_enabled = True
    solver.boundary_kind = "bloch"

    shape = (5, 5, 5)
    field_mapping = _complex_field_mapping(shape)
    term = _delayed_bloch_source_term()
    term["offsets"] = (2, 2, 2)  # interior: no Bloch wrap contributions
    time_value = 2.5e-9

    updated = _apply_source_term_list(
        field_mapping,
        terms=[term],
        source_time=_PULSE_SOURCE_TIME,
        omega=term["omega"],
        time_value=time_value,
        solver=solver,
    )

    patch = term["patch"]
    delay_patch = term["delay_patch"]
    for i in range(patch.shape[0]):
        expected = evaluate_source_time(
            _PULSE_SOURCE_TIME, time_value - float(delay_patch[i, 0, 0])
        ) * float(patch[i, 0, 0])
        # rel/abs absorb the float32 waveform evaluation vs the float64 reference.
        assert updated["Ez"][2 + i, 2, 2].item() == pytest.approx(expected, rel=1e-4, abs=1e-6)
        # A real current adds nothing to the imaginary field for an interior patch.
        assert updated["Ez_imag"][2 + i, 2, 2].item() == pytest.approx(0.0, abs=1e-7)
