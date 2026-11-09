"""Independent complex-carrier and Floquet phase checks for CW injection."""
import math

import torch

from witwin.maxwell.fdtd.cuda import backend
from witwin.maxwell.fdtd.adjoint import core as adjoint_core


class _CaptureExtension:
    def __init__(self):
        self.calls = []

    def add_source_patch_bloch(self, *arguments):
        self.calls.append(arguments)


def _carrier(monkeypatch, phase, temporal_phase):
    extension = _CaptureExtension()
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", extension)
    field = torch.zeros_like(phase).reshape(1, 1, -1)
    backend._add_cw_phased_source_patch_bloch(
        ExReal=field, ExImag=field, EyReal=field, EyImag=field, EzReal=field, EzImag=field,
        sourcePatchCos=torch.cos(phase).reshape_as(field), sourcePatchSin=torch.sin(phase).reshape_as(field),
        offsetI=0, offsetJ=0, offsetK=0, signalCos=math.cos(temporal_phase), signalSin=math.sin(temporal_phase),
        axisCode=2, phaseCosA=1., phaseSinA=0., phaseCosB=1., phaseSinB=0., wrapAxisA=1, wrapAxisB=1)
    assert len(extension.calls) == 2
    return torch.complex(extension.calls[0][6], extension.calls[1][6]).reshape_as(phase)


def test_bloch_cw_carrier_uses_negative_temporal_and_positive_spatial_phase(monkeypatch):
    phase = torch.tensor([-.7, 0., .3, 1.2], dtype=torch.float64)
    for temporal_phase in (0., .37, math.pi / 2, 1.7):
        actual = _carrier(monkeypatch, phase, temporal_phase)
        expected = torch.exp(1j * (phase - temporal_phase))
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_bloch_cw_translation_matches_boundary_phase_and_positive_frequency_dft(monkeypatch):
    bloch_phase = .61
    phase = torch.tensor([.37, .37 + bloch_phase], dtype=torch.float64)
    samples = []
    for step in range(4):
        temporal_phase = step * math.pi / 2
        carrier = _carrier(monkeypatch, phase, temporal_phase)
        torch.testing.assert_close(carrier[1], carrier[0] * torch.exp(torch.tensor(1j * bloch_phase, dtype=torch.complex128)), rtol=1e-14, atol=1e-14)
        samples.append(carrier * complex(math.cos(temporal_phase), math.sin(temporal_phase)))
    # The solver's positive-frequency projection must retain the carrier. The
    # opposite imaginary quadrature cancels this sum instead of measuring it.
    measured = torch.stack(samples).mean(0)
    torch.testing.assert_close(measured, torch.exp(1j * phase), rtol=1e-14, atol=1e-14)


def test_functional_bloch_cw_replay_matches_native_complex_carrier(monkeypatch):
    phase = torch.tensor([-.9, .2, 1.4], dtype=torch.float64)
    temporal_phase = .73
    captured = {}

    def capture(mapping, **kwargs):
        captured.update(kwargs)
        return mapping

    monkeypatch.setattr(adjoint_core, "_add_complex_bloch_source_patch", capture)
    mapping = {"Ez": torch.zeros_like(phase)}
    assert adjoint_core._apply_bloch_cw_source_term(
        mapping, solver=object(), field_name="Ez", offsets=(0, 0, 0),
        cw_cos_patch=phase.cos(), cw_sin_patch=phase.sin(),
        signal_cos=math.cos(temporal_phase), signal_sin=math.sin(temporal_phase)) is mapping
    expected = torch.exp(1j * (phase - temporal_phase))
    torch.testing.assert_close(captured["delta"], expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(captured["delta"], _carrier(monkeypatch, phase, temporal_phase),
                               rtol=1e-14, atol=1e-14)
