"""CUDA-only FDTD adjoint acceptance tests.

The independent numerical checks live in the bridge, material, and rigorous
gradient suites.  This file locks the native-only dispatch contract itself.
"""

from __future__ import annotations

import pytest
import torch

from witwin.maxwell.fdtd.adjoint.capabilities import NATIVE_ADJOINT_CAPABILITIES
from witwin.maxwell.fdtd.adjoint.dispatch import (
    _NATIVE_REVERSE_LABELS,
    _NATIVE_REVERSE_RUNNERS,
    _ReverseBackend,
)


def test_native_capability_inventory_has_no_remainder_or_fallback_entries():
    assert NATIVE_ADJOINT_CAPABILITIES
    assert {"GENERAL_NONLINEAR", "DISPERSIVE", "MIXED_BLOCH_CPML", "GRATING_TFSF"} <= set(
        NATIVE_ADJOINT_CAPABILITIES
    )
    serialized = repr(NATIVE_ADJOINT_CAPABILITIES).lower()
    assert "remainder" not in serialized
    assert "fallback" not in serialized


def test_every_internal_variant_has_a_registered_native_runner():
    assert {variant.name for variant in _ReverseBackend} == set(NATIVE_ADJOINT_CAPABILITIES)
    assert set(_ReverseBackend) == set(_NATIVE_REVERSE_LABELS)
    assert set(_ReverseBackend) == set(_NATIVE_REVERSE_RUNNERS)
    assert all(label.startswith("native_") for label in _NATIVE_REVERSE_LABELS.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_missing_native_runner_fails_during_prepare(monkeypatch):
    from tests.gradients.test_fdtd_adjoint_bridge import _DensityPointScene, _build_simulation

    model = _DensityPointScene().cuda()
    monkeypatch.setattr(
        "witwin.maxwell.fdtd.adjoint.dispatch._NATIVE_REVERSE_RUNNERS", {}
    )
    with pytest.raises(RuntimeError, match="extension unavailable|native CUDA adjoint extension|variant"):
        _build_simulation(model, time_steps=4).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_missing_native_extension_fails_during_prepare(monkeypatch):
    from tests.gradients.test_fdtd_adjoint_bridge import _DensityPointScene, _build_simulation

    model = _DensityPointScene().cuda()
    monkeypatch.setattr(
        "witwin.maxwell.fdtd.cuda.backend.get_compiled_extension",
        lambda: (_ for _ in ()).throw(RuntimeError("extension unavailable")),
    )
    with pytest.raises(RuntimeError, match="extension unavailable|native CUDA adjoint extension|variant"):
        _build_simulation(model, time_steps=4).prepare()
