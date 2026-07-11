"""Unit coverage for the FDTD adjoint reverse-execution dispatch switch.

These tests exercise the P6 foundation plumbing:

- ``WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND`` override parsing
  (``auto`` / ``native`` / ``torch_reference`` / ``torch_vjp``),
- the mode-aware dispatch switch in :func:`reverse_step`,
- the native reverse-execution harness (runner registry), and
- the ``native_*`` profiler labels wired into ``reverse_backend_counts``.

Everything here runs on CPU. The native standard runner is registered, but its
qualifier only accepts a CUDA scene with the compiled extension, so on CPU
``auto`` mode must still reproduce the analytic torch reference behaviour exactly
and an explicit ``native`` override must still raise.
"""

from __future__ import annotations

import pytest
import torch

from witwin.maxwell.fdtd.adjoint import dispatch
from witwin.maxwell.fdtd.adjoint.dispatch import (
    _NATIVE_REVERSE_LABELS,
    _ReverseBackend,
    register_native_reverse_backend,
    resolve_fdtd_adjoint_backend_name,
    reverse_step,
)
from witwin.maxwell.fdtd.adjoint.profiler import _BackwardProfiler, _ReverseStepResult
from tests.gradients.test_fdtd_adjoint_bridge import _fake_standard_reverse_solver


_ENV = "WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"


def _standard_case():
    """A standard (open-boundary, non-CPML) reverse-step invocation on CPU.

    ``_select_reverse_backend`` routes this to ``PYTHON_STANDARD`` / the
    ``python_reference_standard`` analytic backend.
    """
    solver = _fake_standard_reverse_solver()
    torch.manual_seed(0)
    forward_state = {
        "Ex": torch.randn(2, 4, 5),
        "Ey": torch.randn(3, 3, 5),
        "Ez": torch.randn(3, 4, 4),
        "Hx": torch.randn(3, 3, 4),
        "Hy": torch.randn(2, 4, 4),
        "Hz": torch.randn(2, 3, 5),
    }
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    eps_ex = torch.full((2, 4, 5), 2.0, requires_grad=True)
    eps_ey = torch.full((3, 3, 5), 2.5, requires_grad=True)
    eps_ez = torch.full((3, 4, 4), 3.0, requires_grad=True)
    return solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez


def _run(solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez):
    return reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=0.0,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )


# ---------------------------------------------------------------------------
# resolve_fdtd_adjoint_backend_name
# ---------------------------------------------------------------------------


def test_valid_backend_vocabulary_is_exactly_the_four_modes():
    assert dispatch._VALID_ADJOINT_BACKENDS == {"auto", "native", "torch_reference", "torch_vjp"}


def test_resolve_defaults_to_auto(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_fdtd_adjoint_backend_name() == "auto"


@pytest.mark.parametrize("mode", ["auto", "native", "torch_reference", "torch_vjp"])
def test_resolve_reads_environment(monkeypatch, mode):
    monkeypatch.setenv(_ENV, mode)
    assert resolve_fdtd_adjoint_backend_name() == mode


def test_resolve_is_case_insensitive_and_stripped(monkeypatch):
    monkeypatch.setenv(_ENV, "  Torch_VJP \n")
    assert resolve_fdtd_adjoint_backend_name() == "torch_vjp"


def test_resolve_explicit_request_overrides_environment(monkeypatch):
    monkeypatch.setenv(_ENV, "auto")
    assert resolve_fdtd_adjoint_backend_name("torch_reference") == "torch_reference"


@pytest.mark.parametrize("bad", ["python", "cuda", "vjp", ""])
def test_resolve_rejects_unknown_backends(monkeypatch, bad):
    monkeypatch.setenv(_ENV, bad)
    with pytest.raises(ValueError, match="WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"):
        resolve_fdtd_adjoint_backend_name()


# ---------------------------------------------------------------------------
# dispatch switch: auto / torch_reference / torch_vjp
# ---------------------------------------------------------------------------


def test_auto_mode_routes_to_analytic_reference(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    result = _run(*_standard_case())
    assert result.backend == "python_reference_standard"


def test_force_torch_reference(monkeypatch):
    monkeypatch.setenv(_ENV, "torch_reference")
    result = _run(*_standard_case())
    assert result.backend == "python_reference_standard"


def test_force_torch_vjp(monkeypatch):
    monkeypatch.setenv(_ENV, "torch_vjp")
    result = _run(*_standard_case())
    assert result.backend == "torch_vjp"


def test_force_torch_vjp_matches_auto_reference_gradients(monkeypatch):
    # Forcing the VJP fallback on a config the analytic backend also supports
    # must produce the same eps gradients (the analytic backend is derived from
    # the same reverse recurrence), guarding the routing against silent drift.
    case = _standard_case()
    monkeypatch.setenv(_ENV, "torch_reference")
    reference = _run(*case)
    monkeypatch.setenv(_ENV, "torch_vjp")
    vjp = _run(*case)
    for name in ("grad_eps_ex", "grad_eps_ey", "grad_eps_ez"):
        torch.testing.assert_close(getattr(vjp, name), getattr(reference, name), rtol=1e-5, atol=1e-6)


def test_force_torch_reference_raises_when_only_vjp_supported(monkeypatch):
    monkeypatch.setenv(_ENV, "torch_reference")
    monkeypatch.setattr(
        dispatch, "_select_reverse_backend", lambda *a, **k: _ReverseBackend.TORCH_VJP
    )
    with pytest.raises(ValueError, match="torch_reference"):
        _run(*_standard_case())


# ---------------------------------------------------------------------------
# native reverse-execution harness
# ---------------------------------------------------------------------------


def test_native_labels_cover_every_analytic_reference_variant():
    analytic = set(_ReverseBackend) - {_ReverseBackend.TORCH_VJP}
    assert set(_NATIVE_REVERSE_LABELS) == analytic
    # Labels are unique and all carry the native_ prefix.
    labels = list(_NATIVE_REVERSE_LABELS.values())
    assert len(labels) == len(set(labels))
    assert all(label.startswith("native_") for label in labels)


def test_force_native_raises_when_runner_does_not_qualify(monkeypatch):
    # The standard native runner is registered, but its qualifier only accepts a
    # CUDA scene with the compiled extension. On this CPU standard case the
    # runner does not qualify, so an explicit ``native`` override must still raise.
    monkeypatch.setenv(_ENV, "native")
    solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez = _standard_case()
    assert not dispatch._native_backend_available(
        _ReverseBackend.PYTHON_STANDARD, solver, forward_state
    )
    with pytest.raises(ValueError, match="no .*native CUDA reverse backend is registered"):
        _run(solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez)


def _fake_native_runner(recorder, label):
    def runner(solver, forward_state, adjoint_state, *, time_value, eps_ex, eps_ey, eps_ez, resolved_source_terms, profiler):
        recorder["called"] = True
        recorder["time_value"] = time_value
        return _ReverseStepResult(
            pre_step_adjoint={name: tensor.clone() for name, tensor in adjoint_state.items()},
            grad_eps_ex=torch.zeros_like(eps_ex),
            grad_eps_ey=torch.zeros_like(eps_ey),
            grad_eps_ez=torch.zeros_like(eps_ez),
            backend=label,
        )

    return runner


def test_native_harness_dispatches_registered_runner(monkeypatch):
    label = _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_STANDARD]
    recorder: dict = {}
    monkeypatch.setitem(
        dispatch._NATIVE_REVERSE_RUNNERS,
        _ReverseBackend.PYTHON_STANDARD,
        _fake_native_runner(recorder, label),
    )
    monkeypatch.setenv(_ENV, "native")
    result = _run(*_standard_case())
    assert recorder.get("called") is True
    assert result.backend == label


def test_auto_prefers_native_when_registered(monkeypatch):
    label = _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_STANDARD]
    recorder: dict = {}
    monkeypatch.setitem(
        dispatch._NATIVE_REVERSE_RUNNERS,
        _ReverseBackend.PYTHON_STANDARD,
        _fake_native_runner(recorder, label),
    )
    monkeypatch.setenv(_ENV, "auto")
    result = _run(*_standard_case())
    assert result.backend == label


def test_torch_reference_ignores_registered_native_runner(monkeypatch):
    # An explicit torch_reference override must never fall through to native.
    monkeypatch.setitem(
        dispatch._NATIVE_REVERSE_RUNNERS,
        _ReverseBackend.PYTHON_STANDARD,
        _fake_native_runner({}, _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_STANDARD]),
    )
    monkeypatch.setenv(_ENV, "torch_reference")
    result = _run(*_standard_case())
    assert result.backend == "python_reference_standard"


def test_register_native_reverse_backend_rejects_torch_vjp():
    with pytest.raises(ValueError, match="No native reverse label"):
        register_native_reverse_backend(_ReverseBackend.TORCH_VJP, lambda *a, **k: None)


# ---------------------------------------------------------------------------
# native labels wired into the backward profiler counts
# ---------------------------------------------------------------------------


def test_profiler_counts_native_reverse_labels(monkeypatch):
    # Mirror the bridge's per-step recording: a native reverse step feeds its
    # native_* label straight into reverse_backend_counts.
    label = _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_STANDARD]
    recorder: dict = {}
    monkeypatch.setitem(
        dispatch._NATIVE_REVERSE_RUNNERS,
        _ReverseBackend.PYTHON_STANDARD,
        _fake_native_runner(recorder, label),
    )
    monkeypatch.setenv(_ENV, "native")

    profiler = _BackwardProfiler(enabled=True, device=None)
    for _ in range(3):
        step_result = _run(*_standard_case())
        profiler.record_reverse_backend(step_result.backend)

    summary = profiler.summary(steps=3, segments=1, checkpoint_stride=1)
    assert summary["reverse_backend_counts"] == {label: 3}
    assert summary["reverse_backend_counts"].get("torch_vjp", 0) == 0
