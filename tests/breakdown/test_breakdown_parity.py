"""Zero-cost / parity gates for the breakdown machinery (decision 4).

* A scene with a breakdown descriptor whose field never reaches ``critical_field``
  must produce *bitwise-identical* fields to the same scene without the descriptor:
  intact capable edges are rewritten with their exact stored base coefficients, so
  the six-field state after N steps is reproduced bit for bit.
* A scene with no breakdown material takes the untouched code path and reports
  ``Result.breakdown is None``.

The parity comparison uses the raw last-step Yee fields on the solver (all six
components), which is the strictest observable.
"""

from __future__ import annotations

import pytest
import torch

from tests.breakdown._common import build_breakdown_scene, build_plain_scene, run_breakdown

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _last_step_fields(result):
    solver = result.solver
    return {name: getattr(solver, name).clone() for name in _FIELDS}


def test_below_threshold_descriptor_is_bitwise_identical():
    """Descriptor present but never triggered == no descriptor, bit for bit."""
    # critical_field = 1e12 V/m is unreachable for this weak CW source.
    with_descriptor = run_breakdown(
        build_breakdown_scene(
            critical_field=1.0e12,
            minimum_duration=0.0,
            source_amplitude=50.0,
        )
    )
    without = run_breakdown(
        build_plain_scene(source_amplitude=50.0)
    )

    # The descriptor scene still exposes a breakdown result (material is present),
    # but it never triggered.
    assert with_descriptor.breakdown is not None
    assert with_descriptor.breakdown.triggered_count == 0
    # A never-triggering descriptor deposits exactly zero breakdown energy: the
    # dissipation channel only accumulates on conducting edges. Assert exact 0.0
    # (not int-truncated, which would silently pass any value in [0, 1)).
    assert with_descriptor.breakdown.total_dissipated_energy == 0.0
    assert float(with_descriptor.breakdown.dissipated_energy.abs().sum().item()) == 0.0

    a = _last_step_fields(with_descriptor)
    b = _last_step_fields(without)
    for name in _FIELDS:
        assert torch.equal(a[name], b[name]), (
            f"{name} diverged: max|delta|={(a[name]-b[name]).abs().max().item():.3e}"
        )


def test_scene_without_breakdown_material_reports_none():
    """No breakdown descriptor anywhere -> Result.breakdown is None (zero cost)."""
    result = run_breakdown(build_plain_scene(source_amplitude=50.0))
    assert result.breakdown is None
    assert result.breakdown_events == ()


def test_plain_scene_never_compiles_breakdown_layout(monkeypatch):
    """Zero-impact-when-unused gate: a breakdown-free scene never invokes the
    layout compiler, so it allocates none of the seven full-grid breakdown
    tensors. Monkeypatch the compiler with a raising stub; prepare+run must still
    succeed because the cheap structure pre-scan short-circuits ahead of it."""
    import witwin.maxwell.fdtd.runtime.breakdown as breakdown_runtime

    def _forbidden(*_args, **_kwargs):
        raise AssertionError(
            "compile_breakdown_layout must not run for a breakdown-free scene"
        )

    monkeypatch.setattr(breakdown_runtime, "compile_breakdown_layout", _forbidden)

    result = run_breakdown(build_plain_scene(source_amplitude=50.0))
    assert result.breakdown is None
    assert result.breakdown_events == ()


def test_plain_scene_run_is_bitwise_deterministic():
    """Field-level no-breakdown parity: with the breakdown module imported and its
    prepare-time hook active, two identical plain-scene runs reproduce all six Yee
    fields bit for bit -- the (guarded) breakdown machinery perturbs nothing."""
    a = _last_step_fields(run_breakdown(build_plain_scene(source_amplitude=50.0)))
    b = _last_step_fields(run_breakdown(build_plain_scene(source_amplitude=50.0)))
    for name in _FIELDS:
        assert torch.equal(a[name], b[name]), (
            f"{name} diverged across identical plain-scene runs: "
            f"max|delta|={(a[name]-b[name]).abs().max().item():.3e}"
        )


def test_triggering_breaks_parity_control():
    """Control for the parity gate: a triggering descriptor DOES change the fields.

    This is the falsification companion -- if lowering the threshold left the
    fields bit-identical, the below-threshold parity test would be vacuous.
    """
    triggering = run_breakdown(
        build_breakdown_scene(
            critical_field=1.0e-3,  # reached almost everywhere the wave arrives
            minimum_duration=0.0,
            post_sigma=5.0,
            source_amplitude=50.0,
        )
    )
    without = run_breakdown(build_plain_scene(source_amplitude=50.0))

    assert triggering.breakdown is not None
    assert triggering.breakdown.triggered_count > 0

    a = _last_step_fields(triggering)
    b = _last_step_fields(without)
    diverged = any(not torch.equal(a[name], b[name]) for name in _FIELDS)
    assert diverged, "a triggering breakdown must perturb the fields"
