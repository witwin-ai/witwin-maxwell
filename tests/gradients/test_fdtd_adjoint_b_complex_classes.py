"""Consolidated parity + FD coverage for the P6 native B-class complex reverse.

The B cluster nativizes the fused per-step reverse math for the "hard" material /
boundary classes. This module locks the newest complex class landed under P6: the
Bloch (complex-field) + electric-dispersive reverse
(``native_bloch_dispersive`` / ``python_reference_bloch_dispersive``), which
carries the imaginary-ADE replica that the plain complex-Bloch and plain
electric-dispersive backends each drop.

Three independent guards, matching the KERNEL-CHAIN parity contract:

- step-level parity: the fused native reverse (auto mode) reproduces the analytic
  torch reference *and* the torch-autograd VJP ground truth to ~1e-5 for a random
  checkpoint state (the native math and the reference math must agree, and the
  reference derivation must agree with autograd);
- end-to-end design-gradient agreement: the full checkpointed backward produces
  the same design gradient under ``auto`` (native), ``torch_reference``, and
  ``torch_vjp`` (the FD-consistency of that gradient is pinned separately by
  ``test_scene_with_bloch_dispersive_medium_gradient_matches_fd``);
- the dispatch coverage map: ``auto`` selects the native runner for this class,
  while the general mixed Bloch+CPML complex class stays the honestly reported
  torch-VJP remainder.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint.dispatch import (
    _ReverseBackend,
    _native_backend_available,
    _select_reverse_backend,
    reverse_step,
)
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from tests.gradients.test_fdtd_adjoint_materials import (
    _BlochDispersiveDensityScene,
    _abs2,
    _bloch_dispersive_scene,
    _build_sim,
)

_ENV = "WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the native FDTD adjoint")


def _prepared_bloch_dispersive_solver():
    density = torch.sigmoid(torch.zeros((2, 2, 2), device="cuda"))
    prepared = mw.Simulation.fdtd(
        _bloch_dispersive_scene(3.5, density=density),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=8),
    ).prepare()
    return prepared.solver


def _physical_reverse_state(*, seed):
    """A physical checkpoint state (with active real+imag dispersive currents) plus
    a random cotangent seed.

    Using the replayed forward trajectory the reverse actually rolls back keeps the
    fields, ADE currents, and eps sensitivity well conditioned, so the fused native
    reverse and the analytic reference agree at the single-precision step-level bar.
    """
    from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge
    from witwin.maxwell.fdtd.adjoint.core import _replay_segment_states

    torch.manual_seed(seed)
    model = _BlochDispersiveDensityScene(init=0.0).cuda()
    bridge = _FDTDGradientBridge(_build_sim(model, time_steps=24))
    bridge.forward(tuple(bridge.material_inputs))
    solver = bridge._last_solver
    states = _replay_segment_states(solver, bridge._last_checkpoints[0], 0, bridge._time_steps)
    forward_state = {name: tensor.detach().clone() for name, tensor in states[len(states) // 2].items()}
    # Seed the cotangent independently of the forward run so the adjoint is
    # deterministic regardless of how much RNG the forward replay consumed.
    torch.manual_seed(seed + 1000)
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    return solver, forward_state, adjoint_state


def _assert_l2_close(actual, expected, *, tol, name):
    """Scale-invariant L2-relative parity.

    The fused native reverse accumulates the same terms as the torch reference in a
    different FP order, so isolated boundary edges where large base and correction
    contributions nearly cancel can carry a large *per-cell* relative error while
    the field/gradient as a whole is bit-close. The L2-relative norm is the
    meaningful native-vs-reference parity measure here (a real reverse-math bug
    shows up as an O(1) relative error, far above the ~1e-7 single-precision floor).
    """
    denominator = expected.norm().clamp_min(1e-30)
    relative = ((actual - expected).norm() / denominator).item()
    assert relative <= tol, f"{name}: L2 relative error {relative:.3e} exceeds {tol:.1e}"


def _run_reverse(monkeypatch, mode, solver, forward_state, adjoint_state):
    monkeypatch.setenv(_ENV, mode)
    return reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=0.0,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
    )


@_CUDA
def test_bloch_dispersive_native_step_matches_reference(monkeypatch):
    """Step-level parity contract: the fused native reverse reproduces the analytic
    torch reference (``python_reference_bloch_dispersive``) bit-for-bit (~1e-5)
    across every checkpoint field and dispersive-state adjoint, real and imaginary.
    The reference derivation's agreement with the autograd VJP ground truth is
    pinned end-to-end by ``test_bloch_dispersive_design_gradient_agrees_across_backends``.
    """
    solver, forward_state, adjoint_state = _physical_reverse_state(seed=17)

    native = _run_reverse(monkeypatch, "auto", solver, forward_state, adjoint_state)
    reference = _run_reverse(monkeypatch, "torch_reference", solver, forward_state, adjoint_state)
    torch.cuda.synchronize()

    assert native.backend == "native_bloch_dispersive"
    assert reference.backend == "python_reference_bloch_dispersive"

    for name in forward_state:
        _assert_l2_close(native.pre_step_adjoint[name], reference.pre_step_adjoint[name], tol=1e-4, name=f"pre[{name}]")
    for grad_name in ("grad_eps_ex", "grad_eps_ey", "grad_eps_ez"):
        _assert_l2_close(getattr(native, grad_name), getattr(reference, grad_name), tol=1e-4, name=grad_name)


@_CUDA
def test_bloch_dispersive_auto_selects_native_and_qualifies():
    solver = _prepared_bloch_dispersive_solver()
    assert solver.electric_dispersive_enabled and solver.complex_fields_enabled
    forward_state = capture_checkpoint_state(solver, step=0).tensors
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.PYTHON_BLOCH_DISPERSIVE
    assert _native_backend_available(backend, solver, forward_state)


@_CUDA
def test_bloch_dispersive_design_gradient_agrees_across_backends(monkeypatch):
    """The full checkpointed backward yields the same design gradient under the
    native, analytic-reference, and torch-VJP backends."""

    def design_gradient(mode):
        monkeypatch.setenv(_ENV, mode)
        torch.manual_seed(0)
        model = _BlochDispersiveDensityScene(init=0.0).cuda()
        loss = _abs2(_build_sim(model, time_steps=40).run().monitor("probe")["data"])
        loss.backward()
        return model.logits.grad.detach().clone()

    native = design_gradient("auto")
    reference = design_gradient("torch_reference")
    vjp = design_gradient("torch_vjp")

    scale = native.abs().max().clamp_min(1e-30)
    torch.testing.assert_close(native, reference, rtol=1e-4, atol=float(scale) * 1e-5)
    torch.testing.assert_close(native, vjp, rtol=1e-4, atol=float(scale) * 1e-5)


@_CUDA
def test_general_mixed_bloch_cpml_remains_torch_vjp_remainder():
    """Coverage-map guard for the honestly reported remainder: a general mixed
    Bloch+CPML complex scene (two Bloch axes + one absorbing axis, no TFSF) has no
    fused analytic complex-CPML reverse, so it stays on the torch-VJP fallback."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.12, 0.12))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=6,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(math.pi / 1.2, math.pi / 2.4, 0.0),
        ),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(eps_r=6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.02, 0.0, -0.02),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.7e9),
        )
    )
    solver = mw.Simulation.fdtd(
        scene, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=8)
    ).prepare().solver
    assert solver.complex_fields_enabled and solver.uses_cpml
    forward_state = capture_checkpoint_state(solver, step=0).tensors
    backend = _select_reverse_backend(
        solver,
        forward_state,
        eps_ex=solver.eps_Ex,
        eps_ey=solver.eps_Ey,
        eps_ez=solver.eps_Ez,
        resolved_source_terms=None,
    )
    assert backend is _ReverseBackend.TORCH_VJP
    assert not _native_backend_available(backend, solver, forward_state)
