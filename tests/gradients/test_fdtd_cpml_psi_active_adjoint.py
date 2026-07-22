"""psi-active single-GPU CPML adjoint gradient regression.

Guards the Hy/Ey CPML psi axis convention on both sides of the adjoint:

* the forward replay (``adjoint/core._step_state`` and the distributed CPML
  forward mirror) must store the advanced psi under the canonical key
  (``psi_hy_z``/``psi_ey_z`` = the z-family, ``psi_hy_x``/``psi_ey_x`` = the
  x-family, per ``fdtd/boundary/cpml._CPML_MEMORY_SPECS``), and
* the native reverse (``adjoint/native``) must read the incoming post-step psi
  cotangent (``AdjPsiPosPost``/``AdjPsiNegPost``) from the SAME key its
  ``AdjPsiPosPrev``/``AdjPsiNegPrev`` write into, so the recursive psi carry
  connects step-to-step.

The pre-existing single-GPU CPML adjoint tests all sit in a psi-inert regime
(few steps, monitor in the interior, design region away from any PML) where the
psi memory is ~machine-zero and a swap on either side is undetectable. This test
deliberately drives the psi machinery hard: a long run with the probe placed
INSIDE the high x-PML band so the objective's sensitivity flows back through the
absorbing-layer psi recursion, where a Hy/Ey key swap corrupts the gradient at
the percent level.

Recorded falsifications (2026-07-19, RTX A6000, this scene):
* Reverting the native reverse ``AdjPsiPost`` keys to the swapped
  ``psi_hy_x``/``psi_ey_x`` (pos) form drives the analytic-vs-FD relative error
  from ~9e-6 to 3.4e-2 (h-converged), i.e. 17x above the 2e-3 gate.
* The forward-replay ``_step_state`` swap is inert for THIS scene (the design
  region lies in the non-PML interior where psi ~ 0), but corrupts the replayed
  psi state 4400x relative to the native kernel and would corrupt the gradient
  for any design region overlapping a PML, so it is fixed and pinned by the
  distributed replay-parity suite.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FREQ = 1.0e9
_STEPS = 400
_DENS_SHAPE = (5, 4, 4)
# Source in the interior, probe inside the high x-PML band (domain x in
# [-0.6, 0.6], 4-cell PML => high PML starts at x ~ 0.36; 0.48 is well inside it).
_SOURCE_X = -0.18
_MONITOR_X = 0.48


def _scene(density, *, device):
    x = np.linspace(-0.6, 0.6, 21, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(_SOURCE_X, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (_MONITOR_X, 0.0, 0.0), fields=("Ez",)))
    return scene


def _solve(density_values, *, want_grad, return_solver=False):
    density = density_values.clone().to("cuda:0")
    if want_grad:
        density.requires_grad_(True)
    scene = _scene(density, device="cuda:0")
    result = mw.Simulation.fdtd(
        scene,
        frequency=_FREQ,
        run_time=mw.TimeConfig(time_steps=_STEPS),
        absorber="cpml",
    ).run()
    spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    grad = None
    if want_grad:
        loss.backward()
        grad = density.grad.detach().cpu().clone()
    payload = (float(loss.detach().cpu()), grad)
    if return_solver:
        return payload + (result.solver,)
    return payload


def _base_density():
    torch.manual_seed(1)
    return (0.4 + 0.2 * torch.rand(_DENS_SHAPE)).double()


@_CUDA
def test_cpml_psi_active_adjoint_matches_central_fd():
    base = _base_density()
    loss, grad, solver = _solve(base, want_grad=True, return_solver=True)

    # (1) Non-vacuity: the run must be genuinely psi-active. The CPML memory of
    # the Hy/Ey components (the ones whose (pos, neg) = (z, x) axis order makes
    # the key order error-prone) must be a significant fraction of the field
    # scale, otherwise a swap on either side would be undetectable.
    field_scale = float(solver.Ez.abs().max())
    assert field_scale > 0.0
    for attr in ("psi_hy_z", "psi_hy_x", "psi_ey_z", "psi_ey_x"):
        psi = getattr(solver, attr)
        assert psi is not None
    assert float(solver.psi_hy_z.abs().max()) > 0.1 * field_scale, (
        "psi memory is inert; the psi-convention gate would be vacuous"
    )
    # The two families must be distinguishable, so a key swap actually changes
    # the numbers (equal families would make the swap a no-op).
    assert not torch.allclose(solver.psi_hy_z, solver.psi_hy_x)

    # (2) The objective must carry real signal at the in-PML probe.
    grad_scale = float(grad.abs().max())
    assert grad_scale > 1.0, f"objective too weak to be a meaningful gate: grad_scale={grad_scale:.3e}"

    # (3) Central finite-difference check of the analytic gradient on the most
    # sensitive design texel. With the correct psi convention on both replay and
    # reverse this matches to well under the FD floor; a Hy/Ey swap pushes it to
    # ~3.4e-2 (recorded falsification in the module docstring).
    texel = np.unravel_index(int(grad.abs().argmax()), grad.shape)
    analytic = float(grad[texel])
    best_rel = None
    for h in (5.0e-3, 1.0e-3):
        plus = base.clone()
        plus[texel] += h
        minus = base.clone()
        minus[texel] -= h
        loss_plus, _ = _solve(plus, want_grad=False)
        loss_minus, _ = _solve(minus, want_grad=False)
        fd = (loss_plus - loss_minus) / (2.0 * h)
        rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
        best_rel = rel if best_rel is None else min(best_rel, rel)
    assert best_rel <= 2.0e-3, (
        f"psi-active CPML adjoint grad {analytic:.6e} at texel {texel} disagrees with "
        f"central FD: best rel {best_rel:.3e} exceeds the 2e-3 gate (Hy/Ey psi "
        "convention regression)"
    )
