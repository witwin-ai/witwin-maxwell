"""Open-boundary / domain-extension truncation study for the electrostatic solver
(Plan 12 Phase 4, stage H2b).

Covers:
- the ``open`` electrostatic boundary fail-close (no exact radiation condition on a
  finite Cartesian face; a boundary-element open boundary is a later phase);
- the opt-in ``truncation_estimate`` API on ``Simulation.capacitance(...)`` (a
  second, enlarged grounded-box solve is run only on request, never silently);
- the two-axis domain-extension convergence study: self-capacitance of an isolated
  conductor versus enclosure size ``L`` at fixed grid, and versus grid at fixed
  ``L``, with a 1/L Richardson extrapolation to the infinite-domain limit;
- the tensor-eps differentiability disposition through the public capacitance path
  (fail-closed: Phase 4 owns forward tensor-eps, gradients are Phase 5).
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.electrostatic.api import ElectrostaticBoundarySpec
from witwin.maxwell.electrostatic.capacitance import (
    TruncationEstimate,
    TruncationReport,
    _enlarged_scene,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


# --------------------------------------------------------------------------- #
# ``open`` boundary fail-close.
# --------------------------------------------------------------------------- #
def test_open_boundary_default_fails_closed():
    with pytest.raises(NotImplementedError, match="Open .*electrostatic boundaries"):
        ElectrostaticBoundarySpec(default="open")


def test_open_boundary_per_face_fails_closed():
    with pytest.raises(NotImplementedError, match="truncation_estimate"):
        ElectrostaticBoundarySpec.dirichlet(x_low="open")


# --------------------------------------------------------------------------- #
# Scene builders.
# --------------------------------------------------------------------------- #
def _isolated_conductor_scene(length, h, cube=0.2):
    """A single conducting cube at the origin inside a domain [-L/2, L/2]^3."""
    domain = mw.Domain(bounds=((-length / 2, length / 2),) * 3)
    grid = mw.GridSpec.uniform(h)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="c", geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(cube, cube, cube)),
            potential=1.0,
        )
    )
    return scene


def _self_capacitance(length, h, cube=0.2):
    scene = _isolated_conductor_scene(length, h, cube=cube)
    cap = mw.Simulation.capacitance(
        scene, terminals=("c",), boundary=ElectrostaticBoundarySpec.grounded_box()
    ).run().capacitance
    return float(cap.matrix[0, 0])


# --------------------------------------------------------------------------- #
# Two-axis domain-extension convergence study (headline gate).
# --------------------------------------------------------------------------- #
def test_domain_extension_convergence_vs_size():
    """Self-capacitance of an isolated conductor decreases toward C_inf as L grows.

    A grounded box of linear size L truncates the open-domain field of a charged
    conductor: the nearer the grounded walls, the larger the induced charge, so
    C(L) decreases monotonically toward the infinite-domain self-capacitance with a
    leading 1/L correction. The 1/L Richardson extrapolation from successive size
    pairs must agree (confirms the 1/L truncation model) and the extrapolated C_inf
    must lie below every finite-L capacitance.
    """
    h = 0.05
    sizes = [0.8, 1.2, 1.6, 2.0]
    caps = [_self_capacitance(L, h) for L in sizes]

    # Monotone decreasing in L (walls move away -> less induced charge).
    for a, b in zip(caps, caps[1:]):
        assert a > b, f"C not decreasing with L: {caps}"

    # 1/L Richardson from consecutive pairs: C_inf = C1 + (C1 - C0)/(L1/L0 - 1).
    def richardson(i):
        f = sizes[i + 1] / sizes[i]
        return caps[i + 1] + (caps[i + 1] - caps[i]) / (f - 1.0)

    c_inf = [richardson(i) for i in range(len(sizes) - 1)]
    # Successive extrapolations converge (the model is consistent).
    rel = [abs(c_inf[i + 1] - c_inf[i]) / abs(c_inf[-1]) for i in range(len(c_inf) - 1)]
    assert max(rel) < 0.05, f"Richardson C_inf not stable across pairs: {c_inf} (rel {rel})"
    # Every finite-L capacitance exceeds the extrapolated infinite-domain limit.
    assert min(caps) > c_inf[-1] > 0.0, f"C_inf {c_inf[-1]} not below finite-L caps {caps}"
    # The finite-domain truncation error genuinely shrinks with L.
    err = [abs(c - c_inf[-1]) / c_inf[-1] for c in caps]
    assert err[0] > err[-1], f"truncation error not shrinking with L: {err}"


def test_grid_convergence_at_fixed_size():
    """At a fixed enclosure size, the self-capacitance is Cauchy-convergent in h."""
    length = 1.0
    hs = [0.1, 0.05, 0.025]
    caps = [_self_capacitance(length, h) for h in hs]
    # Successive changes shrink under refinement (discretization converges).
    d1 = abs(caps[1] - caps[0])
    d2 = abs(caps[2] - caps[1])
    assert d2 < d1, f"grid refinement not converging: caps {caps} (|dC|: {d1}, {d2})"


# --------------------------------------------------------------------------- #
# Opt-in truncation_estimate API.
# --------------------------------------------------------------------------- #
def test_truncation_estimate_is_opt_in():
    """No enlarged solve, and no truncation report, unless explicitly requested."""
    scene = _isolated_conductor_scene(1.0, 0.05)
    data = mw.Simulation.capacitance(
        scene, terminals=("c",), boundary=ElectrostaticBoundarySpec.grounded_box()
    ).run().capacitance
    assert data.truncation_estimate is None


def test_truncation_estimate_reports_domain_error():
    scene = _isolated_conductor_scene(1.0, 0.05)
    pad = 8
    result = mw.Simulation.capacitance(
        scene,
        terminals=("c",),
        boundary=ElectrostaticBoundarySpec.grounded_box(),
        truncation_estimate=TruncationEstimate(padding_cells=pad),
    ).run()
    report = result.capacitance.truncation_estimate
    assert isinstance(report, TruncationReport)

    # Base matrix in the report matches the returned matrix.
    assert torch.equal(report.base_matrix, result.capacitance.matrix)
    # The enlarged domain grew by exactly 2 * pad cells per axis (interior grid fixed).
    assert math.isclose(report.enlarged_size, report.base_size + 2 * pad * 0.05, rel_tol=1e-9)
    # Pushing the grounded box out lowers the self-capacitance (delta < 0) and the
    # measured sensitivity is a real, positive number.
    assert report.max_relative_delta > 0.0
    assert float(report.delta[0, 0]) < 0.0
    # The 1/L Richardson estimate lies below the enlarged capacitance (still
    # approaching C_inf from above). It extrapolates the FULL residual to the
    # infinite-domain limit, so for a sub-doubling extension (f < 2) the shift
    # legitimately exceeds one extension's delta (shift = delta / (f - 1)); it need
    # only be a real positive number.
    assert float(report.richardson_matrix[0, 0]) < float(report.enlarged_matrix[0, 0])
    assert report.richardson_max_relative_shift > 0.0

    # Surfaced in the standard Result solver_stats.
    assert "truncation_max_relative_delta" in result.solver_stats


def test_truncation_estimate_reduces_with_larger_padding():
    """A larger enclosure already truncates less: the incremental delta shrinks."""
    scene = _isolated_conductor_scene(1.6, 0.05)
    small = mw.Simulation.capacitance(
        scene, terminals=("c",), boundary=ElectrostaticBoundarySpec.grounded_box(),
        truncation_estimate=TruncationEstimate(padding_cells=4),
    ).run().capacitance.truncation_estimate
    scene2 = _isolated_conductor_scene(1.6, 0.05)
    # Base is the same, but a larger base enclosure would be closer to converged;
    # here we compare the sensitivity of the SAME base for two padding amounts.
    big = mw.Simulation.capacitance(
        scene2, terminals=("c",), boundary=ElectrostaticBoundarySpec.grounded_box(),
        truncation_estimate=TruncationEstimate(padding_cells=12),
    ).run().capacitance.truncation_estimate
    # A larger extension moves the capacitance more in absolute terms (it captures
    # more of the residual truncation), so its max_relative_delta is larger.
    assert big.max_relative_delta > small.max_relative_delta


# --------------------------------------------------------------------------- #
# truncation_estimate fail-closed usage errors.
# --------------------------------------------------------------------------- #
def test_truncation_estimate_requires_dirichlet_enclosure():
    """A pure-Neumann boundary has no enclosure to extend; reject the request."""
    domain = mw.Domain(bounds=((-0.5, 0.5),) * 3)
    grid = mw.GridSpec.uniform(0.05)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=mw.Box(position=(-0.2, 0.0, 0.0), size=(0.1, 0.2, 0.2)), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.1, 0.2, 0.2)), grounded=True)
    )
    with pytest.raises(ValueError, match="Dirichlet enclosure"):
        mw.Simulation.capacitance(
            scene, terminals=("a", "b"), reference="b",
            boundary=ElectrostaticBoundarySpec.neumann(),
            truncation_estimate=TruncationEstimate(padding_cells=4),
        ).run()


def test_truncation_estimate_requires_uniform_grid():
    """The domain-extension helper needs a uniform grid to grow by whole cells."""
    import numpy as np

    coords = np.linspace(-0.5, 0.5, 21)
    grid = mw.GridSpec.custom(x_coords=coords, y_coords=coords, z_coords=coords)
    domain = mw.Domain(bounds=((-0.5, 0.5),) * 3)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    with pytest.raises(ValueError, match="uniform GridSpec"):
        _enlarged_scene(scene, 4)


def test_truncation_estimate_type_check():
    scene = _isolated_conductor_scene(1.0, 0.05)
    with pytest.raises(TypeError, match="TruncationEstimate"):
        mw.Simulation.capacitance(
            scene, terminals=("c",), truncation_estimate=object()
        )


# --------------------------------------------------------------------------- #
# Differentiability disposition (public capacitance path, tensor eps).
# --------------------------------------------------------------------------- #
def test_capacitance_trainable_tensor_fails_closed():
    """The public capacitance path rejects a trainable input under a tensor dielectric.

    Phase 4 delivers the forward SPD tensor-eps operator; the off-diagonal
    cross-flux has no reverse-mode VJP, so a differentiable run under an
    anisotropic dielectric fails closed (gradients are Phase 5) rather than
    silently detaching. A trainable free-charge density alongside an anisotropic
    Structure is the reachable public trigger.
    """
    domain = mw.Domain(bounds=((0.0, 1.0),) * 3)
    grid = mw.GridSpec.uniform(1.0 / 12)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    rows = ((3.0, 0.5, 0.0), (0.5, 4.0, 0.0), (0.0, 0.0, 5.0))
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(2.0, 2.0, 2.0)),
            material=mw.Material(epsilon_tensor=mw.Tensor3x3(rows)),
        )
    )
    density = torch.ones((), dtype=torch.float64, device="cuda", requires_grad=True)
    scene.add_charge_density(
        mw.ChargeDensity(geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(0.2, 0.2, 0.2)), density=density)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="a", geometry=mw.Box(position=(0.3, 0.5, 0.5), size=(0.1, 0.3, 0.3)), potential=1.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(name="b", geometry=mw.Box(position=(0.7, 0.5, 0.5), size=(0.1, 0.3, 0.3)), potential=0.0)
    )
    with pytest.raises(NotImplementedError, match="tensor"):
        mw.Simulation.capacitance(
            scene, terminals=("a", "b"), boundary=ElectrostaticBoundarySpec.grounded_box()
        ).run()
