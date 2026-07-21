"""SPD tensor-permittivity electrostatic operator: symmetry, positive-definiteness,
rotated-frame manufactured-solution convergence, diagonal-reduction parity, energy
identity / Gauss closure, and anisotropic capacitance reciprocity.

These gates back the Phase-4 SPD tensor-eps finite-volume operator. The cross-flux
discretization is derived from a discrete energy ``W(phi) = 0.5 phi^T A phi`` so the
operator is symmetric by construction; the off-diagonal terms carry only the
off-diagonal permittivity, so a diagonal tensor reduces exactly to the isotropic /
per-axis face-flux path.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.electrostatic import (
    CompiledElectrostatics,
    TensorEpsilon,
    compile_electrostatics,
)
from witwin.maxwell.electrostatic.api import ElectrostaticBoundarySpec, ElectrostaticSolverConfig
from witwin.maxwell.electrostatic.runtime import ElectrostaticOperator, _reduced_solve

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="electrostatics is GPU-first")


def _rotation(alpha: float, beta: float) -> np.ndarray:
    """R = Rz(alpha) @ Rx(beta): a generic rotation that mixes all three axes."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    return rz @ rx


def _rotated_spd(eigenvalues, alpha=math.radians(30.0), beta=math.radians(45.0)) -> np.ndarray:
    """A constant SPD tensor: rotate diag(eigenvalues) by 30 deg / 45 deg."""
    r = _rotation(alpha, beta)
    return r @ np.diag(np.asarray(eigenvalues, dtype=np.float64)) @ r.T


def _const_compiled(n, length, matrix, boundary_spec, dtype=torch.float64):
    """CompiledElectrostatics on a uniform [0, length]^3 grid with a constant tensor."""
    domain = mw.Domain(bounds=((0.0, length),) * 3)
    grid = mw.GridSpec.uniform(length / n)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    compiled = compile_electrostatics(scene, boundary_spec, dtype=dtype)
    ones = torch.ones(compiled.shape, dtype=dtype, device=compiled.device)
    m = np.asarray(matrix, dtype=np.float64)
    tensor = TensorEpsilon(
        xx=ones * float(m[0, 0]),
        yy=ones * float(m[1, 1]),
        zz=ones * float(m[2, 2]),
        xy=ones * float(m[0, 1]),
        xz=ones * float(m[0, 2]),
        yz=ones * float(m[1, 2]),
    )
    epsilon_r = ones * float((m[0, 0] + m[1, 1] + m[2, 2]) / 3.0)
    return dataclasses.replace(compiled, epsilon_tensor=tensor, epsilon_r=epsilon_r)


def _dense_operator(operator) -> torch.Tensor:
    """Materialize the full operator as a dense matrix by applying to unit vectors."""
    shape = operator.shape
    ncells = int(np.prod(shape))
    columns = []
    for idx in range(ncells):
        e = torch.zeros(ncells, dtype=operator.dtype, device=operator.device)
        e[idx] = 1.0
        col = operator.apply_full(e.reshape(shape)).reshape(-1)
        columns.append(col)
    return torch.stack(columns, dim=1)  # A[:, idx] = A e_idx


# --------------------------------------------------------------------------- #
# Symmetry and positive-definiteness (headline gate).
# --------------------------------------------------------------------------- #
def test_operator_symmetry_and_positive_definite():
    """The rotated-SPD-tensor operator is symmetric and positive-definite.

    Built on a small grounded box so the Dirichlet ghost makes the full operator
    strictly positive-definite (no constant null space). Symmetry is checked as a
    dense matrix property; positive-definiteness from the symmetric eigenvalues.
    """
    matrix = _rotated_spd((1.0, 2.0, 3.0))
    compiled = _const_compiled(6, 1.0, matrix, ElectrostaticBoundarySpec.grounded_box())
    operator = ElectrostaticOperator(compiled)
    assert operator.tensor_mode

    A = _dense_operator(operator)
    scale = float(A.abs().max())
    asymmetry = float((A - A.T).abs().max()) / scale
    assert asymmetry < 1e-9, f"operator not symmetric to fp tolerance: {asymmetry:.2e}"

    eigenvalues = torch.linalg.eigvalsh(0.5 * (A + A.T))
    assert float(eigenvalues.min()) > 0.0, (
        f"operator is not positive-definite: min eigenvalue {float(eigenvalues.min()):.3e}"
    )


def test_operator_symmetry_random_vectors():
    """<A x, y> == <x, A y> to fp tolerance for random vectors (Neumann box)."""
    matrix = _rotated_spd((1.0, 2.5, 4.0))
    compiled = _const_compiled(10, 1.0, matrix, ElectrostaticBoundarySpec.neumann())
    operator = ElectrostaticOperator(compiled)
    torch.manual_seed(0)
    x = torch.randn(compiled.shape, dtype=torch.float64, device=compiled.device)
    y = torch.randn(compiled.shape, dtype=torch.float64, device=compiled.device)
    lhs = float((operator.apply_full(x) * y).sum())
    rhs = float((x * operator.apply_full(y)).sum())
    # Normalize by the magnitude of the bilinear form itself (the absolute
    # values are ~eps0-scale, so a fixed 1.0 floor would mask any asymmetry).
    scale = max(abs(lhs), abs(rhs))
    assert abs(lhs - rhs) / scale < 1e-10


def test_positive_energy_for_nonzero_field():
    """Field energy is strictly positive for any nonzero potential (grounded box)."""
    matrix = _rotated_spd((1.0, 2.0, 3.0))
    compiled = _const_compiled(8, 1.0, matrix, ElectrostaticBoundarySpec.grounded_box())
    operator = ElectrostaticOperator(compiled)
    torch.manual_seed(1)
    for _ in range(5):
        phi = torch.randn(compiled.shape, dtype=torch.float64, device=compiled.device)
        assert float(operator.field_energy(phi)) > 0.0


# --------------------------------------------------------------------------- #
# Energy identity: field_energy(phi) == 0.5 phi^T A phi.
# --------------------------------------------------------------------------- #
def test_energy_identity_matches_quadratic_form():
    matrix = _rotated_spd((1.0, 2.0, 3.0))
    compiled = _const_compiled(8, 1.0, matrix, ElectrostaticBoundarySpec.grounded_box())
    operator = ElectrostaticOperator(compiled)
    torch.manual_seed(2)
    phi = torch.randn(compiled.shape, dtype=torch.float64, device=compiled.device)
    energy = float(operator.field_energy(phi))
    quadratic = 0.5 * float((phi * operator.apply_full(phi)).sum())
    assert abs(energy - quadratic) / max(abs(energy), 1.0) < 1e-10


# --------------------------------------------------------------------------- #
# Diagonal reduction parity (headline gate).
# --------------------------------------------------------------------------- #
def test_isotropic_material_stays_on_scalar_path():
    """An isotropic scalar (or an isotropic diagonal tensor) keeps epsilon_tensor None.

    So every existing isotropic analytic result is byte-identical (untouched path).
    """
    domain = mw.Domain(bounds=((0.0, 1.0),) * 3)
    grid = mw.GridSpec.uniform(1.0 / 8)
    for material in (
        mw.Material(eps_r=2.5),
        mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.5, 2.5, 2.5)),
    ):
        scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
        scene.add_structure(
            mw.Structure(geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(0.4, 0.4, 0.4)), material=material)
        )
        compiled = compile_electrostatics(scene, ElectrostaticBoundarySpec.grounded_box())
        assert compiled.epsilon_tensor is None


def test_diagonal_tensor_has_zero_cross_coupling():
    """A diagonal tensor (zero off-diagonals) reduces exactly to the face-flux path.

    The cross operator carries only the off-diagonal permittivity, so it maps every
    field to exactly zero; the operator is then the per-axis two-point flux stencil.
    """
    matrix = np.diag([2.0, 3.0, 4.0]).astype(np.float64)
    compiled = _const_compiled(8, 1.0, matrix, ElectrostaticBoundarySpec.grounded_box())
    operator = ElectrostaticOperator(compiled)
    assert operator.tensor_mode
    torch.manual_seed(3)
    phi = torch.randn(compiled.shape, dtype=torch.float64, device=compiled.device)
    cross = operator._apply_cross(phi)
    assert float(cross.abs().max()) == 0.0
    # And it equals the isotropic operator when all diagonal entries coincide.
    iso_matrix = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    iso_tensor = _const_compiled(8, 1.0, iso_matrix, ElectrostaticBoundarySpec.grounded_box())
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(1.0 / 8),
        boundary=mw.BoundarySpec.none(),
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(2.0, 2.0, 2.0)),
            material=mw.Material(eps_r=3.0),
        )
    )
    iso_scalar = compile_electrostatics(scene, ElectrostaticBoundarySpec.grounded_box())
    op_tensor = ElectrostaticOperator(iso_tensor)
    op_scalar = ElectrostaticOperator(iso_scalar)
    diff = (op_tensor.apply_full(phi) - op_scalar.apply_full(phi)).abs().max()
    assert float(diff) < 1e-12


# --------------------------------------------------------------------------- #
# Rotated-frame method-of-manufactured-solutions convergence (headline gate).
# --------------------------------------------------------------------------- #
#
# Manufactured solution: phi(x,y,z) = f(x) f(y) f(z) with f(t) = sin^2(pi t) on
# the unit cube. f and f' both vanish at t = 0, 1, so phi is a smooth bump with
# homogeneous Dirichlet data AND zero tangential gradient at the walls, yet its
# mixed second derivatives are nonzero throughout the interior, so the full
# rotated tensor (all off-diagonal entries active) is genuinely exercised. The
# forcing is the analytic ``rho = -div(eps0 eps_r grad phi)``, injected as the
# free charge; the boundary is pinned to zero.
def _bump_fields(coords):
    f = torch.sin(math.pi * coords) ** 2
    df = math.pi * torch.sin(2.0 * math.pi * coords)
    d2f = 2.0 * math.pi * math.pi * torch.cos(2.0 * math.pi * coords)
    return f, df, d2f


def _mms_solve(n, matrix):
    length = 1.0
    compiled = _const_compiled(n, length, matrix, ElectrostaticBoundarySpec.neumann())
    op = ElectrostaticOperator(compiled)
    device = compiled.device
    xx, yy, zz = torch.meshgrid(compiled.xc, compiled.yc, compiled.zc, indexing="ij")
    fx, dfx, d2x = _bump_fields(xx)
    fy, dfy, d2y = _bump_fields(yy)
    fz, dfz, d2z = _bump_fields(zz)
    phi_exact = fx * fy * fz
    m = np.asarray(matrix, dtype=np.float64)
    # Hessian components of phi_exact.
    hxx, hyy, hzz = d2x * fy * fz, fx * d2y * fz, fx * fy * d2z
    hxy, hxz, hyz = dfx * dfy * fz, dfx * fy * dfz, fx * dfy * dfz
    div = (
        m[0, 0] * hxx + m[1, 1] * hyy + m[2, 2] * hzz
        + 2.0 * m[0, 1] * hxy + 2.0 * m[0, 2] * hxz + 2.0 * m[1, 2] * hyz
    )
    free_charge = compiled.eps0 * (-div) * compiled.cell_volume

    shape = compiled.shape
    boundary = torch.zeros(shape, dtype=torch.bool, device=device)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    free_mask = ~boundary
    fixed_value = torch.zeros_like(phi_exact)

    config = ElectrostaticSolverConfig(tolerance=1e-13, max_iterations=80000)
    phi, report = _reduced_solve(op, free_mask, fixed_value, free_charge, config)
    assert report.converged
    err = (phi - phi_exact)[free_mask]
    volume = compiled.cell_volume[free_mask]
    l2 = float(torch.sqrt((err * err * volume).sum() / volume.sum()))
    gauss = float((op.apply_full(phi) - free_charge)[free_mask].abs().max())
    gauss_rel = gauss / float(free_charge.abs().max())
    return l2, gauss_rel


def test_rotated_mms_second_order_convergence():
    matrix = _rotated_spd((1.0, 2.0, 3.0))
    ns = (24, 36, 54)
    results = [_mms_solve(n, matrix) for n in ns]
    errors = [r[0] for r in results]
    # Monotone refinement.
    assert errors[0] > errors[1] > errors[2]
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i])
        for i in range(len(ns) - 1)
    ]
    # Second-order scheme (diagonal two-point flux + symmetric cross-derivative terms).
    assert min(orders) > 1.8, f"observed convergence orders {orders} below 2nd order"


def test_mms_gauss_closure():
    """The MMS solve satisfies the discrete Gauss law on interior cells."""
    matrix = _rotated_spd((1.0, 2.0, 3.0))
    _, gauss_rel = _mms_solve(36, matrix)
    assert gauss_rel < 1e-9


# --------------------------------------------------------------------------- #
# Anisotropic capacitance reciprocity (headline gate, end-to-end public API).
# --------------------------------------------------------------------------- #
def _anisotropic_capacitance():
    """Three asymmetrically-placed terminals inside a rotated anisotropic block.

    The asymmetric placement (no mirror symmetry) means the Maxwell matrix is
    symmetric only because the operator is: reciprocity here genuinely probes the
    cross-flux symmetrization (a mirror-symmetric two-terminal cell would report
    a symmetric matrix even with a broken operator).
    """
    length = 1.0
    domain = mw.Domain(bounds=((0.0, length),) * 3)
    grid = mw.GridSpec.uniform(length / 32)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.none())
    rows = _rotated_spd((2.0, 3.0, 5.0))
    material = mw.Material(
        epsilon_tensor=mw.Tensor3x3(tuple(tuple(float(v) for v in row) for row in rows))
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(2.0, 2.0, 2.0)), material=material)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="a", geometry=mw.Box(position=(0.28, 0.45, 0.5), size=(0.12, 0.3, 0.4)), potential=1.0
        )
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="b", geometry=mw.Box(position=(0.55, 0.72, 0.42), size=(0.16, 0.12, 0.3)), potential=0.0
        )
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal(
            name="c", geometry=mw.Box(position=(0.72, 0.35, 0.62), size=(0.12, 0.24, 0.16)), potential=0.0
        )
    )
    return mw.Simulation.capacitance(
        scene,
        terminals=("a", "b", "c"),
        boundary=mw.ElectrostaticBoundarySpec.grounded_box(),
    ).run().capacitance


def test_anisotropic_capacitance_reciprocity():
    cap = _anisotropic_capacitance()
    matrix = cap.matrix
    # Maxwell matrix stays symmetric under an anisotropic (cross-coupled) dielectric.
    assert cap.reciprocity_error < 1e-6, f"reciprocity_error {cap.reciprocity_error:.2e}"
    # Diagonal self terms positive, off-diagonal mutual terms non-positive.
    for i in range(len(cap.terminal_order)):
        assert float(matrix[i, i]) > 0.0
        for j in range(len(cap.terminal_order)):
            if i != j:
                assert float(matrix[i, j]) <= 0.0


def test_trainable_tensor_eps_fails_closed():
    """A trainable free charge alongside a tensor dielectric fails closed.

    The tensor operator's off-diagonal cross-flux has no reverse-mode VJP, so the
    solve must reject a trainable input rather than silently detaching its
    gradient.
    """
    from witwin.maxwell.electrostatic.runtime import solve_electrostatics

    matrix = _rotated_spd((1.0, 2.0, 3.0))
    compiled = _const_compiled(6, 1.0, matrix, ElectrostaticBoundarySpec.grounded_box())
    trainable_charge = compiled.free_charge.clone().requires_grad_(True)
    compiled = dataclasses.replace(compiled, free_charge=trainable_charge)
    with pytest.raises(NotImplementedError, match="tensor"):
        solve_electrostatics(compiled)


def test_anisotropic_capacitance_energy_consistency():
    """Excitation energy equals 0.5 * self capacitance (W = 0.5 C V^2, V = 1)."""
    cap = _anisotropic_capacitance()
    for j in range(len(cap.terminal_order)):
        assert abs(float(cap.energy[j]) - 0.5 * float(cap.matrix[j, j])) / float(cap.matrix[j, j]) < 1e-6
