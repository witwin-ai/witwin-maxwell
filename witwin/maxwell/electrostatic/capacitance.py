from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..compiler.electrostatic import CompiledElectrostatics, compile_electrostatics
from .api import ElectrostaticBoundarySpec, ElectrostaticSolverConfig
from .runtime import (
    ElectrostaticOperator,
    _pinned_value,
    _reject_trainable_tensor,
    solve_fixed_potential,
)


@dataclass
class CapacitanceData:
    """Maxwell capacitance matrix extracted from unit-voltage electrostatic solves.

    ``matrix[i, j] = Q_i`` (Coulombs, and Farads since each excitation drives a
    unit potential) is the free charge induced on active terminal ``i`` when
    terminal ``j`` is held at 1 V and every other conductor (active terminals and
    the ``reference``) is grounded to 0 V. ``terminal_order`` labels the rows and
    columns (the active, non-reference terminals in the requested order).

    The matrix is reported raw: no symmetrization is applied. In the standard
    passive-dielectric convention it is symmetric (``reciprocity_error``), has
    non-negative diagonal self terms and non-positive off-diagonal mutual terms.

    ``charges`` holds the same raw induced-charge measurements as ``matrix`` (unit
    excitation makes charge and capacitance numerically identical) and is kept as
    the primary physical quantity the matrix is derived from. ``energy[j]`` is the
    field energy of excitation ``j`` (equal to ``0.5 * matrix[j, j]``).

    ``reciprocity_error`` is ``max|C - Cᵀ| / max|C|``. ``row_sum_error`` is the
    largest induced charge, relative to ``max|C|``, when every conductor (active
    terminals and the reference) is driven to 1 V simultaneously; under an
    insulating (Neumann) outer boundary with no free charge the whole conductor
    set is isolated, so this must vanish (a discrete charge-conservation check).
    With a Dirichlet outer boundary the field terminates on the enclosure and this
    quantity is legitimately non-zero.
    """

    matrix: torch.Tensor
    terminal_order: tuple[str, ...]
    reference: str | None
    charges: torch.Tensor
    energy: torch.Tensor
    reciprocity_error: float
    row_sum_error: float
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def _resolve(self, name: str) -> int:
        if name not in self._index:
            raise KeyError(f"No active terminal named {name!r}; available: {self.terminal_order}.")
        return self._index[name]

    def capacitance(self, a: str, b: str) -> torch.Tensor:
        """Raw Maxwell matrix entry ``C[a, b]`` (F)."""
        return self.matrix[self._resolve(a), self._resolve(b)]

    def mutual_capacitance(self, a: str, b: str) -> torch.Tensor:
        """Partial mutual capacitance ``-C[a, b]`` (F, Maxwell convention).

        In the Maxwell matrix the off-diagonal entries are non-positive; the
        physical coupling capacitance between two electrodes is their negation.
        """
        if a == b:
            raise ValueError("mutual_capacitance requires two distinct terminals.")
        return -self.matrix[self._resolve(a), self._resolve(b)]

    def capacitance_to_reference(self, a: str) -> torch.Tensor:
        """Self capacitance of terminal ``a`` to the reference: row sum ``sum_j C[a, j]``."""
        i = self._resolve(a)
        return self.matrix[i].sum()

    def two_terminal_capacitance(self, a: str | None = None, b: str | None = None) -> torch.Tensor:
        """Equivalent capacitance seen between terminals ``a`` and ``b``.

        Places ``+Q`` on ``a`` and ``-Q`` on ``b`` (all other active terminals
        held at the reference) and returns ``Q / (V_a - V_b)``, computed from the
        2x2 submatrix ``[[Caa, Cab], [Cba, Cbb]]`` as
        ``det / (Caa + Cbb + Cab + Cba)``. Defaults to the two terminals when the
        matrix is exactly 2x2.
        """
        if a is None and b is None:
            if len(self.terminal_order) != 2:
                raise ValueError(
                    "two_terminal_capacitance needs explicit terminals unless the matrix is 2x2."
                )
            a, b = self.terminal_order
        if a is None or b is None or a == b:
            raise ValueError("two_terminal_capacitance requires two distinct terminals.")
        i, j = self._resolve(a), self._resolve(b)
        caa = self.matrix[i, i]
        cbb = self.matrix[j, j]
        cab = self.matrix[i, j]
        cba = self.matrix[j, i]
        det = caa * cbb - cab * cba
        return det / (caa + cbb + cab + cba)


def _excitation_charges(operator, compiled, drive: dict[str, float], b_full, config):
    """Solve one unit-excitation problem and return (charges_by_name, phi).

    The solve routes through the implicit-diff wrapper and the induced charges are
    read back with the grad-enabled ``operator`` (built from the live
    ``compiled.epsilon_r``), so each capacitance entry is differentiable in the
    permittivity. Capacitance measures the pure conductor response, so the free
    charge is excluded (``use_free_charge=False``).
    """
    shape, dtype, device = compiled.shape, compiled.dtype, compiled.device
    entries = [(t.mask, drive.get(t.name, 0.0)) for t in compiled.terminals]
    fixed_value = _pinned_value(shape, dtype, device, entries)
    phi, _ = solve_fixed_potential(
        compiled, operator, fixed_value, free_mask_of(compiled), config, use_free_charge=False
    )
    reaction = operator.apply_full(phi) - b_full
    charges = {t.name: reaction[t.mask].sum() for t in compiled.terminals}
    return charges, phi


def free_mask_of(compiled: CompiledElectrostatics) -> torch.Tensor:
    all_mask = torch.zeros(compiled.shape, dtype=torch.bool, device=compiled.device)
    for terminal in compiled.terminals:
        all_mask = all_mask | terminal.mask
    return ~all_mask


def extract_capacitance(
    compiled: CompiledElectrostatics,
    terminals: tuple[str, ...] | None,
    reference: str | None,
    config: ElectrostaticSolverConfig | None = None,
) -> CapacitanceData:
    if config is None:
        config = ElectrostaticSolverConfig()
    if compiled.dtype != config.dtype:
        raise ValueError(
            "CompiledElectrostatics dtype must match the solver dtype; compile with the "
            "same dtype the solver uses."
        )
    _reject_trainable_tensor(compiled)

    names = [t.name for t in compiled.terminals]
    if not names:
        raise ValueError("Capacitance extraction requires at least one electrostatic terminal.")
    if terminals is None:
        terminals = tuple(names)
    else:
        terminals = tuple(str(name) for name in terminals)
        for name in terminals:
            if name not in names:
                raise ValueError(
                    f"Capacitance terminal {name!r} is not a scene electrostatic terminal "
                    f"(available: {tuple(names)})."
                )
        if len(set(terminals)) != len(terminals):
            raise ValueError("Capacitance terminals must be distinct.")

    if reference is not None:
        reference = str(reference)
        if reference not in names:
            raise ValueError(
                f"Capacitance reference {reference!r} is not a scene electrostatic terminal."
            )

    active = tuple(name for name in terminals if name != reference)
    if not active:
        raise ValueError(
            "Capacitance extraction needs at least one non-reference terminal to excite."
        )
    if reference is None and not compiled.boundary.has_dirichlet:
        raise ValueError(
            "Capacitance extraction needs a charge return path: pass reference=<terminal> or "
            "use a Dirichlet (grounded_box) boundary so induced flux has somewhere to terminate."
        )

    operator = ElectrostaticOperator(compiled)
    # Capacitance measures the pure conductor response; ignore any free charge.
    b_full = operator.rhs_boundary()

    n = len(active)
    dtype, device = compiled.dtype, compiled.device
    matrix = torch.zeros((n, n), dtype=dtype, device=device)
    energy = torch.zeros((n,), dtype=dtype, device=device)
    for col, name in enumerate(active):
        charges, phi = _excitation_charges(operator, compiled, {name: 1.0}, b_full, config)
        for row, other in enumerate(active):
            matrix[row, col] = charges[other]
        energy[col] = operator.field_energy(phi)

    # Charge conservation check: all conductors (active + reference) at 1 V.
    uniform_drive = {name: 1.0 for name in terminals}
    uniform_charges, _ = _excitation_charges(operator, compiled, uniform_drive, b_full, config)
    scale = float(matrix.detach().abs().max()) + 1.0e-300
    row_sum_error = max(abs(float(uniform_charges[name].detach())) for name in active) / scale
    reciprocity_error = float((matrix - matrix.T).detach().abs().max()) / scale

    index = {name: i for i, name in enumerate(active)}
    return CapacitanceData(
        matrix=matrix,
        terminal_order=active,
        reference=reference,
        charges=matrix.clone(),
        energy=energy,
        reciprocity_error=reciprocity_error,
        row_sum_error=row_sum_error,
        _index=index,
    )


class CapacitanceSimulation:
    """Runner returned by ``Simulation.capacitance(...)``.

    Keeps the public entry ``Scene -> Simulation -> Result``: ``run()`` returns a
    standard ``Result(method="capacitance")`` whose ``result.capacitance`` accessor
    is the typed :class:`CapacitanceData`.
    """

    def __init__(self, scene, *, terminals=None, reference=None, boundary=None, solver=None):
        self.scene = scene
        self.terminals = None if terminals is None else tuple(terminals)
        self.reference = None if reference is None else str(reference)
        if boundary is None:
            boundary = ElectrostaticBoundarySpec.grounded_box()
        if not isinstance(boundary, ElectrostaticBoundarySpec):
            raise TypeError("boundary must be an ElectrostaticBoundarySpec.")
        self.boundary = boundary
        if solver is None:
            solver = ElectrostaticSolverConfig()
        if not isinstance(solver, ElectrostaticSolverConfig):
            raise TypeError("solver must be an ElectrostaticSolverConfig.")
        self.solver = solver

    def prepare(self) -> CompiledElectrostatics:
        return compile_electrostatics(self.scene, self.boundary, dtype=self.solver.dtype)

    def run(self):
        from ..result import Result

        compiled = self.prepare()
        data = extract_capacitance(compiled, self.terminals, self.reference, self.solver)
        return Result(
            method="capacitance",
            scene=self.scene,
            frequency=0.0,
            solver_stats={
                "reciprocity_error": data.reciprocity_error,
                "row_sum_error": data.row_sum_error,
                "terminal_order": data.terminal_order,
                "reference": data.reference,
            },
            raw_output=data,
        )
