"""Torch reference arithmetic of the conformal thin-sheet patch (frozen test oracle).

These are the Torch twins of the fused CUDA thin-sheet kernels that used to run behind
``WITWIN_MAXWELL_THIN_SHEET_KERNEL=torch`` and on the retired CPU reference profile.
The runtime (``witwin.maxwell.fdtd.thin_sheet``) now has only the fused kernels; each
function here keeps the signature of its fused counterpart (``gather`` for
``_gather_fused`` and so on), so :func:`install` swaps them into a prepared run on the
same device and a test compares fused execution against this reference.
"""
from __future__ import annotations

import torch

from witwin.maxwell.fdtd import thin_sheet


def gather(solver, plan, extra_buffer, values):
    for name, positions, flat in plan["native"]:
        values.index_copy_(0, positions, getattr(solver, name).reshape(-1).index_select(0, flat).double())
    if len(plan["extra_positions"]):
        values.index_copy_(0, plan["extra_positions"], extra_buffer.index_select(0, plan["extra_slots"]))
    return values


def scatter(solver, plan, extra_buffer, values):
    for name, positions, flat in plan["native"]:
        field = getattr(solver, name)
        field.view(-1).index_copy_(0, flat, values.index_select(0, positions).to(field.dtype))
    if len(plan["extra_positions"]):
        extra_buffer.index_copy_(0, plan["extra_slots"], values.index_select(0, plan["extra_positions"]))


def copper_rhs(self, solver, old, new):
    cu, dt = self.copper, float(solver.dt)
    count = cu["count"]
    history = (cu["history"] @ cu["state"]).T
    current, previous = self.chord_e.reshape(-1), self.chord_e_prev.reshape(-1)
    mass, conductance = cu["mass"].reshape(-1), cu["conductance"].reshape(-1)
    rhs = (mass / dt - conductance / 2) * current + self.chord_f.reshape(-1)
    if self.feeds:
        local_current = torch.einsum("tij,tj->ti", cu["local"][:count], self.chord_e)
        # Feed matrices already include the side-average factors.
        chord_area = self.feed_area
        rhs = rhs - (chord_area[:, None] * local_current / 2
                     + cu["area"][:count, None] * history[:count]).reshape(-1)
        rhs = rhs - torch.mv(cu["feed_conductance"], current) / 2
        for feed in self.feeds:
            for column in feed["columns"]:
                if column["kind"] == "chord":
                    rhs[column["dofs"]] += column["conductance"] * column["u"] * feed["binding"].drive
    else:
        rhs = rhs - (cu["area"][:count, None] * (self.chord_e @ cu["effective"].T / 2 + history[:count])).reshape(-1)
    rhs = rhs - dt / 4 * (torch.mv(cu["Acx"], new + 2 * old + self.x_prev)
                          + torch.mv(cu["Acc"], 2 * current + previous))
    return history, rhs


def chord_forcing(self, solver, old, new, updated):
    cu, dt = self.copper, float(solver.dt)
    current, previous = self.chord_e.reshape(-1), self.chord_e_prev.reshape(-1)
    self.chord_f.sub_((dt / 4 * (torch.mv(cu["Acx"], new + 2 * old + self.x_prev)
                                   + torch.mv(cu["Acc"], updated + 2 * current + previous))).reshape(-1, 2))


def copper_state(self, solver, old, new, updated, pair_old, pair_new):
    cu, dt = self.copper, float(solver.dt)
    trace_old = torch.cat((self.chord_e, pair_old)).T
    trace_new = torch.cat((updated.reshape(-1, 2), pair_new)).T
    state = cu["state"]
    next_state = cu["decay"][:, None] * state + cu["drive"] @ (trace_new + trace_old)
    for feed in self.feeds:
        next_state.index_fill_(1, feed["ids"], 0)
    midpoint_e, midpoint_state = (trace_new + trace_old) / 2, (next_state + state) / 2
    loss = (cu["rates"][:, None] * midpoint_state.square()).sum(0)
    loss = loss + (midpoint_e * (cu["D"] @ midpoint_e)).sum(0)
    self.loss.add_(dt * torch.sum(loss * cu["area"]))
    for feed in self.feeds:
        binding = feed["binding"]
        for index, column in enumerate(feed["columns"]):
            binding.column_voltage[index] = torch.dot(midpoint_e.T[column["ids"]].reshape(-1), column["u"])
        g = binding.column_conductance
        binding.voltage.copy_(torch.dot(g, binding.column_voltage) / g.sum())
        binding.current.copy_(torch.sum(g * (binding.drive - binding.column_voltage)))
    state.copy_(next_state)
    self.chord_e_prev.copy_(self.chord_e)
    self.chord_e.copy_(updated.reshape(-1, 2))
    self.x_prev.copy_(old)
    new[cu["promoted_position"].reshape(-1)] = self.chord_e[cu["promoted_chords"]].reshape(-1)


def chord_solve(self, rhs):
    """Even Chebyshev residual s_k(mu)^2: 0 < P <= S^-1, with block Jacobi."""
    cu = self.copper
    if not cu["count"]:
        return rhs.clone()
    a, b, k = cu["chebyshev"]
    theta, delta = (b+a)/2, (b-a)/2
    sigma = theta / delta
    y, residual = torch.zeros_like(rhs), rhs.clone()
    for _ in range(2):
        rho = 1 / sigma
        direction = torch.mv(cu["R"], torch.mv(cu["R"], residual)) / theta
        for step in range(k):
            y = y + direction
            residual = residual - torch.mv(cu["S"], direction)
            if step + 1 < k:
                rho_next = 1 / (2 * sigma - rho)
                direction = rho_next * rho * direction + (2 * rho_next / delta) * torch.mv(cu["R"], torch.mv(cu["R"], residual))
                rho = rho_next
    return y


KERNELS = {
    "_gather_fused": gather,
    "_scatter_fused": scatter,
    "_copper_rhs_fused": copper_rhs,
    "_chord_forcing_fused": chord_forcing,
    "_copper_state_fused": copper_state,
    "_chord_solve_fused": chord_solve,
}


def install(monkeypatch):
    """Run every thin-sheet patch in this test on the Torch reference arithmetic."""
    for name, reference in KERNELS.items():
        monkeypatch.setattr(thin_sheet, name, reference)
