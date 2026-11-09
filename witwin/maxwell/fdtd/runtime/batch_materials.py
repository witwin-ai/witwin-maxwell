"""Activity-gated GPU material histories around native batched Yee updates."""
from __future__ import annotations

import torch

from ..checkpoint import iter_dispersive_state_specs, iter_magnetic_dispersive_state_specs


def material_state_tensors(solver, *, magnetic=False):
    iterator = iter_magnetic_dispersive_state_specs if magnetic else iter_dispersive_state_specs
    return tuple(entry[name] for _component, _kind, _index, names, entry in iterator(solver) or () for name in names)


class NativeMaterialLane:
    """Keep each lane's native ADE recurrence and freeze inactive endpoints.

    Scratch tensors have fixed addresses and are allocated once at cohort bind.
    Their GPU selections run inside the same CUDA graph as the batched curls.
    No material fields or histories are copied to the host.
    """

    def __init__(self, solver, activity):
        self.solver, self.activity = solver, activity
        self.electric = material_state_tensors(solver)
        self.magnetic = material_state_tensors(solver, magnetic=True)
        self.initial = tuple((tensor, tensor.clone()) for tensor in self.electric + self.magnetic)
        self.electric_scratch = tuple(torch.empty_like(tensor) for tensor in self.electric)
        self.magnetic_scratch = tuple(torch.empty_like(tensor) for tensor in self.magnetic)
        self.fields = tuple(getattr(solver, name) for name in ("Hx", "Hy", "Hz")) if self.magnetic else ()
        self.field_scratch = tuple(torch.empty_like(tensor) for tensor in self.fields)

    def _apply(self, operation, tensors, scratch):
        if not tensors:
            return
        for tensor, saved in zip(tensors, scratch):
            saved.copy_(tensor)
        operation()
        for tensor, saved in zip(tensors, scratch):
            torch.where(self.activity != 0, tensor, saved, out=tensor)

    def advance_magnetic(self):
        self._apply(self.solver._advance_magnetic_dispersive_state, self.magnetic, self.magnetic_scratch)

    def correct_magnetic(self):
        self._apply(self.solver._apply_magnetic_dispersive_corrections, self.fields, self.field_scratch)

    def advance_electric(self):
        self._apply(self.solver._advance_dispersive_state, self.electric, self.electric_scratch)
