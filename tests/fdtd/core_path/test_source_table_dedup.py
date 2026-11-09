"""Auxiliary TFSF source tables evaluate each step once and stay bitwise equal to per-entry evaluation."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from witwin.maxwell.fdtd.runtime import core_tables


class _Aux:
    def __init__(self, source_index, incident):
        self.time_step, self.source_index, self.dt = 7, source_index, 1.3e-12
        self.incident_reference = object() if incident else None
        self.electric = self.magnetic = self.electric_decay = self.electric_curl = torch.empty(0)
        self.calls = []

    def source_value(self, step):
        self.calls.append(step)
        t = (step + 0.5) * self.dt
        return math.exp(-((t - 2e-10) / 4e-11) ** 2) * math.sin(2 * math.pi * 3e9 * t)


@pytest.mark.parametrize("source_index,incident", [(1, False), (2, True), (3, True), (0, True)])
def test_auxiliary_tables_equal_per_entry_evaluation(source_index, incident):
    aux = _Aux(source_index, incident)
    solver = SimpleNamespace(device="cpu", dt=aux.dt, tfsf_enabled=True, _tfsf_state={"auxiliary_grid": aux},
                             _source_terms=[], _magnetic_source_terms=[], _electric_source_terms=[])
    extension = SimpleNamespace(CoreStepLaunch=lambda *args: args)
    steps = 37
    tables = core_tables.compile_source_tables(solver, extension, torch.zeros(1, dtype=torch.int64), steps)
    reference = torch.as_tensor(np.asarray([aux.source_value(aux.time_step + n) for n in range(steps)],
                                           dtype=np.float32))
    assert torch.equal(tables.rows["auxiliary/signals"], reference)
    if incident:
        boundary = torch.as_tensor(np.asarray(
            [[aux.source_value(aux.time_step + n + k) for k in range(source_index)] for n in range(steps)],
            dtype=np.float32))
        assert tables.rows["auxiliary/boundary_signals"].shape == (steps, source_index)
        assert torch.equal(tables.rows["auxiliary/boundary_signals"], boundary)
    else:
        assert "auxiliary/boundary_signals" not in tables.rows
