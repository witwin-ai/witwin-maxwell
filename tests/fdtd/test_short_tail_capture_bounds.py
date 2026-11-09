"""Host-only table-bound and snapshot oracles; no GPU physics is simulated."""
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.runtime.stepping import _make_tail_runner

pytestmark = pytest.mark.host_validation


def _solver(rows, step):
    return SimpleNamespace(
        device="cpu",
        _cuda_graph_active=True,
        _dft_weighted_cos_table=torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        _dft_weighted_sin_table=torch.zeros((rows, 3)),
        _dft_step=torch.tensor([step], dtype=torch.int64),
        field=torch.tensor([7.0]),
        mur=torch.tensor([0.2, 0.3]),
        _dft_batched_fields={"Ex": {"real": torch.tensor([0.4]), "imag": torch.tensor([0.6])}},
    )


def _install_schedule_model(monkeypatch, solver, *, fail_capture=False):
    calls, warmups, record_positions = [], [], []

    def block(state):
        index = int(state._dft_step.item())
        state.field.add_(state._dft_weighted_cos_table[index, 0] + 1)
        state.mur.add_(1)
        state._dft_batched_fields["Ex"]["real"].add_(2)
        state._dft_step.add_(1)
        calls.append(index)

    class ScheduleModel:
        def __init__(self, *, warmup_steps, **kwargs):
            self.warmup_steps = warmup_steps
            warmups.append(warmup_steps)

        def capture(self, function):
            for _ in range(self.warmup_steps):
                function()
            record_positions.append(int(solver._dft_step.item()))
            if fail_capture:
                raise RuntimeError("Injected capture failure after host schedule warmup")
            # CUDA recording does not execute device operations. This host model
            # returns the block as replay after the production helper restores.
            return function

    monkeypatch.setattr("witwin.maxwell.fdtd.cuda.runtime.graph.CudaGraphRunner", ScheduleModel)
    monkeypatch.setattr(
        "witwin.maxwell.fdtd.checkpoint.graph_state_tensors",
        lambda state, tail=True: {("field",): state.field, ("mur",): state.mur},
    )
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.stepping._post_source_block", block)
    return calls, warmups, record_positions


@pytest.mark.parametrize(
    "rows,step,expected",
    [(1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 0, 3), (8, 0, 3), (8, 5, 2), (8, 6, 1), (8, 7, 0)],
)
def test_remaining_rows_bound_and_complete_nonzero_state_restore(rows, step, expected, monkeypatch):
    initial_cuda = torch.cuda.is_initialized()
    solver = _solver(rows, step)
    calls, warmups, positions = _install_schedule_model(monkeypatch, solver)
    saved = {name: getattr(solver, name).clone() for name in ("field", "mur", "_dft_step")}
    dft = solver._dft_batched_fields["Ex"]["real"].clone()

    replay = _make_tail_runner(solver, True)

    assert replay is not None
    assert warmups == [expected]
    assert calls == list(range(step, step + expected))
    assert positions == [step + expected] and positions[0] < rows
    for name, value in saved.items():
        assert torch.equal(getattr(solver, name), value)
    assert torch.equal(solver._dft_batched_fields["Ex"]["real"], dft)
    replay()
    assert int(solver._dft_step.item()) == step + 1
    assert float(solver.field) == float(saved["field"] + solver._dft_weighted_cos_table[step, 0] + 1)
    assert torch.cuda.is_initialized() == initial_cuda


@pytest.mark.parametrize("rows,step", [(0, 0), (1, 1), (4, 4)])
def test_exhausted_horizon_does_not_capture_or_touch_state(rows, step, monkeypatch):
    solver = _solver(rows, step)
    calls, warmups, _ = _install_schedule_model(monkeypatch, solver)
    assert _make_tail_runner(solver, True) is None
    assert not calls and not warmups
    assert float(solver.field) == 7.0
    assert torch.equal(solver.mur, torch.tensor([0.2, 0.3]))


@pytest.mark.parametrize(
    "change",
    [
        {"_dft_step": torch.tensor([-1])},
        {"_dft_step": torch.tensor([5])},
        {"_dft_step": torch.tensor([0.0])},
        {"_dft_step": torch.tensor(0)},
        {"_dft_step": None},
        {"_dft_weighted_cos_table": None},
        {"_dft_weighted_sin_table": torch.zeros((3, 3))},
        {"_dft_weighted_sin_table": torch.zeros((4, 2))},
        {
            "_dft_weighted_cos_table": torch.zeros((4, 3), dtype=torch.int64),
            "_dft_weighted_sin_table": torch.zeros((4, 3), dtype=torch.int64),
        },
        {
            "_dft_weighted_cos_table": torch.zeros((4, 3), dtype=torch.float64),
            "_dft_weighted_sin_table": torch.zeros((4, 3), dtype=torch.float64),
        },
    ],
)
def test_invalid_table_counter_rejected_before_capture_or_mutation(change, monkeypatch):
    solver = _solver(4, 0)
    solver.__dict__.update(change)
    calls, warmups, _ = _install_schedule_model(monkeypatch, solver)
    with pytest.raises(ValueError):
        _make_tail_runner(solver, True)
    assert not calls and not warmups
    assert float(solver.field) == 7.0
    assert torch.equal(solver.mur, torch.tensor([0.2, 0.3]))


def test_capture_failure_restores_nonzero_mur_field_counter_and_DFT(monkeypatch):
    solver = _solver(3, 0)
    _install_schedule_model(monkeypatch, solver, fail_capture=True)
    assert _make_tail_runner(solver, True) is None
    assert float(solver.field) == 7.0
    assert torch.equal(solver.mur, torch.tensor([0.2, 0.3]))
    assert int(solver._dft_step.item()) == 0
    assert torch.equal(solver._dft_batched_fields["Ex"]["real"], torch.tensor([0.4]))


def test_original_fixed_three_warmups_overrun_one_row_before_replay():
    solver = _solver(1, 0)
    with pytest.raises(IndexError):
        for _ in range(3):
            solver._dft_weighted_cos_table[int(solver._dft_step.item())]
            solver._dft_step.add_(1)
    assert int(solver._dft_step.item()) == 1
