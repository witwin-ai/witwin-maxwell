from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.distributed.persistence import (
    export_distributed_field_shards,
)


_SENTINEL = -98765.0
_FREQUENCIES = (0.8e9, 1.2e9)
_LAYOUTS = (
    SimpleNamespace(
        storage_cell_owned=slice(0, 3),
        global_cell_owned=slice(0, 3),
        storage_node_owned=slice(0, 3),
        global_node_owned=slice(0, 3),
    ),
    SimpleNamespace(
        storage_cell_owned=slice(1, 3),
        global_cell_owned=slice(3, 5),
        storage_node_owned=slice(1, 4),
        global_node_owned=slice(3, 6),
    ),
)


class _DftSolver:
    dft_enabled = True

    def __init__(self, fields, frequencies):
        self._fields = fields
        self._frequencies = frequencies
        self.calls = []

    def get_frequency_solution(self, *, all_frequencies=False):
        self.calls.append(all_frequencies)
        return {**self._fields, "frequencies": self._frequencies}


def _global_fields(*, frequency_count=None):
    prefix = () if frequency_count is None else (frequency_count,)
    return {
        "Ex": torch.arange(5 * 2 * 2 * (frequency_count or 1), dtype=torch.float32)
        .reshape(prefix + (5, 2, 2)),
        "Ey": (
            1000
            + torch.arange(6 * 2 * 2 * (frequency_count or 1), dtype=torch.float32)
        ).reshape(prefix + (6, 2, 2)),
        "Ez": (
            2000
            + torch.arange(6 * 2 * 2 * (frequency_count or 1), dtype=torch.float32)
        ).reshape(prefix + (6, 2, 2)),
    }


def _with_ghosts(field, *, storage_slice, global_slice, local_x):
    shape = list(field.shape)
    x_axis = field.ndim - 3
    shape[x_axis] = local_x
    local = torch.full(shape, _SENTINEL, dtype=field.dtype)
    source = [slice(None)] * field.ndim
    destination = [slice(None)] * field.ndim
    source[x_axis] = global_slice
    destination[x_axis] = storage_slice
    local[tuple(destination)] = field[tuple(source)]
    return local


def _local_fields(global_fields, rank):
    layout = _LAYOUTS[rank]
    result = {}
    for name, field in global_fields.items():
        is_cell = name == "Ex"
        storage = layout.storage_cell_owned if is_cell else layout.storage_node_owned
        global_slice = layout.global_cell_owned if is_cell else layout.global_node_owned
        local_x = 4 if rank == 0 or not is_cell else 3
        result[name] = _with_ghosts(
            field,
            storage_slice=storage,
            global_slice=global_slice,
            local_x=local_x,
        )
    return result


def _distributed_solver(global_fields, *, dft=True, frequencies=_FREQUENCIES):
    shards = []
    local_solvers = []
    for rank in range(2):
        fields = _local_fields(global_fields, rank)
        if dft:
            metadata = (
                torch.tensor(frequencies, dtype=torch.float64)
                if rank == 0
                else frequencies
            )
            local_solver = _DftSolver(fields, metadata)
        else:
            local_solver = SimpleNamespace(dft_enabled=False, **fields)
        local_solvers.append(local_solver)
        shards.append(
            SimpleNamespace(
                rank=rank,
                device=f"cuda:{rank}",
                layout=_LAYOUTS[rank],
                solver=local_solver,
            )
        )
    root = SimpleNamespace(frequency=1.5e9, shards=tuple(reversed(shards)))
    return root, tuple(local_solvers)


def _assert_owned_fields(artifacts, global_fields, frequencies):
    assert tuple(artifact.rank for artifact in artifacts) == (0, 1)
    assert all(artifact.frequencies == frequencies for artifact in artifacts)
    for name, expected in global_fields.items():
        components = [
            next(component for component in artifact.components if component.name == name)
            for artifact in artifacts
        ]
        x_axis = expected.ndim - 3
        gathered = torch.cat([component.tensor for component in components], dim=x_axis)
        torch.testing.assert_close(gathered, expected)
        assert all(component.tensor.is_contiguous() for component in components)
        assert all(not torch.any(component.tensor == _SENTINEL) for component in components)


def test_export_last_step_fields_crops_halos_and_uses_yee_owned_slices():
    fields = _global_fields()
    solver, _locals = _distributed_solver(fields, dft=False)

    artifacts = export_distributed_field_shards(solver)

    _assert_owned_fields(artifacts, fields, (solver.frequency,))
    assert tuple(artifact.components[0].global_x_slice for artifact in artifacts) == (
        (0, 3),
        (3, 5),
    )
    assert tuple(artifact.components[1].global_x_slice for artifact in artifacts) == (
        (0, 3),
        (3, 6),
    )
    assert tuple(artifact.components[2].global_x_slice for artifact in artifacts) == (
        (0, 3),
        (3, 6),
    )


def test_export_multifrequency_dft_crops_x_axis_after_frequency_axis():
    fields = _global_fields(frequency_count=len(_FREQUENCIES))
    solver, local_solvers = _distributed_solver(fields)

    artifacts = export_distributed_field_shards(solver)

    _assert_owned_fields(artifacts, fields, _FREQUENCIES)
    assert all(
        component.x_axis == 1
        for artifact in artifacts
        for component in artifact.components
    )
    assert all(local_solver.calls == [True] for local_solver in local_solvers)


def test_export_rejects_mixed_dft_and_last_step_shards_before_reading_fields():
    fields = _global_fields(frequency_count=len(_FREQUENCIES))
    solver, local_solvers = _distributed_solver(fields)
    local_solvers[1].dft_enabled = False

    with pytest.raises(RuntimeError, match="mixes DFT and last-step"):
        export_distributed_field_shards(solver)
    assert all(local_solver.calls == [] for local_solver in local_solvers)


def test_export_rejects_inconsistent_dft_frequencies():
    fields = _global_fields(frequency_count=len(_FREQUENCIES))
    solver, local_solvers = _distributed_solver(fields)
    local_solvers[1]._frequencies = (0.8e9, 1.3e9)

    with pytest.raises(RuntimeError, match=r"rank 1 DFT frequencies.*do not match"):
        export_distributed_field_shards(solver)
