from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from witwin.maxwell.fdtd.cuda import backend
from witwin.maxwell.fdtd.distributed import solver as distributed_solver
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan


class _RecordedLaunch:
    def __init__(self, calls, name: str, kwargs: dict) -> None:
        self._calls = calls
        self._name = name
        self._kwargs = kwargs

    def launchRaw(self):  # noqa: N802 - mirrors the CUDA backend launch API
        self._calls.append((self._name, self._kwargs))


class _RecordingModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        def bind(**kwargs):
            return _RecordedLaunch(self.calls, name, kwargs)

        return bind


def _sentinel_solver(module: _RecordingModule):
    names = (
        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "chx_decay", "chx_curl", "chy_decay", "chy_curl", "chz_decay", "chz_curl",
        "inv_dx_e", "inv_dy_e", "inv_dz_e", "inv_dx_h", "inv_dy_h", "inv_dz_h",
    )
    return SimpleNamespace(
        fdtd_module=module,
        boundary_y_low_code=11,
        boundary_y_high_code=12,
        boundary_z_low_code=13,
        boundary_z_high_code=14,
        **{name: object() for name in names},
    )


def test_bounded_helpers_pass_full_allocations_and_component_coordinates(monkeypatch):
    plan = FDTDPartitionPlan(
        global_shape=(13, 7, 6),
        devices=("cuda:0", "cuda:1", "cuda:2"),
    )
    layout = plan.layout(1)
    module = _RecordingModule()
    solver = _sentinel_solver(module)
    electric_decay = (object(), object(), object())
    electric_curl = (object(), object(), object())
    monkeypatch.setattr(
        distributed_solver,
        "_electric_coefficients",
        lambda _solver: (electric_decay, electric_curl),
    )

    node_owned = layout.storage_node_owned
    cell_owned = layout.storage_cell_owned
    distributed_solver._launch_magnetic_hx(solver, layout, node_owned)
    distributed_solver._launch_magnetic_hy_hz(solver, layout, cell_owned)
    distributed_solver._launch_electric_ex(solver, layout, cell_owned)
    distributed_solver._launch_electric_ey_ez(
        solver,
        layout,
        node_owned,
        x_low_mode=21,
        x_high_mode=22,
    )

    expected = (
        ("updateMagneticFieldHxStandardBounded3D", "Hx", node_owned),
        ("updateMagneticFieldHyStandardBounded3D", "Hy", cell_owned),
        ("updateMagneticFieldHzStandardBounded3D", "Hz", cell_owned),
        ("updateElectricFieldExStandardBounded3D", "Ex", cell_owned),
        ("updateElectricFieldEyStandardBounded3D", "Ey", node_owned),
        ("updateElectricFieldEzStandardBounded3D", "Ez", node_owned),
    )
    assert tuple(name for name, _ in module.calls) == tuple(item[0] for item in expected)
    for (name, field_name, x_slice), (actual_name, kwargs) in zip(expected, module.calls):
        assert actual_name == name
        assert kwargs[field_name] is getattr(solver, field_name)
        component_layout = layout.component(field_name)
        assert kwargs["localXBegin"] == x_slice.start
        assert kwargs["localXEnd"] == x_slice.stop
        assert kwargs["globalXOffset"] == component_layout.global_origin[0]
        assert kwargs["globalXExtent"] == component_layout.global_shape[0]

    assert module.calls[3][1]["ExDecay"] is electric_decay[0]
    assert module.calls[4][1]["EyCurl"] is electric_curl[1]
    assert module.calls[5][1]["EzCurl"] is electric_curl[2]
    assert module.calls[4][1]["xLowBoundaryMode"] == 21
    assert module.calls[4][1]["xHighBoundaryMode"] == 22

    before = len(module.calls)
    distributed_solver._launch_magnetic_hx(solver, layout, slice(2, 2))
    distributed_solver._launch_electric_ex(solver, layout, slice(3, 3))
    assert len(module.calls) == before


def test_bounded_operators_have_schemas_and_backend_bindings():
    op_names = (
        "update_magnetic_hx_standard_bounded",
        "update_magnetic_hy_standard_bounded",
        "update_magnetic_hz_standard_bounded",
        "update_electric_ex_standard_bounded",
        "update_electric_ey_standard_bounded",
        "update_electric_ez_standard_bounded",
    )
    public_names = (
        "updateMagneticFieldHxStandardBounded3D",
        "updateMagneticFieldHyStandardBounded3D",
        "updateMagneticFieldHzStandardBounded3D",
        "updateElectricFieldExStandardBounded3D",
        "updateElectricFieldEyStandardBounded3D",
        "updateElectricFieldEzStandardBounded3D",
    )
    source = Path(backend.__file__).with_name("extension.cpp").read_text(encoding="utf-8")
    for name in op_names:
        schema = source.split(f'_({name}, "', maxsplit=1)[1].split('"', maxsplit=1)[0]
        assert "int local_x_begin, int local_x_end" in schema
        assert "int global_x_offset, int global_x_extent" in schema
    assert all(name in backend._KERNELS for name in public_names)


class _RecordingStream:
    def __init__(self, rank: int, events: list[tuple]) -> None:
        self.rank = rank
        self.events = events

    def wait_event(self, event) -> None:
        self.events.append(("wait", self.rank, event))


class _RecordingTransport:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def exchange_electric(self, shards) -> None:
        self.events.append(("exchange_electric",))

    def exchange_magnetic(self, shards) -> None:
        self.events.append(("exchange_magnetic",))


def _assert_exact_owned_partition(boxes: list[slice], owned: slice) -> None:
    assert boxes
    assert boxes[0].start == owned.start
    assert boxes[-1].stop == owned.stop
    assert all(box.start >= owned.start and box.stop <= owned.stop for box in boxes)
    assert all(left.stop == right.start for left, right in zip(boxes, boxes[1:]))


def test_overlap_schedules_wait_between_disjoint_owned_launch_boxes(monkeypatch):
    from contextlib import nullcontext

    plan = FDTDPartitionPlan(global_shape=(13, 7, 6), devices=("cuda:0", "cuda:1"))
    events: list[tuple] = []
    shards = []
    for rank, layout in enumerate(plan.shard_layouts):
        solver = SimpleNamespace(boundary_x_low_code=31, boundary_x_high_code=32)
        shards.append(
            SimpleNamespace(
                rank=rank,
                layout=layout,
                solver=solver,
                device=layout.device,
                compute_stream=_RecordingStream(rank, events),
                electric_received=f"electric-{rank}",
                magnetic_received=f"magnetic-{rank}",
                is_first=rank == 0,
                is_last=rank == len(plan.shard_layouts) - 1,
            )
        )
    coordinator = distributed_solver.DistributedFDTD.__new__(
        distributed_solver.DistributedFDTD
    )
    coordinator.shards = tuple(shards)
    coordinator.transport = _RecordingTransport(events)

    monkeypatch.setattr(distributed_solver.torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(distributed_solver.torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(
        distributed_solver,
        "_launch_magnetic_hx",
        lambda solver, layout, box: events.append(("hx", layout.rank, box)),
    )
    monkeypatch.setattr(
        distributed_solver,
        "_launch_magnetic_hy_hz",
        lambda solver, layout, box: events.append(("hyhz", layout.rank, box)),
    )
    monkeypatch.setattr(
        distributed_solver,
        "_launch_electric_ex",
        lambda solver, layout, box: events.append(("ex", layout.rank, box)),
    )
    monkeypatch.setattr(
        distributed_solver,
        "_launch_electric_ey_ez",
        lambda solver, layout, box, **_: events.append(("eyez", layout.rank, box)),
    )

    coordinator._advance_magnetic_overlapped()
    nonlast = plan.layout(0)
    h_owned = nonlast.storage_cell_owned
    h_boxes = [event[2] for event in events if event[:2] == ("hyhz", 0)]
    assert h_boxes == [
        slice(h_owned.start, h_owned.stop - 1),
        slice(h_owned.stop - 1, h_owned.stop),
    ]
    _assert_exact_owned_partition(h_boxes, h_owned)
    h_interior = events.index(("hyhz", 0, h_boxes[0]))
    h_wait = events.index(("wait", 0, "electric-0"))
    h_boundary = events.index(("hyhz", 0, h_boxes[1]))
    assert h_interior < h_wait < h_boundary
    assert ("hx", 0, nonlast.storage_node_owned) in events

    events.clear()
    coordinator._advance_electric_overlapped()
    nonfirst = plan.layout(1)
    e_owned = nonfirst.storage_node_owned
    e_boxes = [event[2] for event in events if event[:2] == ("eyez", 1)]
    assert e_boxes == [
        slice(e_owned.start + 1, e_owned.stop),
        slice(e_owned.start, e_owned.start + 1),
    ]
    ordered_boxes = sorted(e_boxes, key=lambda box: box.start)
    _assert_exact_owned_partition(ordered_boxes, e_owned)
    e_interior = events.index(("eyez", 1, e_boxes[0]))
    e_wait = events.index(("wait", 1, "magnetic-1"))
    e_boundary = events.index(("eyez", 1, e_boxes[1]))
    assert e_interior < e_wait < e_boundary
    assert e_owned.start > 0
    assert all(box.start >= e_owned.start for box in e_boxes)
    assert ("ex", 1, nonfirst.storage_cell_owned) in events
