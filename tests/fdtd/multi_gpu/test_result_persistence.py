from __future__ import annotations

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.result import Result


_FREQUENCIES = (0.8e9, 1.2e9)


def _scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )


def _parallel_stats(device: torch.device) -> dict:
    return {
        "parallel_stats": {
            "devices": ("cuda:0", "cuda:1"),
            "partition_extents": ((0, 4), (4, 8)),
            "transport": "cuda_p2p",
            "halo_bytes_total": 4096,
            "communication_time_s": 0.01,
            "compute_time_s": 0.04,
            "peak_memory_bytes": {"cuda:0": 1024, "cuda:1": 1088},
            "device_counter": torch.tensor((7, 9), device=device),
        }
    }


def test_monitor_first_result_round_trip_preserves_order_and_stats_without_runtime_state(
    tmp_path,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    coords = np.asarray((-0.3, -0.1, 0.1, 0.3), dtype=np.float64)
    left = torch.tensor((1.0 + 2.0j, 3.0 + 4.0j), device=cuda_p2p_devices[0])
    right = torch.tensor((5.0 + 6.0j, 7.0 + 8.0j), device=cuda_p2p_devices[1])
    monitors = {
        "right_declared_first": {
            "kind": "point",
            "monitor_type": "point",
            "fields": ("Ez",),
            "frequencies": _FREQUENCIES,
            "data": right,
            "components": {"Ez": right},
        },
        "plane_declared_second": {
            "kind": "plane",
            "monitor_type": "plane",
            "fields": ("Ez",),
            "frequencies": _FREQUENCIES,
            "axis": "y",
            "x": coords,
            "z": np.asarray((-0.1, 0.1), dtype=np.float64),
            "data": left[:, None, None].expand(-1, coords.size, 2),
        },
    }
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=_FREQUENCIES,
        fields={},
        monitors=monitors,
        metadata={"rank_metric": right.real},
        solver_stats=_parallel_stats(cuda_p2p_devices[1]),
        solver=lambda: None,
        raw_output={"transport_handle": lambda: None},
    )

    path = tmp_path / "monitor-first.pt"
    result.save(path)
    loaded = Result.load(path, scene=_scene())

    assert loaded.fields == {}
    assert tuple(loaded.monitors) == (
        "right_declared_first",
        "plane_declared_second",
    )
    torch.testing.assert_close(loaded.monitors["right_declared_first"]["data"], right.cpu())
    np.testing.assert_array_equal(loaded.monitors["plane_declared_second"]["x"], coords)
    assert loaded.monitors["plane_declared_second"]["data"].device.type == "cpu"
    stats = loaded.solver_stats["parallel_stats"]
    assert stats["devices"] == ("cuda:0", "cuda:1")
    assert stats["partition_extents"] == ((0, 4), (4, 8))
    assert stats["halo_bytes_total"] == 4096
    assert stats["device_counter"].device.type == "cpu"
    torch.testing.assert_close(stats["device_counter"], torch.tensor((7, 9)))
    assert loaded._metadata["rank_metric"].device.type == "cpu"
    assert loaded.solver is None
    assert loaded.raw_output is None


def test_gathered_field_result_round_trip_preserves_frequency_selection_on_cpu(
    tmp_path,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    ex = torch.arange(48, device=cuda_p2p_devices[1], dtype=torch.float32).reshape(
        2,
        4,
        3,
        2,
    )
    ez = (ex + 1.0).to(torch.complex64)
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=_FREQUENCIES,
        fields={"EX": ex, "EZ": ez},
        monitors={},
        solver_stats=_parallel_stats(cuda_p2p_devices[0]),
    )

    path = tmp_path / "gathered-fields.pt"
    result.save(path)
    loaded = Result.load(path, scene=_scene())

    assert loaded.frequencies == _FREQUENCIES
    assert tuple(loaded.fields) == ("EX", "EZ")
    assert all(tensor.device.type == "cpu" for tensor in loaded.fields.values())
    torch.testing.assert_close(loaded.tensor("EX", freq_index=1), ex[1].cpu())
    torch.testing.assert_close(loaded.tensor("EZ", frequency=_FREQUENCIES[0]), ez[0].cpu())
    assert loaded.stats()["num_fields"] == 2
    assert loaded.stats()["num_monitors"] == 0
