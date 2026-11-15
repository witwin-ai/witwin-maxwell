from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.persistence import (
    FieldComponentArtifact,
    FieldShardArtifact,
)
from witwin.maxwell.result import Result


_FREQUENCIES = (0.8e9, 1.2e9)


def _scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
    )


def _global_fields() -> dict[str, torch.Tensor]:
    return {
        "Ex": torch.arange(2 * 9 * 4 * 5, dtype=torch.float32).reshape(2, 9, 4, 5),
        "Ey": (
            1000
            + torch.arange(2 * 10 * 3 * 5, dtype=torch.float32).reshape(2, 10, 3, 5)
        ),
        "Ez": (
            2000
            + torch.arange(2 * 10 * 4 * 4, dtype=torch.float32).reshape(2, 10, 4, 4)
        ),
    }


def _artifacts(
    fields: dict[str, torch.Tensor],
    *,
    order=(2, 0, 1),
) -> tuple[FieldShardArtifact, ...]:
    cell_slices = ((0, 4), (4, 7), (7, 9))
    node_slices = ((0, 4), (4, 7), (7, 10))
    by_rank = []
    for rank in range(3):
        components = []
        for name in ("Ex", "Ey", "Ez"):
            extent = cell_slices[rank] if name == "Ex" else node_slices[rank]
            components.append(
                FieldComponentArtifact(
                    name=name,
                    tensor=fields[name][:, extent[0] : extent[1]],
                    global_x_slice=extent,
                )
            )
        by_rank.append(
            FieldShardArtifact(
                rank=rank,
                device=f"cuda:{rank % 2}",
                frequencies=_FREQUENCIES,
                components=tuple(components),
            )
        )
    return tuple(by_rank[rank] for rank in order)


def _circuit_data(*, requires_grad=False) -> mw.circuits.CircuitData:
    return mw.circuits.CircuitData(
        circuit_name="network",
        times=torch.tensor([0.0, 1.0], dtype=torch.float64),
        node_names=("0", "signal"),
        node_voltages=torch.tensor(
            [[0.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        branch_names=("R1",),
        branch_currents=torch.tensor(
            [[0.0], [0.02]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        device_powers={
            "R1": torch.tensor([0.0, 0.02], dtype=torch.float64)
        },
        energy_balance=torch.tensor([0.0, 1.0e-12], dtype=torch.float64),
        diagnostics={"condition": torch.tensor([1.0, 1.5], dtype=torch.float64)},
    )


def _result(artifacts, *, fields=None, circuits=None) -> Result:
    solver = SimpleNamespace(export_field_shards=lambda: artifacts)
    monitor_data = torch.tensor((1.0 + 2.0j, 3.0 + 4.0j))
    return Result(
        method="fdtd",
        scene=_scene(),
        frequencies=_FREQUENCIES,
        solver=solver,
        fields=fields or {},
        monitors={
            "probe": {
                "kind": "point",
                "fields": ("Ez",),
                "frequencies": _FREQUENCIES,
                "components": {"Ez": monitor_data},
                "field_indices": {"Ez": (0, 0, 0)},
                "position": (0.0, 0.0, 0.0),
            }
        },
        circuits=circuits,
        metadata={"run": "uneven-multifrequency"},
        solver_stats={"parallel_stats": {"devices": ("cuda:0", "cuda:1")}},
    )


def _save_fixture(tmp_path):
    fields = _global_fields()
    directory = tmp_path / "sharded-result"
    manifest = mw.execution.save_sharded(_result(
        _artifacts(fields),
        fields={name.upper(): tensor for name, tensor in fields.items()},
    ), directory)
    return directory, manifest, fields


def test_mixed_sharded_result_restores_ratio_only_ports(tmp_path):
    port = mw.rf.PortData.from_power_waves(
        port_name="feed", frequencies=torch.tensor(_FREQUENCIES),
        a=torch.tensor([1. + .2j, .8 - .1j]), b=torch.tensor([.2j, .1j]), z0=50.,
    )
    result = Result(
        method="fdtd", scene=_scene(), frequencies=_FREQUENCIES, ports={"feed": port},
        solver=SimpleNamespace(export_field_shards=lambda: _artifacts(_global_fields())),
        solver_stats={"time_plan": {"mixed_sources": True}, "normalize_source": False},
    )
    directory = tmp_path / "mixed-sharded"
    mw.execution.save_sharded(result, directory)
    loaded = mw.execution.load_sharded(directory, scene=_scene())
    from witwin.maxwell.result import _RatioOnlyPortData
    assert isinstance(loaded.port("feed"), _RatioOnlyPortData)
    assert torch.equal(loaded.port("feed").reflection_coefficient, port.reflection_coefficient)
    with pytest.raises(ValueError, match="Absolute spectra of a run mixing ongoing"):
        loaded.port("feed").voltage


def test_sharded_round_trip_sorts_ranks_and_gathers_uneven_multifrequency_fields(
    tmp_path,
):
    directory, manifest, fields = _save_fixture(tmp_path)

    assert tuple(path.name for path in sorted(directory.iterdir())) == (
        "manifest.json",
        "rank-0000.pt",
        "rank-0001.pt",
        "rank-0002.pt",
        "result.pt",
    )
    assert tuple(shard.rank for shard in manifest.shards) == (0, 1, 2)
    assert tuple(component.shape for component in manifest.components) == (
        (2, 9, 4, 5),
        (2, 10, 3, 5),
        (2, 10, 4, 4),
    )
    assert all(component.x_axis == 1 for component in manifest.components)

    encoded = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert [shard["rank"] for shard in encoded["shards"]] == [0, 1, 2]
    assert [
        shard["components"][0]["global_x_slice"] for shard in encoded["shards"]
    ] == [[0, 4], [4, 7], [7, 9]]
    assert [
        shard["components"][1]["global_x_slice"] for shard in encoded["shards"]
    ] == [[0, 4], [4, 7], [7, 10]]

    rank = torch.load(directory / "rank-0001.pt", weights_only=True)
    ex = rank["components"]["Ex"]
    assert ex["tensor"].device.type == "cpu"
    assert ex["dtype"] == "torch.float32"
    assert ex["shape"] == (2, 3, 4, 5)
    assert ex["frequencies"] == _FREQUENCIES
    assert ex["x_axis"] == 1

    lazy = mw.execution.load_sharded(directory, scene=_scene())
    assert mw.execution.is_sharded(lazy) is True
    assert lazy.fields == {}
    assert tuple(path.name for path in mw.execution.shard_paths(lazy)) == (
        "rank-0000.pt",
        "rank-0001.pt",
        "rank-0002.pt",
    )
    assert tuple(shard.rank for shard in mw.execution.sharded_manifest(lazy).shards) == (0, 1, 2)
    assert tuple(lazy.monitors) == ("probe",)

    gathered = mw.execution.load_sharded(
        directory,
        scene=_scene(),
        gather_fields=True,
        map_location="cpu",
    )
    # The synthetic shards hold arbitrary array shapes, not Yee grids, so read the stored arrays.
    for name, reference in fields.items():
        torch.testing.assert_close(gathered._fields[name.upper()], reference)
    torch.testing.assert_close(gathered._fields["EX"][1], fields["Ex"][1])
    assert gathered.stats()["parallel_stats"]["devices"] == (
        "cuda:0",
        "cuda:1",
    )


def test_sharded_v3_stores_circuits_once_in_result_metadata(tmp_path):
    fields = _global_fields()
    data = _circuit_data(requires_grad=True)
    directory = tmp_path / "sharded-circuit"

    mw.execution.save_sharded(_result(
        _artifacts(fields),
        circuits={"network": data},
    ), directory)

    payload = torch.load(directory / "result.pt", weights_only=True)
    assert payload["schema_version"] == 3
    assert tuple(payload["circuits"]) == ("network",)
    assert payload["circuits"]["network"]["schema_version"] == 1
    for rank in range(3):
        rank_payload = torch.load(
            directory / f"rank-{rank:04d}.pt",
            weights_only=True,
        )
        assert "circuits" not in rank_payload

    lazy = mw.execution.load_sharded(directory, scene=_scene())
    restored = mw.circuits.circuit(lazy, "network")
    torch.testing.assert_close(restored.node_voltages, data.node_voltages.detach())
    torch.testing.assert_close(
        restored.diagnostics["condition"],
        data.diagnostics["condition"],
    )
    assert restored.node_voltages.requires_grad is False

    gathered = mw.execution.load_sharded(
        directory,
        scene=_scene(),
        gather_fields=True,
    )
    torch.testing.assert_close(
        mw.circuits.circuit(gathered, "network").branch_currents,
        data.branch_currents.detach(),
    )


def test_sharded_rejects_superseded_v1_and_v2_without_circuits(tmp_path):
    fields = _global_fields()
    directory = tmp_path / "sharded-v1"
    mw.execution.save_sharded(_result(
        _artifacts(fields),
        circuits={"network": _circuit_data()},
    ), directory)
    result_path = directory / "result.pt"
    payload = torch.load(result_path, weights_only=True)

    # Older schemas are refused with SchemaError: there is no backward-support path.
    payload["schema_version"] = 2
    torch.save(payload, result_path)
    with pytest.raises(
        mw.SchemaError,
        match="Unsupported sharded result metadata schema_version=2",
    ):
        mw.execution.load_sharded(directory, scene=_scene())

    payload["schema_version"] = 3
    payload.pop("circuits")
    torch.save(payload, result_path)
    with pytest.raises(mw.SchemaError, match="schema v3 is missing required key: circuits"):
        mw.execution.load_sharded(directory, scene=_scene())


def test_save_sharded_prevalidates_circuits_before_calling_exporter(tmp_path):
    calls = 0

    def exporter():
        nonlocal calls
        calls += 1
        return ()

    data = _circuit_data()
    data.diagnostics["unsafe"] = object()
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequency=1.0e9,
        solver=SimpleNamespace(export_field_shards=exporter),
        circuits={"network": data},
    )
    directory = tmp_path / "unsafe-circuit"

    with pytest.raises(TypeError, match="CircuitData.diagnostics.*object"):
        mw.execution.save_sharded(result, directory)

    assert calls == 0
    assert directory.exists() is False


def test_save_sharded_rejects_existing_destination_before_exporting(tmp_path):
    calls = 0

    def exporter():
        nonlocal calls
        calls += 1
        return ()

    destination = tmp_path / "existing"
    destination.mkdir()
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequency=1.0e9,
        solver=SimpleNamespace(export_field_shards=exporter),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        mw.execution.save_sharded(result, destination)

    assert calls == 0


def test_lazy_load_does_not_deserialize_existing_corrupt_rank_until_gather(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    corrupt = directory / "rank-0001.pt"
    corrupt.write_bytes(b"not a torch shard")

    lazy = mw.execution.load_sharded(directory, scene=_scene(), gather_fields=False)
    assert lazy.fields == {}
    assert mw.execution.shard_paths(lazy)[1] == corrupt

    with pytest.raises(ValueError, match=r"rank 1 file.*Failed|rank 1 file"):
        mw.execution.load_sharded(directory, scene=_scene(), gather_fields=True)


def test_missing_rank_is_reported_before_lazy_result_is_returned(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    missing = directory / "rank-0002.pt"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=r"rank 2 file is missing"):
        mw.execution.load_sharded(directory, scene=_scene())


def test_corrupt_manifest_is_rejected(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    (directory / "manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid sharded Result manifest"):
        mw.execution.load_sharded(directory, scene=_scene())


def test_save_sharded_requires_export_field_shards_provider(tmp_path):
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequency=1.0e9,
        solver=object(),
    )

    with pytest.raises(RuntimeError, match=r"requires solver\.export_field_shards"):
        mw.execution.save_sharded(result, tmp_path / "missing-exporter")


def test_noncontiguous_owned_intervals_are_rejected_before_publication(tmp_path):
    fields = _global_fields()
    artifacts = list(_artifacts(fields, order=(0, 1, 2)))
    broken = artifacts[1]
    components = list(broken.components)
    components[0] = FieldComponentArtifact(
        name="Ex",
        tensor=fields["Ex"][:, 5:8],
        global_x_slice=(5, 8),
    )
    artifacts[1] = FieldShardArtifact(
        rank=broken.rank,
        device=broken.device,
        frequencies=broken.frequencies,
        components=tuple(components),
    )

    directory = tmp_path / "nested" / "gap"
    with pytest.raises(ValueError, match="leaves a gap"):
        mw.execution.save_sharded(_result(tuple(artifacts)), directory)
    assert directory.exists() is False
    assert directory.parent.exists() is False


def test_rank_tensors_are_detached_and_written_on_cpu(
    tmp_path,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    # Full EX Yee-grid shape of the 10 x 4 x 4 cell scene: (Nx, Ny + 1, Nz + 1).
    global_ex = torch.arange(
        2 * 10 * 5 * 5,
        dtype=torch.float32,
        device=cuda_p2p_devices[0],
    ).reshape(2, 10, 5, 5)
    artifacts = (
        {
            "rank": 1,
            "device": "cuda:1",
            "frequencies": _FREQUENCIES,
            "components": {
                "Ex": {
                    "tensor": global_ex[:, 5:].to(cuda_p2p_devices[1]),
                    "global_x_slice": (5, 10),
                }
            },
        },
        {
            "rank": 0,
            "device": "cuda:0",
            "frequencies": _FREQUENCIES,
            "components": {
                "Ex": {
                    "tensor": global_ex[:, :5].requires_grad_(),
                    "global_x_slice": (0, 5),
                }
            },
        },
    )
    directory = tmp_path / "cuda-shards"

    mw.execution.save_sharded(_result(artifacts), directory)

    for rank in range(2):
        payload = torch.load(directory / f"rank-{rank:04d}.pt", weights_only=True)
        tensor = payload["components"]["Ex"]["tensor"]
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False
    loaded = mw.execution.load_sharded(directory, scene=_scene(), gather_fields=True)
    torch.testing.assert_close(loaded.fields["EX"].values, global_ex.cpu())
