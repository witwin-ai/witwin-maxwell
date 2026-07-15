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
        device="cpu",
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


def _result(artifacts, *, fields=None) -> Result:
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
                "data": monitor_data,
            }
        },
        metadata={"run": "uneven-multifrequency"},
        solver_stats={"parallel_stats": {"devices": ("cuda:0", "cuda:1")}},
    )


def _save_fixture(tmp_path):
    fields = _global_fields()
    directory = tmp_path / "sharded-result"
    manifest = _result(
        _artifacts(fields),
        fields={name.upper(): tensor for name, tensor in fields.items()},
    ).save_sharded(directory)
    return directory, manifest, fields


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

    lazy = Result.load_sharded(directory, scene=_scene())
    assert lazy.is_sharded is True
    assert lazy.fields == {}
    assert tuple(path.name for path in lazy.shard_paths) == (
        "rank-0000.pt",
        "rank-0001.pt",
        "rank-0002.pt",
    )
    assert tuple(shard.rank for shard in lazy.sharded_manifest.shards) == (0, 1, 2)
    assert tuple(lazy.monitors) == ("probe",)

    gathered = Result.load_sharded(
        directory,
        scene=_scene(),
        gather_fields=True,
        map_location="cpu",
    )
    assert tuple(gathered.fields) == ("EX", "EY", "EZ")
    for name, reference in fields.items():
        torch.testing.assert_close(gathered.fields[name.upper()], reference)
    torch.testing.assert_close(gathered.tensor("EX", freq_index=1), fields["Ex"][1])
    assert gathered.solver_stats["parallel_stats"]["devices"] == (
        "cuda:0",
        "cuda:1",
    )


def test_lazy_load_does_not_deserialize_existing_corrupt_rank_until_gather(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    corrupt = directory / "rank-0001.pt"
    corrupt.write_bytes(b"not a torch shard")

    lazy = Result.load_sharded(directory, scene=_scene(), gather_fields=False)
    assert lazy.fields == {}
    assert lazy.shard_paths[1] == corrupt

    with pytest.raises(ValueError, match=r"rank 1 file.*Failed|rank 1 file"):
        Result.load_sharded(directory, scene=_scene(), gather_fields=True)


def test_missing_rank_is_reported_before_lazy_result_is_returned(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    missing = directory / "rank-0002.pt"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=r"rank 2 file is missing"):
        Result.load_sharded(directory, scene=_scene())


def test_corrupt_manifest_is_rejected(tmp_path):
    directory, _manifest, _fields = _save_fixture(tmp_path)
    (directory / "manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid sharded Result manifest"):
        Result.load_sharded(directory, scene=_scene())


def test_save_sharded_requires_export_field_shards_provider(tmp_path):
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequency=1.0e9,
        solver=object(),
    )

    with pytest.raises(RuntimeError, match=r"requires solver\.export_field_shards"):
        result.save_sharded(tmp_path / "missing-exporter")


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

    directory = tmp_path / "gap"
    with pytest.raises(ValueError, match="leaves a gap"):
        _result(tuple(artifacts)).save_sharded(directory)
    assert directory.exists() is False


def test_rank_tensors_are_detached_and_written_on_cpu(
    tmp_path,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    global_ex = torch.arange(
        2 * 6 * 2 * 2,
        dtype=torch.float32,
        device=cuda_p2p_devices[0],
    ).reshape(2, 6, 2, 2)
    artifacts = (
        {
            "rank": 1,
            "device": "cuda:1",
            "frequencies": _FREQUENCIES,
            "components": {
                "Ex": {
                    "tensor": global_ex[:, 3:].to(cuda_p2p_devices[1]),
                    "global_x_slice": (3, 6),
                }
            },
        },
        {
            "rank": 0,
            "device": "cuda:0",
            "frequencies": _FREQUENCIES,
            "components": {
                "Ex": {
                    "tensor": global_ex[:, :3].requires_grad_(),
                    "global_x_slice": (0, 3),
                }
            },
        },
    )
    directory = tmp_path / "cuda-shards"

    _result(artifacts).save_sharded(directory)

    for rank in range(2):
        payload = torch.load(directory / f"rank-{rank:04d}.pt", weights_only=True)
        tensor = payload["components"]["Ex"]["tensor"]
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False
    loaded = Result.load_sharded(directory, scene=_scene(), gather_fields=True)
    torch.testing.assert_close(loaded.fields["EX"], global_ex.cpu())
