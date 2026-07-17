from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_SCHEMA = "witwin.maxwell.fdtd.sharded-result"
_FORMAT_VERSION = 1
_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")


def _component_name(value: str) -> str:
    text = str(value).strip()
    normalized = text[0].upper() + text[1:].lower() if text else text
    if normalized not in _ELECTRIC_COMPONENTS:
        choices = ", ".join(_ELECTRIC_COMPONENTS)
        raise ValueError(
            f"Sharded Result component must be one of {choices}, got {value!r}."
        )
    return normalized


def _half_open_slice(value, *, label: str) -> tuple[int, int]:
    if isinstance(value, slice):
        if value.step not in (None, 1):
            raise ValueError(f"{label} must have unit stride.")
        start, stop = value.start, value.stop
    else:
        try:
            start, stop = value
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be a (start, stop) pair or slice.") from exc
    if isinstance(start, bool) or isinstance(stop, bool):
        raise TypeError(f"{label} endpoints must be integers.")
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise ValueError(
            f"{label} must be a nonempty half-open interval, got ({start}, {stop})."
        )
    return start, stop


def _frequencies(values) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("Sharded Result frequencies must not be empty.")
    return result


def _cpu_data(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_data(item) for item in value)
    return value


@dataclass(frozen=True)
class FieldComponentArtifact:
    """Owned-only tensor and global x interval for one Yee E component."""

    name: str
    tensor: torch.Tensor
    global_x_slice: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _component_name(self.name))
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"{self.name} shard tensor must be a torch.Tensor.")
        if self.tensor.ndim < 3:
            raise ValueError(
                f"{self.name} shard tensor must have at least three field dimensions."
            )
        extent = _half_open_slice(
            self.global_x_slice,
            label=f"{self.name} global_x_slice",
        )
        object.__setattr__(self, "global_x_slice", extent)
        x_axis = self.tensor.ndim - 3
        if int(self.tensor.shape[x_axis]) != extent[1] - extent[0]:
            raise ValueError(
                f"{self.name} tensor x size {self.tensor.shape[x_axis]} does not match "
                f"owned interval {extent}."
            )

    @property
    def x_axis(self) -> int:
        return self.tensor.ndim - 3


@dataclass(frozen=True)
class FieldShardArtifact:
    """Pure-data export produced by ``solver.export_field_shards()``."""

    rank: int
    components: tuple[FieldComponentArtifact, ...]
    frequencies: tuple[float, ...]
    device: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or int(self.rank) < 0:
            raise ValueError(f"Shard rank must be a nonnegative integer, got {self.rank!r}.")
        object.__setattr__(self, "rank", int(self.rank))
        components = tuple(self.components)
        if not components:
            raise ValueError(f"Shard rank {self.rank} has no field components.")
        if len({component.name for component in components}) != len(components):
            raise ValueError(f"Shard rank {self.rank} contains duplicate field components.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "frequencies", _frequencies(self.frequencies))
        if self.device is not None:
            object.__setattr__(self, "device", str(self.device))


def _export_frequency_values(values: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, (int, float)):
        values = (values,)
    try:
        return _frequencies(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain one or more numeric values.") from exc


def _dft_frequencies(
    local_solver: Any,
    output: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[float, ...]:
    if "frequencies" in output:
        values = output["frequencies"]
    else:
        values = getattr(local_solver, "dft_frequencies", ())
        if not values:
            frequency = getattr(local_solver, "dft_frequency", None)
            values = () if frequency is None else (frequency,)
    return _export_frequency_values(values, label=f"Shard rank {rank} DFT frequencies")


def _owned_component(
    *,
    name: str,
    tensor: Any,
    layout: Any,
) -> FieldComponentArtifact:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Shard-local {name} field must be a torch.Tensor.")
    if tensor.ndim < 3:
        raise ValueError(f"Shard-local {name} field must have at least three dimensions.")
    is_cell = name == "Ex"
    storage_name = "storage_cell_owned" if is_cell else "storage_node_owned"
    global_name = "global_cell_owned" if is_cell else "global_node_owned"
    storage = _half_open_slice(
        getattr(layout, storage_name),
        label=f"{name} {storage_name}",
    )
    global_extent = _half_open_slice(
        getattr(layout, global_name),
        label=f"{name} {global_name}",
    )
    if storage[1] - storage[0] != global_extent[1] - global_extent[0]:
        raise ValueError(
            f"{name} storage owned interval {storage} and global owned interval "
            f"{global_extent} have different widths."
        )
    x_axis = tensor.ndim - 3
    if storage[1] > int(tensor.shape[x_axis]):
        raise ValueError(
            f"{name} storage owned interval {storage} exceeds local x size "
            f"{tensor.shape[x_axis]}."
        )
    index = [slice(None)] * tensor.ndim
    index[x_axis] = slice(*storage)
    return FieldComponentArtifact(
        name=name,
        tensor=tensor[tuple(index)].contiguous(),
        global_x_slice=global_extent,
    )


def export_distributed_field_shards(solver: Any) -> tuple[FieldShardArtifact, ...]:
    """Export owned Ex/Ey/Ez regions from a distributed solver without gathering."""

    shards = tuple(getattr(solver, "shards", ()))
    if not shards:
        raise ValueError("Distributed solver has no shards to export.")
    ordered = tuple(sorted(shards, key=lambda shard: int(shard.rank)))
    modes = tuple(bool(shard.solver.dft_enabled) for shard in ordered)
    if len(set(modes)) != 1:
        raise RuntimeError("Shard-local field export mixes DFT and last-step modes.")

    expected_frequencies: tuple[float, ...] | None = None
    if not modes[0]:
        expected_frequencies = _export_frequency_values(
            (solver.frequency,),
            label="Distributed solver frequency",
        )

    artifacts = []
    for shard in ordered:
        rank = int(shard.rank)
        local_solver = shard.solver
        if modes[0]:
            output = local_solver.get_frequency_solution(all_frequencies=True)
            if not isinstance(output, Mapping):
                raise TypeError(f"Shard rank {rank} DFT solution must be a mapping.")
            frequencies = _dft_frequencies(local_solver, output, rank=rank)
            if expected_frequencies is None:
                expected_frequencies = frequencies
            elif frequencies != expected_frequencies:
                raise RuntimeError(
                    f"Shard rank {rank} DFT frequencies {frequencies} do not match "
                    f"{expected_frequencies}."
                )
            fields = output
        else:
            frequencies = expected_frequencies
            fields = {
                name: getattr(local_solver, name) for name in _ELECTRIC_COMPONENTS
            }

        components = []
        for name in _ELECTRIC_COMPONENTS:
            if name not in fields:
                raise KeyError(f"Shard rank {rank} field export is missing {name}.")
            component = _owned_component(
                name=name,
                tensor=fields[name],
                layout=shard.layout,
            )
            _validate_frequency_axis(component, frequencies)
            components.append(component)
        artifacts.append(
            FieldShardArtifact(
                rank=rank,
                components=tuple(components),
                frequencies=frequencies,
                device=getattr(shard, "device", None),
            )
        )
    return tuple(artifacts)


@dataclass(frozen=True)
class FieldComponentManifest:
    name: str
    global_x_slice: tuple[int, int]
    shape: tuple[int, ...]
    dtype: str
    x_axis: int
    frequencies: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "global_x_slice": self.global_x_slice,
            "shape": self.shape,
            "dtype": self.dtype,
            "x_axis": self.x_axis,
            "frequencies": self.frequencies,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FieldComponentManifest":
        return cls(
            name=_component_name(payload["name"]),
            global_x_slice=_half_open_slice(
                payload["global_x_slice"],
                label=f"{payload['name']} manifest global_x_slice",
            ),
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
            x_axis=int(payload["x_axis"]),
            frequencies=_frequencies(payload["frequencies"]),
        )


@dataclass(frozen=True)
class FieldShardManifest:
    rank: int
    file: str
    device: str
    components: tuple[FieldComponentManifest, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "file": self.file,
            "device": self.device,
            "components": tuple(component.to_dict() for component in self.components),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FieldShardManifest":
        rank = int(payload["rank"])
        expected_file = f"rank-{rank:04d}.pt"
        file_name = str(payload["file"])
        if file_name != expected_file:
            raise ValueError(
                f"Shard rank {rank} file must be {expected_file!r}, got {file_name!r}."
            )
        return cls(
            rank=rank,
            file=file_name,
            device=str(payload["device"]),
            components=tuple(
                FieldComponentManifest.from_dict(component)
                for component in payload["components"]
            ),
        )


@dataclass(frozen=True)
class GlobalFieldManifest:
    name: str
    shape: tuple[int, ...]
    dtype: str
    x_axis: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "x_axis": self.x_axis,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalFieldManifest":
        return cls(
            name=_component_name(payload["name"]),
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
            x_axis=int(payload["x_axis"]),
        )


@dataclass(frozen=True)
class DistributedResultManifest:
    schema: str
    format_version: int
    result_file: str
    frequencies: tuple[float, ...]
    components: tuple[GlobalFieldManifest, ...]
    shards: tuple[FieldShardManifest, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "result_file": self.result_file,
            "frequencies": self.frequencies,
            "components": tuple(component.to_dict() for component in self.components),
            "shards": tuple(shard.to_dict() for shard in self.shards),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DistributedResultManifest":
        schema = str(payload["schema"])
        version = int(payload["format_version"])
        result_file = str(payload["result_file"])
        if schema != _SCHEMA:
            raise ValueError(f"Unsupported sharded Result schema {schema!r}.")
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported sharded Result format_version={version}.")
        if result_file != "result.pt":
            raise ValueError("Sharded Result manifest result_file must be 'result.pt'.")
        manifest = cls(
            schema=schema,
            format_version=version,
            result_file=result_file,
            frequencies=_frequencies(payload["frequencies"]),
            components=tuple(
                GlobalFieldManifest.from_dict(component)
                for component in payload["components"]
            ),
            shards=tuple(
                FieldShardManifest.from_dict(shard) for shard in payload["shards"]
            ),
        )
        _validate_manifest_structure(manifest)
        return manifest

    @classmethod
    def read(cls, directory: str | Path) -> "DistributedResultManifest":
        path = Path(directory) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Sharded Result manifest is missing: {path}.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("manifest root is not a mapping")
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid sharded Result manifest {path}: {exc}") from exc

    def shard_paths(self, directory: str | Path) -> tuple[Path, ...]:
        root = Path(directory)
        return tuple(root / shard.file for shard in self.shards)


@dataclass(frozen=True)
class LoadedShardedResult:
    manifest: DistributedResultManifest
    result_payload: dict[str, Any]
    shard_paths: tuple[Path, ...]
    fields: dict[str, torch.Tensor]


def _coerce_component(name: str, value: Any) -> FieldComponentArtifact:
    if isinstance(value, FieldComponentArtifact):
        if _component_name(name) != value.name:
            raise ValueError(
                f"Component mapping key {name!r} does not match artifact {value.name!r}."
            )
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Shard component {name!r} must be a mapping or artifact.")
    return FieldComponentArtifact(
        name=name,
        tensor=value["tensor"],
        global_x_slice=value["global_x_slice"],
    )


def _coerce_shard(value: Any) -> FieldShardArtifact:
    if isinstance(value, FieldShardArtifact):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("export_field_shards() entries must be mappings or FieldShardArtifact.")
    components_value = value["components"]
    if isinstance(components_value, Mapping):
        components = tuple(
            _coerce_component(name, component)
            for name, component in components_value.items()
        )
    else:
        components = tuple(components_value)
    return FieldShardArtifact(
        rank=value["rank"],
        device=value.get("device"),
        frequencies=value["frequencies"],
        components=components,
    )


def _validate_frequency_axis(
    component: FieldComponentArtifact,
    frequencies: tuple[float, ...],
) -> None:
    if len(frequencies) <= 1:
        return
    if component.tensor.ndim < 4 or int(component.tensor.shape[0]) != len(frequencies):
        raise ValueError(
            f"{component.name} multi-frequency tensor shape {tuple(component.tensor.shape)} "
            f"does not start with frequency count {len(frequencies)}."
        )


def _build_manifest(
    shards: Iterable[FieldShardArtifact | Mapping[str, Any]],
    *,
    frequencies: tuple[float, ...],
) -> tuple[DistributedResultManifest, tuple[FieldShardArtifact, ...]]:
    expected_frequencies = _frequencies(frequencies)
    ordered = tuple(
        sorted(
            (_coerce_shard(shard) for shard in shards),
            key=lambda item: item.rank,
        )
    )
    if not ordered:
        raise ValueError("export_field_shards() returned no shard artifacts.")
    expected_ranks = tuple(range(len(ordered)))
    ranks = tuple(shard.rank for shard in ordered)
    if ranks != expected_ranks:
        raise ValueError(f"Shard ranks must be contiguous from zero, got {ranks}.")

    component_names = tuple(component.name for component in ordered[0].components)
    for shard in ordered:
        if shard.frequencies != expected_frequencies:
            raise ValueError(
                f"Shard rank {shard.rank} frequencies {shard.frequencies} do not match "
                f"Result frequencies {expected_frequencies}."
            )
        names = tuple(component.name for component in shard.components)
        if names != component_names:
            raise ValueError(
                f"Shard rank {shard.rank} component order {names} does not match "
                f"rank 0 order {component_names}."
            )

    global_components = []
    for component_index, name in enumerate(component_names):
        entries = tuple(shard.components[component_index] for shard in ordered)
        first = entries[0]
        _validate_frequency_axis(first, expected_frequencies)
        expected_start = 0
        for shard, component in zip(ordered, entries):
            _validate_frequency_axis(component, expected_frequencies)
            start, stop = component.global_x_slice
            if start != expected_start:
                relation = "overlaps" if start < expected_start else "leaves a gap before"
                raise ValueError(
                    f"Shard rank {shard.rank} {name} interval {(start, stop)} {relation} "
                    f"global x={expected_start}."
                )
            if component.x_axis != first.x_axis:
                raise ValueError(f"Shard rank {shard.rank} {name} x axis is inconsistent.")
            if component.tensor.dtype != first.tensor.dtype:
                raise ValueError(f"Shard rank {shard.rank} {name} dtype is inconsistent.")
            for axis, (actual, reference) in enumerate(
                zip(component.tensor.shape, first.tensor.shape)
            ):
                if axis != first.x_axis and int(actual) != int(reference):
                    raise ValueError(
                        f"Shard rank {shard.rank} {name} non-x shape is inconsistent."
                    )
            expected_start = stop
        global_shape = list(first.tensor.shape)
        global_shape[first.x_axis] = expected_start
        global_components.append(
            GlobalFieldManifest(
                name=name,
                shape=tuple(int(value) for value in global_shape),
                dtype=str(first.tensor.dtype),
                x_axis=first.x_axis,
            )
        )

    shard_manifests = []
    for shard in ordered:
        source_device = shard.device or str(shard.components[0].tensor.device)
        shard_manifests.append(
            FieldShardManifest(
                rank=shard.rank,
                file=f"rank-{shard.rank:04d}.pt",
                device=source_device,
                components=tuple(
                    FieldComponentManifest(
                        name=component.name,
                        global_x_slice=component.global_x_slice,
                        shape=tuple(int(value) for value in component.tensor.shape),
                        dtype=str(component.tensor.dtype),
                        x_axis=component.x_axis,
                        frequencies=expected_frequencies,
                    )
                    for component in shard.components
                ),
            )
        )

    manifest = DistributedResultManifest(
        schema=_SCHEMA,
        format_version=_FORMAT_VERSION,
        result_file="result.pt",
        frequencies=expected_frequencies,
        components=tuple(global_components),
        shards=tuple(shard_manifests),
    )
    _validate_manifest_structure(manifest)
    return manifest, ordered


def _validate_manifest_structure(manifest: DistributedResultManifest) -> None:
    if not manifest.components or not manifest.shards:
        raise ValueError("Sharded Result manifest must contain components and shards.")
    ranks = tuple(shard.rank for shard in manifest.shards)
    if ranks != tuple(range(len(manifest.shards))):
        raise ValueError(f"Manifest shard ranks must be contiguous from zero, got {ranks}.")
    component_names = tuple(component.name for component in manifest.components)
    if len(set(component_names)) != len(component_names):
        raise ValueError("Manifest contains duplicate global components.")
    for shard in manifest.shards:
        if tuple(component.name for component in shard.components) != component_names:
            raise ValueError(
                f"Manifest shard rank {shard.rank} component order is inconsistent."
            )
    for component_index, global_component in enumerate(manifest.components):
        if len(global_component.shape) < 3 or any(
            int(value) <= 0 for value in global_component.shape
        ):
            raise ValueError(
                f"Manifest global {global_component.name} has invalid shape "
                f"{global_component.shape}."
            )
        if global_component.x_axis != len(global_component.shape) - 3:
            raise ValueError(
                f"Manifest global {global_component.name} x_axis is inconsistent with ndim."
            )
        expected_start = 0
        for shard in manifest.shards:
            local = shard.components[component_index]
            if local.frequencies != manifest.frequencies:
                raise ValueError(
                    f"Manifest shard rank {shard.rank} frequencies are inconsistent."
                )
            if local.dtype != global_component.dtype or local.x_axis != global_component.x_axis:
                raise ValueError(
                    f"Manifest shard rank {shard.rank} {local.name} metadata is inconsistent."
                )
            if (
                len(local.shape) != len(global_component.shape)
                or local.x_axis != len(local.shape) - 3
            ):
                raise ValueError(
                    f"Manifest shard rank {shard.rank} {local.name} shape/x_axis is invalid."
                )
            if local.shape[local.x_axis] != (
                local.global_x_slice[1] - local.global_x_slice[0]
            ):
                raise ValueError(
                    f"Manifest shard rank {shard.rank} {local.name} x size does not "
                    "match its owned interval."
                )
            for axis, (local_size, global_size) in enumerate(
                zip(local.shape, global_component.shape)
            ):
                if axis != local.x_axis and local_size != global_size:
                    raise ValueError(
                        f"Manifest shard rank {shard.rank} {local.name} non-x shape "
                        "does not match the global field."
                    )
            if local.global_x_slice[0] != expected_start:
                raise ValueError(
                    f"Manifest shard rank {shard.rank} {local.name} intervals are not contiguous."
                )
            expected_start = local.global_x_slice[1]
        if expected_start != global_component.shape[global_component.x_axis]:
            raise ValueError(
                f"Manifest global {global_component.name} x size does not match shard intervals."
            )


def _rank_payload(
    artifact: FieldShardArtifact,
    shard_manifest: FieldShardManifest,
) -> dict[str, Any]:
    components = {}
    for component, component_manifest in zip(artifact.components, shard_manifest.components):
        components[component.name] = {
            "tensor": component.tensor,
            "global_x_slice": component_manifest.global_x_slice,
            "shape": component_manifest.shape,
            "dtype": component_manifest.dtype,
            "x_axis": component_manifest.x_axis,
            "frequencies": component_manifest.frequencies,
        }
    return _cpu_data(
        {
            "format_version": _FORMAT_VERSION,
            "rank": artifact.rank,
            "device": shard_manifest.device,
            "frequencies": artifact.frequencies,
            "components": components,
        }
    )


def _sync_file(path: Path) -> None:
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_sharded_result(
    directory: str | Path,
    *,
    result_payload: Mapping[str, Any],
    shard_artifacts: Iterable[FieldShardArtifact | Mapping[str, Any]],
    frequencies: tuple[float, ...],
) -> DistributedResultManifest:
    """Atomically publish ``manifest.json``, ``result.pt``, and rank tensor files."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(
            "Sharded Result directory already exists; refusing non-atomic overwrite: "
            f"{destination}."
        )
    manifest, artifacts = _build_manifest(shard_artifacts, frequencies=frequencies)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        result_path = staging / manifest.result_file
        torch.save(_cpu_data(dict(result_payload)), result_path)
        _sync_file(result_path)
        for artifact, shard_manifest in zip(artifacts, manifest.shards):
            shard_path = staging / shard_manifest.file
            torch.save(_rank_payload(artifact, shard_manifest), shard_path)
            _sync_file(shard_path)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _sync_file(manifest_path)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _load_result_payload(
    directory: Path,
    manifest: DistributedResultManifest,
    *,
    map_location: Any,
) -> dict[str, Any]:
    path = directory / manifest.result_file
    if not path.is_file():
        raise FileNotFoundError(f"Sharded Result metadata file is missing: {path}.")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to load sharded Result metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Sharded Result metadata {path} must contain a mapping.")
    payload_frequencies = _frequencies(
        payload.get("frequencies", (payload.get("frequency"),))
    )
    if payload_frequencies != manifest.frequencies:
        raise ValueError("Sharded Result metadata frequencies do not match manifest.")
    return payload


def _load_rank_payload(
    path: Path,
    shard_manifest: FieldShardManifest,
    manifest: DistributedResultManifest,
    *,
    map_location: Any,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load sharded Result rank {shard_manifest.rank} file {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Shard rank {shard_manifest.rank} file must contain a mapping.")
    if int(payload.get("format_version", -1)) != _FORMAT_VERSION:
        raise ValueError(f"Shard rank {shard_manifest.rank} has unsupported format_version.")
    if int(payload.get("rank", -1)) != shard_manifest.rank:
        raise ValueError(f"Shard file {path} rank does not match manifest.")
    if _frequencies(payload.get("frequencies", ())) != manifest.frequencies:
        raise ValueError(f"Shard rank {shard_manifest.rank} frequencies do not match manifest.")
    components = payload.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"Shard rank {shard_manifest.rank} components must be a mapping.")
    expected_names = tuple(component.name for component in shard_manifest.components)
    if tuple(components) != expected_names:
        raise ValueError(f"Shard rank {shard_manifest.rank} component order is corrupt.")
    for component_manifest in shard_manifest.components:
        component = components[component_manifest.name]
        if not isinstance(component, dict):
            raise ValueError(
                f"Shard rank {shard_manifest.rank} {component_manifest.name} is corrupt."
            )
        tensor = component.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Shard rank {shard_manifest.rank} {component_manifest.name} tensor is missing."
            )
        checks = {
            "global_x_slice": tuple(component.get("global_x_slice", ()))
            == component_manifest.global_x_slice,
            "shape": tuple(component.get("shape", ())) == component_manifest.shape,
            "tensor_shape": tuple(tensor.shape) == component_manifest.shape,
            "dtype": str(component.get("dtype")) == component_manifest.dtype,
            "tensor_dtype": str(tensor.dtype) == component_manifest.dtype,
            "x_axis": int(component.get("x_axis", -1)) == component_manifest.x_axis,
            "frequencies": _frequencies(component.get("frequencies", ()))
            == component_manifest.frequencies,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Shard rank {shard_manifest.rank} {component_manifest.name} metadata "
                f"is corrupt: {', '.join(failed)}."
            )
    return payload


def _gather_fields(
    manifest: DistributedResultManifest,
    shard_payloads: tuple[dict[str, Any], ...],
) -> dict[str, torch.Tensor]:
    fields = {}
    for component_index, global_component in enumerate(manifest.components):
        first = shard_payloads[0]["components"][global_component.name]["tensor"]
        destination = torch.empty(
            global_component.shape,
            device=first.device,
            dtype=first.dtype,
        )
        for shard_manifest, payload in zip(manifest.shards, shard_payloads):
            component_manifest = shard_manifest.components[component_index]
            source = payload["components"][global_component.name]["tensor"]
            destination_index = [slice(None)] * destination.ndim
            destination_index[global_component.x_axis] = slice(
                *component_manifest.global_x_slice
            )
            destination[tuple(destination_index)].copy_(source)
        fields[global_component.name.upper()] = destination
    return fields


def load_sharded_result(
    directory: str | Path,
    *,
    gather_fields: bool = False,
    map_location: Any = "cpu",
) -> LoadedShardedResult:
    """Load metadata lazily, reading rank tensors only for an explicit gather."""

    root = Path(directory)
    manifest = DistributedResultManifest.read(root)
    paths = manifest.shard_paths(root)
    for shard, path in zip(manifest.shards, paths):
        if not path.is_file():
            raise FileNotFoundError(
                f"Sharded Result rank {shard.rank} file is missing: {path}."
            )
    result_payload = _load_result_payload(root, manifest, map_location=map_location)
    fields = {}
    if gather_fields:
        shard_payloads = tuple(
            _load_rank_payload(
                path,
                shard,
                manifest,
                map_location=map_location,
            )
            for shard, path in zip(manifest.shards, paths)
        )
        fields = _gather_fields(manifest, shard_payloads)
    return LoadedShardedResult(
        manifest=manifest,
        result_payload=result_payload,
        shard_paths=paths,
        fields=fields,
    )


__all__ = [
    "DistributedResultManifest",
    "FieldComponentArtifact",
    "FieldComponentManifest",
    "FieldShardArtifact",
    "FieldShardManifest",
    "GlobalFieldManifest",
    "LoadedShardedResult",
    "export_distributed_field_shards",
    "load_sharded_result",
    "write_sharded_result",
]
