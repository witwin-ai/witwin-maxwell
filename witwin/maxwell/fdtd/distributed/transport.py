from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch


class _TransportShard(Protocol):
    rank: int
    device: torch.device
    layout: object
    solver: object
    compute_stream: torch.cuda.Stream
    communication_stream: torch.cuda.Stream
    electric_ready: torch.cuda.Event
    electric_received: torch.cuda.Event
    magnetic_ready: torch.cuda.Event
    magnetic_received: torch.cuda.Event
    halo_hy_low: torch.Tensor | None
    halo_hz_low: torch.Tensor | None


@dataclass(frozen=True)
class PeerLink:
    source_rank: int
    destination_rank: int
    source_device: str
    destination_device: str
    direction: str


class HaloTransport(ABC):
    """Transport contract used by the distributed time-step coordinator."""

    name = "abstract"

    @abstractmethod
    def preflight(self) -> None:
        ...

    @abstractmethod
    def exchange_electric(self, shards: tuple[_TransportShard, ...]) -> None:
        ...

    @abstractmethod
    def exchange_magnetic(self, shards: tuple[_TransportShard, ...]) -> None:
        ...

    def teardown(self) -> None:
        return None


class CudaP2PHaloTransport(HaloTransport):
    """One-process CUDA peer transport for asymmetric Yee x halos."""

    name = "cuda_p2p"

    def __init__(self, devices: tuple[torch.device, ...]):
        self.devices = tuple(torch.device(device) for device in devices)
        self.links = tuple(
            PeerLink(
                source_rank=rank + 1,
                destination_rank=rank,
                source_device=str(self.devices[rank + 1]),
                destination_device=str(self.devices[rank]),
                direction="electric_right_to_left",
            )
            for rank in range(len(self.devices) - 1)
        ) + tuple(
            PeerLink(
                source_rank=rank,
                destination_rank=rank + 1,
                source_device=str(self.devices[rank]),
                destination_device=str(self.devices[rank + 1]),
                direction="magnetic_left_to_right",
            )
            for rank in range(len(self.devices) - 1)
        )

    def preflight(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA P2P transport requires torch.cuda.is_available().")
        count = torch.cuda.device_count()
        for device in self.devices:
            if device.type != "cuda" or device.index is None:
                raise ValueError(f"CUDA P2P transport requires indexed CUDA devices, got {device}.")
            if device.index < 0 or device.index >= count:
                raise ValueError(
                    f"CUDA device {device} is unavailable; this process exposes {count} CUDA devices."
                )
        for left, right in zip(self.devices, self.devices[1:]):
            if not torch.cuda.can_device_access_peer(left.index, right.index):
                raise RuntimeError(
                    f"Direct CUDA peer access is unavailable from {left} to {right}; "
                    "host-staged halo fallback is intentionally disabled."
                )
            if not torch.cuda.can_device_access_peer(right.index, left.index):
                raise RuntimeError(
                    f"Direct CUDA peer access is unavailable from {right} to {left}; "
                    "host-staged halo fallback is intentionally disabled."
                )

    def exchange_electric(self, shards: tuple[_TransportShard, ...]) -> None:
        for shard in shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.electric_ready.record(shard.compute_stream)

        for destination, source in zip(shards[:-1], shards[1:]):
            source_node = source.layout.storage_node_owned.start
            destination_ghost = destination.layout.storage_node_owned.stop
            stream = destination.communication_stream
            with torch.cuda.device(destination.device), torch.cuda.stream(stream):
                stream.wait_event(source.electric_ready)
                stream.wait_event(destination.electric_ready)
                destination.solver.Ey[destination_ghost].copy_(
                    source.solver.Ey[source_node], non_blocking=True
                )
                destination.solver.Ez[destination_ghost].copy_(
                    source.solver.Ez[source_node], non_blocking=True
                )
                destination.electric_received.record(stream)

    def exchange_magnetic(self, shards: tuple[_TransportShard, ...]) -> None:
        for shard in shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.magnetic_ready.record(shard.compute_stream)

        for source, destination in zip(shards[:-1], shards[1:]):
            if destination.halo_hy_low is None or destination.halo_hz_low is None:
                raise RuntimeError("Magnetic receive halos were not allocated during prepare().")
            source_last = source.layout.storage_cell_owned.stop - 1
            stream = destination.communication_stream
            with torch.cuda.device(destination.device), torch.cuda.stream(stream):
                stream.wait_event(source.magnetic_ready)
                stream.wait_event(destination.magnetic_ready)
                destination.halo_hy_low.copy_(
                    source.solver.Hy[source_last], non_blocking=True
                )
                destination.halo_hz_low.copy_(
                    source.solver.Hz[source_last], non_blocking=True
                )
                destination.magnetic_received.record(stream)

    @property
    def topology(self) -> dict[str, object]:
        pairs = []
        for left, right in zip(self.devices, self.devices[1:]):
            pairs.append(
                {
                    "devices": (str(left), str(right)),
                    "peer_left_to_right": bool(
                        torch.cuda.can_device_access_peer(left.index, right.index)
                    ),
                    "peer_right_to_left": bool(
                        torch.cuda.can_device_access_peer(right.index, left.index)
                    ),
                }
            )
        return {"kind": self.name, "neighbor_pairs": tuple(pairs)}
