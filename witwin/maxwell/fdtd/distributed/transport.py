from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch


_CELL_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))


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

    @abstractmethod
    def exchange_magnetic_adjoint(
        self,
        shards: tuple[_TransportShard, ...],
        adjoint_states: tuple[dict, ...],
    ) -> None:
        """Transpose of :meth:`exchange_magnetic`.

        The forward magnetic halo copies each left shard's last owned Hy/Hz cell
        plane into the right neighbour's low ghost. Its transpose accumulates the
        right neighbour's ghost adjoint plane back into the left owner's last cell
        and then zeroes the ghost, so the ghost-adjoint-zero invariant the fused
        reverse kernels rely on holds at the next phase boundary.
        """
        ...

    @abstractmethod
    def exchange_electric_adjoint(
        self,
        shards: tuple[_TransportShard, ...],
        adjoint_states: tuple[dict, ...],
    ) -> None:
        """Transpose of :meth:`exchange_electric`.

        The forward electric halo copies each right shard's first owned Ey/Ez node
        plane into the left neighbour's ghost node. Its transpose accumulates the
        left neighbour's ghost adjoint node plane back into the right owner's first
        node and then zeroes the ghost.
        """
        ...

    # -- cross-rank coordinator primitives ---------------------------------
    #
    # The time-step coordinator expresses every operation that reaches beyond a
    # single rank through these primitives, so it never branches on the transport
    # kind. The in-process transport implements them over its shard tuple; a
    # one-process-per-GPU transport implements the same contract with collectives.

    @abstractmethod
    def reduce_owned_energy(self, engines: tuple[_TransportShard, ...]) -> torch.Tensor:
        """Sum every rank's owned electric energy into one scalar.

        Drives the shutoff-energy reduction. Each engine contributes
        :meth:`ShardEngine.owned_electric_energy`; the transport returns the
        global sum on the reduction device.
        """
        ...

    @abstractmethod
    def gather_component_slabs(
        self,
        engines: tuple[_TransportShard, ...],
        component: str,
        local_values: tuple[torch.Tensor, ...],
        *,
        result_device: torch.device,
        global_nx: int,
    ) -> torch.Tensor:
        """Stitch each rank's owned x-slab of ``component`` into a global tensor."""
        ...

    @abstractmethod
    def gather_monitor_payloads(self, engines: tuple[_TransportShard, ...]):
        """Collect every rank's monitor payloads + DFT E fields in rank order.

        Returns ``(shard_monitor_payloads, local_dft_fields, frequencies)`` where
        ``shard_monitor_payloads`` is a rank-ordered ``[(rank, payload), ...]``,
        ``local_dft_fields`` maps ``Ex/Ey/Ez`` to a rank-ordered list of owned
        slabs, and ``frequencies`` is the cross-rank-consistent DFT frequency
        tuple (or ``None`` when no DFT ran).
        """
        ...

    @abstractmethod
    def gather_stats(self, engines: tuple[_TransportShard, ...]) -> dict:
        """Gather per-rank partitions, peak memory, and per-step halo bytes."""
        ...

    def teardown(self) -> None:
        return None


class CudaP2PHaloTransport(HaloTransport):
    """One-process CUDA peer transport for asymmetric Yee x halos."""

    name = "cuda_p2p"

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        *,
        result_device: torch.device | str | None = None,
    ):
        self.devices = tuple(torch.device(device) for device in devices)
        self.result_device = (
            torch.device(result_device) if result_device is not None else self.devices[0]
        )
        # Preallocated reverse-halo staging planes on the destination (owner)
        # device, keyed by (kind, destination_rank). Allocated on first use and
        # reused for every subsequent reverse step so no per-step allocation is
        # introduced into the adjoint time loop.
        self._adjoint_staging: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
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

    def _staging_pair(
        self,
        kind: str,
        destination_rank: int,
        first: torch.Tensor,
        second: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return preallocated staging planes matching the given ghost planes.

        Allocated once per (kind, destination) and reused; a later call with an
        incompatible plane shape/dtype is a programming error (the padded layout
        is fixed at prepare) and rebuilds the buffer rather than silently
        mismatching.
        """
        key = (kind, int(destination_rank))
        cached = self._adjoint_staging.get(key)
        if (
            cached is None
            or cached[0].shape != first.shape
            or cached[1].shape != second.shape
            or cached[0].dtype != first.dtype
            or cached[1].dtype != second.dtype
            or cached[0].device != device
        ):
            with torch.cuda.device(device):
                cached = (
                    torch.empty(first.shape, device=device, dtype=first.dtype),
                    torch.empty(second.shape, device=device, dtype=second.dtype),
                )
            self._adjoint_staging[key] = cached
        return cached

    def prepare_adjoint_staging(
        self,
        shards: tuple[_TransportShard, ...],
        adjoint_states: tuple[dict, ...],
    ) -> None:
        """Preallocate every reverse-halo staging plane before the reverse loop."""

        for destination, source in zip(shards[:-1], shards[1:]):
            source_state = adjoint_states[source.rank]
            self._staging_pair(
                "magnetic",
                destination.rank,
                source_state["Hy"][0],
                source_state["Hz"][0],
                destination.device,
            )
        for source, destination in zip(shards[:-1], shards[1:]):
            source_state = adjoint_states[source.rank]
            ghost = source.layout.storage_node_owned.stop
            self._staging_pair(
                "electric",
                destination.rank,
                source_state["Ey"][ghost],
                source_state["Ez"][ghost],
                destination.device,
            )

    def exchange_magnetic_adjoint(
        self,
        shards: tuple[_TransportShard, ...],
        adjoint_states: tuple[dict, ...],
    ) -> None:
        # Producers: both the right shard's ghost adjoint plane and the left
        # shard's owner plane are produced on the compute stream.
        for shard in shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.magnetic_ready.record(shard.compute_stream)

        for destination, source in zip(shards[:-1], shards[1:]):
            source_state = adjoint_states[source.rank]
            destination_state = adjoint_states[destination.rank]
            source_ghost_hy = source_state["Hy"][0]
            source_ghost_hz = source_state["Hz"][0]
            staging_hy, staging_hz = self._staging_pair(
                "magnetic",
                destination.rank,
                source_ghost_hy,
                source_ghost_hz,
                destination.device,
            )
            owner_cell = destination.layout.storage_cell_owned.stop - 1
            comm = destination.communication_stream
            with torch.cuda.device(destination.device), torch.cuda.stream(comm):
                comm.wait_event(source.magnetic_ready)
                comm.wait_event(destination.magnetic_ready)
                staging_hy.copy_(source_ghost_hy, non_blocking=True)
                staging_hz.copy_(source_ghost_hz, non_blocking=True)
                destination.magnetic_received.record(comm)
            with torch.cuda.device(destination.device), torch.cuda.stream(destination.compute_stream):
                destination.compute_stream.wait_event(destination.magnetic_received)
                destination_state["Hy"][owner_cell].add_(staging_hy)
                destination_state["Hz"][owner_cell].add_(staging_hz)
            with torch.cuda.device(source.device), torch.cuda.stream(source.compute_stream):
                source.compute_stream.wait_event(destination.magnetic_received)
                source_ghost_hy.zero_()
                source_ghost_hz.zero_()

    def exchange_electric_adjoint(
        self,
        shards: tuple[_TransportShard, ...],
        adjoint_states: tuple[dict, ...],
    ) -> None:
        # Producers: the left shard's ghost node plane and the right shard's owner
        # node plane are produced on the compute stream.
        for shard in shards:
            with torch.cuda.device(shard.device), torch.cuda.stream(shard.compute_stream):
                shard.electric_ready.record(shard.compute_stream)

        for source, destination in zip(shards[:-1], shards[1:]):
            source_state = adjoint_states[source.rank]
            destination_state = adjoint_states[destination.rank]
            source_ghost = source.layout.storage_node_owned.stop
            owner_node = destination.layout.storage_node_owned.start
            source_ghost_ey = source_state["Ey"][source_ghost]
            source_ghost_ez = source_state["Ez"][source_ghost]
            staging_ey, staging_ez = self._staging_pair(
                "electric",
                destination.rank,
                source_ghost_ey,
                source_ghost_ez,
                destination.device,
            )
            comm = destination.communication_stream
            with torch.cuda.device(destination.device), torch.cuda.stream(comm):
                comm.wait_event(source.electric_ready)
                comm.wait_event(destination.electric_ready)
                staging_ey.copy_(source_ghost_ey, non_blocking=True)
                staging_ez.copy_(source_ghost_ez, non_blocking=True)
                destination.electric_received.record(comm)
            with torch.cuda.device(destination.device), torch.cuda.stream(destination.compute_stream):
                destination.compute_stream.wait_event(destination.electric_received)
                destination_state["Ey"][owner_node].add_(staging_ey)
                destination_state["Ez"][owner_node].add_(staging_ez)
            with torch.cuda.device(source.device), torch.cuda.stream(source.compute_stream):
                source.compute_stream.wait_event(destination.electric_received)
                source_ghost_ey.zero_()
                source_ghost_ez.zero_()

    # -- cross-rank coordinator primitives ---------------------------------

    def reduce_owned_energy(self, engines) -> torch.Tensor:
        # Each engine computes its owned energy on its own compute stream and
        # records ``electric_ready``; the reduction then orders its cross-device
        # reads after those events on the result device before summing.
        local_energies = [engine.owned_electric_energy() for engine in engines]
        device = self.result_device
        with torch.cuda.device(device):
            result_stream = torch.cuda.current_stream(device)
            total = torch.zeros((), device=device, dtype=torch.float32)
            for engine, local in zip(engines, local_energies):
                result_stream.wait_event(engine.electric_ready)
                total.add_(local.to(device, non_blocking=True))
            return total

    def gather_component_slabs(
        self,
        engines,
        component: str,
        local_values,
        *,
        result_device,
        global_nx: int,
    ) -> torch.Tensor:
        is_cell = component.capitalize() in _CELL_COMPONENTS
        global_x = global_nx - 1 if is_cell else global_nx
        sample = local_values[0]
        x_axis = sample.ndim - 3
        shape = list(sample.shape)
        shape[x_axis] = global_x
        destination = torch.empty(tuple(shape), device=result_device, dtype=sample.dtype)
        for engine, value in zip(engines, local_values):
            local_slice = (
                engine.layout.storage_cell_owned if is_cell else engine.layout.storage_node_owned
            )
            global_slice = (
                engine.layout.global_cell_owned if is_cell else engine.layout.global_node_owned
            )
            src_index = [slice(None)] * value.ndim
            dst_index = [slice(None)] * destination.ndim
            src_index[x_axis] = local_slice
            dst_index[x_axis] = global_slice
            destination[tuple(dst_index)].copy_(value[tuple(src_index)], non_blocking=True)
        return destination

    def gather_monitor_payloads(self, engines):
        shard_monitor_payloads: list[tuple[int, dict]] = []
        local_fields: dict[str, list] = {}
        frequency_metadata: tuple[float, ...] | None = None
        for engine in engines:
            shard_monitors, engine_fields, frequencies = engine.collect_local_monitor_payload()
            for name, tensor in engine_fields.items():
                local_fields.setdefault(name, []).append(tensor)
            if frequencies is not None:
                if frequency_metadata is None:
                    frequency_metadata = frequencies
                elif frequency_metadata != frequencies:
                    raise RuntimeError("Shard-local DFT frequency metadata is inconsistent.")
            shard_monitor_payloads.append((engine.rank, shard_monitors))
        return shard_monitor_payloads, local_fields, frequency_metadata

    def gather_stats(self, engines) -> dict:
        halo_bytes_per_step = 0
        for left, right in zip(engines[:-1], engines[1:]):
            halo_bytes_per_step += (
                left.solver.Ey[-1].numel()
                + left.solver.Ez[-1].numel()
                + right.solver.Hy[0].numel()
                + right.solver.Hz[0].numel()
            ) * left.solver.Ex.element_size()
        partitions = tuple(
            {
                "rank": engine.rank,
                "device": str(engine.device),
                "physical_cells": (
                    engine.layout.physical_cell_begin,
                    engine.layout.physical_cell_end,
                ),
                "global_cells": (
                    engine.layout.global_cell_owned.start,
                    engine.layout.global_cell_owned.stop,
                ),
                "global_nodes": (
                    engine.layout.global_node_owned.start,
                    engine.layout.global_node_owned.stop,
                ),
                "peak_memory_bytes": engine.peak_memory_bytes,
            }
            for engine in engines
        )
        return {
            "halo_bytes_per_step": int(halo_bytes_per_step),
            "partitions": partitions,
            "peak_memory_bytes": {
                str(engine.device): engine.peak_memory_bytes for engine in engines
            },
        }

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
