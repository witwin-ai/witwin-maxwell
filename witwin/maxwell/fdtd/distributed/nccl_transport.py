"""Single-node one-process-per-GPU NCCL halo transport.

This module implements the rank-local half of the distributed FDTD transport
contract for the ``transport="nccl"`` execution shape. Unlike the in-process
:class:`~witwin.maxwell.fdtd.distributed.transport.CudaP2PHaloTransport`, which
holds every shard in a single process and copies between peer tensors directly,
``NcclHaloTransport`` runs one rank per GPU (launched by ``torchrun``) and moves
Yee x-plane halos with ``torch.distributed.batch_isend_irecv``.

torch 2.13 removed same-process multi-device NCCL collectives, so
``init_process_group("nccl", device_id=...)`` binds exactly one device per rank.
This transport therefore owns only the local rank's tensors and exchanges
contiguous x-planes with its immediate chain neighbours. The forward and reverse
halo semantics are the transpose-consistent mirror of the CUDA P2P transport:

* forward electric halo: each rank ships its first owned Ey/Ez node plane to the
  left neighbour's high ghost node (data flows right -> left);
* forward magnetic halo: each rank ships its last owned Hy/Hz cell plane to the
  right neighbour's low ghost cell (data flows left -> right);
* reverse magnetic halo (adjoint transpose of the forward magnetic halo):
  the right neighbour ships its low ghost adjoint plane back to the left owner,
  which accumulates it into its last owned cell before the ghost is zeroed;
* reverse electric halo (adjoint transpose of the forward electric halo):
  the left neighbour ships its high ghost adjoint node plane back to the right
  owner, which accumulates it into its first owned node before zeroing the ghost.

Reverse halos receive into preallocated staging planes and then ``add_`` on the
owner, which keeps the reduction deterministic (no atomics, fixed rank order).
"""

from __future__ import annotations

import datetime as _datetime
import os
import platform
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from .transport import _CELL_COMPONENTS

# Environment variables torchrun always populates for a single-node launch.
_REQUIRED_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK")

# Default collective timeout. Deliberately generous: a stalled peer surfaces as a
# ProcessGroupNCCL watchdog abort rather than an indefinite hang. Callers that
# need a different bound pass ``timeout_s`` explicitly.
_DEFAULT_TIMEOUT_S = 1800.0


def _cuda_backend_is_nccl(backend: str) -> bool:
    """Return whether an adopted group's CUDA collectives run on NCCL.

    ``torch.distributed.get_backend()`` returns either a plain backend token
    (``"nccl"``) or a composite device-to-backend spec such as
    ``"cpu:gloo,cuda:nccl"`` when a group was created with per-device backends.
    A composite group whose CUDA backend is NCCL is fully NCCL-capable for the
    device halos this transport drives, so accept it; reject anything whose CUDA
    collectives are not NCCL. Parsing fails closed: an unrecognisable spec that
    does not positively resolve to a CUDA/NCCL binding is treated as non-NCCL.
    """

    normalized = backend.strip().lower()
    if not normalized:
        return False
    if ":" not in normalized and "," not in normalized:
        return normalized == "nccl"
    for entry in normalized.split(","):
        token = entry.strip()
        if not token:
            continue
        device, sep, name = token.partition(":")
        if not sep:
            # Bare backend token inside a composite spec applies to all devices.
            if device.strip() == "nccl":
                return True
            continue
        if device.strip() == "cuda" and name.strip() == "nccl":
            return True
    return False


@dataclass(frozen=True)
class _DeviceSignature:
    """Homogeneity fingerprint gathered across ranks before the time loop.

    Uses the device name and compute capability only, matching the acceptance
    convention already used by the in-process P2P fixtures. Per-board
    ``total_memory`` reports differ by a few MB (ECC/reserved variance) between
    otherwise identical boards and is deliberately excluded.
    """

    name: str
    major: int
    minor: int
    multi_processor_count: int


class NcclHaloTransport:
    """Rank-local NCCL transport for asymmetric Yee x halos.

    One instance lives per ``torchrun`` rank. The public step-loop surface is the
    same transport-primitive contract the coordinator drives: :meth:`preflight`,
    :meth:`exchange_electric`, :meth:`exchange_magnetic`, the two adjoint
    transposes, :meth:`allreduce_scalar`, :meth:`barrier`, and :meth:`teardown`.
    """

    name = "nccl"

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ):
        if world_size < 2:
            raise ValueError("NCCL transport requires world_size >= 2.")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside [0, {world_size}).")
        if local_rank < 0:
            raise ValueError(f"local_rank {local_rank} must be non-negative.")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.timeout_s = float(timeout_s)
        self.device = torch.device(f"cuda:{self.local_rank}")
        self._connected = False
        # Every rank's partition layout, bound by the coordinator before solve()
        # so rank 0 can size field gathers without a shape-exchange round trip.
        self._shard_layouts = None
        # Chain neighbours: rank r-1 (left) and rank r+1 (right). ``None`` at the
        # ends of the chain, matching the shard endpoints that carry no ghost.
        self.left_rank = self.rank - 1 if self.rank > 0 else None
        self.right_rank = self.rank + 1 if self.rank + 1 < self.world_size else None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        expected_world_size: int | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> "NcclHaloTransport":
        """Build a transport from the ``torchrun`` environment.

        Raises with an explicit, actionable message when the launcher
        environment is absent — the same failure mode the same-process runtime
        surfaces today, now pointing at the required one-process-per-GPU shape.
        """

        source = os.environ if env is None else env
        missing = [key for key in _REQUIRED_ENV if not source.get(key)]
        if missing:
            raise RuntimeError(
                "NCCL transport requires a torchrun one-process-per-GPU launch; "
                f"missing environment variable(s): {', '.join(missing)}. Launch with "
                "`torchrun --nproc-per-node=<gpus> ...` so RANK/WORLD_SIZE/LOCAL_RANK "
                "are populated."
            )
        world_size = int(source["WORLD_SIZE"])
        if expected_world_size is not None and world_size != int(expected_world_size):
            raise RuntimeError(
                f"NCCL world size {world_size} does not match the configured device "
                f"count {int(expected_world_size)}; launch torchrun with "
                "--nproc-per-node equal to the number of participating GPUs."
            )
        return cls(
            rank=int(source["RANK"]),
            world_size=world_size,
            local_rank=int(source["LOCAL_RANK"]),
            timeout_s=timeout_s,
        )

    # -- lifecycle ---------------------------------------------------------

    def preflight(self) -> None:
        """Bind the local device, initialise NCCL, and verify homogeneity.

        Ordered so the cheapest, allocation-free guards fail first: platform and
        CUDA availability are checked before any process group is created, so a
        misconfigured launch never leaves a half-initialised group behind.
        """

        if platform.system() != "Linux":
            raise RuntimeError(
                "NCCL transport is single-node Linux only; "
                f"detected platform {platform.system()!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL transport requires torch.cuda.is_available().")
        if self.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"local_rank {self.local_rank} exceeds the {torch.cuda.device_count()} "
                "CUDA device(s) visible to this process."
            )

        torch.cuda.set_device(self.device)
        if dist.is_initialized():
            self._validate_adopted_group()
        else:
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
                device_id=self.device,
                timeout=_datetime.timedelta(seconds=self.timeout_s),
            )
        self._connected = True
        self._verify_homogeneity()

    def _validate_adopted_group(self) -> None:
        """Reject a pre-existing process group that disagrees with this rank.

        ``preflight`` adopts an already-initialised default group instead of
        creating a second one, but a group whose world size, rank, or backend
        does not match this transport's expectations would make every halo
        exchange address the wrong peer. Fail closed with an explicit message
        rather than silently binding to a mismatched group.
        """

        actual_world = dist.get_world_size()
        if actual_world != self.world_size:
            raise RuntimeError(
                "NCCL transport adopted an existing process group with world size "
                f"{actual_world}, but this rank expects {self.world_size}; the "
                "launcher and the configured device count disagree."
            )
        actual_rank = dist.get_rank()
        if actual_rank != self.rank:
            raise RuntimeError(
                "NCCL transport adopted an existing process group reporting rank "
                f"{actual_rank}, but this transport was constructed for rank "
                f"{self.rank}; the launcher rank assignment is inconsistent."
            )
        backend = str(dist.get_backend())
        if not _cuda_backend_is_nccl(backend):
            raise RuntimeError(
                "NCCL transport requires a NCCL-backed process group; the adopted "
                f"group uses backend {backend!r}."
            )

    def _verify_homogeneity(self) -> None:
        properties = torch.cuda.get_device_properties(self.device)
        local = _DeviceSignature(
            name=properties.name,
            major=properties.major,
            minor=properties.minor,
            multi_processor_count=int(properties.multi_processor_count),
        )
        gathered: list[_DeviceSignature | None] = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        distinct = {signature for signature in gathered if signature is not None}
        if len(distinct) != 1:
            raise RuntimeError(
                "NCCL transport requires homogeneous CUDA devices across ranks; "
                f"observed heterogeneous signatures: {sorted(str(s) for s in distinct)}."
            )

    def teardown(self) -> None:
        """Destroy the process group deterministically (idempotent)."""

        if dist.is_initialized():
            dist.destroy_process_group()
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("NCCL transport used before preflight().")
        # Defensive guard for the shared-group pattern: several transports may
        # adopt the same default process group, and a teardown on any one of them
        # destroys it for all. A sibling that still reads ``_connected == True``
        # would otherwise drive a collective against a destroyed group. Detect the
        # vanished group, flip the local flag, and fail closed with an actionable
        # message instead of surfacing an opaque torch.distributed error.
        if not dist.is_initialized():
            self._connected = False
            raise RuntimeError(
                "NCCL transport process group has been destroyed; this transport is "
                "no longer connected. Build and preflight() a fresh transport."
            )

    # -- primitive batched point-to-point ----------------------------------

    def _exchange_planes(
        self,
        *,
        send: Sequence[torch.Tensor] | None,
        send_to: int | None,
        recv: Sequence[torch.Tensor] | None,
        recv_from: int | None,
    ) -> None:
        """Post one batched send/recv group and wait on it.

        Every plane must be a contiguous device tensor. Operations are appended
        in a fixed order (receives before sends) so both peers of a pair post the
        complementary ops in the same relative order; ``batch_isend_irecv`` fuses
        them into a single NCCL group that cannot deadlock on a chain topology.
        """

        self._require_connected()
        ops: list[dist.P2POp] = []
        if recv is not None and recv_from is not None:
            for plane in recv:
                if not plane.is_contiguous():
                    raise ValueError("NCCL halo receive plane must be contiguous.")
                ops.append(dist.P2POp(dist.irecv, plane, recv_from))
        if send is not None and send_to is not None:
            for plane in send:
                if not plane.is_contiguous():
                    raise ValueError("NCCL halo send plane must be contiguous.")
                ops.append(dist.P2POp(dist.isend, plane, send_to))
        if not ops:
            return
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()

    # -- forward halo plane primitives -------------------------------------

    def _electric_halo_planes(
        self,
        *,
        first_owned_node_planes: Sequence[torch.Tensor] | None,
        ghost_node_planes: Sequence[torch.Tensor] | None,
    ) -> None:
        """Forward electric halo: first owned Ey/Ez node plane -> left ghost.

        ``first_owned_node_planes`` are shipped to the left neighbour; the right
        neighbour's contribution is received into ``ghost_node_planes``.
        """

        self._exchange_planes(
            send=first_owned_node_planes if self.left_rank is not None else None,
            send_to=self.left_rank,
            recv=ghost_node_planes if self.right_rank is not None else None,
            recv_from=self.right_rank,
        )

    def _magnetic_halo_planes(
        self,
        *,
        last_owned_cell_planes: Sequence[torch.Tensor] | None,
        low_ghost_planes: Sequence[torch.Tensor] | None,
    ) -> None:
        """Forward magnetic halo: last owned Hy/Hz cell plane -> right ghost.

        ``last_owned_cell_planes`` are shipped to the right neighbour; the left
        neighbour's contribution is received into ``low_ghost_planes``.
        """

        self._exchange_planes(
            send=last_owned_cell_planes if self.right_rank is not None else None,
            send_to=self.right_rank,
            recv=low_ghost_planes if self.left_rank is not None else None,
            recv_from=self.left_rank,
        )

    # -- reverse (adjoint transpose) halo plane primitives -----------------

    def _magnetic_adjoint_planes(
        self,
        *,
        ghost_adjoint_planes: Sequence[torch.Tensor] | None,
        owner_adjoint_planes: Sequence[torch.Tensor] | None,
        staging_planes: Sequence[torch.Tensor] | None,
    ) -> None:
        """Transpose of :meth:`_magnetic_halo_planes`.

        Data flows right -> left: this rank ships its low ghost adjoint plane to
        the left neighbour and then zeroes it, while receiving the right
        neighbour's ghost adjoint plane into ``staging_planes`` and accumulating
        it into ``owner_adjoint_planes`` (the last owned cell).
        """

        recv = staging_planes if self.right_rank is not None else None
        self._exchange_planes(
            send=ghost_adjoint_planes if self.left_rank is not None else None,
            send_to=self.left_rank,
            recv=recv,
            recv_from=self.right_rank,
        )
        if self.right_rank is not None and owner_adjoint_planes is not None and staging_planes is not None:
            for owner, staged in zip(owner_adjoint_planes, staging_planes):
                owner.add_(staged)
        if self.left_rank is not None and ghost_adjoint_planes is not None:
            for plane in ghost_adjoint_planes:
                plane.zero_()

    def _electric_adjoint_planes(
        self,
        *,
        ghost_adjoint_planes: Sequence[torch.Tensor] | None,
        owner_adjoint_planes: Sequence[torch.Tensor] | None,
        staging_planes: Sequence[torch.Tensor] | None,
    ) -> None:
        """Transpose of :meth:`_electric_halo_planes`.

        Data flows left -> right: this rank ships its high ghost adjoint node
        plane to the right neighbour and then zeroes it, while receiving the left
        neighbour's ghost adjoint plane into ``staging_planes`` and accumulating
        it into ``owner_adjoint_planes`` (the first owned node).
        """

        recv = staging_planes if self.left_rank is not None else None
        self._exchange_planes(
            send=ghost_adjoint_planes if self.right_rank is not None else None,
            send_to=self.right_rank,
            recv=recv,
            recv_from=self.left_rank,
        )
        if self.left_rank is not None and owner_adjoint_planes is not None and staging_planes is not None:
            for owner, staged in zip(owner_adjoint_planes, staging_planes):
                owner.add_(staged)
        if self.right_rank is not None and ghost_adjoint_planes is not None:
            for plane in ghost_adjoint_planes:
                plane.zero_()

    # -- engine-based coordinator primitives -------------------------------
    #
    # The coordinator drives these exactly as it drives the in-process P2P
    # transport: it passes its rank-local engine tuple (one engine for a
    # one-process-per-GPU launch) and never branches on transport kind. Each
    # method extracts the contiguous Yee x-planes from the local engine and
    # drives the plane primitives / collectives above.

    def bind_coordinator_layouts(self, shard_layouts) -> None:
        """Record every rank's layout for sized field gathers to rank 0.

        Layouts are pure deterministic partition metadata available identically
        on every rank, so rank 0 can preallocate each peer's owned slab shape
        without a size-exchange round trip.
        """

        self._shard_layouts = tuple(shard_layouts)

    def exchange_electric(self, engines) -> None:
        engine = engines[0]
        solver = engine.solver
        ns = engine.layout.storage_node_owned
        with torch.cuda.device(engine.device), torch.cuda.stream(engine.compute_stream):
            first_owned = None
            if self.left_rank is not None:
                first_owned = [solver.Ey[ns.start], solver.Ez[ns.start]]
            ghost = None
            if self.right_rank is not None:
                ghost = [solver.Ey[ns.stop], solver.Ez[ns.stop]]
            self._electric_halo_planes(
                first_owned_node_planes=first_owned, ghost_node_planes=ghost
            )
            engine.electric_received.record(engine.compute_stream)

    def exchange_magnetic(self, engines) -> None:
        engine = engines[0]
        solver = engine.solver
        cs = engine.layout.storage_cell_owned
        with torch.cuda.device(engine.device), torch.cuda.stream(engine.compute_stream):
            last_owned = None
            if self.right_rank is not None:
                last_owned = [solver.Hy[cs.stop - 1], solver.Hz[cs.stop - 1]]
            low_ghost = None
            if self.left_rank is not None:
                low_ghost = [solver.Hy[0], solver.Hz[0]]
            self._magnetic_halo_planes(
                last_owned_cell_planes=last_owned, low_ghost_planes=low_ghost
            )
            engine.magnetic_received.record(engine.compute_stream)

    def reduce_owned_energy(self, engines) -> torch.Tensor:
        engine = engines[0]
        return self.allreduce_scalar(engine.owned_electric_energy())

    def gather_component_slabs(
        self,
        engines,
        component: str,
        local_values,
        *,
        result_device,
        global_nx: int,
    ) -> torch.Tensor | None:
        """Gather each rank's owned x-slab of ``component`` onto rank 0.

        Sized point-to-point: every nonzero rank sends its contiguous owned slab
        to rank 0, which stitches them into the global tensor at the owned global
        x-interval each layout declares. Returns the global tensor on rank 0 and
        ``None`` on every other rank (which holds no global result).
        """

        self._require_connected()
        if self._shard_layouts is None:
            raise RuntimeError(
                "NCCL field gather requires bind_coordinator_layouts() before solve()."
            )
        engine = engines[0]
        is_cell = component.capitalize() in _CELL_COMPONENTS
        value = local_values[0]
        x_axis = value.ndim - 3
        local_slice = (
            engine.layout.storage_cell_owned if is_cell else engine.layout.storage_node_owned
        )
        src_index = [slice(None)] * value.ndim
        src_index[x_axis] = local_slice
        owned = value[tuple(src_index)].contiguous()

        if self.rank != 0:
            dist.send(owned, dst=0)
            return None

        global_x = global_nx - 1 if is_cell else global_nx
        shape = list(value.shape)
        shape[x_axis] = global_x
        destination = torch.empty(tuple(shape), device=result_device, dtype=value.dtype)

        def _global_slice(layout):
            return layout.global_cell_owned if is_cell else layout.global_node_owned

        dst_index = [slice(None)] * destination.ndim
        dst_index[x_axis] = _global_slice(engine.layout)
        destination[tuple(dst_index)].copy_(owned.to(result_device))

        for peer_rank in range(1, self.world_size):
            layout = self._shard_layouts[peer_rank]
            gslice = _global_slice(layout)
            recv_shape = list(shape)
            recv_shape[x_axis] = int(gslice.stop) - int(gslice.start)
            buffer = torch.empty(
                tuple(recv_shape), device=result_device, dtype=value.dtype
            )
            dist.recv(buffer, src=peer_rank)
            peer_dst = [slice(None)] * destination.ndim
            peer_dst[x_axis] = gslice
            destination[tuple(peer_dst)].copy_(buffer)
        return destination

    def gather_monitor_payloads(self, engines):
        # Engine-generic collection: with a single rank-local engine this returns
        # only this rank's payloads (monitor-output assembly across ranks is a
        # follow-up gather; NCCL forward runs are field-gather objectives).
        shard_monitor_payloads = []
        local_fields: dict = {}
        frequency_metadata = None
        for engine in engines:
            shard_monitors, engine_fields, frequencies = engine.collect_local_monitor_payload()
            for name, tensor in engine_fields.items():
                local_fields.setdefault(name, []).append(tensor)
            if frequencies is not None:
                frequency_metadata = frequencies
            shard_monitor_payloads.append((engine.rank, shard_monitors))
        return shard_monitor_payloads, local_fields, frequency_metadata

    def gather_stats(self, engines) -> dict:
        # Rank-local partition snapshot; cross-rank stat aggregation is a
        # follow-up gather and is not required for numerical conformance.
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
            "halo_bytes_per_step": 0,
            "partitions": partitions,
            "peak_memory_bytes": {
                str(engine.device): engine.peak_memory_bytes for engine in engines
            },
        }

    # -- scalar collective -------------------------------------------------

    def allreduce_scalar(self, value: float | torch.Tensor) -> torch.Tensor:
        """Sum a rank-local scalar across the world (e.g. shutoff energy)."""

        self._require_connected()
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=self.device, dtype=torch.float64).reshape(())
        else:
            tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def barrier(self) -> None:
        self._require_connected()
        dist.barrier()

    @property
    def topology(self) -> dict[str, object]:
        return {
            "kind": self.name,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "left_rank": self.left_rank,
            "right_rank": self.right_rank,
        }


__all__ = ["NcclHaloTransport"]
