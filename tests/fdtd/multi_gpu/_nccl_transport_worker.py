"""torchrun worker exercising ``NcclHaloTransport`` over two ranks.

Launched by ``tests/fdtd/multi_gpu/test_nccl_transport.py`` via
``torchrun --nproc-per-node=2``. Each rank drives the full transport-primitive
surface against synthetic contiguous Yee x-planes and raises on any mismatch, so
a nonzero process exit signals a conformance failure. Values are deterministic
functions of the source rank, making every assertion a bitwise ``torch.equal``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# torchrun sets sys.path[0] to this file's directory, so make the repository
# root importable before pulling in the package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from witwin.maxwell.fdtd.distributed.nccl_transport import NcclHaloTransport

NY, NZ = 6, 5


def _plane(value: float, device: torch.device) -> torch.Tensor:
    plane = torch.full((NY, NZ), float(value), device=device, dtype=torch.float64)
    assert plane.is_contiguous()
    return plane


def main() -> None:
    transport = NcclHaloTransport.from_env(expected_world_size=2)
    transport.preflight()
    rank = transport.rank
    device = transport.device
    assert transport.world_size == 2

    # -- forward electric: right rank's first owned node -> left rank ghost --
    first_owned = [_plane(100 + rank, device), _plane(200 + rank, device)]
    ghost = [_plane(-1, device), _plane(-1, device)]
    transport.exchange_electric(
        first_owned_node_planes=first_owned,
        ghost_node_planes=ghost,
    )
    if rank == 0:
        # Ghost filled from rank 1's first owned planes.
        assert torch.equal(ghost[0], _plane(101, device)), "electric ghost Ey mismatch"
        assert torch.equal(ghost[1], _plane(201, device)), "electric ghost Ez mismatch"
    if rank == 1:
        # Endpoint negative: the rightmost rank has no right neighbour, so its
        # electric ghost must be left untouched (no phantom receive).
        assert torch.equal(ghost[0], _plane(-1, device)), "endpoint electric ghost Ey mutated"
        assert torch.equal(ghost[1], _plane(-1, device)), "endpoint electric ghost Ez mutated"

    # -- forward magnetic: left rank's last owned cell -> right rank ghost --
    last_owned = [_plane(300 + rank, device), _plane(400 + rank, device)]
    low_ghost = [_plane(-1, device), _plane(-1, device)]
    transport.exchange_magnetic(
        last_owned_cell_planes=last_owned,
        low_ghost_planes=low_ghost,
    )
    if rank == 1:
        assert torch.equal(low_ghost[0], _plane(300, device)), "magnetic ghost Hy mismatch"
        assert torch.equal(low_ghost[1], _plane(400, device)), "magnetic ghost Hz mismatch"
    if rank == 0:
        # Endpoint negative: the leftmost rank has no left neighbour, so its low
        # magnetic ghost must be left untouched.
        assert torch.equal(low_ghost[0], _plane(-1, device)), "endpoint magnetic ghost Hy mutated"
        assert torch.equal(low_ghost[1], _plane(-1, device)), "endpoint magnetic ghost Hz mutated"

    # -- reverse magnetic adjoint: right ghost adj -> left owner (add), zero --
    m_owner = [_plane(10, device), _plane(20, device)]
    m_ghost = [_plane(7, device), _plane(9, device)]
    m_staging = [_plane(0, device), _plane(0, device)]
    transport.exchange_magnetic_adjoint(
        ghost_adjoint_planes=m_ghost,
        owner_adjoint_planes=m_owner,
        staging_planes=m_staging,
    )
    if rank == 0:
        # rank 0 owner accumulates rank 1's ghost adjoint (7, 9).
        assert torch.equal(m_owner[0], _plane(17, device)), "magnetic adjoint owner Hy mismatch"
        assert torch.equal(m_owner[1], _plane(29, device)), "magnetic adjoint owner Hz mismatch"
        # Endpoint negative: leftmost rank has no left neighbour, so it never
        # ships its own ghost adjoint and must not zero it.
        assert torch.equal(m_ghost[0], _plane(7, device)), "endpoint magnetic adjoint ghost Hy zeroed"
        assert torch.equal(m_ghost[1], _plane(9, device)), "endpoint magnetic adjoint ghost Hz zeroed"
    if rank == 1:
        # rank 1's ghost adjoint must be zeroed after shipping it left.
        assert torch.equal(m_ghost[0], _plane(0, device)), "magnetic adjoint ghost Hy not zeroed"
        assert torch.equal(m_ghost[1], _plane(0, device)), "magnetic adjoint ghost Hz not zeroed"
        # Endpoint negative: rightmost rank has no right neighbour, so its owner
        # receives nothing and stays at its initial value.
        assert torch.equal(m_owner[0], _plane(10, device)), "endpoint magnetic adjoint owner Hy mutated"
        assert torch.equal(m_owner[1], _plane(20, device)), "endpoint magnetic adjoint owner Hz mutated"

    # -- reverse electric adjoint: left ghost adj -> right owner (add), zero --
    e_owner = [_plane(30, device), _plane(40, device)]
    e_ghost = [_plane(3, device), _plane(4, device)]
    e_staging = [_plane(0, device), _plane(0, device)]
    transport.exchange_electric_adjoint(
        ghost_adjoint_planes=e_ghost,
        owner_adjoint_planes=e_owner,
        staging_planes=e_staging,
    )
    if rank == 1:
        # rank 1 owner accumulates rank 0's ghost adjoint (3, 4).
        assert torch.equal(e_owner[0], _plane(33, device)), "electric adjoint owner Ey mismatch"
        assert torch.equal(e_owner[1], _plane(44, device)), "electric adjoint owner Ez mismatch"
        # Endpoint negative: rightmost rank has no right neighbour, so it never
        # ships its own ghost adjoint node and must not zero it.
        assert torch.equal(e_ghost[0], _plane(3, device)), "endpoint electric adjoint ghost Ey zeroed"
        assert torch.equal(e_ghost[1], _plane(4, device)), "endpoint electric adjoint ghost Ez zeroed"
    if rank == 0:
        assert torch.equal(e_ghost[0], _plane(0, device)), "electric adjoint ghost Ey not zeroed"
        assert torch.equal(e_ghost[1], _plane(0, device)), "electric adjoint ghost Ez not zeroed"
        # Endpoint negative: leftmost rank has no left neighbour, so its owner
        # receives nothing and stays at its initial value.
        assert torch.equal(e_owner[0], _plane(30, device)), "endpoint electric adjoint owner Ey mutated"
        assert torch.equal(e_owner[1], _plane(40, device)), "endpoint electric adjoint owner Ez mutated"

    # -- scalar all-reduce (shutoff-energy shape): 1 + 2 == 3 --
    total = transport.allreduce_scalar(rank + 1)
    assert torch.equal(total, torch.tensor(3.0, device=device, dtype=torch.float64)), (
        f"allreduce_scalar mismatch on rank {rank}: {total.item()}"
    )

    # -- adopted-group accept path: a second transport with matching rank/world --
    # size must adopt the live group in preflight (not re-initialise it) and pass
    # the world_size/rank/backend validation. Both ranks run this in lockstep so
    # the homogeneity all_gather inside preflight completes.
    adopted = NcclHaloTransport.from_env(expected_world_size=2)
    adopted.preflight()
    assert adopted._connected, "adopted transport failed to bind the live group"

    # Both handles share the one default process group. Tear the adopted handle
    # down first: destroy_process_group() releases the shared group for every
    # transport that adopted it. The original transport still reads
    # _connected == True at this point, but the group is gone -- a primitive call
    # on it must fail closed via the vanished-group guard rather than drive a
    # collective on a destroyed group. This is the safe teardown contract the
    # future coordinator must copy when it holds a rank-local transport per group.
    transport.barrier()
    adopted.teardown()
    assert not adopted._connected, "adopted transport did not disconnect on teardown"
    assert not torch.distributed.is_initialized(), "adopted teardown left the group alive"

    # transport still believes it is connected; the vanished-group guard must fire
    # (a pure local check, no collective) and flip the stale flag.
    try:
        transport.allreduce_scalar(1.0)
    except RuntimeError as error:
        assert "no longer connected" in str(error), f"unexpected teardown error: {error}"
    else:  # pragma: no cover - defensive: the guard must fire once the group is gone
        raise AssertionError("transport primitive did not fail closed after group teardown")
    assert not transport._connected, "vanished-group guard did not flip the stale flag"

    # teardown() remains idempotent even though the shared group is already gone.
    transport.teardown()
    assert not transport._connected, "primary transport did not disconnect on teardown"

    if rank == 0:
        print("NCCL_TRANSPORT_WORKER_OK")


if __name__ == "__main__":
    main()
