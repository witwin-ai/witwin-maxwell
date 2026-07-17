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
    if rank == 1:
        # rank 1's ghost adjoint must be zeroed after shipping it left.
        assert torch.equal(m_ghost[0], _plane(0, device)), "magnetic adjoint ghost Hy not zeroed"
        assert torch.equal(m_ghost[1], _plane(0, device)), "magnetic adjoint ghost Hz not zeroed"

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
    if rank == 0:
        assert torch.equal(e_ghost[0], _plane(0, device)), "electric adjoint ghost Ey not zeroed"
        assert torch.equal(e_ghost[1], _plane(0, device)), "electric adjoint ghost Ez not zeroed"

    # -- scalar all-reduce (shutoff-energy shape): 1 + 2 == 3 --
    total = transport.allreduce_scalar(rank + 1)
    assert torch.equal(total, torch.tensor(3.0, device=device, dtype=torch.float64)), (
        f"allreduce_scalar mismatch on rank {rank}: {total.item()}"
    )

    transport.barrier()
    transport.teardown()
    if rank == 0:
        print("NCCL_TRANSPORT_WORKER_OK")


if __name__ == "__main__":
    main()
