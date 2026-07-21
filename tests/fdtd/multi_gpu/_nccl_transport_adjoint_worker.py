"""torchrun worker: NCCL reverse-halo discrete-transpose identity across 2 ranks.

Launched by ``test_nccl_transport.py`` via ``torchrun --nproc-per-node=2``. The
forward Yee x halos are pure plane copies (owner -> neighbour ghost); their
adjoint must satisfy the discrete-transpose inner-product identity

    <A x, y> == <x, A^T y>

where ``A`` is the forward halo exchange and ``A^T`` is the reverse (adjoint)
exchange. The two inner products live on opposite ranks of a pair, so each side is
formed locally and combined with an all-reduce; because every halo op is a pure
copy the identity holds bitwise (atol == 0). The worker also pins bitwise
determinism across repeats and the ghost-adjoint-zero invariant.

The engine-facing ``exchange_magnetic_adjoint`` / ``exchange_electric_adjoint`` are
driven through lightweight ``SimpleNamespace`` engines built on a real partition
layout, matching the in-process P2P adjoint-transport unit tests. Setting
``NCCL_TRANSPOSE_FALSIFY`` to ``magnetic`` or ``electric`` skips the corresponding
adjoint accumulation so the transpose identity assertion fails: this is the
falsification gate proving the identity check has teeth (the launcher asserts the
worker then exits nonzero).

A clean run prints ``NCCL_TRANSPOSE_ADJOINT_WORKER_OK`` on rank 0. Any mismatch or
the falsification path raises, yielding a nonzero process exit. No timing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from witwin.maxwell.fdtd.distributed.nccl_transport import NcclHaloTransport
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan

_GLOBAL_SHAPE = (12, 7, 6)
_DTYPE = torch.float64


def _build_engine(transport: NcclHaloTransport):
    """A minimal engine on this rank's real partition layout.

    Carries only what the transport's forward/adjoint exchanges read: the padded
    Yee field storage, the owned slices, the bound device, a compute stream, and
    the halo-completion events.
    """

    plan = FDTDPartitionPlan(
        global_shape=_GLOBAL_SHAPE,
        devices=tuple(f"cuda:{r}" for r in range(transport.world_size)),
    )
    layout = plan.shard_layouts[transport.rank]
    device = transport.device
    with torch.cuda.device(device):
        solver = SimpleNamespace(
            Ey=torch.zeros(layout.component("Ey").local_shape, device=device, dtype=_DTYPE),
            Ez=torch.zeros(layout.component("Ez").local_shape, device=device, dtype=_DTYPE),
            Hy=torch.zeros(layout.component("Hy").local_shape, device=device, dtype=_DTYPE),
            Hz=torch.zeros(layout.component("Hz").local_shape, device=device, dtype=_DTYPE),
        )
        engine = SimpleNamespace(
            rank=transport.rank,
            device=device,
            layout=layout,
            solver=solver,
            compute_stream=torch.cuda.Stream(device=device),
            electric_ready=torch.cuda.Event(),
            electric_received=torch.cuda.Event(),
            magnetic_ready=torch.cuda.Event(),
            magnetic_received=torch.cuda.Event(),
        )
    return engine


def _adjoint_state(engine):
    device = engine.device
    with torch.cuda.device(device):
        return {
            name: torch.zeros(getattr(engine.solver, name).shape, device=device, dtype=_DTYPE)
            for name in ("Ey", "Ez", "Hy", "Hz")
        }


def _rand_like(plane, *, seed):
    gen = torch.Generator(device=plane.device).manual_seed(seed)
    return torch.rand(plane.shape, generator=gen, device=plane.device, dtype=_DTYPE)


def _combined(local_value: float, device) -> float:
    """Sum a per-rank scalar across the world (identity's two halves live apart)."""

    tensor = torch.tensor(float(local_value), device=device, dtype=_DTYPE)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _magnetic_identity(transport, engine, *, falsify: bool) -> None:
    """Discrete transpose of the forward magnetic halo (left owner -> right ghost)."""

    ns_cell = engine.layout.storage_cell_owned
    # Forward input x on the left owner's last cell plane; run the forward halo so
    # the right neighbour's low ghost carries A x.
    if transport.left_rank is None:  # left rank of the pair: owns x
        x_hy = _rand_like(engine.solver.Hy[ns_cell.stop - 1], seed=1001)
        x_hz = _rand_like(engine.solver.Hz[ns_cell.stop - 1], seed=1002)
        engine.solver.Hy[ns_cell.stop - 1].copy_(x_hy)
        engine.solver.Hz[ns_cell.stop - 1].copy_(x_hz)
    with torch.cuda.device(engine.device):
        engine.compute_stream.wait_stream(torch.cuda.current_stream(engine.device))
    transport.exchange_magnetic((engine,))
    engine.compute_stream.synchronize()
    transport.barrier()

    # Adjoint cotangent y on the right ghost; the left owner adjoint starts at zero
    # so the accumulation output is exactly A^T y. lhs = <A x, y> is formed on the
    # right rank BEFORE the adjoint exchange, because the exchange (correctly) zeros
    # the shipped-left ghost cotangent as part of the transpose.
    state = _adjoint_state(engine)
    lhs = 0.0
    if transport.right_rank is None:  # right rank of the pair: owns ghost cotangent
        state["Hy"][0].copy_(_rand_like(state["Hy"][0], seed=2001))
        state["Hz"][0].copy_(_rand_like(state["Hz"][0], seed=2002))
        engine.compute_stream.synchronize()
        lhs = float(
            torch.dot(engine.solver.Hy[0].flatten(), state["Hy"][0].flatten())
            + torch.dot(engine.solver.Hz[0].flatten(), state["Hz"][0].flatten())
        )
    transport.prepare_adjoint_staging((engine,), {engine.rank: state})
    with torch.cuda.device(engine.device):
        engine.compute_stream.wait_stream(torch.cuda.current_stream(engine.device))
    # The collective runs on every rank in lockstep (skipping it on one rank would
    # deadlock its peer). Falsification instead drops the accumulated owner on the
    # left rank AFTER the exchange, so A^T y is wrong and the identity assertion
    # must fire -- proving the check catches a broken adjoint accumulation.
    transport.exchange_magnetic_adjoint((engine,), {engine.rank: state})
    engine.compute_stream.synchronize()
    if falsify and transport.right_rank is not None:
        state["Hy"][ns_cell.stop - 1].zero_()
        state["Hz"][ns_cell.stop - 1].zero_()
    transport.barrier()

    # rhs = <x, A^T y> lives on the left rank (holds x and the accumulated owner).
    rhs = 0.0
    if transport.right_rank is None:  # right rank
        # Ghost-adjoint-zero invariant: the shipped-left ghost must be zeroed.
        assert torch.count_nonzero(state["Hy"][0]) == 0
        assert torch.count_nonzero(state["Hz"][0]) == 0
    else:  # left rank
        owner_hy = state["Hy"][ns_cell.stop - 1]
        owner_hz = state["Hz"][ns_cell.stop - 1]
        rhs = float(
            torch.dot(engine.solver.Hy[ns_cell.stop - 1].flatten(), owner_hy.flatten())
            + torch.dot(engine.solver.Hz[ns_cell.stop - 1].flatten(), owner_hz.flatten())
        )
    total_lhs = _combined(lhs, engine.device)
    total_rhs = _combined(rhs, engine.device)
    assert total_lhs == total_rhs, (
        f"magnetic transpose identity broken: <A x, y>={total_lhs!r} != "
        f"<x, A^T y>={total_rhs!r}"
    )


def _electric_identity(transport, engine, *, falsify: bool) -> None:
    """Discrete transpose of the forward electric halo (right owner -> left ghost)."""

    ns = engine.layout.storage_node_owned
    if transport.right_rank is None:  # right rank of the pair: owns x (first node)
        x_ey = _rand_like(engine.solver.Ey[ns.start], seed=3001)
        x_ez = _rand_like(engine.solver.Ez[ns.start], seed=3002)
        engine.solver.Ey[ns.start].copy_(x_ey)
        engine.solver.Ez[ns.start].copy_(x_ez)
    with torch.cuda.device(engine.device):
        engine.compute_stream.wait_stream(torch.cuda.current_stream(engine.device))
    transport.exchange_electric((engine,))
    engine.compute_stream.synchronize()
    transport.barrier()

    state = _adjoint_state(engine)
    ghost = ns.stop
    lhs = 0.0
    if transport.left_rank is None:  # left rank of the pair: owns ghost cotangent
        state["Ey"][ghost].copy_(_rand_like(state["Ey"][ghost], seed=4001))
        state["Ez"][ghost].copy_(_rand_like(state["Ez"][ghost], seed=4002))
        engine.compute_stream.synchronize()
        lhs = float(
            torch.dot(engine.solver.Ey[ghost].flatten(), state["Ey"][ghost].flatten())
            + torch.dot(engine.solver.Ez[ghost].flatten(), state["Ez"][ghost].flatten())
        )
    transport.prepare_adjoint_staging((engine,), {engine.rank: state})
    with torch.cuda.device(engine.device):
        engine.compute_stream.wait_stream(torch.cuda.current_stream(engine.device))
    transport.exchange_electric_adjoint((engine,), {engine.rank: state})
    engine.compute_stream.synchronize()
    if falsify and transport.left_rank is not None:
        state["Ey"][ns.start].zero_()
        state["Ez"][ns.start].zero_()
    transport.barrier()

    rhs = 0.0
    if transport.left_rank is None:  # left rank held A x in the ghost and y
        assert torch.count_nonzero(state["Ey"][ghost]) == 0
        assert torch.count_nonzero(state["Ez"][ghost]) == 0
    else:  # right rank holds x and the accumulated owner (first node)
        owner_ey = state["Ey"][ns.start]
        owner_ez = state["Ez"][ns.start]
        rhs = float(
            torch.dot(engine.solver.Ey[ns.start].flatten(), owner_ey.flatten())
            + torch.dot(engine.solver.Ez[ns.start].flatten(), owner_ez.flatten())
        )
    total_lhs = _combined(lhs, engine.device)
    total_rhs = _combined(rhs, engine.device)
    assert total_lhs == total_rhs, (
        f"electric transpose identity broken: <A x, y>={total_lhs!r} != "
        f"<x, A^T y>={total_rhs!r}"
    )


def _determinism(transport, engine) -> None:
    """Bitwise-identical accumulated owner across two adjoint repeats."""

    ns_cell = engine.layout.storage_cell_owned

    def one_pass():
        state = _adjoint_state(engine)
        if transport.right_rank is None:
            state["Hy"][0].copy_(_rand_like(state["Hy"][0], seed=5001))
            state["Hz"][0].copy_(_rand_like(state["Hz"][0], seed=5002))
        transport.prepare_adjoint_staging((engine,), {engine.rank: state})
        with torch.cuda.device(engine.device):
            engine.compute_stream.wait_stream(torch.cuda.current_stream(engine.device))
        transport.exchange_magnetic_adjoint((engine,), {engine.rank: state})
        engine.compute_stream.synchronize()
        transport.barrier()
        return state["Hy"][ns_cell.stop - 1].clone(), state["Hz"][ns_cell.stop - 1].clone()

    first = one_pass()
    second = one_pass()
    for a, b in zip(first, second):
        assert torch.equal(a, b), "reverse magnetic halo is not bitwise deterministic"


def main() -> None:
    falsify_mode = os.environ.get("NCCL_TRANSPOSE_FALSIFY", "").strip().lower()
    transport = NcclHaloTransport.from_env(expected_world_size=2)
    transport.preflight()
    engine = _build_engine(transport)

    _magnetic_identity(transport, engine, falsify=(falsify_mode == "magnetic"))
    _electric_identity(transport, engine, falsify=(falsify_mode == "electric"))
    if not falsify_mode:
        _determinism(transport, engine)

    transport.barrier()
    transport.teardown()
    if transport.rank == 0 and not falsify_mode:
        print("NCCL_TRANSPOSE_ADJOINT_WORKER_OK")


if __name__ == "__main__":
    main()
