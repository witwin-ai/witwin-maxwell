"""Conformance and failure-matrix tests for the NCCL halo transport.

The guard/failure-matrix cases are pure host checks and run everywhere. The
halo round-trip conformance case launches the two-rank torchrun worker in
:mod:`_nccl_transport_worker` and asserts a clean exit, so the NCCL collectives
are exercised on real devices without asserting any timing.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from witwin.maxwell.fdtd.distributed.nccl_transport import NcclHaloTransport

_WORKER = Path(__file__).with_name("_nccl_transport_worker.py")
_FORWARD_WORKER = Path(__file__).with_name("_nccl_forward_worker.py")
_RANKDEATH_WORKER = Path(__file__).with_name("_nccl_rankdeath_worker.py")


def _torchrun(worker: Path, *, nproc: int = 2, timeout: int = 300, env: dict | None = None):
    run_env = dict(os.environ) if env is None else env
    run_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        str(worker),
    ]
    return subprocess.run(cmd, env=run_env, capture_output=True, text=True, timeout=timeout)


# -- failure matrix (host-only guards) ------------------------------------


def test_from_env_missing_launcher_env_raises():
    with pytest.raises(RuntimeError, match="torchrun one-process-per-GPU"):
        NcclHaloTransport.from_env(env={})


def test_from_env_partial_launcher_env_reports_missing():
    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        NcclHaloTransport.from_env(env={"RANK": "0", "WORLD_SIZE": "2"})


def test_from_env_world_size_mismatch_raises():
    env = {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0"}
    with pytest.raises(RuntimeError, match="does not match the configured device"):
        NcclHaloTransport.from_env(expected_world_size=2, env=env)


def test_constructor_rejects_degenerate_world():
    with pytest.raises(ValueError, match="world_size >= 2"):
        NcclHaloTransport(rank=0, world_size=1, local_rank=0)


def test_constructor_rejects_out_of_range_rank():
    with pytest.raises(ValueError, match="outside"):
        NcclHaloTransport(rank=2, world_size=2, local_rank=0)


def test_non_linux_platform_raises(monkeypatch):
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    monkeypatch.setattr(
        "witwin.maxwell.fdtd.distributed.nccl_transport.platform.system",
        lambda: "Windows",
    )
    with pytest.raises(RuntimeError, match="single-node Linux only"):
        transport.preflight()


def test_use_before_preflight_raises():
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    with pytest.raises(RuntimeError, match="before preflight"):
        transport.allreduce_scalar(1.0)


def _patch_adopted_group(monkeypatch, *, world_size, rank, backend):
    import witwin.maxwell.fdtd.distributed.nccl_transport as mod

    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(mod.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(mod.dist, "get_backend", lambda: backend)


def _skip_without_cuda_linux():
    if platform.system() != "Linux":
        pytest.skip("NCCL transport is single-node Linux only")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_preflight_rejects_adopted_group_world_size_mismatch(monkeypatch):
    _skip_without_cuda_linux()
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    _patch_adopted_group(monkeypatch, world_size=4, rank=0, backend="nccl")
    with pytest.raises(RuntimeError, match="world size"):
        transport.preflight()


def test_preflight_rejects_adopted_group_rank_mismatch(monkeypatch):
    _skip_without_cuda_linux()
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    _patch_adopted_group(monkeypatch, world_size=2, rank=1, backend="nccl")
    # Pin the rank branch's distinctive phrase: a bare "rank" also matches the
    # world-size mismatch message ("this rank expects ..."), so match the phrase
    # unique to the rank-mismatch raise site.
    with pytest.raises(RuntimeError, match="reporting rank"):
        transport.preflight()


def test_preflight_rejects_adopted_group_non_nccl_backend(monkeypatch):
    _skip_without_cuda_linux()
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    _patch_adopted_group(monkeypatch, world_size=2, rank=0, backend="gloo")
    with pytest.raises(RuntimeError, match="NCCL-backed"):
        transport.preflight()


def test_validate_adopted_group_accepts_composite_cuda_nccl_backend(monkeypatch):
    """A composite device->backend spec whose CUDA backend is NCCL is accepted.

    ``get_backend()`` reports ``"cpu:gloo,cuda:nccl"`` for a group built with
    per-device backends; the CUDA collectives that drive the halos are still
    NCCL, so ``_validate_adopted_group`` must not reject it. Exercised directly
    (no CUDA/live group needed): the full ``preflight`` accept path continues
    into the homogeneity all-gather, which is covered by the two-rank worker.
    """

    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    _patch_adopted_group(
        monkeypatch, world_size=2, rank=0, backend="cpu:gloo,cuda:nccl"
    )
    transport._validate_adopted_group()  # must not raise


def test_validate_adopted_group_rejects_composite_without_cuda_nccl(monkeypatch):
    """A composite spec whose CUDA backend is not NCCL still fails closed."""

    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    _patch_adopted_group(
        monkeypatch, world_size=2, rank=0, backend="cpu:gloo,cuda:gloo"
    )
    with pytest.raises(RuntimeError, match="NCCL-backed"):
        transport._validate_adopted_group()


def test_neighbor_topology_is_chain():
    first = NcclHaloTransport(rank=0, world_size=3, local_rank=0)
    middle = NcclHaloTransport(rank=1, world_size=3, local_rank=1)
    last = NcclHaloTransport(rank=2, world_size=3, local_rank=2)
    assert (first.left_rank, first.right_rank) == (None, 1)
    assert (middle.left_rank, middle.right_rank) == (0, 2)
    assert (last.left_rank, last.right_rank) == (1, None)


# -- no-silent-fallback guard ---------------------------------------------


def test_nccl_transport_without_launcher_raises_and_never_builds_p2p(monkeypatch):
    """``transport="nccl"`` in the single-process runtime must fail closed.

    The single-process ``DistributedFDTD`` coordinator cannot drive the
    one-process-per-GPU NCCL shape, and it must raise an explicit torchrun error
    rather than silently constructing the in-process CUDA P2P transport (which
    would run a different execution than the user requested).
    """

    import witwin.maxwell as mw
    import witwin.maxwell.fdtd.distributed.solver as solver_module
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)

    def _forbidden_p2p(*args, **kwargs):
        raise AssertionError(
            "transport='nccl' must not silently fall back to CudaP2PHaloTransport."
        )

    monkeypatch.setattr(solver_module, "CudaP2PHaloTransport", _forbidden_p2p)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    parallel = mw.FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
    )

    with pytest.raises(RuntimeError, match="torchrun"):
        DistributedFDTD(scene, frequency=1.0e9, parallel=parallel)


def test_public_simulation_nccl_transport_raises_before_solver_allocation():
    """The public Simulation path surfaces the same torchrun guard.

    Driven through the public ``Simulation.prepare()`` entrypoint rather than the
    private solver builder: prepare() reaches ``DistributedFDTD`` construction,
    which raises the torchrun ``RuntimeError`` before any shard is allocated
    (``init_field`` runs only after a successful construction). This keeps the
    assertion on the supported public surface.
    """

    _skip_without_cuda_linux()

    import witwin.maxwell as mw

    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    parallel = mw.FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
    )
    simulation = mw.Simulation.fdtd(scene, frequency=1.0e9, parallel=parallel)

    with pytest.raises(RuntimeError, match="torchrun"):
        simulation.prepare()


# -- two-rank conformance (torchrun) --------------------------------------


@pytest.mark.nccl
def test_two_rank_halo_roundtrip():
    if platform.system() != "Linux":
        pytest.skip("NCCL transport is single-node Linux only")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL backend is unavailable")

    env = dict(os.environ)
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(_WORKER),
    ]
    completed = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        "two-rank NCCL worker failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "NCCL_TRANSPORT_WORKER_OK" in completed.stdout, completed.stdout


def _skip_without_two_gpu_nccl():
    if platform.system() != "Linux":
        pytest.skip("NCCL transport is single-node Linux only")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL backend is unavailable")


@pytest.mark.nccl
def test_two_rank_nccl_forward_matches_single_gpu():
    """End-to-end NCCL forward solve matches the single-GPU reference.

    The two-rank worker builds a ShardEngine per rank, runs the distributed
    coordinator loop over ``NcclHaloTransport``, and rank 0 gathers the global
    full-field DFT and compares it to an independent single-GPU ``FDTD`` at the
    plan's monitor tolerances (rtol 5e-5 / atol 5e-6) -- the same gate the
    in-process CUDA P2P conformance leg uses. A nonzero exit means the field
    parity or the halo/gather transport is wrong.
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun(_FORWARD_WORKER, timeout=300)
    assert completed.returncode == 0, (
        "two-rank NCCL forward worker failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "NCCL_FORWARD_WORKER_OK" in completed.stdout, completed.stdout


@pytest.mark.nccl
def test_nccl_rank_death_propagates_failure():
    """A dead peer mid-run must surface as a bounded nonzero exit, not a hang.

    Both ranks join the group; rank 1 hard-exits before the first solve halo
    collective. The survivor's next collective has no peer, so torchrun failure
    propagation and/or the ProcessGroupNCCL watchdog timeout (configured short via
    ``FDTDParallelConfig.timeout_s``) aborts the launch. The subprocess timeout is
    the backstop: if it fired, the survivor hung and this test fails.
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun(_RANKDEATH_WORKER, timeout=180)
    assert completed.returncode != 0, (
        "peer death did not propagate as a failure\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "UNEXPECTED_SURVIVAL" not in completed.stdout, completed.stdout


# -- coordinator-level failure matrix (host-only) -------------------------


def test_coordinator_nccl_world_size_mismatch_raises(monkeypatch):
    """The coordinator rejects a launch whose world size != configured devices.

    Driven at ``DistributedFDTD`` construction: ``NcclHaloTransport.from_env``
    validates ``WORLD_SIZE`` against the configured two-device count and raises
    before any process group is created, so a mismatched launcher fails closed on
    the host without touching CUDA.
    """

    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")

    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    parallel = mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl")
    with pytest.raises(RuntimeError, match="does not match the configured device"):
        DistributedFDTD(scene, frequency=1.0e9, parallel=parallel)


def test_parallel_config_timeout_validation():
    import witwin.maxwell as mw

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s=0.0)
    with pytest.raises(TypeError, match="timeout_s must be a number"):
        mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s="soon")
    # A valid positive timeout is accepted and coerced to float.
    config = mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s=30)
    assert config.timeout_s == 30.0
