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
_TRANSPOSE_WORKER = Path(__file__).with_name("_nccl_transport_adjoint_worker.py")
_ADJOINT_WORKER = Path(__file__).with_name("_nccl_adjoint_worker.py")


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
def test_two_rank_nccl_step_timing_emits_per_rank_json(tmp_path):
    """Opt-in step-rate instrumentation emits one JSON per rank from the worker.

    Enabling ``WITWIN_FDTD_STEP_TIMING`` runs the forward worker's collective
    timing pass and writes ``step_timing_rank{r}.json`` on every rank. This guards
    the worker wiring (the unit test covers the instrument logic); it asserts the
    per-rank artifact schema only, never a wall-clock number, so it is safe on
    shared GPUs. The default (env unset) forward run above must stay a no-op pass.
    """

    _skip_without_two_gpu_nccl()
    env = dict(os.environ)
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["WITWIN_FDTD_STEP_TIMING"] = "1"
    env["WITWIN_FDTD_STEP_TIMING_DIR"] = str(tmp_path)
    env["WITWIN_FDTD_STEP_TIMING_STEPS"] = "32"
    completed = _torchrun(_FORWARD_WORKER, timeout=300, env=env)
    assert completed.returncode == 0, (
        "timing-enabled NCCL forward worker failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "NCCL_FORWARD_WORKER_OK" in completed.stdout, completed.stdout
    import json as _json

    for rank in (0, 1):
        path = tmp_path / f"step_timing_rank{rank}.json"
        assert path.exists(), f"missing per-rank timing artifact for rank {rank}"
        payload = _json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "witwin.fdtd.step_timing/1"
        assert payload["enabled"] is True
        assert payload["rank"] == rank
        assert payload["steps"] == 32
        assert "steps_per_second" in payload


@pytest.mark.nccl
def test_two_rank_nccl_reverse_halo_transpose_identity():
    """The NCCL reverse halos are the discrete transpose of the forward halos.

    The two-rank worker forms ``<A x, y>`` and ``<x, A^T y>`` for both the magnetic
    and electric Yee x halos (each inner product's two halves live on opposite
    ranks and are combined by an all-reduce), asserts bitwise equality, checks the
    ghost-adjoint-zero invariant, and pins bitwise determinism across repeats. A
    nonzero exit means the reverse exchange is not the forward's transpose.
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun(_TRANSPOSE_WORKER, timeout=300)
    assert completed.returncode == 0, (
        "two-rank NCCL transpose-identity worker failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "NCCL_TRANSPOSE_ADJOINT_WORKER_OK" in completed.stdout, completed.stdout


@pytest.mark.nccl
@pytest.mark.parametrize("mode", ("magnetic", "electric"))
def test_two_rank_nccl_reverse_transpose_identity_falsification(mode):
    """No-op'ing one adjoint accumulation must break the transpose identity.

    The falsification gate: with ``NCCL_TRANSPOSE_FALSIFY`` set, the worker skips
    the named reverse accumulation, so ``<x, A^T y>`` no longer equals ``<A x, y>``
    and the worker's identity assertion fires -> nonzero exit. If the exit were
    clean the identity check would be vacuous, so a zero return fails this test.
    """

    _skip_without_two_gpu_nccl()
    env = dict(os.environ)
    env["NCCL_TRANSPOSE_FALSIFY"] = mode
    completed = _torchrun(_TRANSPOSE_WORKER, timeout=300, env=env)
    assert completed.returncode != 0, (
        "falsified NCCL transpose identity unexpectedly passed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )


def _torchrun_adjoint(mode: str, *, timeout: int):
    env = dict(os.environ)
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"] = "false"
    env["WITWIN_NCCL_ADJ_MODE"] = mode
    return _torchrun(_ADJOINT_WORKER, timeout=timeout, env=env)


@pytest.mark.nccl
@pytest.mark.parametrize(
    "mode,timeout",
    (
        ("standard", 400),
        ("cpml", 400),
        ("cpml_psi", 900),
        ("determinism", 600),
    ),
)
def test_two_rank_nccl_adjoint_parity(mode, timeout):
    """Per-rank collective NCCL reverse driver matches the single-GPU adjoint.

    The two-rank worker drives ``run_nccl_distributed_reverse`` (per-rank forward
    with checkpoints, NCCL forward-replay dict halos, local separable point-monitor
    seed, transposed NCCL adjoint reverse, grad_eps gather to rank 0 + rank-0
    pullback) and, on rank 0, compares the world-summed objective and the gathered
    grad_eps material gradient against an independent single-GPU adjoint on the same
    scene. ``standard`` is an open-boundary cross-seam objective; ``cpml`` an
    x-CPML interior probe; ``cpml_psi`` drives the probe deep into the high x-PML so
    the reverse threads the objective back through the CPML psi recursion (the
    worker asserts the world-max psi cotangent is a significant fraction of the E/H
    adjoint scale before the parity gate); ``determinism`` reruns the reverse twice
    and asserts the gathered grad_eps is bitwise identical. Gates mirror
    ``test_adjoint_parity_cpml.py`` (loss rtol 5e-5/atol 5e-6; grad rtol 1e-4 with
    an atol floor 1e-6*max|grad|).
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun_adjoint(mode, timeout=timeout)
    assert completed.returncode == 0, (
        f"NCCL adjoint driver [{mode}] failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert f"NCCL_ADJOINT_WORKER_OK[{mode}" in completed.stdout, completed.stdout


@pytest.mark.nccl
@pytest.mark.parametrize(
    "mode,timeout",
    (
        ("falsify_mag_halo", 400),
        ("falsify_elec_halo", 400),
        ("falsify_psi", 900),
    ),
)
def test_two_rank_nccl_adjoint_falsification(mode, timeout):
    """No-op'ing a reverse halo or zeroing the psi carry must break parity.

    Each falsification perturbs only the distributed reverse (the single-GPU
    reference is untouched) and asserts the 2-GPU gradient moves off the reference
    by >= 1e-3 relative -- well above the ~1e-7 baseline drift and below the smallest
    real error, so the parity gate is non-vacuous. The worker prints OK only when
    the perturbed run has genuinely diverged; a clean parity here would fail the
    worker (nonzero exit), so this test requires exit 0 AND the OK token.
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun_adjoint(mode, timeout=timeout)
    assert completed.returncode == 0, (
        f"NCCL adjoint falsification [{mode}] failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert f"NCCL_ADJOINT_WORKER_OK[{mode}" in completed.stdout, completed.stdout


@pytest.mark.nccl
def test_two_rank_nccl_adjoint_unsupported_rejects_without_deadlock():
    """An unsupported-adjoint scene rejects cleanly on all ranks (no hang).

    The worker drives a trainable-density scene on a legacy graded-sigma absorber,
    which has no verified distributed reverse core. The reject is symmetric (the
    scene is identical on every rank) and fires before any halo collective, so both
    ranks return promptly rather than one blocking in a collective the other
    abandoned. The subprocess timeout is the deadlock witness: a hang would exceed
    it. A clean rejection prints the OK token on rank 0.
    """

    _skip_without_two_gpu_nccl()
    completed = _torchrun_adjoint("guard_deadlock", timeout=200)
    assert completed.returncode == 0, (
        "NCCL adjoint deadlock-freedom guard failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert "NCCL_ADJOINT_WORKER_OK[guard_deadlock]" in completed.stdout, completed.stdout


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


# -- NCCL forward fail-closed fences (host-only) --------------------------
#
# The one-process-per-GPU NCCL forward path drives only the standard field solve
# and the sized full-field gather. Monitors, coupled circuit/network/wire/port
# runtimes, the trainable-density adjoint, and field shutoff are guarded out until
# they are wired over NCCL. These pin each fence's specific message at
# construction/solve time on the host: setting RANK/WORLD_SIZE/LOCAL_RANK lets
# ``NcclHaloTransport.from_env`` bind without a torchrun launch, and the fences run
# in ``DistributedFDTD.__init__`` (``_validate_nccl_capabilities``) or at the top of
# ``solve`` before any CUDA allocation, so no GPU is required.


def _set_nccl_launcher_env(monkeypatch, *, world_size=2):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("LOCAL_RANK", "0")


def _nccl_parallel():
    import witwin.maxwell as mw

    return mw.FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
        gather_fields=True,
        result_device="cuda:0",
    )


def _nccl_base_scene():
    """A scene the NCCL forward path fully accepts (no guarded feature)."""

    import witwin.maxwell as mw

    return mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )


def test_nccl_forward_rejects_monitors(monkeypatch):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    _set_nccl_launcher_env(monkeypatch)
    scene = _nccl_base_scene()
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)))

    with pytest.raises(ValueError, match="per-monitor payload gather"):
        DistributedFDTD(scene, frequency=1.0e9, parallel=_nccl_parallel())


def _coupled_wire_scene():
    import witwin.maxwell as mw

    scene = _nccl_base_scene()
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)),
            radius=1.0e-3,
            conductor=mw.WireConductor.pec(),
        )
    )
    return scene


def _coupled_port_scene():
    import witwin.maxwell as mw

    scene = _nccl_base_scene()
    scene.add_port(
        mw.LumpedPort(
            name="port",
            positive=(0.0, 0.0, 0.1),
            negative=(0.0, 0.0, -0.1),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(0.0, 0.0, -0.05), size=(0.2, 0.2, 0.0)),
            reference_impedance=50.0,
        )
    )
    return scene


@pytest.mark.parametrize(
    "scene_builder",
    (_coupled_wire_scene, _coupled_port_scene),
    ids=("wire", "port"),
)
def test_nccl_forward_rejects_coupled_runtimes(monkeypatch, scene_builder):
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    _set_nccl_launcher_env(monkeypatch)
    scene = scene_builder()

    with pytest.raises(ValueError, match="owner-resident circuit/network/wire/port"):
        DistributedFDTD(scene, frequency=1.0e9, parallel=_nccl_parallel())


def test_nccl_forward_rejects_trainable_density(monkeypatch):
    import torch

    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    _set_nccl_launcher_env(monkeypatch)
    scene = _nccl_base_scene()
    density = torch.rand((4, 4, 4), dtype=torch.float32, requires_grad=True)
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            density=density,
            eps_bounds=(1.0, 4.0),
        )
    )

    with pytest.raises(ValueError, match="Multi-GPU NCCL adjoint"):
        DistributedFDTD(scene, frequency=1.0e9, parallel=_nccl_parallel())


def test_nccl_forward_rejects_field_shutoff(monkeypatch):
    from witwin.maxwell.fdtd.distributed import DistributedFDTD

    _set_nccl_launcher_env(monkeypatch)
    # A fully accepted scene constructs on the host; the shutoff fence is at the top
    # of solve(), before init_field(), so it raises without any CUDA allocation.
    distributed = DistributedFDTD(
        _nccl_base_scene(), frequency=1.0e9, parallel=_nccl_parallel()
    )

    with pytest.raises(ValueError, match="does not support field shutoff"):
        distributed.solve(time_steps=8, shutoff=0.5)


# -- NCCL gather result_device fail-fast (host-only) -----------------------


def test_gather_component_slabs_rejects_non_bound_result_device(monkeypatch):
    """rank 0 must receive peer slabs on its NCCL-bound device.

    ``gather_component_slabs`` posts ``dist.recv`` into buffers on ``result_device``;
    NCCL requires that device to be the rank's bound device. A ``result_device`` on
    any other device fails fast with the constraint named, before any collective.
    Driven as a pure host check: the connected flag and layouts are set directly so
    the validation runs without a live process group.
    """

    import witwin.maxwell.fdtd.distributed.nccl_transport as mod

    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    transport = NcclHaloTransport(rank=0, world_size=2, local_rank=0)
    transport._connected = True
    transport._shard_layouts = (object(), object())

    with pytest.raises(ValueError, match="NCCL-bound device"):
        transport.gather_component_slabs(
            engines=(object(),),
            component="Ez",
            local_values=(None,),
            result_device=torch.device("cuda:1"),
            global_nx=13,
        )


def test_parallel_config_timeout_validation():
    import witwin.maxwell as mw

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s=0.0)
    with pytest.raises(TypeError, match="timeout_s must be a number"):
        mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s="soon")
    # A valid positive timeout is accepted and coerced to float.
    config = mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"), transport="nccl", timeout_s=30)
    assert config.timeout_s == 30.0
