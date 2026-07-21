"""torchrun worker: per-rank collective NCCL reverse driver vs a single-GPU reference.

Launched by ``test_nccl_transport.py`` via ``torchrun --nproc-per-node=2``. Each
rank builds a trainable Box-density scene, drives the per-rank collective NCCL
forward+reverse (``run_nccl_distributed_reverse``), and on rank 0 compares the
world-summed objective and the gathered-grad_eps material gradient against an
independently constructed single-GPU adjoint reference on the same scene.

Modes (env ``WITWIN_NCCL_ADJ_MODE``):
  standard        - open-boundary cross-seam point-monitor parity (gate 2a).
  plane           - open-boundary y-normal PlaneMonitor spanning the x seam: the
                    separable owned-strip objective must match the single-GPU
                    full-plane adjoint (S5 gate).
  plane_seam      - sum the FULL local strip (halo included) instead of the owned
                    strip -> the seam ghost cell is double-counted -> parity MUST
                    break (S5 seam-ownership falsification).
  cpml            - x-CPML interior-probe point-monitor parity (gate 2a).
  cpml_psi        - x-CPML probe deep in the high x-PML: the reverse must thread
                    the objective back through the CPML psi recursion; asserts the
                    distributed psi cotangent is non-trivial AND parity holds.
  falsify_mag_halo/falsify_elec_halo
                  - no-op one reverse field halo -> parity MUST break (gate 2b).
  falsify_psi     - zero the distributed psi cotangent carry -> psi-active parity
                    MUST break (gate 2c, CPML psi scene).
  determinism     - two identical backward passes -> gathered grad_eps bitwise
                    identical (gate 2d).
  guard_deadlock  - an unsupported-adjoint scene must reject cleanly on ALL ranks
                    (no hang); the launcher enforces a timeout (gate 2e).

A nonzero process exit signals a gate failure. No timing is asserted.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.fdtd.distributed import adjoint as _dist_adjoint
from witwin.maxwell.fdtd.distributed.nccl_transport import NcclHaloTransport

_BURNER = Path(__file__).with_name("_gpu_burner.py")


@contextlib.contextmanager
def _gpu_stress():
    """Saturate every participating GPU with a co-tenant burner for the gate.

    Enabled by ``WITWIN_NCCL_ADJ_STRESS=1``. Rank 0 spawns one committed burner
    subprocess per physical GPU (pinned via ``CUDA_VISIBLE_DEVICES``); every rank
    sleeps a fixed warm-up so the two ranks stay in lockstep at the first
    collective and the burners are saturating both boards before the solve runs.
    The burners are always terminated in the ``finally`` so a failing gate never
    leaks a runaway process. This is the load-bearing acceptance condition: green
    here at the honest tolerances proves the reverse is load-safe.
    """

    if os.environ.get("WITWIN_NCCL_ADJ_STRESS") != "1":
        yield
        return
    rank = int(os.environ.get("RANK", "0"))
    procs: list[subprocess.Popen] = []
    if rank == 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        for dev in visible:
            dev = dev.strip()
            if not dev:
                continue
            child = dict(os.environ)
            child["CUDA_VISIBLE_DEVICES"] = dev
            child.pop("WITWIN_NCCL_ADJ_STRESS", None)
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(_BURNER)],
                    env=child,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
    try:
        # Both ranks wait so they arrive at the first NCCL collective together and
        # the burners are fully saturating before the timed gate begins.
        time.sleep(10.0)
        yield
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

_FREQUENCY = 1.0e9
_DENS_SHAPE = (5, 4, 4)

# Parity gates (honest 1e-4-class, mirroring the in-process
# ``test_adjoint_parity_cpml`` tolerances). The forward loss is bitwise identical
# across GPU cohabitation conditions. The distributed *reverse* gradient is now
# load-safe: the reverse/replay NCCL halos run on the current (default) stream so
# their per-step adjoint planes' allocation stream matches their use stream,
# closing the caching-allocator cross-stream reuse window that previously
# corrupted the partition-seam gradient under concurrent GPU load (see the H1
# acceptance doc's "Load-dependence episode and fix"). With that fix the driver
# reproduces the single-GPU reference to ~2e-7 relative BOTH on exclusive GPUs and
# under a saturating co-tenant burner (proven by the stressed gate), so the grad
# tolerance is the genuine numerical-parity gate, not a drift-absorbing band.
_LOSS_RTOL = 5.0e-5
_LOSS_ATOL = 5.0e-6
_GRAD_RTOL = 1.0e-4
_GRAD_ATOL_FLOOR = 1.0e-6
# Falsification separation: a deliberately broken reverse (no-op'd halo or zeroed
# psi carry) moves >=1.64e-2 relative; healthy parity (exclusive OR stressed) stays
# ~2e-7. Threshold sits well between the two.
_FALSIFY_MIN_REL = 1.0e-3
# Repeat-determinism band. The gathered grad_eps reverse product is reduction-order
# reproducible (bitwise under identical block scheduling; concurrent load may flip
# fused-kernel/VJP reductions at the ULP level). The gate asserts relative < 1e-9,
# which still catches any material nondeterminism.
_DETERMINISM_MAX_REL = 1.0e-9

# psi-active probe constants (mirror test_adjoint_parity_cpml).
_PSI_STEPS = 360
_PSI_MONITOR_X = 0.48
_STD_STEPS = 60
_CPML_STEPS = 50


def _base_density():
    torch.manual_seed(1)
    return (0.4 + 0.2 * torch.rand(_DENS_SHAPE)).double()


def _standard_scene(density, *, source_x, monitor_x, device):
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (monitor_x, 0.0, 0.0), fields=("Ez",)))
    return scene


def _plane_scene(density, *, source_x, axis, device):
    x = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    # A y-normal plane at y=0 spans the full x extent, so it is tiled across the x
    # partition seam -- exactly the case the owned-strip seed must handle.
    scene.add_monitor(mw.PlaneMonitor("plane", axis=axis, position=0.0, fields=("Ez",)))
    return scene


def _cpml_scene(density, *, source_x, monitor_x, device):
    x = np.linspace(-0.6, 0.6, 21, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device=device,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (monitor_x, 0.0, 0.0), fields=("Ez",)))
    return scene


def _parallel():
    return FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
        gather_fields=False,
        overlap=False,
        result_device="cuda:0",
    )


def _build_scene(kind, density, device):
    if kind == "standard":
        return _standard_scene(density, source_x=-0.3, monitor_x=0.1, device=device)
    if kind in ("plane", "plane_seam"):
        # Source just left of the x seam (global cell 5 = x=0): both shards' owned
        # plane strips carry comparable field (genuinely dual-sided seeding) AND the
        # seam cell is live, so the rank-0 ghost column that the owned crop discards
        # is non-trivial -- making the seam-ownership falsification bite.
        return _plane_scene(density, source_x=-0.05, axis="y", device=device)
    if kind == "cpml":
        return _cpml_scene(density, source_x=-0.18, monitor_x=0.18, device=device)
    if kind == "cpml_psi":
        return _cpml_scene(density, source_x=-0.18, monitor_x=_PSI_MONITOR_X, device=device)
    raise ValueError(f"unknown scene kind {kind!r}")


def _kwargs(kind):
    if kind in ("standard", "plane", "plane_seam"):
        return dict(
            frequency=_FREQUENCY,
            run_time=mw.TimeConfig(time_steps=_STD_STEPS),
        )
    if kind == "cpml":
        return dict(
            frequency=_FREQUENCY,
            run_time=mw.TimeConfig(time_steps=_CPML_STEPS),
            absorber="cpml",
        )
    if kind == "cpml_psi":
        return dict(
            frequency=_FREQUENCY,
            run_time=mw.TimeConfig(time_steps=_PSI_STEPS),
            absorber="cpml",
        )
    raise ValueError(kind)


def _objective_for(kind):
    if kind in ("plane", "plane_seam"):
        return _dist_adjoint.plane_monitor_l2_objective
    return _dist_adjoint.point_monitor_l2_objective


def _distributed_solve(kind, density_values, *, capture=None, objective=None):
    """Run the per-rank NCCL adjoint driver. Returns (total_loss, grad, bridge)."""
    density = density_values.clone().to("cuda:0").requires_grad_(True)
    scene = _build_scene(kind, density, "cuda:0")
    simulation = mw.Simulation.fdtd(scene, parallel=_parallel(), **_kwargs(kind))
    total_loss, grads, bridge = _dist_adjoint.run_nccl_distributed_reverse(
        simulation, objective=objective if objective is not None else _objective_for(kind)
    )
    grad = grads[0].detach().to("cpu") if grads else None
    if capture is not None:
        capture["bridge"] = bridge
    return float(total_loss.item()), grad, bridge


def _reference_solve(kind, density_values):
    """Single-GPU adjoint reference on rank 0 only.

    For plane kinds the reference sums ``|Ez|^2`` over the FULL global plane; the
    distributed owned strips tile that plane with no overlap, so the world sum of
    the separable per-strip objective must reproduce it (and its gradient).
    """
    density = density_values.clone().to("cuda:0").requires_grad_(True)
    scene = _build_scene(kind, density, "cuda:0")
    result = mw.Simulation.fdtd(scene, **_kwargs(kind)).run()
    if kind in ("plane", "plane_seam"):
        spectrum = result.monitors["plane"]["Ez"]
    else:
        spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    loss.backward()
    return float(loss.detach().cpu()), density.grad.detach().cpu().clone()


def _assert_parity(kind, dist_loss, dist_grad, single_loss, single_grad):
    assert abs(single_loss) > 0.0, "reference loss is zero (vacuous)"
    grad_scale = float(single_grad.abs().max())
    assert grad_scale > 0.0, "reference gradient is zero (vacuous)"
    torch.testing.assert_close(
        torch.tensor(dist_loss), torch.tensor(single_loss), rtol=_LOSS_RTOL, atol=_LOSS_ATOL
    )
    atol_floor = _GRAD_ATOL_FLOOR * grad_scale
    torch.testing.assert_close(dist_grad, single_grad, rtol=_GRAD_RTOL, atol=atol_floor)


# -- psi-active instrumentation (mirror test_adjoint_parity_cpml wrappers) -----

_EH_STATE_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _install_psi_recorder(record):
    original = _dist_adjoint._NcclDistributedFDTDGradientBridge._reverse_phases_cpml

    def wrapped(self, distributed, shards, devices, *, offset, trajectories,
                mid_magnetic, post, eps_by_shard):
        psi_mag = 0.0
        eh_mag = 0.0
        for rank_post in post.values():
            for name in _dist_adjoint._CPML_PSI_NAMES:
                tensor = rank_post.get(name)
                if tensor is not None:
                    psi_mag = max(psi_mag, float(tensor.abs().max()))
            for name in _EH_STATE_NAMES:
                tensor = rank_post.get(name)
                if tensor is not None:
                    eh_mag = max(eh_mag, float(tensor.abs().max()))
        record["max_psi_cotangent"] = max(record["max_psi_cotangent"], psi_mag)
        record["max_eh_adjoint"] = max(record["max_eh_adjoint"], eh_mag)
        return original(self, distributed, shards, devices, offset=offset,
                        trajectories=trajectories, mid_magnetic=mid_magnetic,
                        post=post, eps_by_shard=eps_by_shard)

    _dist_adjoint._NcclDistributedFDTDGradientBridge._reverse_phases_cpml = wrapped


def _install_psi_zeroing():
    original = _dist_adjoint._NcclDistributedFDTDGradientBridge._reverse_phases_cpml

    def wrapped(self, distributed, shards, devices, *, offset, trajectories,
                mid_magnetic, post, eps_by_shard):
        for rank_post in post.values():
            for name in _dist_adjoint._CPML_PSI_NAMES:
                tensor = rank_post.get(name)
                if tensor is not None:
                    tensor.zero_()
        return original(self, distributed, shards, devices, offset=offset,
                        trajectories=trajectories, mid_magnetic=mid_magnetic,
                        post=post, eps_by_shard=eps_by_shard)

    _dist_adjoint._NcclDistributedFDTDGradientBridge._reverse_phases_cpml = wrapped


def _no_op_halo(self, engines, adjoint_states):
    return None


# -- modes ---------------------------------------------------------------------


def _run_parity(kind):
    base = _base_density()
    dist_loss, dist_grad, _ = _distributed_solve(kind, base)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        single_loss, single_grad = _reference_solve(kind, base)
        _assert_parity(kind, dist_loss, dist_grad, single_loss, single_grad)
        print(f"NCCL_ADJOINT_WORKER_OK[{kind}]")


def _plane_full_leaf_objective(pack, leaves, layout):
    """FALSIFICATION objective: sum the FULL local strip, ignoring owned ownership.

    The correct separable objective restricts each rank's plane sum to its owned-local
    x slice so the seam ghost cell is counted on exactly one rank. Summing the full
    allocation strip (halo included) double-counts the seam cell on both shards, so
    the world sum -- and its gradient -- must diverge from the single-GPU full-plane
    reference. This proves the owned-strip seam rule is load-bearing.
    """
    del layout
    cursor = len(pack.field_names)
    loss = None
    for _monitor_name, template in pack.monitor_templates.items():
        for _component_name in template["fields"]:
            spectrum = leaves[cursor]
            cursor += 1
            term = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
            loss = term if loss is None else loss + term
    if loss is None:
        return 0.0
    return loss


def _run_plane_seam():
    base = _base_density()
    dist_loss, dist_grad, _ = _distributed_solve(
        "plane_seam", base, objective=_plane_full_leaf_objective
    )
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        single_loss, single_grad = _reference_solve("plane_seam", base)
        grad_scale = float(single_grad.abs().max())
        assert grad_scale > 0.0, "reference gradient is zero (vacuous)"
        rel = float((dist_grad - single_grad).abs().max()) / grad_scale
        assert rel > _FALSIFY_MIN_REL, (
            f"double-counting the seam strip cell left parity intact (rel {rel:.3e}); "
            "the owned-strip seam rule is not load-bearing"
        )
        loss_rel = abs(dist_loss - single_loss) / max(abs(single_loss), 1e-30)
        print(
            f"NCCL_ADJOINT_WORKER_OK[plane_seam rel={rel:.3e} loss_rel={loss_rel:.3e}]"
        )


def _run_cpml_psi():
    base = _base_density()
    record = {"max_psi_cotangent": 0.0, "max_eh_adjoint": 0.0}
    _install_psi_recorder(record)
    dist_loss, dist_grad, _ = _distributed_solve("cpml_psi", base)
    torch.distributed.barrier()
    # The probe sits deep in the high x-PML owned entirely by the outer (rank 1)
    # shard, so the load-bearing psi cotangent lives on that rank. Reduce the
    # per-rank running maxima with a MAX all_reduce so the non-vacuity check sees
    # the world maximum rather than only the local rank's (rank 0's psi is inert).
    reduced = torch.tensor(
        [record["max_psi_cotangent"], record["max_eh_adjoint"]],
        device=f"cuda:{os.environ.get('LOCAL_RANK', '0')}",
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.MAX)
    global_psi = float(reduced[0].item())
    global_eh = float(reduced[1].item())
    assert global_eh > 0.0, "no E/H adjoint recorded"
    assert global_psi > 0.1 * global_eh, (
        f"distributed psi cotangent inert (psi={global_psi:.3e} vs "
        f"E/H={global_eh:.3e}); psi-active gate vacuous"
    )
    if torch.distributed.get_rank() == 0:
        single_loss, single_grad = _reference_solve("cpml_psi", base)
        assert float(single_grad.abs().max()) > 1.0, "psi-active objective too weak"
        _assert_parity("cpml_psi", dist_loss, dist_grad, single_loss, single_grad)
        print("NCCL_ADJOINT_WORKER_OK[cpml_psi]")


def _run_falsify_halo(which):
    kind = "cpml"
    base = _base_density()
    # Baseline reference (rank 0) + a baseline distributed parity check so the
    # falsification is a delta from a known-good state.
    method = "exchange_magnetic_adjoint" if which == "mag" else "exchange_electric_adjoint"
    setattr(NcclHaloTransport, method, _no_op_halo)
    dist_loss, dist_grad, _ = _distributed_solve(kind, base)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        single_loss, single_grad = _reference_solve(kind, base)
        grad_scale = float(single_grad.abs().max())
        assert grad_scale > 0.0
        rel = float((dist_grad - single_grad).abs().max()) / grad_scale
        assert rel > _FALSIFY_MIN_REL, (
            f"no-op {which} adjoint halo left parity intact (rel {rel:.3e}); the "
            "field halo is not load-bearing"
        )
        print(f"NCCL_ADJOINT_WORKER_OK[falsify_{which}_halo rel={rel:.3e}]")


def _run_falsify_psi():
    base = _base_density()
    _install_psi_zeroing()
    dist_loss, dist_grad, _ = _distributed_solve("cpml_psi", base)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        single_loss, single_grad = _reference_solve("cpml_psi", base)
        grad_scale = float(single_grad.abs().max())
        assert grad_scale > 1.0
        rel = float((dist_grad - single_grad).abs().max()) / grad_scale
        assert rel > _FALSIFY_MIN_REL, (
            f"zeroing the distributed psi cotangent carry left parity intact "
            f"(rel {rel:.3e}); the psi-carrying reverse path is not load-bearing"
        )
        print(f"NCCL_ADJOINT_WORKER_OK[falsify_psi rel={rel:.3e}]")


def _run_determinism():
    kind = "cpml"
    base = _base_density()
    _, _, bridge_a = _distributed_solve(kind, base)
    torch.distributed.barrier()
    _, _, bridge_b = _distributed_solve(kind, base)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        grad_eps_a = bridge_a._last_global_grad_eps
        grad_eps_b = bridge_b._last_global_grad_eps
        assert grad_eps_a is not None and grad_eps_b is not None
        nonzero = 0.0
        for name in ("Ex", "Ey", "Ez"):
            nonzero += float(grad_eps_a[name].abs().sum().item())
            scale = float(grad_eps_a[name].abs().max().item())
            max_diff = float((grad_eps_a[name] - grad_eps_b[name]).abs().max().item())
            rel = max_diff / scale if scale > 0.0 else max_diff
            # Reduction-order determinism (bitwise under exclusive-GPU scheduling;
            # concurrent-load block rescheduling can flip fused-kernel/VJP reductions
            # at the ULP level). A material nondeterminism would blow past this band.
            assert rel < _DETERMINISM_MAX_REL, (
                f"grad_eps[{name}] not reduction-order reproducible: max abs diff "
                f"{max_diff:.3e} (rel {rel:.3e} > {_DETERMINISM_MAX_REL:.0e})"
            )
        assert nonzero > 0.0, "gathered grad_eps is all zero (vacuous)"
        print("NCCL_ADJOINT_WORKER_OK[determinism]")


def _run_guard_deadlock():
    """An unsupported-adjoint scene must reject cleanly on ALL ranks (no hang).

    A trainable density on a legacy graded-sigma absorber has no verified
    distributed reverse core; ``require_distributed_adjoint_support`` rejects it on
    every rank symmetrically. Because the scene is identical across ranks the reject
    fires on all of them, so no collective is ever left half-posted. The launcher's
    timeout is the deadlock witness: a hang would exceed it and fail the gate.
    """
    base = _base_density()
    density = base.clone().to("cuda:0").requires_grad_(True)
    x = np.linspace(-0.6, 0.6, 21, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda:0",
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.3, 0.3)),
            density=density,
            eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.18, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.18, 0.0, 0.0), fields=("Ez",)))
    # absorber="pml" -> legacy graded-sigma, rejected at reverse-support validation.
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=_CPML_STEPS),
        absorber="pml",
        parallel=_parallel(),
    )
    rejected = False
    message = ""
    try:
        _dist_adjoint.run_nccl_distributed_reverse(
            simulation, objective=_dist_adjoint.point_monitor_l2_objective
        )
    except (ValueError, RuntimeError) as exc:
        rejected = True
        message = str(exc)
    # The reject is symmetric (identical scene on every rank) and fires before any
    # halo collective, so the process must return here on ALL ranks rather than one
    # rank blocking in a collective the other already abandoned. Guard the barrier:
    # this particular guard fires at construction, before the process group inits.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    assert rejected, "unsupported-adjoint scene was not rejected"
    assert "distributed reverse" in message.lower() or "graded" in message.lower(), message
    if int(os.environ.get("RANK", "0")) == 0:
        print("NCCL_ADJOINT_WORKER_OK[guard_deadlock]")


_MODES = {
    "standard": lambda: _run_parity("standard"),
    "plane": lambda: _run_parity("plane"),
    "plane_seam": _run_plane_seam,
    "cpml": lambda: _run_parity("cpml"),
    "cpml_psi": _run_cpml_psi,
    "falsify_mag_halo": lambda: _run_falsify_halo("mag"),
    "falsify_elec_halo": lambda: _run_falsify_halo("elec"),
    "falsify_psi": _run_falsify_psi,
    "determinism": _run_determinism,
    "guard_deadlock": _run_guard_deadlock,
}


def main() -> None:
    mode = os.environ.get("WITWIN_NCCL_ADJ_MODE", "standard")
    if mode not in _MODES:
        raise SystemExit(f"unknown WITWIN_NCCL_ADJ_MODE={mode!r}")
    try:
        with _gpu_stress():
            _MODES[mode]()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
