"""Stressed parity + falsification gate for the in-process cuda_p2p adjoint race.

The in-process ``transport="cuda_p2p"`` distributed adjoint reconstructs the
reverse trajectory from mid-forward checkpoints. The checkpoint clone
(``capture_distributed_checkpoint``) reads the persistent padded field storage
that the forward field updates write on each shard's ``compute_stream``. Cloning
on the device default stream left the clone unordered w.r.t. the *next*
``_advance_one_step`` (compute_stream), so under concurrent GPU load the update
overwrote the storage mid-clone and the replayed gradient drifted at the
partition seam -- while the forward output (collected separately) stayed bitwise
identical. The fix clones on the shard's ``compute_stream`` (previous-update ->
clone -> next-update serialize on one stream).

These committed nodes reproduce the load condition with a co-tenant GPU burner on
both boards and assert 1-vs-2-GPU parity over >=6 rounds at the SAME honest
tolerances the unstressed parity gates use (no tolerance changes), plus a
falsification that reverts the fix (clone on the default stream) under the same
load and shows the seam-localized drift return.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.fdtd.distributed import adjoint as _dist_adjoint
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state

_BURNER = Path(__file__).with_name("_gpu_burner.py")
_FREQUENCY = 1.0e9
_DENS_SHAPE = (5, 4, 4)
# Number of stressed distributed rounds asserted against the single-GPU reference.
_STRESS_ROUNDS = 6
# Honest tolerances -- identical to the unstressed parity gates in
# test_adjoint_parity.py / test_adjoint_parity_cpml.py. NOT relaxed for stress.
_LOSS_RTOL = 5.0e-5
_LOSS_ATOL = 5.0e-6
_GRAD_RTOL = 1.0e-4
# Seam-localized drift separation for the falsification (a working reverse holds
# ~2e-7; the reverted clone drifts ~8e-2 under load, deterministic on this scene).
_FALSIFICATION_MIN_REL = 1.0e-3


@contextlib.contextmanager
def _gpu_stress(devices):
    """Saturate both boards with a committed co-tenant burner for the gate.

    One burner subprocess per physical GPU (pinned via ``CUDA_VISIBLE_DEVICES``),
    a fixed warm-up so both boards are saturating before the timed section, and an
    unconditional teardown so a failing gate never leaks a runaway process.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        dev_ids = [d.strip() for d in visible.split(",") if d.strip()]
    else:
        dev_ids = [str(d.index) for d in devices]
    procs: list[subprocess.Popen] = []
    for dev in dev_ids:
        child = dict(os.environ)
        child["CUDA_VISIBLE_DEVICES"] = dev
        procs.append(
            subprocess.Popen(
                [sys.executable, str(_BURNER)],
                env=child,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    try:
        time.sleep(8.0)
        yield
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def _parallel(devices):
    return FDTDParallelConfig(
        devices=devices, transport="cuda_p2p", gather_fields=False,
        overlap=False, result_device=devices[0],
    )


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
            density=density, eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0), polarization="Ez", profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (monitor_x, 0.0, 0.0), fields=("Ez",)))
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
            density=density, eps_bounds=(1.0, 6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0), polarization="Ez", profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (monitor_x, 0.0, 0.0), fields=("Ez",)))
    return scene


def _solve(scene_builder, density_values, *, parallel_devices, source_x, monitor_x,
           steps, absorber=None):
    density = density_values.clone().to("cuda:0").requires_grad_(True)
    scene = scene_builder(density, source_x=source_x, monitor_x=monitor_x, device="cuda:0")
    kwargs = dict(frequency=_FREQUENCY, run_time=mw.TimeConfig(time_steps=steps))
    if absorber is not None:
        kwargs["absorber"] = absorber
    if parallel_devices is not None:
        kwargs["parallel"] = _parallel(parallel_devices)
    result = mw.Simulation.fdtd(scene, **kwargs).run()
    spectrum = result.monitors["probe"]["Ez"]
    loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
    loss.backward()
    return float(loss.detach().cpu()), density.grad.detach().cpu().clone()


def _base_density():
    torch.manual_seed(1)
    return (0.4 + 0.2 * torch.rand(_DENS_SHAPE)).double()


def _assert_parity(dist_loss, dist_grad, ref_loss, ref_grad):
    torch.testing.assert_close(
        torch.tensor(dist_loss), torch.tensor(ref_loss), rtol=_LOSS_RTOL, atol=_LOSS_ATOL
    )
    atol_floor = 1.0e-6 * float(ref_grad.abs().max())
    torch.testing.assert_close(dist_grad, ref_grad, rtol=_GRAD_RTOL, atol=atol_floor)


def _capture_on_default_stream(distributed, step):
    """Pre-fix checkpoint capture: clone on the device default stream.

    Reverts the compute_stream capture fix so the checkpoint clone races the next
    forward update kernel on compute_stream -- the exact site of the race.
    """

    _dist_adjoint.require_distributed_adjoint_support(distributed)
    states = {}
    for shard in distributed.shards:
        with torch.cuda.device(shard.device):
            states[shard.rank] = capture_checkpoint_state(shard.solver, step)
    return _dist_adjoint.DistributedCheckpoint(
        step=int(step),
        partition_signature=_dist_adjoint._partition_signature(distributed),
        states=states,
    )


def test_p2p_adjoint_parity_under_stress_standard(cuda_p2p_devices, cuda_memory_cleanup):
    """Standard open-boundary 1-vs-2-GPU parity holds over >=6 rounds under load."""

    base = _base_density()
    ref_loss, ref_grad = _solve(
        _standard_scene, base, parallel_devices=None,
        source_x=-0.3, monitor_x=0.1, steps=60,
    )
    assert abs(ref_loss) > 0.0 and float(ref_grad.abs().max()) > 0.0

    rels = []
    with _gpu_stress(cuda_p2p_devices):
        for _ in range(_STRESS_ROUNDS):
            d_loss, d_grad = _solve(
                _standard_scene, base, parallel_devices=cuda_p2p_devices,
                source_x=-0.3, monitor_x=0.1, steps=60,
            )
            rels.append(float((d_grad - ref_grad).abs().max()) / float(ref_grad.abs().max()))
            _assert_parity(d_loss, d_grad, ref_loss, ref_grad)
    print(f"\n[stress standard] grad_rel max={max(rels):.3e} min={min(rels):.3e} n={len(rels)}")
    assert max(rels) < _GRAD_RTOL, f"stressed grad rel {max(rels):.3e} exceeded the gate"


def test_p2p_adjoint_parity_under_stress_cpml(cuda_p2p_devices, cuda_memory_cleanup):
    """x-CPML 1-vs-2-GPU parity holds over >=6 rounds under load."""

    base = _base_density()
    ref_loss, ref_grad = _solve(
        _cpml_scene, base, parallel_devices=None,
        source_x=-0.18, monitor_x=0.18, steps=50, absorber="cpml",
    )
    assert abs(ref_loss) > 0.0 and float(ref_grad.abs().max()) > 0.0

    rels = []
    with _gpu_stress(cuda_p2p_devices):
        for _ in range(_STRESS_ROUNDS):
            d_loss, d_grad = _solve(
                _cpml_scene, base, parallel_devices=cuda_p2p_devices,
                source_x=-0.18, monitor_x=0.18, steps=50, absorber="cpml",
            )
            rels.append(float((d_grad - ref_grad).abs().max()) / float(ref_grad.abs().max()))
            _assert_parity(d_loss, d_grad, ref_loss, ref_grad)
    print(f"\n[stress cpml] grad_rel max={max(rels):.3e} min={min(rels):.3e} n={len(rels)}")
    assert max(rels) < _GRAD_RTOL, f"stressed CPML grad rel {max(rels):.3e} exceeded the gate"


def test_p2p_capture_default_stream_falsification_under_stress(
    cuda_p2p_devices, cuda_memory_cleanup, monkeypatch
):
    """Reverting the capture fix (clone on default stream) reddens under load.

    Load-bearing falsification for the compute_stream capture fix: under the same
    co-tenant burner, cloning the checkpoint on the device default stream lets the
    next forward update overwrite the field storage mid-clone, so the replayed
    gradient drifts at the partition seam far above the parity gate. Restoring the
    fix returns the gradient to the ~2e-7 floor. The forward loss stays bitwise
    identical throughout (the output is collected independently of the checkpoints).
    """

    base = _base_density()
    ref_loss, ref_grad = _solve(
        _standard_scene, base, parallel_devices=None,
        source_x=-0.3, monitor_x=0.1, steps=60,
    )
    grad_scale = float(ref_grad.abs().max())
    assert grad_scale > 0.0

    with _gpu_stress(cuda_p2p_devices):
        # Revert the fix: clone on the default stream (the racing site).
        monkeypatch.setattr(
            _dist_adjoint, "capture_distributed_checkpoint", _capture_on_default_stream
        )
        fals_loss, fals_grad = _solve(
            _standard_scene, base, parallel_devices=cuda_p2p_devices,
            source_x=-0.3, monitor_x=0.1, steps=60,
        )
        fals_rel = float((fals_grad - ref_grad).abs().max()) / grad_scale
        # The forward output is independent of the torn checkpoints -> bitwise clean.
        assert abs(fals_loss - ref_loss) <= _LOSS_ATOL + _LOSS_RTOL * abs(ref_loss)
        assert fals_rel > _FALSIFICATION_MIN_REL, (
            f"reverted capture (default-stream clone) left parity intact under load "
            f"(rel {fals_rel:.3e}); the compute_stream capture fix is not load-bearing"
        )

        # Restore the fix -> parity returns to the reduction-order floor.
        monkeypatch.undo()
        good_loss, good_grad = _solve(
            _standard_scene, base, parallel_devices=cuda_p2p_devices,
            source_x=-0.3, monitor_x=0.1, steps=60,
        )
    good_rel = float((good_grad - ref_grad).abs().max()) / grad_scale
    print(f"\n[falsification] reverted(default-stream) grad_rel={fals_rel:.3e} "
          f"restored(compute_stream) grad_rel={good_rel:.3e}")
    _assert_parity(good_loss, good_grad, ref_loss, ref_grad)
