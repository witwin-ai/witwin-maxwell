"""run_many ensemble: CPU-constructible guard/config tests plus CUDA placement.

The guard and configuration tests are CPU-constructible (they raise during plan
construction, before any device pool is built). The identical-to-serial and
device-lease legs require two peer CUDA devices and are skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.execution import (
    DistributedFailure,
    MultiGPUExecution,
    build_ensemble_plan,
    estimate_scene_footprint_bytes,
    execute_plan,
    run_many,
)

_FREQUENCY = 1.0e9


def _vacuum_scene(*, device, amplitude=1.0, trainable=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.25, 0.25), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=amplitude),
            name="src",
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            "probe",
            (0.0, 0.0, 0.0),
            fields=("Ex", "Ey", "Ez"),
        )
    )
    if trainable:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.0),
                    size=torch.tensor([0.2, 0.2, 0.2], requires_grad=True),
                ),
                material=mw.Material(eps_r=2.0),
            )
        )
    return scene


def _sim(scene, *, time_steps=8):
    return mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )


# --------------------------------------------------------------------------- #
# Configuration + guard tests (CPU-constructible).
# --------------------------------------------------------------------------- #


def test_multigpu_execution_ensemble_config():
    execution = MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1"))
    assert execution.devices == ("cuda:0", "cuda:1")
    assert execution.max_concurrency == 2
    assert execution.placement == "memory_aware"
    assert execution.fail_fast is False


def test_multigpu_execution_rejects_bad_devices():
    with pytest.raises(ValueError):
        MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:0"))
    with pytest.raises(ValueError):
        MultiGPUExecution.ensemble(devices=("cpu",))
    with pytest.raises(ValueError):
        MultiGPUExecution.ensemble(devices=("cuda:0",), placement="bogus")


def test_run_many_rejects_non_simulation_and_bad_execution():
    execution = MultiGPUExecution.ensemble(devices=("cuda:0",))
    with pytest.raises(TypeError):
        run_many([object()], execution=execution)
    with pytest.raises(TypeError):
        run_many([], execution=object())


def test_ensemble_rejects_trainable_simulation():
    execution = MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1"))
    sim = _sim(_vacuum_scene(device="cpu", trainable=True))
    with pytest.raises(ValueError, match="trainable"):
        build_ensemble_plan([sim], execution)


def test_ensemble_rejects_joint_solve_parallel_config():
    execution = MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1"))
    scene = _vacuum_scene(device="cpu")
    sim = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=8),
        spectral_sampler=mw.SpectralSampler(window="none"),
        parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1")),
    )
    with pytest.raises(ValueError, match="joint-solve"):
        build_ensemble_plan([sim], execution)


def test_ensemble_rejects_shared_scene_object():
    execution = MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1"))
    scene = _vacuum_scene(device="cpu")
    sim_a = _sim(scene)
    sim_b = _sim(scene)
    with pytest.raises(ValueError, match="share the same"):
        build_ensemble_plan([sim_a, sim_b], execution)


def test_ensemble_rejects_scene_module_input():
    class _Module(mw.SceneModule):
        def __init__(self, scene):
            super().__init__()
            self._scene = scene

        def to_scene(self):
            return self._scene

    execution = MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1"))
    module = _Module(_vacuum_scene(device="cpu"))
    sim = _sim(module)
    with pytest.raises(ValueError, match="SceneModule"):
        build_ensemble_plan([sim], execution)


def test_footprint_estimate_uniform_grid_is_positive_int():
    scene = _vacuum_scene(device="cpu")
    estimated = estimate_scene_footprint_bytes(scene, frequencies=(_FREQUENCY,))
    assert isinstance(estimated, int)
    assert estimated > 0


def test_footprint_estimate_returns_none_for_auto_grid():
    """Custom/auto grids cannot be sized from Domain bounds alone; the estimator
    returns None (order-based placement) rather than doing a coordinator-device
    prepare_scene at plan-build time."""

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.25, 0.25), (-0.25, 0.25))),
        grid=mw.GridSpec.auto(wavelength=0.3),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    assert estimate_scene_footprint_bytes(scene, frequencies=(_FREQUENCY,)) is None


# --------------------------------------------------------------------------- #
# CUDA placement + identical-to-serial legs.
# --------------------------------------------------------------------------- #


def _serial_fields_on_device(scene_kwargs, device):
    scene = _vacuum_scene(device=str(device), **scene_kwargs)
    result = _sim(scene).run()
    return {name: value.detach().clone() for name, value in result.fields.items()}


def test_run_many_matches_serial_and_respects_lease(cuda_p2p_devices, cuda_memory_cleanup):
    devices = tuple(str(device) for device in cuda_p2p_devices)
    execution = MultiGPUExecution.ensemble(devices=devices, max_concurrency=2)

    amplitudes = (1.0, 2.0, 3.0, 4.0)
    sims = [
        _sim(_vacuum_scene(device=devices[0], amplitude=amp)) for amp in amplitudes
    ]

    plan = build_ensemble_plan(sims, execution)
    pool = execution.build_pool(require_cuda=True)
    seq = execute_plan(plan, pool)

    # Order preserved and every task completed.
    assert len(seq) == len(amplitudes)
    assert all(record.status == "completed" for record in seq.records)

    # No device leased beyond its capacity.
    for device in execution.devices:
        assert pool.peak_concurrency(device) <= execution.per_device_concurrency

    for index, amp in enumerate(amplitudes):
        result = seq[index]
        leased = seq.records[index].device
        assert leased in execution.devices
        # Device lease respected: returned tensors live on the leased device.
        for tensor in result.fields.values():
            assert str(tensor.device) == leased
        # Result.scene reports the device its tensors actually live on (the leased
        # device), not the caller's original scene device: the per-run scene copy
        # is pinned to the lease instead of being restored to the original.
        assert str(result.scene.device) == leased
        # Identical to an isolated serial run on the very same device: proves no
        # cross-task state bleed and order fidelity (amplitude drives the field).
        reference = _serial_fields_on_device({"amplitude": amp}, leased)
        for name, tensor in result.fields.items():
            assert torch.equal(tensor, reference[name])


def test_run_many_end_to_end_orders_results(cuda_p2p_devices, cuda_memory_cleanup):
    devices = tuple(str(device) for device in cuda_p2p_devices)
    execution = MultiGPUExecution.ensemble(devices=devices, max_concurrency=2)
    sims = [
        _sim(_vacuum_scene(device=devices[0], amplitude=amp)) for amp in (1.0, 5.0)
    ]
    seq = run_many(sims, execution=execution)
    assert len(seq) == 2
    assert not seq.failed
    # Larger amplitude yields a strictly larger peak |Ez|; order must be preserved.
    peaks = [float(torch.abs(seq[i].fields["EZ"]).max()) for i in range(2)]
    assert peaks[1] > peaks[0]


def test_run_many_isolates_a_failing_task(cuda_p2p_devices, cuda_memory_cleanup):
    devices = tuple(str(device) for device in cuda_p2p_devices)
    execution = MultiGPUExecution.ensemble(
        devices=devices, max_concurrency=2, fail_fast=False
    )
    good = _sim(_vacuum_scene(device=devices[0], amplitude=1.0))
    bad = _sim(_vacuum_scene(device=devices[0], amplitude=1.0))
    # Inject a deterministic failure inside the worker after device binding.
    def _boom():
        raise RuntimeError("injected task failure")

    bad.run = _boom
    good2 = _sim(_vacuum_scene(device=devices[0], amplitude=2.0))

    seq = run_many([good, bad, good2], execution=execution)
    assert not isinstance(seq.entries[0], DistributedFailure)
    assert isinstance(seq.entries[1], DistributedFailure)
    assert not isinstance(seq.entries[2], DistributedFailure)
    assert seq.entries[1].index == 1
