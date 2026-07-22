"""torchrun worker: end-to-end NCCL forward solve vs a single-GPU reference.

Launched by ``test_nccl_transport.py::test_two_rank_nccl_forward_matches_single_gpu``
via ``torchrun --nproc-per-node=2``. Each rank builds its own ShardEngine from the
deterministic partition plan and runs the distributed coordinator loop over
``NcclHaloTransport``; rank 0 gathers the global full-field DFT (Ex/Ey/Ez) with the
sized point-to-point primitive and compares it to an independently constructed
single-GPU ``FDTD`` reference on the same scene, at the plan's monitor tolerances
(rtol 5e-5 / atol 5e-6) -- identical to the in-process CUDA P2P conformance leg.

A nonzero process exit signals a conformance failure. No timing is asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.scene import prepare_scene

_FREQUENCY = 1.0e9
_DFT_FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
# The Gaussian pulse must physically cross the x=0 partition seam before the
# gathered parity check runs, or the halo exchange carries no signal and the gate
# is vacuous: at 48 steps a no-op'd electric halo still passes within atol 5e-6
# because nothing has reached the seam. Measured single-GPU seam-plane amplitudes
# (drive frequency, reference solve on this scene) as a fraction of each
# component's domain max: at 48 steps the largest is Ex 6.1e-3 (below the floor);
# at 120 steps Ex reaches 1.3e-2; at 160 steps Ex reaches 2.65e-2. 160 steps
# clears the 1e-2 precondition floor by ~2.6x while keeping the run short.
_STEPS = 160
# Minimum seam-plane DFT amplitude (largest E component, relative to that
# component's domain max) the reference solve must exhibit for the gathered halo
# parity check to be meaningful. Justified above: 48-step (pre-crossing) configs
# sit at 6.1e-3 and fail this floor, so the worker can never silently regress to a
# pre-seam-crossing step count; 160 steps sits at 2.65e-2.
_SEAM_AMPLITUDE_FLOOR = 1.0e-2


def _scene() -> mw.Scene:
    # 13 x-nodes split 6/6 across two ranks; a dielectric box straddles the split
    # so the interface halo carries a real material discontinuity.
    x = np.linspace(-0.6, 0.6, 13, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.2, 0.2)),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.4e9, amplitude=10.0),
            name="drive",
        )
    )
    return scene


def _assert_seam_carries_signal(reference_output, shard_layouts) -> None:
    """Precondition: the reference solve must carry real signal across the seam.

    Guards the gate against a silent regression to a pre-seam-crossing step count.
    Measures, on the independent single-GPU reference DFT at the drive frequency,
    the seam-plane amplitude of each gathered electric component (the plane the
    coordinator halo exchanges: the internal x partition node for the node-owned
    ``Ey``/``Ez`` and the last owned cell for the cell-owned ``Ex``) relative to
    that component's own domain max. If no component reaches ``_SEAM_AMPLITUDE_FLOOR``
    the pulse has not meaningfully crossed x=0 and the halo parity check would be
    vacuous, so the worker fails closed here.
    """

    layout0 = shard_layouts[0]
    seam_node = int(layout0.global_node_owned.stop)
    seam_cell = int(layout0.global_cell_owned.stop)
    freq_index = _DFT_FREQUENCIES.index(_FREQUENCY)

    ratios: dict[str, float] = {}
    for name in ("Ex", "Ey", "Ez"):
        field = reference_output[name][freq_index]
        seam_index = (seam_cell - 1) if name == "Ex" else seam_node
        domain_max = float(torch.abs(field).max().item())
        assert domain_max > 0.0, f"{name} reference domain max is zero"
        seam_amplitude = float(torch.abs(field[seam_index]).max().item())
        ratios[name] = seam_amplitude / domain_max

    best = max(ratios.values())
    assert best >= _SEAM_AMPLITUDE_FLOOR, (
        "seam-plane DFT amplitude is below the crossing floor; the halo carries no "
        f"meaningful signal and the parity gate is vacuous. Per-component "
        f"seam/domain-max ratios={ratios}, floor={_SEAM_AMPLITUDE_FLOOR}. Raise "
        "_STEPS until the pulse crosses the x=0 seam."
    )


def _run_timing_pass(distributed) -> None:
    """Opt-in per-rank step-rate timing pass (no-op unless env-enabled).

    Runs on every rank in lockstep after the parity solve when
    ``WITWIN_FDTD_STEP_TIMING`` is truthy, so the supervisor's exclusive timing
    window can flip the env var and collect one machine-readable JSON per rank
    without contaminated shared-GPU numbers being asserted here. When disabled
    (the default, and always in CI) the instrument's bracket calls return
    immediately, so this pass drives no collective and adds zero per-step cost --
    the parity behavior above is byte-identical to a build without this pass.
    """

    import os

    from witwin.maxwell.fdtd.distributed.instrumentation import StepRateInstrument

    instrument = StepRateInstrument.from_env(
        rank=distributed.rank,
        world_size=distributed.transport.world_size,
        device=str(distributed.device),
    )
    if not instrument.enabled:
        return
    steps = int(os.environ.get("WITWIN_FDTD_STEP_TIMING_STEPS", "200"))
    distributed._prepare_outputs(steps, _DFT_FREQUENCIES, "none", True)
    overlap_active = distributed._overlap_active()
    distributed._synchronize_all()
    instrument.loop_begin()
    for n in range(steps):
        instrument.step_begin()
        distributed._advance_one_step(n, overlap_active=overlap_active)
        instrument.step_end()
    instrument.loop_end()
    distributed._synchronize_all()
    instrument.finalize()


def _parallel() -> FDTDParallelConfig:
    return FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
        gather_fields=True,
        overlap=False,
        result_device="cuda:0",
    )


def main() -> None:
    distributed = DistributedFDTD(
        _scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(),
        absorber_type="cpml",
    )
    try:
        distributed.init_field()
        output = distributed.solve(
            time_steps=_STEPS,
            dft_frequency=_DFT_FREQUENCIES,
            dft_window="none",
            full_field_dft=True,
            normalize_source=False,
        )
        distributed.transport.barrier()

        # Opt-in per-rank step-rate timing (collective; runs on every rank). A
        # no-op when the timing env var is unset, so the default parity run is
        # untouched.
        _run_timing_pass(distributed)
        distributed.transport.barrier()

        if distributed.rank == 0:
            assert output is not None, "rank 0 produced no gathered output"
            single = FDTD(prepare_scene(_scene()), frequency=_FREQUENCY, absorber_type="cpml")
            single.init_field()
            single_output = single.solve(
                time_steps=_STEPS,
                dft_frequency=_DFT_FREQUENCIES,
                dft_window="none",
                full_field_dft=True,
                normalize_source=False,
                use_cuda_graph=False,
            )
            assert output["frequencies"] == _DFT_FREQUENCIES, output["frequencies"]
            # Fail closed before the parity loop if the pulse has not crossed the
            # seam, so the gate can never silently regress to a vacuous config.
            _assert_seam_carries_signal(single_output, distributed.shard_layouts)
            for name in ("Ex", "Ey", "Ez"):
                dist_field = output[name]
                ref_field = single_output[name]
                assert dist_field.shape == ref_field.shape, (
                    f"{name} shape {tuple(dist_field.shape)} != {tuple(ref_field.shape)}"
                )
                assert torch.is_complex(dist_field) and torch.is_complex(ref_field)
                # Non-vacuity: the field must carry real signal or the check is empty.
                assert float(torch.abs(ref_field).max().item()) > 0.0, f"{name} reference is zero"
                torch.testing.assert_close(
                    dist_field.to(ref_field.device),
                    ref_field,
                    rtol=5.0e-5,
                    atol=5.0e-6,
                )
            print("NCCL_FORWARD_WORKER_OK")
    finally:
        distributed.transport.barrier() if torch.distributed.is_initialized() else None
        distributed.teardown()


if __name__ == "__main__":
    main()
