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
_STEPS = 48


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
