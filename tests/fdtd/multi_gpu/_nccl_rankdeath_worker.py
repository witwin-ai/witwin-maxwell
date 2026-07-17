"""torchrun worker: a peer rank dies mid-run; the survivor must not hang.

Launched by ``test_nccl_transport.py::test_nccl_rank_death_propagates_failure``.
Both ranks join the process group (the preflight homogeneity all_gather completes),
then rank 1 hard-exits before the first halo collective of the solve. Rank 0's
next collective therefore has no matching peer: torchrun detects the dead worker
and/or the ProcessGroupNCCL watchdog aborts after the configured timeout, so the
launch exits nonzero within a bounded time rather than hanging forever.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig

_FREQUENCY = 1.0e9


def _scene() -> mw.Scene:
    x = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.4e9),
            name="drive",
        )
    )
    return scene


def main() -> None:
    parallel = FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        transport="nccl",
        gather_fields=True,
        overlap=False,
        result_device="cuda:0",
        timeout_s=15.0,
    )
    distributed = DistributedFDTD(
        _scene(), frequency=_FREQUENCY, parallel=parallel, absorber_type="cpml"
    )
    # init_field joins the group (preflight all_gather completes on both ranks) and
    # builds each rank's engine locally.
    distributed.init_field()

    if distributed.rank == 1:
        # Die after joining the group but before the first solve halo collective.
        # os._exit skips finalizers/teardown, emulating an ungraceful peer crash.
        os._exit(1)

    # Rank 0 reaches here and posts halo collectives with no peer; the watchdog
    # timeout / launcher failure propagation must abort rather than hang.
    distributed.solve(
        time_steps=200,
        dft_frequency=_FREQUENCY,
        dft_window="none",
        full_field_dft=True,
        normalize_source=False,
    )
    print("NCCL_RANKDEATH_WORKER_UNEXPECTED_SURVIVAL")


if __name__ == "__main__":
    main()
