"""Distributed shards run the fused core Yee update and reproduce the frozen existing-path payloads."""

import pytest
import torch

from benchmark.core_path import distributed_gate
from tests.support import frozen_oracles


def test_simulation_shards_run_core_yee_with_device_dft(cuda_p2p_devices, cuda_memory_cleanup):
    simulation = distributed_gate.build_case("dist_vacuum_none").simulation(True, 12)
    result = simulation.run()
    solver = result.solver
    assert all(shard.solver._core_runtime is not None for shard in solver.shards)
    # Weight rows come from the per-shard device table, indexed by the device step counter.
    assert all(int(shard.solver._dft_step.item()) == 12 for shard in solver.shards)
    stats = solver.parallel_stats
    # The whole-shard fused launch has no interior/halo split, so overlap is requested but inactive.
    assert stats["overlap_requested"] is True and stats["overlap_active"] is False


@pytest.mark.parametrize("scene", ["dist_vacuum_none", "dist_pec_dielectric", "dist_cpml_dense", "dist_shutoff",
                                   "dist_gradient_standard"])
def test_distributed_core_equals_frozen_existing_path(cuda_p2p_devices, cuda_memory_cleanup, scene):
    reference = torch.load(frozen_oracles.scene_directory("distributed", scene) / "tensors.pt", weights_only=True)
    case = distributed_gate.build_case(scene)
    candidate, meta = distributed_gate._run(case, parallel=True, steps=case.steps)
    assert meta["execution_path"] == "core"
    compared = 0
    for key, value in candidate.items():
        if "/shard_state/" in key or key.startswith("shard_state/"):
            continue
        expected = reference[f"dist/full/{key}"]
        assert value.dtype == expected.dtype and value.shape == expected.shape, key
        assert torch.equal(value, expected), key
        compared += 1
    assert compared


def test_seam_cell_probe_reads_the_updated_ghost_hx(cuda_p2p_devices, cuda_memory_cleanup):
    # A point in rank 0's last owned cell interpolates Hx to the high ghost node. The bounded
    # launches never updated that node (Hx read 0 there); the whole-shard fused launch updates it
    # from the received Ey/Ez halo, so the probe equals the single-device run.
    case = distributed_gate.build_case("dist_vacuum_none")
    probes = []
    for parallel in (True, False):
        result = case.simulation(parallel, 24).run()
        probes.append(result.raw_monitor("seam_probe")["components"]["Hx"].cpu())
    assert probes[1].abs().max() > 0
    assert torch.equal(*probes)
