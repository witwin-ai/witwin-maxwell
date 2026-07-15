from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import capacity
from witwin.maxwell.fdtd.distributed.capacity import (
    local_dft_working_set_bytes,
    require_gather_capacity,
    require_local_dft_capacity,
)
from witwin.maxwell.fdtd.distributed.output import electric_field_output_bytes
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig


_FREQUENCY = 1.0e9
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def test_local_dft_capacity_counts_accumulators_and_postprocessed_fields_exactly(
    monkeypatch,
):
    shape = (5, 4, 3)
    frequencies = (0.8e9, 1.2e9)
    complex_fields = electric_field_output_bytes(
        shape,
        frequency_count=len(frequencies),
        complex_output=True,
    )
    # One complex-field equivalent is the real+imag float32 accumulators; the
    # other is the postprocessed complex64 local result.
    expected = 2 * complex_fields

    assert expected == 4256
    assert local_dft_working_set_bytes(
        shape,
        dft_frequency=frequencies,
        full_field_dft=True,
    ) == expected

    monkeypatch.setattr(capacity, "effective_cuda_free_bytes", lambda _device: 1 << 40)
    preflight = require_local_dft_capacity(
        torch.device("cuda:0"),
        shape,
        dft_frequency=frequencies,
        full_field_dft=True,
    )
    assert preflight["required_bytes"] == expected
    assert preflight["pending_local_dft_bytes"] == expected


def test_result_gather_capacity_oom_uses_combined_global_and_local_bytes(monkeypatch):
    shape = (9, 7, 5)
    output_bytes = electric_field_output_bytes(
        shape,
        frequency_count=2,
        complex_output=True,
    )
    pending_local = output_bytes
    combined = output_bytes + pending_local

    # Enough for the global result alone, but not for result + resident local
    # DFT accumulators/postprocess. The preflight must reject the combination.
    monkeypatch.setattr(
        capacity,
        "effective_cuda_free_bytes",
        lambda _device: int(output_bytes * 1.10),
    )
    with pytest.raises(MemoryError, match=rf"requires {combined} bytes"):
        require_gather_capacity(
            torch.device("cuda:0"),
            shape,
            dft_frequency=(0.8e9, 1.2e9),
            full_field_dft=True,
            pending_local_dft_bytes=pending_local,
        )

    monkeypatch.setattr(
        capacity,
        "effective_cuda_free_bytes",
        lambda _device: 2 * combined,
    )
    preflight = require_gather_capacity(
        torch.device("cuda:0"),
        shape,
        dft_frequency=(0.8e9, 1.2e9),
        full_field_dft=True,
        pending_local_dft_bytes=pending_local,
    )
    assert preflight["output_bytes"] == output_bytes
    assert preflight["pending_local_dft_bytes"] == pending_local
    assert preflight["required_bytes"] == combined


def _cpml_scene() -> mw.Scene:
    x = np.linspace(-0.6, 0.6, 13, dtype=np.float64)
    y = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    z = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x[0]), float(x[-1])),
                (float(y[0]), float(y[-1])),
                (float(z[0]), float(z[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0e6),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="source",
        )
    )
    scene.add_monitor(
        mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=_FIELD_NAMES)
    )
    return scene


def _run_cpml(*, parallel=None):
    return mw.Simulation.fdtd(
        _cpml_scene(),
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        absorber="cpml",
        cpml_config={"memory_mode": "slab"},
        full_field_dft=False,
        cuda_graph=False,
        parallel=parallel,
    ).run()


def _multi_component(result, name: str) -> torch.Tensor:
    if name in result.raw_output:
        return result.raw_output[name]
    solver = result.solver
    local = tuple(getattr(shard.solver, name) for shard in solver.shards)
    return solver._gather_component(name, local)


@pytest.fixture(scope="module")
def slab_cpml_results(cuda_p2p_devices):
    single = _run_cpml()
    multi = _run_cpml(
        parallel=FDTDParallelConfig(
            devices=cuda_p2p_devices,
            transport="cuda_p2p",
            overlap=True,
            gather_fields=True,
            result_device=cuda_p2p_devices[0],
        )
    )
    yield single, multi
    del single, multi
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def test_multi_gpu_slab_cpml_preserves_config_and_six_field_parity(slab_cpml_results):
    single, multi = slab_cpml_results
    distributed = multi.solver

    assert distributed.cpml_config["memory_mode"] == "slab"
    assert distributed._cpml_memory_mode_requested == "slab"
    assert distributed._cpml_memory_mode == "slab"
    assert all(
        shard.solver._cpml_memory_mode_requested == "slab"
        and shard.solver._cpml_memory_mode == "slab"
        for shard in distributed.shards
    )

    for name in _FIELD_NAMES:
        torch.testing.assert_close(
            _multi_component(multi, name).to("cuda:0"),
            getattr(single.solver, name).to("cuda:0"),
            rtol=2.0e-5,
            atol=2.0e-6,
        )


def test_public_solver_stats_aggregate_slab_cpml_memory(slab_cpml_results):
    _single, multi = slab_cpml_results
    solver = multi.solver
    stats = multi.stats()

    allocated = sum(
        int(shard.solver._cpml_allocated_memory_bytes) for shard in solver.shards
    )
    dense = sum(int(shard.solver._cpml_dense_memory_bytes) for shard in solver.shards)
    slab = sum(int(shard.solver._cpml_slab_memory_bytes) for shard in solver.shards)

    assert stats["cpml_requested_memory_mode"] == "slab"
    assert stats["cpml_memory_mode"] == "slab"
    assert stats["cpml_allocated_memory_bytes"] == allocated == slab
    assert stats["cpml_dense_memory_bytes"] == dense
    assert stats["cpml_slab_memory_bytes"] == slab
    assert 0 < slab < dense


def test_parallel_stats_report_local_and_including_gather_peaks(slab_cpml_results):
    _single, multi = slab_cpml_results
    parallel_stats = multi.stats()["parallel_stats"]
    local = parallel_stats["peak_memory_bytes"]
    including_gather = parallel_stats["peak_memory_bytes_including_gather"]

    assert set(local) == {"cuda:0", "cuda:1"}
    assert set(including_gather) == set(local)
    assert all(local[device] > 0 for device in local)
    assert all(including_gather[device] >= local[device] for device in local)
    assert {
        partition["device"]: partition["peak_memory_bytes"]
        for partition in parallel_stats["partitions"]
    } == local


def test_parallel_phase_timings_are_explicitly_unavailable_with_explanation(
    slab_cpml_results,
):
    _single, multi = slab_cpml_results
    parallel_stats = multi.stats()["parallel_stats"]

    assert parallel_stats["wall_time_s"] > 0.0
    assert parallel_stats["compute_time_s"] is None
    assert parallel_stats["communication_time_s"] is None
    assert parallel_stats["exposed_communication_time_s"] is None
    assert "external CUDA profiler" in parallel_stats["timing_note"]
    assert "not inserted" in parallel_stats["timing_note"]
