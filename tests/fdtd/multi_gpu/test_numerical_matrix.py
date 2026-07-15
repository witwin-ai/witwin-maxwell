from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig
from witwin.maxwell.result import Result
from witwin.maxwell.scene import prepare_scene


_FREQUENCY = 1.0e9
_FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_DFT_FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)


@dataclass
class _SolvePair:
    single: FDTD
    single_output: dict[str, Any] | None
    distributed: DistributedFDTD
    distributed_output: dict[str, Any] | None


def _parallel(devices, *, gather_fields: bool = True) -> FDTDParallelConfig:
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        overlap=True,
        gather_fields=gather_fields,
        result_device=devices[0],
    )


def _run_pair(
    scene: mw.Scene,
    devices,
    *,
    time_steps: int,
    dft_frequency: float | tuple[float, ...] = _FREQUENCY,
    full_field_dft: bool = False,
    shutoff: float = 0.0,
    shutoff_check_interval: int = 10,
) -> _SolvePair:
    single = FDTD(prepare_scene(scene), frequency=_FREQUENCY, absorber_type="cpml")
    single.init_field()
    single_output = single.solve(
        time_steps=time_steps,
        dft_frequency=dft_frequency,
        dft_window="none",
        full_field_dft=full_field_dft,
        normalize_source=False,
        shutoff=shutoff,
        shutoff_check_interval=shutoff_check_interval,
        use_cuda_graph=False,
    )

    distributed = DistributedFDTD(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(devices),
        absorber_type="cpml",
    )
    distributed.init_field()
    distributed_output = distributed.solve(
        time_steps=time_steps,
        dft_frequency=dft_frequency,
        dft_window="none",
        full_field_dft=full_field_dft,
        normalize_source=False,
        shutoff=shutoff,
        shutoff_check_interval=shutoff_check_interval,
        use_cuda_graph=False,
    )
    return _SolvePair(single, single_output, distributed, distributed_output)


def _small_scene(*, source_time=None, time_monitor: bool = False) -> mw.Scene:
    x = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    y = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    z = np.linspace(-0.2, 0.2, 5, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((float(x[0]), float(x[-1])), (float(y[0]), float(y[-1])), (float(z[0]), float(z[-1])))
        ),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )
    if source_time is not None:
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                profile="ideal",
                source_time=source_time,
                name="interface_pulse",
            )
        )
    if time_monitor:
        scene.add_monitor(
            mw.FieldTimeMonitor(
                "six_field_trace",
                components=_FIELD_NAMES,
                position=(0.1, 0.0, 0.0),
                start=2,
                stop=46,
                interval=3,
            )
        )
    return scene


def _distributed_field(pair: _SolvePair, name: str) -> torch.Tensor:
    local = tuple(getattr(shard.solver, name) for shard in pair.distributed.shards)
    return pair.distributed._gather_component(name, local)


def _assert_step_field_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.to(expected.device)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    error = torch.abs(actual - expected)
    assert float(error.max().item()) <= 2.0e-6

    scale = float(torch.abs(expected).max().item())
    significant = torch.abs(expected) >= max(1.0e-6, 1.0e-4 * scale)
    if bool(significant.any()):
        relative = error[significant] / torch.abs(expected[significant])
        assert float(relative.max().item()) <= 2.0e-5


@pytest.fixture(scope="module")
def pulse_matrix_pair(cuda_p2p_devices):
    scene = _small_scene(
        source_time=mw.GaussianPulse(
            frequency=_FREQUENCY,
            fwidth=0.4e9,
            amplitude=10.0,
        ),
        time_monitor=True,
    )
    pair = _run_pair(
        scene,
        cuda_p2p_devices,
        time_steps=48,
        dft_frequency=_DFT_FREQUENCIES,
        full_field_dft=True,
    )
    yield pair
    for device in cuda_p2p_devices:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def test_zero_field_remains_zero_on_one_and_two_gpus(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    pair = _run_pair(
        _small_scene(),
        cuda_p2p_devices,
        time_steps=8,
        full_field_dft=False,
    )
    for name in _FIELD_NAMES:
        single = getattr(pair.single, name)
        distributed = _distributed_field(pair, name)
        assert torch.count_nonzero(single).item() == 0
        assert torch.count_nonzero(distributed).item() == 0
        torch.testing.assert_close(distributed, single, rtol=0.0, atol=0.0)


def test_gaussian_impulse_six_step_fields_match_one_gpu(pulse_matrix_pair):
    pair = pulse_matrix_pair
    assert float(torch.abs(pair.single.Ez).max().item()) > 0.0
    for name in _FIELD_NAMES:
        _assert_step_field_close(
            _distributed_field(pair, name),
            getattr(pair.single, name),
        )


def test_actual_six_component_field_time_monitor_matches_one_gpu(pulse_matrix_pair):
    pair = pulse_matrix_pair
    single = pair.single_output["observers"]["six_field_trace"]
    distributed = pair.distributed_output["observers"]["six_field_trace"]

    assert distributed["kind"] == single["kind"] == "field_time"
    assert distributed["fields"] == single["fields"] == _FIELD_NAMES
    assert distributed["start"] == single["start"] == 2
    assert distributed["stop"] == single["stop"] == 46
    assert distributed["interval"] == single["interval"] == 3
    torch.testing.assert_close(distributed["t"], single["t"], rtol=0.0, atol=0.0)
    assert tuple(distributed["t"].shape) == (15,)
    for name in _FIELD_NAMES:
        torch.testing.assert_close(
            distributed["components"][name],
            single["components"][name],
            rtol=5.0e-5,
            atol=5.0e-6,
        )
    assert float(torch.abs(single["components"]["Ez"]).max().item()) > 0.0


def test_multifrequency_full_field_dft_values_match_one_gpu(pulse_matrix_pair):
    pair = pulse_matrix_pair
    assert pair.single_output["frequencies"] == _DFT_FREQUENCIES
    assert pair.distributed_output["frequencies"] == _DFT_FREQUENCIES
    for name in ("Ex", "Ey", "Ez"):
        single = pair.single_output[name]
        distributed = pair.distributed_output[name]
        assert single.shape[0] == distributed.shape[0] == len(_DFT_FREQUENCIES)
        assert torch.is_complex(single)
        assert torch.is_complex(distributed)
        assert float(torch.abs(single).max().item()) > 0.0
        torch.testing.assert_close(
            distributed,
            single,
            rtol=5.0e-5,
            atol=5.0e-6,
        )


def test_two_gpu_sharded_result_save_lazy_and_gather_round_trip(
    pulse_matrix_pair,
    tmp_path,
):
    pair = pulse_matrix_pair
    solver = pair.distributed
    output = pair.distributed_output
    references = {name: output[name] for name in ("Ex", "Ey", "Ez")}
    result = Result(
        method="fdtd",
        scene=solver.logical_scene,
        prepared_scene=solver.scene,
        frequencies=_DFT_FREQUENCIES,
        solver=solver,
        fields={name.upper(): tensor for name, tensor in references.items()},
        monitors=output.get("observers", {}),
        solver_stats={"parallel_stats": solver.parallel_stats},
    )

    directory = tmp_path / "two-gpu-sharded-result"
    manifest = result.save_sharded(directory)

    assert tuple(path.name for path in sorted(directory.iterdir())) == (
        "manifest.json",
        "rank-0000.pt",
        "rank-0001.pt",
        "result.pt",
    )
    assert manifest.frequencies == _DFT_FREQUENCIES
    assert tuple(shard.rank for shard in manifest.shards) == (0, 1)
    manifest_components = {component.name: component for component in manifest.components}
    for name, reference in references.items():
        global_component = manifest_components[name]
        assert global_component.shape == tuple(reference.shape)
        assert global_component.x_axis == reference.ndim - 3 == 1

        cursor = 0
        persisted_x = 0
        for shard, path in zip(manifest.shards, manifest.shard_paths(directory)):
            local = next(component for component in shard.components if component.name == name)
            assert local.global_x_slice[0] == cursor
            cursor = local.global_x_slice[1]
            persisted_x += local.shape[local.x_axis]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            tensor = payload["components"][name]["tensor"]
            assert tensor.shape[local.x_axis] == (
                local.global_x_slice[1] - local.global_x_slice[0]
            )
        assert cursor == reference.shape[global_component.x_axis]
        assert persisted_x == reference.shape[global_component.x_axis]

    lazy = Result.load_sharded(
        directory,
        scene=solver.logical_scene,
        prepared_scene=solver.scene,
        gather_fields=False,
        map_location="cpu",
    )
    assert lazy.is_sharded is True
    assert lazy.fields == {}
    assert tuple(path.name for path in lazy.shard_paths) == (
        "rank-0000.pt",
        "rank-0001.pt",
    )

    gathered = Result.load_sharded(
        directory,
        scene=solver.logical_scene,
        prepared_scene=solver.scene,
        gather_fields=True,
        map_location="cpu",
    )
    assert tuple(gathered.fields) == ("EX", "EY", "EZ")
    for name, reference in references.items():
        torch.testing.assert_close(
            gathered.fields[name.upper()],
            reference.detach().cpu(),
            rtol=0.0,
            atol=0.0,
        )


def _magnetic_ade_scene() -> mw.Scene:
    scene = _small_scene(
        source_time=mw.GaussianPulse(
            frequency=_FREQUENCY,
            fwidth=0.4e9,
            amplitude=10.0,
        )
    )
    scene.add_structure(
        mw.Structure(
            name="interface_magnetic_lorentz",
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(0.4, 0.3, 0.3),
            ),
            material=mw.Material(
                mu_lorentz_poles=(
                    mw.LorentzPole(
                        delta_eps=0.5,
                        resonance_frequency=1.2e9,
                        gamma=1.0e8,
                    ),
                ),
            ),
        )
    )
    return scene


def test_magnetic_lorentz_ade_six_fields_match_one_gpu(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    pair = _run_pair(
        _magnetic_ade_scene(),
        cuda_p2p_devices,
        time_steps=48,
        full_field_dft=False,
    )
    assert pair.single.magnetic_dispersive_enabled is True
    assert all(
        shard.solver.magnetic_dispersive_enabled
        for shard in pair.distributed.shards
    )
    for name in _FIELD_NAMES:
        _assert_step_field_close(
            _distributed_field(pair, name),
            getattr(pair.single, name),
        )


def _shutoff_scene() -> mw.Scene:
    nodes = np.linspace(-0.5, 0.5, 26, dtype=np.float64)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda:0",
    )
    scene.add_source(
        mw.UniformCurrentSource(
            size=(0.08, 0.08, 0.08),
            center=(0.0, 0.0, 0.0),
            polarization="Ez",
            source_time=mw.GaussianPulse(
                frequency=_FREQUENCY,
                fwidth=0.4e9,
                amplitude=100.0,
            ),
            name="ringdown_pulse",
        )
    )
    return scene


def test_early_shutoff_step_and_six_fields_match_one_gpu(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    planned_steps = 1500
    pair = _run_pair(
        _shutoff_scene(),
        cuda_p2p_devices,
        time_steps=planned_steps,
        full_field_dft=False,
        shutoff=5.0e-2,
        shutoff_check_interval=25,
    )

    assert pair.single._shutoff_triggered is True
    assert pair.distributed._shutoff_triggered is True
    assert pair.single._shutoff_step == pair.distributed._shutoff_step
    assert pair.single._shutoff_step is not None
    assert pair.single._shutoff_step < planned_steps - 1
    executed_steps = pair.single._shutoff_step + 1
    assert all(
        shard.solver._shutoff_step == pair.single._shutoff_step
        for shard in pair.distributed.shards
    )
    stats = pair.distributed.parallel_stats
    assert stats["halo_bytes_total"] == stats["halo_bytes_per_step"] * executed_steps
    for name in _FIELD_NAMES:
        _assert_step_field_close(
            _distributed_field(pair, name),
            getattr(pair.single, name),
        )
