"""Exact fused observer oracle, storage guards and whole-step integration."""
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from tests.support import frozen_oracles


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Core observers require CUDA.")


def _groups(count, dtype, magnetic=False, field_dtype=torch.float32):
    generator = torch.Generator(device="cuda").manual_seed(142)
    weights = torch.randn((3, 9, 4), device="cuda", dtype=dtype, generator=generator)
    active = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    groups = []
    for index in range(count):
        frequencies, points = 4 + index % 3, 19 + index
        field = torch.randn((3, 3, 4, 5), device="cuda", dtype=field_dtype, generator=generator)
        offsets = torch.randint(field.numel(), (3, points), device="cuda", generator=generator)
        offsets[:, :3] = torch.tensor([-1, field.numel(), field.numel() + 7], device="cuda")
        mapping = torch.randint(weights.numel() // 4, (3, frequencies), device="cuda", generator=generator)
        mapping[0, :3] = torch.tensor([-1, weights.numel() // 4, weights.numel() // 4 + 3], device="cuda")
        real = torch.randn((3, frequencies, points), device="cuda", dtype=dtype, generator=generator)
        imag = torch.randn(real.shape, device="cuda", dtype=dtype, generator=generator)
        groups.append([field, offsets, mapping, weights, active, real, imag, bool(index % 2) ^ magnetic])
    return groups, weights, active


def _gather(extension, groups, weights, active):
    return extension.CoreObserverGather(
        *([args[index] for args in groups] for index in (0, 1, 2, 5, 6, 7)), weights, active,
    )


@pytest.mark.parametrize("count", [1, 3, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("magnetic", [False, True])
def test_fused_gather_equal_native(count, dtype, magnetic):
    _assert_gather_equal_native(*_groups(count, dtype, magnetic))


@pytest.mark.parametrize("count", [1, 20])
@pytest.mark.parametrize("magnetic", [False, True])
def test_fused_gather_float64_fields_equal_native(count, magnetic):
    _assert_gather_equal_native(*_groups(count, torch.float64, magnetic, field_dtype=torch.float64))


def _assert_gather_equal_native(groups, weights, active):
    from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension
    from witwin.maxwell.fdtd.cuda.core.build import build_extension

    native = get_compiled_extension()
    references = [args[:5] + [args[5].clone(), args[6].clone(), args[7]] for args in groups]
    initial = [(args[5].clone(), args[6].clone()) for args in groups]
    gather = _gather(build_extension(), groups, weights, active)
    for iteration in range(3):
        if iteration:
            weights.mul_(-0.75)
        for args in references:
            native.batch_observer_gather(*args)
        gather.launch()
        for args, reference, (real, imag) in zip(groups, references, initial):
            assert torch.equal(args[5], reference[5])
            assert torch.equal(args[6], reference[6])
            # Guarded points, frequencies and inactive lanes retain their input bits.
            assert torch.equal(args[5][1], real[1])
            assert torch.equal(args[6][1], imag[1])
            assert torch.equal(args[5][:, :, :3], real[:, :, :3])
            assert torch.equal(args[6][:, :, :3], imag[:, :, :3])
            assert torch.equal(args[5][0, :3], real[0, :3])
            assert torch.equal(args[6][0, :3], imag[0, :3])
        assert any(not torch.equal(args[5], before[0]) for args, before in zip(groups, initial))


@pytest.mark.parametrize("outputs", [(5, 5), (5, 6), (6, 6)])
@pytest.mark.parametrize("other", [1, 16])
def test_cross_group_accumulator_overlap(outputs, other):
    from witwin.maxwell.fdtd.cuda.core.build import build_extension

    groups, weights, active = _groups(other + 1, torch.float32)
    first, second = groups[0][outputs[0]], groups[other][outputs[1]]
    backing = torch.empty(first.numel() + second.numel(), device="cuda")
    groups[0][outputs[0]] = backing[:first.numel()].view_as(first)
    groups[other][outputs[1]] = backing[first.numel() - 1:-1].view_as(second)
    with pytest.raises(RuntimeError, match="overlap"):
        _gather(build_extension(), groups, weights, active)


def test_cross_group_input_overlap():
    from witwin.maxwell.fdtd.cuda.core.build import build_extension

    groups, weights, active = _groups(2, torch.float32)
    real = groups[0][5]
    groups[1][0] = real.view(3, 1, 4, 19)
    with pytest.raises(RuntimeError, match="overlap"):
        _gather(build_extension(), groups, weights, active)


def _scene_worker(destination):
    import witwin.maxwell as mw
    from benchmark.core_path.gates import api_dialect, make_simulation, tensors
    from benchmark.core_path.scenes import CoreCase, FREQUENCY, _grid_scene
    from witwin.maxwell.fdtd.runtime.core_path import CorePathRuntime

    scene = _grid_scene((32, 32, 32))
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.PlaneMonitor(
            "plane", axis="y", position=0.0, fields=("Ex", "Ez", "Hy", "Hz"), frequencies=(FREQUENCY,),
        ),
        mw.PointMonitor("point", position=(0.1, 0.1, 0.25), fields=("Ez",)),
    ))
    simulation = make_simulation(CoreCase("observer_gather", scene), steps=64, full_field_dft=False)
    prepared = simulation.prepare()
    solver = prepared.solver
    assert max(solver.Nx, solver.Ny, solver.Nz) <= 48
    launches, recorded_steps = [], []
    prepare_step = CorePathRuntime.prepare_step

    def measured_prepare(runtime, *args, **kwargs):
        step, engine = prepare_step(runtime, *args, **kwargs)
        assert 1 < len(engine.groups) <= 16
        assert runtime.observer_gather is not None

        def measured_step():
            before = len(launches)
            step()
            assert len(launches) - before == 1
            recorded_steps.append(1)

        return measured_step, engine

    extension = solver._core_runtime.extension
    launch = extension.CoreObserverGather.launch

    def measured_launch(gather):
        launches.append(1)
        return launch(gather)

    with patch.object(extension.CoreObserverGather, "launch", measured_launch), \
            patch.object(CorePathRuntime, "prepare_step", measured_prepare):
        result = prepared.run()
    assert solver._core_step_graph_active
    # Graph replay runs recorded CUDA launches without reentering Python.
    assert recorded_steps and len(launches) == len(recorded_steps)
    assert sum(solver._core_step_graph_chunks) == 64
    payload = tensors(api_dialect.monitors(result), "monitors")
    assert payload and any(torch.count_nonzero(value) for value in payload.values())
    torch.save(payload, destination)


def test_core_plane_point_monitors_equal_existing_path(tmp_path):
    payloads = [frozen_oracles.load("tests/fdtd/core_path/test_observer_gather.py::plane_point_monitors")]
    destination = tmp_path / "candidate.pt"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(destination)], check=True,
        env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    payloads.append(torch.load(destination, weights_only=True))
    assert payloads[0].keys() == payloads[1].keys()
    for name in payloads[0]:
        assert torch.equal(payloads[0][name], payloads[1][name]), name


if __name__ == "__main__":
    _scene_worker(*sys.argv[1:])
