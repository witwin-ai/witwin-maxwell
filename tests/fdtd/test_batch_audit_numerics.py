"""Audit spectral normalization, boundaries, graded grids, and lane termination."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw

from tests.fdtd.cuda.test_batch_runtime import simulation
from tests.fdtd.test_fdtd_autoshutoff import _vacuum_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Batch audit requires CUDA")


def assert_tensors(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
    elif isinstance(expected, dict):
        for key in expected:
            assert key in actual
            assert_tensors(actual[key], expected[key])


@pytest.mark.parametrize("boundary", [mw.BoundarySpec.pec(), mw.BoundarySpec.pmc(),
                                       mw.BoundarySpec.pml(num_layers=3)])
def test_graded_grid_normalized_multifrequency_planes_and_full_fields(boundary):
    sims = []
    for index in range(2):
        scene = simulation(index, boundary=boundary).scene_input.clone()
        axis = np.linspace(-1., 1., 17)
        coordinates = .2 * (axis + .1 * axis ** 3) / 1.1
        scene.grid = mw.GridSpec.custom(coordinates, coordinates, coordinates)
        scene.add_monitor(mw.PlaneMonitor("plane", axis="z", position=.05,
            fields=("Ex", "Ey", "Hx", "Hy"), compute_flux=True))
        scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=.05))
        sims.append(mw.Simulation.fdtd(scene, frequencies=(.8e9, 1e9, 1.2e9),
            run_time=mw.TimeConfig(time_steps=120, dt=2e-11),
            spectral_sampler=mw.SpectralSampler(window="hanning", normalize_source=True),
            cuda_graph=False, full_field_dft=True))
    batch = mw.Simulation.batch(sims,
        variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3),
        outputs=mw.BatchOutput(fields=("Ex", "Ey", "Ez")))
    with batch.prepare() as prepared:
        result = prepared.run()
        result.results()
        saved = [{name: value.clone() for name, value in item._fields.items()} for item in result]
        repeated = prepared.run()
        repeated.results()
    for index, sim in enumerate(sims):
        reference = sim.run()
        assert_tensors({name.lower(): value for name, value in result[index]._fields.items()},
                       {name.lower(): value for name, value in reference._fields.items()})
        for name in ("probe", "plane", "flux"):
            assert_tensors(result[index].raw_monitor(name), reference.raw_monitor(name))
        for name, value in saved[index].items():
            torch.testing.assert_close(result[index]._fields[name], value, rtol=0, atol=0)
            torch.testing.assert_close(repeated[index]._fields[name], value, rtol=0, atol=0)


def test_shutoff_finishes_one_lane_while_another_keeps_advancing():
    sims = []
    for amplitude in (100., 0.):
        scene = _vacuum_scene(source_time=mw.GaussianPulse(
            frequency=1e9, fwidth=.4e9, amplitude=amplitude))
        scene.add_monitor(mw.PointMonitor("probe", position=(.1, 0., 0.), fields=("Ez",)))
        sims.append(mw.Simulation.fdtd(scene, frequency=1e9,
            run_time=mw.TimeConfig(time_steps=3000), full_field_dft=False,
            shutoff=5e-2, shutoff_check_interval=50, cuda_graph=False))
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("src",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2))
    results = batch.run()
    results.results()
    assert results[0].stats()["shutoff_triggered"]
    assert not results[1].stats()["shutoff_triggered"]
    assert results[0].stats()["steps_run"] < results[1].stats()["steps_run"]
    for index, sim in enumerate(sims):
        reference = sim.run()
        assert results[index].stats()["steps_run"] == reference.stats()["steps_run"]
        assert_tensors(results[index].raw_monitor("probe"), reference.raw_monitor("probe"))
