"""Batched source patches preserve phase, overlap order, and native rounding."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation
from witwin.maxwell.fdtd.excitation.temporal import BatchSourceEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("cw", [False, True])
@pytest.mark.parametrize("position", [(-.05, 0., 0.), (-.2, -.2, 0.)])
@pytest.mark.parametrize("boundary", [mw.BoundarySpec.pml(num_layers=3), mw.BoundarySpec.periodic()])
def test_patch_updates_match_independent_native_arithmetic(cw, boundary, position):
    sims = []
    for index in range(3):
        scene = simulation(index, boundary=boundary).scene_input.clone()
        scene.sources[0] = mw.PointDipole(name="drive", position=position, polarization="Ez",
                                         source_time=scene.sources[0].source_time)
        if cw:
            scene.sources.clear()
            scene.add_source(mw.PointDipole(name="drive", position=position, polarization="Ez",
                source_time=mw.CW(frequency=1e9, amplitude=1.+index, phase=.4)))
        scene.add_source(mw.PointDipole(name="overlap", position=position, polarization="Ez",
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=.3)))
        sims.append(mw.Simulation.fdtd(scene, frequency=1e9,
            run_time=mw.TimeConfig(time_steps=120, dt=2e-11), cuda_graph=False))
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3))
    with batch.prepare() as prepared, torch.cuda.stream(prepared._stream):
        slot = prepared._slot
        source = BatchSourceEngine(slot, 120)
        for n in (0, 20, 60, 119):
            slot.reset()
            for name in ("Ex", "Ey", "Ez"):
                slot.buffers[name].normal_()
            saved = {name: slot.buffers[name].clone() for name in ("Ex", "Ey", "Ez")}
            for solver in slot.solvers:
                solver.add_source(time_value=n*solver.dt)
            expected = {name: slot.buffers[name].clone() for name in saved}
            for name, value in saved.items():
                slot.buffers[name].copy_(value)
            source.advance(torch.tensor([n], device="cuda:0"))
            for name in saved:
                torch.testing.assert_close(slot.buffers[name], expected[name], rtol=0, atol=0)
