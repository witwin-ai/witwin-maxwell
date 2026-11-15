"""Native batch lanes run the fused core Yee launches and equal their solo core runs bitwise."""
from dataclasses import replace
import math

import pytest
import torch
import witwin.maxwell as mw

from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch runtime requires a GPU")


def _execution(**kwargs):
    return mw.execution.BatchExecution(devices=("cuda:0",), reserve_bytes=0, **kwargs)


def _probe(result):
    return result.raw_monitor("probe")["components"]["Ez"]


def _with_graph(sim):
    sim.config = replace(sim.config, cuda_graph=True)
    return sim


@pytest.mark.parametrize("boundary", [mw.BoundarySpec.none(), mw.BoundarySpec.pml(num_layers=3),
                                       mw.BoundarySpec.periodic()])
@pytest.mark.parametrize("graph", ["off", "required"])
def test_lanes_run_on_core_and_equal_solo_core_runs(boundary, graph):
    sims = [_with_graph(simulation(index, boundary)) for index in range(3)]
    batch = mw.Simulation.batch(sims, variation=mw.execution.BatchVariation(sources=("drive",)),
                                outputs=mw.execution.BatchOutput(fields=("Ez",)),
                                execution=_execution(batch_size=3, graph=graph, graph_steps=8))
    results = batch.run()
    results.results()
    for index, sim in enumerate(sims):
        lane = results[index]
        solo = _full_field(sim).run()
        assert solo.solver._core_runtime is not None
        assert torch.equal(_probe(lane), _probe(solo))
        assert torch.equal(lane.field("Ez").values, solo.field("Ez").values)


def _full_field(sim):
    return mw.Simulation.fdtd(sim.scene, frequencies=sim.frequencies, config=replace(sim.config, full_field_dft=True))


def test_shared_coefficient_bank_shares_compressed_tables():
    template = simulation()
    scales = torch.tensor([1.0, -0.5, 2.0], device="cuda:0")
    batch = mw.Simulation.batch(template, parameters=mw.execution.BatchParameters(source_scales={"drive": scales}),
                                execution=_execution(batch_size=3, graph="off"))
    with batch.prepare() as prepared:
        prepared.run().results()
        cores = prepared._slot._core
        for family in ("h", "e"):
            indices = {core.coefficients[family][0].data_ptr() for core in cores}
            assert len(indices) == 1, family
        assert all(not hasattr(solver, "_core_runtime") for solver in prepared._slot.solvers)
    distinct = [simulation(index) for index in range(2)]
    batch = mw.Simulation.batch(distinct, variation=mw.execution.BatchVariation(sources=("drive",)),
                                execution=_execution(batch_size=2, graph="off"))
    with batch.prepare() as prepared:
        prepared.run().results()
        tables = [core.coefficients["e"] for core in prepared._slot._core]
        assert tables[0][0].data_ptr() != tables[1][0].data_ptr()


def _zero_dc_simulation(index):
    # A zero-DC pulse lets the field energy decay below the shutoff fraction at lane-specific checks.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(0.025),
                     boundary=mw.BoundarySpec.pml(num_layers=3),
                     structures=(
                         mw.Structure(name="design",
                             geometry=mw.Box(position=(0.05, 0., 0.), size=(0.07, 0.08, 0.09), device="cuda:0"),
                             material=(mw.Material(eps_r=1.0), mw.Material(eps_r=12.0), mw.Material(eps_r=2.0, lorentz_poles=(
                             mw.media.LorentzPole(delta_eps=3.0, resonance_frequency=1.0e9, gamma=1e7),)))[index]),
                     ),
                     sources=(
                         mw.PointDipole(name="drive", position=(-0.05, 0., 0.), polarization="Ez",
                             source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=1.0 + index, phase=math.pi / 2)),
                     ),
                     monitors=(mw.PointMonitor(name="probe", position=(0.1, 0., 0.), fields=("Ez",)),))
    return mw.Simulation.fdtd(scene, frequencies=(1e9,), config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=640, dt=2e-11), spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
        shutoff=1e-4, shutoff_check_interval=8, device="cuda:0"))


def test_shutoff_lanes_retire_frozen_and_equal_solo_runs():
    sims = [_zero_dc_simulation(index) for index in range(3)]
    batch = mw.Simulation.batch(sims, variation=mw.execution.BatchVariation(sources=("drive",)),
                                execution=_execution(batch_size=3, graph="required", graph_steps=8))
    with batch.prepare() as prepared:
        results = prepared.run()
        results.results()
        lanes = [{name: getattr(solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
                 for solver in prepared._slot.solvers]
        steps = [solver._shutoff_step for solver in prepared._slot.solvers]
    # Lanes retire at different steps (a lane that never trips runs to the horizon).
    assert any(step is not None for step in steps) and len(set(steps)) > 1, steps
    for index, sim in enumerate(sims):
        solo = sim.prepare()
        result = solo.run()
        assert solo.solver._shutoff_step == steps[index]
        assert torch.equal(_probe(results[index]), _probe(result))
        for name, value in lanes[index].items():
            assert torch.equal(value, getattr(solo.solver, name)), (index, name)
