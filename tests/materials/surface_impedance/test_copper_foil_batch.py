"""Native batch lanes carry independent foil physics and spectral histories."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot
from witwin.maxwell.fdtd.copper_foil import iter_copper_state


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CopperFoil batching requires CUDA.")


def _simulation(index=0, *, steps=96):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(), device="cuda:0",
        structures=[mw.Structure(name="foil", geometry=mw.Box(
            position=torch.tensor([.01, 0., 0.], dtype=torch.float64, device="cuda:0"),
            size=torch.tensor([0., .02, .02], dtype=torch.float64, device="cuda:0")),
            material=mw.CopperFoil(thickness=(2 + index) * 1e-6, frequency_range=(.5e9, 2e9)))],
        sources=[mw.PointDipole(name="drive", position=(-.01, 0., 0.), polarization="Ez", profile="ideal",
            source_time=mw.CW(frequency=1e9, amplitude=1 + index))],
        monitors=[mw.PointMonitor(name="probe", position=(.015, 0., 0.), fields=("Ez",))])
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=steps, dt=5e-12),
        spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=True, cuda_graph=False)


def test_native_graph_batch_matches_independent_and_reuses_surface_spectrum_storage():
    simulations = [_simulation(index) for index in range(2)]
    references = [simulation.run() for simulation in simulations]
    batch = mw.Simulation.batch(simulations,
        variation=mw.BatchVariation(structures={"foil": mw.Box(
            position=(.01, 0., 0.), size=(.04, .04, .04), device="cuda:0")}, sources=("drive",)),
        outputs=mw.BatchOutput(fields=("Ey", "Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run()
        first.results()
        assert prepared.report.execution_tier == "native_pipeline_graph"
        states = [[tensor.data_ptr() for _name, tensor in iter_copper_state(solver)]
                  for solver in prepared._slot.solvers]
        graphs = tuple(prepared._graphs.values())
        second = prepared.run()
        second.results()
        assert graphs == tuple(prepared._graphs.values())
        assert states == [[tensor.data_ptr() for _name, tensor in iter_copper_state(solver)]
                          for solver in prepared._slot.solvers]
    for index, expected in enumerate(references):
        for component in ("Ey", "Ez"):
            torch.testing.assert_close(first[index].tensor(component), expected.tensor(component), rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(second[index].tensor(component), first[index].tensor(component), rtol=0, atol=0)
        actual = first[index].conductor_losses["foil"]
        reference = expected.conductor_losses["foil"]
        assert abs(actual["dissipated_energy"] - reference["dissipated_energy"]) / reference["dissipated_energy"] < 1e-4
        torch.testing.assert_close(actual["surface_power"], reference["surface_power"], rtol=1e-4, atol=1e-10)
    assert first[0].conductor_losses["foil"]["dissipated_energy"] != first[1].conductor_losses["foil"]["dissipated_energy"]


def test_inactive_native_lane_freezes_all_foil_fields_and_auxiliary_state():
    solvers = [_simulation(index).prepare().solver for index in range(2)]
    slot = NativeFieldSlot(2, "cuda:0")
    slot.bind(solvers)
    for lane, solver in enumerate(solvers):
        for entry in solver._copper_foils:
            entry["state"].fill_((lane + 1) * 1e-7)
            entry["right_e"].fill_(.1)
    slot.active[0] = 0
    fields = {name: getattr(solvers[0], name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    states = {name: tensor.clone() for name, tensor in iter_copper_state(solvers[0])}
    for _ in range(4):
        slot.advance()
    for name, value in fields.items():
        torch.testing.assert_close(getattr(solvers[0], name), value, rtol=0, atol=0)
    for name, tensor in iter_copper_state(solvers[0]):
        torch.testing.assert_close(tensor, states[name], rtol=0, atol=0)
    assert any(float(entry["cumulative_loss"]) > 0 for entry in solvers[1]._copper_foils)


@pytest.mark.parametrize("graph", ["off", "required"])
def test_rf_excitation_and_passive_rlc_foil_batch_matches_independent(graph):
    simulations = []
    for index in range(2):
        base = _simulation(0)
        port = mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(0., 0., -.0025),
            size=(.015, .015, 0.)))
        load = mw.LumpedPort(name="load", positive=(-.005, 0., .005), negative=(-.005, 0., -.005),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(-.005, 0., -.0025),
            size=(.015, .015, 0.)), termination=mw.SeriesRLC(r=75., l=2e-9, c=1e-12))
        scene = base.scene.clone(sources=(), ports=(port, load))
        simulations.append(mw.Simulation.fdtd(scene, frequency=1e9,
            excitations=mw.PortExcitation("feed", amplitude=1., source_time=mw.CW(frequency=1e9)),
            run_time=mw.TimeConfig(time_steps=96, dt=5e-12),
            spectral_sampler=mw.SpectralSampler(window="none"), full_field_dft=True, cuda_graph=False))
    references = [simulation.run() for simulation in simulations]
    batch = mw.Simulation.batch(simulations, outputs=mw.BatchOutput(fields=("Ez",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph=graph, graph_steps=7))
    with batch.prepare() as prepared:
        results = prepared.run().results()
        graphs = tuple(prepared._graphs.values())
        pointers = [(runtime.accumulator._voltage_sum.data_ptr(), runtime._drive_table.data_ptr())
                    for solver in prepared._slot.solvers for runtime in solver._port_runtimes]
        repeated = prepared.run().results()
        assert graphs == tuple(prepared._graphs.values())
        assert pointers == [(runtime.accumulator._voltage_sum.data_ptr(), runtime._drive_table.data_ptr())
                            for solver in prepared._slot.solvers for runtime in solver._port_runtimes]
    for index, reference in enumerate(references):
        for actual in (results[index], repeated[index]):
            torch.testing.assert_close(actual.tensor("Ez"), reference.tensor("Ez"), rtol=2e-4, atol=1e-6)
            for port_name in ("feed", "load"):
                for name in ("voltage", "current", "a", "b"):
                    torch.testing.assert_close(getattr(actual.port(port_name), name), getattr(reference.port(port_name), name),
                                               rtol=2e-4, atol=1e-8)
            torch.testing.assert_close(actual.conductor_losses["foil"]["surface_power"],
                                       reference.conductor_losses["foil"]["surface_power"], rtol=2e-4, atol=1e-12)
