"""End-to-end native cohorts compared with independent public simulations."""
from dataclasses import replace

import pytest
import torch
import witwin.maxwell as mw

from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot, FIELDS
from witwin.maxwell.fdtd.runtime.stepping import solve_steps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch runtime requires a GPU")


def simulation(index=0, boundary=None, frequencies=(1e9,), steps=80, memory_mode="auto"):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
                     grid=mw.GridSpec.uniform(0.025), device="cuda:0",
                     boundary=boundary or mw.BoundarySpec.pml(num_layers=3))
    scene.add_structure(mw.Structure(name="design",
        geometry=mw.Box(position=(0.05, 0., 0.), size=(0.07, 0.08, 0.09), device="cuda:0"),
        material=mw.Material(eps_r=2.0 + 0.25 * index)))
    scene.add_source(mw.PointDipole(name="drive", position=(-0.05, 0., 0.), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=1.0 + index)))
    scene.add_monitor(mw.PointMonitor(name="probe", position=(0.1, 0., 0.), fields=("Ez",)))
    return mw.Simulation.fdtd(scene, frequencies=frequencies,
        run_time=mw.TimeConfig(time_steps=steps, dt=2e-11),
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False,
        cpml_config={"memory_mode": memory_mode})


def run_cohort(slot, steps=80, frequencies=(1e9,)):
    slot.reset()
    generators = [solve_steps(s, steps, dft_frequency=frequencies, full_field_dft=False,
                             dft_window="none", cooperative=True) for s in slot.solvers]
    for g in generators:
        assert next(g) == 0
    results = {}
    for n in range(steps):
        slot.advance()
        for lane, g in enumerate(generators):
            try:
                assert next(g) == n + 1
            except StopIteration as done:
                results[lane] = done.value
                slot.active[lane] = 0
    return results


@pytest.mark.parametrize("boundary", [mw.BoundarySpec.none(), mw.BoundarySpec.pml(num_layers=3),
                                       mw.BoundarySpec.periodic()])
@pytest.mark.parametrize("capacity,count", [(1, 1), (3, 3), (4, 2)])
@torch.no_grad()
def test_native_cohort_matches_independent_full_solve_and_reset(boundary, capacity, count):
    sims = [simulation(i, boundary) for i in range(count)]
    solvers = [s.prepare().solver for s in sims]
    slot = NativeFieldSlot(capacity, "cuda:0")
    slot.bind(solvers)
    outputs = run_cohort(slot)
    saved = {name: value.clone() for name, value in slot.buffers.items() if name in slot.state_names}
    for lane, sim in enumerate(sims):
        ordinary = sim.prepare()
        result = ordinary.run()
        for name in slot.state_names:
            torch.testing.assert_close(slot.buffers[name][lane], getattr(ordinary.solver, name), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(outputs[lane]["observers"]["probe"]["components"]["Ez"],
                                   result.raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)
    run_cohort(slot)
    for name, value in saved.items():
        torch.testing.assert_close(slot.buffers[name], value, rtol=0, atol=0)
    pointers = {name: value.data_ptr() for name, value in slot.buffers.items()}
    # Reload different material and source values into the same persistent storage.
    slot.bind([simulation(4 + i, boundary).prepare().solver for i in range(count)])
    run_cohort(slot)
    assert {name: value.data_ptr() for name, value in slot.buffers.items()} == pointers
    assert not torch.equal(slot.buffers["Ez"][0], saved["Ez"][0])


@pytest.mark.parametrize("count", [1, 3])
@torch.no_grad()
def test_slab_cpml_and_31_frequency_observers_match_independent(count):
    frequencies = tuple(0.7e9 + i * 0.02e9 for i in range(31))
    sims = [simulation(i, frequencies=frequencies, memory_mode="slab") for i in range(count)]
    slot = NativeFieldSlot(4, "cuda:0")
    slot.bind([s.prepare().solver for s in sims])
    assert all(s._cpml_memory_mode == "slab" for s in slot.solvers)
    result = run_cohort(slot, frequencies=frequencies)
    for lane, sim in enumerate(sims):
        ordinary = sim.prepare()
        reference = ordinary.run()
        for name in slot.state_names:
            torch.testing.assert_close(slot.buffers[name][lane], getattr(ordinary.solver, name), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(result[lane]["observers"]["probe"]["components"]["Ez"],
            reference.raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)
    assert all(not s.dft_enabled for s in slot.solvers)
