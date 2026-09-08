"""Only a source-scale template proves material reuse across cohorts."""
import importlib
import gc
import weakref

import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("density_changes", [False, True])
def test_template_material_compilation_count_and_updated_spectra(monkeypatch, density_changes):
    module = importlib.import_module("witwin.maxwell.scene")
    original = module.compile_material_model
    calls = []
    def compile_materials(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    sim = simulation(steps=31)
    if density_changes:
        sim.scene.add_material_region(mw.MaterialRegion(name="rho",
            geometry=mw.Box(size=(.05, .05, .05), device="cuda:0"),
            density=torch.zeros(2, 2, 2, device="cuda:0"), eps_bounds=(1., 5.)))
    scales = torch.tensor([1., 2., 0., -1., 3.], device="cuda:0")
    parameters = mw.BatchParameters(source_scales={"drive": scales},
        region_densities={"rho": torch.linspace(0., 1., 5, device="cuda:0")[:, None, None, None].expand(5, 2, 2, 2)}
        if density_changes else None)
    batch = mw.Simulation.batch(sim, parameters=parameters,
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required"))
    monkeypatch.setattr(module, "compile_material_model", compile_materials)
    with batch.prepare() as prepared:
        first = prepared.run()
        first.results()
        assert len(calls) == (5 if density_changes else 1)
        if not density_changes:
            assert len(prepared._fixed_material_cache) == 1
        scales.mul_(2)
        second = prepared.run()
        second.results()
        assert len(calls) == (10 if density_changes else 1)
        for a, b in zip(first, second):
            torch.testing.assert_close(b.raw_monitor("probe")["components"]["Ez"],
                2 * a.raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)
    assert not prepared._fixed_material_cache
    monkeypatch.setattr(module, "compile_material_model", original)
    for index in range(5):
        reference = mw.Simulation.fdtd(batch.scene(index), frequencies=sim.frequencies,
            run_time=sim.config.run_time, spectral_sampler=sim.config.spectral_sampler,
            cuda_graph=False).run()
        torch.testing.assert_close(second[index].raw_monitor("probe")["components"]["Ez"],
            reference.raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-6)


def test_different_material_lanes_share_only_equal_static_arrays():
    sims = [simulation(index, steps=31) for index in range(3)]
    for index, sim in enumerate(sims):
        sim.scene.add_structure(mw.Structure(name="dielectric",
            geometry=mw.Box(size=(.08, .08, .08), device="cuda:0"),
            material=mw.Material(eps_r=2. + index)))
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3, graph="required"))
    with batch.prepare() as prepared:
        a, b, c = prepared._slot.solvers
        assert a.mu_Hx.data_ptr() == b.mu_Hx.data_ptr() == c.mu_Hx.data_ptr()
        assert a.material_mu_r.data_ptr() == b.material_mu_r.data_ptr()
        assert a.eps_Ex.data_ptr() != b.eps_Ex.data_ptr()
        assert not torch.equal(a.eps_Ex, b.eps_Ex)
        for solver in (b, c):
            assert solver.scene.compile_materials() is solver._compiled_material_model
            assert solver.mu_r is a.mu_r
            assert solver.mu_r_components["x"] is a.mu_r_components["x"]
        first = prepared.run()
        again = prepared.run()
        first.results()
        again.results()
    for index, sim in enumerate(sims):
        expected = sim.run().raw_monitor("probe")["components"]["Ez"]
        for result in (first[index], again[index]):
            torch.testing.assert_close(result.raw_monitor("probe")["components"]["Ez"],
                                       expected, rtol=1e-4, atol=1e-6)


def test_evicted_cohort_releases_without_cyclic_collection():
    batch = mw.Simulation.batch([simulation(i, steps=8) for i in range(5)],
        variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2, graph="required"))
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with batch.prepare() as prepared:
            evicted = [weakref.ref(solver) for solver in prepared._slot.solvers]
            results = prepared.run()
            results.results()
            assert all(reference() is None for reference in evicted)
            resident = [weakref.ref(solver) for solver in prepared._slot.solvers]
        assert all(reference() is None for reference in resident)
    finally:
        if was_enabled:
            gc.enable()
