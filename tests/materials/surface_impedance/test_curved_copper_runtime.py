"""Rejected spatial prototype: algebra/energy/lifecycle are necessary, insufficient.

Public SurfaceMesh-Cu stays guarded. Private diagnostics preserve evidence that
passing these controls does not imply correct high-contrast transmission.
"""
import pytest
import torch

import witwin.maxwell as mw
from tests.materials.surface_impedance.test_cut_surface import _mesh
from scripts.dev.fdtd.curved_copper_prototype import PrototypeSimulation, prototype_hooks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native copper requires CUDA")


@pytest.fixture(autouse=True)
def _isolated_rejected_prototype_hooks():
    with prototype_hooks():
        yield


def _simulation(*, graph=False, thickness=2e-6, bent=False, source=True):
    geometry = (mw.SurfaceMesh.bent_rectangle(width=.02,length=.016,radius=.03,segments=12,device="cuda")
                if bent else _mesh())
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04,.04),)*3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(),device="cuda",
        structures=(mw.Structure(name="film",geometry=geometry,
            material=mw.CopperFoil(thickness=thickness,frequency_range=(.5e9,2e9))),),
        sources=(mw.PointDipole(position=(-.025,0.,-.02),profile="ideal",polarization="Ez",source_time=mw.CW(1e9)),) if source else ())
    return PrototypeSimulation(scene,frequency=1e9,cuda_graph=graph,full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none"),run_time=mw.TimeConfig(time_steps=64,dt=5e-12))


@pytest.mark.parametrize("bent",(False,True))
def test_joint_curved_native_graph_and_checkpoint_match_eager(bent):
    eager = _simulation(bent=bent).run()
    graph = _simulation(graph=True,bent=bent).run()
    assert graph.solver._cuda_graph_active
    for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
        torch.testing.assert_close(getattr(graph.solver,name),getattr(eager.solver,name),rtol=2e-5,atol=1e-6)
    checkpoint = _simulation(bent=bent).prepare().run_until(27)
    resumed = _simulation(graph=True,bent=bent).prepare().run(resume_from=checkpoint)
    for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
        torch.testing.assert_close(getattr(resumed.solver,name),getattr(eager.solver,name),rtol=2e-5,atol=1e-6)
    reference = eager.solver._curved_copper[0]["runtime"]
    for other in (graph.solver._curved_copper[0]["runtime"],resumed.solver._curved_copper[0]["runtime"]):
        for name in ("ze","zm","cumulative_loss"):
            torch.testing.assert_close(getattr(other,name),getattr(reference,name),rtol=2e-5,atol=1e-20)
    assert reference.cumulative_loss > 0
    assert reference.modal_storage() > 0


def test_curved_resume_rejects_changed_physical_conductor_before_mutation():
    checkpoint = _simulation().prepare().run_until(27)
    prepared = _simulation(thickness=2.1e-6).prepare()
    before = prepared.solver.Ez.clone()
    with pytest.raises(ValueError,match="layout|does not match"):
        prepared.run(resume_from=checkpoint)
    torch.testing.assert_close(prepared.solver.Ez,before,rtol=0,atol=0)


def test_joint_curved_native_modified_energy_plus_both_modal_storages_and_heat_is_closed():
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block,update_magnetic_fields
    from scripts.dev.fdtd.curved_copper_prototype import _gather
    solver = _simulation(source=False,bent=True).prepare().solver
    entry = solver._curved_copper[0]
    runtime = entry["runtime"]
    generator = torch.Generator(device="cuda").manual_seed(734)
    weights = {}
    for component,names in enumerate(zip(("Ex","Ey","Ez"),("Hx","Hy","Hz"))):
        for name in names:
            magnetic = name[0] == "H"
            widths = [torch.as_tensor(getattr(solver.scene,coordinate+("_dual64" if (axis == component)==magnetic else "_primal64")),device="cuda")
                      for axis,coordinate in enumerate(("dx","dy","dz"))]
            weights[name] = getattr(solver,("mu_" if magnetic else "eps_")+name).double()*widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
            if not magnetic:
                field = getattr(solver,name)
                field[3:-3,3:-3,3:-3] = torch.randn(field[3:-3,3:-3,3:-3].shape,device="cuda",generator=generator)
    def energy():
        old_h = tuple(getattr(solver,name).clone() for name in ("Hx","Hy","Hz"))
        _gather(solver,entry["magnetic_groups"],entry["magnetic"])
        trace = runtime.gh@entry["magnetic"]
        tail = runtime.tail*(trace@torch.cholesky_solve(trace[:,None],runtime.surface_cholesky)[:,0])/2
        update_magnetic_fields(solver,solver.Hx,solver.Hy,solver.Hz,solver.Ex,solver.Ey,solver.Ez)
        value = sum((weights[name]*getattr(solver,name).double().square()).sum()/2 for name in ("Ex","Ey","Ez"))
        value += sum((weights[name]*previous.double()*getattr(solver,name).double()).sum()/2
                     for name,previous in zip(("Hx","Hy","Hz"),old_h))
        for name,previous in zip(("Hx","Hy","Hz"),old_h):
            getattr(solver,name).copy_(previous)
        return value+tail+runtime.modal_storage()+runtime.cumulative_loss
    reference = energy()
    values = []
    for _ in range(256):
        _field_update_block(solver,0.)
        values.append(energy())
    drift = (torch.stack(values)-reference).abs().max()/reference
    print(dict(steps=256,maximum_relative_total_energy_drift=float(drift),heat=float(runtime.cumulative_loss)))
    assert float(drift) < 5e-6
    assert runtime.cumulative_loss > 0
