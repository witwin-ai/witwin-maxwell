"""Independent electric polarization histories on both sides of a foil."""
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.copper_foil_ade import (
    initialize_copper_trace_ade, prepare_copper_trace_history,
    finalize_copper_trace_ade, copper_material_energy, iter_copper_trace_ade_state,
)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_trace_ade_matches_independent_two_material_state_recurrence(device, kind, copper_torch_kernel):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for device validation")
    drive = torch.tensor([.1, .2, .3], dtype=torch.float64, device=device).reshape(3, 1, 1)
    template = dict(drive=drive.clone(), decay=.7, restoring=.4)
    entry = dict(face=SimpleNamespace(axis=0), electric_name="Ey",
                 indices=torch.tensor([[1, 0, 0]], device=device),
                 index=(torch.tensor([1], device=device), torch.tensor([0], device=device), torch.tensor([0], device=device)),
                 old_e=torch.zeros((2, 1), device=device, dtype=torch.float64),
                 left_width=.3, right_width=.6, area=torch.tensor([.4], device=device, dtype=torch.float64))
    dt = .05
    entry.update(capacity=torch.ones((2, 1), device=device, dtype=torch.float64),
                 D=torch.eye(2, device=device, dtype=torch.float64),
                 B=torch.ones((1, 2), device=device, dtype=torch.float64),
                 drive=torch.ones((1, 2), device=device, dtype=torch.float64),
                 admittance_scale=torch.ones(1, device=device, dtype=torch.float64))
    solver = SimpleNamespace(_copper_foils=[entry], _dispersive_templates={"Ey": {kind: [template]}},
                             device=device, dt=dt, sigma_e_Ey=torch.zeros_like(drive))
    initialize_copper_trace_ade(solver)
    assert template["drive"][1] == 0
    local = entry["material_ade"][0]
    if "polarization" in local:
        local["polarization"].copy_(torch.tensor([[.2], [-.3]], device=device, dtype=torch.float64))
    local["current"].fill_(.1)
    a = .7
    d = drive[[0, 2], 0, 0].reshape(2, 1)
    for step in range(32):
        old_e = torch.tensor([[.2 + .01 * step], [-.1 + .003 * step]], device=device, dtype=torch.float64)
        new_e = old_e * .9
        midpoint_e = (old_e + new_e) / 2
        entry["old_e"].copy_(old_e)
        previous_energy = copper_material_energy(entry).clone()
        previous_heat = entry["material_dissipated_energy"].clone()
        previous_work = entry["material_work"].clone()
        p = local.get("polarization", torch.zeros_like(old_e)).clone()
        j = local["current"].clone()
        if kind == "debye":
            strength, tau = d / (1-a), .5 * dt * (1+a) / (1-a)
            next_p = ((tau/dt-.5)*p + strength*midpoint_e) / (tau/dt+.5)
            expected_current = (next_p-p)/dt
            next_j = expected_current
        else:
            damping = 2*(1-a)/(dt*(1+a))
            strength = 2*d/(dt*(1+a))
            omega_sq = 2*.4/(dt*(1+a)) if kind == "lorentz" else 0.
            next_j = ((1-damping*dt/2-omega_sq*dt*dt/4)*j - dt*omega_sq*p + dt*strength*midpoint_e) / (1+damping*dt/2+omega_sq*dt*dt/4)
            expected_current = (j+next_j)/2
            next_p = p+dt*expected_current
        prepare_copper_trace_history(entry, dt)
        finalize_copper_trace_ade(entry, new_e, dt)
        torch.testing.assert_close(entry["material_current"], expected_current, rtol=1.e-12, atol=1.e-13)
        torch.testing.assert_close(local["current"], next_j, rtol=1.e-12, atol=1.e-13)
        if kind != "drude":
            torch.testing.assert_close(local["polarization"], next_p, rtol=1.e-12, atol=1.e-13)
        delta_energy = copper_material_energy(entry)-previous_energy
        delta_heat = entry["material_dissipated_energy"]-previous_heat
        delta_work = entry["material_work"]-previous_work
        torch.testing.assert_close(delta_energy+delta_heat, delta_work, rtol=1.e-10, atol=1.e-13)
        assert delta_heat >= 0
    frozen = {name: value.clone() for name, value in iter_copper_trace_ade_state(0, entry)}
    prepare_copper_trace_history(entry, dt)
    finalize_copper_trace_ade(entry, entry["old_e"], dt, activity=torch.zeros((), device=device))
    for name, value in iter_copper_trace_ade_state(0, entry):
        assert torch.equal(value, frozen[name])


def _local_material(kind):
    import witwin.maxwell as mw
    poles = {
        "debye": dict(debye_poles=(mw.DebyePole(delta_eps=1.5, tau=1.e-10),)),
        "drude": dict(drude_poles=(mw.DrudePole(plasma_frequency=2.e9, gamma=.2e9),)),
        "lorentz": dict(lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=2.5e9, gamma=.3e9),)),
        "magnetic": dict(mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=2.5e9, gamma=.3e9),)),
    }
    return mw.Material(eps_r=2., sigma_e=.1, **poles[kind])


def _local_scene(kind, *, one_side=False):
    import witwin.maxwell as mw
    from tests.gradients.test_fdtd_copper_foil_adjoint import FoilDesign
    model = FoilDesign()
    model.requires_grad_(False)
    scene = model.to_scene()
    scene.structures = (*scene.structures, mw.Structure(name="substrate",
        geometry=mw.Box(position=(.01, 0., 0.) if one_side else (0., 0., 0.),
                        size=(.02, .04, .04) if one_side else (.04, .04, .04)), material=_local_material(kind)))
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz"))
def test_gpu_local_material_and_foil_ampere_energy_close_independently(kind):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.copper_foil import begin_copper_step, advance_copper_electric
    solver = mw.Simulation.fdtd(_local_scene(kind), frequency=3.e9).prepare().solver
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(solver, name, getattr(solver, name).double())
    snapshots = []
    for entry in solver._copper_foils:
        getattr(solver, entry["electric_name"])[entry["index"]] = .2
        entry["right_e"] = torch.full_like(entry["right_e"], -.1, dtype=torch.float64)
        for pole in entry["material_ade"]:
            pole["current"].fill_(1.e-3)
            if "polarization" in pole:
                pole["polarization"].fill_(2.e-12)
        snapshots.append(dict(material_energy=copper_material_energy(entry).clone(), state=entry["state"].clone(), current=entry["material_ade"][0]["current"].clone(),
                              polarization=entry["material_ade"][0].get("polarization", torch.zeros_like(entry["old_e"])).clone()))
    begin_copper_step(solver)
    advance_copper_electric(solver)
    for entry, previous in zip(solver._copper_foils, snapshots):
        old = entry["old_e"]
        new = torch.stack((getattr(solver, entry["electric_name"])[entry["index"]], entry["right_e"]))
        midpoint = (old + new) / 2
        pole = entry["material_ade"][0]
        if kind == "debye":
            expected_current = (pole["polarization"] - previous["polarization"]) / solver.dt
        else:
            expected_current = (pole["current"] + previous["current"]) / 2
        torch.testing.assert_close(entry["material_current"], expected_current, rtol=1.e-12, atol=1.e-13)
        indices = entry["indices"]
        left, right = indices.clone(), indices.clone()
        left[:, entry["face"].axis] -= 1
        right[:, entry["face"].axis] += 1
        sigma = getattr(solver, "sigma_e_" + entry["electric_name"])
        conductance = torch.stack((sigma[tuple(left[:, axis] for axis in range(3))].double() * entry["left_width"],
                                   sigma[tuple(right[:, axis] for axis in range(3))].double() * entry["right_width"]))
        foil_midpoint = (previous["state"] + entry["state"]) / 2
        foil_current = entry["B"].T @ foil_midpoint + entry["D"] @ midpoint
        terms = (entry["capacity"] * (new - old) / solver.dt,
                 conductance * midpoint, foil_current, entry["material_widths"] * expected_current)
        residual = sum(terms)
        scale = sum(term.abs() for term in terms).sum().clamp_min(1.e-30)
        assert float(residual.abs().sum() / scale) < 1.e-10
        delta_field = .5 * torch.sum(entry["capacity"] * (new.square() - old.square()) * entry["area"])
        delta_foil = .5 * torch.sum((entry["state"].square() - previous["state"].square()) * entry["area"])
        conductive_loss = solver.dt * torch.sum(conductance * midpoint.square() * entry["area"])
        energy_terms = (delta_field, delta_foil, conductive_loss, entry["cumulative_loss"],
                        copper_material_energy(entry)-previous["material_energy"], entry["material_dissipated_energy"])
        energy_scale = sum(term.abs() for term in energy_terms).clamp_min(1.e-30)
        assert float(sum(energy_terms).abs() / energy_scale) < 1.e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ("debye", "drude", "lorentz", "magnetic"))
@pytest.mark.parametrize("one_side", (False, True))
def test_local_material_foil_public_graph_and_checkpoint_resume(kind, one_side, tmp_path):
    import witwin.maxwell as mw
    def simulation(graph):
        return mw.Simulation.fdtd(_local_scene(kind, one_side=one_side), frequency=3.e9,
            cuda_graph=graph, run_time=mw.TimeConfig(time_steps=128), spectral_sampler=mw.SpectralSampler(window="none"),
            excitations=mw.PortExcitation("feed", source_time=mw.CW(3.e9)))
    eager = simulation(False).run()
    graph = simulation(True).run()
    assert graph.stats()["cuda_graph_active"]
    torch.testing.assert_close(graph.port("feed").voltage, eager.port("feed").voltage, rtol=2.e-5, atol=1.e-8)
    assert eager.conductor_losses["foil"]["dissipated_energy"] > 0
    for entry in eager.solver._copper_foils:
        if kind == "magnetic":
            continue
        assert entry["material_ade"]
        assert torch.any(entry["material_ade"][0]["current"] != 0)
        if one_side:
            assert torch.all(entry["material_ade"][0]["drive"][0] == 0)
            assert torch.all(entry["material_ade"][0]["current"][0] == 0)
        templates = eager.solver._dispersive_templates[entry["electric_name"]]
        for pole in templates[kind]:
            assert torch.all(pole["drive"][entry["index"]] == 0)
            assert torch.all(pole["current"][entry["index"]] == 0)
    checkpoint = simulation(True).prepare().run_until(47)
    path = tmp_path / "local-material.pt"
    checkpoint.save(path)
    resumed = simulation(True).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    torch.testing.assert_close(resumed.port("feed").voltage, graph.port("feed").voltage, rtol=2.e-5, atol=1.e-8)
    assert resumed.conductor_losses["foil"]["dissipated_energy"] == pytest.approx(
        graph.conductor_losses["foil"]["dissipated_energy"], rel=1.e-6)
