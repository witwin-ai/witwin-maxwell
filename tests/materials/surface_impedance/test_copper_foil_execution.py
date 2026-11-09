"""Load-bearing split traces, passive energy, graph and resume qualification."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.copper_foil import begin_copper_step, advance_copper_electric


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CopperFoil execution requires CUDA.")


def _simulation(*, graph=False, thickness=2e-6, steps=180, contact=False):
    port = mw.LumpedPort("feed", positive=(0., 0., .005), negative=(0., 0., -.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)))
    foil = mw.Structure(geometry=mw.Box(position=torch.tensor([.01, 0., 0.], dtype=torch.float64),
        size=torch.tensor([0., .02, .02], dtype=torch.float64)),
        material=mw.CopperFoil(thickness=thickness, frequency_range=(.5e9, 2e9)), name="foil")
    structures = [foil]
    if contact:
        structures.append(mw.Structure(geometry=mw.Box(position=torch.tensor([.01, 0., 0.], dtype=torch.float64),
            size=torch.tensor([.01, .01, .01], dtype=torch.float64)), material=mw.Material.pec(), name="via"))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(),
        ports=[port], structures=structures, device="cuda")
    return mw.Simulation.fdtd(scene, frequency=1e9,
        run_time=mw.TimeConfig(time_steps=steps), cuda_graph=graph,
        excitations=mw.PortExcitation("feed"), spectral_sampler=mw.SpectralSampler(window="none"))


def _assert_same_fields(actual, expected, *, rtol=1e-6, atol=2e-8):
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=rtol, atol=atol)


def test_eager_graph_and_disk_resume_preserve_both_traces_states_and_loss(tmp_path):
    expected = _simulation().run()
    graph = _simulation(graph=True).run()
    assert graph.solver._cuda_graph_active
    _assert_same_fields(graph.solver, expected.solver)
    checkpoint = _simulation().prepare().run_until(90)
    keys = [key for key in checkpoint.physics.tensors if key.startswith("copper_foil_")]
    assert keys
    for suffix in ("state", "right_e", "cumulative_loss", "spectral_real", "spectral_imag", "spectral_step"):
        assert any(key.endswith(suffix) for key in keys)
    assert any(float(checkpoint.physics.tensors[key].abs().max()) > 0 for key in keys if key.endswith("right_e"))
    path = tmp_path / "foil.pt"
    checkpoint.save(path)
    loaded = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    actual = _simulation().prepare().run(resume_from=loaded)
    _assert_same_fields(actual.solver, expected.solver)
    for a, b in zip(actual.solver._copper_foils, expected.solver._copper_foils):
        for name in ("state", "right_e", "cumulative_loss", "spectral_real", "spectral_imag", "spectral_norm"):
            torch.testing.assert_close(a[name], b[name], rtol=1e-6, atol=2e-10)
    assert actual.conductor_losses["foil"]["dissipated_energy"] > 0
    assert torch.all(actual.conductor_losses["foil"]["surface_power"] >= 0)
    saved = tmp_path / "result.pt"
    actual.save(saved)
    snapshot = mw.Result.load(saved, scene=actual.scene)
    assert snapshot.conductor_losses["foil"]["dissipated_energy"] > 0
    torch.testing.assert_close(snapshot.conductor_losses["foil"]["surface_power"],
                               actual.conductor_losses["foil"]["surface_power"])


def test_resume_rejects_changed_thickness_before_mutating_fields():
    checkpoint = _simulation().prepare().run_until(90)
    prepared = _simulation(thickness=2.01e-6).prepare()
    before = prepared.solver.Ez.clone()
    with pytest.raises(ValueError, match="layout|does not match"):
        prepared.run(resume_from=checkpoint)
    torch.testing.assert_close(prepared.solver.Ez, before, rtol=0, atol=0)


def test_dropping_the_right_trace_and_state_changes_the_resumed_solution():
    checkpoint = _simulation().prepare().run_until(90)
    expected = _simulation().prepare().run(resume_from=checkpoint)
    for key, value in checkpoint.physics.tensors.items():
        if key.startswith("copper_foil_") and (key.endswith("right_e") or key.endswith("_state")):
            value.zero_()
    actual = _simulation().prepare().run(resume_from=checkpoint)
    scale = float(expected.solver.Ez.abs().max())
    assert float((actual.solver.Ez - expected.solver.Ez).abs().max()) / scale > 1e-7


def test_explicit_via_contact_retains_the_remaining_foil_state():
    prepared = _simulation(contact=True).prepare()
    assert prepared.solver._copper_foils
    assert sum(entry["pec_contact_edge_count"] for entry in prepared.solver._copper_foils) > 0
    result = prepared.run()
    assert result.conductor_losses["foil"]["dissipated_energy"] > 0
    assert torch.isfinite(result.solver.Ez).all()


def test_local_midpoint_step_closes_capacitor_and_foil_energy():
    solver = _simulation().prepare().solver
    generator = torch.Generator(device="cuda").manual_seed(173)
    for entry in solver._copper_foils:
        entry["state"].normal_(generator=generator)
        entry["state"].mul_(1e-7)
        entry["right_e"].normal_(generator=generator)
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).normal_(generator=generator)
    # With zero exterior H the local system exchanges energy only between the
    # two exterior half-cell capacitors and the foil's diffusion states.
    def energy():
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        for entry in solver._copper_foils:
            electric = torch.stack((getattr(solver, entry["electric_name"])[entry["index"]], entry["right_e"])).double()
            density = .5 * (entry["capacity"] * electric.square()).sum(0)
            density += .5 * entry["state"].square().sum(0)
            total += (density * entry["area"]).sum()
        return total
    initial = energy()
    for _ in range(20):
        begin_copper_step(solver)
        advance_copper_electric(solver)
    final = energy()
    heat = sum(entry["cumulative_loss"] for entry in solver._copper_foils)
    residual = abs(float(final + heat - initial)) / float(initial)
    assert residual < 1e-5
    assert float(heat) > 0


def test_compressed_foil_coupled_field_step_has_no_growing_eigenmode():
    """Compare old/new coupled E/H/copper operators in FP64 at the final dt.

    For split-trace curl K, field masses Me/Mh and modal mass W = area/kappa,
    the leapfrog/midpoint update is (Q-dt*A/2)u+ = (Q+dt*A/2)u, where
    Q = [[Me, -dt*K.T/2, 0], [-dt*K/2, Mh, 0], [0, 0, W]].
    Its symmetric dissipation is diag(S.T*W*D*S, 0, W*p) >= 0.
    Thus the coupled modified energy contracts when Q > 0 at the field CFL;
    The existing Torch curl kernels run on CUDA here because native PEC field
    kernels are FP32-only. Coefficients are re-derived in FP64, and the copper
    coupling is the ordinary runtime, including its split magnetic correction.
    """
    from scipy.linalg import eigvals
    from witwin.maxwell.copper_foil import EPSILON_0, MU_0
    from witwin.maxwell.fdtd.cpu_reference import get_cpu_reference_module
    from witwin.maxwell.fdtd.copper_foil import initialize_copper_foils
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, enforce_pec_boundaries

    reports = []
    for mode_count in (256, None):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.01, .01),) * 3),
            grid=mw.GridSpec.uniform(.01), boundary=mw.BoundarySpec.pec(), device="cuda",
            structures=[mw.Structure(geometry=mw.Box(position=torch.tensor([0., .005, .005], dtype=torch.float64),
                size=torch.tensor([0., .01, .01], dtype=torch.float64)), name="foil",
                material=mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 10e9), mode_count=mode_count))])
        solver = mw.Simulation.fdtd(scene, frequency=10e9,
            run_time=mw.TimeConfig(time_steps=1)).prepare().solver
        for name, value in list(vars(solver).items()):
            if torch.is_tensor(value) and value.dtype == torch.float32:
                setattr(solver, name, value.double())
        for name in ("Ex", "Ey", "Ez"):
            getattr(solver, "eps_" + name).fill_(EPSILON_0)
        for name in ("Hx", "Hy", "Hz"):
            getattr(solver, "mu_" + name).fill_(MU_0)
        for axis in "xyz":
            for kind, spacing in (("e", "dual"), ("h", "primal")):
                setattr(solver, f"inv_d{axis}_{kind}", torch.as_tensor(
                    1 / getattr(solver.scene, f"d{axis}_{spacing}64"), dtype=torch.float64, device="cuda"))
        solver.fdtd_module = get_cpu_reference_module()
        solver._build_update_coefficients()
        initialize_copper_foils(solver)
        fields = [getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        tensors = fields + [tensor for entry in solver._copper_foils
                           for tensor in (entry["right_e"], entry["state"])]
        assert all(value.dtype == torch.float64 and value.is_cuda for value in tensors)
        scales = [1.] * 3 + [1 / 376.730313668] * 3
        scales += [scale for _ in solver._copper_foils for scale in (1., math.sqrt(EPSILON_0 * .005))]
        size = sum(tensor.numel() for tensor in tensors)
        operator = torch.empty((size, size), dtype=torch.float64, device="cuda")

        def step():
            _field_update_block(solver, 0.)
            enforce_pec_boundaries(solver)

        runner = CudaGraphRunner(device=solver.device)
        replay = runner.capture(step)
        column = 0
        for tensor, scale in zip(tensors, scales):
            for index in range(tensor.numel()):
                for value in tensors:
                    value.zero_()
                tensor.view(-1)[index] = scale
                replay()
                operator[:, column] = torch.cat([value.reshape(-1) / factor
                                                 for value, factor in zip(tensors, scales)])
                column += 1
        radius = float(abs(eigvals(operator.cpu().numpy())).max())
        report = dict(mode_count=mode_count, coupled_state_count=size, dt_s=solver.dt, spectral_radius=radius)
        reports.append(report)
        print(report, flush=True)
        assert radius <= 1 + 2e-12
    assert reports[0]["dt_s"] == reports[1]["dt_s"]


@pytest.mark.parametrize("thickness", [2e-8, 60e-6])
def test_one_hundred_thousand_passive_full_field_steps_remain_bounded(thickness):
    # Batch tier: 1e5 steps with the same assertions; the 1e6-step horizon is the
    # milestone-tier test_copper_foil_long_horizon.py (launch-bound, contention-fragile).
    from tests.support.copper_long_horizon import run_passive_full_field_steps
    run_passive_full_field_steps(thickness, replays=1_000)
