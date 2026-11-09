"""Conductive field/circuit updates against an independent midpoint block solve."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.lumped import apply_lumped_runtime, prepare_lumped_runtime
from witwin.maxwell.fdtd.ports import _validate_local_update_coefficient


def _block_reference(e, eps, volume, sigma, weights, dt, r, l, c, il, vc, drive):
    """Solve all edge fields and branch current simultaneously, without a Norton reduction."""
    mass = eps * volume
    conductance = sigma * volume
    n = e.numel()
    matrix = e.new_zeros((n + 1, n + 1))
    matrix[:n, :n] = torch.diag(mass / dt + conductance / 2)
    matrix[:n, n] = weights
    matrix[n, :n] = -weights / 2
    matrix[n, n] = r + 2 * l / dt + (dt / (2 * c) if c else 0)
    rhs = torch.cat(((mass / dt - conductance / 2) * e,
                     (weights.dot(e) / 2 - drive - vc + 2 * l / dt * il).reshape(1)))
    solution = torch.linalg.solve(matrix, rhs)
    current = solution[n]
    return solution[:n], current, 2 * current - il if l else il * 0, vc + dt / c * current if c else vc * 0


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("sigma_ratio", (0.0, 1.e-5, 0.1, 2.0, 20.0, 1000.0))
@pytest.mark.parametrize("components", ((0.7, 0., 0.), (0., .2, 0.), (0., 0., .3), (.7, .2, .3)))
def test_lossy_midpoint_matches_independent_block_and_energy(device, sigma_ratio, components):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for device validation")
    dtype = torch.float64
    eps = torch.tensor([1.2, 2.1, 3.4], device=device, dtype=dtype)
    volume = torch.tensor([.5, .2, .8], device=device, dtype=dtype)
    weights = torch.tensor([.4, -.2, .3], device=device, dtype=dtype)
    dt = .15
    sigma = eps / dt * sigma_ratio * torch.tensor([1., .4, .7], device=device, dtype=dtype)
    coefficient = dt / (eps + sigma * dt / 2)
    geometry = SimpleNamespace(port_name="test", voltage_weights=weights,
                               voltage_indices=torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], device=device))
    r, l, c = components
    runtime = prepare_lumped_runtime(geometry, dt=dt, eps_edge=eps.reshape(-1, 1, 1),
                                    yee_control_volume=volume.reshape(-1, 1, 1),
                                    electric_update_coefficient=coefficient.reshape(-1, 1, 1),
                                    termination=mw.SeriesRLC(r=eps.new_tensor(r), l=eps.new_tensor(l) if l else None,
                                                             c=eps.new_tensor(c) if c else None), diagnostics=True)
    e = torch.tensor([.8, -.3, .2], device=device, dtype=dtype)
    il = e.new_zeros(())
    vc = e.new_zeros(())
    runtime.last_voltage_after.copy_(weights.dot(e))
    initial_energy = .5 * torch.sum(eps * volume * e.square())
    cumulative_residual = initial_energy.new_zeros(())
    for step in range(32):
        drive = .3 * torch.cos(e.new_tensor(step * .2)) if step < 8 else e.new_zeros(())
        expected_e, expected_i, next_il, next_vc = _block_reference(
            e, eps, volume, sigma, weights, dt, r, l, c, il, vc, drive)
        free = ((eps - sigma * dt / 2) / (eps + sigma * dt / 2) * e).reshape(-1, 1, 1)
        current = apply_lumped_runtime(runtime, free, thevenin_voltage=drive)
        torch.testing.assert_close(free.flatten(), expected_e, rtol=1.e-11, atol=1.e-12)
        torch.testing.assert_close(current, expected_i, rtol=1.e-11, atol=1.e-12)
        midpoint = (e + free.flatten()) / 2
        conductive_loss = dt * torch.sum(sigma * volume * midpoint.square())
        field_change = .5 * torch.sum(eps * volume * (free.flatten().square() - e.square()))
        circuit_change = .5 * l * (next_il.square() - il.square()) + .5 * c * (next_vc.square() - vc.square())
        residual = field_change + circuit_change + conductive_loss + dt * r * current.square() + dt * drive * current
        cumulative_residual += residual
        torch.testing.assert_close(residual, residual.new_zeros(()), rtol=0, atol=2.e-13)
        torch.testing.assert_close(runtime.last_voltage_midpoint, weights.dot(midpoint), rtol=1.e-12, atol=1.e-13)
        e, il, vc = free.flatten().clone(), next_il, next_vc
    assert abs(float(cumulative_residual)) / float(initial_energy) < 1.e-10


def test_local_response_validation_accepts_loss_and_rejects_extra_suppression():
    eps = torch.full((2, 1, 1), 2., dtype=torch.float64)
    sigma = torch.full_like(eps, 3.)
    coefficient = .2 / (eps + .1 * sigma)
    runtime = SimpleNamespace(dt=.2, linear_indices=torch.tensor([0, 1]), port_name="lossy")
    solver = SimpleNamespace(eps_Ex=eps, sigma_e_Ex=sigma, cex_curl=coefficient)
    _validate_local_update_coefficient(solver, runtime, "Ex")
    solver.cex_curl = coefficient * .9
    with pytest.raises(ValueError, match="static conductive material response"):
        _validate_local_update_coefficient(solver, runtime, "Ex")


def _scene(*, sigma, remote=False, circuit=False, driven=False):
    port = mw.LumpedPort(name="feed", positive=(0., 0., .005), negative=(0., 0., -.005),
                         voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                             position=(0., 0., -.0025), size=(.015, .015, 0.)),
                         termination=None if circuit or driven else mw.SeriesRLC(r=25., l=.5e-9, c=1.e-12))
    circuits = ()
    if circuit:
        network = mw.Circuit("load")
        a, b, c = network.node("a"), network.node("b"), network.node("c")
        network.add(mw.Resistor("r", a, b, 25.))
        network.add(mw.Inductor("l", b, c, .5e-9))
        network.add(mw.Capacitor("c", c, network.ground, 1.e-12))
        network.bind_port("feed", positive=a, negative=network.ground)
        circuits = (network,)
    box = mw.Box(position=(.0125, 0., 0.) if remote else (0., 0., 0.),
                 size=(.005, .02, .02) if remote else (.03, .03, .03))
    return mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
                    boundary=mw.BoundarySpec.none(), ports=(port,), circuits=circuits,
                    structures=(mw.Structure(geometry=box, material=mw.Material(eps_r=2., sigma_e=sigma)),), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("remote", (False, True))
@pytest.mark.parametrize("sigma", (.01, 2., 100.))
def test_lossy_native_series_and_mna_agree_through_actual_gpu_field_steps(remote, sigma):
    from witwin.maxwell.fdtd.circuits import prepare_circuit_time_series
    from witwin.maxwell.fdtd.ports import apply_port_runtimes
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block
    solvers = [mw.Simulation.fdtd(_scene(sigma=sigma, remote=remote, circuit=mna),
                                 frequency=3.e9).prepare().solver for mna in (False, True)]
    native, coupled = solvers
    prepare_circuit_time_series(coupled, 64)
    for solver in solvers:
        solver.Ez[4, 4, 3:5] = 1.
        runtime = solver._port_runtimes[0]
        state = runtime.lumped if runtime.lumped is not None else runtime.circuit_port.field
        state.last_voltage_after.copy_(torch.dot(state.voltage_weights,
                                                solver.Ez.flatten().index_select(0, state.linear_indices)))
    transferred = 0.
    for step in range(64):
        for solver in solvers:
            _field_update_block(solver, step * solver.dt)
            apply_port_runtimes(solver)
        # Compare against the fixed unit initial field scale, including zero crossings.
        assert float(torch.max(torch.abs(native.Ez - coupled.Ez))) < 1.e-5
        torch.testing.assert_close(native._port_runtimes[0].lumped.last_branch_current,
                                   coupled._port_runtimes[0].circuit_port.field.last_current,
                                   rtol=3.e-5, atol=1.e-10)
        transferred += abs(float(native._port_runtimes[0].lumped.last_branch_current))
    assert transferred > 1.e-7
    assert torch.isfinite(native.Ez).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("remote", (False, True))
def test_lossy_source_public_run_has_nonzero_voltage_current_and_power(remote):
    result = mw.Simulation.fdtd(
        _scene(sigma=.1, remote=remote, driven=True), frequencies=(2.5e9, 3.e9),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=1.e9)),
        run_time=mw.TimeConfig(time_steps=256), spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False).run()
    data = result.port("feed")
    assert torch.isfinite(data.voltage).all() and torch.isfinite(data.current).all()
    assert bool(torch.all(data.accepted_power > 0.))
