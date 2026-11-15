"""C2 GPU gates: extraction quantities are differentiable through the GR1 adjoint.

- complex128 gradcheck of the complete re-solving map Zs_groups -> Y -> (L, Q, R)
  (the GR1 native VJP composed with the PyTorch extraction functions);
- Scene-level d(quantity)/d(sigma) through Result.network against central
  differences of fresh solves, inside the GR1 truncation-plus-solve-noise bound
  propagated through the extraction Jacobian.
"""

import dataclasses
import gc

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import extraction as ex
from witwin.maxwell.mom.layered.adjoint import network_admittance
from witwin.maxwell.mom.layered.lowfreq import scaled_system
from witwin.maxwell.mom.layered.network import solve_prepared_network
from witwin.maxwell.mom.layered.operator import BACKWARD
from witwin.maxwell.network import NetworkData
from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, U, loop_config, loop_scene

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def budget():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    yield
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def quantities(network):
    """One-port (p2 shorted), floating differential and pi-series quantities."""
    return torch.cat((ex.quality_factor(network, "p1", others="short"),
                      ex.inductance(network, ("p1", "p2"), others="open") * 1e9,
                      ex.resistance(network, ("p1", "p2"), others="open"),
                      ex.pi_model(network, "p1", "p2").series_resistance))


def test_gradcheck_through_gr1():
    sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    runtime = mw.Simulation.mom(loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5), frequencies=FREQUENCIES,
                                config=loop_config("sheet", max_edge=32 * U)).prepare().solver
    prepared = runtime.prepared
    adjoint, impedance = runtime.gradients.loss_adjoint(prepared)
    groups = runtime.gradients.face_groups(prepared)
    base = impedance.detach()
    frequencies = prepared.table.frequencies

    def extracted(u):
        values = base * u
        padded = torch.cat((values.new_zeros(len(FREQUENCIES), 1), values.detach()), 1)
        loss = dataclasses.replace(prepared.loss, impedance=padded[:, groups + 1].contiguous())
        solved, currents = solve_prepared_network(dataclasses.replace(prepared, loss=loss))
        y = network_admittance(values, solved.to_y(), currents, adjoint)
        network = NetworkData.from_y(frequencies=frequencies, y=y, z0=solved.z0, port_names=solved.port_names)
        return quantities(network)

    u = torch.ones_like(base, requires_grad=True)
    assert torch.autograd.gradcheck(extracted, (u,), eps=1e-5, atol=1e-6, rtol=1e-3)


def run(sigma):
    return mw.Simulation.mom(loop_scene(sheet_sigma=sigma), frequencies=FREQUENCIES,
                             config=loop_config("sheet")).run()


def test_scene_gradient_against_central_differences(record_property):
    sigma = torch.tensor(SIGMA, dtype=torch.float64, device="cuda", requires_grad=True)
    result = run(sigma)
    values = quantities(result.network)
    gradient = torch.stack([torch.autograd.grad(v, sigma, retain_graph=True)[0] for v in values])
    y = result.network.to_y().detach()
    kappa = torch.linalg.cond(scaled_system(result.solver.prepared).matrix).real
    # Extraction Jacobian |dq/dY| (complex entries) at the forward point.
    def from_y(parts):
        matrix = torch.view_as_complex(parts)
        return quantities(NetworkData.from_y(frequencies=result.network.frequencies, y=matrix,
                                             z0=result.network.z0, port_names=result.network.port_names))

    # Real/imaginary parts each perturbed by at most |dY|: weight both partials.
    jacobian = torch.autograd.functional.jacobian(from_y, torch.view_as_real(y).contiguous())
    sensitivity = jacobian.abs().sum(-1).flatten(1)
    h = 1e-3

    def fd(step):
        with torch.no_grad():
            plus, minus = run(SIGMA * (1 + step)), run(SIGMA * (1 - step))
        return (quantities(plus.network) - quantities(minus.network)) / (2 * step * SIGMA)

    first, second = fd(h), fd(2 * h)
    truncation = (second - first).abs() / 3
    # Each solve meets backward error BACKWARD: |dY| <= 2 kappa BACKWARD max|Y| per frequency.
    noise_y = (2 * kappa * BACKWARD * y.abs().amax((-1, -2)))[:, None, None].expand_as(y).flatten()
    noise = 2 * (sensitivity @ noise_y) / (2 * h * SIGMA)
    bound = 2 * (truncation + noise)
    error = (gradient - first).abs()
    record_property("relative_error", (error / gradient.abs()).max().item())
    record_property("relative_bound", (bound / gradient.abs()).max().item())
    assert bool((error <= bound).all())
    assert (bound / gradient.abs()).max().item() <= 1e-2
