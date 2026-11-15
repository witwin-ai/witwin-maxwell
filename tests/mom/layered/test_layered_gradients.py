"""GR1 gates: native adjoint network gradients of layered local-loss parameters.

dY/dtheta = -X^T (dZ/dtheta) X reuses the forward currents X. Gates:
- the native loss_adjoint contraction against a dense torch contraction of
  the same local Gram rows (SheetLoss.add_to; fp64 setup bound) and bitwise repeats;
- torch.autograd.gradcheck in complex128 on the complete re-solving map
  Zs_groups -> Y (conjugate Wirtinger convention of complex inputs);
- central finite differences through fresh Scene -> Simulation -> Result runs
  on the same mesh, inside a derived bound (see fd_bound);
- gradient repeat determinism (torch.equal, three fresh runs);
- a SceneModule + torch.optim loop verified by an independent forward run.
"""

import dataclasses
import gc
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.adjoint import network_admittance
from witwin.maxwell.mom.layered.lowfreq import scaled_system
from witwin.maxwell.mom.layered.network import solve_prepared_network
from witwin.maxwell.mom.layered.operator import BACKWARD
from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, U, loop_config, loop_scene

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]

NOMINAL = {"sheet_sigma": SIGMA, "via_sigma": SIGMA, "thickness": 2 * U,
           "rms_height": .5e-6, "sphere_radius": .5e-6}


@pytest.fixture(autouse=True)
def budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    record_property("seconds", seconds)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def simulation(model, **parameters):
    return mw.Simulation.mom(loop_scene(**parameters), frequencies=FREQUENCIES, config=loop_config(model))


def derivative(y, parameter):
    """Complex dY/dtheta of every entry for a real scalar theta (2*K*P*P reverse passes)."""
    flat = y.reshape(-1)
    parts = []
    for value in flat:
        real, = torch.autograd.grad(value.real, parameter, retain_graph=True)
        imag, = torch.autograd.grad(value.imag, parameter, retain_graph=True)
        assert not real.is_complex() and not imag.is_complex()
        parts.append(torch.complex(real, imag))
    return torch.stack(parts).reshape(y.shape)


def admittance(model, name, scale):
    with torch.no_grad():
        result = simulation(model, **{name: scale * NOMINAL[name]}).run()
    return result.network.to_y()


def fd_bound(model, name, gradient, kappa, y, h):
    """Central-difference bound on the same mesh.

    FD_h = g + c h^2 + O(h^4); with FD_2h the Richardson truncation estimate is
    T = |FD_2h - FD_h| / 3. Each solve meets the normwise backward-error gate
    beta = 64*2^-53, so |dY| <= 2 kappa beta max|Y| per frequency and the FD
    noise is N = 2 kappa beta max|Y| / h. The bound 2 (T + N) also absorbs the
    O(h^4) remainder and the noise of FD_2h inside T.
    """
    plus, minus = admittance(model, name, 1 + h), admittance(model, name, 1 - h)
    plus2, minus2 = admittance(model, name, 1 + 2 * h), admittance(model, name, 1 - 2 * h)
    fd = (plus - minus) / (2 * h)
    fd2 = (plus2 - minus2) / (4 * h)
    truncation = (fd2 - fd).abs() / 3
    noise = (2 * kappa * BACKWARD * y.abs().amax((-1, -2)) / h)[:, None, None]
    bound = 2 * (truncation + noise)
    return fd, bound, truncation, noise


def condition(result):
    # The in-place dense chunk system holds the complete equilibrated matrix.
    matrix = scaled_system(result.solver.prepared).matrix
    return torch.linalg.cond(matrix).real


CASES = [(model, name) for model in ("sheet", "closed") for name in NOMINAL
         if not (model == "closed" and name == "thickness")]


@pytest.mark.parametrize("model,name", CASES)
def test_finite_difference_agreement(model, name, record_property):
    scale = torch.tensor(1., dtype=torch.float64, device="cuda", requires_grad=True)
    result = simulation(model, **{name: scale * NOMINAL[name]}).run()
    y = result.network.to_y()
    gradient = derivative(y, scale)
    kappa = condition(result)
    h = 1e-3
    fd, bound, truncation, noise = fd_bound(model, name, gradient, kappa, y.detach(), h)
    error = (gradient - fd).abs()
    scale_g = gradient.abs().max()
    record_property("max_abs_gradient", scale_g.item())
    record_property("max_relative_error", (error.max() / scale_g).item())
    record_property("max_relative_bound", (bound.max() / scale_g).item())
    record_property("truncation_relative", (truncation.max() / scale_g).item())
    record_property("noise_relative", (noise.max() / scale_g).item())
    record_property("condition", kappa.tolist())
    assert bool((error <= bound).all())
    # The bound must be informative: far below the derivative itself.
    assert bound.max() <= 1e-3 * scale_g


@pytest.mark.parametrize("model", ("sheet", "closed"))
def test_gradient_repeat_determinism(model):
    first = None
    for _ in range(3):
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        rough = torch.tensor(.5e-6, dtype=torch.float64, requires_grad=True)
        result = simulation(model, sheet_sigma=sigma, via_sigma=sigma * .5, rms_height=rough).run()
        loss = result.network.s.abs().square().sum() + result.network.to_z().imag.sum()
        grads = torch.autograd.grad(loss, (sigma, rough))
        assert all(g.dtype == torch.float64 for g in grads)
        if first is None:
            first = grads
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, grads))
        del result
        gc.collect()


@pytest.mark.parametrize("model", ("sheet", "closed"))
def test_gradcheck_complex128_and_impedance_map(model):
    sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    runtime = mw.Simulation.mom(loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5,
                                           rms_height=torch.tensor(.5e-6, requires_grad=True)),
                                frequencies=FREQUENCIES,
                                config=loop_config(model, max_edge=32 * U)).prepare().solver
    prepared = runtime.prepared
    adjoint, impedance = runtime.gradients.loss_adjoint(prepared)
    groups = runtime.gradients.face_groups(prepared)
    # The PyTorch-native impedance map reproduces the forward face impedance.
    padded = torch.cat((impedance.new_zeros(len(FREQUENCIES), 1), impedance.detach()), 1)
    faces = padded[:, groups + 1]
    reference = prepared.loss.impedance
    assert torch.equal(faces == 0, reference == 0)
    assert ((faces - reference).abs().max() <= 1e-14 * reference.abs().max()).item()
    base = impedance.detach()
    y0 = solve_prepared_network(prepared)[0].to_y()
    scale = y0.abs().max()

    def network(u):
        values = base * u
        padded = torch.cat((values.new_zeros(len(FREQUENCIES), 1), values.detach()), 1)
        loss = dataclasses.replace(prepared.loss, impedance=padded[:, groups + 1].contiguous())
        solved, currents = solve_prepared_network(dataclasses.replace(prepared, loss=loss))
        return network_admittance(values, solved.to_y(), currents, adjoint) / scale

    u = torch.ones_like(base, requires_grad=True)
    assert torch.autograd.gradcheck(network, (u,), eps=1e-5, atol=1e-6, rtol=1e-3)


def test_loss_adjoint_kernel_matches_dense_contraction():
    sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    runtime = mw.Simulation.mom(loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5), frequencies=FREQUENCIES,
                                config=loop_config("sheet")).prepare().solver
    prepared, gradients = runtime.prepared, runtime.gradients
    adjoint, impedance = gradients.loss_adjoint(prepared)
    groups = gradients.face_groups(prepared)
    loss = prepared.loss
    generator = torch.Generator().manual_seed(7)
    K, N, P, G = len(FREQUENCIES), loss.blocks.shape[0], 2, impedance.shape[1]
    currents = torch.randn((K, N, P), generator=generator, dtype=torch.complex128).cuda()
    upstream = torch.randn((K, P, P), generator=generator, dtype=torch.complex128).cuda()
    op = torch.ops.witwin_maxwell_mom_layered
    native = [op.loss_adjoint(currents, upstream, adjoint.blocks, adjoint.columns, adjoint.rows,
                              adjoint.row_offsets, adjoint.faces, adjoint.face_offsets) for _ in range(3)]
    assert all(torch.equal(native[0], value) for value in native[1:])
    reference = torch.empty((K, G), device="cuda", dtype=torch.complex128)
    for g in range(G):
        indicator = (groups == g).to(torch.complex128).expand(K, -1).contiguous()
        # Dense dZ/dZs_g = sum_{f in g} G_f from the torch local-Gram writes of SheetLoss.
        unit = dataclasses.replace(loss, impedance=indicator)
        derivative_matrix = torch.zeros((K, N, N), device="cuda", dtype=torch.complex128)
        for k in range(K):
            unit.add_to(derivative_matrix[k], k)
        dy = -currents.transpose(-1, -2) @ derivative_matrix @ currents
        reference[:, g] = (dy.conj() * upstream).sum((-1, -2))
    assert ((native[0] - reference).abs().max() <= 1e-12 * reference.abs().max()).item()


def test_reference_impedance_and_port_load_gradients(record_property):
    """z0 and port loads act on Result.network through the PyTorch NetworkData algebra.

    A trainable LumpedPort.reference_impedance is refused by the shared Simulation
    admission for non-FDTD methods; renormalize(z0) is the supported route.
    """
    sigma = torch.tensor(SIGMA, dtype=torch.float64, device="cuda", requires_grad=True)
    network = simulation("sheet", sheet_sigma=sigma).run().network
    z0 = torch.tensor(50., dtype=torch.float64, device="cuda", requires_grad=True)
    load = torch.tensor(20., dtype=torch.float64, device="cuda", requires_grad=True)

    def objective(network, reference, resistance):
        renormalized = network.renormalize(torch.stack((reference, reference.new_tensor(50.))))
        return (renormalized.s[:, 1, 0].abs().sum()
                + renormalized.terminate("p2", impedance=resistance).s.abs().sum())

    gradient = torch.autograd.grad(objective(network, z0, load), (z0, load, sigma))
    fixed = network.s.detach()
    detached = type(network)(frequencies=network.frequencies, s=fixed, z0=network.z0,
                             port_names=network.port_names)
    h = 1e-4

    def value(reference, resistance):
        return objective(detached, torch.tensor(reference, dtype=torch.float64, device="cuda"),
                         torch.tensor(resistance, dtype=torch.float64, device="cuda")).item()

    fd = ((value(50 * (1 + h), 20.) - value(50 * (1 - h), 20.)) / (100 * h),
          (value(50., 20 * (1 + h)) - value(50., 20 * (1 - h))) / (40 * h))
    record_property("gradients", [g.item() for g in gradient])
    record_property("fd", list(fd))
    assert gradient[2].dtype == torch.float64 and gradient[2].item() != 0
    for g, f in zip(gradient, fd):
        # Analytic S algebra of a fixed network: O(h^2) truncation with unit-scale
        # curvature (~1e-8 relative) plus fp64 rounding / h (~1e-12); no solve noise.
        assert abs(g.item() - f) <= 1e-6 * abs(g.item()) + 1e-12
    port = loop_scene().ports[0]
    trainable = mw.LumpedPort(port.name, positive=port.positive, negative=port.negative,
                              voltage_path=port.voltage_path, reference_impedance=z0)
    scene = loop_scene()
    scene = scene.updated(ports=(trainable, *scene.ports[1:]))
    with pytest.raises(NotImplementedError, match="require Simulation.fdtd"):
        mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config())


def test_scene_module_optimizer_raises_q(record_property):
    class Coil(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.log_sigma = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64, device="cuda"))

        def to_scene(self):
            return loop_scene(sheet_sigma=1e7 * torch.exp(self.log_sigma))

    def quality(network):
        zin = 1 / network.to_y()[-1, 0, 0]
        return -zin.imag / zin.real

    module = Coil()
    optimizer = torch.optim.Adam(module.parameters(), lr=.3)
    sim = mw.Simulation.mom(module, frequencies=FREQUENCIES, config=loop_config("sheet"))
    history = []
    for _ in range(3):
        optimizer.zero_grad()
        q = quality(sim.run().network)
        (-q).backward()
        history.append(q.item())
        optimizer.step()
    with torch.no_grad():
        final = quality(mw.Simulation.mom(loop_scene(sheet_sigma=1e7 * torch.exp(module.log_sigma.detach())),
                                          frequencies=FREQUENCIES, config=loop_config("sheet")).run().network)
    record_property("Q_history", history + [final.item()])
    assert history[0] < history[1] < history[2] < final.item()


def test_gradient_refusals_and_forward_equality():
    trainable = torch.tensor(2 * U, dtype=torch.float64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="geometry derivative"):
        simulation("closed", thickness=trainable).prepare()
    scene = loop_scene()
    scene.structures[-1].geometry.position.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="geometry gradient"):
        mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config()).prepare()
    scene = loop_scene(z0=(50.0, torch.tensor(50., requires_grad=True)))
    with pytest.raises(NotImplementedError, match="require Simulation.fdtd"):
        mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config())
    # Any termination, trainable or fixed, is refused by the mom.layered capability table.
    scene = loop_scene()
    object.__setattr__(scene.ports[1], "termination", mw.rf.SeriesRLC(r=torch.tensor(5., requires_grad=True)))
    with pytest.raises(mw.UnsupportedFeatureError, match="does not apply a termination"):
        mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config())
    with pytest.raises(NotImplementedError, match=r"Trainable Layer\.eps_r"):
        mw.mom.Layer("air", 1e-3, torch.tensor(1., requires_grad=True))
    # A fixed stack tensor made trainable after construction is still refused at preparation.
    eps_r = torch.tensor(1.)
    scene = loop_scene(sheet_sigma=torch.tensor(SIGMA, requires_grad=True))
    scene = scene.updated(background=mw.mom.LayerStack((mw.mom.Layer("air", 1e-3, eps_r),), mw.mom.HalfSpace(), mw.mom.HalfSpace()))
    eps_r.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="process-stack"):
        mw.Simulation.mom(scene, frequencies=FREQUENCIES, config=loop_config()).prepare()
    sweep = mw.mom.MoMConfig(layered=dataclasses.replace(
        loop_config().layered, sweep=mw.mom.AdaptiveSweep(1e9, 1e10)))
    with pytest.raises(NotImplementedError, match="sweep"):
        mw.Simulation.mom(loop_scene(sheet_sigma=torch.tensor(SIGMA, requires_grad=True)),
                          config=sweep).prepare()
    # Gradient attachment keeps the forward currents bitwise and S to rounding.
    with torch.no_grad():
        fixed = simulation("sheet").run()
    attached = simulation("sheet", sheet_sigma=torch.tensor(SIGMA, dtype=torch.float64,
                                                            requires_grad=True)).run()
    assert attached.network.s.requires_grad and not fixed.network.s.requires_grad
    assert torch.equal(mw.mom.surface_currents(fixed).currents, mw.mom.surface_currents(attached).currents)
    assert ((fixed.network.s - attached.network.s.detach()).abs().max() <= 1e-13).item()
