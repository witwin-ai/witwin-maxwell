"""Continuum plane-wave convergence using a source-free native Yee evolution."""
import math

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("epsilon,mu", [(1., 1.), (0.25, 0.5)])
def test_material_wave_speed_has_second_order_grid_convergence(epsilon, mu):
    errors = []
    for cells in (20, 40, 80):
        dx = 1 / cells
        scene = mw.Scene(domain=mw.Domain(bounds=((0., 1.), (0., 0.2), (0., 0.2))),
            grid=mw.GridSpec.uniform(dx), boundary=mw.BoundarySpec.periodic(), device="cuda",
            structures=(mw.Structure(geometry=mw.Box(position=(0.5, 0.1, 0.1), size=(2., 1., 1.)),
                material=mw.Material(eps_r=epsilon, mu_r=mu)),),
            monitors=(mw.FieldTimeMonitor("trace", components=("Ez",), position=(0.5, 0.1, 0.1)),))
        velocity = 299792458. / math.sqrt(epsilon * mu)
        simulation = mw.Simulation.fdtd(scene, frequency=velocity,
            run_time=mw.TimeConfig(time_steps=1), cuda_graph=True)
        prepared = simulation.prepare()
        solver = prepared.solver
        steps = round(2 / velocity / solver.dt)
        simulation.config.run_time = mw.TimeConfig(time_steps=steps, dt=solver.dt)
        x = torch.as_tensor(solver.scene.x_nodes64, device="cuda", dtype=solver.Ex.dtype)
        xh = (x[:-1] + x[1:]) / 2
        impedance = math.sqrt(solver.mu0 * mu / (solver.eps0 * epsilon))
        solver.Ez.copy_(torch.cos(2 * math.pi * x)[:, None, None].expand_as(solver.Ez))
        solver.Hy.copy_((-torch.cos(2 * math.pi * (xh + velocity * solver.dt / 2)) / impedance)[:, None, None].expand_as(solver.Hy))
        result = prepared.run()
        time = steps * solver.dt
        expected = torch.cos(2 * math.pi * (x - velocity * time))
        actual = result.solver.Ez[:, 1, 1]
        error = float(torch.linalg.vector_norm(actual[:-1] - expected[:-1]) / torch.linalg.vector_norm(expected[:-1]))
        errors.append(error)
        # Bound the homogeneous wave energy after two transit times, excluding
        # the duplicated periodic endpoint from the electric quadrature.
        energy = float(actual[:-1].square().mean() + (impedance * solver.Hy[:, 1, 1]).square().mean())
        assert abs(energy - 1) < 0.025
    assert errors[0] < 0.1
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3
