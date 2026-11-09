"""Tensor-current material derivatives when the source overlaps a design interface."""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _loss(value, kind, stride, *, trainable=False, nonuniform=False):
    parameter = torch.tensor(value, dtype=torch.float64, device="cuda", requires_grad=trainable)
    direction = torch.tensor([.3, -.7, .5, .2, -.4, .8, -.2, .6], device="cuda").reshape(2, 2, 2)
    nodes = np.array([-.6, -.48, -.36, -.25, -.13, 0., .09, .21, .36, .48, .6])
    grid = mw.GridSpec.custom(x_coords=nodes, y_coords=nodes, z_coords=nodes) if nonuniform else mw.GridSpec.uniform(.12)
    scene = mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.6, .6),) * 3), grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=2), device="cuda")
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.,) * 3, size=(.36, .36, .12)),
        material=mw.Material(epsilon_tensor=mw.Tensor3x3(((5., 1.5, .8), (1.5, 4., .7), (.8, .7, 3.))))))
    scene.add_material_region(mw.MaterialRegion(name="interface",
        # Its lower z face passes through the tensor's central node, retaining
        # a fractional tensor/design blend and a density-dependent cross inverse.
        geometry=mw.Box(position=(.03, .03, .12), size=(.24,) * 3),
        density=torch.sigmoid(.3 + parameter * direction), eps_bounds=(1., 6.)))
    waveform = mw.GaussianPulse(frequency=1e9, fwidth=.25e9, amplitude=50.)
    if kind in {"point", "point_cross"}:
        position = (0., 0., 0.) if kind == "point_cross" else (0., 0., .06)
        polarization = "Ex" if kind == "point_cross" else (.4, -.3, 1.)
        source = mw.PointDipole(position=position, polarization=polarization, width=.04, source_time=waveform)
    elif kind == "uniform":
        source = mw.UniformCurrentSource(center=(0., 0., .06), size=(.12,) * 3,
            polarization=(.4, -.3, 1.), source_time=waveform)
    else:
        axes = ((-.06, .06), (-.06, .06), (0., .12))
        currents = (("My", 40.),) if kind == "magnetic" else (("Jx", .4), ("Jy", -.3), ("Jz", 1.))
        dataset = mw.CurrentDataset(axes, {name: value * np.ones((2, 2, 2))
            for name, value in currents})
        source = mw.CustomCurrentSource(dataset, source_time=waveform)
    scene.add_source(source)
    scene.add_monitor(mw.PointMonitor("probe", (.12, .12, .18), fields=("Ex", "Ey", "Ez")))
    simulation = mw.Simulation.fdtd(scene, frequency=1e9,
        run_time=mw.TimeConfig(time_steps=240, dt=2e-11),
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
    simulation.config.adjoint_checkpoint_stride = stride
    components = simulation.run().monitor("probe")["components"]
    field = torch.stack([components[name] for name in ("Ex", "Ey", "Ez")]).to(torch.complex128)
    weights = [0., 1., 0.] if kind == "point_cross" else [.7, 1.2, .9]
    return (field.abs().square() * field.real.new_tensor(weights)[:, None]).sum(), parameter


@pytest.mark.parametrize("kind,nonuniform", [("point", False), ("uniform", False), ("custom", False),
                                          ("point", True), ("magnetic", False), ("point_cross", False)])
def test_tensor_current_interface_gradient_matches_fd_across_replay_strides(kind, nonuniform):
    fixed, _ = _loss(.2, kind, 17, nonuniform=nonuniform)
    gradients = []
    for stride in (1, 17, 61):
        loss, parameter = _loss(.2, kind, stride, trainable=True, nonuniform=nonuniform)
        torch.testing.assert_close(loss.detach(), fixed, rtol=2e-6, atol=1e-8)
        loss.backward()
        gradients.append(parameter.grad.detach())
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, gradients[0], rtol=2e-5, atol=1e-8)
    for delta in (.003, .01):
        upper, _ = _loss(.2 + delta, kind, 17, nonuniform=nonuniform)
        lower, _ = _loss(.2 - delta, kind, 17, nonuniform=nonuniform)
        numerical = (upper - lower) / (2 * delta)
        torch.testing.assert_close(gradients[0], numerical, rtol=.002, atol=0.)
