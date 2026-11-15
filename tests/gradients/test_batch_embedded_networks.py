"""Batch lanes report embedded networks on forward-only and differentiable runs."""
import pytest
import torch

import witwin.maxwell as mw
from tests.gradients.test_fdtd_network_adjoint import _rational_network
from tests.rf.network.test_network_runtime import _fdtd_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

_NETWORK = "differentiable_load"


def _template():
    block, _residues, _direct = _rational_network(residue_value=2.0e8, direct_value=1.0e-2, trainable=False)
    scene = _fdtd_scene(network=block)
    scene = scene.updated(subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"))
    scene = scene.updated(material_regions=(
        *scene.material_regions,
        mw.MaterialRegion(
            name="near_terminal",
            geometry=mw.Box(position=(0.005, 0.0, 0.005), size=(0.005, 0.005, 0.005)),
            density=torch.zeros(1, 1, 1, device="cuda", dtype=torch.float64),
            eps_bounds=(1.0, 4.0), mu_bounds=(1.0, 1.0)),
    ))
    return mw.Simulation.fdtd(scene, frequencies=2.75e9, config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=48),
                              spectral_sampler=mw.fdtd.SpectralSampler(window="none")))


def _batch(density):
    return mw.Simulation.batch(_template(), parameters=mw.execution.BatchParameters(region_densities={"near_terminal": density}),
        execution=mw.execution.BatchExecution(devices=("cuda:0",), batch_size=2, graph="off",
                                    memory_limit_bytes=256 * 1024**2, reserve_bytes=0))


def _objective(results):
    return sum(mw.rf.embedded_network(result, _NETWORK).voltage.abs().square().sum()
               + 0.3 * mw.rf.embedded_network(result, _NETWORK).absorbed_power.sum() / 1e-11
               + 0.7 * result.port("feed").current.real.sum() / 1e-6
               for result in results)


@pytest.mark.parametrize("trainable", [False, True])
def test_batch_lanes_keep_embedded_networks(trainable):
    density = torch.sigmoid(torch.tensor([0.1, -0.4], device="cuda", dtype=torch.float64)).reshape(2, 1, 1, 1)
    density = density.detach().requires_grad_(trainable)
    results = _batch(density).run()
    results.results()
    assert all((result.stats().get("batch_adjoint") == "native_checkpoint_per_lane") is trainable
               for result in results)
    for result in results:
        network = mw.rf.embedded_network(result, _NETWORK)
        torch.testing.assert_close(network.voltage[0], result.port("feed").voltage, rtol=0, atol=0)
        assert network.voltage.requires_grad is trainable
        assert result.port("feed").voltage.requires_grad is trainable
    if not trainable:
        return
    actual, = torch.autograd.grad(_objective(results), density)
    reference = density.detach().clone().requires_grad_()
    expected_results = []
    for lane in range(2):
        scene = _batch(reference).scene(lane)
        expected_results.append(mw.Simulation.fdtd(scene, frequencies=2.75e9, config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=48),
                                                   spectral_sampler=mw.fdtd.SpectralSampler(window="none"))).run())
    expected, = torch.autograd.grad(_objective(expected_results), reference)
    assert expected.abs().min() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-12)
