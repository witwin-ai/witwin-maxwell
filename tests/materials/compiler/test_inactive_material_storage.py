"""Inactive material channels use bounded storage without losing active physics."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("samples", [(1, 1, 1), (2, 2, 2)])
def test_inactive_channels_and_pole_templates_do_not_allocate_volume_storage(samples):
    sim = simulation(steps=8)
    model = sim.prepare().solver.scene.compile_materials(subpixel_samples=mw.SubpixelSpec(samples=samples, averaging="arithmetic"))
    channels = [model[key] for key in ("kerr_chi3", "chi2", "tpa_sigma",
                                      "modulation_cos", "modulation_sin", "modulation_omega")]
    channels.extend(model["eps_offdiag_components"].values())
    for value in channels:
        assert value.shape == model["eps_r"].shape
        assert value.untyped_storage().nbytes() == 4
        assert value.count_nonzero() == 0
    solver = sim.prepare().solver
    assert not solver._dispersive_templates
    assert not solver._magnetic_dispersive_templates


def test_zero_blend_retains_zero_derivative_for_trainable_occupancy():
    from witwin.maxwell.compiler.materials import _constant_zero_field, _blend_material
    occupancy = torch.rand((3, 4, 5), device="cuda:0", requires_grad=True)
    result = _blend_material(_constant_zero_field(occupancy.shape, occupancy.device), occupancy, value=0.)
    gradient, = torch.autograd.grad(result.sum(), occupancy)
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_later_zero_material_displaces_an_activated_channel():
    from witwin.maxwell.compiler.materials import _constant_zero_field, _blend_material
    occupancy = torch.full((3, 4, 5), .5, device="cuda:0")
    original = _constant_zero_field(occupancy.shape, occupancy.device)
    activated = _blend_material(original, occupancy, value=4.)
    replaced = _blend_material(activated, occupancy, value=0.)
    torch.testing.assert_close(replaced, torch.ones_like(replaced), rtol=0, atol=0)
    assert original.count_nonzero() == 0
