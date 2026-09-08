"""Published scene descriptions cannot change when optimization inputs change."""
import pytest
import torch
from tests.gradients.test_batch_adjoint import density_batch, objective

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("slots", [1, 2])
def test_result_density_snapshot_survives_optimizer_update_and_close(slots):
    density = torch.full((3, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density, slots=slots)
    with batch.prepare() as prepared:
        first = prepared.run()
        objective(first).backward()
        assert density.grad is not None
        with torch.no_grad():
            density.fill_(.8)
        second = prepared.run()
    for index in range(3):
        before = first[index].scene.material_regions[0].density
        after = second[index].scene.material_regions[0].density
        torch.testing.assert_close(before, torch.full_like(before, .4), rtol=0, atol=0)
        torch.testing.assert_close(after, torch.full_like(after, .8), rtol=0, atol=0)
        assert not before.requires_grad and not after.requires_grad
        assert before.untyped_storage().data_ptr() != density.untyped_storage().data_ptr()


def test_snapshot_detaches_nested_geometry_and_metadata_but_tape_guards_original_inputs():
    density = torch.full((1, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density)
    original = batch.template.scene
    original.metadata["values"] = [torch.tensor([3.], device="cuda:0")]
    with batch.prepare() as prepared:
        result = prepared.run()[0]
    stored_size = result.scene.structures[0].geometry.size.clone()
    with torch.no_grad():
        original.structures[0].geometry.size.mul_(2)
        original.metadata["values"][0].add_(1)
        density.add_(.1)
    torch.testing.assert_close(result.scene.structures[0].geometry.size, stored_size, rtol=0, atol=0)
    torch.testing.assert_close(result.scene.metadata["values"][0], torch.tensor([3.], device="cuda:0"), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.raw_monitor("probe")["components"]["Ez"].real.sum().backward()
