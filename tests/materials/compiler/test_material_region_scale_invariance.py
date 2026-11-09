"""MaterialRegion occupancy must not depend on the absolute length scale of the scene."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _region_profile(scale, *, trainable):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0 * scale, 1.0 * scale), (-0.4 * scale, 0.4 * scale), (-0.4 * scale, 0.4 * scale))),
        grid=mw.GridSpec.uniform(0.04 * scale),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
    )
    density = torch.full((20, 10, 1), 0.5, requires_grad=trainable)
    scene.add_material_region(mw.MaterialRegion(
        name="design", geometry=mw.Box(position=(0.4 * scale, 0.0, 0.0), size=(0.8 * scale, 0.4 * scale, 0.22 * scale)),
        density=density, eps_bounds=(1.0, 12.0)))
    prepared = prepare_scene(scene)
    eps = prepared.compile_materials()["eps_r"].detach()
    iy, iz = int(torch.argmin(prepared.y.abs())), int(torch.argmin(prepared.z.abs()))
    # Sample the x row at the region face and 1, 2, 5 and 15 cells outside it.
    indices = [int(torch.argmin((prepared.x + cells * 0.04 * scale).abs())) for cells in (0, 1, 2, 5, 15)]
    return torch.stack([eps[i, iy, iz] for i in indices])


@pytest.mark.parametrize("trainable", [False, True])
def test_material_region_occupancy_is_scale_invariant(trainable):
    # The smoothing width is cell-relative, so micrometre and metre scenes compile identically.
    metre = _region_profile(1.0, trainable=trainable)
    micrometre = _region_profile(1e-6, trainable=trainable)
    torch.testing.assert_close(micrometre, metre, rtol=0.0, atol=1e-6)
    assert torch.all(micrometre[1:] == 1.0)
