import math

import pytest
import torch

from witwin.maxwell import invdes

KINDS = ("conic", "disk", "gaussian")
PRESERVING = ("edge", "reflect", "periodic", "normalized")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spacing", [(0.01, 0.01), (0.01, 0.02), (0.01, 0.01, 0.01), (0.02, 0.01, 0.015)])
def test_kernel_is_normalized_and_mirror_symmetric(kind, spacing):
    kernel = invdes.filter_kernel(0.05, spacing, kind=kind, dtype=torch.float64)
    assert kernel.ndim == len(spacing)
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert (kernel >= 0).all()
    for dim in range(kernel.ndim):
        torch.testing.assert_close(kernel, kernel.flip(dim))


def test_kernel_half_width_follows_physical_radius():
    kernel = invdes.filter_kernel(0.05, (0.01, 0.025), kind="conic")
    assert kernel.shape == (11, 5)
    gaussian = invdes.filter_kernel(0.04, (0.01, 0.01), kind="gaussian", dtype=torch.float64)
    assert gaussian.shape == (13, 13)  # truncated at 1.5 radius = 3 sigma
    sigma_cells = 2  # sigma = radius / 2 = 0.02 m = 2 cells
    ratio = gaussian[6 + sigma_cells, 6] / gaussian[6, 6]
    assert math.isclose(float(ratio), math.exp(-0.5), rel_tol=1e-12)
    conic = invdes.filter_kernel(0.05, (0.01, 0.01), kind="conic", dtype=torch.float64)
    assert math.isclose(float(conic[5 + 2, 5] / conic[5, 5]), 1.0 - 0.02 / 0.05, rel_tol=1e-12)


def test_kernel_below_one_cell_is_identity_and_singleton_axes_are_extruded():
    identity = invdes.filter_kernel(0.005, (0.01, 0.01), kind="conic")
    assert identity.shape == (1, 1)
    slab = invdes.filter_kernel(0.05, (0.01, 0.01, 0.01), kind="conic", shape=(20, 20, 1))
    assert slab.shape == (11, 11, 1)
    x = torch.rand(20, 20, dtype=torch.float64)
    torch.testing.assert_close(
        invdes.conic_filter(x[..., None], 0.05, (0.01, 0.01, 0.01))[..., 0],
        invdes.conic_filter(x, 0.05, (0.01, 0.01)),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("padding", PRESERVING)
@pytest.mark.parametrize("shape", [(12, 9), (8, 7, 6)])
def test_filters_preserve_constant_fields(kind, padding, shape):
    spacing = (0.01,) * len(shape)
    x = torch.full(shape, 0.37, dtype=torch.float64)
    module = invdes.DensityFilter(0.03, spacing, kind=kind, padding=padding)
    torch.testing.assert_close(module(x), x)


def test_constant_padding_uses_pad_value():
    x = torch.full((10, 10), 0.6, dtype=torch.float64)
    same = invdes.conic_filter(x, 0.03, 0.01, padding="constant", pad_value=0.6)
    torch.testing.assert_close(same, x)
    void = invdes.conic_filter(x, 0.03, 0.01, padding="constant", pad_value=0.0)
    assert void[0, 0] < 0.6 and math.isclose(float(void[5, 5]), 0.6, rel_tol=1e-12)


def test_periodic_padding_is_translation_equivariant():
    x = torch.rand(16, 12, dtype=torch.float64)
    filtered = invdes.gaussian_filter(x, 0.03, 0.01, padding="periodic")
    shifted = invdes.gaussian_filter(torch.roll(x, (3, -2), (0, 1)), 0.03, 0.01, padding="periodic")
    torch.testing.assert_close(shifted, torch.roll(filtered, (3, -2), (0, 1)))


def test_per_axis_padding_and_invalid_modes():
    x = torch.full((10, 8), 0.4, dtype=torch.float64)
    torch.testing.assert_close(invdes.conic_filter(x, 0.03, 0.01, padding=("periodic", "normalized")), x)
    with pytest.raises(ValueError, match="cannot be mixed"):
        invdes.conic_filter(x, 0.03, 0.01, padding=("constant", "normalized"))
    with pytest.raises(ValueError, match="reflect"):
        invdes.conic_filter(torch.ones(3, 3), 0.05, 0.01, padding="reflect")
    with pytest.raises(ValueError):
        invdes.filter_kernel(0.05, 0.01, kind="box", ndim=2)


def test_physical_radius_is_resolution_independent():
    """The same radius in metres smooths a step edge identically on a refined grid."""
    coarse_h, fine_h, radius = 0.01, 0.005, 0.06
    xc = (torch.arange(40, dtype=torch.float64) + 0.5) * coarse_h
    xf = (torch.arange(80, dtype=torch.float64) + 0.5) * fine_h
    coarse = (xc < 0.2).to(torch.float64)[:, None].expand(40, 3).contiguous()
    fine = (xf < 0.2).to(torch.float64)[:, None].expand(80, 3).contiguous()
    fc = invdes.conic_filter(coarse, radius, (coarse_h, coarse_h))[:, 1]
    ff = invdes.conic_filter(fine, radius, (fine_h, fine_h))[:, 1]
    # Compare at the coarse cell centres (average of the two fine cells inside).
    ff_on_coarse = 0.5 * (ff[0::2] + ff[1::2])
    assert torch.max(torch.abs(fc - ff_on_coarse)) < 0.02


def test_design_spacing_maps_box_size_to_cells():
    assert invdes.design_spacing((1.6e-6, 0.8e-6, 0.22e-6), (40, 20, 1)) == pytest.approx((4e-8, 4e-8, 2.2e-7))
