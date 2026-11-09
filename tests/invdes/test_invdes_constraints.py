import itertools

import pytest
import torch

from witwin.maxwell import invdes

H = 0.01
RADIUS = 0.05  # five design cells
SATISFIED = 1e-10
VIOLATED = 1e-6


def _line(width_cells, n=80, *, void=False):
    x = torch.zeros(n, n, dtype=torch.float64)
    start = n // 2 - width_cells // 2
    x[start : start + width_cells, :] = 1.0
    return 1.0 - x if void else x


def _constraints(x, beta=32.0):
    filtered = invdes.conic_filter(x, RADIUS, H)
    projected = invdes.tanh_projection(filtered, beta)
    linewidth = invdes.minimum_linewidth_constraint(filtered, projected, spacing=H, radius=RADIUS)
    gap = invdes.minimum_gap_constraint(filtered, projected, spacing=H, radius=RADIUS)
    return float(linewidth), float(gap)


def test_gray_level_penalty_limits():
    assert float(invdes.gray_level_penalty(torch.tensor([0.0, 1.0, 1.0, 0.0]))) == 0.0
    assert float(invdes.gray_level_penalty(torch.full((5, 5), 0.5))) == pytest.approx(1.0)


@pytest.mark.parametrize("width", [6, 10, 20])
def test_length_scale_constraints_vanish_for_wide_features(width):
    linewidth, gap = _constraints(_line(width))
    assert linewidth < SATISFIED and gap < SATISFIED
    linewidth, gap = _constraints(_line(width, void=True))
    assert linewidth < SATISFIED and gap < SATISFIED


def test_linewidth_constraint_detects_thin_solid_line():
    linewidth, _ = _constraints(_line(4))
    assert linewidth > VIOLATED
    assert linewidth > 1e4 * _constraints(_line(10))[0]


def test_gap_constraint_detects_narrow_gap():
    _, gap = _constraints(_line(4, void=True))
    assert gap > VIOLATED
    assert gap > 1e4 * _constraints(_line(10, void=True))[1]


def test_length_scale_constraints_work_on_extruded_3d_slabs():
    slab = _line(4)[..., None]
    filtered = invdes.conic_filter(slab, RADIUS, (H, H, 0.05))
    projected = invdes.tanh_projection(filtered, 32.0)
    value = invdes.minimum_linewidth_constraint(filtered, projected, spacing=(H, H, 0.05), radius=RADIUS)
    assert float(value) == pytest.approx(_constraints(_line(4))[0], rel=1e-12)
    with pytest.raises(ValueError, match="radius"):
        invdes.minimum_gap_constraint(filtered, projected, spacing=(H, H, 0.05))


def test_length_scale_coefficient_reproduces_element_unit_default():
    # c = R_e^4 in element units equals radius^4 / h^2 with physical gradients.
    assert invdes.length_scale_coefficient(0.05, (0.01, 0.02)) == pytest.approx(5.0**4 * 0.01**2)


def _pipeline(spacing, padding, kind):
    def objective(params):
        filtered = invdes.apply_filter(
            params,
            invdes.filter_kernel(0.025, spacing, kind=kind, shape=params.shape, dtype=params.dtype),
            padding=padding,
        )
        projected = invdes.tanh_projection(filtered, 6.0, 0.45)
        eroded, _, dilated = invdes.robust_projections(filtered, 6.0, 0.45, 0.1)
        return (
            invdes.gray_level_penalty(projected)
            + (eroded * dilated).mean()
            + invdes.minimum_linewidth_constraint(filtered, projected, spacing=spacing, radius=0.025, c=0.5)
            + invdes.minimum_gap_constraint(filtered, projected, spacing=spacing, radius=0.025, c=0.5)
        )

    return objective


@pytest.mark.parametrize(
    "shape,padding,kind",
    [
        ((7, 6), "edge", "conic"),
        ((7, 6), "normalized", "gaussian"),
        ((7, 6), ("periodic", "reflect"), "disk"),
        ((5, 4, 4), "edge", "conic"),
        ((5, 4, 4), "constant", "gaussian"),
    ],
)
def test_gradcheck_filter_projection_penalty(shape, padding, kind):
    torch.manual_seed(0)
    spacing = (0.01,) * len(shape)
    params = torch.rand(shape, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_pipeline(spacing, padding, kind), (params,))


def test_gradcheck_symmetry_and_smoothed_projection():
    torch.manual_seed(1)
    params = torch.rand(6, 6, dtype=torch.float64, requires_grad=True)

    def objective(p):
        p = invdes.symmetrize(p, mirror=(1,), rotation=4)
        filtered = invdes.conic_filter(p, 0.025, 0.01)
        return invdes.smoothed_projection(filtered, 8.0, 0.5, spacing=0.01).pow(2).sum()

    assert torch.autograd.gradcheck(objective, (params,))


def test_symmetry_projectors_are_idempotent_and_invariant():
    torch.manual_seed(2)
    x = torch.rand(8, 8, 5, dtype=torch.float64)
    cases = [
        dict(mirror=(0,)),
        dict(mirror=(0, 2)),
        dict(rotation=2, rotation_dims=(0, 2)),
        dict(rotation=4),
        dict(diagonal=(0, 1)),
        dict(mirror=(0,), diagonal=(0, 1)),  # generates the full dihedral group D4
        dict(mirror=(1, 2), rotation=4),
    ]
    for kwargs in cases:
        once = invdes.symmetrize(x, **kwargs)
        torch.testing.assert_close(invdes.symmetrize(once, **kwargs), once)
    once = invdes.mirror_symmetrize(x, 0)
    torch.testing.assert_close(once, once.flip(0))
    rot = invdes.rotational_symmetrize(x, 4)
    torch.testing.assert_close(rot, torch.rot90(rot, 1, (0, 1)))
    diag = invdes.diagonal_symmetrize(x)
    torch.testing.assert_close(diag, diag.transpose(0, 1))
    d4 = invdes.symmetrize(x, mirror=(0,), diagonal=(0, 1))
    for flip_dims, transpose in itertools.product([(), (0,), (1,), (0, 1)], [False, True]):
        image = d4.flip(flip_dims) if flip_dims else d4
        image = image.transpose(0, 1) if transpose else image
        torch.testing.assert_close(image, d4)


def test_symmetry_survives_filtering_and_rejects_bad_grids():
    x = invdes.symmetrize(torch.rand(10, 10, dtype=torch.float64), rotation=4, mirror=(0,))
    filtered = invdes.conic_filter(x, 0.03, 0.01, padding="edge")
    torch.testing.assert_close(filtered, torch.rot90(filtered, 1, (0, 1)))
    torch.testing.assert_close(filtered, filtered.flip(0))
    with pytest.raises(ValueError, match="equal grid lengths"):
        invdes.rotational_symmetrize(torch.rand(6, 5), 4)
    with pytest.raises(ValueError):
        invdes.rotational_symmetrize(torch.rand(6, 6), 3)
