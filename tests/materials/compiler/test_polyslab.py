import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _build_scene(device="cpu"):
    return mw.Scene(
        domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
        grid=mw.GridSpec.uniform(0.1),
        device=device,
    )


def _scalar_sd(geometry, point):
    x, y, z = (torch.tensor(coord, dtype=torch.float32) for coord in point)
    return geometry.signed_distance(x, y, z).item()


def test_rectangular_polyslab_matches_equivalent_box():
    center = (0.1, -0.05)
    half = (0.3, 0.25)
    bounds = (-0.2, 0.2)

    box_scene = _build_scene()
    box_scene.add_structure(
        mw.Structure(
            name="box",
            geometry=mw.Box(position=(center[0], center[1], 0.0), size=(2 * half[0], 2 * half[1], bounds[1] - bounds[0])),
            material=mw.Material(eps_r=4.0),
        )
    )

    poly_scene = _build_scene()
    poly_scene.add_structure(
        mw.Structure(
            name="poly",
            geometry=mw.PolySlab(
                vertices=[
                    (center[0] - half[0], center[1] - half[1]),
                    (center[0] + half[0], center[1] - half[1]),
                    (center[0] + half[0], center[1] + half[1]),
                    (center[0] - half[0], center[1] + half[1]),
                ],
                bounds=bounds,
                axis="z",
            ),
            material=mw.Material(eps_r=4.0),
        )
    )

    eps_box = prepare_scene(box_scene).permittivity
    eps_poly = prepare_scene(poly_scene).permittivity

    assert torch.allclose(eps_poly, eps_box, atol=1.0e-5)


@pytest.mark.parametrize("reference_plane", ["bottom", "middle", "top"])
def test_tapered_polyslab_cross_sections_match_analytic_half_widths(reference_plane):
    half_width = 0.4
    angle = 0.2
    bounds = (-0.3, 0.3)
    slab = mw.PolySlab(
        vertices=[
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width),
        ],
        bounds=bounds,
        axis="z",
        sidewall_angle=angle,
        reference_plane=reference_plane,
    )
    reference = {"bottom": bounds[0], "middle": 0.5 * (bounds[0] + bounds[1]), "top": bounds[1]}[reference_plane]

    eps = 2.0e-3
    for z in (-0.25, 0.0, 0.25):
        expected = half_width - math.tan(angle) * (z - reference)
        assert _scalar_sd(slab, (expected - eps, 0.0, z)) < 0.0
        assert _scalar_sd(slab, (expected + eps, 0.0, z)) > 0.0
        # Cross-sections stay square under the taper, so the v-direction matches too.
        assert _scalar_sd(slab, (0.0, expected - eps, z)) < 0.0
        assert _scalar_sd(slab, (0.0, expected + eps, z)) > 0.0


def test_polyslab_axis_permutations_are_consistent():
    vertices = [(-0.4, -0.2), (0.4, -0.1), (0.3, 0.3), (-0.2, 0.4)]
    bounds = (-0.25, 0.35)
    u = torch.linspace(-0.6, 0.6, 7, dtype=torch.float32)
    v = torch.linspace(-0.5, 0.5, 6, dtype=torch.float32)
    w = torch.linspace(-0.4, 0.5, 5, dtype=torch.float32)
    uu, vv, ww = torch.meshgrid(u, v, w, indexing="ij")

    slab_z = mw.PolySlab(vertices, bounds, axis=2, sidewall_angle=0.15)
    slab_y = mw.PolySlab(vertices, bounds, axis=1, sidewall_angle=0.15)
    slab_x = mw.PolySlab(vertices, bounds, axis=0, sidewall_angle=0.15)

    sd_z = slab_z.signed_distance(uu, vv, ww)
    # Plane coordinates per axis: z -> (x, y), y -> (x, z), x -> (y, z).
    sd_y = slab_y.signed_distance(uu, ww, vv)
    sd_x = slab_x.signed_distance(ww, uu, vv)

    assert torch.allclose(sd_y, sd_z, atol=1.0e-6)
    assert torch.allclose(sd_x, sd_z, atol=1.0e-6)


def test_complex_polyslab_bowtie_fills_two_triangles():
    bowtie = mw.ComplexPolySlab(
        loops=[[(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)]],
        bounds=(-0.5, 0.5),
    )

    assert _scalar_sd(bowtie, (0.0, 0.5, 0.0)) < 0.0
    assert _scalar_sd(bowtie, (0.0, -0.5, 0.0)) < 0.0
    assert _scalar_sd(bowtie, (0.5, 0.0, 0.0)) > 0.0
    assert _scalar_sd(bowtie, (-0.5, 0.0, 0.0)) > 0.0


def test_complex_polyslab_hole_leaves_interior_empty():
    outer = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    hole = [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)]
    slab = mw.ComplexPolySlab(loops=[outer, hole], bounds=(-0.5, 0.5))

    assert _scalar_sd(slab, (0.0, 0.0, 0.0)) > 0.0
    assert _scalar_sd(slab, (0.7, 0.0, 0.0)) < 0.0
    assert _scalar_sd(slab, (1.3, 0.0, 0.0)) > 0.0
    assert abs(_scalar_sd(slab, (0.4, 0.0, 0.0))) < 1.0e-5

    scene = _build_scene()
    scene.add_structure(
        mw.Structure(name="ring", geometry=slab, material=mw.Material(eps_r=6.0))
    )
    prepared = prepare_scene(scene)
    x_coords = prepared.X[:, 0, 0]
    hole_i = torch.argmin(x_coords.abs()).item()
    ring_i = torch.argmin((x_coords - 0.7).abs()).item()
    j = torch.argmin(prepared.Y[0, :, 0].abs()).item()
    k = torch.argmin(prepared.Z[0, 0, :].abs()).item()

    assert prepared.permittivity[hole_i, j, k].item() == pytest.approx(1.0, abs=1.0e-3)
    assert prepared.permittivity[ring_i, j, k].item() == pytest.approx(6.0, abs=1.0e-3)


def test_polyslab_compiled_occupancy_propagates_gradients_through_vertices():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.2),
        device="cpu",
    )
    vertices = torch.nn.Parameter(
        torch.tensor([[-0.31, -0.24], [0.29, -0.24], [0.29, 0.26], [-0.31, 0.26]], dtype=torch.float32)
    )
    scene.add_structure(
        mw.Structure(
            name="trainable_slab",
            geometry=mw.PolySlab(vertices, bounds=(-0.29, 0.28), axis="z"),
            material=mw.Material(eps_r=4.0),
        )
    )

    eps_r, mu_r = prepare_scene(scene).compile_material_tensors()
    loss = eps_r.sum() + mu_r.sum()
    loss.backward()

    assert vertices.grad is not None
    assert torch.all(torch.isfinite(vertices.grad))
    assert torch.any(vertices.grad.abs() > 1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU compile smoke test.")
def test_polyslab_scene_compiles_material_tensors_on_gpu():
    scene = _build_scene(device="cuda")
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.PolySlab(
                vertices=[(-0.5, -0.4), (0.5, -0.4), (0.3, 0.5)],
                bounds=(-0.3, 0.3),
                axis="z",
                sidewall_angle=0.1,
            ),
            material=mw.Material(eps_r=3.0),
        )
    )

    prepared = prepare_scene(scene)
    eps_r, mu_r = prepared.compile_material_tensors()

    assert eps_r.device.type == "cuda"
    assert mu_r.device.type == "cuda"
    assert torch.all(torch.isfinite(eps_r))
    assert eps_r.max().item() > 2.0
