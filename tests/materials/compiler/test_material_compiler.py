import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def _build_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-1, 1), (-1, 1), (-1, 1))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    )


def _prepared_scene(scene):
    return prepare_scene(scene)


def _cube_mesh(*, size=0.6, requires_grad=False):
    half = size / 2.0
    vertices = torch.tensor(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        ],
        dtype=torch.int64,
    )
    return vertices, faces


def test_shared_material_compiler_matches_rotated_box_scene_grid():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(
            name="box",
            geometry=mw.Box(position=(0, 0, 0), size=(0.8, 0.4, 0.4), rotation=(0.0, 0.0, 0.6)),
            material=mw.Material(eps_r=3.5),
        )
    )

    prepared_scene = _prepared_scene(scene)
    eps_before = prepared_scene.permittivity.clone()
    mu_before = prepared_scene.permeability.clone()
    eps_compiled, mu_compiled = prepared_scene.compile_material_tensors()

    assert torch.equal(eps_compiled, eps_before)
    assert torch.equal(mu_compiled, mu_before)


def test_shared_material_compiler_matches_extended_geometries():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(name="ellipsoid", geometry=mw.Ellipsoid(position=(-0.2, 0.0, 0.0), radii=(0.35, 0.2, 0.25)), material=mw.Material(eps_r=4.0))
    )
    scene.add_structure(
        mw.Structure(name="torus", geometry=mw.Torus(position=(0.35, 0.0, 0.0), major_radius=0.2, minor_radius=0.08), material=mw.Material(eps_r=5.0))
    )
    scene.add_structure(
        mw.Structure(name="hollow_box", geometry=mw.HollowBox(position=(0.0, 0.35, 0.0), outer_size=(0.5, 0.5, 0.5), inner_size=(0.2, 0.2, 0.2)), material=mw.Material(eps_r=6.0))
    )

    prepared_scene = _prepared_scene(scene)
    eps_before = prepared_scene.permittivity.clone()
    mu_before = prepared_scene.permeability.clone()
    eps_compiled, mu_compiled = prepared_scene.compile_material_tensors()

    assert torch.equal(eps_compiled, eps_before)
    assert torch.equal(mu_compiled, mu_before)


def test_refresh_material_grids_rebuilds_scene_from_geometry_records():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(name="cone", geometry=mw.Cone(position=(0.0, 0.0, -0.2), radius=0.25, height=0.5), material=mw.Material(eps_r=4.5))
    )
    scene.add_structure(
        mw.Structure(name="prism", geometry=mw.Prism(position=(0.0, -0.25, 0.0), radius=0.2, height=0.4, num_sides=5), material=mw.Material(eps_r=3.0))
    )

    prepared_scene = _prepared_scene(scene)
    expected_eps = prepared_scene.permittivity.clone()
    expected_mu = prepared_scene.permeability.clone()

    prepared_scene.permittivity.fill_(1.0)
    prepared_scene.permeability.fill_(1.0)
    prepared_scene.refresh_material_grids()

    assert torch.equal(prepared_scene.permittivity, expected_eps)
    assert torch.equal(prepared_scene.permeability, expected_mu)


def test_soft_occupancy_returns_fractional_interface_fill_without_subpixel_sampling():
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)

    scene = mw.Scene(domain=domain, grid=grid, device="cpu", subpixel_samples=1)

    structure = mw.Structure(
        name="edge_box",
        geometry=mw.Box(position=(0.125, 0.5, 0.5), size=(0.25, 1.0, 1.0)),
        material=mw.Material(eps_r=5.0),
    )
    scene.add_structure(structure)

    interface_value = _prepared_scene(scene).permittivity[0, 1, 1].item()
    assert 1.0 < interface_value < 5.0
    assert abs(interface_value - 3.0) < 0.35


def test_soft_occupancy_polarized_normal_component_uses_harmonic_mean():
    # Domain [0,1]^3, grid 0.25 -> nodes at 0, 0.25, 0.5, 0.75, 1.0. The slab low face
    # lands on the interior node x=0.25 with a neighbouring interior node at x=0.5, so
    # the node-based finite-difference normal at x=0.25 resolves along x.
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)

    scene = mw.Scene(
        domain=domain,
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(averaging="polarized"),
    )
    scene.add_structure(
        mw.Structure(
            name="slab",
            geometry=mw.Box(position=(0.625, 0.5, 0.5), size=(0.75, 1.0, 1.0)),
            material=mw.Material(eps_r=5.0),
        )
    )

    eps, _ = _prepared_scene(scene).compile_material_components()
    node = (1, 1, 1)
    harmonic = 1.0 / (0.5 / 1.0 + 0.5 / 5.0)
    assert abs(eps["x"][node].item() - harmonic) < 1e-2
    assert abs(eps["y"][node].item() - 3.0) < 1e-2
    assert abs(eps["z"][node].item() - 3.0) < 1e-2


def test_scene_frequency_specific_material_tensors_match_dispersive_medium_response():
    scene = _build_scene()
    material = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.DebyePole(delta_eps=3.0, tau=5e-10),),
        lorentz_poles=(
            mw.LorentzPole(
                delta_eps=1.5,
                resonance_frequency=4.0e9,
                gamma=0.2e9,
            ),
        ),
    )
    scene.add_structure(
        mw.Structure(
            name="fill",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.8, 1.8, 1.8)),
            material=material,
        )
    )

    frequency = 2.5e9
    prepared_scene = _prepared_scene(scene)
    eps_r, mu_r = prepared_scene.compile_relative_materials(frequency=frequency)
    center_index = (prepared_scene.Nx // 2, prepared_scene.Ny // 2, prepared_scene.Nz // 2)
    expected = torch.tensor(material.relative_permittivity(frequency), dtype=torch.complex64)

    assert torch.is_complex(eps_r)
    assert torch.allclose(eps_r[center_index], expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(mu_r, prepared_scene.permeability)
    assert prepared_scene.permittivity[center_index].item() == material.eps_r


def test_compiled_material_model_tracks_partial_dispersive_fill_weight():
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)
    material = mw.Material.debye(eps_inf=2.0, delta_eps=4.0, tau=1.0e-9)
    scene = mw.Scene(domain=domain, grid=grid, device="cpu", subpixel_samples=4)
    scene.add_structure(
        mw.Structure(
            name="edge_box",
            geometry=mw.Box(position=(0.125, 0.5, 0.5), size=(0.25, 1.0, 1.0)),
            material=material,
        )
    )

    prepared_scene = _prepared_scene(scene)
    model = prepared_scene.compile_materials()
    weight = model["debye_poles"][0]["weight"][0, 1, 1].item()
    assert 0.0 < weight < 1.0

    frequency = 1.0e9
    eps_r, _ = prepared_scene.compile_relative_materials(frequency=frequency)
    expected = prepared_scene.permittivity[0, 1, 1].item() + weight * (
        material.relative_permittivity(frequency) - material.eps_r
    )
    assert torch.allclose(
        eps_r[0, 1, 1],
        torch.tensor(expected, dtype=torch.complex64),
        atol=1e-5,
        rtol=1e-5,
    )


def test_subpixel_sphere_average_is_closer_to_analytic_volume_fraction():
    domain = mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))
    grid = mw.GridSpec.uniform(1.0)
    radius = 0.25
    eps_r = 5.0
    analytic_fraction = (4.0 / 3.0) * np.pi * radius**3
    analytic_eps = 1.0 + (eps_r - 1.0) * analytic_fraction

    single_sample_scene = mw.Scene(domain=domain, grid=grid, device="cpu", subpixel_samples=1)
    smooth_scene = mw.Scene(domain=domain, grid=grid, device="cpu", subpixel_samples=12)

    structure = mw.Structure(
        name="small_sphere",
        geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=radius),
        material=mw.Material(eps_r=eps_r),
    )
    single_sample_scene.add_structure(structure)
    smooth_scene.add_structure(structure)

    single_sample_value = _prepared_scene(single_sample_scene).permittivity[1, 1, 1].item()
    smoothed_value = _prepared_scene(smooth_scene).permittivity[1, 1, 1].item()

    assert abs(smoothed_value - analytic_eps) < 0.08
    assert abs(smoothed_value - analytic_eps) < abs(single_sample_value - analytic_eps)


def test_compile_material_tensors_propagate_gradients_through_box_parameters():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.2),
        device="cpu",
    )
    position = torch.nn.Parameter(torch.tensor([0.05, -0.03, 0.02], dtype=torch.float32))
    size = torch.nn.Parameter(torch.tensor([0.45, 0.35, 0.40], dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.15, -0.10, 0.20], dtype=torch.float32))
    scene.add_structure(
        mw.Structure(
            name="trainable_box",
            geometry=mw.Box(position=position, size=size, rotation=rotation),
            material=mw.Material(eps_r=4.0),
        )
    )

    eps_r, mu_r = _prepared_scene(scene).compile_material_tensors()
    loss = eps_r.sum() + mu_r.sum()
    loss.backward()

    for grad in (position.grad, size.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)


def test_compile_material_tensors_propagate_gradients_through_mesh_vertices():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.2),
        device="cpu",
    )
    vertices, faces = _cube_mesh(size=0.7, requires_grad=True)
    scene.add_structure(
        mw.Structure(
            name="trainable_mesh",
            geometry=mw.Mesh(vertices, faces, recenter=False, fill_mode="solid"),
            material=mw.Material(eps_r=4.0),
        )
    )

    eps_r, mu_r = _prepared_scene(scene).compile_material_tensors()
    loss = eps_r.sum() + mu_r.sum()
    loss.backward()

    assert vertices.grad is not None
    assert torch.all(torch.isfinite(vertices.grad))
    assert torch.any(vertices.grad.abs() > 1.0e-6)


def test_compile_materials_recomputes_for_trainable_geometry_without_manual_cache_invalidation():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    position = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    scene.add_structure(
        mw.Structure(
            name="moving_box",
            geometry=mw.Box(position=position, size=(0.4, 0.4, 0.4)),
            material=mw.Material(eps_r=3.0),
        )
    )

    prepared_scene = _prepared_scene(scene)
    before = prepared_scene.compile_materials()["eps_r"].clone()
    with torch.no_grad():
        position.copy_(torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32))
    after = prepared_scene.compile_materials()["eps_r"]

    assert not torch.allclose(before, after)


def test_material_region_density_overlay_updates_compiled_materials_without_changing_static_path():
    scene = _build_scene()
    baseline_eps = _prepared_scene(scene).permittivity.clone()

    density = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.25, 1.0], [0.5, 0.0]],
        ],
        dtype=torch.float32,
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6)),
            density=density,
            eps_bounds=(2.0, 6.0),
            mu_bounds=(1.0, 1.0),
        )
    )

    prepared_scene = _prepared_scene(scene)
    model = prepared_scene.compile_materials()
    assert "eps_r_base" in model
    assert "eps_r_design" in model
    assert "design_mask" in model
    assert torch.equal(model["eps_r_base"], baseline_eps)
    assert torch.any(model["design_mask"])
    assert torch.any(prepared_scene.permittivity != baseline_eps)


def test_material_region_preserves_tensor_device_dtype_and_gradients():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    density = torch.nn.Parameter(torch.full((2, 2, 2), 0.5, dtype=torch.float32))
    scene.add_material_region(
        mw.MaterialRegion(
            name="design_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            density=density,
            eps_bounds=(1.0, 5.0),
            filter_radius=0.05,
            projection_beta=4.0,
        )
    )

    eps_r, mu_r = _prepared_scene(scene).compile_material_tensors()
    loss = eps_r.sum() + mu_r.sum()
    loss.backward()

    assert eps_r.dtype == torch.float32
    assert mu_r.dtype == torch.float32
    assert eps_r.device.type == "cpu"
    assert density.grad is not None
    assert density.grad.shape == density.shape


def test_structure_priority_overrides_append_order():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(
            name="late_low_priority",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6)),
            material=mw.Material(eps_r=5.0),
            priority=0,
        )
    )
    scene.add_structure(
        mw.Structure(
            name="early_high_priority",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6)),
            material=mw.Material(eps_r=2.0),
            priority=10,
        )
    )

    prepared_scene = _prepared_scene(scene)
    center_index = (prepared_scene.Nx // 2, prepared_scene.Ny // 2, prepared_scene.Nz // 2)
    assert prepared_scene.permittivity[center_index].item() == 2.0


def test_anisotropic_medium_compiles_component_fields_and_scalar_summary():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(
            name="aniso_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.Material(
                eps_r=1.0,
                mu_r=1.0,
                epsilon_tensor=mw.DiagonalTensor3(2.0, 4.0, 8.0),
                mu_tensor=mw.DiagonalTensor3(1.5, 2.5, 3.5),
            ),
        )
    )

    prepared_scene = _prepared_scene(scene)
    eps_components, mu_components = prepared_scene.compile_material_components()
    eps_r, mu_r = prepared_scene.compile_material_tensors()
    center_index = (prepared_scene.Nx // 2, prepared_scene.Ny // 2, prepared_scene.Nz // 2)

    assert eps_components["x"][center_index].item() == pytest.approx(2.0)
    assert eps_components["y"][center_index].item() == pytest.approx(4.0)
    assert eps_components["z"][center_index].item() == pytest.approx(8.0)
    assert mu_components["x"][center_index].item() == pytest.approx(1.5)
    assert mu_components["y"][center_index].item() == pytest.approx(2.5)
    assert mu_components["z"][center_index].item() == pytest.approx(3.5)
    assert eps_r[center_index].item() == pytest.approx((2.0 + 4.0 + 8.0) / 3.0)
    assert mu_r[center_index].item() == pytest.approx((1.5 + 2.5 + 3.5) / 3.0)
    assert prepared_scene.permittivity[center_index].item() == pytest.approx((2.0 + 4.0 + 8.0) / 3.0)
    assert prepared_scene.permeability[center_index].item() == pytest.approx((1.5 + 2.5 + 3.5) / 3.0)


def test_full_anisotropic_medium_compiles_off_diagonal_permittivity_fields():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(
            name="full_aniso_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.Material(
                epsilon_tensor=mw.Tensor3x3(
                    ((2.0, 0.2, 0.1), (0.2, 3.0, 0.3), (0.1, 0.3, 4.0))
                )
            ),
        )
    )

    prepared_scene = _prepared_scene(scene)
    model = prepared_scene.compile_materials()
    center_index = (prepared_scene.Nx // 2, prepared_scene.Ny // 2, prepared_scene.Nz // 2)

    assert model["eps_components"]["x"][center_index].item() == pytest.approx(2.0)
    assert model["eps_components"]["y"][center_index].item() == pytest.approx(3.0)
    assert model["eps_components"]["z"][center_index].item() == pytest.approx(4.0)
    assert model["eps_offdiag_components"]["xy"][center_index].item() == pytest.approx(0.2)
    assert model["eps_offdiag_components"]["xz"][center_index].item() == pytest.approx(0.1)
    assert model["eps_offdiag_components"]["yz"][center_index].item() == pytest.approx(0.3)


def test_anisotropic_sigma_tensor_produces_component_specific_complex_permittivity():
    scene = _build_scene()
    scene.add_structure(
        mw.Structure(
            name="conductive_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.Material(
                eps_r=2.0,
                sigma_e_tensor=mw.DiagonalTensor3(0.0, 1.0, 5.0),
            ),
        )
    )

    frequency = 2.0e9
    prepared_scene = _prepared_scene(scene)
    eps_components, _ = prepared_scene.compile_material_components(frequency=frequency)
    eps_r, _ = prepared_scene.compile_relative_materials(frequency=frequency)
    center_index = (prepared_scene.Nx // 2, prepared_scene.Ny // 2, prepared_scene.Nz // 2)
    expected_summary = (
        eps_components["x"][center_index]
        + eps_components["y"][center_index]
        + eps_components["z"][center_index]
    ) / 3.0

    assert torch.is_complex(eps_components["x"])
    assert abs(eps_components["x"][center_index].imag.item()) < 1.0e-6
    assert abs(eps_components["z"][center_index].imag.item()) > abs(eps_components["y"][center_index].imag.item())
    assert torch.allclose(eps_r[center_index], expected_summary, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: mw.Material(
                eps_r=2.0,
                epsilon_tensor=mw.DiagonalTensor3(2.0, 2.5, 3.0),
                orientation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            "orientation",
        ),
        (
            lambda: mw.Material(
                eps_r=2.0,
                epsilon_tensor=mw.DiagonalTensor3(2.0, 2.5, 3.0),
                debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-9),),
            ),
            "Anisotropic electric dispersion",
        ),
        (
            lambda: mw.Material(
                mu_r=1.0,
                mu_tensor=mw.DiagonalTensor3(1.2, 1.3, 1.4),
                mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=1.0e9, gamma=1.0e8),),
            ),
            "Magnetic dispersion with mu_tensor",
        ),
        (
            lambda: mw.Material(
                eps_r=2.0,
                debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-9),),
                kerr_chi3=1.0e-10,
            ),
            "Kerr media cannot be combined with dispersive poles",
        ),
        (
            lambda: mw.Material(
                eps_r=2.0,
                epsilon_tensor=mw.DiagonalTensor3(2.0, 2.5, 3.0),
                kerr_chi3=1.0e-10,
            ),
            "Kerr media cannot be combined with anisotropic tensors",
        ),
    ],
)
def test_medium_rejects_unsupported_tensor_and_nonlinear_combinations(factory, message):
    with pytest.raises(NotImplementedError, match=message):
        factory()


def test_scene_rejects_kerr_combined_with_other_dispersive_or_anisotropic_materials():
    dispersive_scene = _build_scene()
    dispersive_scene.add_structure(
        mw.Structure(
            name="kerr_box",
            geometry=mw.Box(position=(-0.2, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-10),
        )
    )
    with pytest.raises(NotImplementedError, match="same Scene"):
        dispersive_scene.add_structure(
            mw.Structure(
                name="debye_box",
                geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
                material=mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1.0e-9),
            )
        )

    anisotropic_scene = _build_scene()
    anisotropic_scene.add_structure(
        mw.Structure(
            name="kerr_box",
            geometry=mw.Box(position=(-0.2, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-10),
        )
    )
    with pytest.raises(NotImplementedError, match="same Scene"):
        anisotropic_scene.add_structure(
            mw.Structure(
                name="aniso_box",
                geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
                material=mw.Material(eps_r=2.0, epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0)),
            )
        )
