from __future__ import annotations

from pathlib import Path

import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene

TESTS_ROOT = Path(__file__).resolve()
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if TESTS_ROOT.name != "tests":
    raise RuntimeError("Unable to locate tests root directory.")

ASSET_DIR = TESTS_ROOT / "assets"


def _prepared_scene(scene):
    return prepare_scene(scene)


def _cube_vertices_faces():
    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
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


def test_mesh_from_obj_loads_and_reports_topology():
    mesh = mw.Mesh.from_obj(
        ASSET_DIR / "cube.obj",
        position=(1.0, 2.0, 3.0),
        scale=2.0,
        recenter=True,
    )

    assert mesh.vertex_count == 8
    assert mesh.face_count == 12
    assert mesh.is_watertight is True
    assert mesh.boundary_edge_count == 0
    assert mesh.non_manifold_edge_count == 0
    torch.testing.assert_close(
        mesh.bounds_world,
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
                     dtype=mesh.bounds_world.dtype, device=mesh.bounds_world.device),
    )
    assert mesh.source_path is not None and mesh.source_path.endswith("cube.obj")


def test_teapot_asset_loads_and_reports_mesh_stats():
    mesh = mw.Mesh.from_obj(ASSET_DIR / "teapot.obj")

    assert mesh.vertex_count == 206
    assert mesh.face_count == 392
    assert mesh.is_watertight is False
    assert mesh.boundary_edge_count == 0
    assert mesh.non_manifold_edge_count == 0
    assert mesh.inconsistent_edge_orientation_count > 0
    assert mesh.source_path is not None and mesh.source_path.endswith("teapot.obj")


def test_mesh_material_compilation_matches_box_for_closed_cube():
    mesh_scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    mesh_scene.add_structure(
        mw.Structure(
            name="mesh_cube",
            geometry=mw.Mesh.from_obj(ASSET_DIR / "cube.obj", fill_mode="solid"),
            material=mw.Material(eps_r=4.0),
        )
    )

    box_scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    box_scene.add_structure(
        mw.Structure(
            name="analytic_cube",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
            material=mw.Material(eps_r=4.0),
        )
    )

    mesh_prepared = _prepared_scene(mesh_scene)
    box_prepared = _prepared_scene(box_scene)

    assert torch.equal(mesh_prepared.permittivity != 1.0, box_prepared.permittivity != 1.0)
    assert torch.equal(mesh_prepared.permeability != 1.0, box_prepared.permeability != 1.0)


def test_teapot_auto_matches_solid_and_surface_is_thinner():
    domain = mw.Domain(bounds=((-1.5, 1.5), (-1.2, 1.2), (-1.2, 1.2)))
    grid = mw.GridSpec.uniform(0.12)
    material = mw.Material(eps_r=4.0)

    solid_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    solid_scene.add_structure(
        mw.Structure(
            name="teapot_solid",
            geometry=mw.Mesh.from_obj(ASSET_DIR / "teapot.obj", fill_mode="solid"),
            material=material,
        )
    )

    auto_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    auto_scene.add_structure(
        mw.Structure(
            name="teapot_auto",
            geometry=mw.Mesh.from_obj(ASSET_DIR / "teapot.obj", fill_mode="auto"),
            material=material,
        )
    )

    surface_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    surface_scene.add_structure(
        mw.Structure(
            name="teapot_surface",
            geometry=mw.Mesh.from_obj(ASSET_DIR / "teapot.obj", fill_mode="surface"),
            material=material,
        )
    )

    solid_prepared = _prepared_scene(solid_scene)
    auto_prepared = _prepared_scene(auto_scene)
    surface_prepared = _prepared_scene(surface_scene)

    solid_count = int((solid_prepared.permittivity != 1.0).sum().item())
    surface_count = int((surface_prepared.permittivity != 1.0).sum().item())

    assert solid_count > 0
    assert surface_count > 0
    assert solid_count > surface_count
    assert torch.equal(auto_prepared.permittivity, surface_prepared.permittivity)


def test_non_watertight_auto_mesh_matches_surface_mode():
    vertices, faces = _cube_vertices_faces()
    faces = faces.clone()
    faces[0] = faces[0, [0, 2, 1]]

    auto_mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="auto")
    surface_mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="surface")
    solid_mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")

    assert auto_mesh.is_watertight is False
    assert auto_mesh.inconsistent_edge_orientation_count > 0

    domain = mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)
    material = mw.Material(eps_r=4.0)

    auto_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    auto_scene.add_structure(mw.Structure(name="auto", geometry=auto_mesh, material=material))

    surface_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    surface_scene.add_structure(mw.Structure(name="surface", geometry=surface_mesh, material=material))

    solid_scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    solid_scene.add_structure(mw.Structure(name="solid", geometry=solid_mesh, material=material))

    auto_prepared = _prepared_scene(auto_scene)
    surface_prepared = _prepared_scene(surface_scene)
    solid_prepared = _prepared_scene(solid_scene)

    assert torch.equal(auto_prepared.permittivity, surface_prepared.permittivity)
    assert not torch.equal(auto_prepared.permittivity, solid_prepared.permittivity)


def test_mesh_topology_reports_degenerate_faces():
    vertices, faces = _cube_vertices_faces()
    faces = faces.clone()
    faces[0] = torch.tensor([0, 0, 1], dtype=faces.dtype)

    mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="auto")

    assert mesh.is_watertight is False
    assert mesh.degenerate_face_count > 0
