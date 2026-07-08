import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


def test_pec_material_constructor_and_flag():
    metal = mw.Material.pec(name="metal")
    assert metal.is_pec is True
    assert mw.Material(eps_r=2.0).is_pec is False


def test_pec_material_rejects_dispersion_anisotropy_kerr_and_nondefault():
    with pytest.raises(ValueError):
        mw.Material(pec=True, eps_r=2.0)
    with pytest.raises(ValueError):
        mw.Material(pec=True, sigma_e=1.0)
    with pytest.raises(ValueError):
        mw.Material(pec=True, kerr_chi3=1e-20)
    with pytest.raises(ValueError):
        mw.Material(pec=True, debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1e-10),))
    with pytest.raises(ValueError):
        mw.Material(pec=True, epsilon_tensor=mw.DiagonalTensor3(2.0, 2.0, 2.0))


def test_compiler_emits_pec_occupancy_with_fractional_face_fill():
    # Domain [0,1]^3, grid 0.25 -> PEC box low face on the node x=0.25.
    domain = mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    grid = mw.GridSpec.uniform(0.25)
    scene = mw.Scene(domain=domain, grid=grid, device="cpu")
    scene.add_structure(
        mw.Structure(
            name="pec",
            geometry=mw.Box(position=(0.625, 0.5, 0.5), size=(0.75, 1.0, 1.0)),
            material=mw.Material.pec(),
        )
    )
    model = prepare_scene(scene).compile_materials()
    occupancy = model["pec_occupancy"]
    assert occupancy is not None
    assert model["pec_mode"] == "staircase"
    assert occupancy[3, 2, 2].item() > 0.9  # x = 0.75 interior
    assert occupancy[0, 2, 2].item() < 0.1  # x = 0.0 exterior
    assert abs(occupancy[1, 2, 2].item() - 0.5) < 0.05  # x = 0.25 node-aligned face


def test_non_pec_scene_has_none_pec_occupancy_and_unchanged_components():
    domain = mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))
    grid = mw.GridSpec.uniform(0.1)

    def build():
        scene = mw.Scene(domain=domain, grid=grid, device="cpu")
        scene.add_structure(
            mw.Structure(
                name="diel",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6)),
                material=mw.Material(eps_r=4.0),
            )
        )
        return scene

    model = prepare_scene(build()).compile_materials()
    assert model["pec_occupancy"] is None

    eps_ref, _ = prepare_scene(build()).compile_material_components()
    eps_now = model["eps_components"]
    for axis in ("x", "y", "z"):
        assert torch.equal(eps_ref[axis], eps_now[axis])


def test_fdfd_rejects_in_domain_pec_material():
    if not torch.cuda.is_available():
        pytest.skip("FDFD requires CUDA")
    domain = mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)))
    grid = mw.GridSpec.uniform(0.05)
    scene = mw.Scene(domain=domain, grid=grid, boundary=mw.BoundarySpec.pml(num_layers=6), device="cuda")
    scene.add_structure(
        mw.Structure(
            name="pec",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            material=mw.Material.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            width=0.03,
            polarization=(0, 0, 1),
            source_time=mw.CW(frequency=5e9, amplitude=1.0),
            name="src",
        )
    )
    with pytest.raises(NotImplementedError):
        mw.Simulation.fdfd(scene, frequency=5e9, solver=mw.GMRES(max_iter=10)).run()
