import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.mass_density import (
    BACKGROUND_TISSUE_ID,
    compile_mass_density,
)
from witwin.maxwell.scene import prepare_scene


def test_material_mass_density_validation_and_flags():
    tissue = mw.Material(eps_r=41.4, sigma_e=0.87, mass_density=1100.0, name="skin")
    assert tissue.has_mass_density
    assert tissue.mass_density == 1100.0
    assert tissue.is_electrically_lossy

    lossless = mw.Material(eps_r=2.0)
    assert not lossless.has_mass_density
    assert not lossless.is_electrically_lossy

    grid = torch.full((4, 4, 4), 900.0)
    assert torch.is_tensor(mw.Material(mass_density=grid).mass_density)

    with pytest.raises(ValueError, match="strictly positive"):
        mw.Material(mass_density=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        mw.Material(mass_density=-5.0)
    with pytest.raises(ValueError, match="strictly positive"):
        mw.Material(mass_density=torch.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="3D tensor"):
        mw.Material(mass_density=torch.ones(4))


def _scene(*materials_and_boxes, device="cpu"):
    structures = tuple(
        mw.Structure(name=f"s{i}", geometry=box, material=material)
        for i, (material, box) in enumerate(materials_and_boxes)
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        structures=structures,
        device=device,
    )


def test_compile_mass_density_fills_effective_density_and_ids():
    scene = _scene(
        (
            mw.Material(sigma_e=0.5, mass_density=1000.0, name="tissue"),
            mw.Box(position=(0.5, 0.5, 0.5), size=(4.0, 4.0, 4.0)),
        )
    )
    prepared = prepare_scene(scene)
    mass = compile_mass_density(prepared)

    assert mass.shape == (prepared.Nx, prepared.Ny, prepared.Nz)
    interior = mass.occupancy > 0.999
    assert bool(torch.any(interior))
    # Effective density equals the true density where fully occupied.
    torch.testing.assert_close(
        mass.rho_cell[interior],
        torch.full_like(mass.rho_cell[interior], 1000.0),
    )
    assert torch.all(mass.tissue_id[interior] == 0)
    assert mass.tissue_names[0] == "tissue"
    assert torch.all(mass.cell_volume > 0)


def test_compile_mass_density_excludes_materials_without_density():
    scene = _scene(
        (
            mw.Material(eps_r=2.0),  # no mass_density -> excluded
            mw.Box(position=(0.5, 0.5, 0.5), size=(4.0, 4.0, 4.0)),
        )
    )
    prepared = prepare_scene(scene)
    mass = compile_mass_density(prepared)
    assert torch.all(mass.rho_cell == 0)
    assert torch.all(mass.tissue_id == BACKGROUND_TISSUE_ID)
    assert dict(mass.tissue_names) == {}


def test_compile_mass_density_priority_overwrite_two_tissues():
    outer = mw.Material(mass_density=1000.0, name="outer")
    inner = mw.Material(mass_density=1800.0, name="bone")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="outer",
                geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(0.8, 0.8, 0.8)),
                material=outer,
                priority=0,
            ),
            mw.Structure(
                name="bone",
                geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(0.3, 0.3, 0.3)),
                material=inner,
                priority=1,
            ),
        ),
        device="cpu",
    )
    prepared = prepare_scene(scene)
    mass = compile_mass_density(prepared)
    center = (prepared.Nx // 2, prepared.Ny // 2, prepared.Nz // 2)
    assert int(mass.tissue_id[center]) == 1
    assert float(mass.rho_cell[center]) == pytest.approx(1800.0, rel=1e-5)
    assert set(mass.tissue_names.values()) == {"outer", "bone"}
