"""Dump compiled grid coordinates for a fixed set of scenes (fresh-interpreter helper).

Run as ``python grid_identity_dump.py OUT_DIR`` with ``PYTHONPATH`` selecting the
``witwin.maxwell`` tree under test. Every scene uses only grid APIs that predate
opt-in corner / layer-bounds refinement, so a baseline tree and the current tree
must write byte-identical node arrays. ``test_autogrid_default_identity.py``
drives this script against a ``git archive`` of the baseline commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.meshing import resolve_auto_grid
from witwin.maxwell.scene import prepare_scene

_FREQ = 1.0e9
_WIDTH, _HEIGHT, _FOIL = 0.010, 0.005, 35e-6


def _dipole(position=(0.0, 0.0, 0.0), frequency=_FREQ):
    return mw.PointDipole(
        position=position,
        polarization="Ez",
        source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5 * frequency),
    )


def _dielectric_scene(grid):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12), (-0.1, 0.1), (-0.08, 0.09))),
        grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cpu",
        structures=[
            mw.Structure(mw.Box(position=(0.01, -0.02, 0.0), size=(0.07, 0.05, 0.013)), mw.Material(eps_r=4.0)),
            mw.Structure(mw.Sphere(position=(-0.05, 0.03, 0.01), radius=0.021), mw.Material(eps_r=9.0)),
        ],
        sources=[_dipole((0.0, 0.0, 0.04))],
    )


def _printed_custom_nodes(dx=0.0025, half=(0.05, 0.06, 0.05), radius=0.025, outer=0.005):
    """Printed-line style explicit nodes: fine core, coarse shell, exact layer planes."""
    nodes = []
    for index, h in enumerate(half):
        if index == 0:
            nodes.append(np.arange(-round(h / dx), round(h / dx) + 1, dtype=np.float64) * dx)
            continue
        fine = np.arange(-round(radius / dx), round(radius / dx) + 1, dtype=np.float64) * dx
        coarse = np.arange(-round(h / outer), round(h / outer) + 1, dtype=np.float64) * outer
        nodes.append(np.r_[coarse[coarse < -radius - 1e-12], fine, coarse[coarse > radius + 1e-12]])
    nodes[2] = np.unique(np.r_[nodes[2], -_HEIGHT, 0.0, _FOIL])
    return nodes


def _pcb_structures(extent):
    return [
        mw.Structure(
            mw.Box(position=(0.0, 0.0, -_HEIGHT / 2), size=(2 * extent, 2 * extent, _HEIGHT)),
            mw.Material(eps_r=4.4, sigma_e=1e-3), name="substrate",
        ),
        mw.Structure(
            mw.Box(position=(0.0, 0.0, -_HEIGHT - 0.0004), size=(2 * extent, 2 * extent, 0.0008)),
            mw.Material(eps_r=3.6), name="prepreg",
        ),
        mw.Structure(
            mw.PlanarPEC.rectangle(((-extent, extent), (-extent, extent)), coordinate=-_HEIGHT,
                                   physical_thickness=_FOIL),
            mw.Material.pec(), name="ground",
        ),
        mw.Structure(
            mw.Box(position=(0.0, 0.0, _FOIL / 2), size=(2 * extent, _WIDTH, _FOIL)),
            mw.Material(eps_r=1.0, sigma_e=5.8e7), name="trace",
        ),
        mw.Structure(
            mw.PlanarPEC.rectangle(((-0.004, 0.004), (_WIDTH / 2, _WIDTH / 2 + 0.003)),
                                   coordinate=0.0, physical_thickness=_FOIL),
            mw.Material.pec(), name="stub",
        ),
    ]


def scenes():
    out = {}
    out["uniform"] = _dielectric_scene(mw.GridSpec.uniform(0.006))
    out["anisotropic"] = _dielectric_scene(mw.GridSpec.anisotropic(0.006, 0.0075, 0.005))
    out["auto_default"] = _dielectric_scene(mw.GridSpec.auto())
    out["auto_options"] = _dielectric_scene(
        mw.GridSpec.auto(
            min_steps_per_wavelength=14,
            wavelength=0.25,
            max_ratio=1.25,
            override_structures=(
                mw.MeshOverrideStructure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.03, 0.04, 0.02)),
                                         dl=(0.002, 0.003, 0.001)),
            ),
            layer_refinement=mw.LayerRefinementSpec(min_cells=3),
        )
    )
    out["auto_layer_axes"] = _dielectric_scene(
        mw.GridSpec.auto(layer_refinement=mw.LayerRefinementSpec(min_cells=2, axes=("z",)))
    )

    custom = _printed_custom_nodes()
    bounds = tuple((float(a[0]), float(a[-1])) for a in custom)
    out["printed_custom"] = mw.Scene(
        domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.custom(*custom),
        boundary=mw.BoundarySpec.pml(num_layers=6), device="cpu",
        structures=_pcb_structures(0.2), sources=[_dipole((0.0, 0.0, -0.002))],
    )

    pcb_bounds = ((-0.04, 0.04), (-0.03, 0.03), (-0.012, 0.012))
    pcb_auto = mw.Scene(
        domain=mw.Domain(bounds=pcb_bounds),
        grid=mw.GridSpec.auto(
            min_steps_per_wavelength=20,
            max_ratio=1.3,
            override_structures=(
                mw.MeshOverrideStructure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.08, 0.016, 0.002)),
                                         dl=(0.002, 0.0005, 0.0002)),
            ),
            layer_refinement=mw.LayerRefinementSpec(min_cells=2, axes=("z",)),
        ),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, z_low="pec"),
        device="cpu",
        structures=_pcb_structures(0.2),
        sources=[_dipole((0.0, 0.0, -0.002), frequency=2.0e9)],
    )
    out["pcb_auto_layered"] = pcb_auto
    # Autogrid plus explicit nodes: resolve, insert port / contour planes, go custom.
    resolved = [np.asarray(a) for a in resolve_auto_grid(pcb_auto)]
    explicit = ([-0.025, -0.0249, 0.025, 0.0251], [-0.0051, 0.0051], [-0.0025, 0.0007])
    merged = [np.union1d(a, np.asarray(e, dtype=np.float64)) for a, e in zip(resolved, explicit)]
    out["pcb_auto_plus_explicit"] = pcb_auto.clone(grid=mw.GridSpec.custom(*merged))

    region_scene = _dielectric_scene(mw.GridSpec.auto(wavelength=0.3, max_ratio=1.3))
    region_scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.03, 0.02)),
            density=torch.full((4, 3, 2), 0.5),
            eps_bounds=(1.0, 6.0),
        )
    )
    out["auto_material_region"] = region_scene
    return out


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, scene in scenes().items():
        prepared = prepare_scene(scene, lazy_meshgrid=True)
        arrays = {f"{axis}_nodes64": getattr(prepared, f"{axis}_nodes64") for axis in "xyz"}
        np.savez(out / f"{name}.npz", **arrays)
        torch.save({axis: getattr(prepared, axis).cpu() for axis in "xyz"}, out / f"{name}.pt")
        print(name, *(arrays[f"{a}_nodes64"].size for a in "xyz"))


if __name__ == "__main__":
    main(sys.argv[1])
