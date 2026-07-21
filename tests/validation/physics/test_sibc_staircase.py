"""Staircased (voxelized) surface-impedance boundary for curved conductors.

The all-orientation generalization staircases any voxelized good-conductor: a node belongs
to the metal when its center is inside the geometry, and every axis-aligned voxel face on
the metal/vacuum boundary becomes a surface-impedance face carrying the Leontovich relation
``E_t = R * (n_hat x H)``. This replaces the axis-aligned-Box-only layout with an occupancy
extraction that handles curved surfaces (cylinder/sphere), all six orientations, and mixed
orientations in one scene. True oblique/conformal (non-staircase) SIBC stays fail-closed.

The physics anchor is a flat plate driven through the voxel path: a large-radius cylinder
whose axis is x fills a slab normal to x spanning the whole transverse cross-section, i.e.
a flat plate assembled entirely from staircased voxel faces. Its reflection must reproduce
the analytic Leontovich value the resolved good conductor converges to, confirming the
staircase write applies the surface impedance correctly (not merely a PEC termination).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Cylinder
from witwin.maxwell.compiler.materials import compile_surface_impedance_layout
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.scene import prepare_scene

_MU0 = 4.0e-7 * np.pi
_EPS0 = 8.8541878128e-12
_C = 299792458.0
_SIGMA = 50.0


def _analytic_gamma(sigma, frequency):
    omega = 2.0 * np.pi * frequency
    eta0 = np.sqrt(_MU0 / _EPS0)
    r = np.sqrt(omega * _MU0 / (2.0 * sigma))
    z_s = r + 1j * r
    return abs((z_s - eta0) / (z_s + eta0))


def _field(result, component):
    value = result.field(component)
    value = value["data"] if isinstance(value, dict) else value
    return value.detach().cpu().numpy()


# --- staircase physics: a flat plate built from voxel faces matches analytic -----------


def _run_voxel_slab(kind, frequency):
    """A flat plate normal to x, assembled from staircased voxel faces via a big cylinder."""
    dx = (_C / frequency) / 40.0
    trans = 4.0 * dx
    material = mw.LossyMetalMedium(conductivity=_SIGMA) if kind == "sibc" else mw.Material.pec()
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-trans / 2, trans / 2), (-trans / 2, trans / 2))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.3, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * dx,
                source_time=mw.CW(frequency=frequency, amplitude=40.0),
                name="s",
            )
        ],
        structures=[
            # axis=x, radius >> transverse extent -> the curved side never enters the domain,
            # so the occupancy is a flat slab spanning the full cross-section, flush against +x.
            mw.Structure(
                geometry=Cylinder(radius=5.0, height=0.8, axis="x", position=(0.5, 0.0, 0.0)),
                material=material,
            )
        ],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=16, transient_cycles=24),
        full_field_dft=True,
    ).run()
    ez = _field(result, "Ez")
    assert np.isfinite(ez).all(), "staircased slab diverged"
    line = np.abs(ez).reshape(ez.shape[0], -1).mean(axis=1)
    xs = result.solver.scene.x_nodes64
    v = line[(xs >= -0.28) & (xs <= 0.08)]
    return float((v.max() - v.min()) / (v.max() + v.min()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_staircased_slab_matches_analytic_leontovich_at_three_frequencies():
    """A flat plate built from voxel faces reflects the analytic Leontovich value (<5%)."""
    for frequency in (1.0e9, 2.0e9, 3.0e9):
        gamma = _run_voxel_slab("sibc", frequency)
        analytic = _analytic_gamma(_SIGMA, frequency)
        rel = abs(gamma - analytic) / analytic
        assert rel < 0.05, (
            f"f={frequency/1e9:.0f} GHz: staircased |Gamma|={gamma:.4f} vs analytic "
            f"{analytic:.4f} (rel {rel:.3f})"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_staircased_slab_absorbs_more_than_pec():
    """The staircased good-conductor slab reflects measurably less than a PEC slab."""
    gamma_sibc = _run_voxel_slab("sibc", 2.0e9)
    gamma_pec = _run_voxel_slab("pec", 2.0e9)
    assert gamma_pec > 0.98
    assert gamma_sibc < gamma_pec - 0.02


# --- occupancy layout: all orientations, masked faces ----------------------------------


def _cylinder_scene(axis="z", radius=0.08, material=None):
    material = material or mw.LossyMetalMedium(conductivity=_SIGMA)
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.26, 0.26), (-0.26, 0.26), (-0.26, 0.26))),
            grid=mw.GridSpec.uniform((_C / 2.0e9) / 24.0),
            boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(
                    geometry=Cylinder(radius=radius, height=0.3, axis=axis, position=(0.0, 0.0, 0.0)),
                    material=material,
                )
            ],
        )
    )


def test_voxel_cylinder_enumerates_all_orientations_with_masks():
    """A staircased cylinder exposes faces on all three axes, each carrying a boolean mask."""
    layout = compile_surface_impedance_layout(_cylinder_scene(axis="z"))
    assert len(layout.metals) == 1
    assert layout.metals[0].interior_node_mask is not None
    assert len(layout.faces) > 6  # a curved surface has many staircase steps per orientation
    assert sorted({face.axis for face in layout.faces}) == [0, 1, 2]
    assert all(face.transverse_mask is not None for face in layout.faces)
    assert all(not face.full_plane for face in layout.faces)
    assert layout.total_area > 0.0


def test_generic_rational_surface_on_curved_geometry_fails_closed():
    """A generic rational surface on a curved (non-Box) conductor fails closed with a phase.

    The staircased voxel path wires only the narrowband good-conductor (order-0 resistance)
    surface; the per-edge rational ADE stays Box-only.
    """
    frequencies = torch.logspace(9.0, np.log10(40.0e9), 64, dtype=torch.float64)
    admittance = (frequencies.to(torch.complex128) * 0.0 + (1.0 + 0.0j) * 1.0e-3)
    model = RationalSurfaceImpedance.fit(frequencies, admittance, order=4, band=(1.0e9, 40.0e9))
    medium = SurfaceImpedanceMedium(impedance=model, name="rough")
    with pytest.raises(NotImplementedError) as info:
        compile_surface_impedance_layout(_cylinder_scene(axis="z", material=medium))
    message = str(info.value)
    assert "Phase" in message
    assert not any(
        phrase in message.lower() for phrase in ("not implemented yet", "not supported yet", "in v1")
    )


# --- staircase stability + orientation equivalence -------------------------------------


def _run_cylinder(axis, frequency=2.0e9, steps=4000):
    dx = (_C / frequency) / 22.0
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.26, 0.26), (-0.26, 0.26), (-0.26, 0.26))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.17, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * dx,
                source_time=mw.CW(frequency=frequency, amplitude=40.0),
                name="s",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Cylinder(radius=0.08, height=0.34, axis=axis, position=(0.0, 0.0, 0.0)),
                material=mw.LossyMetalMedium(conductivity=_SIGMA),
            )
        ],
    )
    return mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig(time_steps=steps),
        full_field_dft=True,
    ).run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_staircased_cylinder_is_stable_over_a_long_run():
    """The stateless Leontovich update stays finite/bounded on a curved staircase surface."""
    result = _run_cylinder("z", steps=6000)
    ez = _field(result, "Ez")
    assert np.isfinite(ez).all(), "staircased cylinder diverged"
    assert np.abs(ez).max() < 1.0e12, "staircased cylinder grew unbounded"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_staircased_orientation_equivalence_z_to_x():
    """A cylinder along z and the same cylinder along x agree under the axis permutation.

    The source is symmetric under the cyclic map base(x, y, z) -> world(y, z, x): a
    base-z-axis cylinder maps to a world-x-axis cylinder, the base-x source at -0.17 maps
    to a world-y source, and the base-z polarization maps to world-x. The Yee solver is
    exactly permutation-covariant (the Box orientation test agrees to ~1e-7); the residual
    here is set by the hard-threshold staircase, where a node whose center lies within
    float round-off of the curved surface can flip between orientations and change one
    voxel face. That is a staircase-discretization effect of order 1e-4, still two-plus
    orders below any face-normal-sign or enumeration bug (which scatters ~1e-2 or diverges).
    """
    perm = (1, 2, 0)  # base axis -> world axis

    def permute(vector):
        out = [0.0, 0.0, 0.0]
        for base_axis in range(3):
            out[perm[base_axis]] = vector[base_axis]
        return tuple(out)

    frequency, dx = 2.0e9, (_C / 2.0e9) / 22.0

    def build(cyl_axis, source_pos, polarization):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.26, 0.26), (-0.26, 0.26), (-0.26, 0.26))),
            grid=mw.GridSpec.uniform(dx),
            boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
            device="cuda",
            sources=[
                mw.PointDipole(
                    position=source_pos,
                    polarization=polarization,
                    width=2.0 * dx,
                    source_time=mw.CW(frequency=frequency, amplitude=40.0),
                    name="s",
                )
            ],
            structures=[
                mw.Structure(
                    geometry=Cylinder(radius=0.08, height=0.34, axis=cyl_axis, position=(0.0, 0.0, 0.0)),
                    material=mw.LossyMetalMedium(conductivity=_SIGMA),
                )
            ],
        )
        return mw.Simulation.fdtd(
            scene,
            frequencies=[frequency],
            run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=12),
            full_field_dft=True,
        ).run()

    reference = build("z", (-0.17, 0.0, 0.0), (0.0, 0.0, 1.0))
    permuted = build("x", permute((-0.17, 0.0, 0.0)), permute((0.0, 0.0, 1.0)))
    components = ("Ex", "Ey", "Ez")
    worst = 0.0
    for base_axis, base_component in enumerate(components):
        ref = np.abs(_field(reference, base_component))
        world = np.abs(_field(permuted, components[perm[base_axis]]))
        mapped = np.transpose(world, perm)
        assert mapped.shape == ref.shape
        scale = np.linalg.norm(ref)
        if scale == 0.0:
            continue
        worst = max(worst, np.linalg.norm(mapped - ref) / scale)
    assert worst < 1.0e-3, f"staircase orientation equivalence residual {worst:.3e}"
