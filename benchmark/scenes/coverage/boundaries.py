"""Additional boundary-family coverage: PML variants, periodic/Bloch, symmetry planes, mixed faces."""

from __future__ import annotations

import numpy as np

import witwin.maxwell as mw
from benchmark.models import ScenarioDefinition
from benchmark.scenes._common import DX, HALF_SPAN, base_scene
from benchmark.scenes.planned import (
    GRATING_FREQUENCIES,
    GRATING_PULSE,
    PULSE,
    _C0,
    _make,
    _observed,
    _plane_scene,
)

# Physical uniform-grid step actually used by both runtimes on the base domain.
# Interfaces placed on multiples of this step land on primal nodes, so a slab is
# resolved by whole cells and the two solvers do not differ through subpixel
# averaging of a partially filled boundary cell.
STEP = 2.0 * HALF_SPAN / int(np.ceil(2.0 * HALF_SPAN / DX))


def _pml_slab_through_scene() -> mw.Scene:
    """Dielectric half-space that runs through the x/y/z PML regions.

    The interface sits on a primal node at z = 0, and the medium extends far past
    every physical face, so the lateral PMLs and the z-high PML all terminate a
    loaded (eps_r = 4) medium instead of vacuum.
    """
    scene = base_scene()
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 1.0), size=(4 * HALF_SPAN, 4 * HALF_SPAN, 2.0)),
            material=mw.Material(eps_r=4.0),
            name="loaded_half_space",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _periodic_slab_scene() -> mw.Scene:
    """Normal-incidence transmission through a uniform slab under periodic x/y faces."""
    reference = base_scene()
    boundary = mw.BoundarySpec.pml(num_layers=12).with_faces(
        x_low="periodic",
        x_high="periodic",
        y_low="periodic",
        y_high="periodic",
    )
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        subpixel_samples=reference.subpixel,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4 * HALF_SPAN, 4 * HALF_SPAN, 4 * STEP)),
            material=mw.Material(eps_r=3.0),
            name="periodic_slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _bloch_p_polarized_scene() -> mw.Scene:
    """Oblique Bloch incidence with the in-plane (p) polarization.

    The existing ``bloch_oblique`` case launches the out-of-plane (s) vector at
    theta = 20 deg / phi = 30 deg; this one launches the orthogonal in-plane vector
    at theta = 35 deg / phi = 60 deg. The Bloch wavevector is derived from the same
    k = 2 pi f d / c relation the campaign grating uses, so both runtimes phase the
    periodic faces identically.
    """
    theta = np.deg2rad(35.0)
    phi = np.deg2rad(60.0)
    direction = (
        float(np.sin(theta) * np.cos(phi)),
        float(np.sin(theta) * np.sin(phi)),
        float(np.cos(theta)),
    )
    polarization = (
        float(np.cos(theta) * np.cos(phi)),
        float(np.cos(theta) * np.sin(phi)),
        float(-np.sin(theta)),
    )
    bloch_kx = 2.0 * np.pi * GRATING_FREQUENCIES[0] * direction[0] / _C0
    bloch_ky = 2.0 * np.pi * GRATING_FREQUENCIES[0] * direction[1] / _C0
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=12,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector=(bloch_kx, bloch_ky, 0.0),
    )

    reference = base_scene()
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.52, 2 * HALF_SPAN, 0.12)),
            material=mw.Material(eps_r=4.0),
            name="periodic_bar",
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=GRATING_PULSE,
            injection=mw.TFSF.slab(axis="z", bounds=(-0.45, 0.45)),
            name="grating_tfsf",
        )
    )
    return _observed(
        scene,
        component="Ex",
        flux_positions=(-0.50, 0.30),
        frequencies=GRATING_FREQUENCIES,
    )


def _symmetry_scene(symmetry) -> mw.Scene:
    """Plane-wave scene carrying a real ``Scene(symmetry=...)`` declaration.

    An Ex-polarized wave travelling along +z has E = (Ex, 0, 0) and H = (0, Hy, 0)
    and is uniform in x and y. Every x-normal plane is therefore an exact PEC plane
    (n x E = 0, n . B = 0) and every y-normal plane an exact PMC plane
    (n x H = 0, n . D = 0). Because the field carries no transverse variation, the
    reduction is exact wherever the plane is placed: maxwell folds about the domain
    face while the Tidy3D export folds about the domain center, and both describe
    the identical physical problem. Any laterally structured scene would not share
    that property with the current export.
    """
    reference = base_scene()
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=reference.boundary,
        subpixel_samples=reference.subpixel,
        symmetry=symmetry,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4 * HALF_SPAN, 4 * HALF_SPAN, 4 * STEP)),
            material=mw.Material(eps_r=3.0),
            name="symmetric_slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene)


def _mixed_faces_scene() -> mw.Scene:
    """Per-face boundary mix: periodic x, PEC y, PML z.

    The wave is Ey-polarized so E is normal to the PEC walls and the uniform plane
    wave satisfies them exactly; the slab supplies the reflected/transmitted
    observable.
    """
    reference = base_scene()
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=8,
        x="periodic",
        y="pec",
        z="pml",
    )
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        subpixel_samples=reference.subpixel,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4 * HALF_SPAN, 4 * HALF_SPAN, 4 * STEP)),
            material=mw.Material(eps_r=4.0),
            name="mixed_slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ey", source_time=PULSE)
    )
    return _observed(scene, component="Ey")


def _asymmetric_faces_scene() -> mw.Scene:
    """Different low/high conditions on one axis: x-low PEC and x-high PML."""
    reference = base_scene()
    boundary = mw.BoundarySpec.pml(num_layers=8).with_faces(
        x_low="pec",
        x_high="pml",
    )
    scene = mw.Scene(
        domain=reference.domain,
        grid=reference.grid,
        boundary=boundary,
        subpixel_samples=reference.subpixel,
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0),
                size=(4 * HALF_SPAN, 4 * HALF_SPAN, 4 * STEP),
            ),
            material=mw.Material(eps_r=3.0),
            name="asymmetric_boundary_slab",
        )
    )
    scene.add_source(
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=PULSE)
    )
    return _observed(scene, component="Ex")


BOUNDARY_COVERAGE_SCENARIOS = (
    _make(
        "pml_thin",
        "boundaries",
        "6-layer PML vacuum absorption",
        lambda: _plane_scene(boundary=mw.BoundarySpec.pml(num_layers=6)),
    ),
    _make(
        "pml_slab_through",
        "boundaries",
        "material-loaded PML with a dielectric half-space",
        _pml_slab_through_scene,
    ),
    _make(
        "periodic_slab",
        "boundaries",
        "periodic faces at normal incidence on a uniform slab",
        _periodic_slab_scene,
    ),
    _make(
        "bloch_oblique_te",
        "boundaries",
        "Bloch faces at 35 deg with the in-plane polarization",
        _bloch_p_polarized_scene,
        frequencies=GRATING_FREQUENCIES,
    ),
    _make(
        "symmetry_pec_center",
        "boundaries",
        "PEC symmetry plane on the antisymmetric axis",
        lambda: _symmetry_scene(("PEC", None, None)),
    ),
    _make(
        "symmetry_pmc_center",
        "boundaries",
        "PMC symmetry plane on the symmetric axis",
        lambda: _symmetry_scene((None, "PMC", None)),
    ),
    _make(
        "mixed_faces",
        "boundaries",
        "per-face periodic/PEC/PML mix",
        _mixed_faces_scene,
        component="Ey",
    ),
    _make(
        "asymmetric_boundary_faces",
        "boundaries",
        "x-low PEC and x-high PML with a transmitted slab field",
        _asymmetric_faces_scene,
    ),
)
