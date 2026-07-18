import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene

_MU0 = 4.0e-7 * np.pi


def test_lossy_metal_construction_and_surface_impedance():
    sigma = 5.8e7  # copper [S/m]
    metal = mw.LossyMetalMedium(conductivity=sigma, name="copper")
    assert metal.is_lossy_metal
    assert metal.conductivity == pytest.approx(sigma)

    frequency = 10.0e9
    omega = 2.0 * np.pi * frequency
    delta = np.sqrt(2.0 / (omega * _MU0 * sigma))
    assert metal.skin_depth(frequency) == pytest.approx(delta, rel=1e-12)

    z_s = metal.surface_impedance_at_freq(frequency)
    magnitude = np.sqrt(omega * _MU0 / (2.0 * sigma))
    assert z_s.real == pytest.approx(magnitude, rel=1e-12)
    assert z_s.imag == pytest.approx(-magnitude, rel=1e-12)
    # Leontovich relation |Z_s| = sqrt(2) / (sigma * delta).
    assert abs(z_s) == pytest.approx(np.sqrt(2.0) / (sigma * delta), rel=1e-12)

    with pytest.raises(ValueError):
        mw.LossyMetalMedium(conductivity=0.0)
    with pytest.raises(ValueError):
        metal.surface_impedance(0.0)


def _slab_scene(*, device="cpu", conductivity=5.8e7, side="high"):
    # Metal slab flush against the +x domain boundary spanning the full transverse
    # cross-section: a single axis-aligned face at normal incidence.
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
        device=device,
        structures=[
            mw.Structure(
                geometry=Box(
                    position=((0.3 if side == "high" else -0.3), 0.0, 0.0),
                    size=(0.4, 0.4, 0.4),
                ),
                material=mw.LossyMetalMedium(conductivity=conductivity),
            )
        ],
    )


def test_lossy_metal_compiles_to_order0_surface_layout():
    scene = prepare_scene(_slab_scene())
    layout = scene.compile_materials().get("surface_impedance")
    assert layout is not None and bool(layout)
    assert len(layout.faces) == 1
    face = layout.faces[0]
    # Slab is bounded along x and flush against the +x edge, so the metal occupies the
    # high side of the illuminated surface node.
    assert face.axis == 0
    assert face.metal_side == "high"
    assert face.full_plane is True
    metal = layout.metals[face.metal_index]
    # A narrowband LossyMetalMedium is realized as an order-0 pure-resistance surface.
    assert metal.conductivity == pytest.approx(5.8e7)
    assert scene.x_nodes64[face.surface_node] == pytest.approx(0.1)


def test_lossy_metal_low_side_surface_uses_upper_geometry_face():
    scene = prepare_scene(_slab_scene(side="low"))
    layout = scene.compile_materials()["surface_impedance"]
    assert len(layout.faces) == 1
    face = layout.faces[0]
    assert face.axis == 0
    assert face.metal_side == "low"
    assert scene.x_nodes64[face.surface_node] == pytest.approx(-0.1)


def _box_scene(box):
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
            device="cpu",
            structures=[mw.Structure(geometry=box, material=mw.LossyMetalMedium(conductivity=5.8e7))],
        )
    )


def test_lossy_metal_finite_block_exposes_all_faces():
    # A laterally finite block now compiles: every axis-aligned exposed face is a
    # surface-impedance plane (S1.1 replaced the single-plane restriction).
    scene = _box_scene(Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)))
    layout = scene.compile_materials()["surface_impedance"]
    assert len(layout.faces) == 6
    assert layout.total_area == pytest.approx(0.24)


def test_lossy_metal_mid_domain_double_sided_plate_compiles():
    # A slab with vacuum on both sides now compiles to a two-face double-sided plate.
    scene = _box_scene(Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.4, 0.4)))
    layout = scene.compile_materials()["surface_impedance"]
    assert len(layout.faces) == 2
    assert sorted(face.metal_side for face in layout.faces) == ["high", "low"]


def test_lossy_metal_multiple_slabs_compile():
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
            device="cpu",
            structures=[
                mw.Structure(geometry=Box(position=(0.3, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
                             material=mw.LossyMetalMedium(conductivity=5.8e7)),
                mw.Structure(geometry=Box(position=(-0.3, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
                             material=mw.LossyMetalMedium(conductivity=5.8e7)),
            ],
        )
    )
    layout = scene.compile_materials()["surface_impedance"]
    assert len(layout.metals) == 2
    # Each full-span slab flush against a domain boundary exposes one illuminated face.
    assert len(layout.faces) == 2


def test_lossy_metal_adjoint_is_rejected():
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    message = _unsupported_adjoint_medium(_slab_scene())
    assert message is not None
    assert "surface-impedance boundary" in message
    assert "not implemented yet" not in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_metal_simulation_prepare_configures_surface_impedance():
    solver = mw.Simulation.fdtd(_slab_scene(device="cuda"), frequencies=[1.0e9]).prepare().solver
    assert solver.surface_impedance_enabled
    state = solver._surface_impedance
    # One full-plane order-0 face -> two tangential-component writes, each a resistive
    # surface with R = sqrt(omega0*mu0/(2*sigma)) and no ADE state.
    writes = state["writes"]
    assert len(writes) == 2
    omega0 = 2.0 * np.pi * solver.source_frequency
    expected_r = float(np.sqrt(omega0 * solver.mu0 / (2.0 * 5.8e7)))
    for write in writes:
        assert write["surface_r"] == pytest.approx(expected_r, rel=1e-9)
        assert write.get("ade") is None
        assert write["full_plane"] is True
