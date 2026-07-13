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


def test_lossy_metal_compiles_to_sibc_descriptor():
    scene = prepare_scene(_slab_scene())
    model = scene.compile_materials()
    descriptor = model.get("sibc")
    assert descriptor is not None
    # Slab is bounded along x and flush against the +x edge, so the metal occupies
    # the high side of the illuminated surface node.
    assert descriptor["axis"] == 0
    assert descriptor["metal_side"] == "high"
    assert descriptor["conductivity"] == pytest.approx(5.8e7)
    assert scene.x_nodes64[descriptor["surface_node"]] == pytest.approx(0.1)


def test_lossy_metal_low_side_surface_uses_upper_geometry_face():
    scene = prepare_scene(_slab_scene(side="low"))
    descriptor = scene.compile_materials()["sibc"]

    assert descriptor["axis"] == 0
    assert descriptor["metal_side"] == "low"
    assert scene.x_nodes64[descriptor["surface_node"]] == pytest.approx(-0.1)


def test_lossy_metal_finite_block_raises_physics_guard():
    # A laterally finite block exposes edge faces the scalar normal-incidence Zs
    # does not model; the guard must be physics-worded, not "not implemented yet".
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(
                    geometry=Box(position=(0.3, 0.0, 0.0), size=(0.4, 0.2, 0.2)),
                    material=mw.LossyMetalMedium(conductivity=5.8e7),
                )
            ],
        )
    )
    with pytest.raises(NotImplementedError, match="transverse cross-section"):
        scene.compile_materials()
    with pytest.raises(NotImplementedError) as info:
        scene.compile_materials()
    assert "not implemented yet" not in str(info.value)


def test_lossy_metal_mid_domain_slab_raises_physics_guard():
    # A slab with vacuum on both sides exposes two faces; v1 supports a single
    # illuminated face flush against a domain boundary.
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.faces(default="periodic", num_layers=4, strength=1.0, x=("pml", "pml")),
            device="cpu",
            structures=[
                mw.Structure(
                    geometry=Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.4, 0.4)),
                    material=mw.LossyMetalMedium(conductivity=5.8e7),
                )
            ],
        )
    )
    with pytest.raises(NotImplementedError, match="single illuminated face"):
        scene.compile_materials()


def test_lossy_metal_multiple_slabs_raise_physics_guard():
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
    with pytest.raises(NotImplementedError, match="single metal slab"):
        scene.compile_materials()


def test_lossy_metal_adjoint_is_rejected():
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    message = _unsupported_adjoint_medium(_slab_scene())
    assert message is not None
    assert "LossyMetalMedium" in message
    assert "not implemented yet" not in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_metal_simulation_prepare_configures_sibc():
    solver = mw.Simulation.fdtd(_slab_scene(device="cuda"), frequencies=[1.0e9]).prepare().solver
    assert solver.sibc_enabled
    state = solver._sibc
    assert state["surface_r"] > 0.0
    # Narrowband series R-L: Ls = R / omega0 (surface inductance).
    omega0 = 2.0 * np.pi * solver.source_frequency
    assert state["surface_l"] == pytest.approx(state["surface_r"] / omega0, rel=1e-9)
    # Two tangential faces are configured on the illuminated surface.
    assert len(state["faces"]) == 2
