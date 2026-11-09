"""PECFrame: PlanarPEC side walls around a modal injection aperture."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.coax_thru import coax_thru_scene

DX = 0.01


def _framed(device, length=2):
    scene = coax_thru_scene(dx=DX, device=device)
    for port in scene.ports:
        for structure in mw.PECFrame(length).structures(port, scene):
            scene.add_structure(structure)
    return scene


def test_pec_frame_builds_four_planar_walls_toward_the_injection_direction():
    scene = _framed("cpu")
    frames = [s for s in scene.structures if "pec_frame" in s.tags]
    assert len(frames) == 8
    left = {s.name: s for s in frames if s.metadata["pec_frame"]["port"] == "left"}
    assert set(left) == {f"left::pec_frame::{axis}{side}" for axis in "yz" for side in "-+"}
    wall = left["left::pec_frame::y-"].geometry
    assert isinstance(wall, mw.PlanarPEC) and wall.normal_axis == "y"
    assert wall.coordinate == pytest.approx(-0.20)
    vertices = wall.vertices.numpy()  # y-normal plane: (x, z) coordinates
    assert vertices[:, 0].min() == pytest.approx(-0.02) and vertices[:, 0].max() == pytest.approx(0.0)
    assert vertices[:, 1].min() == pytest.approx(-0.20) and vertices[:, 1].max() == pytest.approx(0.20)
    right = next(s for s in frames if s.name == "right::pec_frame::z+").geometry
    xs = right.vertices.numpy()[:, 0]
    assert xs.min() == pytest.approx(0.0) and xs.max() == pytest.approx(0.02)
    assert all(s.material.is_pec and s.metadata["pec_frame"]["length"] == 2 for s in frames)


@pytest.mark.parametrize("length, error", [(0, ValueError), (1.5, TypeError), (True, TypeError)])
def test_pec_frame_rejects_invalid_lengths(length, error):
    with pytest.raises(error):
        mw.PECFrame(length)


def test_pec_frame_rejects_non_aperture_targets_and_frames_leaving_the_grid():
    scene = coax_thru_scene(dx=DX, device="cpu")
    with pytest.raises(TypeError, match="WavePort, ModePort or ModeSource"):
        mw.PECFrame().structures(mw.Box(position=(0, 0, 0), size=(1, 1, 1)), scene)
    with pytest.raises(ValueError, match="leaves the grid"):
        mw.PECFrame(500).structures(scene.ports[0], scene)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="PEC frame FDTD checks require CUDA")


@cuda
def test_pec_frame_shorts_exactly_its_wall_edges_on_the_yee_grid():
    def solver(scene):
        # Port-free clone: the frame is ordinary scene geometry once materialized.
        return mw.Simulation.fdtd(scene.clone(ports=()), frequencies=(1.0e9,),
                                  run_time=mw.TimeConfig(time_steps=10)).prepare().solver

    plain = solver(coax_thru_scene(dx=DX, device="cuda"))
    framed = solver(_framed("cuda"))
    scene = framed.scene
    x, y, z = scene.x_nodes64, scene.y_nodes64, scene.z_nodes64
    ix = np.flatnonzero((x >= -0.02 - 1e-9) & (x <= 0.0 + 1e-9))
    jy = int(np.argmin(abs(y + 0.20)))
    kz = np.flatnonzero(abs(z) <= 0.20 + 1e-9)
    ex = framed.cex_curl[ix[0]:ix[-1], jy, kz[0]:kz[-1] + 1]
    ez = framed.cez_curl[ix[0]:ix[-1] + 1, jy, kz[0]:kz[-1]]
    assert torch.all(ex == 0) and torch.all(ez == 0)
    changed = framed.cex_curl != plain.cex_curl
    # Only edges lying on frame walls change; the coaxial interior is untouched.
    assert torch.all(plain.cex_curl[changed] > 0) and not torch.any(framed.cex_curl[changed] > 0)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    interior = torch.as_tensor(np.hypot(yy, zz) < 0.16, device=framed.cex_curl.device)
    assert torch.equal(framed.cex_curl[:, interior], plain.cex_curl[:, interior])


@cuda
def test_pec_frame_preserves_the_shielded_coax_thru_response():
    def transmission(scene):
        result = mw.Simulation.fdtd(scene, frequencies=(1.0e9,), excitations=mw.PortSweep(),
                                    run_time=mw.TimeConfig(time_steps=1500),
                                    spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False).run()
        return result.network.s[:, 1, 0].cpu().numpy()

    plain = transmission(coax_thru_scene(dx=DX, device="cuda"))
    framed = transmission(_framed("cuda"))
    # The frame walls sit at the shielded aperture edge: |S21| changes by < 1e-3.
    assert abs(plain[0]) > 0.9
    assert np.max(np.abs(framed - plain)) < 1.0e-3
