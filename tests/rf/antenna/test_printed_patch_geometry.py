"""Physical geometry and feed remain fixed while independently refining a grid."""
import numpy as np
import pytest
import torch
from benchmark.scenes.antenna.printed_patch import printed_patch_scene
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.ports import compile_port_geometry


def test_declared_si_probe_faces_and_feed_quadrature_keep_float64_precision():
    scene = printed_patch_scene(device="cpu")
    probe = next(item.geometry for item in scene.structures if item.name == "probe")
    assert probe.position.dtype == probe.size.dtype == torch.float64
    assert torch.equal(probe.position-probe.size/2,
                       torch.tensor((-.006, -.001, .001), dtype=torch.float64))
    assert torch.equal(probe.position+probe.size/2,
                       torch.tensor((-.004, .001, .003), dtype=torch.float64))
    feed = scene.ports[0]
    assert feed.voltage_aperture.position.dtype == torch.float64
    compiled = compile_port_geometry(prepare_scene(scene), feed, device="cpu")
    assert abs(float(compiled.voltage_weights.sum())-.001) < 1e-17


@pytest.mark.parametrize("lateral,cells", [(.001, 3), (.001, 6), (.001, 12), (.0005, 12), (.00025, 12)])
def test_refining_printed_patch_preserves_physical_geometry_aperture_and_reference(lateral, cells):
    reference = printed_patch_scene(device="cpu")
    scene = printed_patch_scene(lateral_dl=lateral, substrate_cells=cells, device="cpu")
    assert scene.domain.bounds == reference.domain.bounds
    for actual, fixed in zip(scene.structures, reference.structures):
        assert actual.name == fixed.name
        for attribute in ("position", "size", "vertices"):
            if hasattr(actual.geometry, attribute):
                assert torch.equal(getattr(actual.geometry, attribute), getattr(fixed.geometry, attribute))
        if hasattr(actual.geometry, "coordinate"):
            assert actual.geometry.coordinate == fixed.geometry.coordinate
            assert actual.geometry.physical_thickness == 35e-6
    a, b = scene.ports[0], reference.ports[0]
    assert a.positive == b.positive and a.negative == b.negative
    for box in ("current_surface", "voltage_aperture"):
        assert torch.equal(getattr(a, box).position, getattr(b, box).position)
        assert torch.equal(getattr(a, box).size, getattr(b, box).size)
    prepared = prepare_scene(scene)
    port = compile_port_geometry(prepared, a, device="cpu")
    assert len(port.voltage_indices) >= 9
    assert np.isclose(float(port.voltage_weights.sum()), .001, rtol=1e-6)


def test_edge_focused_grid_preserves_geometry_and_uses_fewer_air_interior_cells():
    uniform=printed_patch_scene(lateral_dl=.000125,substrate_cells=12,device="cpu")
    local=printed_patch_scene(lateral_dl=.0005,substrate_cells=12,edge_dl=.000125,device="cpu")
    for a,b in zip(local.structures,uniform.structures):
        for name in ("position","size","vertices"):
            if hasattr(a.geometry,name):
                assert torch.equal(getattr(a.geometry,name),getattr(b.geometry,name))
    u,l=prepare_scene(uniform),prepare_scene(local)
    assert l.Nx*l.Ny < .6*u.Nx*u.Ny
    assert np.isclose(min(np.diff(l.x_nodes64)),.000125,rtol=1e-10)
    compile_port_geometry(l,local.ports[0],device="cpu")


def test_finite_copper_control_preserves_physical_support_and_ideal_probe_contact():
    reference=printed_patch_scene(device="cpu")
    copper=printed_patch_scene(device="cpu",metal="copper",dielectric_sigma=.001)
    assert copper.domain.bounds==reference.domain.bounds
    assert copper.ports[0].positive==reference.ports[0].positive
    for name in ("ground","patch"):
        ideal=next(s for s in reference.structures if s.name==name)
        finite=next(s for s in copper.structures if s.name==name)
        vertices=ideal.geometry.vertices
        lo=torch.cat((vertices.amin(0), vertices.new_tensor([ideal.geometry.coordinate])))
        hi=torch.cat((vertices.amax(0), vertices.new_tensor([ideal.geometry.coordinate])))
        center=finite.geometry.position
        size=finite.geometry.size
        assert torch.allclose(center-size/2,lo.to(center))
        assert torch.allclose(center+size/2,hi.to(center))
        assert finite.material.thickness==35e-6
        assert finite.material.conductivity==5.8e7
    probe=next(s for s in copper.structures if s.name=="probe")
    assert probe.material.is_pec
    substrate=next(s for s in copper.structures if s.name=="substrate")
    assert float(substrate.material.sigma_e)==.001
