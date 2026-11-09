"""Multi-sample coverage fill and scalar-interface Kottke tensors (CPU).

The FDTD edge coefficients of a z-layered slab are compared with closed forms:
an off-grid face gives the exact volume fraction (arithmetic) or the exact
series mean (polarized E normal to the face) at any sub-cell position, and a
face on a grid plane leaves both neighbouring cells at their exact bulk value.
"""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import compile_material_model
from witwin.maxwell.compiler.subpixel import _inverse_diagonal_blend
from witwin.maxwell.scene import prepare_scene

DZ = 0.001
EPS = 2.25


def _edges(top, spec):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.004), (0.0, 0.004), (0.0, 0.010))),
        grid=mw.GridSpec.uniform(DZ),
        boundary=mw.BoundarySpec.faces(default="periodic", z="pec"),
        device="cpu",
        subpixel_samples=spec,
    )
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, top - 0.01), size=(1.0, 1.0, 0.02)),
                                     material=mw.Material(eps_r=EPS)))
    prepared = prepare_scene(scene)
    edges = compile_material_model(prepared, subpixel=spec)["edge_components"]
    return {name: edges["eps"][name][1, 1, :].double() for name in ("Ex", "Ez")}


@pytest.mark.parametrize("samples", [2, 4, (1, 1, 5)])
@pytest.mark.parametrize("phase", [0.07, 0.5, 0.81])
def test_off_grid_face_gets_exact_fill_and_series_mean(samples, phase):
    top = (4 + phase) * DZ
    arithmetic = _edges(top, mw.SubpixelSpec(samples=samples, averaging="arithmetic"))
    polarized = _edges(top, mw.SubpixelSpec(samples=samples, averaging="polarized"))
    # Ez of cell [4, 5] dz is normal to the face: fill fraction = phase.
    assert float(arithmetic["Ez"][4]) == pytest.approx(1.0 + phase * (EPS - 1.0), abs=2e-6)
    assert float(polarized["Ez"][4]) == pytest.approx(1.0 / (phase / EPS + (1.0 - phase)), abs=2e-6)
    # Ex at node 4 or 5 is tangential; its window [k - 1/2, k + 1/2] dz holds the face.
    node = 4 if phase < 0.5 else 5
    fraction = phase + 0.5 if node == 4 else phase - 0.5
    for edges in (arithmetic, polarized):
        assert float(edges["Ex"][node]) == pytest.approx(1.0 + fraction * (EPS - 1.0), abs=2e-6)


def test_uniform_slab_uses_resolved_yee_averaging_windows():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.05), (0.0, 0.05), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.025),
        boundary=mw.BoundarySpec.pec(),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
    )
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(2.56, 2.56, 0.12)),
        material=mw.Material(eps_r=3.0),
    ))
    prepared = prepare_scene(scene)
    edges = compile_material_model(prepared, subpixel=scene.subpixel)["edge_components"]["eps"]
    spacing = 1.28 / 52
    for component in ("Ex", "Ey", "Ez"):
        centers = torch.as_tensor(
            prepared.z_half64 if component == "Ez" else prepared.z_nodes64,
            dtype=torch.float64, device=scene.device,
        )
        overlap = (torch.minimum(centers + spacing / 2, centers.new_tensor(0.06))
                   - torch.maximum(centers - spacing / 2, centers.new_tensor(-0.06)))
        fill = (overlap / spacing).clamp(0.0, 1.0)
        interface = (fill > 0.0) & (fill < 1.0)
        expected = 1.0 / (1.0 - fill + fill / 3.0) if component == "Ez" else 1.0 + 2.0 * fill
        torch.testing.assert_close(edges[component][1, 1, :][interface].double(),
                                   expected[interface], rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("averaging", ["arithmetic", "polarized"])
@pytest.mark.parametrize("samples", [2, 4])
def test_grid_aligned_face_keeps_exact_bulk_values(averaging, samples):
    edges = _edges(4 * DZ, mw.SubpixelSpec(samples=samples, averaging=averaging))
    assert torch.equal(edges["Ez"][:4], torch.full((4,), EPS, dtype=torch.float64))
    assert torch.equal(edges["Ez"][4:], torch.ones_like(edges["Ez"][4:]))
    assert torch.equal(edges["Ex"][:4], torch.full((4,), EPS, dtype=torch.float64))
    assert float(edges["Ex"][4]) == 0.5 * (1.0 + EPS)
    assert torch.equal(edges["Ex"][5:], torch.ones_like(edges["Ex"][5:]))


def test_polarized_blend_is_the_inverse_tensor_diagonal():
    arithmetic = torch.tensor([3.0, 3.0, 3.0, 2.0])
    inverse = torch.tensor([0.6, 0.6, 0.6, 0.5])
    weight = torch.tensor([0.0, 1.0, 0.5, 0.3])
    blended = _inverse_diagonal_blend(arithmetic, inverse, weight)
    expected = 1.0 / (weight * inverse + (1.0 - weight) / arithmetic)
    # Homogeneous windows (<eps><1/eps> == 1) return <eps> exactly.
    expected[3] = 2.0
    torch.testing.assert_close(blended, expected, rtol=1e-6, atol=0.0)
    assert float(blended[0]) == 3.0


def test_coverage_fill_is_differentiable_in_geometry():
    size = torch.tensor(0.0063, requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.004), (0.0, 0.004), (0.0, 0.010))),
        grid=mw.GridSpec.uniform(DZ),
        boundary=mw.BoundarySpec.faces(default="periodic", z="pec"),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="polarized"),
    )
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, size)),
                                     material=mw.Material(eps_r=EPS)))
    edges = compile_material_model(prepare_scene(scene), subpixel=scene.subpixel)["edge_components"]
    (gradient,) = torch.autograd.grad(edges["eps"]["Ez"].sum(), size)
    assert math.isfinite(float(gradient)) and float(gradient) > 0.0


def test_oblique_interface_compiles_symmetric_kottke_tensor():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(0.1), device='cpu',
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging='polarized_tensor'))
    # A half-filled window at the origin, normal to (1, 1, 0).
    scene.add_structure(mw.Structure(mw.Box(position=(1/math.sqrt(2), 1/math.sqrt(2), 0.),
        size=(2., 2., 2.), rotation=(0., 0., math.pi/4)), mw.Material(eps_r=4.)))
    model = prepare_scene(scene).compile_materials()
    i = (3, 3, 3)
    diagonal = model['eps_components']
    off = model['eps_offdiag_components']
    actual = torch.stack([torch.stack([diagonal['x'][i], off['xy'][i], off['xz'][i]]),
                          torch.stack([off['xy'][i], diagonal['y'][i], off['yz'][i]]),
                          torch.stack([off['xz'][i], off['yz'][i], diagonal['z'][i]])])
    normal = torch.tensor([1., 1., 0.]) / math.sqrt(2)
    expected = 2.5 * torch.eye(3) + (1.6 - 2.5) * torch.outer(normal, normal)
    torch.testing.assert_close(actual, expected, rtol=0., atol=2e-5)
    torch.testing.assert_close(actual @ normal, 1.6 * normal, rtol=0., atol=2e-5)


@pytest.mark.parametrize("offdiagonal", [0.0, 0.2])
def test_anisotropic_constituent_excludes_only_its_occupied_windows(offdiagonal):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.6, 0.6),) * 3),
        grid=mw.GridSpec.uniform(0.1), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized_tensor"))
    scene.add_structure(mw.Structure(mw.Sphere(position=(-0.3, 0., 0.), radius=0.16),
        mw.Material(eps_r=4.)))
    baseline = prepare_scene(scene).compile_materials()
    scene.add_structure(mw.Structure(mw.Sphere(position=(0.3, 0., 0.), radius=0.16),
        mw.Material(epsilon_tensor=mw.Tensor3x3(
            ((2., offdiagonal, 0.), (offdiagonal, 3., 0.), (0., 0., 5.))))))
    mixed = prepare_scene(scene).compile_materials()
    # A distant tensor body must not disable the scalar interface's coupling.
    scalar_interface = (4, 7, 6)
    assert baseline["eps_offdiag_components"]["xy"][scalar_interface].abs() > 0.01
    for pair in ("xy", "xz", "yz"):
        assert torch.equal(mixed["eps_offdiag_components"][pair][scalar_interface],
                           baseline["eps_offdiag_components"][pair][scalar_interface])
    # The anisotropic body retains its declared tensor, without scalar projection.
    tensor_center = (9, 6, 6)
    for axis, expected in zip("xyz", (2., 3., 5.)):
        assert float(mixed["eps_components"][axis][tensor_center]) == pytest.approx(expected, abs=2e-5)
    assert float(mixed["eps_offdiag_components"]["xy"][tensor_center]) == pytest.approx(offdiagonal, abs=2e-5)
    if offdiagonal == 0:
        assert torch.count_nonzero(mixed["eps_offdiag_components"]["xy"][8:]) == 0


@pytest.mark.parametrize("eps", [4., 12.])
def test_tensor_mode_preserves_conductivity_and_component_wise_edges(eps):
    from witwin.maxwell.compiler.materials import _PCBConductivityReplay

    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(0.1), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"))
    scene.add_structure(mw.Structure(mw.Sphere(position=(0.017, -0.023, 0.031), radius=0.19),
        mw.Material(eps_r=eps, sigma_e=0.2, sigma_m=0.03)))
    diagonal = prepare_scene(scene).compile_materials()
    scene = scene.clone(subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized_tensor"))
    tensor = prepare_scene(scene).compile_materials()
    assert "subpixel_full_tensor" not in diagonal
    assert tensor["subpixel_full_tensor"]
    for key in ("sigma_e_components", "sigma_m_components"):
        for axis in "xyz":
            assert torch.equal(diagonal[key][axis], tensor[key][axis])
    assert torch.equal(diagonal["sigma_e"], tensor["sigma_e"])
    for key, components in diagonal["edge_components"].items():
        for component, value in components.items():
            assert torch.equal(value, tensor["edge_components"][key][component])
    with pytest.raises(NotImplementedError, match="polarized_tensor"):
        _PCBConductivityReplay(prepare_scene(scene))
