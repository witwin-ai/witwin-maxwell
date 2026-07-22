"""Edge-native (per-Yee-component) material sampling gates.

The compiler evaluates the diagonal background permittivity / permeability and
the static conductivities at each Yee component's own staggered location rather
than Kottke-blending at the node grid and arithmetically smearing the node values
onto the edges/faces. These gates pin:

* the staggered shapes of the six component fields;
* homogeneous exactness at each component location;
* the manufactured axis-aligned dielectric interface, where each E component's
  permittivity must equal the analytic Kottke value AT ITS OWN location -- the
  normal (parallel-to-field) component takes the harmonic (series) mean, a
  tangential component the arithmetic (parallel) mean -- which the node->edge
  smear provably cannot reproduce;
* the fail-closed fallback (``edge_components is None``) for material families
  whose per-component staggering is not yet validated.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
_DX = 0.1
# Node coordinates are x_i = -0.5 + 0.1*i (i = 0..10); node index 5 sits at x=0.0.
# Ex edge centres are -0.45 + 0.1*k (k = 0..9); edge 5 sits at x=0.05.
_NODE0 = 5
_EDGE_MAT = 5  # Ex edge centred at +0.05 (inside a x>=0 half-space)
_EDGE_VAC = 4  # Ex edge centred at -0.05 (outside a x>=0 half-space)


def _scene(*, structures=(), samples=(9, 9, 9), averaging="polarized", regions=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=_BOUNDS),
        grid=mw.GridSpec.uniform(_DX),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging),
    )
    for structure in structures:
        scene.add_structure(structure)
    for region in regions:
        scene.add_material_region(region)
    return prepare_scene(scene)


def _half_space(interface_x, eps_r, *, span=10.0):
    """Axis-aligned half-space filling x >= interface_x with material ``eps_r``."""
    return mw.Structure(
        name="half_space",
        geometry=mw.Box(position=(interface_x + span / 2.0, 0.0, 0.0), size=(span, span, span)),
        material=mw.Material(eps_r=eps_r),
    )


def test_edge_components_have_staggered_shapes():
    prepared = _scene(structures=[_half_space(0.0, 4.0)])
    edge = prepared.compile_materials()["edge_components"]
    assert edge is not None
    nx, ny, nz = prepared.Nx, prepared.Ny, prepared.Nz
    assert tuple(edge["eps"]["Ex"].shape) == (nx - 1, ny, nz)
    assert tuple(edge["eps"]["Ey"].shape) == (nx, ny - 1, nz)
    assert tuple(edge["eps"]["Ez"].shape) == (nx, ny, nz - 1)
    assert tuple(edge["mu"]["Hx"].shape) == (nx, ny - 1, nz - 1)
    assert tuple(edge["mu"]["Hy"].shape) == (nx - 1, ny, nz - 1)
    assert tuple(edge["mu"]["Hz"].shape) == (nx - 1, ny - 1, nz)


def test_homogeneous_material_is_exact_at_every_component():
    eps_r = 6.0
    box = mw.Structure(
        name="fill",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0)),
        material=mw.Material(eps_r=eps_r),
    )
    edge = _scene(structures=[box]).compile_materials()["edge_components"]
    for component in ("Ex", "Ey", "Ez"):
        interior = edge["eps"][component][2:-2, 2:-2, 2:-2]
        assert torch.allclose(interior, torch.full_like(interior, eps_r), atol=1e-4)
    for component in ("Hx", "Hy", "Hz"):
        interior = edge["mu"][component][2:-2, 2:-2, 2:-2]
        assert torch.allclose(interior, torch.ones_like(interior), atol=1e-4)


def test_manufactured_interface_normal_component_is_harmonic_kottke():
    """Ex is normal to a x-normal interface, so at the straddled edge it must take
    the harmonic (series) Kottke mean 2*eps2/(1+eps2), edge-native."""
    eps2 = 4.0
    # Interface at the Ex edge-5 centre (x=0.05): that edge straddles it 50/50.
    edge = _scene(structures=[_half_space(0.05, eps2)]).compile_materials()["edge_components"]
    harmonic = 2.0 * eps2 / (1.0 + eps2)  # = 1.6
    value = float(edge["eps"]["Ex"][_EDGE_MAT, _NODE0, _NODE0])
    assert abs(value - harmonic) < 2e-2
    # The node->edge smear (Kottke at the nodes, then arithmetic edge average)
    # gives a markedly different value: this gate is what excludes re-adding it.
    node = _scene(structures=[_half_space(0.05, eps2)]).compile_materials()["eps_components"]["x"]
    smear = 0.5 * (float(node[_EDGE_MAT, _NODE0, _NODE0]) + float(node[_EDGE_MAT + 1, _NODE0, _NODE0]))
    assert abs(smear - harmonic) > 0.3


def test_manufactured_interface_tangential_component_is_arithmetic_kottke():
    """Ey is tangential to a x-normal interface, so at the node straddling it the
    permittivity must take the arithmetic (parallel) mean (eps1+eps2)/2."""
    eps2 = 4.0
    # Interface at node x=0.0: the Ey sample there straddles the interface 50/50.
    edge = _scene(structures=[_half_space(0.0, eps2)]).compile_materials()["edge_components"]
    arithmetic = 0.5 * (1.0 + eps2)  # = 2.5
    value = float(edge["eps"]["Ey"][_NODE0, _NODE0, _NODE0])
    assert abs(value - arithmetic) < 5e-3


def test_normal_component_clean_on_each_side_of_a_node_aligned_interface():
    eps2 = 4.0
    edge = _scene(structures=[_half_space(0.0, eps2)]).compile_materials()["edge_components"]
    ex = edge["eps"]["Ex"]
    assert abs(float(ex[_EDGE_VAC, _NODE0, _NODE0]) - 1.0) < 3e-2  # fully vacuum edge
    assert abs(float(ex[_EDGE_MAT, _NODE0, _NODE0]) - eps2) < 2e-1  # fully material edge


def test_arithmetic_averaging_applies_no_normal_projection():
    """With arithmetic averaging the SAME straddled edge gives the arithmetic mean,
    not the harmonic one -- confirming the polarized path is what selects Kottke."""
    eps2 = 4.0
    edge = _scene(
        structures=[_half_space(0.05, eps2)], averaging="arithmetic"
    ).compile_materials()["edge_components"]
    value = float(edge["eps"]["Ex"][_EDGE_MAT, _NODE0, _NODE0])
    assert abs(value - 0.5 * (1.0 + eps2)) < 2e-2


def test_full_offdiag_anisotropy_falls_back_to_node_path():
    structure = mw.Structure(
        name="tensor",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
        material=mw.Material(
            epsilon_tensor=mw.Tensor3x3(((4.0, 0.5, 0.0), (0.5, 4.0, 0.0), (0.0, 0.0, 4.0)))
        ),
    )
    edge = _scene(structures=[structure], averaging="arithmetic").compile_materials()["edge_components"]
    assert edge is None


def test_conductivity_is_edge_native():
    sigma = 0.05
    box = mw.Structure(
        name="lossy",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0)),
        material=mw.Material(eps_r=2.0, sigma_e=sigma),
    )
    edge = _scene(structures=[box]).compile_materials()["edge_components"]
    interior = edge["sigma_e"]["Ex"][2:-2, 2:-2, 2:-2]
    assert torch.allclose(interior, torch.full_like(interior, sigma), atol=1e-4)
