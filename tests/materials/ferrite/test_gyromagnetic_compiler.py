"""Slice 1b: gyromagnetic ferrite compiler layout (SoA, local basis, cache).

Gates (design brief slice 1b):
  * local-basis orthogonality + rotation covariance (axis-aligned fast path = no
    rotation / identity);
  * sparse/dense index parity;
  * staircase partial-fill (no scalar averaging of the anti-symmetric tensor);
  * compiler-cache / serialization units.

Falsification: the compiler's per-cell state-space / Cayley matrices must agree
bit-for-bit with the independent Phase-0 verification oracle
(``fdtd.ferrite_reference``); a flipped sign or convention in either twin fails
here.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell import media
from witwin.maxwell.compiler.gyromagnetic import (
    CompiledGyromagneticLayout,
    compile_gyromagnetic_layout,
)
from witwin.maxwell.fdtd import ferrite_reference as oracle
from witwin.maxwell.scene import prepare_scene

MU_0 = 4.0e-7 * math.pi


def _ferrite(bias=(0.0, 0.0, 1.75e5), **kwargs):
    base = dict(
        eps_r=14.5,
        saturation_magnetization=1.40e5,
        bias_field=bias,
        gilbert_damping=2.0e-3,
    )
    base.update(kwargs)
    return mw.GyromagneticFerrite(**base)


def _scene(material, *, size=(0.4, 0.4, 0.4), position=(0.0, 0.0, 0.0), spacing=0.1, extra=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(position=position, size=size), material=material))
    if extra is not None:
        for structure in extra:
            scene.add_structure(structure)
    return prepare_scene(scene)


# --- local basis: orthogonality + rotation covariance ------------------------


@pytest.mark.parametrize(
    "bias",
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.3, -0.7, 0.2),
    ],
)
def test_local_basis_orthonormal_right_handed(bias):
    R = media.gyromagnetic_local_basis(bias)
    eye = torch.eye(3, dtype=R.dtype)
    assert torch.allclose(R.T @ R, eye, atol=1e-12)
    assert torch.det(R).item() == pytest.approx(1.0, abs=1e-12)
    b = torch.as_tensor(bias, dtype=R.dtype)
    b = b / torch.linalg.vector_norm(b)
    assert torch.allclose(R[:, 2], b, atol=1e-12)


def test_axis_aligned_fast_path_is_identity():
    # z-aligned bias -> local frame is exactly the Cartesian frame (no rotation).
    R = media.gyromagnetic_local_basis((0.0, 0.0, 1.0))
    assert torch.allclose(R, torch.eye(3, dtype=R.dtype), atol=1e-15)


def test_layout_fast_axis_codes():
    for axis, code in ((0, 0), (1, 1), (2, 2)):
        bias = [0.0, 0.0, 0.0]
        bias[axis] = 2.0e5
        layout = compile_gyromagnetic_layout(_scene(_ferrite(bias=tuple(bias))))
        assert int(layout.fast_axis[0]) == code
    layout = compile_gyromagnetic_layout(_scene(_ferrite(bias=(1.0e5, 1.0e5, 1.0e5))))
    assert int(layout.fast_axis[0]) == -1


def test_rotation_covariance_of_permeability_tensor():
    """A rotated bias rotates mu covariantly: eigenvalues are bias-direction invariant."""
    freq = 8.0e9
    mag = 1.75e5 / math.sqrt(3.0)
    axis = compile_gyromagnetic_layout(_scene(_ferrite(bias=(0.0, 0.0, 1.75e5))))
    tilted = compile_gyromagnetic_layout(_scene(_ferrite(bias=(mag, mag, mag))))
    mu_axis = axis.permeability_tensor(freq)[0]
    mu_tilted = tilted.permeability_tensor(freq)[0]
    def _sorted_eig(tensor):
        eig = torch.linalg.eigvals(tensor)
        order = torch.argsort(eig.real + 1e-6 * eig.imag)
        return eig[order]

    assert torch.allclose(_sorted_eig(mu_axis), _sorted_eig(mu_tilted), atol=1e-9)


def test_permeability_tensor_matches_material():
    freq = 9.5e9
    material = _ferrite(bias=(0.0, 0.0, 1.75e5))
    layout = compile_gyromagnetic_layout(_scene(material))
    mu_layout = layout.permeability_tensor(freq)[0]
    mu_material = material.permeability_tensor_at_freq(freq)
    assert torch.allclose(mu_layout, mu_material, atol=1e-9)


# --- sparse / dense index parity ---------------------------------------------


def test_sparse_dense_index_parity():
    layout = compile_gyromagnetic_layout(_scene(_ferrite(), size=(0.4, 0.4, 0.4)))
    dense = layout.dense_active_mask()
    assert int(dense.sum()) == layout.num_active
    # active_index must round-trip through the dense mask.
    recovered = torch.nonzero(dense.reshape(-1), as_tuple=False).reshape(-1)
    assert torch.equal(recovered, layout.active_index)
    owner = layout.dense_owner()
    assert torch.equal(owner.reshape(-1)[layout.active_index], layout.slot_owner)
    assert int((owner >= 0).sum()) == layout.num_active


def test_empty_layout_when_no_ferrite():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(0.4, 0.4, 0.4)), material=mw.Material(eps_r=4.0)))
    layout = compile_gyromagnetic_layout(prepare_scene(scene))
    assert layout.num_active == 0
    assert not layout.enabled
    assert layout.permeability_tensor(1e9).shape == (0, 3, 3)


# --- staircase partial fill --------------------------------------------------


def test_staircase_partial_fill_not_blended():
    """A partially-filled box staircases: active cells carry FULL params, unblended."""
    material = _ferrite(bias=(0.0, 0.0, 1.75e5))
    # Box edge deliberately off the grid so boundary cells are partially filled.
    layout = compile_gyromagnetic_layout(_scene(material, size=(0.35, 0.35, 0.35)))
    assert layout.num_active > 0
    # No active cell has blended parameters: every active cell reads exactly the
    # full ferrite omega_0/omega_m, never a fractional average.
    assert torch.allclose(layout.omega_0, torch.full_like(layout.omega_0, material.omega_0), rtol=1e-12)
    assert torch.allclose(layout.omega_m, torch.full_like(layout.omega_m, material.omega_m), rtol=1e-12)
    # A cell is active iff its owning-structure occupancy reached the staircase
    # threshold; the raw occupancy is retained (>= 0.5) but never scales params.
    assert torch.all(layout.occupancy >= 0.5 - 1e-9)


def test_priority_override_suppresses_ferrite():
    """A higher-priority non-ferrite structure on top removes the ferrite cell."""
    ferrite = _ferrite()
    cover = mw.Structure(
        geometry=Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
        material=mw.Material(eps_r=3.0),
        priority=10,
    )
    scene = _scene(ferrite, size=(0.4, 0.4, 0.4), extra=[cover])
    layout = compile_gyromagnetic_layout(scene)
    assert layout.num_active == 0


# --- falsification tie to the Phase-0 oracle ---------------------------------


def test_state_matrices_match_oracle():
    material = _ferrite(bias=(0.0, 0.0, 1.75e5), gilbert_damping=5e-3)
    layout = compile_gyromagnetic_layout(_scene(material))
    P_ref, Q_ref = oracle.state_space_matrices(
        material.omega_0, material.omega_m, material.gilbert_damping
    )
    assert torch.allclose(layout.state_P[0], P_ref, atol=0.0, rtol=1e-12)
    assert torch.allclose(layout.state_Q[0], Q_ref, atol=0.0, rtol=1e-12)


def test_cayley_matches_oracle_and_orthogonal_lossless():
    dt = 1.0e-13
    # Lossless: Phi must be orthogonal (energy conservation building block).
    lossless = _ferrite(bias=(0.0, 0.0, 1.75e5), gilbert_damping=0.0)
    layout = compile_gyromagnetic_layout(_scene(lossless), dt=dt)
    phi = layout.phi[0]
    assert torch.allclose(phi.T @ phi, torch.eye(2, dtype=phi.dtype), atol=1e-12)
    # Matches the oracle's discrete transfer function across the band.
    params = oracle.FerriteReferenceParameters(
        saturation_magnetization=lossless.saturation_magnetization,
        bias_magnitude=lossless.bias_magnitude,
        bias_unit_vector=lossless.bias_unit_vector,
        gilbert_damping=lossless.gilbert_damping,
        gyromagnetic_ratio=lossless.gyromagnetic_ratio,
        mu_infinity=lossless.mu_infinity,
    )
    P_ref, Q_ref = oracle.state_space_matrices(params.omega_0, params.omega_m, params.gilbert_damping)
    eye = torch.eye(2, dtype=torch.float64)
    a_inv = torch.linalg.inv(eye - (dt / 2.0) * P_ref)
    phi_ref = a_inv @ (eye + (dt / 2.0) * P_ref)
    gamma_ref = (a_inv * dt) @ Q_ref
    assert torch.allclose(layout.phi[0], phi_ref, rtol=1e-12, atol=1e-14)
    assert torch.allclose(layout.gamma[0], gamma_ref, rtol=1e-12, atol=1e-14)


def test_cayley_contraction_when_lossy():
    dt = 1.0e-13
    lossy = _ferrite(bias=(0.0, 0.0, 1.75e5), gilbert_damping=1e-2)
    layout = compile_gyromagnetic_layout(_scene(lossy), dt=dt)
    phi = layout.phi[0]
    # Strict contraction: spectral radius / largest singular value < 1.
    sv = torch.linalg.svdvals(phi)
    assert float(sv.max()) < 1.0


# --- cache / serialization ---------------------------------------------------


def test_scene_cache_returns_same_layout():
    material = _ferrite()
    scene = _scene(material)
    a = scene.compile_gyromagnetic_materials()
    b = scene.compile_gyromagnetic_materials()
    assert a is b
    scene.add_structure(mw.Structure(geometry=Box(size=(0.1, 0.1, 0.1)), material=mw.Material(eps_r=2.0)))
    c = scene.compile_gyromagnetic_materials()
    assert c is not a  # cache invalidated on structure change


def test_with_timestep_idempotent_and_dt_binding():
    dt = 2.0e-13
    scene = _scene(_ferrite())
    base = scene.compile_gyromagnetic_materials()
    assert base.dt is None and base.phi is None
    bound = scene.compile_gyromagnetic_materials(dt=dt)
    assert bound.dt == dt and bound.phi is not None
    assert bound.with_timestep(dt) is bound  # idempotent for equal dt


def test_serialization_round_trip():
    dt = 1.5e-13
    layout = compile_gyromagnetic_layout(_scene(_ferrite()), dt=dt)
    restored = CompiledGyromagneticLayout.from_dict(layout.to_dict())
    assert restored.grid_shape == layout.grid_shape
    assert restored.dt == layout.dt
    for name in ("active_index", "occupancy", "bias_unit", "local_basis", "omega_0", "phi", "gamma", "state_P"):
        assert torch.equal(getattr(restored, name), getattr(layout, name))
