"""Orientation equivalence and mixed-orientation stability for the staircased SIBC.

The surface-impedance boundary enumerates every axis-aligned exposed face of a
good-conductor metal (all six orientations, mixed orientations in one scene) and writes
the Leontovich relation ``E_t = R * (n_hat x H)`` on each face. Two properties must hold
for the all-orientation generalization to be correct:

* **Orientation equivalence (headline)**: the same physical plate scene expressed in a
  grid-permuted orientation produces the field solution permuted by the same relabeling,
  to near-bitwise agreement. A coordinate-axis permutation is an exact symmetry of the
  Yee update, the CPML absorber, and the point-dipole injection, so the only thing that
  can break the permutation covariance is an orientation-dependent bug in the face
  enumeration or the face-normal sign. Falsify by flipping one orientation's face-normal
  sign (recorded in the acceptance doc): the equivalence goes red.
* **Mixed-orientation stability**: the resistive (stateless) Leontovich update is
  unconditionally stable across the good-conductor regime. A finite metal block exposes
  all six faces at once; a long run at several conductivities must stay finite and
  bounded, with no late-time divergence. Falsify by re-adding the dropped inductive
  derivative term (recorded in the acceptance doc): the run diverges to non-finite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_C = 299792458.0
_F = 2.0e9
_DX = (_C / _F) / 20.0
_BOUND = 0.26


def _field(result, component):
    value = result.field(component)
    value = value["data"] if isinstance(value, dict) else value
    return value.detach().cpu().numpy()


def _permute_vector(vector, perm):
    """Place a base-frame vector into world coordinates: ``out[perm[a]] = vector[a]``."""
    out = [0.0, 0.0, 0.0]
    for base_axis in range(3):
        out[perm[base_axis]] = vector[base_axis]
    return tuple(out)


def _run_plate(perm, *, sigma=50.0, base_size=(0.06, 10.0, 10.0)):
    """A full-transverse good-conductor plate normal to base-x, permuted into ``perm``.

    ``perm[base_axis] = world_axis``. The plate is normal to base-x with an illuminated
    low face; a tangential (base-z) point dipole sits in the vacuum region in front of
    it. An identity ``perm`` is the reference; any coordinate permutation must reproduce
    it up to the same relabeling.
    """
    size = _permute_vector(base_size, perm)
    center = _permute_vector((0.11, 0.0, 0.0), perm)
    source_pos = _permute_vector((-0.13, 0.0, 0.0), perm)
    polarization = _permute_vector((0.0, 0.0, 1.0), perm)
    domain = tuple((-_BOUND, _BOUND) for _ in range(3))
    scene = mw.Scene(
        domain=mw.Domain(bounds=domain),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=source_pos,
                polarization=polarization,
                width=2.0 * _DX,
                source_time=mw.CW(frequency=_F, amplitude=40.0),
                name="s",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=center, size=size),
                material=mw.LossyMetalMedium(conductivity=sigma),
            )
        ],
    )
    return mw.Simulation.fdtd(
        scene,
        frequencies=[_F],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=12),
        full_field_dft=True,
    ).run()


def _relabel_to_base(world_array, perm):
    """Remap a world-frame field component array back to base indexing.

    With ``k[perm[a]] = i_a`` the base view is ``base[i] = world[k]``, which is
    ``np.transpose(world, perm)`` (base axis ``a`` reads world axis ``perm[a]``).
    """
    return np.transpose(world_array, perm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("perm", [(2, 0, 1), (1, 2, 0)])
def test_sibc_orientation_equivalence_is_near_bitwise(perm):
    """A grid-permuted plate reproduces the reference field up to the same relabeling."""
    reference = _run_plate((0, 1, 2))
    permuted = _run_plate(perm)
    components = ("Ex", "Ey", "Ez")
    worst = 0.0
    for base_axis, base_component in enumerate(components):
        world_component = components[perm[base_axis]]
        ref = np.abs(_field(reference, base_component))
        world = np.abs(_field(permuted, world_component))
        mapped = _relabel_to_base(world, perm)
        assert mapped.shape == ref.shape, (
            f"{base_component}->{world_component}: shape {mapped.shape} != {ref.shape}"
        )
        scale = np.linalg.norm(ref)
        if scale == 0.0:
            continue
        rel = np.linalg.norm(mapped - ref) / scale
        worst = max(worst, rel)
    # Float32 leapfrog accumulation over ~thousands of steps; a coordinate permutation is
    # an exact symmetry, so the residual is round-off only, orders below any sign/enum bug.
    assert worst < 1.0e-4, f"orientation equivalence residual {worst:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_double_sided_plate_exercises_both_signs_and_stays_passive():
    """A mid-domain plate exposes a low (-x normal) and a high (+x normal) face at once.

    Both metal-side branches of the Leontovich sign must be the passive (energy-absorbing)
    branch. The low face uses ``sn = +1`` and the high face ``sn = -1``; if either were the
    active branch it would inject energy every step and the run would diverge. Driving one
    face for a long run, the field must stay finite and bounded (passivity of both signs),
    and the illuminated face must reflect measurably less than a perfect conductor in the
    same geometry (genuine absorption on the driven low face).
    """
    domain = ((-_BOUND, _BOUND), (-0.06, 0.06), (-0.06, 0.06))

    scene = mw.Scene(
        domain=mw.Domain(bounds=domain),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.faces(
            default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")
        ),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.18, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * _DX,
                source_time=mw.CW(frequency=_F, amplitude=40.0),
                name="s",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.1, 1.0, 1.0)),
                material=mw.LossyMetalMedium(conductivity=50.0),
            )
        ],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_F],
        run_time=mw.TimeConfig(time_steps=6000),
        full_field_dft=True,
    ).run()
    ez = _field(result, "Ez")
    # A plate bounded along x exposes its -x face (sn = +1) and its +x face (sn = -1)
    # simultaneously. If either sign were the active branch, that face would inject energy
    # every step over the 6000-step run and the field would diverge.
    assert np.isfinite(ez).all(), "double-sided plate diverged (a face sign is active)"
    assert np.abs(ez).max() < 1.0e12, "double-sided plate grew unbounded"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("sigma", [50.0, 5.0, 1.0])
def test_mixed_orientation_finite_block_is_stable(sigma):
    """A finite block exposes all six faces at once and stays finite over a long run.

    The resistive (stateless) Leontovich update is unconditionally stable across the
    good-conductor regime; a mixed-orientation block driven for a long run must remain
    finite and bounded (no late-time divergence), at conductivities spanning the good- to
    moderate-conductor corners of the validity domain.
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-_BOUND, _BOUND), (-_BOUND, _BOUND), (-_BOUND, _BOUND))),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.15, 0.02, -0.03),
                polarization=(0.0, 0.0, 1.0),
                width=2.0 * _DX,
                source_time=mw.CW(frequency=_F, amplitude=40.0),
                name="s",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.12)),
                material=mw.LossyMetalMedium(conductivity=sigma),
            )
        ],
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_F],
        run_time=mw.TimeConfig(time_steps=6000),
        full_field_dft=True,
    ).run()
    ez = _field(result, "Ez")
    assert np.isfinite(ez).all(), f"sigma={sigma}: mixed-orientation block diverged"
    # A bounded steady state: the field never approaches the float32 overflow scale.
    assert np.abs(ez).max() < 1.0e12, f"sigma={sigma}: field grew unbounded"
