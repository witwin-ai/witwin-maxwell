"""P5.2 material combination matrix -- the phase acceptance artifact.

This harness documents the TRUE post-P5.2 pairwise composability of the FDTD
material and environmental features. It is written from measurements (construct
the combination, then run a short FDTD solve and check the fields are finite),
not from the plan's projected matrix.

Axes (every ordered pair is exercised):

  material:      dispersive, diagonal-aniso, full-aniso, nonlinear, modulated, sigma_e
  environmental: bloch (x/y Bloch + z PML), pml (structure overlapping the CPML)

Contract asserted per pair:

  * physically meaningful  -> the scene compiles and a short FDTD run produces
    finite fields (the combination is genuinely supported).
  * deferred (unsupported) -> the framework raises ``NotImplementedError`` whose
    message states a physical or mathematical reason and never the phrase
    "not implemented yet".
  * meaningless            -> ``diagonal-aniso + full-aniso`` is not a pair: a
    permittivity is either diagonal or full, both occupy the single
    ``epsilon_tensor`` slot, and a full ``Tensor3x3`` already subsumes the
    diagonal case. The public API cannot express both at once.

The measured matrix (symmetric):

              disp  diagA  fullA  nonlin  mod   sigma  bloch  pml
  disp         -     ok     ok     ok      ok    ok     ok     ok
  diagA        ok    -      X      NO      NO    ok     ok     ok
  fullA        ok    X      -      NO      NO    ok     NO     ok
  nonlin       ok    NO     NO     -       ok    ok     NO     ok
  mod          ok    NO     NO     ok      -     NO     NO     ok
  sigma        ok    ok     ok     ok      NO    -      ok     ok
  bloch        ok    ok     NO     NO      NO    ok     -      ok
  pml          ok    ok     ok     ok      ok    ok     ok     -

  ok = supported (compiles + finite run); NO = deferred (physics-worded raise);
  X = meaningless (same epsilon_tensor slot).
"""

import itertools

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FREQ = 1.0e9


class _ExclusiveFeatures(Exception):
    """Two axes that occupy the same material slot cannot form a pair."""


# ---------------------------------------------------------------------------
# Axis appliers: each mutates a Material keyword dict (material axes) or is an
# environmental flag consumed by the scene builder.
# ---------------------------------------------------------------------------
def _rotated_uniaxial(exx, eo, ee):
    """A full off-diagonal Tensor3x3 (uniaxial rotated 45 deg about x)."""
    mean = 0.5 * (eo + ee)
    delta = 0.5 * (ee - eo)
    return mw.Tensor3x3(((exx, 0.0, 0.0), (0.0, mean, delta), (0.0, delta, mean)))


def _set_eps_tensor(kw, tensor):
    if "epsilon_tensor" in kw:
        raise _ExclusiveFeatures("both axes write epsilon_tensor")
    kw["epsilon_tensor"] = tensor


def _apply_dispersive(kw):
    kw.setdefault("lorentz_poles", ())
    kw["lorentz_poles"] = kw["lorentz_poles"] + (
        mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.0e9, gamma=2.0e8),
    )


def _apply_diagonal_aniso(kw):
    _set_eps_tensor(kw, mw.DiagonalTensor3(2.25, 2.25, 4.0))


def _apply_full_aniso(kw):
    _set_eps_tensor(kw, _rotated_uniaxial(2.0, 2.0, 3.0))


def _apply_nonlinear(kw):
    kw["kerr_chi3"] = 1.0e-18


def _apply_modulated(kw):
    kw["modulation"] = mw.ModulationSpec(frequency=2.0e8, amplitude=0.1)


def _apply_sigma_e(kw):
    kw["sigma_e"] = 0.02


_MATERIAL_APPLIERS = {
    "dispersive": _apply_dispersive,
    "diagonal-aniso": _apply_diagonal_aniso,
    "full-aniso": _apply_full_aniso,
    "nonlinear": _apply_nonlinear,
    "modulated": _apply_modulated,
    "sigma_e": _apply_sigma_e,
}
_ENV_AXES = ("bloch", "pml")
_AXES = tuple(_MATERIAL_APPLIERS) + _ENV_AXES


def _build_and_run(axes):
    """Construct the combination and run a short FDTD solve; return the Ez field.

    Raises whatever the framework raises (``_ExclusiveFeatures`` for the
    same-slot pair, ``NotImplementedError`` for a deferred combination).
    """
    mat_axes = [a for a in axes if a in _MATERIAL_APPLIERS]
    kw = {}
    for a in mat_axes:
        _MATERIAL_APPLIERS[a](kw)

    bloch = "bloch" in axes
    pml_overlap = "pml" in axes

    if bloch:
        boundary = mw.BoundarySpec.faces(
            default="pml", num_layers=5, strength=1.0,
            x=("bloch", "bloch"), y=("bloch", "bloch"),
            bloch_wavevector=(0.9, 0.7, 0.0),
        )
    else:
        boundary = mw.BoundarySpec.pml(num_layers=5, strength=1.0)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=boundary,
        device="cuda",
    )
    # A structure overlapping the absorber when the pml axis is active, else a
    # compact interior box. When the pair is env-only, use a plain dielectric so
    # the run still has a material interface to propagate through.
    box_size = 0.5 if pml_overlap else 0.28
    material = mw.Material(**kw) if mat_axes else mw.Material(eps_r=4.0)
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(box_size,) * 3),
                     material=material)
    )
    scene.add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", width=0.06,
                       source_time=mw.CW(frequency=_FREQ, amplitude=20.0), name="src")
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=[_FREQ],
        run_time=mw.TimeConfig.auto(steady_cycles=1, transient_cycles=2),
        full_field_dft=True,
    ).run()
    ez = result.field("Ez")
    return (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# The measured matrix. Keys are frozensets (the matrix is symmetric).
# ---------------------------------------------------------------------------
_MEANINGLESS = {frozenset({"diagonal-aniso", "full-aniso"})}

# deferred pair -> physics keywords that must appear in the (physics-worded) message.
_DEFERRED = {
    frozenset({"diagonal-aniso", "nonlinear"}): ("nonlinear", "anisotropic"),
    frozenset({"full-aniso", "nonlinear"}): ("nonlinear", "anisotropic"),
    frozenset({"diagonal-aniso", "modulated"}): ("modulated", "anisotropic"),
    frozenset({"full-aniso", "modulated"}): ("modulated", "anisotropic"),
    frozenset({"modulated", "sigma_e"}): ("modulated", "conductivity"),
    frozenset({"full-aniso", "bloch"}): ("bloch", "anisotropic"),
    frozenset({"nonlinear", "bloch"}): ("bloch", "nonlinear"),
    frozenset({"modulated", "bloch"}): ("bloch", "modulated"),
}


def _pair_ids(pairs):
    return [f"{a}+{b}" for a, b in pairs]


_ORDERED_PAIRS = list(itertools.permutations(_AXES, 2))


@pytest.mark.parametrize("a,b", _ORDERED_PAIRS, ids=_pair_ids(_ORDERED_PAIRS))
def test_combination_matrix(a, b):
    key = frozenset({a, b})

    if key in _MEANINGLESS:
        # A permittivity is either diagonal or full; the two axes share the
        # single epsilon_tensor slot, so the pair cannot be expressed at all.
        with pytest.raises(_ExclusiveFeatures):
            _build_and_run([a, b])
        return

    if key in _DEFERRED:
        with pytest.raises(NotImplementedError) as excinfo:
            _build_and_run([a, b])
        message = str(excinfo.value)
        assert "not implemented yet" not in message.lower(), message
        low = message.lower()
        for keyword in _DEFERRED[key]:
            assert keyword.lower() in low, (keyword, message)
        return

    # Physically meaningful: compiles and runs to a finite field, in both orders.
    ez = _build_and_run([a, b])
    assert np.isfinite(ez).all(), (a, b)
    assert np.abs(ez).max() > 0.0, (a, b)


# ---------------------------------------------------------------------------
# Nonlinear flavor coverage: the pair matrix uses chi3 as the nonlinear
# representative; confirm chi2 and TPA compose with the same partners and are
# rejected on the same anisotropic edge with a physics-worded message.
# ---------------------------------------------------------------------------
def _nonlinear_material(flavor, partner_kw):
    kw = dict(partner_kw)
    if flavor == "chi2":
        kw["nonlinearity"] = mw.NonlinearSusceptibility(chi2=1.0e-12)
    elif flavor == "chi3":
        kw["kerr_chi3"] = 1.0e-18
    elif flavor == "tpa":
        kw["nonlinearity"] = mw.TwoPhotonAbsorption(beta=1.0e-11, n0=1.5)
    return mw.Material(**kw)


def _run_material(material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.pml(num_layers=5, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(0.28,) * 3), material=material)
    )
    scene.add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", width=0.06,
                       source_time=mw.CW(frequency=_FREQ, amplitude=20.0), name="src")
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=[_FREQ],
        run_time=mw.TimeConfig.auto(steady_cycles=1, transient_cycles=2),
        full_field_dft=True,
    ).run()
    ez = result.field("Ez")
    return (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()


@pytest.mark.parametrize("flavor", ["chi2", "chi3", "tpa"])
@pytest.mark.parametrize(
    "partner",
    [
        ({"lorentz_poles": (mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.0e9, gamma=2.0e8),)}, "dispersive"),
        ({"modulation": mw.ModulationSpec(frequency=2.0e8, amplitude=0.1)}, "modulated"),
        ({"sigma_e": 0.02}, "sigma_e"),
    ],
    ids=lambda p: p[1],
)
def test_nonlinear_flavor_composes(flavor, partner):
    """chi2 / chi3 / TPA each compose with dispersion, modulation, and sigma_e."""
    partner_kw, _ = partner
    ez = _run_material(_nonlinear_material(flavor, partner_kw))
    assert np.isfinite(ez).all()
    assert np.abs(ez).max() > 0.0


@pytest.mark.parametrize("flavor", ["chi2", "chi3", "tpa"])
def test_nonlinear_flavor_rejects_anisotropy(flavor):
    """Every nonlinear flavor is rejected with an anisotropic full tensor, with a
    physics-worded reason (the field-dependent scalar permittivity cannot be a
    coupled tensor inverse)."""
    with pytest.raises(NotImplementedError) as excinfo:
        _nonlinear_material(flavor, {"epsilon_tensor": _rotated_uniaxial(2.0, 2.0, 3.0)})
    message = str(excinfo.value).lower()
    assert "not implemented yet" not in message
    assert "nonlinear" in message and "anisotropic" in message
