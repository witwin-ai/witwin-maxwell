import types

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import _normalize_symmetry
from witwin.maxwell.fdtd.boundary.runtime import _configure_face_boundary_codes
from witwin.maxwell.fdtd.boundary.common import BOUNDARY_PEC, BOUNDARY_PMC, BOUNDARY_NONE
from witwin.maxwell.result import _expand_tensor_with_symmetry


# --------------------------------------------------------------------------- #
# Tier A: normalization / declaration parsing
# --------------------------------------------------------------------------- #
def test_normalize_symmetry_accepts_bare_and_extended_declarations():
    assert _normalize_symmetry(None) == (None, None, None)
    # Bare strings default to the low face.
    assert _normalize_symmetry(("PEC", None, "pmc")) == (
        ("PEC", "low"),
        None,
        ("PMC", "low"),
    )
    # Explicit (mode, face) pairs on any axis / either face.
    assert _normalize_symmetry((("PEC", "high"), ("PMC", "low"), None)) == (
        ("PEC", "high"),
        ("PMC", "low"),
        None,
    )


def test_normalize_symmetry_rejects_invalid_declarations():
    with pytest.raises(ValueError):
        _normalize_symmetry(("PEC", None))  # wrong length
    with pytest.raises(ValueError):
        _normalize_symmetry((("PEC", "middle"), None, None))  # bad face
    with pytest.raises(ValueError):
        _normalize_symmetry(("PML", None, None))  # bad mode
    with pytest.raises(ValueError):
        _normalize_symmetry((("PEC", "low", "x"), None, None))  # bad tuple len


# --------------------------------------------------------------------------- #
# Tier A: forward-fold face-code injection
# --------------------------------------------------------------------------- #
def _fake_solver(scene):
    return types.SimpleNamespace(scene=scene)


def _cpu_scene(*, boundary, symmetry, bounds=((0.0, 0.6), (0.0, 0.6), (-0.6, 0.6)), source_x=0.0):
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(0.15),
        boundary=boundary,
        symmetry=symmetry,
        device="cpu",
    )
    scene.add_source(
        mw.PointDipole(
            position=(source_x, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
            name="src",
        )
    )
    return scene


def test_configure_places_image_code_on_selected_face():
    # Low face on x, high face on y.
    scene = _cpu_scene(
        boundary=mw.BoundarySpec.none(),
        symmetry=(("PEC", "low"), ("PMC", "high"), None),
    )
    solver = _fake_solver(scene)
    _configure_face_boundary_codes(solver)

    assert solver.boundary_x_low_code == BOUNDARY_PEC
    assert solver.boundary_x_high_code == BOUNDARY_NONE
    assert solver.boundary_y_high_code == BOUNDARY_PMC
    assert solver.boundary_y_low_code == BOUNDARY_NONE
    assert solver.has_pec_faces is True


def test_configure_rejects_incompatible_symmetry_face():
    # PEC boundary on the low x-face conflicts with a low-face symmetry plane.
    scene = _cpu_scene(
        boundary=mw.BoundarySpec.pec(),
        symmetry=(("PMC", "low"), None, None),
    )
    with pytest.raises(ValueError, match="x-low face"):
        _configure_face_boundary_codes(_fake_solver(scene))


def test_configure_rejects_source_in_folded_away_half():
    # Low-face symmetry plane at x=0; a source at x=-0.2 lives in the removed half.
    scene = _cpu_scene(
        boundary=mw.BoundarySpec.none(),
        symmetry=(("PEC", "low"), None, None),
        source_x=-0.2,
    )
    with pytest.raises(ValueError, match="folded-away half"):
        _configure_face_boundary_codes(_fake_solver(scene))


# --------------------------------------------------------------------------- #
# Tier A: result-side expansion round-trip (both faces, PEC/PMC parity)
# --------------------------------------------------------------------------- #
def _expand_1d(values, *, entry, component, half_size):
    scene = types.SimpleNamespace(Nx=half_size, Ny=1, Nz=1, symmetry=(entry, None, None))
    tensor = torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
    out = _expand_tensor_with_symmetry(tensor, scene, component=component)
    return out.reshape(-1).numpy()


def test_expand_low_face_tangential_pec_prepends_odd_mirror():
    # Tangential PEC component is odd under reflection; the shared plane node
    # (index 0) is not duplicated.
    out = _expand_1d([1.0, 2.0, 3.0], entry=("PEC", "low"), component="EY", half_size=3)
    np.testing.assert_allclose(out, [-3.0, -2.0, 1.0, 2.0, 3.0])


def test_expand_high_face_tangential_pec_appends_odd_mirror():
    out = _expand_1d([1.0, 2.0, 3.0], entry=("PEC", "high"), component="EY", half_size=3)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, -2.0, -1.0])


def test_expand_low_face_tangential_pmc_prepends_even_mirror():
    # Tangential PMC component is even under reflection.
    out = _expand_1d([1.0, 2.0, 3.0], entry=("PMC", "low"), component="EZ", half_size=3)
    np.testing.assert_allclose(out, [3.0, 2.0, 1.0, 2.0, 3.0])


def test_expand_high_face_normal_pec_even_mirror():
    # Normal PEC component is even under reflection.
    out = _expand_1d([1.0, 2.0, 3.0], entry=("PEC", "high"), component="EX", half_size=3)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 2.0, 1.0])


def test_expand_staggered_component_mirrors_full_half():
    # A component staggered off the plane has (half_size - 1) samples and is
    # mirrored in full (no shared node). Ex normal to x is even under PEC.
    out = _expand_1d([1.0, 2.0], entry=("PEC", "low"), component="EX", half_size=3)
    np.testing.assert_allclose(out, [2.0, 1.0, 1.0, 2.0])
    out_high = _expand_1d([1.0, 2.0], entry=("PEC", "high"), component="EX", half_size=3)
    np.testing.assert_allclose(out_high, [1.0, 2.0, 2.0, 1.0])


# --------------------------------------------------------------------------- #
# Tier B: physical folded-vs-full equivalence (CUDA)
# --------------------------------------------------------------------------- #
def _run(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
        absorber="cpml",
    ).run()


def _dipole(position, polarization):
    return mw.PointDipole(
        position=position,
        polarization=polarization,
        width=0.05,
        source_time=mw.CW(frequency=1e9, amplitude=25.0),
        name="src",
    )


def _rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_low_face_y_symmetry_expands_to_full_domain():
    # PMC on the y-low face; folded half [0, 0.625]. Ez is tangential to the
    # y-plane so it is even under PMC. dl and the spans are exact binary
    # fractions so the folded and full grids mesh commensurately.
    half = mw.Scene(
        domain=mw.Domain(bounds=((-0.625, 0.625), (0.0, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        symmetry=(None, ("PMC", "low"), None),
        device="cuda",
    )
    half.add_source(_dipole((0.0, 0.0, 0.0), "Ez"))

    full = mw.Scene(
        domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    full.add_source(_dipole((0.0, 0.0, 0.0), "Ez"))

    half_ez = _run(half).tensor("Ez", expand_symmetry=True).detach().cpu().numpy()
    full_ez = _run(full).tensor("Ez").detach().cpu().numpy()
    assert half_ez.shape == full_ez.shape
    assert _rel_err(half_ez, full_ez) < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_high_face_x_symmetry_expands_to_full_domain():
    # PEC on the x-high face; folded half nodes end at the plane x=0. Ex is
    # normal to the x-plane so it is even under PEC.
    half = mw.Scene(
        domain=mw.Domain(bounds=((-0.625, 0.0), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        symmetry=(("PEC", "high"), None, None),
        device="cuda",
    )
    half.add_source(_dipole((0.0, 0.0, 0.0), "Ex"))

    full = mw.Scene(
        domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    full.add_source(_dipole((0.0, 0.0, 0.0), "Ex"))

    half_ex = _run(half).tensor("Ex", expand_symmetry=True).detach().cpu().numpy()
    full_ex = _run(full).tensor("Ex").detach().cpu().numpy()
    assert half_ex.shape == full_ex.shape
    assert _rel_err(half_ex, full_ex) < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_quarter_domain_two_plane_symmetry_expands_to_full_domain():
    # PMC on x-low and y-low; folded quarter. Ez is tangential to both planes.
    quarter = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.625), (0.0, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        symmetry=(("PMC", "low"), ("PMC", "low"), None),
        device="cuda",
    )
    quarter.add_source(_dipole((0.0, 0.0, 0.0), "Ez"))

    full = mw.Scene(
        domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    full.add_source(_dipole((0.0, 0.0, 0.0), "Ez"))

    quarter_ez = _run(quarter).tensor("Ez", expand_symmetry=True).detach().cpu().numpy()
    full_ez = _run(full).tensor("Ez").detach().cpu().numpy()
    assert quarter_ez.shape == full_ez.shape
    assert _rel_err(quarter_ez, full_ez) < 0.12
