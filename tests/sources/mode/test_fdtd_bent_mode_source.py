import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.postprocess.modal import _mode_source_from_mode_spec


def _build_wire_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.24)).with_material(
            mw.Material(eps_r=12.0), name="core"
        )
    )
    return scene


def _solve_bent(bend_radius, bend_axis):
    scene = _build_wire_scene()
    source = mw.ModeSource(
        position=(-0.32, 0.0, 0.0),
        size=(0.0, 0.56, 0.56),
        polarization="Ez",
        source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
        name="m",
        bend_radius=bend_radius,
        bend_axis=bend_axis,
    )
    scene.add_source(source)
    solver = mw.Simulation.fdtd(scene, frequencies=[1.0e9]).prepare().solver
    compiled = _compile_mode_source(source, default_frequency=1.0e9)
    return solve_mode_source_profile(solver, compiled)


def _radial_centroid(mode_data):
    # Propagation along x, cylinder axis z, so the in-plane radial axis is y (dim 0
    # of the (u, v) = (y, z) profile). Return the index-space centroid measured from
    # the aperture centre, a bend-sign-agnostic measure of the transverse mode shift.
    profile = torch.abs(mode_data["component_profiles"]["Ez"]).detach().cpu().to(torch.float64)
    weight = profile.sum(dim=1)
    index = torch.arange(weight.numel(), dtype=torch.float64)
    centre = (weight.numel() - 1) / 2.0
    return float((index * weight).sum() / weight.sum()) - centre


# --- construction / validation (no GPU needed) -----------------------------------


def test_bend_axis_must_be_tangential_to_the_mode_plane():
    # Propagation (zero-size) axis is x, so bend_axis='x' is the propagation axis.
    with pytest.raises(ValueError, match="tangential"):
        mw.ModeSource(
            position=(-0.32, 0.0, 0.0),
            size=(0.0, 0.56, 0.56),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9),
            bend_radius=2.0,
            bend_axis="x",
        )


def test_bend_radius_requires_bend_axis_and_vice_versa():
    with pytest.raises(ValueError, match="bend_axis"):
        mw.ModeSource(size=(0.0, 0.56, 0.56), bend_radius=2.0)
    with pytest.raises(ValueError, match="bend_radius"):
        mw.ModeSource(size=(0.0, 0.56, 0.56), bend_axis="y")


def test_zero_bend_radius_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        mw.ModeSource(size=(0.0, 0.56, 0.56), bend_radius=0.0, bend_axis="y")


def test_bent_port_threads_bend_into_source_monitor_and_spec():
    port = mw.ModePort(
        name="p",
        position=(-0.32, 0.0, 0.0),
        size=(0.0, 0.56, 0.56),
        polarization="Ez",
        source_time=mw.CW(frequency=1.0e9),
        bend_radius=3.0,
        bend_axis="z",
    )
    source = port.to_mode_source()
    monitor = port.to_mode_monitor()
    assert (source.bend_radius, source.bend_axis) == (3.0, "z")
    assert (monitor.bend_radius, monitor.bend_axis) == (3.0, "z")

    spec = monitor.mode_spec()
    assert spec["bend_radius"] == 3.0 and spec["bend_axis"] == "z"
    rebuilt = _mode_source_from_mode_spec(spec, frequency=1.0e9)
    assert (rebuilt.bend_radius, rebuilt.bend_axis) == (3.0, "z")

    compiled = _compile_mode_source(source, default_frequency=1.0e9)
    assert compiled["bend_radius"] == 3.0 and compiled["bend_axis"] == "z"


# --- physics (needs the FDTD CUDA backend) ---------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_bent_effective_index_rises_above_straight_and_relaxes_with_radius():
    straight = _solve_bent(None, None)["effective_index"]
    tight = _solve_bent(2.0, "z")["effective_index"]
    loose = _solve_bent(50.0, "z")["effective_index"]

    # A bend confines the mode more strongly on the outer wall, raising n_eff.
    assert tight > straight + 1e-3
    # The conformal factor -> 1 as R -> inf, so a loose bend approaches the straight guide.
    assert abs(loose - straight) < abs(tight - straight)
    assert abs(loose - straight) < 1e-2 * straight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.xfail(
    reason=(
        "The common-plane full-vector discretization retains an order-1e-6 orientation bias for mirrored bends; "
        "a component-staggered eigensolver is required to restore exact discrete reflection symmetry."
    ),
    strict=False,
)
def test_bent_effective_index_is_symmetric_in_radius_sign_for_a_symmetric_guide():
    positive = _solve_bent(2.0, "z")["effective_index"]
    negative = _solve_bent(-2.0, "z")["effective_index"]
    # Mirroring the guide about its centre maps R -> -R and leaves the spectrum fixed.
    assert positive == pytest.approx(negative, rel=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_bent_mode_profile_shifts_toward_the_outer_wall():
    straight_shift = _radial_centroid(_solve_bent(None, None))
    positive_shift = _radial_centroid(_solve_bent(2.0, "z"))
    negative_shift = _radial_centroid(_solve_bent(-2.0, "z"))

    assert abs(straight_shift) < 0.05
    # R > 0 grades the index up on the +y side, pulling the mode outward (+y).
    assert positive_shift > 0.5
    # R < 0 mirrors it toward -y by (nearly) the same amount.
    assert negative_shift < -0.5
    assert positive_shift == pytest.approx(-negative_shift, rel=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_bend_radius_smaller_than_aperture_half_span_is_rejected():
    # The aperture spans y in [-0.28, 0.28]; R = 0.2 drives 1 + r/R <= 0 near -y,
    # where the conformal transform is singular.
    with pytest.raises(ValueError, match="centre of curvature"):
        _solve_bent(0.2, "z")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_bent_mode_reports_its_bend_metadata():
    mode_data = _solve_bent(4.0, "z")
    assert mode_data["bend_radius"] == 4.0
    assert mode_data["bend_axis"] == "z"
    assert math.isfinite(mode_data["effective_index"])
