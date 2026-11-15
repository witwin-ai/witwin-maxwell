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
    return types.SimpleNamespace(scene=types.SimpleNamespace(scene=scene))


def _cpu_scene(*, boundary, symmetry, bounds=((0.0, 0.6), (0.0, 0.6), (-0.6, 0.6)), source_x=0.0):
    scene = mw.Scene(
                domain=mw.Domain(bounds=bounds),
                grid=mw.GridSpec.uniform(0.15),
                boundary=boundary,
                symmetry=symmetry,
                sources=(
            mw.PointDipole(
                position=(source_x, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=1.0),
                name="src",
            ),
            ),
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
# Tier A: monitors that would need the image half are refused at construction
# --------------------------------------------------------------------------- #
_FOLDED_X = (("PMC", "low"), None, None)


def _folded_simulation(monitor, symmetry=_FOLDED_X):
    scene = _cpu_scene(boundary=mw.BoundarySpec.pml(num_layers=2), symmetry=symmetry)
    scene = scene.updated(monitors=(*scene.monitors, monitor))
    return mw.Simulation.fdtd(scene, frequencies=[1e9])


@pytest.mark.parametrize("monitor", [
    mw.monitors.AngleProjectionMonitor("far", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3), theta=(0.5,), phi=(0.0,)),
    mw.monitors.CartesianProjectionMonitor("far", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3), x=(0.0,), y=(0.0,), distance=1.0),
    mw.monitors.KSpaceProjectionMonitor("far", position=(0.2, 0.2, 0.0), size=(0.3, 0.3, 0.3), ux=(0.0,), uy=(0.0,)),
], ids=lambda monitor: type(monitor).__name__)
def test_projection_monitor_under_symmetry_is_refused_at_construction(monitor):
    with pytest.raises(NotImplementedError, match="cannot project from a Scene.symmetry domain"):
        _folded_simulation(monitor)


def test_diffraction_monitor_with_folded_tangential_axis_is_refused():
    monitor = mw.monitors.DiffractionMonitor("orders", position=(0.3, 0.3, 0.3), size=(0.6, 0.6, 0.0))
    with pytest.raises(NotImplementedError, match="folds its tangential axis 'x'"):
        _folded_simulation(monitor)
    # A symmetry plane normal to the monitor leaves its transverse period intact.
    _folded_simulation(monitor, symmetry=(None, None, ("PMC", "low")))


def test_plane_flux_monitors_are_admitted_over_the_declared_domain():
    _folded_simulation(mw.FluxMonitor("flux_tangential", axis="z", position=0.3))
    _folded_simulation(mw.FluxMonitor("flux_normal", axis="x", position=0.3))


def _flux_rectangle(x_lower, x_upper):
    return mw.monitors.FinitePlaneMonitor(
        "rect", position=(0.5 * (x_lower + x_upper), 0.3, 0.3), size=(x_upper - x_lower, 0.3, 0.0),
        fields=("Ex", "Ey", "Hx", "Hy"), compute_flux=True)


def test_finite_flux_plane_crossing_the_image_plane_asymmetrically_is_refused():
    with pytest.raises(NotImplementedError, match="crosses the Scene.symmetry image plane on axis 'x'"):
        _folded_simulation(_flux_rectangle(-0.1, 0.3))


def test_finite_flux_planes_symmetric_about_or_beside_the_image_plane_are_admitted():
    _folded_simulation(_flux_rectangle(-0.3, 0.3))
    _folded_simulation(_flux_rectangle(0.0, 0.3))
    # Without flux the rectangle only reports its kept fields.
    _folded_simulation(mw.monitors.FinitePlaneMonitor("rect", position=(0.1, 0.3, 0.3), size=(0.4, 0.3, 0.0)))


def _mode_planes(x_center, x_size, z=0.3):
    position, size = (x_center, 0.3, z), (x_size, 0.3, 0.0)
    return {
        "ModeSource": lambda: mw.ModeSource(position=position, size=size, polarization="Ey",
                                            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9), name="mode"),
        "ModeMonitor": lambda: mw.ModeMonitor("mode", position=position, size=size, polarization="Ey",
                                              frequencies=(1e9,)),
        "ModePort": lambda: mw.ModePort("mode", position=position, size=size, polarization="Ey",
                                        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9)),
    }


def _mode_plane_simulation(kind, plane):
    scene = _cpu_scene(boundary=mw.BoundarySpec.pml(num_layers=2), symmetry=_FOLDED_X)
    field = {"ModeSource": "sources", "ModeMonitor": "monitors", "ModePort": "ports"}[kind]
    scene = scene.updated(**{field: (*getattr(scene, field), plane)})
    return mw.Simulation.fdtd(scene, frequencies=[1e9])


# The aperture [0, 0.4] reaches the x-low image plane at x = 0 (also within half a 0.15 m cell, or a
# plane lying on it); the transverse mode solver has no PEC/PMC symmetry edge.
@pytest.mark.parametrize("kind", ["ModeSource", "ModeMonitor", "ModePort"])
@pytest.mark.parametrize(("case", "x_center", "x_size"), [("touching", 0.2, 0.4), ("snapping", 0.26, 0.4)])
def test_mode_planes_touching_an_image_plane_are_refused(kind, case, x_center, x_size):
    with pytest.raises(NotImplementedError, match="mode plane that touches the Scene.symmetry image plane on axis 'x'"):
        _mode_plane_simulation(kind, _mode_planes(x_center, x_size)[kind]())


def test_mode_plane_lying_on_an_image_plane_is_refused():
    plane = mw.ModeMonitor("mode", position=(0.0, 0.3, 0.0), size=(0.0, 0.3, 0.6), polarization="Ey",
                           frequencies=(1e9,))
    with pytest.raises(NotImplementedError, match="image plane on axis 'x'"):
        _mode_plane_simulation("ModeMonitor", plane)


@pytest.mark.parametrize("kind", ["ModeSource", "ModeMonitor", "ModePort"])
def test_mode_planes_clear_of_the_image_plane_are_admitted(kind):
    _mode_plane_simulation(kind, _mode_planes(0.35, 0.4)[kind]())


# The 0.15 m cell puts the low-x image plane at x = 0; faces within 0.075 m snap onto it.
@pytest.mark.parametrize(("case", "x_lower"), [("touching", 0.0), ("crossing", -0.15), ("snapping", 0.06)])
def test_closed_surface_reaching_the_image_plane_is_refused(case, x_lower):
    box = mw.monitors.ClosedSurfaceMonitor.box("box", position=(x_lower + 0.15, 0.3, 0.0), size=(0.3, 0.3, 0.3))
    with pytest.raises(NotImplementedError, match="touches or crosses the Scene.symmetry image plane on axis 'x'"):
        _folded_simulation(box)


def test_closed_surface_reaching_a_high_face_image_plane_is_refused():
    box = mw.monitors.ClosedSurfaceMonitor.box("box", position=(0.3, 0.45, 0.0), size=(0.3, 0.3, 0.3))
    with pytest.raises(NotImplementedError, match="image plane on axis 'y'"):
        _folded_simulation(box, symmetry=(None, ("PEC", "high"), None))


def test_closed_surface_inside_the_folded_domain_is_admitted():
    _folded_simulation(mw.monitors.ClosedSurfaceMonitor.box("box", position=(0.3, 0.3, 0.0), size=(0.3, 0.3, 0.3)))


def test_closed_surface_under_symmetry_on_an_auto_grid_is_refused():
    scene = _cpu_scene(boundary=mw.BoundarySpec.pml(num_layers=2), symmetry=_FOLDED_X)
    scene = scene.updated(grid=mw.GridSpec.auto(wavelength=0.3))
    scene = scene.updated(monitors=(
        *scene.monitors,
        mw.monitors.ClosedSurfaceMonitor.box("box", position=(0.3, 0.3, 0.0), size=(0.3, 0.3, 0.3)),
    ))
    with pytest.raises(NotImplementedError, match="needs a uniform or custom grid"):
        mw.Simulation.fdtd(scene, frequencies=[1e9])


# --------------------------------------------------------------------------- #
# Tier A: result-side expansion round-trip (both faces, PEC/PMC parity)
# --------------------------------------------------------------------------- #
def _expand_1d(values, *, entry, component, half_size):
    scene = types.SimpleNamespace(Nx=half_size, Ny=1, Nz=1, scene=types.SimpleNamespace(symmetry=(entry, None, None)))
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
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
            spectral_sampler=mw.fdtd.SpectralSampler(window="hanning"),
            full_field_dft=True,
            absorber="cpml",
        ),
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
               sources=(_dipole((0.0, 0.0, 0.0), "Ez"),),
           )

    full = mw.Scene(
               domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
               grid=mw.GridSpec.uniform(0.125),
               boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
               sources=(_dipole((0.0, 0.0, 0.0), "Ez"),),
           )

    half_ez = _run(half).field("Ez", expand_symmetry=True).isel(f=0).values.detach().cpu().numpy()
    full_ez = _run(full).field("Ez").isel(f=0).values.detach().cpu().numpy()
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
               sources=(_dipole((0.0, 0.0, 0.0), "Ex"),),
           )

    full = mw.Scene(
               domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
               grid=mw.GridSpec.uniform(0.125),
               boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
               sources=(_dipole((0.0, 0.0, 0.0), "Ex"),),
           )

    half_ex = _run(half).field("Ex", expand_symmetry=True).isel(f=0).values.detach().cpu().numpy()
    full_ex = _run(full).field("Ex").isel(f=0).values.detach().cpu().numpy()
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
                  sources=(_dipole((0.0, 0.0, 0.0), "Ez"),),
              )

    full = mw.Scene(
               domain=mw.Domain(bounds=((-0.625, 0.625), (-0.625, 0.625), (-0.625, 0.625))),
               grid=mw.GridSpec.uniform(0.125),
               boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
               sources=(_dipole((0.0, 0.0, 0.0), "Ez"),),
           )

    quarter_ez = _run(quarter).field("Ez", expand_symmetry=True).isel(f=0).values.detach().cpu().numpy()
    full_ez = _run(full).field("Ez").isel(f=0).values.detach().cpu().numpy()
    assert quarter_ez.shape == full_ez.shape
    assert _rel_err(quarter_ez, full_ez) < 0.12
