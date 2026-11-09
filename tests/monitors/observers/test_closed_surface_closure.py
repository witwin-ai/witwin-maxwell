"""Closed surfaces are watertight on the grid; standalone planes keep their declared area.

A closed-surface face is snapped to its nearest tangential-E Yee node along its
normal, and its in-plane rectangle is bounded by the snapped Yee planes of the
adjacent faces. The discrete faces then enclose one closed region exactly, for
any off-node declared box. A standalone ``FinitePlaneMonitor`` keeps the exact
declared-area intersection (``test_finite_cell_intersection.py``).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Cylinder
from witwin.maxwell.fdtd.observers import finite_plane_integration_bounds
from witwin.maxwell.result import _crop_plane_monitor_payload


def test_closed_surface_faces_record_adjacent_normal_planes():
    box = mw.ClosedSurfaceMonitor.box("box", position=(.001, -.002, .0005), size=(.0203, .0161, .0147))
    planes = {face.face_label: face.plane_position for face in box.faces}
    for face in box.faces:
        assert face.closure_planes == face.tangential_bounds
        for axis, (low, high) in face.closure_planes.items():
            assert low == planes[f"{axis}_neg"] and high == planes[f"{axis}_pos"]
    # User faces: an edge off its neighbour by round-off takes the neighbour's plane.
    b, noise = .0101, 3e-13
    faces = []
    for axis in "xyz":
        for sign in "+-":
            position = [0., 0., 0.]
            position["xyz".index(axis)] = b if sign == "+" else -b
            size = [2 * b + 2 * noise] * 3
            size["xyz".index(axis)] = 0.
            faces.append(mw.FinitePlaneMonitor(f"{axis}{sign}", position=position, size=size,
                                               fields=mw.monitors.required_flux_fields(axis), normal_direction=sign))
    surface = mw.ClosedSurfaceMonitor("user", faces)
    assert all(face.closure_planes is None for face in faces)
    for face in surface.faces:
        assert all(planes == (-b, b) for planes in face.closure_planes.values())
    projection = mw.AngleProjectionMonitor("far", position=(0., 0., 0.), size=(.02, .02, .02),
                                           theta=(0., 1.), phi=(0.,), exclude_surfaces=("z_neg",))
    assert len(projection.faces) == 5
    assert all(face.closure_planes == face.tangential_bounds for face in projection.faces)
    planar = mw.AngleProjectionMonitor("plane", position=(0., 0., .01), size=(.02, .02, 0.), theta=(0.,), phi=(0.,))
    assert planar.faces[0].closure_planes is None
    with pytest.raises(ValueError, match="closure_planes"):
        mw.FinitePlaneMonitor("bad", size=(1., 1., 0.), closure_planes={"x": (0., 1.)})


def _nodes():
    return (np.array([-2., -1., -.4, .1, .7, 1.5, 2.]), np.array([-1.8, -.8, -.2, .4, 1., 1.9]),
            np.array([-2., -1., 0., 1., 2.]))


def test_closure_bounds_snap_to_the_adjacent_yee_planes_and_select_whole_cells():
    x, y, z = _nodes()
    prepared = SimpleNamespace(x_nodes64=x, y_nodes64=y, z_nodes64=z)
    # z-normal face; x edges at -.32/.96 snap to the nodes -.4/.7, y edges to -.2/1.0.
    declared = dict(position=(.32, .4, .17), size=(1.28, 1.15, 0.), fields=("Ex", "Ey", "Hx", "Hy"),
                    compute_flux=True)
    closed = mw.FinitePlaneMonitor("closed", **declared, closure_planes={"x": (-.32, .96), "y": (-.175, .975)})
    standalone = mw.FinitePlaneMonitor("plane", **declared)
    assert finite_plane_integration_bounds(prepared, closed) == {"x": (-.4, .7), "y": (-.2, 1.)}
    assert finite_plane_integration_bounds(prepared, standalone) == standalone.tangential_bounds
    # The observer record resolves through the same rule.
    record = {"closure_planes": dict(closed.closure_planes), "tangential_bounds": dict(closed.tangential_bounds)}
    assert finite_plane_integration_bounds(prepared, record) == {"x": (-.4, .7), "y": (-.2, 1.)}

    centers = tuple(.5 * (n[1:] + n[:-1]) for n in (x, y))
    widths = tuple(np.diff(n) for n in (x, y))
    u, v = np.meshgrid(*centers, indexing="ij")
    fields = {"Ex": (2. + .3 * u - .2 * v).astype(complex), "Hy": np.full(u.shape, 3. + 0j),
              "Ey": np.zeros(u.shape, complex), "Hx": np.zeros(u.shape, complex)}
    payload = dict(axis="z", position=.17, frequency=1e9, frequencies=(1e9,), normal_direction="+",
                   fields=tuple(fields), components={k: dict(data=f, coords=centers) for k, f in fields.items()},
                   cell_widths=dict(zip("xy", widths)), coords=centers, compute_flux=True, collocated=True,
                   x=centers[0], y=centers[1], **fields)
    snapped = _crop_plane_monitor_payload(payload, closed, finite_plane_integration_bounds(prepared, closed))
    np.testing.assert_array_equal(snapped["cell_widths"]["x"], widths[0][2:4])
    np.testing.assert_array_equal(snapped["cell_widths"]["y"], widths[1][2:4])
    np.testing.assert_array_equal(snapped["x"], centers[0][2:4])
    assert snapped["finite_cell_selection"]["rule"] == "watertight_closure_at_snapped_adjacent_planes"
    assert snapped["tangential_bounds"] == {"x": (-.4, .7), "y": (-.2, 1.)}
    assert snapped["declared_tangential_bounds"] == closed.tangential_bounds
    exact = _crop_plane_monitor_payload(payload, standalone)
    area = float(np.sum(exact["cell_widths"]["x"]) * np.sum(exact["cell_widths"]["y"]))
    assert area == pytest.approx(1.28 * 1.15, rel=2e-15)
    assert "declared_tangential_bounds" not in exact


_F = 6e9
_DX = 1e-3
# Two on-node boxes, then off-node boxes that snap onto them, including the
# half-cell ties 10.5 mm and 9.5 mm, which IEEE rounding of the nodes resolves.
_BOXES = (.010, .011, .0102, .0097, .0105, .0108, .0095)


def _run(structures=(), monitors=()):
    surfaces = []
    for index, half in enumerate(_BOXES):
        surfaces.append(mw.ClosedSurfaceMonitor.box(f"native{index}", size=(2 * half,) * 3, frequencies=(_F,)))
        surfaces.append(mw.ClosedSurfaceMonitor.box(f"sampled{index}", size=(2 * half,) * 3, frequencies=(_F,),
                                                    sample_spacing=_DX / 2))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-.02, .02),) * 3), grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=8, strength=1.), device="cuda",
        sources=[mw.PointDipole(position=(-.0155, .0012, .0007), polarization=(0., 0., 1.), width=_DX,
                                source_time=mw.CW(frequency=_F, amplitude=40.), name="s")],
        structures=list(structures), monitors=[*surfaces, *monitors])
    return mw.Simulation.fdtd(scene, frequencies=[_F], full_field_dft=bool(monitors),
                              run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=15)).run()


def _value(value):
    return float(np.asarray(value.detach().cpu() if hasattr(value, "detach") else value).reshape(-1)[0])


def _fluxes(result, prefix):
    return [_value(result.monitor(f"{prefix}{index}")["flux"]) for index in range(len(_BOXES))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vacuum_closed_surface_is_watertight_for_off_node_boxes():
    result = _run()
    native, sampled = _fluxes(result, "native"), _fluxes(result, "sampled")
    snapped = []
    for index, half in enumerate(_BOXES):
        surface = result.monitor(f"native{index}")
        faces = surface["faces"].values()
        planes = {(face["axis"], face["normal_direction"]): face["yee_plane_position"] for face in faces}
        snapped.append(tuple(sorted(planes.items())))
        for face in faces:
            assert abs(abs(face["yee_plane_position"]) - half) <= _DX / 2 + 1e-12
            # Every in-plane edge is the snapped plane of the adjacent face.
            for axis, (low, high) in face["tangential_bounds"].items():
                assert (low, high) == (planes[(axis, "-")], planes[(axis, "+")])
        through = sum(abs(_value(face["flux"])) for face in faces)
        # Source outside, vacuum inside: zero net flux up to float32 accumulation
        # and the finite-run spectral residual.
        assert abs(native[index]) < 2e-5 * through
        # Midpoint-sampled faces interpolate fields, so they close to quadrature accuracy.
        assert abs(sampled[index]) < 1e-3 * through
        # Closure does not depend on where the declared box lies within a cell.
        reference = snapped.index(snapped[-1])
        assert native[index] == pytest.approx(native[reference], rel=1e-6, abs=1e-6 * through)
        assert sampled[index] == pytest.approx(sampled[reference], rel=1e-6, abs=1e-6 * through)
    assert len(set(snapped)) >= 2 and len(set(snapped)) < len(_BOXES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lossy_absorption_is_box_independent_and_matches_volume_loss():
    cylinder = mw.Structure(geometry=Cylinder(radius=.005, height=.01, axis="z", position=(0., 0., 0.)),
                            material=mw.Material(eps_r=1., sigma_e=30.))
    loss = mw.PowerLossMonitor("loss", position=(0., 0., 0.), size=(.019,) * 3, frequencies=(_F,))
    result = _run(structures=[cylinder], monitors=[loss])
    absorbed = [-value for value in _fluxes(result, "native")]
    sampled = [-value for value in _fluxes(result, "sampled")]
    volume = _value(result.power_loss("loss").total)
    assert volume > 0
    for index in range(len(_BOXES)):
        # Divergence theorem against the native conduction loss of the enclosed body.
        assert absorbed[index] == pytest.approx(volume, rel=2e-3)
        assert sampled[index] == pytest.approx(volume, rel=5e-3)
        # Different snapped boxes differ only by the vacuum discretization residual.
        assert absorbed[index] == pytest.approx(absorbed[0], rel=2e-4)
