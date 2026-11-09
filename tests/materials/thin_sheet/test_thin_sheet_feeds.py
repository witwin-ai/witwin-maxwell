"""Feed geometry, admission, and the declared axis-aligned runtime gate."""
import itertools

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.compiler.thin_sheet import CHORD_MASS_FACTOR, compile_thin_sheet
from witwin.maxwell.scene import prepare_scene

from . import harness as h

CELL = 0.005


@pytest.fixture(scope="session", autouse=True)
def _single_thread_reference():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _port(negative, positive):
    axis = int(np.argmax(np.abs(np.array(positive) - negative)))
    return mw.LumpedPort(
        "feed", negative=negative, positive=positive, voltage_path=mw.AxisPath("xyz"[axis]),
        current_surface=Box(position=tuple(np.mean((negative, positive), axis=0)),
                            size=tuple(0.0 if a == axis else 0.01 for a in range(3))))


def _plate(*, offset=0.001, width=0.02, tilted=False):
    return h.rectangle_mesh((0, 0, offset), (0, -0.6, 0.8) if tilted else (0, 0, 1),
                            0.03, width / 2, u_direction=(1, 0, 0), sheet_model="conformal_two_sided")


def _scene(mesh, port, material="copper", others=()):
    material = h.pec_material() if material == "pec" else mw.CopperFoil(
        thickness=35e-6, frequency_range=(0.5e9, 4e9))
    return prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((-0.05, 0.05),) * 3), grid=mw.GridSpec.uniform(CELL),
        boundary=mw.BoundarySpec.faces(default="pec"), device="cpu",
        structures=[mw.Structure(geometry=mesh, material=material), *others],
        ports=[] if port is None else [port]))


def _compile(mesh, port, **kwargs):
    scene = _scene(mesh, port, **kwargs)
    return compile_thin_sheet(scene, scene.structures[0])


def _cylinder():
    return h.extruded_polyline_mesh(h.polygon_points(0.03, 8, phase=np.pi / 8), (-0.01, 0.01),
                                     closed=True, sheet_model="conformal_two_sided")


@pytest.mark.parametrize("material", ["pec", "copper"])
@pytest.mark.parametrize("kind", ["aligned", "tilted", "cylinder"])
def test_gap_geometry_against_hand_values(kind, material):
    # d = x. The tilted transverse section is z = .001 + .75 y,
    # -.01 <= y <= .01, so its physical width is .02/.8 = .025 m.
    if kind == "cylinder":
        mesh = _cylinder()
        level = 0.03 * np.cos(np.pi / 8)
        negative, positive = (-0.005, level, 0), (0.005, level, 0)
        expected_width, driven_count, open_count = 0.02, 10, 4
        cells = {(i, 15, k) for i in (9, 10) for k in range(8, 12)}
    else:
        mesh = _plate(tilted=kind == "tilted", width=0.025 if kind == "tilted" else 0.02)
        negative, positive = (-0.005, 0, 0.001), (0.005, 0, 0.001)
        expected_width = 0.025 if kind == "tilted" else 0.02
        driven_count, open_count = (16, 7) if kind == "tilted" else (10, 4)
        # Grid crossings of the tilted yz section, in increasing y order.
        yz_cells = [(8, 8), (8, 9), (9, 9), (9, 10), (10, 10), (11, 10), (11, 11)]
        cells = ({(i, j, k) for i in (9, 10) for j, k in yz_cells} if kind == "tilted"
                 else {(i, j, 10) for i in (9, 10) for j in range(8, 12)})
    c = _compile(mesh, _port(negative, positive), material=material)
    feed, = c.feeds
    assert feed.port_name == "feed"
    np.testing.assert_array_equal(feed.direction, [1, 0, 0])
    assert feed.length == pytest.approx(0.01, abs=1e-15)
    assert feed.width == pytest.approx(expected_width, abs=1e-15)
    assert set(map(tuple, feed.cut_cells)) == cells
    driven = ~feed.open_chords
    assert driven.sum() == driven_count
    assert feed.open_chords.sum() == open_count
    np.testing.assert_allclose(np.abs(feed.chord_projection[driven]), 1, atol=1e-14)
    np.testing.assert_allclose(feed.chord_projection[feed.open_chords], 0, atol=1e-14)
    np.testing.assert_allclose(feed.chord_length[driven], CELL, atol=1e-15)
    assert np.all(feed.chord_volume > 0)
    if kind == "tilted":
        # y-normal faces have w_c=.8h; z-normal faces have w_c=.6h.
        widths = np.sort(feed.chord_width[driven])
        np.testing.assert_allclose(widths, [0.003] * 6 + [0.004] * 10, atol=1e-15)
        lengths = np.diff([-0.01, -0.008, -0.005, -0.001 / 0.75, 0, 0.005,
                           0.004 / 0.75, 0.01]) / 0.8
        np.testing.assert_allclose(np.sort(feed.chord_length[feed.open_chords]), np.sort(lengths), atol=1e-15)
        np.testing.assert_allclose(feed.chord_width[feed.open_chords], CELL, atol=1e-15)
        # For each yz cell, a corner triangle has mean signed distance one
        # third of its corner's; subtract its moment from the full rectangle.
        # The central cell is two trapezoids: <distance> = .8(a^2+ab+b^2)/(3(a+b)).
        distances = CELL * np.array([
            [(.46 + .06 * .08) / .94, .08],
            [.12, (.34 + .135 * .12) / .865],
            [(.26 + (121/600) * (11/75)) / (479/600), 11/75],
            [4/75, (.54 + (2/75) * (4/75)) / (73/75)],
            [.8 * (.2**2 + .2*.95 + .95**2) / (3*1.15),
             .8 * (.8**2 + .8*.05 + .05**2) / (3*.85)],
            [(.66 + (1/600) * (1/75)) / (599/600), 1/75],
            [14/75, (.14 + (49/150) * (14/75)) / (101/150)],
        ])
        order = sorted(np.flatnonzero(feed.open_chords), key=lambda i: feed.chord_path[i].mean(0)[1])
        expected = CHORD_MASS_FACTOR * lengths[:, None] * CELL * np.maximum(distances, CELL / 16)
        np.testing.assert_allclose(feed.chord_volume[order], expected, rtol=1e-12, atol=1e-22)
        # A driven chord cuts a Cartesian rectangle into two rectangles.
        # Their centroid distances are half the normal-projected heights.
        for i in np.flatnonzero(driven):
            center = feed.chord_path[i].mean(0)
            if np.isclose(feed.chord_width[i], 0.004):
                z0 = CELL * np.floor((center[2] + 1e-14) / CELL)
                distances = 0.8 * np.array((center[2] - z0, z0 + CELL - center[2])) / 2
            else:
                y0 = CELL * np.floor((center[1] + 1e-14) / CELL)
                distances = 0.6 * np.array((y0 + CELL - center[1], center[1] - y0)) / 2
            expected = CHORD_MASS_FACTOR * CELL * feed.chord_width[i] * np.maximum(distances, CELL / 16)
            np.testing.assert_allclose(feed.chord_volume[i], expected, rtol=1e-12, atol=1e-22)
    else:
        np.testing.assert_allclose(feed.chord_length, CELL, atol=1e-15)
        np.testing.assert_allclose(feed.chord_width, CELL, atol=1e-15)
        level = level if kind == "cylinder" else 0.001
        fraction = (level % CELL) / CELL
        distances = CELL * np.array((fraction, 1 - fraction)) / 2
        expected = CHORD_MASS_FACTOR * CELL**2 * np.maximum(distances, CELL / 16)
        np.testing.assert_allclose(feed.chord_volume, np.tile(expected, (len(driven), 1)), rtol=1e-12, atol=1e-22)
    if material == "copper":
        np.testing.assert_array_equal(feed.chord_length, c.chord_length[feed.chord_index])
        np.testing.assert_array_equal(feed.chord_width, c.chord_width[feed.chord_index])
        np.testing.assert_array_equal(feed.chord_volume, c.chord_volume[feed.chord_index])
        np.testing.assert_array_equal(feed.chord_face, c.chord_face[feed.chord_index])
        np.testing.assert_array_equal(feed.chord_rows, c.chord_rows[feed.chord_index])
        np.testing.assert_array_equal(feed.chord_orientation, c.chord_orientation[feed.chord_index])
    else:
        assert c.chord_length is None
        assert np.all(feed.chord_index == -1)


@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("material", ["pec", "copper"])
def test_gap_includes_exact_on_sheet_pairs(tilted, material):
    c = _compile(_plate(offset=0, tilted=tilted), _port((-0.005, 0, 0), (0.005, 0, 0)), material=material)
    feed, = c.feeds
    on_sheet = feed.trace_dofs[:, 0] >= 0
    assert on_sheet.any()
    assert np.all(c.e_side[feed.trace_dofs[on_sheet]] == [-1, 1])
    np.testing.assert_array_equal(feed.chord_volume[on_sheet], c.e_volume[feed.trace_dofs[on_sheet]])
    if not tilted and material == "copper":
        assert np.all(feed.chord_index == -1)  # Aligned pairs stay on the explicit foil path.


def _notched_plate(x0, x1):
    xs, ys = [-0.03, x0, x1, 0.03], [-0.01, 0.002, 0.01]
    vertices = [(x, y, 0.001) for x, y in itertools.product(xs, ys)]
    triangles = []
    for i, j in itertools.product(range(3), range(2)):
        if (i, j) == (1, 0):
            continue
        a, b, c, d = 3*i+j, 3*(i+1)+j, 3*(i+1)+j+1, 3*i+j+1
        triangles.extend(((a, b, c), (a, c, d)))
    return mw.SurfaceMesh(np.array(vertices), triangles, sheet_model="conformal_two_sided")


@pytest.mark.parametrize("notch", [(-0.003, 0.003), (-0.008, -0.006)])
def test_gap_outline_must_cover_the_whole_rectangle(notch):
    with pytest.raises(ValueError, match="sheet outline does not cover"):
        _compile(_notched_plate(*notch), _port((-0.012, 0, 0.001), (0.012, 0, 0.001)))


@pytest.mark.parametrize("terminal", ["negative", "positive"])
@pytest.mark.parametrize("clearance", [0.0, 0.004])
def test_gap_requires_one_cell_beyond_each_terminal(terminal, clearance):
    x = 0.03 - clearance
    ends = ((x - 0.01, 0, 0.001), (x, 0, 0.001))
    if terminal == "negative":
        ends = tuple((-p[0], p[1], p[2]) for p in reversed(ends))
    with pytest.raises(ValueError, match=rf"{terminal} terminal is at the sheet rim.*one cell"):
        _compile(_plate(), _port(*ends))


@pytest.mark.parametrize("terminal_x", [0.008, 0.03 * np.sin(np.pi / 8)])
def test_fold_cannot_cross_gap_or_one_cell_margin(terminal_x):
    y = 0.03 * np.cos(np.pi / 8)
    with pytest.raises(ValueError, match="fold crosses the gap or its one-cell margin"):
        _compile(_cylinder(), _port((-0.005, y, 0), (terminal_x, y, 0)))


def test_terminals_on_different_facets_are_rejected():
    mesh = h.extruded_polyline_mesh([(-0.03, -0.03), (0, 0), (0.03, -0.03)], (-0.01, 0.01),
                                     sheet_model="conformal_two_sided")
    with pytest.raises(ValueError, match="fold crosses the gap"):
        _compile(mesh, _port((-0.01, -0.01, 0), (0.01, -0.01, 0)))


def test_terminals_on_different_sheets_are_rejected():
    other = mw.Structure(geometry=_plate(offset=0.011), material=h.pec_material())
    with pytest.raises(ValueError, match="terminals on different sheets"):
        _compile(_plate(), _port((0, 0, 0.001), (0, 0, 0.011)), others=[other])


def test_subcell_gap_names_length_cell_size_and_fix():
    with pytest.raises(ValueError, match=r"g=0.004 m.*cell size 0.005 m.*refine the grid or lengthen the gap"):
        _compile(_plate(), _port((-0.002, 0, 0.001), (0.002, 0, 0.001)))


@pytest.mark.parametrize("polarity", [-1, 1])
def test_one_cell_gap_owns_one_chord_per_transverse_line(polarity):
    c = _compile(_plate(), _port((-polarity * CELL / 2, 0, 0.001), (polarity * CELL / 2, 0, 0.001)))
    feed, = c.feeds
    driven = ~feed.open_chords
    assert driven.sum() == 5
    np.testing.assert_allclose(feed.chord_length[driven], CELL, atol=1e-15)
    assert all(path.mean(0)[0] == pytest.approx(-CELL / 2) for path, open_ in
               zip(feed.chord_path, feed.open_chords) if not open_)


def test_width_below_two_cells_warns():
    with pytest.warns(UserWarning, match=r"w=0.0075 m.*fewer than 2 cells.*0.005 m"):
        c = _compile(_plate(width=0.0075), _port((-0.005, 0, 0.001), (0.005, 0, 0.001)))
    assert c.feeds[0].width == pytest.approx(0.0075)


def test_transverse_rim_contact_and_exact_one_cell_arm_margin_are_allowed():
    # The positive arm runs from x=.025 to the rim at .030. The terminal
    # line is itself on the transverse rim, which still defines w=.020.
    c = _compile(_plate(), _port((0.015, 0.01, 0.001), (0.025, 0.01, 0.001)))
    assert c.feeds[0].width == pytest.approx(0.02)
    assert c.feeds[0].length == pytest.approx(0.01)


@pytest.mark.parametrize("multiple, admitted", [(0.5, True), (2, False)])
def test_sheet_roundoff_classification(multiple, admitted):
    z = 0.001 + multiple * 1e-9 * CELL
    c = _compile(_plate(), _port((-0.005, 0, z), (0.005, 0, z)))
    assert bool(c.feeds) == admitted


# Runtime gates use the same full-width gap and voltage aperture on both paths.
def _runtime_scene(material="copper", *, reference=False, device="cpu", tilted=False, cell=CELL, width=.02):
    foil = (h.pec_material() if material == "pec" else
            mw.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 4e9)))
    z = 0.001 if tilted else 0.0
    narrow = width == cell
    terminal_y = -cell/2 if narrow else 0
    negative, positive = (-0.005, terminal_y, z), (0.005, terminal_y, z)
    port = mw.LumpedPort(
        "feed", negative=negative, positive=positive, voltage_path=mw.AxisPath("x"),
        current_surface=Box(position=(cell/2, 0, z), size=(0, width+cell, 0.01+cell)),
        voltage_aperture=Box(position=(0, 0, z), size=(0.01, width, 0)))
    if reference:
        structures = []
        for x in (-0.0175, 0.0175):
            geometry = (mw.PlanarPEC([(x-.0125, -width/2), (x+.0125, -width/2),
                                      (x+.0125, width/2), (x-.0125, width/2)], normal_axis="z", coordinate=0)
                        if material == "pec" else Box(position=(x, 0, 0), size=(.025, width, 0)))
            structures.append(mw.Structure(geometry=geometry, material=foil))
    else:
        structures = [mw.Structure(geometry=_plate(offset=z, tilted=tilted, width=width), material=foil)]
    bounds = ((-.05, .05), (-.05-cell/2, .05+cell/2) if narrow else (-.05, .05), (-.05, .05))
    return mw.Scene(domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(cell),
                    boundary=mw.BoundarySpec.faces(default="pec"), device=device,
                    structures=structures, ports=[port])


def _simulation(material="copper", *, reference=False, device="cpu", tilted=False,
                steps=400, driven=True, graph=False, cell=CELL, width=.02, core_path="auto"):
    return mw.Simulation.fdtd(
        _runtime_scene(material, reference=reference, device=device, tilted=tilted, cell=cell, width=width),
        frequencies=[0.8e9, 1e9, 1.2e9], run_time=mw.TimeConfig(time_steps=steps, dt=2e-12 * cell/CELL),
        backend="cpu_reference" if device == "cpu" else "native", cuda_graph=graph, fdtd_core_path=core_path,
        excitations=[mw.PortExcitation("feed", source_time=mw.GaussianPulse(
            frequency=1e9, fwidth=0.5e9))] if driven else [])


@pytest.mark.parametrize("kind, exception, message", [
    ("circuit", NotImplementedError, "circuits or embedded networks"),
    ("reactive", NotImplementedError, "purely resistive termination"),
    ("resistance", ValueError, "positive and finite"),
])
def test_sheet_feed_runtime_load_admission(kind, exception, message):
    scene = _runtime_scene(tilted=True)
    port, = scene.ports
    if kind == "circuit":
        circuit = mw.Circuit("load")
        node = circuit.node("input")
        circuit.add(mw.Resistor("R", node, circuit.ground, 50.))
        circuit.bind_port("feed", positive=node, negative=circuit.ground)
        scene.circuits = (circuit,)
    elif kind == "reactive":
        object.__setattr__(port, "termination", mw.SeriesRLC(r=50., l=1e-9))
    else:
        # Exercise the runtime check independently of constructor validation.
        object.__setattr__(port, "reference_impedance", 0.)
    with pytest.raises(exception, match=message):
        mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=1),
                           backend="cpu_reference").prepare()


def test_off_sheet_port_is_rejected_in_a_sheet_scene():
    scene = _runtime_scene()
    scene.structures = [mw.Structure(geometry=_plate(offset=.011), material=scene.structures[0].material)]
    with pytest.raises(NotImplementedError, match="only sheet gap feeds as ports"):
        mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=1),
                           backend="cpu_reference").prepare()


def _rel_l2(a, b):
    return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b))


@pytest.mark.parametrize("material", ["pec", "copper"])
@pytest.mark.parametrize("width_cells", [pytest.param(1, id="narrow"), 2, 4])
def test_f1_axis_aligned_feed_equivalence(material, width_cells, monkeypatch):
    from witwin.maxwell.fdtd import ports
    samples = []
    original = ports.accumulate_port_observers

    def observe(solver, **kwargs):
        runtime, = solver._port_runtimes
        if runtime.sheet_feed is not None:
            v, i = runtime.sheet_feed.voltage, runtime.sheet_feed.current
        else:
            v, i = runtime.lumped.last_voltage_midpoint, -runtime.lumped.last_branch_current
        samples.append(torch.stack((v, i)).detach().cpu().clone())
        original(solver, **kwargs)

    monkeypatch.setattr(ports, "accumulate_port_observers", observe)
    # stepping imports this function directly for the shared R1 block.
    from witwin.maxwell.fdtd.runtime import stepping
    monkeypatch.setattr(stepping, "accumulate_port_observers", observe)
    errors_by_grid = []
    cases = [(CELL, width_cells*CELL), (CELL/2, width_cells*CELL/2)]
    if width_cells == 4:
        cases.append((CELL/2, width_cells*CELL))
    for cell, width in cases:
        results, traces = [], []
        for reference in (True, False):
            samples.clear()
            results.append(_simulation(material, reference=reference, cell=cell,
                                       width=width,
                                       steps=round(1200*CELL/cell)).run())
            traces.append(torch.stack(samples))
        native, sheet = (r.ports["feed"] for r in results)
        native_z, sheet_z = native.voltage/native.current, sheet.voltage/sheet.current
        native_s, sheet_s = (native_z-50)/(native_z+50), (sheet_z-50)/(sheet_z+50)
        errors = dict(voltage=_rel_l2(traces[1][:, 0], traces[0][:, 0]),
                      current=_rel_l2(traces[1][:, 1], traces[0][:, 1]),
                      impedance=float(((sheet_z-native_z)/native_z).abs().max()),
                      s11=float(((sheet_s-native_s)/native_s).abs().max()))
        feed = results[1].solver._thin_sheets[0].feeds[0]
        geometry = feed["geometry"]
        area_ratio = float(torch.dot(feed["length"], feed["width"])) / (geometry.width * geometry.length)
        diagnostic = sheet.metadata["sheet_gap"]
        p_acc = diagnostic["port_accepted_power"]
        p_var = diagnostic["variance_power"]
        closure = float(((diagnostic["field_accepted_power"] - p_acc - diagnostic["power_correction"])
                         / p_acc).abs().max())
        print(dict(F1=material, cell=cell, width_cells=width/cell, **errors,
                   gap_area_ratio=area_ratio, effective_load=50/area_ratio,
                   p_acc=p_acc.tolist(), p_var=p_var.tolist(), p_var_over_p_acc=(p_var/p_acc).tolist(),
                   exact_source_power=diagnostic["field_accepted_power"].tolist(), power_identity_error=closure),
              flush=True)
        assert area_ratio == pytest.approx(1, rel=1e-14)
        assert closure <= 1e-12
        errors_by_grid.append(errors)
    if width_cells == 1:
        assert max(error for errors in errors_by_grid for error in errors.values()) <= 2e-3



@pytest.mark.parametrize("kind", ["aligned", "tilted", "one_cell", "rim", "single_column"])
def test_feed_quadrature_partitions_the_physical_gap(kind):
    from witwin.maxwell.fdtd.thin_sheet import _feed_quadrature
    ends = ((-.0025, 0, .001), (.0025, 0, .001)) if kind == "one_cell" else (
        (-.005, .01 if kind == "rim" else 0, .001), (.005, .01 if kind == "rim" else 0, .001))
    c = _compile(_plate(tilted=kind == "tilted", width=CELL if kind == "single_column" else .02), _port(*ends))
    feed, = c.feeds
    length, width, columns = _feed_quadrature(c, feed)
    assert np.dot(length, width) == pytest.approx(feed.length * feed.width, rel=1e-14)
    assert np.sum(length * width) / (50 * feed.width * feed.length) == pytest.approx(1/50, rel=1e-14)
    assert np.all(width[feed.open_chords] == 0)
    for column in np.unique(columns[columns >= 0]):
        assert length[columns == column].sum() == pytest.approx(feed.length, rel=1e-14)
    if kind not in ("tilted", "single_column"):
        np.testing.assert_allclose(length[~feed.open_chords], CELL, rtol=1e-14)
        np.testing.assert_allclose(np.sort(np.unique(np.round(width[~feed.open_chords], 12))), [.0025, .005])
    if kind == "single_column":
        simulation = _simulation(steps=32, width=CELL)
        simulation.scene.domain = mw.Domain(bounds=((-0.05, 0.05),)*3)
        result = simulation.run()
        diagnostic = result.ports["feed"].metadata["sheet_gap"]
        assert len(diagnostic["column_conductance"]) == 1
        assert torch.equal(diagnostic["variance_power"], torch.zeros(3, dtype=torch.float64))



_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _state(solver):
    return [*(getattr(solver, name) for name in _FIELDS), *solver._thin_sheets[0].state_tensors()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("material", ["pec", "copper"])
def test_f5_float64_patch_and_graph_replay(material, monkeypatch):
    from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators, prepare_port_drive_tables
    from witwin.maxwell.fdtd.runtime.stepping import _port_step_block
    # The existing runner's float64 port patch and port-graph replay are under test;
    # eligible sheet feeds would otherwise take the core path under the default.
    solvers = {device: _simulation(material, device=device, tilted=True, steps=200, core_path=False).prepare().solver
               for device in ("cpu", "cuda")}
    generator = torch.Generator().manual_seed(43)
    for name in _FIELDS:
        value = torch.randn(getattr(solvers["cpu"], name).shape, dtype=torch.float64, generator=generator)
        for device, solver in solvers.items():
            setattr(solver, name, value.to(device).clone())
    for solver in solvers.values():
        prepare_port_spectral_accumulators(solver, 200, "none")
        prepare_port_drive_tables(solver, 200)
    solvers["cuda"]._port_runtimes[0]._drive_table.copy_(solvers["cpu"]._port_runtimes[0]._drive_table)
    errors = {}
    for n in range(200):
        for solver in solvers.values():
            sheet = solver._thin_sheets[0]
            sheet.begin(solver)
            sheet.magnetic(solver)
            step = solver._port_observer_step
            _port_step_block(solver, step, post_source=lambda: None)
            step.add_(1)
        if n in (0, 199):
            errors[n+1] = max(float((a-b.cpu()).abs().max()) / max(float(a.abs().max()), 1e-30)
                              for a, b in zip(_state(solvers["cpu"]), _state(solvers["cuda"])) if a.numel())
    print(dict(F5=material, float64_patch_relative_error=errors), flush=True)
    assert errors[1] <= 1e-12
    graph = _simulation(material, device="cuda", tilted=True, steps=200, graph=True, core_path=False).run()
    eager = _simulation(material, device="cuda", tilted=True, steps=200, graph=False, core_path=False).run()
    assert graph.solver._cuda_graph_active and graph.solver._port_cuda_graph_active, getattr(
        graph.solver, "_port_graph_capture_error", "capture was not attempted")
    for a, b in zip(_state(graph.solver), _state(eager.solver)):
        assert torch.equal(a, b)
    for key in ("voltage", "current"):
        assert torch.equal(getattr(graph.ports["feed"], key), getattr(eager.ports["feed"], key))
    for key in ("column_voltage", "variance_power"):
        assert torch.equal(graph.ports["feed"].metadata["sheet_gap"][key],
                           eager.ports["feed"].metadata["sheet_gap"][key])
    print(dict(F5=material, graph_eager_max_difference=0, steps=200), flush=True)

    # Compare nonzero no-feed fields and private state against the merged baseline.
    import subprocess
    from witwin.maxwell.fdtd import thin_sheet
    from witwin.maxwell.fdtd.runtime import stepping
    namespace = {"__name__": "witwin.maxwell.fdtd._merged_sheet", "__package__": "witwin.maxwell.fdtd"}
    source = subprocess.check_output(["git", "show", "7eaea197:witwin/maxwell/fdtd/thin_sheet.py"], text=True)
    exec(compile(source, "<merged-thin-sheet>", "exec"), namespace)

    def no_feed_run():
        simulation = _simulation(material, device="cuda", tilted=True, steps=200, graph=True, driven=False,
                                 core_path=False)
        simulation.scene.ports = ()
        simulation.scene.add_source(mw.PointDipole(position=(-.04, .03, .035), polarization=(.3, .5, .8),
            profile="ideal", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)))
        return simulation.run()

    # This stored-output pin preserves the tensor reference's operation order.
    # The CPU/CUDA parity and graph/eager gates above retain the fused default.
    with monkeypatch.context() as patch:
        patch.setenv("WITWIN_MAXWELL_THIN_SHEET_KERNEL", "torch")
        current = no_feed_run()
    with monkeypatch.context() as patch:
        patch.setattr(thin_sheet, "_SheetRuntime", namespace["_SheetRuntime"])
        patch.setattr(stepping, "advance_thin_sheet_electric", namespace["advance_thin_sheet_electric"])
        merged = no_feed_run()
    assert any(bool(getattr(current.solver, name).any()) for name in _FIELDS)
    for a, b in zip(_state(current.solver), _state(merged.solver)):
        assert torch.equal(a, b)
    print(dict(F6=material, merged_baseline="7eaea197", no_feed_max_difference=0, steps=200), flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_f7_checkpoint_resume_includes_feed_and_column_spectra(tmp_path):
    reference = _simulation(device="cuda", tilted=True, steps=400, graph=True).run()
    checkpoint = _simulation(device="cuda", tilted=True, steps=400, graph=True).prepare().run_until(173)
    path = tmp_path / "sheet-feed.pt"
    checkpoint.save(path)
    resumed = _simulation(device="cuda", tilted=True, steps=400, graph=True).prepare().run(
        resume_from=mw.FDTDResumeCheckpoint.load(path))
    for a, b in zip(_state(reference.solver), _state(resumed.solver)):
        assert torch.equal(a, b)
    for key in ("voltage", "current"):
        assert torch.equal(getattr(reference.ports["feed"], key), getattr(resumed.ports["feed"], key))
    for key in ("column_voltage", "variance_power", "field_accepted_power"):
        assert torch.equal(reference.ports["feed"].metadata["sheet_gap"][key],
                           resumed.ports["feed"].metadata["sheet_gap"][key])
    changed = _simulation(device="cuda", tilted=True, steps=400, graph=True)
    changed.excitations = (mw.PortExcitation("feed", source_impedance=100,
        source_time=mw.GaussianPulse(1e9, fwidth=.5e9)),)
    with pytest.raises(ValueError, match="configuration does not match"):
        changed.prepare().run(resume_from=checkpoint)
    print(dict(F7_steps=400, checkpoint_step=173, max_difference=0, changed_resistance_rejected=True), flush=True)


def test_f4_loaded_unexcited_feed_for_twenty_thousand_steps():
    from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, _port_step_block
    from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators
    from witwin.maxwell.fdtd.thin_sheet import thin_sheet_energy
    solver = _simulation(tilted=True, steps=20000, driven=False).prepare().solver
    prepare_port_spectral_accumulators(solver, 20000, "none")
    generator = torch.Generator().manual_seed(61)
    sheet = solver._thin_sheets[0]
    # A nonzero private chord state exercises the column load and modal storage.
    sheet.chord_e.copy_(torch.randn(sheet.chord_e.shape, dtype=torch.float64, generator=generator))
    sheet.chord_e_prev.copy_(sheet.chord_e)
    modified = [thin_sheet_energy(solver, modified=True)]
    for n in range(20000):
        _field_update_block(solver, n*solver.dt)
        _port_step_block(solver, None, post_source=lambda: None)
        if n % 100 == 0 or n == 19999:
            modified.append(thin_sheet_energy(solver, modified=True))
    bound = max(modified)/modified[0]
    print(dict(F4_steps=20000, sample_interval=100, modified_initial=modified[0],
               modified_max_over_initial=bound,
               modified_final_over_initial=modified[-1]/modified[0]), flush=True)
    assert max(modified) <= modified[0] * (1 + 1e-12)
    assert modified[-1] < modified[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("material", ["pec", "copper"])
def test_f3_exact_field_side_power_with_variance_correction(material, monkeypatch):
    scene = _runtime_scene(material, device="cuda", tilted=True)
    scene.domain = mw.Domain(bounds=((-0.08, 0.08),)*3)
    scene.boundary = mw.BoundarySpec.faces(default="pml", num_layers=8)
    scene.add_monitor(mw.ClosedSurfaceMonitor.box("radiation", size=(.08, .08, .08)))
    simulation = mw.Simulation.fdtd(scene, frequencies=[.8e9, 1e9, 1.2e9],
        run_time=mw.TimeConfig(time_steps=5000, dt=2e-12), cuda_graph=material == "pec",
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(1e9, fwidth=.5e9)))
    prepared = simulation.prepare()
    solver = prepared.solver
    sheet = solver._thin_sheets[0]
    cu = sheet.copper
    if material == "copper":
        e_sum = torch.zeros((3, 2, len(cu["area"])), dtype=torch.complex128, device=solver.device)
        z_sum = torch.zeros((3, *cu["state"].shape), dtype=torch.complex128, device=solver.device)
        electric = sheet.electric

        def collect_copper_spectra(solver):
            previous_state = cu["state"].clone()
            electric(solver)
            pair = cu["pair_position"]
            midpoint_e = torch.cat(((sheet.chord_e+sheet.chord_e_prev)/2,
                                   (sheet.e_old[pair]+sheet.e_new[pair])/2)).T
            port, = solver._port_runtimes
            kernel = port.current_phase_weights[port.sample_index, :, None, None]
            e_sum.add_(kernel*midpoint_e[None])
            z_sum.add_(kernel*((previous_state+cu["state"])/2)[None])

        monkeypatch.setattr(sheet, "electric", collect_copper_spectra)
    result = prepared.run()
    diagnostic = result.ports["feed"].metadata["sheet_gap"]
    copper_loss = torch.zeros_like(diagnostic["field_accepted_power"])
    if material == "copper":
        port, = solver._port_runtimes
        scale = solver._spectral_integration_dt * port.frequencies.to(e_sum.real.dtype)[:, None, None]
        e_hat, z_hat = e_sum*scale, z_sum*scale
        loss = (cu["rates"][None, :, None]*z_hat.abs().square()).sum(1)
        loss += (e_hat.conj()*torch.einsum("ij,fjn->fin", cu["D"].to(e_hat.dtype), e_hat)).real.sum(1)
        copper_loss = .5*(loss*cu["area"]).sum(1)
    radiation = torch.stack([torch.as_tensor(result.monitor("radiation", frequency=f)["flux"])
                             for f in (.8e9, 1e9, 1.2e9)]).reshape(-1)
    exact = diagnostic["field_accepted_power"]
    closure = (radiation+copper_loss-exact).abs()/exact.abs()
    print(dict(F3=material, F3_port_accepted=diagnostic["port_accepted_power"].tolist(),
               variance_power=diagnostic["variance_power"].tolist(),
               correction=diagnostic["power_correction"].tolist(), exact_field_accepted=exact.tolist(),
               radiation=radiation.tolist(), copper_loss=copper_loss.tolist(), other_load_loss=0,
               relative_closure=closure.tolist()), flush=True)
    assert float(closure.max()) <= .01
