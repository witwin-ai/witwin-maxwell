"""G4 gates: combinations of round-2 layered MoM features that no input branch validated.

(a) acceptance gates
- polygon layouts (B1) x dispersive layers (E1): a multi-frequency dispersive run against
  per-frequency runs of the frozen constant permittivity eps(f) (the E1 reference; a kernel
  that read the wrong permittivity row shows an O(delta eps) difference);
- surface_admittance (A1c) x dispersive layers: the same frozen-eps(f) reference, which also
  checks the body admittance (background permittivity of the embedding layer at each frequency);
- GR1 gradients x polygon path: central finite differences inside the GR1 derived bound.
(b) lifted round-2 refusals
- polygon x hmatrix: D1 acceptance on planar meshes (|dY|_F/|Y|_F vs dense <= 1e-4, the exact
  far-action check <= 1e-5, GMRES bound <= 1e-5), cluster partition and determinism;
- polygon x edge ports: the planar-mesher edge port of a Manhattan strip loop matches the
  rectangle-mesher edge port of the same layout (planar_mesher="triangle" vs "auto": inductance
  within the mesher discretization bound) and, on both meshers, the C1 property that an edge port
  equals the delta-gap port of the same width (bitwise);
- surface_admittance x edge ports: an edge-fed split lossy strip reproduces R = l/(sigma A) at DC
  (the aperture-wide PEC block contacts the full end faces), and the edge port equals the delta-gap
  block of the same width.
"""

import dataclasses
import gc
import json
import math
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.network import solve_prepared_network
from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, U, loop_config, loop_scene
from tests.support.rfic_closed_references import network_metrics, strip_fixture

shapely = pytest.importorskip("shapely")

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]

UM = 1e-6
FR4 = dict(eps_r=4.0, tan_delta=0.02, frequency=1e9)


@pytest.fixture(autouse=True)
def budget(record_property):
    """Memory cap per test; wall time is recorded only (budgets live in tests/perf, 0.7.0 test tiers)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    record_property("seconds", seconds)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def relative(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b)).tolist()


def layered(config, **changes):
    return dataclasses.replace(config, layered=dataclasses.replace(config.layered, **changes))


def run(scene, frequencies, config):
    return mw.Simulation.mom(scene, frequencies=frequencies, config=config).run()


def polygon_structure(polygon, z, material, thickness=0.0):
    loops = [list(polygon.exterior.coords)[:-1], *[list(r.coords)[:-1] for r in polygon.interiors]]
    loops = [torch.tensor(loop, dtype=torch.float64) * UM for loop in loops]
    return mw.Structure(mw.geometry.PlanarPEC(loops[0], holes=loops[1:], normal_axis="z", coordinate=z * UM,
                                              physical_thickness=thickness * UM), material)


def prism_via(x, y, z0, z1, radius, material):
    return mw.Structure(mw.geometry.Prism(
        position=torch.tensor((x, y, (z0 + z1) / 2), dtype=torch.float64) * UM,
        radius=torch.tensor(radius * UM, dtype=torch.float64), height=torch.tensor((z1 - z0) * UM, dtype=torch.float64),
        num_sides=8, axis="z"), material)


def gap_port(name, positive, negative, axis, aperture=None):
    extra = {} if aperture is None else dict(voltage_aperture=mw.Box(
        position=tuple(v * UM for v in aperture[0]), size=tuple(v * UM for v in aperture[1])))
    return mw.LumpedPort(name, positive=tuple(v * UM for v in positive), negative=tuple(v * UM for v in negative),
                         voltage_path=mw.AxisPath(axis), reference_impedance=50.0, **extra)


# ---------------------------------------------------------------- polygon two-plane fixture


def octagon_pair_scene(stack, *, lossy=True):
    """Two-port polygon layout on two metal planes (z = 100 and 120 um) joined by octagonal Prism vias.

    Lower plane: a gapped 32-gon ring (R = 150 um, w = 12 um, port p1 across its bottom gap).
    Upper plane: a 45-degree routed strip from above the ring's left via to its right via with
    a gap (port p2). Vias (octagons, r = 3 um) at the ring's left and right extremes connect the
    planes, so the vertical tables, wall assembly and both horizontal planes carry current.
    """
    scene = mw.Scene(background=stack)
    material = mw.media.LossyMetalMedium(conductivity=5.8e7) if lossy else mw.Material.pec()
    thickness = 1.0 if lossy else 0.0
    ring = shapely.Polygon([(150 * math.cos(2 * math.pi * k / 32), 150 * math.sin(2 * math.pi * k / 32))
                            for k in range(32)])
    ring = ring.difference(ring.buffer(-12, join_style="mitre"))
    flat = 150 * math.cos(math.pi / 32) - 6
    ring = ring.difference(shapely.box(-5, -flat - 10, 5, -flat + 10))
    scene = scene.updated(structures=(
        *scene.structures,
        *(polygon_structure(part, 100., material, thickness) for part in shapely.get_parts(ring)),
    ))
    route = shapely.LineString(((-144, 0), (-100, 44), (100, 44), (144, 0)))
    upper = route.buffer(6, cap_style="square", join_style="mitre")
    upper = upper.difference(shapely.box(-5, 30, 5, 60))
    scene = scene.updated(structures=(
        *scene.structures,
        *(polygon_structure(part, 120., material, thickness) for part in shapely.get_parts(upper)),
    ))
    scene = scene.updated(structures=(
        *scene.structures,
        *(prism_via(x, 0., 100., 120., 3., material) for x in (-144., 144.)),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        gap_port("p1", (5, -flat, 100.), (-5, -flat, 100.), "x"),
        gap_port("p2", (5, 44, 120.), (-5, 44, 120.), "x"),
    ))
    return scene


def octagon_config(max_edge=30.):
    # Polygon layouts are sheet-only; auto would select surface_admittance for thick copper (A2).
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=8 * UM, max_edge_length_m=max_edge * UM,
                                                        edge_cell_m=4 * UM, conductor_model="sheet"))


def dispersive_stack(model):
    return LayerStack((Layer("substrate", 100 * UM, model), Layer("film", 40 * UM, model)), "pec", HalfSpace())


# ---------------------------------------------------------------- (a1) polygon x dispersive


def test_polygon_dispersive_matches_frozen_permittivity(record_property):
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    frequencies = (1e8, 5e9, 2e10)
    result = run(octagon_pair_scene(dispersive_stack(model)), frequencies, octagon_config())
    metadata = result.network.metadata
    assert metadata["mesher"] == "planar_conforming_delaunay_v1" and metadata["vias"]
    y = result.network.to_y()
    errors = []
    for k, f in enumerate(frequencies):
        frozen = dispersive_stack(model.relative_permittivity(f))
        reference = run(octagon_pair_scene(frozen), (f,), octagon_config())
        assert reference.network.metadata["mesh_sha256"] == metadata["mesh_sha256"]
        errors.append(relative(y[k], reference.network.to_y()[0]))
    # The constant permittivity at the lowest and highest frequency differ by O(1e-1): the gate resolves rows.
    spread = abs(model.relative_permittivity(frequencies[0]) - model.relative_permittivity(frequencies[-1]))
    record_property("frozen_relative", json.dumps(errors))
    record_property("permittivity_spread", spread)
    assert spread > 1e-2
    assert max(errors) <= 1e-6
    reciprocity = ((y - y.transpose(-1, -2)).abs().amax() / y.abs().amax()).item()
    assert reciprocity <= 1e-10
    assert torch.linalg.svdvals(result.network.s).amax().item() <= 1 + 1e-9
    again = run(octagon_pair_scene(dispersive_stack(model)), frequencies, octagon_config())
    assert torch.equal(again.network.s, result.network.s)
    assert torch.equal(mw.mom.surface_currents(again).currents, mw.mom.surface_currents(result).currents)


# ---------------------------------------------------------------- (a2) surface_admittance x dispersive


def admittance_strip(stack):
    # Coarser than the A1c gates: the frozen-permittivity identity does not depend on the mesh.
    scene, config = strip_fixture(model="surface_admittance", maximum_um=40., edge_um=4.)
    moved = mw.Scene(background=stack)
    moved = moved.updated(structures=(*moved.structures, *scene.structures))
    moved = moved.updated(ports=(*moved.ports, *scene.ports))
    return moved, config


def strip_stack(eps):
    # The strip (z = 99..101 um) lies inside the dispersive layer over a PEC ground plane.
    return LayerStack((Layer("film", 300 * UM, eps),), "pec", HalfSpace())


def test_surface_admittance_dispersive_matches_frozen_permittivity(record_property):
    from witwin.maxwell.mom.layered.admittance import background_permittivity, regime_weight
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    # Blend window and DSA regime of the 2 um copper body; the low-regime interior (uniform local
    # conduction) reads no background permittivity, only the Green table, which (a1) and E1 cover.
    frequencies = (2e8, 2e9)
    scene, config = admittance_strip(strip_stack(model))
    simulation = mw.Simulation.mom(scene, frequencies=frequencies, config=config).prepare()
    result = simulation.run()
    prepared = simulation.solver.prepared
    body = prepared.admittance.bodies[0]
    assert 0 < regime_weight(body, frequencies[0]) < 1 and regime_weight(body, frequencies[1]) == 1
    y = result.network.to_y()
    errors = []
    for k, f in enumerate(frequencies):
        eps = model.relative_permittivity(f)
        # The body admittance reads the embedding layer's model at its own frequency.
        assert background_permittivity(prepared.admittance.stack, *body.z_range, f) == \
            background_permittivity(strip_stack(eps), *body.z_range, f)
        frozen, _ = admittance_strip(strip_stack(eps))
        reference = run(frozen, (f,), config)
        errors.append(relative(y[k], reference.network.to_y()[0]))
    record_property("frozen_relative", json.dumps(errors))
    record_property("R_ohm", network_metrics(result)["R_ohm"])
    assert max(errors) <= 1e-6
    assert torch.linalg.svdvals(result.network.s).amax().item() <= 1 + 1e-9
    again, currents = solve_prepared_network(prepared)
    assert torch.equal(again.s, result.network.s) and torch.equal(currents, mw.mom.surface_currents(result).currents)


# ---------------------------------------------------------------- (a3) GR1 gradients x polygon path


NOMINAL = {"sheet_sigma": SIGMA, "via_sigma": SIGMA, "thickness": 2 * U, "rms_height": .5e-6,
           "sphere_radius": .5e-6}


def prism_loop_scene(**parameters):
    """loop_scene with its Box vias replaced by octagonal Prism vias of the same height (polygon path)."""
    scene = loop_scene(**parameters)
    polygon = mw.Scene(background=scene.background)
    from witwin.core import Box
    for structure in scene.structures:
        geometry = structure.geometry
        if isinstance(geometry, Box):
            position = geometry.position.detach().to(torch.float64)
            height = geometry.size.detach().to(torch.float64)[2]
            geometry = mw.geometry.Prism(position=position, radius=torch.tensor(4 * U, dtype=torch.float64),
                                         height=height, num_sides=8, axis="z")
        polygon = polygon.updated(structures=(*polygon.structures, mw.Structure(geometry, structure.material)))
    polygon = polygon.updated(ports=(*polygon.ports, *scene.ports))
    return polygon


GRADIENT_FIXTURES = {
    "triangle": (loop_scene, lambda: layered(loop_config("sheet"), planar_mesher="triangle")),
    "prism": (prism_loop_scene, lambda: loop_config("sheet")),
}


def derivative(y, parameter):
    parts = []
    for value in y.reshape(-1):
        real, = torch.autograd.grad(value.real, parameter, retain_graph=True)
        imag, = torch.autograd.grad(value.imag, parameter, retain_graph=True)
        parts.append(torch.complex(real, imag))
    return torch.stack(parts).reshape(y.shape)


@pytest.mark.parametrize("fixture,name", [(f, n) for f in GRADIENT_FIXTURES for n in NOMINAL])
def test_polygon_gradients_match_finite_differences(fixture, name, record_property):
    """GR1 central-FD gate (tests/mom/layered/test_layered_gradients.py::fd_bound) on the planar mesher."""
    from witwin.maxwell.mom.layered.lowfreq import scaled_system
    from witwin.maxwell.mom.layered.operator import BACKWARD
    make_scene, make_config = GRADIENT_FIXTURES[fixture]
    scale = torch.tensor(1., dtype=torch.float64, device="cuda", requires_grad=True)
    result = run(make_scene(**{name: scale * NOMINAL[name]}), FREQUENCIES, make_config())
    assert result.network.metadata["mesher"] == "planar_conforming_delaunay_v1"
    y = result.network.to_y()
    gradient = derivative(y, scale)
    kappa = torch.linalg.cond(scaled_system(result.solver.prepared).matrix).real
    h = 1e-3

    def admittance(factor):
        with torch.no_grad():
            return run(make_scene(**{name: factor * NOMINAL[name]}), FREQUENCIES, make_config()).network.to_y()
    fd = (admittance(1 + h) - admittance(1 - h)) / (2 * h)
    fd2 = (admittance(1 + 2 * h) - admittance(1 - 2 * h)) / (4 * h)
    truncation = (fd2 - fd).abs() / 3
    noise = (2 * kappa * BACKWARD * y.detach().abs().amax((-1, -2)) / h)[:, None, None]
    bound = 2 * (truncation + noise)
    error = (gradient - fd).abs()
    scale_g = gradient.abs().max()
    record_property("max_relative_error", (error.max() / scale_g).item())
    record_property("max_relative_bound", (bound.max() / scale_g).item())
    assert scale_g > 0
    assert bool((error <= bound).all())
    assert bound.max() <= 1e-3 * scale_g


def test_polygon_gradients_hmatrix_match_dense(record_property):
    records = {}
    for backend in ("dense", "hmatrix"):
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        rough = torch.tensor(.5e-6, dtype=torch.float64, requires_grad=True)
        config = layered(loop_config("sheet"), operator_backend=backend)
        result = run(prism_loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5, rms_height=rough), FREQUENCIES, config)
        y = result.network.to_y()
        rows = [torch.stack(torch.autograd.grad(part, (sigma, rough), retain_graph=True))
                for value in y.reshape(-1) for part in (value.real, value.imag)]
        records[backend] = (y.detach(), torch.stack(rows))
        del result
    (y_dense, g_dense), (y, g) = records["dense"], records["hmatrix"]
    errors = ((g - g_dense).abs().amax(0) / g_dense.abs().amax(0)).tolist()
    record_property("gradient_relative_error", json.dumps(errors))
    assert max(relative(y, y_dense)) <= 1e-4
    assert max(errors) <= 1e-4


# ---------------------------------------------------------------- (b1) polygon x hmatrix


def hmatrix_cases():
    from tests.mom.layered.test_layered_planar_network import two_rings, via_loop
    from tests.support.rfic_planar_fixtures import spiral_scene
    spiral = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=5 * UM, max_edge_length_m=40 * UM,
                                                          edge_cell_m=4 * UM, conductor_model="sheet"))
    return {
        "rings": (lambda: two_rings(True), octagon_config(20.).layered, (1e8, 1e10)),
        "octagon_pair": (lambda: octagon_pair_scene(dispersive_stack(mw.media.WidebandDebye.from_loss_tangent(**FR4))),
                         octagon_config().layered, (1e9, 2e10)),
        "landing_junction": (lambda: via_loop("landing")[0], octagon_config(10.).layered, (1e6, 1e9)),
        "octagonal_spiral": (lambda: spiral_scene(True, lossy=True), spiral.layered, (1e9, 1e10)),
    }


@pytest.mark.parametrize("case", ["rings", "octagon_pair", "landing_junction", "octagonal_spiral"])
def test_polygon_hmatrix_matches_dense(case, record_property):
    make_scene, options, frequencies = hmatrix_cases()[case]
    dense = run(make_scene(), frequencies, mw.mom.MoMConfig(layered=options))
    config = mw.mom.MoMConfig(layered=dataclasses.replace(options, operator_backend="hmatrix", hmatrix_exact_check=True))
    result = run(make_scene(), frequencies, config)
    metadata = result.network.metadata
    errors = relative(result.network.to_y(), dense.network.to_y())
    record_property("agreement", json.dumps(dict(
        unknowns=metadata["unknowns"], relative_error=errors, exact=metadata["compression_exact_check"],
        bound=metadata["admittance_bound"], steps=metadata["block_gmres_steps"], hmatrix=metadata["hmatrix"]),
        default=str))
    assert metadata["operator_backend"] == "hmatrix"
    assert metadata["mesher"] == "planar_conforming_delaunay_v1"
    assert metadata["mesh_sha256"] == dense.network.metadata["mesh_sha256"]
    assert max(errors) <= 1e-4
    assert max(metadata["compression_exact_check"]) <= 1e-5
    assert max(metadata["admittance_bound"]) <= 1e-5


def test_polygon_hmatrix_partition_and_determinism(record_property):
    from witwin.maxwell.mom.layered import hmatrix as hm
    from witwin.maxwell.mom.layered.matrix_free import build_near_structure
    make_scene, options, _ = hmatrix_cases()["octagonal_spiral"]
    config = mw.mom.MoMConfig(layered=dataclasses.replace(options, operator_backend="hmatrix"))
    prepared = mw.Simulation.mom(make_scene(), frequencies=(1e10,), config=config).prepare().solver.prepared
    structure = build_near_structure(prepared)
    clusters = hm.build_cluster_blocks(prepared, structure)
    stats = clusters.stats
    record_property("clusters", json.dumps(stats, default=str))
    assert stats["unordered_pairs_covered"] == stats["unordered_pairs"]
    assert stats["near_pairs_in_leaves"] == stats["near_pairs_unordered"]
    # Two metal planes and the via walls, with compressed cross-plane blocks.
    assert stats["z_classes"] >= 3 and stats["cross_class_blocks"] > 0
    assert stats["admissible_face_pairs"] > stats["leaf_far_pairs"]
    first = solve_prepared_network(prepared)
    again = solve_prepared_network(prepared)
    assert torch.equal(again[0].s, first[0].s) and torch.equal(again[1], first[1])


# ---------------------------------------------------------------- (b2) polygon x edge ports


def square_loop_scene(aperture):
    """PEC square strip loop (outer 200 um, strip 10 um, Manhattan) in vacuum with a 10 um gap at x = 95..105.

    ``aperture`` True declares the port as the full-width edge port across the gap.
    """
    loop = shapely.box(0., 0., 200., 200.).difference(shapely.box(10., 10., 190., 190.))
    loop = loop.difference(shapely.box(95., -1., 105., 11.))
    aperture = ((100., 5., 100.), (10., 10., 0.)) if aperture else None
    scene = mw.Scene(background=LayerStack((Layer("air", 1e-3),), HalfSpace(), HalfSpace()), structures=(
                polygon_structure(loop, 100., mw.Material.pec()),
            ), ports=(
                gap_port("p", (105., 5., 100.), (95., 5., 100.), "x", aperture),
            ))
    return scene


def edge_config(mesher, bridge=4.):
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=bridge * UM, max_edge_length_m=20 * UM,
                                                        edge_cell_m=2. * UM, planar_mesher=mesher))


def loop_inductance(result):
    frequencies = result.network.frequencies
    return (-result.network.to_z()[:, 0, 0].imag / (2 * math.pi * frequencies)).tolist()


def test_polygon_edge_port_matches_rectangle_path_and_delta_gap(record_property):
    frequencies = (1e8, 1e9)
    planar = run(square_loop_scene(True), frequencies, edge_config("triangle"))
    rectangle = run(square_loop_scene(True), frequencies, edge_config("auto"))
    record = planar.network.metadata["ports"][0]
    assert planar.network.metadata["mesher"] == "planar_conforming_delaunay_v1"
    assert "mesher" not in rectangle.network.metadata
    assert (record["kind"], record["width_m"]) == ("edge", pytest.approx(10 * UM, rel=1e-12))
    assert record == {**rectangle.network.metadata["ports"][0], "width_m": record["width_m"]}
    # Edge port == delta-gap port of the aperture width (identical bridge and cut) on each mesher.
    for mesher, edge in (("triangle", planar), ("auto", rectangle)):
        delta = run(square_loop_scene(False), frequencies, edge_config(mesher, 10.))
        assert torch.equal(edge.network.s, delta.network.s)
    # Different meshers of the same layout: B1 measured the square spiral through both within 0.28%.
    values = [loop_inductance(planar), loop_inductance(rectangle)]
    across = [a / b - 1 for a, b in zip(*values)]
    record_property("inductance_H", json.dumps(values))
    record_property("planar_vs_rectangle", json.dumps(across))
    assert max(map(abs, across)) <= 1e-2
    again = run(square_loop_scene(True), frequencies, edge_config("triangle"))
    assert torch.equal(again.network.s, planar.network.s)


# ---------------------------------------------------------------- (b3) surface_admittance x edge ports


def split_strip_scene(edge):
    """Copper strip x = 0..200, y = -5..5 um (2 um thick) split by a 10 um gap at x = 95..105 um.

    PEC end posts and a continuous PEC return close the loop. The port spans the gap; as an edge
    port its PEC block covers both copper end faces completely (sidewalls flush with the strip).
    """
    stack = LayerStack((Layer("air", 1e-3),), HalfSpace(), HalfSpace())
    scene = mw.Scene(background=stack)
    copper = mw.media.LossyMetalMedium(conductivity=SIGMA)
    pec = mw.Material.pec()
    scene = scene.updated(structures=(
        *scene.structures,
        *(polygon_structure(shapely.box(x0, y0, x1, y1), 100., medium, 2.) for x0, x1, y0, y1, medium in ((0., 95., -5., 5., copper), (105., 200., -5., 5., copper),
                                   (-10., 0., -115., 5., pec), (200., 210., -115., 5., pec),
                                   (0., 200., -115., -105., pec))),
    ))
    aperture = ((100., 0., 100.), (10., 10., 0.)) if edge else None
    scene = scene.updated(ports=(*scene.ports, gap_port("p", (105., 0., 100.), (95., 0., 100.), "x", aperture)))
    return scene


def admittance_config(bridge=10.):
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=bridge * UM, max_edge_length_m=40 * UM,
                                                        edge_cell_m=4 * UM, conductor_model="surface_admittance"))


def test_surface_admittance_edge_port_dc_resistance(record_property):
    frequencies = (1.0, 1e9)
    simulation = mw.Simulation.mom(split_strip_scene(True), frequencies=frequencies, config=admittance_config(4.))
    simulation = simulation.prepare()
    result = simulation.run()
    metadata = result.network.metadata
    record = metadata["ports"][0]
    assert (record["kind"], record["bridge"]) == ("edge", "ideal_pec_block")
    assert record["width_m"] == pytest.approx(10 * UM, rel=1e-12)
    (block,) = metadata["port_blocks"]
    assert block[2:4] == (-5 * UM, 5 * UM)
    metrics = network_metrics(result)
    expected = 190e-6 / (SIGMA * 10e-6 * 2e-6)
    record_property("metrics", json.dumps(metrics))
    record_property("dc_relative", metrics["R_ohm"][0] / expected - 1)
    assert abs(metrics["R_ohm"][0] / expected - 1) < 1e-6
    assert torch.linalg.svdvals(result.network.s).amax().item() <= 1 + 1e-12
    # Same block as the delta-gap port whose bridge_width is the strip width.
    delta = run(split_strip_scene(False), frequencies, admittance_config(10.))
    assert torch.equal(result.network.s, delta.network.s)
    again, currents = solve_prepared_network(simulation.solver.prepared)
    assert torch.equal(again.s, result.network.s) and torch.equal(currents, mw.mom.surface_currents(result).currents)


def test_surface_admittance_edge_port_on_the_a1c_fixture(record_property):
    """The A1c strip fixture with its delta-gap port declared as the equivalent full-width edge port."""
    scene, config = strip_fixture(model="surface_admittance", maximum_um=40., edge_um=4.)
    edge = mw.Scene(background=scene.background)
    edge = edge.updated(structures=(*edge.structures, *scene.structures))
    port = scene.ports[0]
    edge = edge.updated(ports=(
        *edge.ports,
        mw.LumpedPort("p", positive=port.positive, negative=port.negative, voltage_path=mw.AxisPath("x"),
                      reference_impedance=50., voltage_aperture=mw.Box(
                          position=(100 * UM, -110 * UM, 100 * UM), size=(10 * UM, 10 * UM, 0.))),
    ))
    frequencies = (1e9,)
    reference = run(scene, frequencies, config)
    result = run(edge, frequencies, layered(config, bridge_width_m=4 * UM))
    assert result.network.metadata["ports"][0]["kind"] == "edge"
    assert torch.equal(result.network.s, reference.network.s)
    assert torch.equal(mw.mom.surface_currents(result).currents, mw.mom.surface_currents(reference).currents)
