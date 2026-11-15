"""B2 (R7-1): arbitrary-outline vias through the core ExtrudedPolygon, and via arrays against a quasi-static reference.

Host contracts: ExtrudedPolygon footprints compile to exact counter-clockwise PolygonVia outlines and route to the
planar mesher; non-z or tilted extrusions are refused. CUDA gates: a regular-octagon ExtrudedPolygon via loop is
bitwise the Prism loop; an L-shaped via bar loop has KCL <= 1e-8 and L within 3% of the partial-inductance loop
reference with the bar's 2D logarithmic-capacity radius; a 2x2 via array inside an 8-via fence between PEC pads,
de-embedded by the height difference, gives R and L within 2% of the independent 2D BEM with the same Leontovich
wall impedance (tests/support/via_array_reference.py), with KCL on every wall.
"""

import gc
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import ExtrudedPolygon
from tests.support.layered_via_references import MU0
from tests.support.rfic_planar_fixtures import UM, gap_port, polygon_structure, vacuum
from tests.support.via_array_reference import converged, regular_outline, solve

shapely = pytest.importorskip("shapely")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
L_BAR = ((-3.0, -3.0), (3.0, -3.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (-3.0, 3.0))


@pytest.fixture(autouse=True)
def release():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_extruded_polygon_via_record_is_exact_and_counter_clockwise():
    from witwin.maxwell.mom.layered.planar import via_record
    planes = (0.0, 3 * UM)
    vertices = torch.tensor(L_BAR, dtype=torch.float64) * UM
    for authored in (vertices, vertices.flip(0)):
        geometry = ExtrudedPolygon(authored, height=torch.tensor(3 * UM, dtype=torch.float64),
                                   position=torch.tensor((10 * UM, -5 * UM, 1.5 * UM), dtype=torch.float64))
        record = via_record(geometry, planes)
        outline = np.array(record.outline)
        assert (record.z0, record.z1) == planes
        assert np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1]) > 0
        expected = vertices.numpy() + np.array((10 * UM, -5 * UM))
        assert {tuple(p) for p in outline} == {tuple(p) for p in expected}
    rotated = ExtrudedPolygon(vertices, height=3 * UM, position=(0, 0, 1.5 * UM), rotation=(0.0, 0.0, 0.4))
    turned = np.array(via_record(rotated, planes).outline)
    c, s = math.cos(0.4), math.sin(0.4)
    assert np.allclose(turned, vertices.numpy() @ np.array(((c, s), (-s, c))), rtol=0, atol=1e-6 * UM)


@pytest.mark.parametrize("options,message", [
    (dict(axis="x"), "rotated about z only"),
    (dict(rotation=(0.3, 0.0, 0.0)), "rotated about z only"),
])
def test_extruded_polygon_via_refusals(options, message):
    from witwin.maxwell.mom.layered.planar import via_record
    geometry = ExtrudedPolygon(torch.tensor(L_BAR) * UM, height=3 * UM, position=(0, 0, 1.5 * UM), **options)
    with pytest.raises(NotImplementedError, match=message):
        via_record(geometry, (0.0, 3 * UM))


def test_extruded_polygon_via_routes_to_the_planar_mesher():
    from witwin.maxwell.mom.layered.planar_mesh import PolygonVia
    from witwin.maxwell.mom.layered.runtime import compile_layered_scene
    scene, _ = bar_loop("l_bar")
    compiled = compile_layered_scene(scene)
    vias = compiled[5]
    assert len(vias) == 2 and all(isinstance(v, PolygonVia) and len(v.outline) == 6 for v in vias)


def test_via_array_reference_closed_forms_and_pec_limit():
    """Independence audit of the 2D via-array reference (R7-1). Regular N-gons stand in for circles:
    (a) coaxial line with the Leontovich impedance: R' = Rs (1/a + 1/b)/2pi and L' = mu0/2pi ln(b/a) + R'/omega
        with errors falling as N^-2 (N = 48/96/192: R' 1.8e-3/4.4e-4/1.1e-4, L' 3.7e-5/4.6e-6/5.8e-7);
    (b) PEC-limit two-wire line (proximity factor x/sqrt(x^2-1)): R' 2.9e-4/7.2e-5/1.8e-5;
    (c) Zs -> 0: the impedance solution approaches the PEC-limit R' and L'_ext + R'/omega (gate array, Rs scaled by
        1e-1..1e-4: R' -6.3e-2, -1.6e-2, -2.4e-3, +1.1e-4; L' -2.3e-4 .. 5e-9).
    The gate uses the impedance model because the PEC-limit current assumes omega L' >> R'; at copper
    Rs/(omega L') ~ 0.1 the wall resistance relaxes proximity crowding (2x2 array: R' 197.4 -> 152.9 ohm/m)."""
    from tests.support.via_array_reference import MU0 as mu0, closed_form_checks
    omega, rs = 2 * math.pi * 1e9, math.sqrt(math.pi * 1e9 * mu0 / 5.8e7)
    errors = [closed_form_checks(n, omega=omega, surface_resistance=rs) for n in (48, 96)]
    for key in ("coax_R", "coax_L", "pair_R", "pair_L"):
        assert abs(errors[1][key]) < abs(errors[0][key]) / 3.5, key
    assert abs(errors[1]["coax_R"]) < 5e-4 and abs(errors[1]["coax_L"]) < 1e-5
    outlines = [regular_outline((sx * 6 * UM, 0.0), 4 * UM, 8) for sx in (-1, 1)] + [regular_outline((0.0, 0.0), 40 * UM, 64)]
    deviations = []
    for scale in (1e-2, 1e-3):
        z = solve(outlines, [1, 1, -1], panels=6, surface_resistance=rs * scale, omega=omega)
        p = solve(outlines, [1, 1, -1], panels=6, surface_resistance=rs * scale)
        deviations.append((abs(z["R_per_m"] / p["R_per_m"] - 1), abs(z["L_per_m"] / (p["L_per_m"] + p["R_per_m"] / omega) - 1)))
    assert deviations[1][0] < deviations[0][0] / 5 and deviations[1][0] < 1e-2
    assert deviations[1][1] < 1e-5


def via_geometry(kind, x, z0, z1):
    """Via of ``kind`` centred at (x, 0): an octagon as Prism or ExtrudedPolygon, or the L-shaped bar."""
    position = torch.tensor((x * UM, 0.0, (z0 + z1) / 2 * UM), dtype=torch.float64)
    height = torch.tensor((z1 - z0) * UM, dtype=torch.float64)
    if kind == "prism":
        return mw.geometry.Prism(position=position, radius=torch.tensor(3. * UM, dtype=torch.float64),
                                 height=height, num_sides=8, axis="z")
    if kind == "octagon":
        # The same fp64 vertices via_record builds for the Prism, so the PolygonVia is identical.
        angles = 2 * math.pi * np.arange(8) / 8
        local = np.stack((3. * UM * np.cos(angles), 3. * UM * np.sin(angles)), 1)
        return ExtrudedPolygon(torch.tensor(local), height=height, position=position, axis="z")
    return ExtrudedPolygon(torch.tensor(L_BAR, dtype=torch.float64) * UM, height=height, position=position, axis="z")


def bar_loop(kind):
    """Vertical rectangular loop (length 200 um, planes 100/300 um) with one via of ``kind`` per side."""
    length, width, gap, za, zb = 200., 10., 10., 100., 300.
    scene = mw.Scene(background=vacuum())
    pec = mw.Material.pec()
    scene = scene.updated(structures=(
        *scene.structures,
        *(polygon_structure(shapely.box(box[0], box[2], box[1], box[3]), z, pec) for box, z in (((-width, length + width, -width / 2, width / 2), zb),
                   ((-width, length / 2 - gap / 2, -width / 2, width / 2), za),
                   ((length / 2 + gap / 2, length + width, -width / 2, width / 2), za))),
    ))
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(via_geometry(kind, x, za, zb), pec) for x in (0., length)),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        gap_port("loop", (length / 2 + gap / 2, 0, za), (length / 2 - gap / 2, 0, za), "x"),
    ))
    return scene, (length, zb - za, width)


def loop_config():
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=10. * UM, max_edge_length_m=20. * UM,
                                                                edge_cell_m=2. * UM))


def inductance(result):
    frequency = result.network.frequencies[0].item()
    return -result.network.to_z()[0, 0, 0].imag.item() / (2 * math.pi * frequency)


def wall_currents(prepared, currents, z_range, footprint, tolerance):
    """Signed vertical current through each horizontal wall-edge level inside ``footprint`` (shapely, metres)."""
    layout = prepared.solver.prepared.layout
    basis, mesh = layout.basis, layout.mesh
    a, b = mesh.vertices[basis.edges[:, 0]], mesh.vertices[basis.edges[:, 1]]
    flow = mesh.centroids[basis.minus_faces] - mesh.centroids[basis.plus_faces]
    terminal = (prepared.solver.prepared.excitation[:, 0] * currents[0, :, 0]).sum()
    near = shapely.buffer(footprint, tolerance)
    inside = torch.tensor([bool(near.covers(shapely.Point(p[0], p[1])) and near.covers(shapely.Point(q[0], q[1])))
                           for p, q in zip(a[:, :2].tolist(), b[:, :2].tolist())], device=a.device)
    tube = (a[:, 2] == b[:, 2]) & (a[:, 2] > z_range[0]) & (a[:, 2] < z_range[1]) & inside
    levels = a[tube, 2].unique()
    values = [(torch.sign(flow[tube & (a[:, 2] == z), 2]) * basis.edge_lengths[tube & (a[:, 2] == z)]
               * currents[0, tube & (a[:, 2] == z), 0]).sum() / terminal for z in levels]
    return torch.stack(values), levels


@requires_cuda
@pytest.mark.cuda
def test_regular_octagon_extruded_polygon_via_loop_is_bitwise_prism(record_property):
    results = {}
    for kind in ("prism", "octagon"):
        scene, _ = bar_loop(kind)
        result = mw.Simulation.mom(scene, frequencies=(1e6,), config=loop_config()).run()
        results[kind] = (result.network.s, mw.mom.surface_currents(result).currents,
                         result.network.metadata["mesh_sha256"])
    record_property("mesh_sha256", results["prism"][2])
    assert results["prism"][2] == results["octagon"][2]
    assert torch.equal(results["prism"][0], results["octagon"][0])
    assert torch.equal(results["prism"][1], results["octagon"][1])


@requires_cuda
@pytest.mark.cuda
def test_l_shaped_via_bar_loop_kcl_and_inductance(record_property):
    """L-shaped via bars (ExtrudedPolygon): KCL <= 1e-8 on every wall level at 1 MHz and L within 3% of the
    partial-inductance loop with the bar's 2D logarithmic-capacity radius (PEC limit, BEM)."""
    scene, (length, height, width) = bar_loop("l_bar")
    prepared = mw.Simulation.mom(scene, frequencies=(1e6,), config=loop_config()).prepare()
    result = prepared.run()
    currents = mw.mom.surface_currents(result).currents
    kcl = []
    for x in (0., length):
        footprint = shapely.Polygon([((u + x) * UM, v * UM) for u, v in L_BAR])
        values, levels = wall_currents(prepared, currents, (100 * UM, 300 * UM), footprint, 1e-9 * UM)
        assert len(levels) >= 3
        kcl.append((values.abs() - 1).abs().max().item())
    # Equivalent radius: L' of the bar inside a far coaxial circle of radius b is mu0/2pi ln(b/r_eq).
    far = 2000 * UM
    coax = solve([tuple((u * UM, v * UM) for u, v in L_BAR), regular_outline((0, 0), far, 256)], [1, -1], panels=24)
    r_eq = far / math.exp(coax["L_per_m"] * 2 * math.pi / MU0)
    side = r_eq * 4 * math.pi ** 1.5 / math.gamma(.25) ** 2
    from tests.support.layered_via_references import rectangular_loop_inductance
    expected = rectangular_loop_inductance(length * UM, height * UM, width * UM, side)
    value = inductance(result)
    record_property("inductance_H", value)
    record_property("reference_H", expected)
    record_property("equivalent_radius_m", r_eq)
    record_property("kcl_max", max(kcl))
    record_property("unknowns", result.network.metadata["unknowns"])
    assert max(kcl) <= 1e-8
    assert abs(value / expected - 1) <= .03


VIA_ARRAY_HEIGHTS = (60.0, 120.0)


@pytest.fixture(scope="module")
def via_array_runs():
    """2x2 octagonal via array (ExtrudedPolygon) inside an 8-via fence between PEC pads, lossy copper vias,
    edge cell 4 um, at two via heights, plus the converged 2D BEM (Leontovich wall impedance) of the via cross-sections."""
    from tests.support.via_array_fixtures import measure_coax, via_outlines
    frequencies, sigma = (1e6, 1e9), 5.8e7
    runs = [measure_coax(h, "array", 4.0, frequencies, sigma) for h in VIA_ARRAY_HEIGHTS]
    nets = [s for _, s in via_outlines("array")]
    rs = math.sqrt(math.pi * frequencies[1] * MU0 / sigma)
    omega = 2 * math.pi * frequencies[1]
    reference = converged([[(x * UM, y * UM) for x, y in o] for o, _ in via_outlines("array")], nets,
                          surface_resistance=rs, omega=omega)
    dh = (VIA_ARRAY_HEIGHTS[1] - VIA_ARRAY_HEIGHTS[0]) * UM
    dz = runs[1]["z"][1] - runs[0]["z"][1]
    # exp(-i omega t): Z = R - i omega L; the reference is the 2D impedance of the same Leontovich wall model.
    return dict(runs=runs, nets=nets, reference=reference,
                r_error=dz.real.item() / (dh * reference["R_per_m"]) - 1,
                l_error=-dz.imag.item() / omega / (dh * reference["L_per_m"]) - 1)


@requires_cuda
@pytest.mark.cuda
def test_via_array_r_l_kcl_and_repeat(via_array_runs, record_property):
    """B2 acceptance: the height difference 60 -> 120 um de-embeds pads, gap and port. dR and dL at 1 GHz are
    within 2% of (h2-h1) R' and (h2-h1) L' of the independent 2D BEM with the same Leontovich wall impedance
    (converged to < 1e-4). Measured (edge 4 um): dR -1.2%, dL +1.5% (edge 2 um: -1.2%, +1.4%). Every via wall
    carries a z-constant current at 1 MHz and the array carries the port current (KCL <= 1e-8); a repeat is
    bitwise. The per-via split approaches the BEM's equal split as about 2.7 um / h (end coupling through
    the pads), so it is recorded, not gated. History: the first gate used the PEC-limit reference
    (R' 197.4 ohm/m, L'_ext + R'/omega 3.237e-7 H/m) and failed R by -23.5% at both meshes; the impedance
    reference (152.9 ohm/m, 3.206e-7 H/m) solves the via walls' own Leontovich model, see
    test_via_array_reference_closed_forms_and_pec_limit and <P>/r7-1/via_array_reference_evidence.json."""
    from tests.support.via_array_fixtures import measure_coax
    runs, reference = via_array_runs["runs"], via_array_runs["reference"]
    nets = via_array_runs["nets"]
    kcl, split = [], []
    for run in runs:
        walls = run["walls"][0]
        kcl.append(max((w - w[:1]).abs().max().item() for w in walls))
        kcl.append(abs(abs(sum(w[0] for w, s in zip(walls, nets) if s > 0)) - 1).item())
        middle = torch.stack([w[len(w) // 2].real for w in run["walls"][1]])
        split.append((middle.abs() / torch.tensor(reference["via_currents"]).abs() - 1).abs().max().item())
    record_property("R_error", via_array_runs["r_error"])
    record_property("L_error", via_array_runs["l_error"])
    record_property("reference_convergence", (reference["R_change"], reference["L_change"]))
    record_property("kcl_max", max(kcl))
    record_property("split_deviation_max", split)
    record_property("unknowns", [run["unknowns"] for run in runs])
    assert max(reference["R_change"], reference["L_change"]) < 1e-4
    assert abs(via_array_runs["r_error"]) <= .02 and abs(via_array_runs["l_error"]) <= .02
    assert max(kcl) <= 1e-8
    again = measure_coax(VIA_ARRAY_HEIGHTS[0], "array", 4.0, (1e6, 1e9))
    assert torch.equal(again["s"], runs[0]["s"]) and torch.equal(again["currents"], runs[0]["currents"])
