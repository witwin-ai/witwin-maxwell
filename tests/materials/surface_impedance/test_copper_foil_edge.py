"""Finite-thickness free-edge impedance of zero-thickness CopperFoil supports.

The zero-thickness foil's Yee rim trace carries the singular edge current of
a strip. Without a thickness-aware edge impedance its loss is 38-57% below an
independent 35 um cross-section reference (and grows ~log under refinement).
"""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import (
    _EDGE_C0, _EDGE_G, _EDGE_LOG2_ASPECT, compile_copper_faces, copper_edge_half_plane_constants,
    copper_edge_impedance_factor, copper_edge_stopping_distance, copper_free_edge_sides,
)
from witwin.maxwell.scene import prepare_scene

MU0 = 1.25663706212e-6
EPS0 = 8.8541878128e-12
SIGMA = 5.8e7
# Independent cross-section reference (pcb/line-accuracy
# benchmark/line_cross_section_reference.py, reference.json): homogeneous
# eps_r 2.25 stripline, W = 10 mm, b = 10 mm, 35 um Cu, Richardson alpha_c.
STRIPLINE_REFERENCE_ALPHA_C_1GHZ = 11.448e-3


def test_stopping_distance_matches_the_rectangular_edge_conformal_map():
    """Thick-edge surface-current loss equals the 1/sqrt(s) integral cut at Delta.

    z = -(t/pi)[w sqrt(w^2-1) - acosh(w)] + i t/2 maps the upper half plane to
    the exterior of a semi-infinite slab of thickness t. With the uniform
    half-plane potential, the per-face current is 1/|dz/dw|, so the loss within
    the image of |w| < W is (pi/2t) * integral dw/|sqrt(w^2-1)|. The
    zero-thickness edge with K^2 = pi/t loses (pi/2t) ln(a/Delta) on [Delta, a].
    """
    thickness = 35e-6
    for limit in (1e2, 1e3, 1e4):
        side = quad(lambda w: 1 / math.sqrt(1 - w * w), 0, 1)[0]
        face = quad(lambda w: 1 / math.sqrt(w * w - 1), 1, limit, limit=200)[0]
        thick = 2 * (side + face)
        distance = thickness / math.pi * (limit * math.sqrt(limit**2 - 1) - math.acosh(limit))
        thin = math.log(distance / copper_edge_stopping_distance(thickness))
        assert thick == pytest.approx(thin, abs=4 / limit)


@pytest.mark.parametrize("aspect", (1., 2.))
def test_tabulated_half_plane_constants_match_an_independent_discrete_solve(aspect):
    from benchmark.copper_foil_edge_constants import edge_constants
    c0, g = edge_constants(aspect, n=160)
    table_c0, table_g = copper_edge_half_plane_constants(torch.tensor([aspect], dtype=torch.float64))
    assert float(table_c0) == pytest.approx(c0, rel=2e-4)
    assert float(table_g) == pytest.approx(g, abs=5e-3)


def test_edge_factor_is_geometric_monotone_and_reduces_to_the_bare_rim_at_fine_cells():
    nodes = torch.tensor(_EDGE_LOG2_ASPECT, dtype=torch.float64)
    c0, g = copper_edge_half_plane_constants(2 ** nodes)
    torch.testing.assert_close(c0, torch.tensor(_EDGE_C0, dtype=torch.float64), rtol=1e-14, atol=0)
    torch.testing.assert_close(g, torch.tensor(_EDGE_G, dtype=torch.float64), rtol=1e-14, atol=0)
    widths = torch.tensor((1e-6, 1e-4, 6.25e-4, 1.25e-3, 2.5e-3), dtype=torch.float64)
    kappa = copper_edge_impedance_factor(35e-6, widths, torch.ones_like(widths), torch.ones_like(widths))
    assert float(kappa[0]) == 1.
    assert bool(torch.all(kappa[1:] > kappa[:-1]))
    expected = 1 + (math.log(2.5e-3 * 4 * math.pi * math.exp(math.pi) / 35e-6) - _EDGE_G[6]) / _EDGE_C0[6]**2
    assert float(kappa[-1]) == pytest.approx(expected, rel=1e-14)
    two = copper_edge_impedance_factor(35e-6, widths[-1:], torch.ones(1), torch.full((1,), 2.))
    assert float(two) - 1 == pytest.approx(2 * (expected - 1), rel=1e-14)


def test_edge_model_selection_follows_the_good_conductor_regime():
    band = (.8e9, 1.2e9)
    thick = mw.CopperFoil(thickness=35e-6, frequency_range=band)
    assert thick.edge_model == mw.CopperFoil.DEFAULT_EDGE_MODEL == "auto" and thick.edge_correction_active
    assert float(thick.edge_weight()) == 1.
    assert not mw.CopperFoil(thickness=35e-6, frequency_range=band, edge_model="none").edge_correction_active
    assert not mw.CopperFoil(thickness=2e-6, frequency_range=band).edge_correction_active
    with pytest.raises(ValueError, match="3 skin depths"):
        mw.CopperFoil(thickness=2e-6, frequency_range=band, edge_model="finite_thickness")
    with pytest.raises(ValueError, match="edge_model"):
        mw.CopperFoil(thickness=35e-6, frequency_range=band, edge_model="rounded")


def _skin_depth(frequency):
    return math.sqrt(1 / (math.pi * frequency * MU0 * SIGMA))


def test_broadband_gating_uses_the_upper_band_edge_and_warns_for_a_thin_low_end():
    # 35 um is thinner than 3 skin depths below ~45 MHz but thick at 1 GHz.
    with pytest.warns(UserWarning, match="lower part has thickness < 3"):
        broadband = mw.CopperFoil(thickness=35e-6, frequency_range=(1e6, 1e9))
    assert float(broadband.edge_weight()) == 1.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 4e9))
        mw.CopperFoil(thickness=1e-6, frequency_range=(1e6, 1e9))  # weight 0: nothing applied


def test_edge_weight_is_a_continuous_differentiable_blend_over_two_to_four_skin_depths():
    band = (1e8, 1e9)
    delta = _skin_depth(band[1])
    ratios = torch.tensor([1., 2., 2.5, 3., 3.5, 4., 6.], dtype=torch.float64)
    weights = []
    for ratio in ratios:
        thickness = torch.tensor(float(ratio) * delta, dtype=torch.float64, requires_grad=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            foil = mw.CopperFoil(thickness=thickness, frequency_range=band, mode_count=64)
        weight = foil.edge_weight(thickness=thickness)
        weights.append(float(weight))
        if 2 < float(ratio) < 4:
            weight.backward()
            s = (float(ratio) - 2) / 2
            assert float(thickness.grad) * delta == pytest.approx(6 * s * (1 - s) / 2, rel=1e-9)
    expected = [0., 0., .15625, .5, .84375, 1., 1.]
    assert weights == pytest.approx(expected, abs=1e-12)
    # kappa follows the blend continuously (no 1 -> 2.5 jump at a threshold).
    kappas = [float(copper_edge_impedance_factor(35e-6, torch.tensor([.00125]), torch.ones(1), torch.ones(1), w))
              for w in weights]
    assert all(b >= a for a, b in zip(kappas, kappas[1:])) and kappas[0] == 1.
    assert max(b - a for a, b in zip(kappas, kappas[1:])) < .6


def test_feature_width_limits_are_measured_and_warned():
    from witwin.maxwell.compiler.copper_foil import copper_edge_feature_cells, copper_edge_geometry, warn_copper_edge_features
    scene = _strip_scene()
    faces = compile_copper_faces(scene, device="cpu")
    strip = next(face for face in faces if face.name == "strip" and face.component == 0)
    extension = next(face for face in faces if face.name == "extension" and face.component == 0)
    for face, expected in ((strip, (4, 1)), (extension, (3, 1))):
        sides = copper_free_edge_sides(scene, face, face.indices, faces)
        assert copper_edge_feature_cells(scene, face, face.indices, faces, sides) == expected
    geometry = copper_edge_geometry(scene, extension, extension.indices, faces)
    with pytest.warns(UserWarning) as record:
        warn_copper_edge_features(scene, extension, extension.indices, faces, geometry)
    messages = " ".join(str(item.message) for item in record)
    assert "strip is 3 tangential cells" in messages and "slot between free edges is 1" in messages


def test_periodic_seam_wraps_free_edge_detection():
    h = .0025
    nodes = [np.arange(-4, 5) * h, np.arange(-8, 9) * h, np.arange(-4, 5) * h]
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9))
    def scene(y0, y1):
        return prepare_scene(mw.Scene(domain=mw.Domain(bounds=tuple((float(a[0]), float(a[-1])) for a in nodes)),
            grid=mw.GridSpec.custom(*nodes), device="cpu",
            boundary=mw.BoundarySpec.faces(default="pec", y=("periodic", "periodic")),
            structures=[mw.Structure(mw.Box(position=torch.tensor((0., (y0 + y1) / 2, 0.), dtype=torch.float64),
                size=torch.tensor((1., y1 - y0, 0.), dtype=torch.float64)), foil, name="sheet")]))
    # Ends exactly at the y = -20 mm seam node: that end is a free edge.
    ending = scene(-.02, .005)
    faces = compile_copper_faces(ending, device="cpu")
    ex = next(face for face in faces if face.component == 0)
    sides = copper_free_edge_sides(ending, ex, ex.indices, faces)
    y = np.asarray(ending.y_nodes64)[ex.indices[:, 1].numpy()]
    assert set(np.round(y[sides.numpy() > 0], 6)) == {-.02, .005}
    # Spanning the periodic axis leaves no free edge, including at the seam.
    spanning = scene(-1., 1.)
    faces = compile_copper_faces(spanning, device="cpu")
    ex = next(face for face in faces if face.component == 0)
    assert not bool(copper_free_edge_sides(spanning, ex, ex.indices, faces).any())


def test_sharded_x_supports_give_the_unsharded_free_edges():
    from witwin.maxwell.fdtd.distributed.shard_engine import _build_local_scene, _physical_axis_nodes
    from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan
    h = .0025
    nodes = [np.arange(-8, 9) * h, np.arange(-6, 7) * h, np.arange(-3, 4) * h]
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9))
    # A rectangle whose x ends (Ey free edges, tangent x) sit on and next to the split.
    logical = mw.Scene(domain=mw.Domain(bounds=tuple((float(a[0]), float(a[-1])) for a in nodes)),
        grid=mw.GridSpec.custom(*nodes), boundary=mw.BoundarySpec.pec(), device="cpu",
        structures=[mw.Structure(mw.Box(position=torch.tensor((-.00625, 0., 0.), dtype=torch.float64),
            size=torch.tensor((.0125, .01, 0.), dtype=torch.float64)), foil, name="patch"),
            mw.Structure(mw.Box(position=torch.tensor((.0075, 0., 0.), dtype=torch.float64),
            size=torch.tensor((.005, .01, 0.), dtype=torch.float64)), foil, name="pad")])
    prepared = prepare_scene(logical)
    expected = {}
    faces = compile_copper_faces(prepared, device="cpu")
    for face in faces:
        sides = copper_free_edge_sides(prepared, face, face.indices, faces)
        expected.update({(face.name, face.component, tuple(i)): float(v)
                         for i, v in zip(face.indices.tolist(), sides.tolist())})
    plan = FDTDPartitionPlan((prepared.Nx, prepared.Ny, prepared.Nz), ("cuda:0", "cuda:1"))
    actual = {}
    for layout in plan.shard_layouts:
        from dataclasses import replace
        local = _build_local_scene(logical, prepared, replace(layout, device="cpu"), _physical_axis_nodes(prepared, "x"))
        local = prepare_scene(local)
        local._fdtd_shard_layout = layout
        local._fdtd_global_x_nodes64 = prepared.x_nodes64
        owned = compile_copper_faces(local, device="cpu")
        halos = compile_copper_faces(local, device="cpu", include_halos=True)
        for face in owned:
            sides = copper_free_edge_sides(local, face, face.indices, halos)
            origin = torch.tensor(layout.component("E" + "xyz"[face.component]).global_origin)
            for index, value in zip((face.indices + origin).tolist(), sides.tolist()):
                key = (face.name, face.component, tuple(index))
                assert key not in actual
                actual[key] = float(value)
    assert actual == expected
    assert sum(value > 0 for (name, component, _), value in expected.items() if component == 1) > 0


def test_normal_contact_at_a_rim_node_removes_its_free_side():
    scene = _strip_scene()
    faces = compile_copper_faces(scene, device="cpu")
    strip = next(face for face in faces if face.name == "strip" and face.component == 0)
    high = strip.indices[:, 1] == strip.indices[:, 1].max()
    via = torch.zeros(tuple(len(getattr(scene, f"{axis}_nodes64")) - (axis == "z") for axis in "xyz"), dtype=torch.bool)
    # One vertical contact edge below a single rim node (a via touching only the rim).
    rim = strip.indices[high][3]
    via[int(rim[0]), int(rim[1]), strip.normal_index - 1] = True
    sides = copper_free_edge_sides(scene, strip, strip.indices, faces, contact={"Ez": via})
    touched = (strip.indices[:, 1] == rim[1]) & ((strip.indices[:, 0] == rim[0]) | (strip.indices[:, 0] == rim[0] - 1))
    assert int(touched.sum()) == 2
    assert bool(torch.all(sides[touched] == 0)) and bool(torch.all(sides[high & ~touched] == 1))


def _strip_scene(*, pec_contact=False):
    h = .0025
    nodes = [np.arange(-8, 9) * h, np.arange(-8, 9) * h, np.arange(-4, 5) * h]
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9))
    structures = [
        mw.Structure(mw.Box(position=torch.tensor((0., 0., 0.), dtype=torch.float64),
                            size=torch.tensor((1., .010, 0.), dtype=torch.float64)), foil, name="strip"),
        mw.Structure(mw.Box(position=torch.tensor((0., 0., -.005), dtype=torch.float64),
                            size=torch.tensor((1., 1., 0.), dtype=torch.float64)), foil, name="ground"),
        # Abutting second foil on the same plane continues the strip at y = -5 mm.
        mw.Structure(mw.Box(position=torch.tensor((0., -.01125, 0.), dtype=torch.float64),
                            size=torch.tensor((1., .0075, 0.), dtype=torch.float64)), foil, name="extension"),
    ]
    scene = mw.Scene(domain=mw.Domain(bounds=tuple((float(a[0]), float(a[-1])) for a in nodes)),
                     grid=mw.GridSpec.custom(*nodes), boundary=mw.BoundarySpec.pec(), device="cpu",
                     structures=structures)
    return prepare_scene(scene)


def test_free_edges_are_only_open_strip_rims():
    scene = _strip_scene()
    faces = compile_copper_faces(scene, device="cpu")
    y = np.asarray(scene.y_nodes64)
    for face in faces:
        sides = copper_free_edge_sides(scene, face, face.indices, faces)
        coordinate = y[face.indices[:, 1].numpy()]
        if face.name == "ground":
            assert not bool(sides.any())
        elif face.component == 0:
            # Ex traces: both strip rims are free. The extension ends one cell
            # short of the strip (no shared edge), so both of its ends are free.
            free = coordinate[sides.numpy() > 0]
            expected = {"strip": {-.005, .005}, "extension": {-.015, -.0075}}[face.name]
            np.testing.assert_allclose(sorted(set(np.round(free, 6))), sorted(expected), atol=1e-9)
            assert bool(torch.all(sides <= 1))
        else:
            # Ey traces run across the strip: their tangent axis is x, where the
            # foil reaches the domain boundary, so they never see a free edge.
            assert not bool(sides.any())


def test_connecting_foil_or_contact_edges_continue_the_conductor():
    scene = _strip_scene()
    faces = compile_copper_faces(scene, device="cpu")
    strip = next(face for face in faces if face.name == "strip" and face.component == 0)
    bridge_source = next(face for face in faces if face.name == "strip" and face.component == 1)
    low = strip.indices[:, 1] == strip.indices[:, 1].min()
    high = strip.indices[:, 1] == strip.indices[:, 1].max()
    alone = copper_free_edge_sides(scene, strip, strip.indices, faces)
    assert bool(torch.all(alone[low] == 1)) and bool(torch.all(alone[high] == 1))
    # A foil Ey edge spanning the one-cell gap joins the strip to the extension.
    bridge = bridge_source.indices[bridge_source.indices[:, 1] == bridge_source.indices[:, 1].min()].clone()
    bridge[:, 1] = strip.indices[:, 1].min() - 1
    joined_face = type(bridge_source)("bridge", bridge_source.axis, bridge_source.normal_index, 1,
                                      bridge, bridge_source.material)
    joined = copper_free_edge_sides(scene, strip, strip.indices, (*faces, joined_face))
    assert bool(torch.all(joined[low] == 0)) and bool(torch.all(joined[high] == 1))
    # A conductor-contact Ey edge (PEC, conformal PEC, good conductor, port...) also continues it.
    mask = torch.zeros(tuple(len(getattr(scene, f"{axis}_nodes64")) - (axis == "y") for axis in "xyz"), dtype=torch.bool)
    mask[:, int(strip.indices[:, 1].max()), strip.normal_index] = True
    contacted = copper_free_edge_sides(scene, strip, strip.indices, faces, contact={"Ey": mask})
    assert bool(torch.all(contacted[high] == 0)) and bool(torch.all(contacted[low] == 1))


# ----------------------------------------------------------------- CUDA runs


def _line(kind, edge_model, *, h=.0025, half_x=.06, half_y=.03, pml=10):
    xs = np.arange(-round(half_x / h) - pml, round(half_x / h) + pml + 1) * h
    ys = np.arange(-round(half_y / h), round(half_y / h) + 1) * h
    zs = np.arange(-3, 4) * h
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9), edge_model=edge_model)
    structures = [mw.Structure(mw.Box(size=(10., 10., 10.)), mw.Material(eps_r=2.25), name="dielectric")]
    for name, y0, y1, z in (("low", -5., 5., -.005), ("high", -5., 5., .005)) + (
            (("signal", -.005, .005, 0.),) if kind == "stripline" else ()):
        structures.append(mw.Structure(mw.Box(position=torch.tensor((0., (y0 + y1) / 2, z), dtype=torch.float64),
            size=torch.tensor((10., y1 - y0, 0.), dtype=torch.float64)), foil, name=name))
    source_x = -round(.6 * half_x / h) * h
    if kind == "stripline":
        sources = [mw.UniformCurrentSource(size=(h / 2, .010, .005), polarization=(0., 0., sign),
                   center=(source_x, 0., sign * .0025), source_time=mw.CW(1e9)) for sign in (1., -1.)]
        boundary = mw.BoundarySpec.faces(default="pec", num_layers=pml, strength=1e6, x=("pml", "pml"))
    else:
        sources = [mw.UniformCurrentSource(size=(h / 2, 2 * half_y, .010), polarization="Ez",
                   center=(source_x, 0., 0.), source_time=mw.CW(1e9))]
        boundary = mw.BoundarySpec.faces(default="pec", num_layers=pml, strength=1e6, x=("pml", "pml"),
                                         y=("periodic", "periodic"))
    scene = mw.Scene(domain=mw.Domain(bounds=((xs[0], xs[-1]), (ys[0], ys[-1]), (zs[0], zs[-1]))),
                     grid=mw.GridSpec.custom(xs, ys, zs), boundary=boundary, device="cuda", structures=structures)
    for source in sources:
        scene.add_source(source)
    return mw.Simulation.fdtd(scene, absorber="cpml", frequencies=[1e9], full_field_dft=True,
                              run_time=mw.TimeConfig(time_steps=6000),
                              spectral_sampler=mw.SpectralSampler(window="hanning")).run()


def _local_alpha(result):
    """alpha = P'_loss / (2 P): native foil heat per length over E-field Poynting flux.

    H is the exact discrete Faraday image of the sampled E phasors, using each
    foil's high-face trace for the H row above it.
    """
    from witwin.maxwell.fdtd.runtime.representation import spectral_phasor_factor
    solver = result.solver
    xs, ys, zs = (np.asarray(getattr(solver.scene, f"{axis}_nodes64")) for axis in "xyz")
    E = {name: result.tensor(name).cpu().numpy().astype(np.complex128) for name in ("Ex", "Ey", "Ez")}
    dt = float(solver.dt)
    factor = 1 / (1j * MU0 * 2 * math.sin(math.pi * 1e9 * dt) / dt)
    dexdz = (E["Ex"][:, :, 1:] - E["Ex"][:, :, :-1]) / np.diff(zs)[None, None, :]
    ex_plane, ey_plane = E["Ex"].copy(), E["Ey"].copy()
    for entry in solver._copper_foils:
        phasor = torch.complex(entry["spectral_real"], entry["spectral_imag"]) * spectral_phasor_factor(solver)
        low, high = (phasor / entry["spectral_norm"].clamp_min(1)[:, None, None])[0].cpu().numpy()
        index = tuple(entry["indices"].cpu().numpy().T)
        if entry["electric_name"] == "Ex":
            dexdz[index[0], index[1], entry["face"].normal_index] -= (high - low) / np.diff(zs)[entry["face"].normal_index]
            ex_plane[index] = (low + high) / 2
        else:
            ey_plane[index] = (low + high) / 2
    hy = factor * (dexdz - (E["Ez"][1:] - E["Ez"][:-1]) / np.diff(xs)[:, None, None])
    hz = factor * ((ey_plane[1:] - ey_plane[:-1]) / np.diff(xs)[:, None, None]
                   - (ex_plane[:, 1:] - ex_plane[:, :-1]) / np.diff(ys)[None, :, None])
    dy, dz = np.diff(ys), np.diff(zs)
    dyd = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    dzd = np.r_[dz[0] / 2, (dz[:-1] + dz[1:]) / 2, dz[-1] / 2]
    if solver.scene.boundary.axis_kind("y") == "periodic":
        dyd[0] = dyd[-1] = dy[0] / 2
    pml = solver.scene.pml_thickness_for_face("x", "low")
    alphas = []
    for station in range(len(xs) // 2 + 4, len(xs) - pml - 4):
        flux = .5 * np.real(np.sum(E["Ey"][station] * np.conj((hz[station - 1] + hz[station]) / 2) * dy[:, None] * dzd[None])
                            - np.sum(E["Ez"][station] * np.conj((hy[station - 1] + hy[station]) / 2) * dyd[:, None] * dz[None]))
        lower, upper = xs[station] - (xs[1] - xs[0]) / 2, xs[station] + (xs[1] - xs[0]) / 2
        heat = 0.
        for record in result.conductor_losses.values():
            for trace in record["surface_power_traces"]:
                low, high = trace["lower_bounds"][:, 0].numpy(), trace["upper_bounds"][:, 0].numpy()
                fraction = np.clip(np.minimum(high, upper) - np.maximum(low, lower), 0, None) / (high - low)
                heat += float((trace["power"][0].numpy() * fraction).sum())
        alphas.append(heat / (upper - lower) / (2 * flux))
    return float(np.mean(alphas)), float(np.std(alphas))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
def test_edgeless_parallel_plate_is_bit_identical_and_matches_the_analytic_surface_loss(copper_torch_kernel):
    bare, edged = _line("plate", "none"), _line("plate", "auto")
    for name in ("Ex", "Ey", "Ez"):
        assert torch.equal(bare.tensor(name), edged.tensor(name))
    assert all(record["free_edge_trace_count"] == 0 for record in edged.conductor_losses.values())
    alpha, spread = _local_alpha(edged)
    surface_resistance = math.sqrt(math.pi * 1e9 * MU0 / SIGMA)
    analytic = surface_resistance / (math.sqrt(MU0 / (EPS0 * 2.25)) * .010)
    assert spread < 1e-3 * alpha
    assert alpha == pytest.approx(analytic, rel=3e-3)


@cuda
def test_stripline_copper_attenuation_matches_the_independent_35um_reference():
    result = _line("stripline", "auto")
    record = result.conductor_losses["signal"]
    assert record["free_edge_trace_count"] > 0
    assert all(result.conductor_losses[name]["free_edge_trace_count"] == 0 for name in ("low", "high"))
    factors = torch.cat([trace["edge_impedance_factor"] for trace in record["surface_power_traces"]])
    assert set(np.round(factors.unique().numpy(), 6)) == {1., round(float(copper_edge_impedance_factor(
        35e-6, torch.tensor([.0025]), torch.ones(1), torch.ones(1))), 6)}
    alpha, spread = _local_alpha(result)
    assert spread < 2e-3 * alpha
    # 2.5 mm grid: +3.7% (1.25 mm: +0.9%, 0.625 mm: -0.1%). The bare
    # zero-thickness rim gives 7.06e-3 (-38%) here and only 7.95e-3 at 0.625 mm.
    assert alpha == pytest.approx(STRIPLINE_REFERENCE_ALPHA_C_1GHZ, rel=.06)


class _ThickFoil(mw.SceneModule):
    """A finite 20 mm foil square (four free edges) in the good-conductor regime."""

    def __init__(self, thickness=10e-6):
        super().__init__()
        self.log_thickness = torch.nn.Parameter(torch.tensor(math.log(thickness), device="cuda", dtype=torch.float64))

    def to_scene(self):
        return mw.Scene(domain=mw.Domain(bounds=((-.03, .03),) * 3), grid=mw.GridSpec.uniform(.005),
            boundary=mw.BoundarySpec.pec(), device="cuda",
            structures=(mw.Structure(mw.Box(size=(0., .02, .02)), mw.CopperFoil(
                thickness=self.log_thickness.exp(), frequency_range=(2e9, 4e9), mode_count=32), name="foil"),),
            sources=(mw.UniformCurrentSource(name="drive", center=(-.015, 0., -.0025), size=(.005,) * 3,
                polarization="Ez", source_time=mw.CW(3e9, amplitude=1e-5)),))

    def run(self):
        simulation = mw.Simulation.fdtd(self, frequencies=(3e9,), full_field_dft=True,
            run_time=mw.TimeConfig(time_steps=72), spectral_sampler=mw.SpectralSampler(window="none"))
        simulation.config.adjoint_checkpoint_stride = 7
        return simulation.run()


@cuda
@pytest.mark.filterwarnings("ignore:CopperFoil free-edge model")
@pytest.mark.parametrize("kind,thickness", (("dissipated_energy", 10e-6), ("stored_energy", 10e-6),
                                            ("surface_power", 10e-6), ("dissipated_energy", 3.1e-6)))
def test_thickness_gradient_includes_the_edge_impedance_factor(kind, thickness):
    # 3.1 um sits inside the 2-4 skin-depth blend at 4 GHz (weight ~0.5).
    model = _ThickFoil(thickness)
    def objective():
        record = model.run().conductor_losses["foil"]
        return record[kind].sum() if kind == "surface_power" else record[kind], record
    value, record = objective()
    assert record["free_edge_trace_count"] > 0
    value.backward()
    actual = model.log_thickness.grad.detach().clone()
    base = model.log_thickness.detach().clone()
    values = []
    model.requires_grad_(False)
    for delta in (.02, -.02):
        with torch.no_grad():
            model.log_thickness.copy_(base + delta)
        values.append(torch.as_tensor(objective()[0], dtype=torch.float64))
    expected = (values[0] - values[1]) / .04
    # In this regime the heat depends on thickness almost only through kappa.
    assert abs(float(expected)) > 1e-3 * abs(float(value))
    # Compare logarithmic derivatives: the energies are O(1e-32) J.
    assert float(actual / value) == pytest.approx(float(expected / value), rel=.02, abs=0)


@cuda
def test_runtime_conductor_and_port_contacts_are_not_free_edges():
    h = .0025
    nodes = [np.arange(-8, 9) * h, np.arange(-8, 9) * h, np.arange(-4, 5) * h]
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(.8e9, 1.2e9))
    def foil_box(name, y0, y1, z):
        return mw.Structure(mw.Box(position=torch.tensor((0., (y0 + y1) / 2, z), dtype=torch.float64),
            size=torch.tensor((1., y1 - y0, 0.), dtype=torch.float64)), foil, name=name)
    def prepared(*extra, ports=()):
        scene = mw.Scene(domain=mw.Domain(bounds=tuple((float(a[0]), float(a[-1])) for a in nodes)),
            grid=mw.GridSpec.custom(*nodes), boundary=mw.BoundarySpec.pec(), device="cuda",
            structures=[foil_box("strip", -.005, .005, 0.), foil_box("ground", -1., 1., -.005), *extra],
            ports=list(ports))
        options = dict(excitations=mw.PortExcitation(ports[0].name)) if ports else {}
        simulation = mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=4), **options)
        return simulation.prepare().solver
    def rim_factors(solver):
        entry = next(e for e in solver._copper_foils if e["face"].name == "strip" and e["electric_name"] == "Ex")
        y = torch.as_tensor(solver.scene.y_nodes64, device=entry["indices"].device)[entry["indices"][:, 1]]
        factor = 1 / entry["admittance_scale"]
        return {round(float(v), 4): factor[torch.isclose(y, torch.tensor(v, dtype=y.dtype, device=y.device))]
                for v in (-.005, .005)}, entry
    bare, _ = rim_factors(prepared())
    assert bool(torch.all(bare[-.005] > 2)) and bool(torch.all(bare[.005] > 2))
    # A good-conductor volume along the +5 mm edge continues the conductor there.
    block = mw.Structure(mw.Box(position=(0., .00875, 0.), size=(1., .0075, .0025)),
                         mw.Material(eps_r=1., sigma_e=5.8e7), name="block")
    blocked, _ = rim_factors(prepared(block))
    assert bool(torch.all(blocked[.005] == 1)) and bool(torch.all(blocked[-.005] > 2))
    # A lumped port fed vertically at one rim node removes that node's free side.
    port = mw.LumpedPort("feed", positive=(0., -.005, 0.), negative=(0., -.005, -.005),
        voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(0., -.005, -.00375), size=(.0025, .0025, 0.)))
    fed, entry = rim_factors(prepared(ports=(port,)))
    touched = int((fed[-.005] == 1).sum())
    assert touched == 2 and bool(torch.all(fed[.005] > 2))
