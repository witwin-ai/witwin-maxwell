"""Centre-owned bulk poles, shared by Yee ADE and frequency-domain sampling."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdfd.materials import _evaluate_edge_components
from witwin.maxwell.scene import prepare_scene


def sphere_scene(material, averaging="kottke_edge", samples=4):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
        grid=mw.GridSpec.uniform(0.05), boundary=mw.BoundarySpec.pec(), device="cpu",
        subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging),
        structures=[mw.Structure(geometry=mw.Sphere(radius=0.15), material=material)],
    )


def _assert_compiled_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.is_tensor(expected) and torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_compiled_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_compiled_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind,magnetic", [(kind, magnetic) for kind in ("debye", "drude", "lorentz")
    for magnetic in (False, True)] + [("sellmeier", False), ("zero_perturbation", False)])
def test_uniform_pole_fields_compile_bitwise_like_scalar_materials(kind, magnetic):
    grid = torch.ones(3, 3, 3, dtype=torch.float64)
    if kind == "sellmeier":
        custom = mw.Material.custom_sellmeier(b_coefficients=(1.2 * grid,), c_coefficients=(.0625,), eps_inf=1.5)
        scalar = mw.Material.sellmeier(b_coefficients=(1.2,), c_coefficients=(.0625,), eps_inf=1.5)
    elif kind == "zero_perturbation":
        scalar = mw.Material.lorentz(eps_inf=2., delta_eps=1., resonance_frequency=2e9, gamma=2e8)
        custom = mw.PerturbationMedium(scalar, perturbation=torch.zeros_like(grid),
            eps_sensitivity=.7, lorentz_sensitivities=(.25,))
    else:
        scalar_pole, custom_pole = {
            "debye": lambda: (mw.DebyePole(3., 1e-10), mw.CustomDebyePole(3. * grid, 1e-10)),
            "drude": lambda: (mw.DrudePole(2e9, 1e8), mw.CustomDrudePole(2e9 * grid, 1e8)),
            "lorentz": lambda: (mw.LorentzPole(3., 2e9, 2e8), mw.CustomLorentzPole(3. * grid, 2e9, 2e8)),
        }[kind]()
        key = ("mu_" if magnetic else "") + kind + "_poles"
        custom = mw.Material(eps_r=2., mu_r=1.5, **{key: (custom_pole,)})
        scalar = mw.Material(eps_r=2., mu_r=1.5, **{key: (scalar_pole,)})
    def compile(material):
        scene = sphere_scene(material)
        scene.structures = [mw.Structure(geometry=mw.Box(size=(.27, .27, .27)), material=material)]
        return prepare_scene(scene).compile_materials()
    _assert_compiled_equal(compile(custom), compile(scalar))



def _compile_box(material):
    scene = sphere_scene(material)
    scene.structures = [mw.Structure(geometry=mw.Box(size=(.27, .27, .27)), material=material)]
    return prepare_scene(scene).compile_materials()


def test_near_uniform_custom_pole_field_stays_fractional():
    # Uniformity is exact equality: one value one ulp away keeps the graded fractional path.
    grid = torch.full((3, 3, 3), 3., dtype=torch.float64)
    grid[0, 0, 0] = torch.nextafter(grid[0, 0, 0], grid.new_tensor(float("inf")))
    near = _compile_box(mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(grid, 2e9, 2e8),)))["lorentz_poles"][0]
    exact = _compile_box(mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(torch.full_like(grid, 3.), 2e9, 2e8),)))
    assert "edge_weights" not in near and near["amplitude"] is not None
    assert "edge_weights" in exact["lorentz_poles"][0] and exact["lorentz_poles"][0]["amplitude"] is None


def test_tiny_uniform_pole_strength_is_not_thresholded():
    # A 1e-30 strength is an ordinary pole: it compiles like its scalar twin, not like no pole.
    grid = torch.full((3, 3, 3), 1e-30, dtype=torch.float64)
    custom = _compile_box(mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(grid, 2e9, 2e8),)))
    scalar = _compile_box(mw.Material(eps_r=2., lorentz_poles=(mw.LorentzPole(1e-30, 2e9, 2e8),)))
    _assert_compiled_equal(custom, scalar)
    assert len(custom["lorentz_poles"]) == 1
    assert any(bool((w == 1).any()) for w in custom["lorentz_poles"][0]["edge_weights"].values())

@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz"])
def test_bulk_pole_edges_are_binary_and_share_frequency_background(kind):
    averaging = "kottke_edge"
    pole = {"debye": mw.DebyePole(3., 1e-10), "drude": mw.DrudePole(5e9, 1e8),
            "lorentz": mw.LorentzPole(3., 1.8e9, 1e8)}[kind]
    material = mw.Material(eps_r=2., **{kind + "_poles": (pole,)})
    scene = prepare_scene(sphere_scene(material, averaging))
    model = scene.compile_materials()
    epsilon, _ = _evaluate_edge_components(model, 1.5e9)
    for component, weight in model[kind + "_poles"][0]["edge_weights"].items():
        assert bool(((weight == 0) | (weight == 1)).all())
        assert bool((weight == 0).any()) and bool((weight == 1).any())
        from witwin.maxwell.compiler.materials import _edge_component_coords, _edge_axis_base_1d
        bases = {a: _edge_axis_base_1d(scene, a, a == component[-1].lower(), dtype=torch.float64) for a in "xyz"}
        coords = _edge_component_coords(bases, (0., 0., 0.))
        inside = sum(x.square() for x in coords) <= scene.structures[0].geometry.radius.double().square()
        assert torch.equal(weight.bool(), inside)
        assert torch.equal(model["edge_components"]["eps"][component], 1 + weight)
        expected = 1 + weight * (1 + pole.susceptibility_at_freq(1.5e9))
        torch.testing.assert_close(epsilon[component[-1].lower()], expected.to(torch.complex64))
    solver = mw.Simulation.fdtd(sphere_scene(material, averaging), frequencies=(1.5e9,),
        backend="cpu_reference", run_time=mw.TimeConfig(time_steps=2, dt=3e-12), cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(normalize_source=False)).prepare().solver
    for c, weight in model[kind+"_poles"][0]["edge_weights"].items():
        torch.testing.assert_close(getattr(solver, "eps_"+c)/solver.eps0, 1+weight)


def test_modulated_pole_constituent_keeps_binary_modulation_support():
    material = mw.Material(eps_r=2., drude_poles=(mw.DrudePole(5e9, 0.),),
        modulation=mw.ModulationSpec(frequency=1e8, amplitude=.2))
    solver = mw.Simulation.fdtd(sphere_scene(material), frequencies=(1.5e9,), backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=2, dt=3e-12), cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(normalize_source=False)).prepare().solver
    for c, weight in solver._compiled_material_model["drude_poles"][0]["edge_weights"].items():
        torch.testing.assert_close(getattr(solver, "mod_cos_"+c), .2*weight)
        mask = solver._compiled_material_model["pole_edge_modulation"][c]["mask"]
        torch.testing.assert_close(getattr(solver, "mod_omega_"+c)[mask], (2*math.pi*1e8*weight)[mask])


def test_overlapping_constituents_select_one_complete_pole_set():
    scene = sphere_scene(mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=0.))
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(.1, 0., 0.), size=(.2, .5, .5)),
        material=mw.Material.lorentz(eps_inf=4., delta_eps=3., resonance_frequency=1.8e9, gamma=0.)))
    model = prepare_scene(scene).compile_materials()
    for c in ("Ex", "Ey", "Ez"):
        a = model["drude_poles"][0]["edge_weights"][c]
        b = model["lorentz_poles"][0]["edge_weights"][c]
        assert not bool((a.bool() & b.bool()).any())
        torch.testing.assert_close(model["edge_components"]["eps"][c][(a+b)>0], (1+a+3*b)[(a+b)>0])


@pytest.mark.parametrize("averaging,samples", [("arithmetic", 1), ("arithmetic", 4),
    ("polarized", 4), ("polarized_tensor", 4), ("kottke_smooth", 4)])
def test_gradient_modes_keep_fractional_poles_and_density_gradients(averaging, samples):
    scene = sphere_scene(mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=1e8), averaging, samples)
    density = torch.full((3, 3, 3), .5, requires_grad=True)
    scene.add_material_region(mw.MaterialRegion(name="design", geometry=mw.Box(position=(.15, 0., 0.), size=(.2, .4, .4)),
        density=density, eps_bounds=(1., 3.)))
    model = prepare_scene(scene).compile_materials()
    entry = model["drude_poles"][0]
    assert "edge_weights" not in entry
    assert bool(((entry["weight"] > 0) & (entry["weight"] < 1)).any())
    epsilon, _ = _evaluate_edge_components(model, 1.5e9)
    gradient, = torch.autograd.grad(sum(v.real.sum() for v in epsilon.values()), density)
    assert torch.isfinite(gradient).all() and bool((gradient != 0).any())


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("point_location,owned", [("face", True), ("inside", True), ("outside", False)])
def test_closed_box_node_face_ownership_is_exact_to_float64_ulp(axis, point_location, owned):
    # Move the declared face one ulp either side of a fixed Yee node plane.
    face = torch.tensor(.25, dtype=torch.float64)
    if point_location != "face":
        direction = float("inf") if point_location == "inside" else -float("inf")
        face = torch.nextafter(face, torch.tensor(direction, dtype=torch.float64))
    size = torch.full((3,), .75, dtype=torch.float64)
    size["xyz".index(axis)] = 2 * face
    scene = prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(.125),
        boundary=mw.BoundarySpec.pec(), device="cpu",
        structures=[mw.Structure(geometry=mw.Box(position=torch.zeros(3, dtype=torch.float64), size=size),
            material=mw.Material.lorentz(eps_inf=2., delta_eps=1., resonance_frequency=1e9, gamma=0.))]))
    model = scene.compile_materials()
    component = "E" + "xyz"[("xyz".index(axis) + 1) % 3]
    nodes = torch.as_tensor(getattr(scene, f"{axis}_nodes64"))
    weight = model["lorentz_poles"][0]["edge_weights"][component]
    for sign in (-1, 1):
        plane = int(torch.nonzero(nodes == sign * .25).item())
        index = [s // 2 for s in weight.shape]
        index["xyz".index(axis)] = plane
        assert bool(weight[tuple(index)]) == owned


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("direction,owned", [(-float("inf"), False), (float("inf"), True)])
def test_float64_non_node_box_bound_keeps_exact_ulp_comparison(axis, direction, owned):
    from witwin.maxwell.compiler.materials import _geometry_occupancy
    point = torch.tensor(.23, dtype=torch.float64)
    face = torch.nextafter(point, torch.tensor(direction, dtype=torch.float64))
    size = torch.full((3,), .75, dtype=torch.float64)
    size["xyz".index(axis)] = 2 * face
    box = mw.Box(position=torch.zeros(3, dtype=torch.float64), size=size)
    scene = prepare_scene(sphere_scene(mw.Material()))
    for sign in (-1, 1):
        coords = [torch.zeros((1, 1, 1), dtype=torch.float64) for _ in "xyz"]
        coords["xyz".index(axis)].fill_(sign * point)
        weight = _geometry_occupancy(scene, box, coords=coords, surface_inclusive=True)
        assert bool(weight.item()) == owned


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("ulps,owned", [(0, True), (1, True), (2, False)])
@pytest.mark.parametrize("staggered", [False, True])
def test_float32_centered_box_snaps_both_faces_within_propagated_roundoff(axis, ulps, owned, staggered):
    from witwin.maxwell.compiler.materials import _geometry_occupancy
    node = torch.tensor(.2, dtype=torch.float32)
    face = node.clone()
    for _ in range(ulps):
        face = torch.nextafter(face, face.new_tensor(-float("inf")))
    size = torch.full((3,), .75)
    size["xyz".index(axis)] = 2 * face
    box = mw.Box(position=torch.zeros(3), size=size)
    extent = 2 * float(node) if staggered else .4
    nodes = (-extent, 0., extent) if staggered else (-extent, -float(node), 0., float(node), extent)
    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-extent, extent),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes), device="cpu", boundary=mw.BoundarySpec.pec()))
    for sign in (-1, 1):
        coords = [torch.zeros((1, 1, 1), dtype=torch.float64) for _ in "xyz"]
        coords["xyz".index(axis)].fill_(sign * node.double())
        weight = _geometry_occupancy(scene, box, coords=coords, surface_inclusive=True)
        assert bool(weight.item()) == owned


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("location,owned", [("inside", True), ("bound", True), ("outside", False)])
def test_float32_box_propagated_snap_bound_is_inclusive_in_float64(axis, sign, location, owned):
    from witwin.maxwell.compiler.materials import _geometry_occupancy
    # p=.3125, s=.25, f=.1875: ULP(p)/2 + ULP(s)/4 + ULP(f)/2 = 2**-25.
    # The division is exact. Move the grid plane one float64 ULP across that bound.
    node = torch.tensor(.1875 - 2**-25, dtype=torch.float64)
    if location != "bound":
        direction = float("inf") if location == "inside" else -float("inf")
        node = torch.nextafter(node, node.new_tensor(direction))
    node *= sign
    position = torch.zeros(3)
    size = torch.full((3,), .75)
    index = "xyz".index(axis)
    position[index] = sign * .3125
    size[index] = .25
    box = mw.Box(position=position, size=size)
    nodes = sorted((-.5, 0., float(node), .5))
    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-.5, .5),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes), device="cpu", boundary=mw.BoundarySpec.pec()))
    coords = [torch.zeros((1, 1, 1), dtype=torch.float64) for _ in "xyz"]
    coords[index].fill_(node)
    weight = _geometry_occupancy(scene, box, coords=coords, surface_inclusive=True)
    assert bool(weight.item()) == owned


def test_slab_surface_roundoff_keeps_poles_and_later_constituent_priority():
    from witwin.maxwell.compiler.materials import _edge_axis_base_1d, _edge_component_coords
    scene = mw.Scene(domain=mw.Domain(bounds=((-1., 1.), (-.4, .4), (-.4, .4))),
        grid=mw.GridSpec.uniform(.04), boundary=mw.BoundarySpec.pec(), device="cpu",
        structures=[mw.Structure(geometry=mw.Box(position=(.1, 0., 0.), size=(.16, .8, .8)),
            material=mw.Material.lorentz(eps_inf=1., delta_eps=1.2, resonance_frequency=1e9, gamma=1.5e8))])
    prepared = prepare_scene(scene)
    model = prepared.compile_materials()
    coords = _edge_component_coords({a: _edge_axis_base_1d(prepared, a, a == "x") for a in "xyz"}, (0., 0., 0.))
    sdf = scene.structures[0].geometry.signed_distance(*coords)
    weight = model["lorentz_poles"][0]["edge_weights"]["Ex"]
    entrance = torch.isclose(coords[0], torch.tensor(.02), rtol=0., atol=1e-8)
    surface = torch.isclose(coords[0], torch.tensor(.18), rtol=0., atol=1e-8)
    assert int(entrance.sum()) == 441
    assert int(surface.sum()) == 441
    # The entrance is 1.76 face ULPs away, within the propagated p +/- s/2 bound.
    assert bool((weight[entrance] == 1).all())
    assert bool((sdf[surface] > 0).all())
    assert bool((weight[surface] == 1).all())
    coords64 = _edge_component_coords({a: _edge_axis_base_1d(prepared, a, a == "x", dtype=torch.float64)
                                      for a in "xyz"}, (0., 0., 0.))
    box = scene.structures[0].geometry
    lower, upper = (bound.double() for bound in (box.position-box.size/2, box.position+box.size/2))
    inside = torch.ones_like(weight, dtype=torch.bool)
    for i, coord in enumerate(coords64):
        inside &= (coord >= lower[i]) & (coord <= upper[i])
    inside |= entrance | surface
    assert torch.equal(weight.bool(), inside)
    changed = weight.bool() != (sdf <= 0)
    assert int((changed & surface).sum()) == 441
    assert not bool((changed & (sdf < -1e-6)).any())
    away = sdf.abs() > 1e-6
    assert torch.equal(weight[away].bool(), sdf[away] <= 0)
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(.26, 0., 0.), size=(.16, .8, .8)),
        material=mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=0.)))
    model = prepare_scene(scene).compile_materials()
    assert bool((model["lorentz_poles"][0]["edge_weights"]["Ex"][surface] == 0).all())
    assert bool((model["drude_poles"][0]["edge_weights"]["Ex"][surface] == 1).all())


def test_nondispersive_interfaces_keep_their_subpixel_average():
    scene = sphere_scene(mw.Material(eps_r=3.))
    scene.structures = [mw.Structure(geometry=mw.Sphere(position=(.23, 0., 0.), radius=.08), material=mw.Material(eps_r=3.))]
    before = prepare_scene(scene).compile_materials()["edge_components"]["eps"]
    scene.add_structure(mw.Structure(geometry=mw.Sphere(position=(-.23, 0., 0.), radius=.08),
        material=mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=0.)))
    after = prepare_scene(scene).compile_materials()["edge_components"]["eps"]
    for component in before:
        assert torch.equal(before[component][-6:], after[component][-6:])
        assert bool(((before[component] > 1) & (before[component] < 3)).any())


def test_dispersive_dielectric_interface_uses_complete_center_constituent():
    scene = sphere_scene(mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=0.))
    scene.structures.insert(0, mw.Structure(geometry=mw.Box(size=(2., 2., 2.)), material=mw.Material(eps_r=3.)))
    model = prepare_scene(scene).compile_materials()
    for c, weight in model["drude_poles"][0]["edge_weights"].items():
        assert torch.equal(model["edge_components"]["eps"][c], 3-weight)


def test_graded_partial_window_keeps_full_centre_background():
    averaging = "kottke_edge"
    scene = mw.Scene(domain=mw.Domain(bounds=((0., 1.), (0., .1), (0., .1))),
        grid=mw.GridSpec.custom((0., .1, 1.), (0., .01, .1), (0., .01, .1)),
        device="cpu", boundary=mw.BoundarySpec.pec(),
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging=averaging),
        structures=[mw.Structure(geometry=mw.Box(position=(1., .05, .05), size=(.12, 1., 1.)),
            material=mw.Material.drude(eps_inf=2., plasma_frequency=5e9, gamma=0.))])
    # The final Ex window reaches x=1, but its centre is x=.55 and its last
    # quadrature midpoint is x=.8875. The pole body begins at x=.94.
    model = prepare_scene(scene).compile_materials()
    weight = model["drude_poles"][0]["edge_weights"]["Ex"]
    assert not bool(weight.any())
    assert torch.equal(model["edge_components"]["eps"]["Ex"], torch.ones_like(weight))


def test_static_conductivity_sampling_is_unchanged_by_pole_ownership():
    materials = (mw.Material(eps_r=2., sigma_e=.1),
                 mw.Material(eps_r=2., sigma_e=.1, drude_poles=(mw.DrudePole(5e9, 0.),)))
    models = [prepare_scene(sphere_scene(material)).compile_materials() for material in materials]
    for c in ("Ex", "Ey", "Ez"):
        assert torch.equal(models[0]["edge_components"]["sigma_e"][c], models[1]["edge_components"]["sigma_e"][c])


@pytest.mark.parametrize("samples", [1, 4])
def test_tensor_background_runtime_keeps_scalar_pole_interface_binary(samples):
    scene = sphere_scene(mw.Material.lorentz(eps_inf=2., delta_eps=3., resonance_frequency=1.8e9, gamma=0.), samples=samples)
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(.28, 0., 0.), size=(.08, .16, .16)),
        material=mw.Material(epsilon_tensor=mw.Tensor3x3(((3., .2, 0.), (.2, 3., 0.), (0., 0., 3.))))))
    solver = mw.Simulation.fdtd(scene, frequencies=(1.5e9,), backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=2, dt=3e-12), cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(normalize_source=False)).prepare().solver
    assert solver.full_aniso_enabled
    model = solver._compiled_material_model
    for c, weight in model["lorentz_poles"][0]["edge_weights"].items():
        x = torch.as_tensor(solver._get_component_coords(c)[0])[:, None, None]
        mask = (abs(x) < .18).expand_as(weight)
        torch.testing.assert_close((getattr(solver, "eps_"+c)/solver.eps0)[mask], (1+weight)[mask])


def test_shard_coordinate_sampling_matches_global_binary_weights():
    scene = sphere_scene(mw.Material(mu_lorentz_poles=(mw.LorentzPole(3., 1.8e9, 0.),)))
    whole = prepare_scene(scene).compile_materials()
    local = scene.clone(domain=mw.Domain(bounds=((-0.2, .2), (-.4, .4), (-.4, .4))))
    shard = prepare_scene(local).compile_materials()
    for c in ("Hx", "Hy", "Hz"):
        weight = whole["mu_lorentz_poles"][0]["edge_weights"][c]
        assert bool(((weight == 0) | (weight == 1)).all())
        assert torch.equal(weight[4:-4], shard["mu_lorentz_poles"][0]["edge_weights"][c])


def test_edge_pole_coefficients_are_part_of_resume_identity():
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint
    scene = sphere_scene(mw.Material.drude(eps_inf=1., plasma_frequency=5e9, gamma=0.))
    solver = mw.Simulation.fdtd(scene, frequencies=(1.5e9,), backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=2, dt=3e-12), cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(normalize_source=False)).prepare().solver
    before = _configuration_fingerprint(solver)
    weights = solver._compiled_material_model["drude_poles"][0]["edge_weights"]["Ex"]
    weights[0, 0, 0] = 1 - weights[0, 0, 0]
    assert _configuration_fingerprint(solver) != before


def test_uniform_custom_pole_strength_gradient_uses_binary_support():
    strength = torch.full((3, 3, 3), 3., requires_grad=True)
    def evaluate(x):
        scene = sphere_scene(mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(
            delta_eps=x, resonance_frequency=1.8e9, gamma=1e8),)))
        scene.structures = [mw.Structure(geometry=mw.Box(size=(.27, .27, .27)), material=scene.structures[0].material)]
        model = prepare_scene(scene).compile_materials()
        for weight in model["lorentz_poles"][0]["edge_weights"].values():
            assert bool(((weight == 0) | (weight == 1)).all())
        eps, _ = _evaluate_edge_components(model, 1.5e9)
        return sum(v.imag.sum() for v in eps.values())
    grad, = torch.autograd.grad(evaluate(strength), strength)
    step = .001
    fd = (evaluate(strength.detach()+step) - evaluate(strength.detach()-step)) / (2*step)
    torch.testing.assert_close(grad.sum(), fd, rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("averaging", ["kottke_edge", "arithmetic", "polarized", "polarized_tensor", "kottke_smooth"])
@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz", "sellmeier"])
def test_custom_constituents_keep_previous_fractional_response(monkeypatch, averaging, kind):
    import witwin.maxwell.compiler.materials as compiler
    grid = torch.linspace(.5, 1., 27).reshape(3, 3, 3)
    material = {
        "debye": lambda: mw.Material(eps_r=2., debye_poles=(mw.CustomDebyePole(delta_eps=grid, tau=1e-10),)),
        "drude": lambda: mw.Material(eps_r=2., drude_poles=(mw.CustomDrudePole(plasma_frequency=grid*5e9, gamma=1e8),)),
        "lorentz": lambda: mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(delta_eps=grid, resonance_frequency=1.8e9, gamma=1e8),)),
        "sellmeier": lambda: mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(.01,)),
    }[kind]()
    scene = sphere_scene(material, averaging)
    scene.structures = [mw.Structure(geometry=mw.Box(size=(.27, .27, .27)), material=material)]
    model = prepare_scene(scene).compile_materials()
    assert all("edge_weights" not in entry for key in ("debye_poles", "drude_poles", "lorentz_poles")
               for entry in model[key])
    monkeypatch.setattr(compiler, "_compile_binary_pole_edges", lambda *args, **kwargs: None)
    previous = prepare_scene(scene).compile_materials()
    for actual, expected in zip(_evaluate_edge_components(model, 1.5e9), _evaluate_edge_components(previous, 1.5e9)):
        for axis in "xyz":
            assert torch.equal(actual[axis], expected[axis])


def test_mixed_custom_scalar_window_keeps_all_fractional_weights_and_background(monkeypatch):
    import witwin.maxwell.compiler.materials as compiler
    from witwin.maxwell.fdtd.runtime.materials import average_node_to_component
    scene = sphere_scene(mw.Material())
    scene.structures = [
        mw.Structure(geometry=mw.Box(position=(-.1, 0., 0.), size=(.3, .5, .5)),
            material=mw.Material(eps_r=2., lorentz_poles=(mw.CustomLorentzPole(
                delta_eps=torch.linspace(.5, 1., 27).reshape(3, 3, 3), resonance_frequency=1.8e9, gamma=1e8),))),
        mw.Structure(geometry=mw.Box(position=(.15, 0., 0.), size=(.3, .5, .5)),
            material=mw.Material.drude(eps_inf=3., plasma_frequency=5e9, gamma=1e8)),
    ]
    model = prepare_scene(scene).compile_materials()
    monkeypatch.setattr(compiler, "_compile_binary_pole_edges", lambda *args, **kwargs: None)
    previous = prepare_scene(scene).compile_materials()
    for component in ("Ex", "Ey", "Ez"):
        custom = average_node_to_component(None, previous["lorentz_poles"][0]["weight"], component)
        mask = custom > 0
        scalar = average_node_to_component(None, previous["drude_poles"][0]["weight"], component)
        assert bool(((scalar > 0) & mask).any())
        for key in ("lorentz_poles", "drude_poles"):
            actual = model[key][0]["edge_weights"][component]
            expected = average_node_to_component(None, previous[key][0]["weight"], component)
            assert torch.equal(actual[mask], expected[mask])
        assert torch.equal(model["edge_components"]["eps"][component][mask],
                           previous["edge_components"]["eps"][component][mask])
        # The far scalar face still receives binary ownership in the same scene.
        far = model["drude_poles"][0]["edge_weights"][component][-3:]
        assert bool(((far == 0) | (far == 1)).all())


class _CpuADE:
    """Test-local missing ADE launches; ordinary cpu_reference Yee kernels remain."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        if name not in ("updateDrudeCurrent3D", "updateLorentzCurrent3D", "applyPolarizationCurrent3D"):
            return getattr(self.original, name)
        def bind(**kw):
            class Launch:
                def launchRaw(self):
                    if name == "applyPolarizationCurrent3D":
                        kw["ElectricField"].sub_(kw["dt"] * kw["InvPermittivity"] * kw["PolarizationCurrent"])
                    else:
                        j = kw["PolarizationCurrent"]
                        value = kw["decay"] * j
                        kind = "Lorentz" if name == "updateLorentzCurrent3D" else "Drude"
                        if kind == "Lorentz":
                            value = value - kw["restoring"] * kw["Polarization"]
                        j.copy_(value + kw[kind + "Drive"] * kw["ElectricField"])
                        if kind == "Lorentz":
                            kw["Polarization"].add_(kw["dt"] * j)
            return Launch()
        return bind


@pytest.mark.parametrize("kind", ["drude", "lorentz"])
def test_closed_pec_lossless_partial_sphere_energy(kind):
    """Port of the factorial closed-box gate: exact cross-time Yee/ADE energy.

    E=P=J start at zero. Five divergence-free magnetic cavity modes excite all
    polarizations. Every step is measured for at least twenty 1.5 GHz periods.
    """
    from witwin.maxwell.fdtd.runtime.stepping import enforce_pec_boundaries
    material = (mw.Material.drude(eps_inf=1., plasma_frequency=5e9, gamma=0.) if kind == "drude" else
                mw.Material.lorentz(eps_inf=2., delta_eps=3., resonance_frequency=1.8e9, gamma=0.))
    h, frequency = .05, 1.5e9
    dt = .8 / (2*299792458.*math.sqrt(3)/h + 2*math.pi*12e9)
    solver = mw.Simulation.fdtd(sphere_scene(material), frequencies=(frequency,),
        run_time=mw.TimeConfig(time_steps=2, dt=dt), full_field_dft=False, cuda_graph=False,
        spectral_sampler=mw.SpectralSampler(normalize_source=False, window="none"), backend="cpu_reference").prepare().solver
    solver.fdtd_module = _CpuADE(solver.fdtd_module)
    solver._initialize_dispersive_state()
    electric, magnetic = ("Ex", "Ey", "Ez"), ("Hx", "Hy", "Hz")
    for c in electric:
        torch.testing.assert_close(getattr(solver, "eps_"+c)/solver.eps0,
            solver._compiled_material_model["edge_components"]["eps"][c])
    for i, c in enumerate(magnetic):
        xyz = torch.meshgrid(*(torch.as_tensor(a+.4) for a in solver._get_component_coords(c)), indexing="ij")
        value = torch.zeros_like(getattr(solver, c))
        for a, b, modes, amp in ((0,1,(2,3,1),1.), (1,2,(1,2,3),.83), (2,0,(3,1,2),.67),
                                  (0,2,(5,4,3),.31), (1,0,(4,5,2),.23)):
            if i not in (a, b):
                continue
            wave = [2*math.sin(m*math.pi*h/(2*.8))/h for m in modes]
            term = torch.ones_like(value) * (amp if i == a else -amp*wave[a]/wave[b])
            for j in range(3):
                phase = modes[j]*math.pi*xyz[j]/.8
                term *= torch.sin(phase) if j == i else torch.cos(phase)
            value.add_(term / math.sqrt(solver.mu0/solver.eps0))
        value.select(i, 0).zero_()
        value.select(i, -1).zero_()
        getattr(solver, c).copy_(value)
    oscillators = [(c, entry) for c, template in solver._dispersive_templates.items() for entry in template[kind]]
    # The sphere has partial quadrature cells despite binary runtime membership.
    nodes = solver._compiled_material_model[kind+"_poles"][0]["weight"]
    assert bool(((nodes > 0) & (nodes < 1)).any())
    coupling = max(float((o["drive"]/dt/getattr(solver, "eps_"+c)).max()) for c, o in oscillators)
    restoring = max(float(o.get("restoring", 0.))/dt for _, o in oscillators)
    bound = dt/2*(2*299792458.*math.sqrt(3)/h + math.sqrt(coupling+restoring))
    assert bound < 1
    def magnetic_energy():
        return sum((getattr(solver, "mu_"+c).double()*getattr(solver, c).double().square()).sum() for c in magnetic)
    initial = magnetic_energy()
    steps = math.ceil(20/frequency/dt)
    drift = 0.
    with torch.no_grad():
        for _ in range(steps):
            old_e = {c: getattr(solver, c).double().clone() for c in electric}
            old_p = [o["polarization"].double().clone() if kind == "lorentz" else None for _, o in oscillators]
            solver._update_magnetic_fields(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
            solver._advance_dispersive_state()
            solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
            solver._apply_dispersive_corrections()
            enforce_pec_boundaries(solver)
            energy = magnetic_energy() + sum((getattr(solver, "eps_"+c).double()*old_e[c]*getattr(solver, c).double()).sum() for c in electric)
            for (_, o), previous in zip(oscillators, old_p):
                alpha = o["drive"].double()/dt
                active = alpha > 0
                numerator = o["current"].double().square()
                if previous is not None:
                    numerator += o["restoring"]/dt*previous*o["polarization"].double()
                energy += (numerator[active]/alpha[active]).sum()
            assert torch.isfinite(energy) and energy > 0
            drift = max(drift, abs(float(energy/initial)-1))
    assert steps*dt*frequency >= 20
    print(f"{kind}: {steps} steps, {steps*dt*frequency:.9f} periods, energy drift={drift:.9g}")
    assert drift <= 1e-3, drift
