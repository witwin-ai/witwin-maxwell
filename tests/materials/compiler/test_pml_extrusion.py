"""Prepare-time PML continuation, with independent census and material oracles."""
import copy
import warnings
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.scene as scene_module
from benchmark.scenes import build_scene
from witwin.maxwell.scene import prepare_scene


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _census(prepared):
    """Original contact-and-stopping predicate, evaluated on prepared geometry."""
    hits = []
    for i, owner in enumerate([*prepared.structures, *prepared.material_regions]):
        for k, axis in enumerate("xyz"):
            nodes = getattr(prepared, axis)
            others = [a for a in "xyz" if a != axis]
            a, b = torch.meshgrid(*(getattr(prepared, o) for o in others), indexing="ij")

            def inside(value):
                coords = {axis: torch.full_like(a, float(value)), others[0]: a, others[1]: b}
                return owner.geometry.signed_distance(*(coords[o] for o in "xyz")) <= 0

            for side, bound, outer in (("low", prepared.domain.bounds[k][0], nodes[0]),
                                       ("high", prepared.domain.bounds[k][1], nodes[-1])):
                if prepared.boundary.face_kind(axis, side) == "pml":
                    interface = nodes[(nodes - bound).abs().argmin()]
                    if bool((inside(interface) & ~inside(outer)).any()):
                        hits.append((i, axis, side))
    return hits


def _equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    else:
        assert actual == expected


def _scene(geometry=None, *, device="cpu", regions=(), material=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.125),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="arithmetic"),
        device=device, material_regions=regions,
        structures=[] if geometry is None else [mw.Structure(
            geometry=geometry, material=material or mw.Material(eps_r=4), name="sample")],
    )


FLAGGED = [
    "dielectric_slab", "debye_slab", "anisotropic_slab", "sigma_e_drude_slab",
    "ricker_axis_x_anisotropic", "static_medium2d_sheet", "mode_source_higher_order",
    "fdfd_dielectric_slab", "fdfd_diag_aniso_slab", "fdfd_sigma_e_slab",
]


@pytest.mark.parametrize("name", FLAGGED)
def test_flagged_benchmark_census_is_clean_after_compile(name, monkeypatch):
    scene = build_scene(name)
    scene.device = "cpu"
    with monkeypatch.context() as patch:
        patch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
        before = prepare_scene(scene)
    assert _census(before)
    prepared = prepare_scene(scene)
    prepared.compile_materials()
    assert prepared.pml_extrusions
    assert not _census(prepared)
    assert prepared.structures[0] is not scene.structures[0]
    assert "pml_extrusions" not in scene.metadata
    assert prepared.metadata["pml_extrusions"] == prepared.pml_extrusions


def test_dielectric_slab_matches_manual_extension():
    scene = build_scene("dielectric_slab")
    manual = build_scene("dielectric_slab")
    manual.structures[0] = mw.Structure(
        mw.Box(size=(2., 2., .1)), manual.structures[0].material, name="slab")
    _equal(prepare_scene(scene).compile_materials(), prepare_scene(manual).compile_materials())


def test_drude_halfspace_matches_manual_extension():
    # Same physical bounds, spacing, material and half-space as the scattering test.
    material = mw.Material.drude(eps_inf=1., plasma_frequency=2.e9, gamma=1.e8)

    def build(width):
        return mw.Scene(
            domain=mw.Domain(bounds=((-1., 1.), (-.4, .4), (-.4, .4))),
            grid=mw.GridSpec.uniform(.015), boundary=mw.BoundarySpec.pml(num_layers=6),
            device="cpu", structures=[mw.Structure(
                mw.Box(position=(.6, 0., 0.), size=(1.2, width, width)), material)],
        )

    _equal(prepare_scene(build(.8)).compile_materials(), prepare_scene(build(1.2)).compile_materials())


@pytest.mark.parametrize("name", ["dielectric_sphere", "metal_sphere", "pec_box", "cylinder_scatter"])
def test_unflagged_benchmark_is_bitwise_unchanged(name, monkeypatch):
    scene = build_scene(name)
    prepared = prepare_scene(scene)
    assert not _census(prepared)
    assert prepared.pml_extrusions == []
    assert all(a is b for a, b in zip(scene.structures, prepared.structures))
    expected = prepared.compile_materials()
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    _equal(prepare_scene(scene).compile_materials(), expected)


def test_unflagged_density_and_nearby_soft_tails_are_bitwise_unchanged(monkeypatch):
    region = mw.MaterialRegion("interior", mw.Box(size=(.25, .25, .25)),
                               torch.full((2, 2, 2), .5), eps_bounds=(1, 4))
    # A soft tail reaches the interface but there is no hard contact node.
    scene = _scene(mw.Box(size=(.999, .25, .25)), regions=[region])
    prepared = prepare_scene(scene)
    assert prepared.pml_extrusions == []
    expected = prepared.compile_materials()
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    _equal(prepare_scene(scene).compile_materials(), expected)


def test_nonprismatic_sphere_warns_once_and_inside_sphere_does_not():
    touching = _scene(mw.Sphere(radius=.5))
    with pytest.warns(RuntimeWarning, match="structure 0.*non-prismatic.*x-low.*z-high") as caught:
        prepared = prepare_scene(touching)
    assert len(caught) == 1
    assert prepared.pml_extrusions == []
    assert prepared.structures[0] is touching.structures[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert prepare_scene(_scene(mw.Sphere(radius=.25))).pml_extrusions == []
    assert not caught


def _surface_material(kind):
    if kind == "lossy":
        return mw.LossyMetalMedium(conductivity=50.)
    if kind == "foil":
        return mw.CopperFoil(thickness=2.e-6, frequency_range=(.5e9, 5.e9))
    from tests.materials.surface_impedance.smooth_surface_probe import _rational_medium

    # Reuse the probe medium that supports the runtime's passive Z-form refit.
    return _rational_medium()[1]


def _surface_scene(kind, *, device="cpu", contact=True, both_faces=False):
    from tests.materials.surface_impedance.smooth_surface_probe import _metal_scene

    geometry = mw.Box(
        position=(0. if both_faces or not contact else .012, 0., 0.),
        size=(.04 if both_faces else .016, .02, 0. if kind == "foil" else .02))
    return _metal_scene(_surface_material(kind), device=device, geometry=geometry)


def _surface_layout(prepared):
    """Include every face/metal field dumped as layout.* by the smooth probe."""
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces

    compiled = dict(prepared.compile_materials())
    layout = compiled.pop("surface_impedance")
    faces = [vars(face) for face in layout.faces]
    metals = [{key: value for key, value in vars(metal).items() if key != "material"}
              for metal in layout.metals]
    foil_faces = [{key: value for key, value in vars(face).items() if key != "material"}
                  for face in compile_copper_faces(prepared, device=prepared.device)]
    assert faces or foil_faces
    return compiled, faces, metals, layout.total_area, foil_faces


@pytest.mark.parametrize("kind", ["lossy", "rational", "foil"])
@pytest.mark.parametrize("both_faces", [False, True])
def test_surface_impedance_contact_is_bitwise_unchanged(kind, both_faces, monkeypatch):
    scene = _surface_scene(kind, both_faces=both_faces)
    faces = ["x-low", "x-high"] if both_faces else ["x-high"]
    with pytest.warns(RuntimeWarning, match=(
        r"structure 0 \('metal'\).*x-high: surface-impedance conductor reaches a PML interface; not extruded"
    )) as caught:
        prepared = prepare_scene(scene)
    assert len(caught) == 1
    assert prepared.structures[0] is scene.structures[0]
    assert [axis + "-" + side for _, axis, side in _census(prepared)] == faces
    assert prepared.pml_extrusions == [dict(
        structure_index=0, name="metal", faces=faces, kind="skipped_surface_impedance")]
    assert prepared.metadata["pml_extrusions"] == prepared.pml_extrusions
    assert "pml_extrusions" not in scene.metadata
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    baseline = prepare_scene(scene)
    _equal(prepared.compile_material_tensors(), baseline.compile_material_tensors())
    _equal(_surface_layout(prepared), _surface_layout(baseline))


@pytest.mark.parametrize("kind", ["lossy", "rational", "foil"])
def test_interior_surface_impedance_has_no_warning_or_record(kind):
    scene = _surface_scene(kind, contact=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = prepare_scene(scene)
    assert not caught
    assert not _census(prepared)
    assert prepared.pml_extrusions == []
    assert prepared.structures[0] is scene.structures[0]


@pytest.mark.parametrize("flag", ["is_surface_impedance", "is_lossy_metal", "is_copper_foil"])
def test_surface_impedance_predicate_accepts_material_flags(flag):
    material = SimpleNamespace(**{flag: True})
    assert scene_module._pml_surface_impedance_material(material)
    scene = _scene(mw.Box(size=(1., .25, .25)))
    object.__setattr__(scene.structures[0].material, flag, True)
    with pytest.warns(RuntimeWarning, match="surface-impedance conductor reaches a PML interface") as caught:
        prepared = prepare_scene(scene)
    assert len(caught) == 1
    assert prepared.structures[0] is scene.structures[0]
    assert prepared.pml_extrusions[0]["kind"] == "skipped_surface_impedance"


@pytest.mark.parametrize("material", [mw.Material.pec(), mw.Material.pmc()], ids=["pec", "pmc"])
def test_perfect_conductor_box_is_still_extruded(material):
    assert not scene_module._pml_surface_impedance_material(material)
    scene = _scene(mw.Box(size=(1., .25, .25)), material=material)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = prepare_scene(scene)
    assert not caught
    assert prepared.structures[0] is not scene.structures[0]
    assert prepared.pml_extrusions[0]["kind"] == "box"
    assert not _census(prepared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["lossy", "rational"])
def test_surface_impedance_solver_and_fields_bitwise_cuda(kind, monkeypatch):
    from tests.materials.surface_impedance.smooth_surface_probe import _solver_tensors

    scene = _surface_scene(kind, device="cuda")

    def snapshot():
        simulation = mw.Simulation.fdtd(
            # The 8-layer closed-transverse cell exercises the CPML kernels: pin CPML explicitly.
            scene, frequency=2.e9, absorber="cpml", run_time=mw.TimeConfig(time_steps=24),
            full_field_dft=True, spectral_sampler=mw.SpectralSampler(window="none"))
        out = {}
        _solver_tensors(out, "prepared", simulation.prepare().solver)
        solver = simulation.run().solver
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            out[f"run.{name}"] = getattr(solver, name).detach().clone()
        assert any(key.startswith("prepared._surface_impedance") for key in out)
        assert any(bool(out[f"run.{name}"].abs().any()) for name in ("Ex", "Ey", "Ez"))
        return out

    with pytest.warns(RuntimeWarning, match="surface-impedance conductor reaches a PML interface"):
        actual = snapshot()
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    _equal(actual, snapshot())


def test_later_structure_keeps_priority_inside_pml():
    scene = _scene(mw.Box(size=(1., .5, .5)))
    later = mw.Structure(mw.Box(position=(.5, 0, 0), size=(1., .25, .25)), mw.Material(eps_r=9))
    scene.add_structure(later)
    prepared = prepare_scene(scene)
    eps, _ = prepared.compile_relative_materials()
    assert eps[-1, 6, 6] == 9
    assert prepared.structures[1] is later
    assert prepared.structures[0].material is scene.structures[0].material


def test_extrusion_records_reach_simulation_statistics():
    scene = _scene(mw.Box(size=(1., .25, .25)))
    prepared = prepare_scene(scene)
    simulation = mw.Simulation.fdtd(scene, frequency=1.e9)
    runtime = SimpleNamespace(scene=prepared, dt=1.e-12, _spectral_reference_frequency=None)
    stats = simulation._build_fdtd_solver_stats(
        runtime, time_steps=16, use_full_field_dft=False, dft_cfg=simulation.config.spectral_sampler)
    assert stats["pml_extrusions"] == prepared.pml_extrusions


def test_prismatic_cylinder_uses_wrapper_and_preserves_interior():
    original = mw.Cylinder(radius=.25, height=1., axis="x")
    scene = _scene(original)
    prepared = prepare_scene(scene)
    assert prepared.pml_extrusions[0]["kind"] == "wrapper"
    assert not _census(prepared)
    coords = torch.meshgrid(*(torch.linspace(-.375, .375, 7) for _ in range(3)), indexing="ij")
    assert torch.equal(prepared.structures[0].geometry.signed_distance(*coords),
                       original.signed_distance(*coords))
    # The accepted wrapper contract continues the interface SDF itself.
    q = (torch.full((3,), .75), torch.tensor([0., .125, .375]), torch.zeros(3))
    assert torch.equal(prepared.structures[0].geometry.signed_distance(*q),
                       original.signed_distance(q[0].clamp(max=.5), *q[1:]))
    assert all(t.isfinite().all() for t in prepared.compile_material_tensors())


@pytest.mark.parametrize("variable", [False, True])
def test_density_edge_replication_and_nonprismatic_warning(variable):
    density = torch.tensor([.25, .75]).view(1, 2, 1).expand(8, 2, 2).clone()
    if variable:
        density[-2] = 0.
    region = mw.MaterialRegion("design", mw.Box(size=(1., .25, .25)), density, eps_bounds=(1, 5))
    scene = _scene(regions=[region])
    if variable:
        with pytest.warns(RuntimeWarning, match="MaterialRegion.*x-high"):
            prepared = prepare_scene(scene)
        assert prepared.pml_extrusions == []
        return
    prepared = prepare_scene(scene)
    assert prepared.pml_extrusions[0]["kind"] == "density"
    assert not _census(prepared)
    # align_corners=False: 8 cells over width 1, plus 4 replicated cells per side.
    padded = torch.cat((density[:1].expand(4, -1, -1), density, density[-1:].expand(4, -1, -1)))
    manual = _scene(regions=[mw.MaterialRegion(
        "design", mw.Box(size=(2., .25, .25)), padded, eps_bounds=(1, 5))])
    _equal(prepared.compile_material_tensors(), prepare_scene(manual).compile_material_tensors())
    assert scene.material_regions[0] is region
    assert region.geometry.size[0] == 1.


def test_density_prism_decision_uses_exact_native_slices():
    generator = torch.Generator().manual_seed(0)
    density = torch.rand((1, 5, 7), generator=generator).expand(8, -1, -1).clone()
    region = mw.MaterialRegion("design", mw.Box(size=(1., .375, .375)), density, eps_bounds=(1, 5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = prepare_scene(_scene(regions=[region]))
    assert not caught
    assert prepared.pml_extrusions[0]["faces"] == ["x-low", "x-high"]
    # Texture coordinates and native filtering remain identical in the interior.
    from witwin.maxwell.compiler.materials import _sample_material_region_density
    coords = torch.meshgrid(*(torch.linspace(-.375, .375, 7) for _ in range(3)), indexing="ij")
    actual = _sample_material_region_density(prepared, prepared.material_regions[0], density, coords=coords)
    expected = _sample_material_region_density(prepared, region, density, coords=coords)
    assert torch.equal(actual, expected)


def _trainable_slab(parameters, *, manual=False, device="cpu"):
    width, thickness, height = parameters.unbind()
    size = torch.stack((width * 0 + 2. if manual else width, width * 0 + .375, thickness))
    position = torch.stack((height * 0, height * 0, height))
    return _scene(mw.Box(position=position, size=size, device=device), device=device)


def _slab_loss(parameters, *, manual=False, device="cpu"):
    prepared = prepare_scene(_trainable_slab(parameters, manual=manual, device=device))
    eps, _ = prepared.compile_relative_materials()
    return (eps * (1 + prepared.Z)).sum()


def test_trainable_box_extruded_face_gradient_is_zero_and_other_gradients_match():
    parameters = torch.tensor([1., .29, .03], requires_grad=True)
    actual = torch.autograd.grad(_slab_loss(parameters), parameters)[0]
    reference = torch.autograd.grad(_slab_loss(parameters, manual=True), parameters)[0]
    # The two extruded x faces are replaced by grid constants by construction.
    # Their gradient must stay exactly zero; do not restore face sensitivity.
    assert actual[0].item() == 0.
    assert bool((actual[1:].abs() > 0).all())
    assert (actual[1:] - reference[1:]).abs().max() <= 1.e-4 * reference[1:].abs().max()


def test_single_extruded_box_face_has_zero_gradient():
    faces = torch.tensor([-.23, .5], requires_grad=True)
    low, high = faces.unbind()
    geometry = mw.Box(position=torch.stack(((low + high) / 2, low * 0, low * 0)),
                      size=torch.stack((high - low, low * 0 + .375, low * 0 + .375)))
    prepared = prepare_scene(_scene(geometry))
    assert prepared.pml_extrusions[0]["faces"] == ["x-high"]
    eps, _ = prepared.compile_relative_materials()
    gradient = torch.autograd.grad(eps.sum(), faces)[0]
    # The high face is a grid constant after extrusion; only the low face remains trainable.
    assert gradient[1].item() == 0.
    assert gradient[0].abs() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trainable_box_nonextruded_parameter_finite_difference_cuda():
    parameters = torch.tensor([1., .29, .03], device="cuda", requires_grad=True)
    gradient = torch.autograd.grad(_slab_loss(parameters, device="cuda"), parameters)[0]
    step = 1.e-4
    delta = parameters.new_tensor([0., step, 0.])
    finite_difference = (_slab_loss(parameters.detach() + delta, device="cuda")
                         - _slab_loss(parameters.detach() - delta, device="cuda")) / (2 * step)
    torch.testing.assert_close(gradient[1], finite_difference, rtol=.01, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fdtd_trainable_slab_two_horizon_gradients_and_finite_difference_cuda():
    def loss(parameters, horizon, *, manual=False):
        scene = _trainable_slab(parameters, manual=manual, device="cuda")
        scene.add_source(mw.PointDipole(
            position=(0., 0., -.25), polarization="Ex", width=.04,
            source_time=mw.GaussianPulse(frequency=1.e9, fwidth=1.e9, amplitude=50.)))
        scene.add_monitor(mw.PointMonitor("probe", (0., 0., .25), fields=("Ex",)))
        result = mw.Simulation.fdtd(
            scene, frequency=1.e9, run_time=mw.TimeConfig(time_steps=horizon),
            full_field_dft=False, spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        assert result.stats()["pml_extrusions"] == ([] if manual else [
            {"structure_index": 0, "name": "sample", "faces": ["x-low", "x-high"], "kind": "box"}])
        return result.monitor("probe")["data"].abs().square().sum()

    differences = []
    for horizon in (24, 48):
        parameters = torch.tensor([1., .29, .03], device="cuda", requires_grad=True)
        gradient = torch.autograd.grad(loss(parameters, horizon), parameters)[0]
        reference = torch.autograd.grad(loss(parameters, horizon, manual=True), parameters)[0]
        # Extruded face sensitivity is intentionally zero in the full adjoint too.
        assert gradient[0].item() == 0.
        assert bool((reference[1:].abs() > 0).any())
        difference = (gradient[1:] - reference[1:]).abs().max()
        assert difference <= 1.e-4 * reference[1:].abs().max()
        differences.append(difference.item())
    assert differences[1] <= 4 * differences[0]
    delta = parameters.new_tensor([0., 1.e-4, 0.])
    finite_difference = (loss(parameters.detach() + delta, 48)
                         - loss(parameters.detach() - delta, 48)) / (2.e-4)
    torch.testing.assert_close(gradient[1], finite_difference, rtol=.02, atol=0)


def _mode_context(prepared):
    from witwin.core.material import VACUUM_PERMITTIVITY
    return SimpleNamespace(
        scene=prepared, dx=prepared.dx, dy=prepared.dy, dz=prepared.dz,
        Ex=torch.empty(1, device=prepared.device), c=299792458., eps0=VACUUM_PERMITTIVITY,
        boundary_kind=prepared.boundary.kind, _compiled_material_model=prepared.compile_materials(),
        _mode_source_rebuild_from_fields=False,
    )


def test_higher_order_mode_plane_eigenpair_and_profile_unchanged(monkeypatch):
    from witwin.maxwell.compiler.sources import _compile_mode_source
    from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile

    scene = build_scene("mode_source_higher_order")
    source = _compile_mode_source(scene.sources[0], default_frequency=2.e9)
    auto = prepare_scene(scene)
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    baseline = prepare_scene(scene)
    # Seed each independent sparse eigen-solve identically to fix its gauge.
    torch.manual_seed(0)
    a = solve_mode_source_profile(_mode_context(auto), copy.deepcopy(source))
    torch.manual_seed(0)
    b = solve_mode_source_profile(_mode_context(baseline), copy.deepcopy(source))
    assert a["effective_index"] == b["effective_index"]
    assert a["beta"] == b["beta"]
    _equal(a["profile"], b["profile"])


def _sheet_geometry(kind, *, contact=True):
    low = -.5 if contact else -.25
    vertices = [(low, -.25, 0.), (.25, -.25, 0.), (.25, .25, 0.), (low, .25, 0.)]
    faces = [(0, 1, 2), (0, 2, 3)]
    if kind == "planar":
        return mw.PlanarPEC.rectangle(((low, .25), (-.25, .25)))
    if kind == "surface":
        return mw.SurfaceMesh(vertices, faces)
    return mw.Mesh(vertices, faces, fill_mode="surface", recenter=False)


@pytest.mark.parametrize("kind", ["planar", "surface", "mesh_surface"])
@pytest.mark.parametrize("contact", [False, True])
def test_sheet_dispatch_and_contact_records(kind, contact, monkeypatch):
    scene = _scene(_sheet_geometry(kind, contact=contact), material=mw.Material.pec())
    original = scene.structures[0].geometry
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = prepare_scene(scene)
    assert not caught
    assert prepared.structures[0].geometry is original
    assert type(prepared.structures[0].geometry) is type(original)
    assert prepared.pml_extrusions == ([dict(
        structure_index=0, name="sample", faces=["x-low"], kind="skipped_sheet")] if contact else [])
    expected = prepared.compile_materials()
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    _equal(prepare_scene(scene).compile_materials(), expected)


@pytest.mark.parametrize("kind", ["planar", "surface", "mesh_surface"])
def test_boolean_sheet_leaf_skips_without_sampling_or_warning(kind, monkeypatch):
    geometry = mw.Union(mw.Box(size=(1., .25, .25)), mw.Union(_sheet_geometry(kind)))
    scene = _scene(geometry)

    def fail_census(*_):
        pytest.fail("Boolean geometry containing a sheet must not be sampled")

    monkeypatch.setattr(scene_module, "_pml_contact_faces", fail_census)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = prepare_scene(scene)
    assert not caught
    assert prepared.structures[0].geometry is scene.structures[0].geometry
    assert prepared.pml_extrusions == []


@pytest.mark.parametrize("boolean", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
def test_mesh_volume_contact_warns_and_preserves_compilation(boolean, samples, monkeypatch):
    vertices, faces = mw.Box(size=(1., .25, .25)).to_mesh()
    geometry = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
    if boolean:
        geometry = mw.Union(geometry, mw.Box(size=(.25, .125, .125)))
    scene = _scene(geometry)
    scene.subpixel = mw.SubpixelSpec(samples=samples, averaging="arithmetic")
    with pytest.warns(RuntimeWarning, match="structure 0.*mesh geometry") as caught:
        prepared = prepare_scene(scene)
    assert len(caught) == 1
    assert prepared.pml_extrusions == []
    assert prepared.structures[0].geometry is scene.structures[0].geometry
    expected = prepared.compile_materials()
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    _equal(prepare_scene(scene).compile_materials(), expected)


def test_box_euler_zero_rotation_uses_enlarged_box():
    prepared = prepare_scene(_scene(mw.Box(size=(1., .25, .25), rotation=torch.zeros(3))))
    assert prepared.pml_extrusions[0]["kind"] == "box"
    assert isinstance(prepared.structures[0].geometry, mw.Box)


@pytest.mark.parametrize("geometry", [
    mw.Box(size=(.25, .25, .25)),
    mw.Box(size=(1., .25, .25)),
    mw.Box(position=(.49, 0., 0.), size=(.25, .25, .25), rotation=(0., 0., .4)),
    mw.Sphere(radius=.5),
    mw.Sphere(position=(.25, 0., 0.), radius=.25),
    mw.Sphere(radius=.1),
    mw.Mesh(*mw.Box(size=(1., .25, .25)).to_mesh(), recenter=False, fill_mode="solid"),
    mw.Mesh(*_sheet_geometry("surface").to_mesh(), recenter=False, fill_mode="auto"),
    _sheet_geometry("planar"),
    _sheet_geometry("surface"),
    _sheet_geometry("mesh_surface"),
    mw.Mesh([(-.46875, -.25, 0.), (.25, -.25, 0.), (.25, .25, 0.), (-.46875, .25, 0.)],
            [(0, 1, 2), (0, 2, 3)], recenter=False, fill_mode="surface", surface_thickness=.05),
])
def test_conservative_prefilter_preserves_exact_contact_census(geometry, monkeypatch):
    monkeypatch.setattr(scene_module, "_extrude_pml_contacts", lambda _: [])
    prepared = prepare_scene(_scene(geometry))
    actual = scene_module._pml_contact_faces(prepared, geometry)
    monkeypatch.setattr(scene_module, "_pml_conservative_bounds", lambda *_: None)
    expected = scene_module._pml_contact_faces(prepared, geometry)
    _equal(actual, expected)


@pytest.mark.parametrize("projection", [None, 3.])
@pytest.mark.parametrize("filter_radius", [None, .125])
def test_extruded_density_pullback_matches_forward_compile_autograd(projection, filter_radius):
    from witwin.maxwell.fdtd.material_pullback import pullback_density_gradients

    x_slices = 8 if filter_radius is None else 2
    density = torch.linspace(-.6, 1.6, 9).reshape(1, 3, 3).expand(x_slices, -1, -1).clone().requires_grad_()
    region = mw.MaterialRegion("design", mw.Box(size=(1., .375, .375)), density,
                               eps_bounds=(1, 5), bounds=(-.5, 1.5),
                               projection_beta=projection, filter_radius=filter_radius)
    scene = _scene(regions=[region])
    prepared = prepare_scene(scene)
    assert prepared.pml_extrusions[0]["kind"] == "density"
    eps, _ = prepared.compile_relative_materials()
    generator = torch.Generator().manual_seed(41)
    grads = [torch.randn(shape, generator=generator) for shape in (
        (prepared.Nx-1, prepared.Ny, prepared.Nz),
        (prepared.Nx, prepared.Ny-1, prepared.Nz),
        (prepared.Nx, prepared.Ny, prepared.Nz-1))]
    edges = [.5*(eps[:-1]+eps[1:]), .5*(eps[:, :-1]+eps[:, 1:]), .5*(eps[:, :, :-1]+eps[:, :, 1:])]
    expected, = torch.autograd.grad(sum((edge*g).sum() for edge, g in zip(edges, grads)), density)
    repeated = [pullback_density_gradients(scene, density_tensors=(density,), trainable_region_indices=(0,),
        grad_eps_ex=grads[0], grad_eps_ey=grads[1], grad_eps_ez=grads[2], eps0=1.)[0] for _ in range(3)]
    actual = repeated[0]
    assert all(torch.equal(actual, value) for value in repeated[1:])
    assert (actual-expected).abs().max() <= 64*torch.finfo(torch.float32).eps*expected.abs().max()


def test_nonextruded_density_pullback_keeps_legacy_interpolation():
    from witwin.maxwell.fdtd.material_pullback import _pullback_region_density, _resolve_region_layout

    density = torch.full((2, 3, 2), .4)
    region = mw.MaterialRegion("interior", mw.Box(size=(.375, .375, .375)), density, eps_bounds=(1, 5))
    prepared = prepare_scene(_scene(regions=[region]))
    assert prepared.pml_extrusions == []
    layout = _resolve_region_layout(prepared, region)
    patch = torch.arange(torch.tensor(layout.target_shape).prod()).float().reshape(layout.target_shape)/10
    expected = torch.ops.aten.upsample_trilinear3d_backward(
        (patch*4)[None, None].contiguous(), list(layout.target_shape), [1, 1, *density.shape],
        False, None, None, None)[0, 0]
    actual = _pullback_region_density(prepared, region, layout=layout, grad_eps_patch=patch)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extruded_density_edge_texel_fdtd_finite_difference_cuda():
    def loss(density):
        region = mw.MaterialRegion("design", mw.Box(size=(1., .375, .375), device="cuda"),
                                   density, eps_bounds=(1, 5))
        scene = _scene(regions=[region], device="cuda")
        scene.add_source(mw.PointDipole(position=(.5, 0., -.25), polarization="Ex", width=.04,
            source_time=mw.GaussianPulse(frequency=1.e9, fwidth=1.e9, amplitude=50.)))
        scene.add_monitor(mw.PointMonitor("probe", (.5, 0., .25), fields=("Ex",)))
        result = mw.Simulation.fdtd(scene, frequency=1.e9, run_time=mw.TimeConfig(time_steps=48),
            full_field_dft=False, spectral_sampler=mw.SpectralSampler(window="none")).run()
        assert result.stats()["pml_extrusions"][0]["kind"] == "density"
        return result.monitor("probe")["data"].abs().square().sum()

    # Two x slices both border-sample at the physical interface, so perturbing
    # the high edge texel preserves the exact prismatic-contact decision.
    density = torch.full((2, 3, 3), .5, device="cuda", requires_grad=True)
    gradient, = torch.autograd.grad(loss(density), density)
    step = 1.e-3
    delta = torch.zeros_like(density)
    delta[-1, 1, 1] = step
    fd = (loss(density.detach()+delta)-loss(density.detach()-delta))/(2*step)
    assert fd.abs() > 0
    torch.testing.assert_close(gradient[-1, 1, 1], fd, rtol=.02, atol=0)


def test_extrusion_stats_result_save_load_roundtrip(tmp_path):
    scene = _scene(mw.Box(size=(1., .25, .25)))
    records = prepare_scene(scene).pml_extrusions
    assert records
    result = mw.Result(scene=scene, method="fdtd", frequency=1.e9, fields={}, monitors={},
                       solver_stats={"pml_extrusions": records}, metadata={"pml_extrusions": records})
    path = tmp_path / "extruded.pt"
    result.save(path)
    restored = mw.Result.load(path, scene=scene)
    assert restored.stats()["pml_extrusions"] == records
    assert restored.metadata["pml_extrusions"] == records
