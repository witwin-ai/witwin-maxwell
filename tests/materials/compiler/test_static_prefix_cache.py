"""Identity-keyed static prefix cache: cached compiles are torch.equal to uncached ones."""
from dataclasses import fields, is_dataclass
import gc
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler import compile_materials, static_cache
from witwin.maxwell.compiler.materials import LocalMaterialCache
from witwin.maxwell.compiler.static_cache import GEOMETRY_MEMO, STATIC_PREFIX_CACHE
from witwin.maxwell.scene import prepare_scene

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"))]


_COUNTERS = ("hits", "misses", "frames", "evictions", "dropped", "stale", "not_admitted", "not_admitted_bytes")


@pytest.fixture(autouse=True)
def stats():
    """Cache statistics of this test: cumulative counters relative to its start, live sizes as they are."""
    gc.collect()
    STATIC_PREFIX_CACHE.clear()
    GEOMETRY_MEMO.clear()
    baseline = STATIC_PREFIX_CACHE.stats
    yield lambda: {key: value - baseline[key] if key in _COUNTERS else value
                   for key, value in STATIC_PREFIX_CACHE.stats.items()}
    STATIC_PREFIX_CACHE.clear()
    GEOMETRY_MEMO.clear()


def assert_equal_models(actual, expected, path="model"):
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert actual.device == expected.device and torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_equal_models(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_equal_models(a, b, f"{path}[{index}]")
    elif is_dataclass(expected):
        for field in fields(expected):
            assert_equal_models(getattr(actual, field.name), getattr(expected, field.name), f"{path}.{field.name}")
    else:
        assert actual == expected, path


def uncached(prepared, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(static_cache, "FRAME_BYTES", 0)
        return compile_materials(prepared)


def structures(kind, device):
    box = mw.Box(position=(-.1, 0., 0.), size=(.2, .25, .15), device=device)
    sphere = mw.Sphere(position=(.12, .05, 0.), radius=.09, device=device)
    plain = (mw.Structure(name="box", geometry=box, material=mw.Material(eps_r=2.5)),
             mw.Structure(name="sphere", geometry=sphere, material=mw.Material(eps_r=3.1, sigma_e=.02)))
    if kind == "plain":
        return plain
    if kind == "anisotropic":
        return plain + (mw.Structure(name="aniso", geometry=mw.Box(position=(0., -.12, .05), size=(.3, .08, .1), device=device),
                                     material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2., 3., 4.), sigma_m=.5)),)
    if kind == "dispersive":
        return plain + (mw.Structure(name="poles", geometry=mw.Box(position=(0., .12, -.05), size=(.25, .1, .1), device=device),
                                     material=mw.Material(eps_r=1.5, sigma_e=.1, lorentz_poles=(
                                         mw.LorentzPole(delta_eps=2., resonance_frequency=2e9, gamma=2e8),),
                                         debye_poles=(mw.DebyePole(delta_eps=1., tau=1e-10),))),)
    if kind == "pec_sheet":
        return plain + (
            mw.Structure(name="pec", geometry=mw.Box(position=(0., -.15, 0.), size=(.2, .05, .2), device=device),
                         material=mw.Material.pec()),
            mw.Structure(name="sheet", geometry=mw.Box(position=(0., .15, 0.), size=(.2, .2, 0.), device=device),
                         material=mw.media.Medium2D(sigma_s=.3)))
    raise ValueError(kind)


class Design(mw.SceneModule):
    """The recommended pattern: the static part is built once, ``to_scene`` replaces the regions."""

    def __init__(self, *, kind="plain", samples=1, averaging="arithmetic", device="cpu", through_pml=False, trainable=True):
        super().__init__()
        density = torch.full((4, 5, 3), .4, device=device)
        self.density = torch.nn.Parameter(density) if trainable else density
        static = structures(kind, device)
        if through_pml:
            static += (mw.Structure(name="guide", geometry=mw.Box(position=(0., 0., -.2), size=(.64, .1, .06), device=device),
                                    material=mw.Material(eps_r=4.)),)
        self._base = mw.Scene(domain=mw.Domain(bounds=((-.3, .3),) * 3), grid=mw.GridSpec.uniform(.04),
                              boundary=mw.BoundarySpec.pml(num_layers=2),
                              subpixel=mw.SubpixelSpec(samples=(samples,) * 3, averaging=averaging),
                              structures=static)
        self._region = mw.Box(position=(.05, .02, .1), size=(.16, .2, .12), device=device)

    def to_scene(self):
        return self._base.updated(material_regions=(mw.MaterialRegion(
            name="design", geometry=self._region, density=self.density, eps_bounds=(1., 4.), mu_bounds=(1., 1.2)),))


CASES = [("plain", 1, "arithmetic"), ("plain", 2, "arithmetic"), ("plain", 2, "polarized"),
         ("anisotropic", 1, "arithmetic"), ("dispersive", 1, "arithmetic"), ("dispersive", 2, "polarized"),
         ("pec_sheet", 1, "arithmetic")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind,samples,averaging", CASES)
def test_cached_compiles_equal_uncached_and_are_deterministic(kind, samples, averaging, device, monkeypatch, stats):
    module = Design(kind=kind, samples=samples, averaging=averaging, device=device)
    references = [uncached(prepare_scene(module.to_scene(), device=device), monkeypatch) for _ in range(3)]
    for reference in references[1:]:
        assert_equal_models(reference, references[0])
    assert stats()["frames"] == 0
    for call in range(4):
        before = stats()
        model = compile_materials(prepare_scene(module.to_scene(), device=device))
        after = stats()
        assert_equal_models(model, references[0])
        stored = after["entries"]
        assert stored > 0
        if call:
            assert (after["hits"] - before["hits"], after["misses"] - before["misses"]) == (stored, 0)
        else:
            assert (after["hits"], after["misses"]) == (0, stored)


# CUDA grid_sample backward accumulates with atomics, so density gradients are compared on CPU only.
@pytest.mark.parametrize("device", ["cpu"])
def test_density_gradients_are_equal_through_a_hit(device, monkeypatch, stats):
    module = Design(device=device)
    expected = torch.autograd.grad(uncached(prepare_scene(module.to_scene(), device=device), monkeypatch)["eps_r"].sum(),
                                   module.density)[0]
    for _ in range(2):
        model = compile_materials(prepare_scene(module.to_scene(), device=device))
        assert torch.equal(torch.autograd.grad(model["eps_r"].sum(), module.density)[0], expected)
    assert stats()["hits"] > 0


def test_scenemodule_loop_compiles_the_static_prefix_once(stats):
    """Recompile avoidance: one node sample and six edge components are compiled once, then reused."""
    module = Design()
    for _ in range(3):
        compile_materials(prepare_scene(module.to_scene(), device="cpu"))
    counts = stats()
    assert (counts["misses"], counts["hits"], counts["entries"], counts["frames"]) == (7, 14, 7, 1)


def test_scenemodule_to_scene_is_structural_sharing():
    """``to_scene`` on 1,000 static structures shares them and stays under 0.2 ms.

    The best of five batches is compared with the plan budget; the measured mean is about 0.02 ms.
    """
    material = mw.Material(eps_r=2.)
    base = mw.Scene(domain=mw.Domain(bounds=((-1., 1.),) * 3), grid=mw.GridSpec.uniform(.1),
                    boundary=mw.BoundarySpec.pml(num_layers=2),
                    structures=tuple(mw.Structure(name=f"s{i}", geometry=mw.Box(position=(0., 0., 0.), size=(.1, .1, .1), device="cpu"),
                                                  material=material) for i in range(1000)))
    region = mw.MaterialRegion(name="design", geometry=mw.Box(size=(.4, .4, .4), device="cpu"), density=torch.full((4, 4, 4), .5))
    scene = base.updated(material_regions=(region,))
    assert scene.structures is base.structures
    batches = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(200):
            base.updated(material_regions=(region,))
        batches.append((time.perf_counter() - start) / 200)
    assert min(batches) <= 2e-4


def test_returned_tensors_never_alias_the_cache(monkeypatch):
    module = Design()
    prepared = prepare_scene(module.to_scene(), device="cpu")
    frame = STATIC_PREFIX_CACHE.frame(prepared, (1, 1, 1), 1.0, 1.0)
    for _ in range(3):
        value = STATIC_PREFIX_CACHE.prefix(frame, ("probe",), lambda: {"a": [torch.ones(3)]})
        assert torch.equal(value["a"][0], torch.ones(3))
        value["a"][0].add_(5.)
    expected = uncached(prepare_scene(module.to_scene(), device="cpu"), monkeypatch)
    for _ in range(3):
        model = compile_materials(prepare_scene(module.to_scene(), device="cpu"))
        assert_equal_models(model, expected)
        with torch.no_grad():
            for channel in (*model["eps_components"].values(), *model["edge_components"]["eps"].values(),
                            model["sigma_e"], model["mu_r"]):
                channel.add_(7.)


@pytest.mark.parametrize("device", DEVICES)
def test_in_place_geometry_edit_misses(device, monkeypatch, stats):
    module = Design(device=device)
    compile_materials(prepare_scene(module.to_scene(), device=device))
    box = module._base.structures[0].geometry
    box.position.data[0] += .03  # bypasses the version counter
    expected = uncached(prepare_scene(module.to_scene(), device=device), monkeypatch)
    before = stats()
    model = compile_materials(prepare_scene(module.to_scene(), device=device))
    after = stats()
    assert_equal_models(model, expected)
    assert after["stale"] == before["stale"] + 1 and after["hits"] == before["hits"]


def test_structure_death_drops_the_frame(stats):
    module = Design()
    compile_materials(prepare_scene(module.to_scene(), device="cpu"))
    assert stats()["live_frames"] == 1 and stats()["bytes"] > 0
    del module
    gc.collect()
    counts = stats()
    assert (counts["dropped"], counts["live_frames"], counts["bytes"]) == (1, 0, 0)
    fresh = Design()
    compile_materials(prepare_scene(fresh.to_scene(), device="cpu"))
    assert stats()["live_frames"] == 1


def test_trainable_structures_and_oversized_scenes_are_not_admitted(monkeypatch, stats):
    module = Design()
    trainable = mw.Box(position=(0., 0., 0.), size=(.1, .1, .1), device="cpu")
    trainable.position.requires_grad_(True)
    scene = module.to_scene()
    scene = scene.updated(structures=(*scene.structures, mw.Structure(name="t", geometry=trainable, material=mw.Material(eps_r=2.))),
                          subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"))
    compile_materials(prepare_scene(scene, device="cpu"))
    assert (stats()["not_admitted"], stats()["frames"]) == (1, 0)
    monkeypatch.setattr(static_cache, "FRAME_BYTES", 1024)
    compile_materials(prepare_scene(module.to_scene(), device="cpu"))
    assert (stats()["not_admitted_bytes"], stats()["frames"]) == (1, 0)


def test_frames_are_evicted_least_recently_used_first(stats):
    modules = [Design() for _ in range(3)]
    for module in modules:
        compile_materials(prepare_scene(module.to_scene(), device="cpu"))
    counts = stats()
    assert (counts["frames"], counts["evictions"], counts["live_frames"]) == (3, 1, 2)


def test_local_material_cache_crop_frame_is_equal(monkeypatch, stats):
    module = Design(trainable=False)
    base = prepare_scene(module.to_scene(), device="cpu")
    with monkeypatch.context() as patch:
        patch.setattr(static_cache, "FRAME_BYTES", 0)
        reference = LocalMaterialCache(base, ("design",))
        expected = [reference.compile(prepare_scene(module.to_scene(), device="cpu")) for _ in range(2)]
    STATIC_PREFIX_CACHE.clear()
    local = LocalMaterialCache(base, ("design",))
    actual = [local.compile(prepare_scene(module.to_scene(), device="cpu")) for _ in range(2)]
    for a, b in zip(actual, expected):
        assert_equal_models(a, b)
    counts = stats()
    assert counts["frames"] == 2 and counts["hits"] > 0


def test_pml_contact_census_is_reused_by_geometry_identity(monkeypatch):
    hits = GEOMETRY_MEMO.stats["hits"]
    module = Design(through_pml=True)
    with monkeypatch.context() as patch:
        patch.setattr(static_cache, "GEOMETRY_ENTRIES", 0)
        first = prepare_scene(module.to_scene(), device="cpu")
    second, third = (prepare_scene(module.to_scene(), device="cpu") for _ in range(2))
    assert GEOMETRY_MEMO.stats["hits"] > hits
    assert any(record["name"] == "guide" and record["kind"] == "box" for record in first.pml_extrusions)
    for prepared in (second, third):
        assert prepared.pml_extrusions == first.pml_extrusions
        for a, b in zip(prepared.scene.structures, first.scene.structures):
            assert type(a.geometry) is type(b.geometry)
            for name, value in vars(b.geometry).items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(getattr(a.geometry, name), value)
    assert_equal_models(compile_materials(third), uncached(first, monkeypatch))



def _pml_scene(geometry):
    return mw.Scene(domain=mw.Domain(bounds=((-.3, .3),) * 3), grid=mw.GridSpec.uniform(.04),
                    boundary=mw.BoundarySpec.pml(num_layers=2),
                    structures=(mw.Structure(name="g", geometry=geometry, material=mw.Material(eps_r=2.)),))


@pytest.mark.parametrize("device", DEVICES)
def test_pml_memo_entries_die_with_their_geometry(device):
    """A loop of fresh geometries, each dropped after its prepare, leaves no live entry or snapshot."""
    import weakref

    before = GEOMETRY_MEMO.stats
    probe = None
    for index in range(2000):
        geometry = mw.Box(position=(0., 0., -.2 + 1e-4 * index), size=(.64, .1, .06), device=device)
        if index == 10:
            probe = weakref.ref(geometry.position)
        prepared = prepare_scene(_pml_scene(geometry), device=device)
        assert prepared.pml_extrusions[0]["kind"] == "box"
        del geometry, prepared
    gc.collect()
    stats = GEOMETRY_MEMO.stats
    assert stats["misses"] - before["misses"] == 2000
    assert (stats["entries"], stats["bytes"]) == (0, 0)
    assert probe() is None


def test_pml_memo_bytes_are_bounded_least_recently_used_first(monkeypatch):
    monkeypatch.setattr(static_cache, "GEOMETRY_BYTES", 40 * 10)
    geometries = [mw.Box(position=(0., 0., .01 * i), size=(.64, .1, .06), device="cpu") for i in range(30)]
    for geometry in geometries:
        prepare_scene(_pml_scene(geometry), device="cpu")
    stats = GEOMETRY_MEMO.stats
    # A Box snapshot is 10 float32 values (position, size, rotation): 40 bytes.
    assert stats["bytes"] <= 40 * 10 and stats["entries"] == 10 and stats["evictions"] >= 20
    before = GEOMETRY_MEMO.stats["hits"]
    prepare_scene(_pml_scene(geometries[-1]), device="cpu")
    prepare_scene(_pml_scene(geometries[0]), device="cpu")
    assert GEOMETRY_MEMO.stats["hits"] == before + 1


def test_oversize_mesh_is_not_memoized_and_matches_uncached(monkeypatch):
    box = mw.Box(position=(0., 0., -.2), size=(.64, .1, .06), device="cpu")
    vertices, faces = box.to_mesh()
    with monkeypatch.context() as patch:
        patch.setattr(static_cache, "GEOMETRY_ENTRIES", 0)
        expected = prepare_scene(_pml_scene(mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")), device="cpu")
    monkeypatch.setattr(static_cache, "GEOMETRY_ENTRY_BYTES", 64)
    mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
    before = GEOMETRY_MEMO.stats
    actual = [prepare_scene(_pml_scene(mesh), device="cpu") for _ in range(2)]
    stats = GEOMETRY_MEMO.stats
    assert stats["not_memoized"] - before["not_memoized"] == 2 and stats["hits"] == before["hits"]
    assert stats["entries"] == before["entries"]
    from witwin.maxwell.scene import _pml_contact_faces

    reference = _pml_contact_faces(expected, expected.scene.structures[0].geometry)
    for prepared in actual:
        assert prepared.pml_extrusions == expected.pml_extrusions
        census = _pml_contact_faces(prepared, mesh)
        assert [(a, s) for a, s, *_ in census[0]] == [(a, s) for a, s, *_ in reference[0]]
        assert [(a, s) for a, s, *_ in census[1]] == [(a, s) for a, s, *_ in reference[1]]


class _FluxDesign(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.density = torch.nn.Parameter(torch.full((3, 3, 2), .6, device="cuda"))
        self._base = mw.Scene(
            domain=mw.Domain(bounds=((-0.36, 0.36), (-0.36, 0.36), (-0.48, 0.48))), grid=mw.GridSpec.uniform(0.06),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
            structures=(
                mw.Structure(name="slab", geometry=mw.Box(position=(0., 0., .3), size=(.5, .5, .1), device="cuda"),
                             material=mw.Material(eps_r=2.2, sigma_e=.01)),
                mw.Structure(name="guide", geometry=mw.Box(position=(0., 0., -.3), size=(.12, .12, 1.), device="cuda"),
                             material=mw.Material(eps_r=3.)),
            ),
            sources=(mw.PointDipole(position=(0., 0., -0.18), polarization="Ex", width=0.04,
                                    source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.8e9, amplitude=50.)),),
            monitors=(mw.FluxMonitor("flux_probe", axis="z", position=0.18, normal_direction="+"),))
        self._box = mw.Box(position=(0., 0., 0.), size=(.36, .36, .12), device="cuda")

    def to_scene(self):
        return self._base.updated(material_regions=(mw.MaterialRegion(
            name="design", geometry=self._box, density=self.density, eps_bounds=(1., 6.)),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_scenemodule_density_loop_runs_are_equal_on_cuda(monkeypatch, stats):
    """A SceneModule density loop: FDTD forwards on cached materials equal uncached ones."""
    def flux(model):
        simulation = mw.Simulation.fdtd(model, frequencies=[1e9],
                                        config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=64)))
        return simulation.run().monitor("flux_probe").values.detach().clone()

    model = _FluxDesign()
    with monkeypatch.context() as patch:
        patch.setattr(static_cache, "FRAME_BYTES", 0)
        expected = [flux(model) for _ in range(3)]
    assert all(torch.equal(item, expected[0]) for item in expected)
    actual = [flux(model) for _ in range(3)]
    assert all(torch.equal(item, expected[0]) for item in actual)
    assert stats()["hits"] > 0 and stats()["frames"] == 1
