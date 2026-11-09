from unittest.mock import patch

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.material_cache import _clone_tree, _tree_bytes, _scene_key
from witwin.maxwell.scene import prepare_scene


def test_cache_bounds_and_snapshot_aliases():
    for kwargs in ({"max_bytes": 0}, {"max_entries": True}, {"max_entries": -1}):
        with pytest.raises(ValueError):
            mw.FDTDMaterialCache(**kwargs)
    x = torch.ones(4)
    original = {"a": x, "b": [x]}
    copied = _clone_tree(original)
    assert copied["a"] is copied["b"][0]
    copied["a"].zero_()
    assert torch.equal(x, torch.ones(4))
    assert _tree_bytes(copied) == 16
    with pytest.raises(ValueError, match="autograd"):
        _clone_tree(torch.ones(4, requires_grad=True))


def scene(eps=2., shift=0., dx=.1, pml=4, subpixel=1):
    # A small closed tetrahedron, with a fresh geometry object on every call.
    vertices = np.array([[0,0,.2],[.2,0,-.1],[-.1,.2,-.1],[-.1,-.2,-.1]], dtype=np.float64)
    vertices[:, 0] += shift
    faces = np.array([[0,2,1],[0,3,2],[0,1,3],[1,2,3]], dtype=np.int64)
    mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid", device="cuda")
    return mw.Scene(domain=mw.Domain(bounds=((-0.5,.5),)*3), grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=pml), device="cuda",
        subpixel_samples=subpixel, structures=[mw.Structure(mesh,mw.Material(eps_r=eps))],
        sources=[mw.PointDipole(position=(.3,0,0),polarization="Ez",source_time=mw.CW(frequency=1e8))],
        monitors=[mw.PointMonitor("probe",position=(.1,0,0),fields=("Ez",))])


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


@cuda
def test_cache_content_identity_invalidation_and_trainable_rejection():
    key = lambda value: _scene_key(prepare_scene(value), 1e8)
    original = scene()
    expected = key(original)
    assert key(scene()) == expected
    prepared = prepare_scene(scene())
    prepared.device = str(prepared.x.device)
    assert _scene_key(prepared, 1e8) == expected
    original.sources.clear()
    original.monitors.clear()
    assert key(original) == expected
    for changed in (scene(eps=3),scene(shift=.01),scene(dx=.125),scene(pml=5),scene(subpixel=2)):
        assert key(changed) != expected
    original.structures[0].geometry.position[0] += .01
    assert key(original) != expected
    original.structures[0].geometry.position.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        key(original)
    unsupported = scene()
    unsupported.structures[0] = mw.Structure(unsupported.structures[0].geometry,mw.Material(eps_r=2.,kerr_chi3=1e-20))
    with pytest.raises(NotImplementedError,match="kerr"):
        key(unsupported)
    mutated = scene()
    mutated.structures[0].geometry.vertices[0, 0] += .01
    assert key(mutated) != expected
    diagnosed = scene()
    geometry = diagnosed.structures[0].geometry
    from witwin.core.geometry.mesh import _tensor_revision
    geometry._topology_diagnostic_revision = (_tensor_revision(geometry.vertices), _tensor_revision(geometry.faces))
    geometry.vertices.add_(0.)
    with pytest.raises(RuntimeError, match="stale"):
        key(diagnosed)


@cuda
def test_cached_public_solves_match_and_materials_are_owned():
    cache = mw.FDTDMaterialCache()
    def simulation(cache=None):
        return mw.Simulation.fdtd(scene(), frequency=1e8,
                                  run_time=mw.TimeConfig(time_steps=400, transient_cycles=0, steady_cycles=1),
                                  spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
                                  material_cache=cache, full_field_dft=True)
    from witwin.maxwell import scene as scene_module
    compiler = scene_module.compile_material_model
    with patch.object(scene_module, "compile_material_model", wraps=compiler) as compile_call:
        uncached = simulation().prepare()
        first = simulation(cache).prepare()
        second = simulation(cache).prepare()
        assert compile_call.call_count == 2
        assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1
        assert first.solver._compiled_material_model["eps_r"].data_ptr() != second.solver._compiled_material_model["eps_r"].data_ptr()
        # Corrupt one consumer before it runs; the cache and next consumer must survive.
        first.solver._compiled_material_model["eps_r"].zero_()
        third = simulation(cache).prepare()
        assert torch.equal(third.solver._compiled_material_model["eps_r"], uncached.solver._compiled_material_model["eps_r"])
        reference = uncached.run()
        actual = second.run()
        assert reference.fields.keys() == actual.fields.keys()
        for name in reference.fields:
            assert torch.equal(reference.fields[name], actual.fields[name])
        assert actual.solver_stats["material_cache"]["outcome"] == "hit"
    cache.clear()
    assert cache.stats["entries"] == 0 and cache.stats["bytes"] == 0


@cuda
def test_cache_lru_and_oversized_not_retained():
    cache = mw.FDTDMaterialCache(max_entries=1)
    cache._bind(prepare_scene(scene()),frequency=1e8)
    cache._bind(prepare_scene(scene(eps=3)),frequency=1e8)
    assert cache.stats["evictions"] == 1
    cache._bind(prepare_scene(scene()),frequency=1e8)
    assert cache.stats["misses"] == 3
    small = mw.FDTDMaterialCache(max_bytes=1)
    info = small._bind(prepare_scene(scene()),frequency=1e8)
    assert info["outcome"] == "miss_oversized_not_retained"
    assert small.stats["entries"] == 0
