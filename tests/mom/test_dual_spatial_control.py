"""The spatial control refines geometry without changing physical inputs."""
from dataclasses import fields, replace
import pytest
import torch
from scripts.dev.solve_mom_dual_harmonic_power_control import setup
from scripts.dev import solve_mom_dual_harmonic_spatial_control as spatial
from scripts.dev.solve_mom_dual_harmonic_spatial_control import refined_setup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_spatial_control_preserves_physics_and_conforming_boundary(monkeypatch):
    coarse_scene, coarse = setup()
    monkeypatch.setattr(spatial, 'setup', lambda: (coarse_scene, coarse))
    fine_scene, fine = refined_setup(1)
    a, b = coarse.harmonic_balance.volumes[0], fine.harmonic_balance.volumes[0]
    assert len(b.cells) == 8*len(a.cells) == 32
    assert len(b.boundary.faces) == 4*len(a.boundary.faces) == 16
    torch.testing.assert_close(b.volumes.sum(), a.volumes.sum(), rtol=1e-14, atol=0)
    for item in fields(coarse.harmonic_balance):
        if item.name != 'volumes':
            assert getattr(coarse.harmonic_balance, item.name) == getattr(fine.harmonic_balance, item.name)
    for item in fields(coarse):
        if item.name != 'harmonic_balance':
            assert getattr(coarse, item.name) == getattr(fine, item.name)
    assert coarse_scene.structures[0].material == fine_scene.structures[0].material
    assert coarse_scene.sources == fine_scene.sources
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integers'):
            refined_setup(invalid)

    for name in ('source_max_order', 'source_max_inner_order'):
        for invalid in (2, 0, True, 3.5):
            with pytest.raises(ValueError, match=name):
                replace(coarse.harmonic_balance, **{name: invalid})
        assert getattr(replace(coarse.harmonic_balance, **{name: 64}), name) == 64
