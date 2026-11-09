"""Static foil/pole content identity and private evolving consumer histories."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.material_cache import _scene_key
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.checkpoint import graph_state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Device material snapshots require CUDA.")


def _simulation(cache=None, *, kind="debye", thickness=2e-6, conductivity=5.8e7,
                band=(.5e9, 2e9), mode_count=16, eps=2., coordinate=0., pole_scale=1.):
    poles = {
        "debye": dict(debye_poles=(mw.DebyePole(delta_eps=1.5*pole_scale, tau=1e-10),)),
        "drude": dict(drude_poles=(mw.DrudePole(plasma_frequency=2e9*pole_scale, gamma=.2e9),)),
        "lorentz": dict(lorentz_poles=(mw.LorentzPole(delta_eps=1.5*pole_scale,
            resonance_frequency=2.5e9, gamma=.3e9),)),
        "magnetic": dict(mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.5*pole_scale,
            resonance_frequency=2.5e9, gamma=.3e9),)),
    }
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),)*3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda:0", structures=[
            mw.Structure(name="substrate", geometry=mw.Box(position=(0., 0., 0.), size=(.04,)*3),
                material=mw.Material(eps_r=eps, sigma_e=.01, **poles[kind])),
            mw.Structure(name="foil", geometry=mw.Box(
                position=torch.tensor([coordinate, 0., 0.], dtype=torch.float64),
                size=torch.tensor([0., .02, .02], dtype=torch.float64)),
                material=mw.CopperFoil(thickness=thickness, conductivity=conductivity,
                    frequency_range=band, mode_count=mode_count))],
        sources=[mw.UniformCurrentSource(center=(-.01, 0., -.0025), size=(.005,)*3, polarization="Ez",
            source_time=mw.CW(frequency=1e9))])
    return mw.Simulation.fdtd(scene, frequency=1e9, material_cache=cache, full_field_dft=True,
        cuda_graph=False, run_time=mw.TimeConfig(time_steps=64, dt=5e-12),
        spectral_sampler=mw.SpectralSampler(window="none"))


@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz", "magnetic"])
def test_cached_foil_and_static_poles_have_private_histories_and_exact_forward_parity(kind):
    cache = mw.FDTDMaterialCache(max_entries=2)
    reference = _simulation(kind=kind).prepare()
    first = _simulation(cache, kind=kind).prepare()
    second = _simulation(cache, kind=kind).prepare()
    assert cache.stats["misses"] == cache.stats["hits"] == 1
    a, b = graph_state_tensors(first.solver), graph_state_tensors(second.solver)
    assert set(a) == set(b)
    for name in a:
        if a[name].numel():
            assert a[name].data_ptr() != b[name].data_ptr(), name
        a[name].fill_(.123)
    third = _simulation(cache, kind=kind).prepare()
    for name, value in graph_state_tensors(third.solver).items():
        torch.testing.assert_close(value, b[name], rtol=0, atol=0)
    expected, actual = reference.run(), second.run()
    for name in expected.fields:
        torch.testing.assert_close(actual.tensor(name), expected.tensor(name), rtol=0, atol=0)
    for name in ("surface_power", "frequencies"):
        torch.testing.assert_close(actual.conductor_losses["foil"][name],
                                   expected.conductor_losses["foil"][name], rtol=0, atol=0)
    assert actual.conductor_losses["foil"]["dissipated_energy"] == expected.conductor_losses["foil"]["dissipated_energy"]


def test_foil_cache_identity_covers_response_substrate_poles_and_support():
    key = lambda **kwargs: _scene_key(prepare_scene(_simulation(**kwargs).scene), 1e9)
    original = key()
    assert key() == original
    for changed in (dict(thickness=3e-6), dict(conductivity=4e7), dict(band=(.6e9, 2e9)),
                    dict(mode_count=32), dict(eps=3.), dict(pole_scale=1.2), dict(coordinate=.005)):
        assert key(**changed) != original
    for parameter in ("thickness", "conductivity"):
        value = torch.tensor(2e-6 if parameter == "thickness" else 5.8e7,
                             dtype=torch.float64, device="cuda:0", requires_grad=True)
        with pytest.raises(ValueError, match="trainable"):
            key(**{parameter: value})


def test_foil_cache_lru_and_budget_exclude_live_state():
    cache = mw.FDTDMaterialCache(max_entries=1)
    _simulation(cache).prepare()
    _simulation(cache, thickness=3e-6).prepare()
    assert cache.stats["evictions"] == 1
    _simulation(cache).prepare()
    assert cache.stats["misses"] == 3
    small = mw.FDTDMaterialCache(max_bytes=1)
    _simulation(small).prepare()
    assert small.stats["entries"] == 0 and small.stats["oversized"] == 1
