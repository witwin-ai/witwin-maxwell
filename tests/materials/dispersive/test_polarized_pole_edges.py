"""Polarized-family subpixel averaging of pole media: interface pole weights follow the static window rule."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import compile_material_model
from witwin.maxwell.scene import prepare_scene


_SAMPLES = {"polarized": 3, "polarized_tensor": 3, "kottke_smooth": 4}


def _compile(material, averaging="polarized", samples=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4),) * 3), grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pec(), subpixel=mw.SubpixelSpec(samples=samples or _SAMPLES[averaging], averaging=averaging),
        structures=[mw.Structure(geometry=mw.Sphere(radius=0.17), material=material)],
    )
    return compile_material_model(prepare_scene(scene, device="cpu"))


_SELLMEIER = ("eps", mw.Material.sellmeier(eps_inf=1.5, b_coefficients=(1.5,), c_coefficients=(2.5e-3,)), 1.5, 3.0)
_MAGNETIC = ("mu", mw.Material(mu_r=1.5, mu_lorentz_poles=(mw.LorentzPole(1.0, 3.5e9, 2.5e8),)), 1.5, 2.5)


@pytest.mark.parametrize("averaging,channel,material,eps_inf,static", [
    ("polarized", *_SELLMEIER),
    ("polarized", "eps", mw.Material(eps_r=2.0, debye_poles=(mw.DebyePole(1.0, 1e-10), mw.DebyePole(0.5, 3e-10))), 2.0, 3.5),
    ("polarized", "eps", mw.Material(eps_r=4.0, lorentz_poles=(mw.LorentzPole(1.5, 3e9, 2e8),)), 4.0, 5.5),
    ("polarized", *_MAGNETIC),
    ("polarized_tensor", *_SELLMEIER),
    ("polarized_tensor", *_MAGNETIC),
    ("kottke_smooth", *_SELLMEIER),
    ("kottke_smooth", *_MAGNETIC),
])
def test_interface_edges_match_the_window_rule_at_both_frequency_limits(averaging, channel, material, eps_inf, static):
    model = _compile(material, averaging)
    high = _compile(mw.Material(**{channel + "_r": eps_inf}), averaging)
    low = _compile(mw.Material(**{channel + "_r": static}), averaging)
    prefix = "" if channel == "eps" else "mu_"
    components = ("Ex", "Ey", "Ez") if channel == "eps" else ("Hx", "Hy", "Hz")
    poles = [entry for kind in ("debye", "lorentz") for entry in model[prefix + kind + "_poles"]]
    for component in components:
        edge = model["edge_components"][channel][component]
        assert torch.equal(edge, high["edge_components"][channel][component])
        realized_static = edge + sum(entry["edge_weights"][component] * float(entry["pole"].delta_eps)
                                     for entry in poles)
        torch.testing.assert_close(realized_static, low["edge_components"][channel][component],
                                   rtol=0, atol=2e-6)
        for entry in poles:
            weight = entry["edge_weights"][component]
            inside, outside = edge == eps_inf, edge == 1.0
            assert torch.equal(weight[inside], torch.ones_like(weight[inside]))
            assert torch.equal(weight[outside], torch.zeros_like(weight[outside]))


@pytest.mark.parametrize("averaging", ["polarized", "kottke_smooth"])
def test_drude_weights_are_the_edge_volume_fraction(averaging):
    # A Drude pole has no finite static limit; it keeps the arithmetic window fraction.
    drude = _compile(mw.Material(eps_r=2.0, drude_poles=(mw.DrudePole(2e9, 1e8),)), averaging)
    unit = _compile(mw.Material(eps_r=3.0), "arithmetic", _SAMPLES[averaging])["edge_components"]["eps"]
    plain = _compile(mw.Material(eps_r=2.0), "arithmetic", _SAMPLES[averaging])["edge_components"]["eps"]
    for component in ("Ex", "Ey", "Ez"):
        weight = drude["drude_poles"][0]["edge_weights"][component]
        torch.testing.assert_close(weight, (unit[component] - plain[component]).clamp(0, 1), rtol=0, atol=1e-5)
        assert int(((weight > 0) & (weight < 1)).sum()) > 0
