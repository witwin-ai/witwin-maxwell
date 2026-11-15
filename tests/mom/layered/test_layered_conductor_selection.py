"""conductor_model="auto" (A2): physics-only selection of the thick-metal model, refusals, metadata."""
from dataclasses import replace
import math

import pytest
import torch

import witwin.maxwell as mw
from tests.support.rfic_closed_references import strip_fixture
from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.runtime import THICK_T_OVER_DELTA, select_conductor_model
from witwin.maxwell.mom.layered.runtime import compile_layered_scene

UM = 1e-6
SIGMA = 5.8e7


def _compiled(**kwargs):
    return compile_layered_scene(fixture_fa(**kwargs).scene)


def _select(compiled, frequencies, *, sweep=None, gradients=False, planar=False):
    return select_conductor_model("auto", compiled, frequencies, sweep=sweep, gradients=gradients, planar=planar)


def _onset(thickness, sigma=SIGMA):
    """Frequency where t/delta = THICK_T_OVER_DELTA."""
    return (THICK_T_OVER_DELTA / thickness) ** 2 / (math.pi * 4e-7 * math.pi * sigma)


def test_default_is_auto_and_explicit_models_pass_through():
    config = mw.mom.LayeredSurfaceConfig(bridge_width_m=10 * UM, max_edge_length_m=10 * UM)
    assert config.conductor_model == "auto"
    assert config.payload()["conductor_model"] == "auto"
    # The sheet descriptor stays absent (frozen evidence configs).
    assert "conductor_model" not in replace(config, conductor_model="sheet").payload()
    compiled = _compiled()
    for model in ("sheet", "closed", "surface_admittance"):
        assert select_conductor_model(model, compiled, (1e9,), sweep=None, gradients=True, planar=True) == (model, None)


def test_rule_is_t_over_delta_at_the_highest_frequency():
    compiled = _compiled()
    onset = _onset(2 * UM)  # F-A top copper is 2 um: the thickest lossy sheet.
    model, record = _select(compiled, (1e9,))
    assert model == "surface_admittance"
    assert record["max_t_over_delta"] == pytest.approx(2 * UM * math.sqrt(math.pi * 1e9 * 4e-7 * math.pi * SIGMA), rel=1e-12)
    assert record["threshold_t_over_delta"] == THICK_T_OVER_DELTA == .3
    assert _select(compiled, (onset * 1.001,))[0] == "surface_admittance"
    assert _select(compiled, (onset * .999,))[0] == "sheet"
    # One sample above the onset selects the model for the whole request.
    assert _select(compiled, (1e6, onset * 1.001))[0] == "surface_admittance"
    sweep = mw.mom.AdaptiveSweep(f_min=1e6, f_max=onset * 1.001)
    assert _select(compiled, sweep.seed_frequencies()[:1], sweep=sweep)[1]["f_max_Hz"] == sweep.f_max


def test_pec_and_thin_metal_keep_sheet():
    assert _select(_compiled(lossy=False), (1e11,)) == ("sheet", {
        "requested": "auto", "selected": "sheet", "rule": "max t/delta at f_max >= threshold",
        "threshold_t_over_delta": .3, "f_max_Hz": 1e11, "max_t_over_delta": 0.0})
    thin = _compiled(top_thickness_um=.1)
    # The 1-um underpass is then the thickest lossy sheet.
    assert _select(thin, (_onset(1 * UM) * .999,))[0] == "sheet"
    assert _select(thin, (_onset(1 * UM) * 1.001,))[0] == "surface_admittance"


def test_auto_refuses_surface_admittance_features():
    """No fallback: a thick selection with a refused feature raises and names the feature."""
    compiled = _compiled()
    stack, rectangles, ports, materials, sheets, vias, via_materials = compiled
    vertical = mw.LumpedPort("v", positive=(130 * UM, 95 * UM, 305 * UM), negative=(130 * UM, 95 * UM, 0.),
                             voltage_path=mw.AxisPath("z"), reference_impedance=50.)
    rough = (SIGMA, 2 * UM, ("hammerstad", 1e-6))
    cases = {"polygon_layout": (compiled, dict(planar=True)),
             "zero_thickness_sheet": ((stack, rectangles, ports, [*materials[:-1], (None, 0., None)], sheets, vias,
                                       via_materials), {}),
             "surface_roughness": ((stack, rectangles, ports, [rough, *materials[1:]], sheets, vias, via_materials), {}),
             "vertical_port": ((stack, rectangles, (*ports, vertical), materials, sheets, vias, via_materials), {}),
             "gradients": (compiled, dict(gradients=True))}
    for feature, (records, flags) in cases.items():
        with pytest.raises(NotImplementedError, match=f"selects 'surface_admittance'.*{feature}.*conductor_model='sheet'"):
            _select(records, (1e9,), **flags)
        # Below the onset the same declarations select sheet.
        assert _select(records, (1e6,), **flags)[0] == "sheet"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_runtime_equals_explicit_surface_admittance():
    scene, config = strip_fixture(model="surface_admittance")
    auto = replace(config, layered=replace(config.layered, conductor_model="auto"))
    explicit = mw.Simulation.mom(scene, frequencies=[1e9], config=config).run()
    selected = mw.Simulation.mom(scene, frequencies=[1e9], config=auto).run()
    assert torch.equal(explicit.network.s, selected.network.s)
    assert torch.equal(mw.mom.surface_currents(explicit).currents, mw.mom.surface_currents(selected).currents)
    for metadata in (selected.network.metadata, selected.metadata):
        assert metadata["conductor_model"] == "surface_admittance"
        assert metadata["conductor_model_selection"]["selected"] == "surface_admittance"
    assert "conductor_model_selection" not in explicit.network.metadata
    low = mw.Simulation.mom(scene, frequencies=[1e6], config=auto).run()
    assert low.network.metadata["conductor_model"] == "sheet"
    sheet = replace(config, layered=replace(config.layered, conductor_model="sheet"))
    assert torch.equal(low.network.s, mw.Simulation.mom(scene, frequencies=[1e6], config=sheet).run().network.s)
