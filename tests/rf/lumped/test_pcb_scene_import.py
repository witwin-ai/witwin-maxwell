import copy
import json
from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from witwin.maxwell.scene import prepare_scene


FIXTURES = Path(__file__).resolve().parents[3] / "docs/assessments/thin-copper-2026-10-22/site-scenes"
if not FIXTURES.exists():
    pytest.skip("repository evidence not in this checkout: docs/assessments/thin-copper-2026-10-22/site-scenes", allow_module_level=True)
NAMES = ("metamaterial-solid", "metamaterial-hollow", "metamaterial-open-top", "pcb-flexible", "pcb-rigid")


def definition(name):
    return json.loads((FIXTURES / f"metahz-scene-{name}.json").read_text())


@pytest.mark.parametrize('change', [{'type': 'wave'}, {'deembed_m': 1e-3}])
def test_import_rejects_unimplemented_port_contract_without_rewriting_it(change):
    data = definition('pcb-flexible')
    assert import_pcb_scene(data, frequency=2.45e9, device='cpu').ports
    data['ports'][0].update(change)
    original = copy.deepcopy(data)
    with pytest.raises(NotImplementedError, match='zero-deembedded lumped port'):
        import_pcb_scene(data, frequency=2.45e9, device='cpu')
    assert data == original


@pytest.mark.parametrize("name", NAMES)
def test_import_keeps_exact_geometry_selection_loss_and_finite_aperture(name):
    data = definition(name)
    snapshot = copy.deepcopy(data)
    scene = import_pcb_scene(data, frequency=2.45e9, device="cpu")
    assert data == snapshot
    imported = scene.metadata["pcb_import"]
    expected = [c["id"] for c in data["conductors"] if c["role"] == "fixed" or c["id"] in data["initial_conductor_ids"]]
    assert imported["active_conductor_ids"] == expected
    assert scene.sources == []
    assert all(p.voltage_aperture is not None for p in scene.ports)
    compiled = prepare_scene(scene).compile_ports()
    assert all(p.voltage_weights.numel() > 1 for p in compiled)
    assert any(float(s.material.sigma_e) > 0 for s in scene.structures)
    assert len(imported["definition_sha256"]) == 64
    # An approximate tangent-loss material must not silently become broadband.
    with pytest.raises(ValueError, match="reference frequency"):
        mw.Simulation.fdtd(scene, frequencies=(2.45e9, 2.5e9))


def test_import_rejects_fake_copper_selection_invalid_material_and_missing_terminals():
    data = definition("pcb-flexible")
    with pytest.raises(ValueError, match="existing IDs"):
        import_pcb_scene(data, frequency=2.45e9, selected_conductors=["imaginary"])
    data["materials"][0]["loss"]["kind"] = "unknown"
    with pytest.raises(NotImplementedError, match="loss"):
        import_pcb_scene(data, frequency=2.45e9)
    data = definition("pcb-flexible")
    data["ports"][0]["positive_conductor_ids"] = ["not-connected"]
    with pytest.raises(ValueError, match="active conductors"):
        import_pcb_scene(data, frequency=2.45e9)


def test_import_preserves_explicit_material_quadrature_without_changing_geometry():
    data = definition('pcb-flexible')
    ordinary = import_pcb_scene(data, frequency=2.45e9, device='cpu')
    data['mesh']['subpixel_samples'] = {'samples': [1, 1, 256], 'averaging': 'polarized'}
    scene = import_pcb_scene(data, frequency=2.45e9, device='cpu')
    default = mw.SubpixelSpec()
    assert ordinary.subpixel == default
    assert ordinary.metadata['pcb_import']['material_quadrature'] == {
        'samples': list(default.samples), 'averaging': default.averaging, 'pec': default.pec}
    assert scene.subpixel_samples == (1, 1, 256)
    assert scene.subpixel.averaging == 'polarized'
    assert scene.metadata['pcb_import']['material_quadrature']['samples'] == [1, 1, 256]
    for actual, expected in zip(scene.structures, ordinary.structures):
        torch.testing.assert_close(actual.geometry.position, expected.geometry.position)
        torch.testing.assert_close(actual.geometry.size, expected.geometry.size)
    for invalid in ({'samples': [1, 0, 4]}, {'unknown': True}, [1, 1, 4]):
        data['mesh']['subpixel_samples'] = invalid
        with pytest.raises((TypeError, ValueError)):
            import_pcb_scene(data, frequency=2.45e9, device='cpu')


def foil_definition():
    data = definition("pcb-flexible")
    for record in data["conductors"]:
        if record["role"] == "fixed":
            record["kind"] = "sheet"
            record["bounds_m"][4] = record["bounds_m"][5]
            record["constitutive_model"] = {"kind": "copper_foil", "thickness_m": 18e-6,
                "conductivity_s_per_m": 5.8e7, "frequency_range_hz": [1e9, 4e9]}
    return data


def test_explicit_current_loop_preserves_its_physical_plane_under_refinement():
    data = foil_definition()
    original = import_pcb_scene(data, frequency=2.45e9, device="cpu")
    surface = original.ports[0].current_surface
    bounds = torch.stack((surface.position - surface.size/2, surface.position + surface.size/2), dim=-1)
    data["ports"][0]["current_contour_bounds_m"] = bounds.reshape(-1).tolist()
    old = data["mesh"]["axes_m"]["y"]
    refined = [old[0]]
    for low, high in zip(old, old[1:]):
        refined.extend((low+(high-low)/3, low+2*(high-low)/3, high))
    data["mesh"]["axes_m"]["y"] = refined
    imported = import_pcb_scene(data, frequency=2.45e9, device="cpu")
    # Bounds -> center/size introduces at most a few float64 arithmetic ULPs.
    torch.testing.assert_close(imported.ports[0].current_surface.position, surface.position, rtol=8e-16, atol=0)
    torch.testing.assert_close(imported.ports[0].current_surface.size, surface.size, rtol=8e-16, atol=0)
    assert any(index.numel() > 0 for index in prepare_scene(imported).compile_ports()[0].current_indices)
    data["ports"][0]["current_contour_bounds_m"][2] = float("nan")
    with pytest.raises(ValueError, match="finite SI bounds"):
        import_pcb_scene(data, frequency=2.45e9, device="cpu")


def test_explicit_foil_import_preserves_physical_constitutive_contract():
    data = foil_definition()
    scene = import_pcb_scene(data, frequency=2.45e9, device="cpu")
    foils = [s for s in scene.structures if isinstance(s.material, mw.CopperFoil)]
    assert len(foils) == 2
    assert all(float(s.geometry.size[2]) == 0 and s.material.thickness == 18e-6 for s in foils)
    assert scene.metadata["pcb_import"]["conductor_models"]["positive-pad"] == data["conductors"][1]["constitutive_model"]
    with pytest.raises(ValueError, match="frequency_range"):
        import_pcb_scene(data, frequency=6e9)
    with pytest.raises(ValueError, match="explicitly declared copper loss"):
        import_pcb_scene(data, frequency=2.45e9, losses=False)
    data["conductors"][0]["kind"] = "solid"
    with pytest.raises(ValueError, match="collapse physical layers"):
        import_pcb_scene(data, frequency=2.45e9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
@pytest.mark.parametrize("quadrature", [False, True])
def test_explicit_imported_foil_executes_native_lossy_aperture_feed(quadrature):
    data = foil_definition()
    if quadrature:
        data["mesh"]["subpixel_samples"] = {"samples": [1, 1, 8], "averaging": "polarized"}
    scene = import_pcb_scene(data, frequency=2.45e9)
    result = mw.Simulation.fdtd(scene, frequency=2.45e9,
        excitations=mw.PortExcitation("P1", source_time=mw.CW(frequency=2.45e9)),
        run_time=mw.TimeConfig(time_steps=256, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window="none")).run()
    assert float(result.port("P1").voltage.abs().max()) > 0
    assert set(result.conductor_losses) == {"positive-pad", "negative-pad"}
    assert all(float(row["dissipated_energy"]) > 0 for row in result.conductor_losses.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
@pytest.mark.parametrize("name", NAMES)
def test_every_imported_scene_runs_with_declared_loss_and_real_aperture(name):
    scene = import_pcb_scene(definition(name), frequency=2.45e9)
    prepared = mw.Simulation.fdtd(
        scene, frequency=2.45e9,
        excitations=mw.PortExcitation(scene.ports[0].name, source_time=mw.CW(frequency=2.45e9)),
        run_time=mw.TimeConfig(time_steps=256, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare()
    assert prepared.diagnostics["conductive"]
    result = prepared.run()
    port = result.port(scene.ports[0].name)
    assert bool(torch.isfinite(port.voltage).all())
    assert bool(torch.isfinite(port.current).all())
    assert float(port.voltage.abs().max()) > 0
    assert float(port.current.abs().max()) > 0
