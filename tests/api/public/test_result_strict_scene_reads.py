"""Result reads its scene strictly: a wrong scene object fails loudly, a scene-less Result keeps working.

``Result.scene`` is the declarative ``Scene`` of the run, or ``None`` for a scene-less Result (for example
``Result.load(path, scene=None)``). Any other object, such as the ``PreparedScene`` that p7b left in wave-port
results, must raise instead of silently reading as "no coordinates", "no monitors", "no sources" or "no circuits".
"""
from types import SimpleNamespace

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


FREQUENCY = 1.0e9


def _scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.25),
    )


def _probe_payload():
    return {
        "kind": "point",
        "frequencies": (FREQUENCY,),
        "data": torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex128),
    }


def _result(scene):
    return mw.Result(method="fdtd", scene=scene, frequency=FREQUENCY, monitors={"probe": _probe_payload()})


def _wrong_scenes():
    return {
        "prepared_scene": prepare_scene(_scene(), device="cpu"),
        "plain_object": SimpleNamespace(),
    }


@pytest.mark.parametrize("kind", ("prepared_scene", "plain_object"))
def test_construction_and_load_with_a_wrong_scene_object_raise(kind, tmp_path):
    wrong = _wrong_scenes()[kind]
    with pytest.raises(AttributeError, match="coordinates"):
        _result(wrong)
    path = tmp_path / "result.pt"
    _result(_scene()).save(path)
    with pytest.raises(AttributeError, match="coordinates"):
        mw.Result.load(path, scene=wrong)


@pytest.mark.parametrize("kind", ("prepared_scene", "plain_object"))
@pytest.mark.parametrize(
    ("read", "field"),
    (
        (lambda result: result.raw_monitor("probe"), "monitors"),
        (lambda result: result.material("eps_r"), "coordinates"),
        (lambda result: mw.esd.esd_waveform_names(result), "sources"),
        (lambda result: mw.esd.esd_generator_names(result), "circuits"),
        (lambda result: mw.esd.component_stress_names(result), "resolved_monitors"),
    ),
    ids=("monitor_lookup", "coordinates", "esd_sources", "esd_circuits", "component_stress"),
)
def test_reads_through_a_wrong_scene_object_raise(kind, read, field):
    result = _result(_scene())
    # A Result built with a valid Scene whose scene was later replaced: the read paths must not default.
    result.scene = _wrong_scenes()[kind]
    with pytest.raises(AttributeError, match=field):
        read(result)


def _assert_scene_less_reads(result):
    payload = result.raw_monitor("probe")
    expected = _probe_payload()
    assert set(payload) == set(expected)
    assert payload["kind"] == "point"
    assert payload["frequencies"] == expected["frequencies"]
    assert torch.equal(payload["data"], expected["data"])
    assert "native_product_disabled" not in payload
    assert "field_frame" not in result.metadata
    assert mw.esd.esd_waveform_names(result) == ()
    assert mw.esd.esd_generator_names(result) == ()
    assert mw.esd.component_stress_names(result) == ()
    with pytest.raises(KeyError, match="ComponentStressMonitor 'probe' is not present"):
        mw.esd.component_stress(result, "probe")


def test_scene_less_result_constructs_loads_and_reads_its_stored_data(tmp_path):
    result = _result(None)
    assert result.scene is None
    _assert_scene_less_reads(result)
    path = tmp_path / "scene_less.pt"
    result.save(path)
    loaded = mw.Result.load(path, scene=None)
    assert loaded.scene is None
    assert loaded.frequencies == (FREQUENCY,)
    _assert_scene_less_reads(loaded)
