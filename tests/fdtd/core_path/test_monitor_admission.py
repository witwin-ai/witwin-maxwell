"""Every public monitor type runs on the core path (W7 row 16a): admission and core == existing path."""
import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.scenes import GATE_SCENES, MONITOR_SCENES, build_core_case
from witwin.maxwell.capability.keys import MONITOR_TYPES
from witwin.maxwell.fdtd.runtime import core_path


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **kw: pytest.fail("Host check initialized CUDA."))


def _monitors():
    """One instance of every public monitor type (the capability registry's list)."""
    from tests.fdfd.test_fdfd_monitor_refusal import MONITORS

    box = dict(position=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1))
    monitors = (*MONITORS, mw.monitors.PermittivityMonitor("requested", **box),
                mw.monitors.MediumMonitor("requested", **box))
    assert sorted(type(monitor).__name__ for monitor in monitors) == sorted(MONITOR_TYPES)
    return monitors


@pytest.mark.host_validation
@pytest.mark.parametrize("monitor", _monitors(), ids=lambda monitor: type(monitor).__name__)
def test_every_public_monitor_type_is_admitted(monitor):
    # There is no admission list or path decision left (W7 rows 16c/16d): a scene with any
    # public monitor type prepares on the core path.
    mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.05),
             boundary=mw.BoundarySpec.none(), monitors=(monitor,))
    assert not hasattr(core_path, "core_path_decision")


@pytest.mark.host_validation
@pytest.mark.parametrize("name", MONITOR_SCENES)
def test_monitor_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    assert build_core_case(name).scene.monitors


def _simulation(name, steps):
    from benchmark.core_path.gates import make_simulation

    return make_simulation(build_core_case(name), steps=steps, full_field_dft=True)


def _payload(result):
    from benchmark.core_path.gates import export_tensors, physics_tensors, result_tensors, tensors

    solver = result.solver
    payload = tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
    return payload | result_tensors(result) | physics_tensors(solver, result) | export_tensors(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core execution requires CUDA.")
@pytest.mark.parametrize("name", MONITOR_SCENES)
def test_monitor_scene_core_equals_existing_path(name):
    # Reference: the existing-runner run of the same scene, frozen in addendum w7-row16e (W7 row 16e).
    from tests.support import frozen_oracles
    reference = frozen_oracles.load(f"tests/fdtd/core_path/test_monitor_admission.py::{name}")
    result = _simulation(name, 40).prepare().run()
    assert result.solver._core_runtime is not None
    assert any(key.startswith("exports/") or key.startswith("monitors/") for key in reference)
    assert any(value.abs().max() > 0 for key, value in reference.items()
               if key.startswith("fields/E") and value.is_floating_point())
    frozen_oracles.assert_equal(_payload(result), reference, name)
