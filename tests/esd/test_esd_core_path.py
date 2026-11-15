"""ESD current sources on the CUDA core path (W7 row 16c).

``ESDCurrentSource`` sits in ``scene.sources`` and lowers to a ``UniformCurrentSource``
through ``Scene.resolved_sources()``. The former ``"unsupported source type"`` blocker read
the unresolved list and kept these scenes on the existing stepping path; with the blocker
gone they run on the core path. The core run equals the existing-runner reference bitwise:
the reference is the frozen addendum ``w7-row16e`` captured on master a462f31ba before the
existing runner was deleted (W7 row 16e).
"""
from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.gates import result_tensors, tensors
from tests.esd import test_prebias
from tests.support import frozen_oracles

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")]

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _injection_scene():
    def pec(name, position):
        return mw.Structure(geometry=mw.Box(position=position, size=(0.10, 0.10, 0.04)),
                            material=mw.Material.pec(), name=name)

    return mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3),) * 3), grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        structures=(pec("signal", (0.0, 0.0, 0.06)), pec("ground", (0.0, 0.0, -0.06))),
        ports=(mw.rf.TerminalPort("feed", mw.rf.TerminalRef("signal"), mw.rf.TerminalRef("ground"),
                                  mw.AxisPath("z"), 0.01),),
        sources=(mw.esd.ESDCurrentSource("gun", port="feed", waveform=mw.esd.ESDWaveform.iec_61000_4_2(8000.0)),),
        monitors=(mw.monitors.FieldTimeMonitor("vprobe", components=("Ez",), position=(0.0, 0.0, 0.0), interval=1),),
    )


def _run(scene, **config):
    config = mw.FDTDConfig(spectral_sampler=mw.fdtd.SpectralSampler(window="none"), full_field_dft=False, **config)
    result = mw.Simulation.fdtd(scene, frequencies=[1e9], config=config).run()
    assert result.solver._core_runtime is not None
    payload = tensors({name: getattr(result.solver, name) for name in FIELDS}, "fields")
    payload.update(result_tensors(result))
    return payload


@pytest.mark.parametrize("case", ["injection", "prebias_breakdown"])
def test_esd_current_source_runs_core_bitwise_equal_to_the_existing_runner(case):
    def build():
        if case == "injection":
            return _injection_scene(), dict(run_time=mw.TimeConfig(time_steps=240))
        return test_prebias._fdtd_scene(), dict(run_time=mw.TimeConfig(time_steps=120),
                                                initial_condition=test_prebias._prebias())

    reference = frozen_oracles.load(f"tests/esd/test_esd_core_path.py::{case}")
    scene, config = build()
    actual = _run(scene, **config)
    assert any(torch.count_nonzero(value) for key, value in reference.items() if key.startswith("fields/"))
    frozen_oracles.assert_equal(actual, reference, case)
