from __future__ import annotations

from types import SimpleNamespace

import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import prepare_port_runtimes


def test_scene_without_rf_objects_creates_no_port_or_circuit_runtime():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    solver = SimpleNamespace(scene=scene, device=torch.device("cpu"))

    runtimes = prepare_port_runtimes(solver, (1.0e9,))

    assert runtimes == ()
    assert solver._port_runtimes == ()
    assert solver._lumped_element_runtimes == ()
