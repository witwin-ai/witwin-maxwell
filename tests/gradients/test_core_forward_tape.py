"""Differentiable FDTD forward on the core path: forward tape and gradients vs frozen oracles.

The frozen payloads (store keys ``gates/grad/<scene>/full.pt``, addenda ``w7-row14`` and ``w7-row16b-grad``) are the
existing-path gradient forward of each scene: the adjoint checkpoints at the checkpoint stride,
the bridge's forward output tensors, the loss and the gradients of every trainable input.
"""
import pytest
import torch

from benchmark.core_path.grad_gates import GRADIENT_SCENES, capture_scene, compare_payloads, frozen_payload

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")]

# Configurations that keep stepping's per-step loop (with the fused Yee pair) instead of K-step graphs.
PER_STEP = {"circuit_port", "time_monitor", "eager_graphs_off"}


@pytest.mark.parametrize("name", sorted(GRADIENT_SCENES))
def test_gradient_forward_tape_matches_existing_path(name):
    payload, meta = capture_scene(name, GRADIENT_SCENES[name][1])
    assert meta["execution_path"] == "core"
    assert meta["core_step_graph_active"] is (name not in PER_STEP)
    # The core forward is the ordinary solve, so closed-transverse CPML samples the passivity guard.
    assert (meta["passivity_guard_status"] == "active") is (name == "guarded_cpml")
    report = compare_payloads(frozen_payload(name, "full"), payload)
    assert report["unequal"] == [], report["unequal"][:10]


def test_failed_core_forward_raises(monkeypatch):
    # As in the ordinary solve (W7 row 16c): a failed core forward raises its own error.
    from witwin.maxwell.fdtd.runtime import core_path

    def fail(*args, **kwargs):
        raise RuntimeError("forced capture failure")

    with monkeypatch.context() as patch:
        patch.setattr(core_path.CorePathRuntime, "run_chunks", fail)
        with pytest.raises(RuntimeError, match="^forced capture failure$"):
            capture_scene("pml_module_eps", GRADIENT_SCENES["pml_module_eps"][1])
    payload, meta = capture_scene("pml_module_eps", GRADIENT_SCENES["pml_module_eps"][1])
    assert meta["execution_path"] == "core"
    assert compare_payloads(frozen_payload("pml_module_eps", "full"), payload)["unequal"] == []
