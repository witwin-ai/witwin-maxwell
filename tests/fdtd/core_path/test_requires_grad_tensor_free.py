"""The requires_grad walk skips declarations that cannot hold tensors, with unchanged answers."""
import dataclasses

import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.accel import dispatch


def _reference_walk(value, seen=None):
    # The walk before the tensor-free skip.
    if torch.is_tensor(value):
        return value.requires_grad
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        children = vars(value).values()
    else:
        return False
    return any(_reference_walk(child, seen) for child in children)


def _leaves(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def test_point_monitor_stores_only_python_scalars():
    monitor = mw.PointMonitor("p", torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64), fields=("Ex", "Ez"))
    for field in dataclasses.fields(mw.PointMonitor):
        assert all(isinstance(leaf, (str, float)) for leaf in _leaves(getattr(monitor, field.name))), field.name
    assert dispatch._TENSOR_FREE == (mw.PointMonitor,)


def _scene(trainable):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.05), device="cpu")
    size = torch.tensor([0.2, 0.2, 0.2], requires_grad=trainable)
    scene.add_structure(mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=size), mw.Material(eps_r=2.0)))
    for index in range(256):
        scene.add_monitor(mw.PointMonitor(f"p{index}", (0.01 * (index % 16), 0.0, 0.0), fields=("Ex", "Ey", "Ez")))
    return scene


def test_walk_answers_match_reference_and_skip_point_monitors(monkeypatch):
    for trainable in (False, True):
        scene = _scene(trainable)
        assert dispatch._walk_requires_grad(scene) is _reference_walk(scene) is trainable
    def visits(tensor_free):
        monkeypatch.setattr(dispatch, "_TENSOR_FREE", tensor_free)
        count = 0
        original = dispatch._walk_requires_grad.__wrapped__ if hasattr(dispatch._walk_requires_grad, "__wrapped__") \
            else dispatch._walk_requires_grad
        def counted(value, seen=None):
            nonlocal count
            count += 1
            return original(value, seen)
        monkeypatch.setattr(dispatch, "_walk_requires_grad", counted)
        dispatch._walk_requires_grad(_scene(False))
        monkeypatch.setattr(dispatch, "_walk_requires_grad", original)
        return count
    skipped, full = visits((mw.PointMonitor,)), visits(())
    # A monitor holds name, kind, a 3-float position and a 3-name field tuple:
    # at least 10 visits each without the skip, exactly one with it.
    assert full - skipped >= 256 * 10
