"""CPU-only fingerprints of the gate scenes, pinned in ``gate_scene_fingerprints.json``.

The G1/G4 frozen store holds payloads captured on fixed scene definitions. A gate scene
must not move under it: a benchmark re-spec reaching a gate scene through the benchmark
registry (validation/bench-references moved ``full_tensor_slab``, ``mode_source_higher_order``,
``dipole_vacuum`` and ``dipole_ey``) leaves the gate comparing a different scene to the store.
``tests/validation/benchmark/test_gate_scene_fingerprints.py`` compares every gate scene with
the recorded fingerprint.

A fingerprint is a few sha256 digests per scene, one per part, so a mismatch names the part
that moved: ``grid`` (the prepared primal nodes of all three axes, absorber included),
``geometry`` (domain, grid and boundary specs, symmetry, subpixel and coordinates),
``structures``, ``sources``, ``monitors``, ``circuits`` (ports, networks, lumped elements,
circuits, terminals), ``scene_other`` (every remaining scene attribute) and ``case`` (the
CoreCase fields besides the scene: frequencies, steps, absorber, dtype, ...). Objects are
canonicalized structurally: class name and attributes (None-valued ones and integer ``*_id``
counters left out), floats as exact hex, tensors and arrays by dtype, shape and a digest of their bytes, callables by module
and qualified name.

Regenerate after an intended gate-scene change (which needs its own gate evidence):
``python -m benchmark.core_path.fingerprint --write``.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

FINGERPRINTS = Path(__file__).with_name("gate_scene_fingerprints.json")
_GEOMETRY = ("domain", "grid", "boundary", "symmetry", "subpixel", "coordinates")
_STRUCTURES = ("structures", "background", "material_regions", "thin_wires", "internal_absorbers")
_CIRCUITS = ("ports", "networks", "lumped_elements", "circuits", "electrostatic_terminals", "charge_densities")


def _counter(key: str, value) -> bool:
    """Process-global object counters (``material_id``, ``structure_id``, ...) depend on build order."""
    return key.endswith("_id") and type(value) is int


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value, _stack=None):
    """A JSON-serializable, process-independent structural description of ``value``."""
    stack = set() if _stack is None else _stack
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if isinstance(value, enum.Enum):
        return ["enum", type(value).__qualname__, value.name]
    if isinstance(value, (torch.dtype, torch.device)):
        return ["torch", str(value)]
    if isinstance(value, torch.Tensor):
        # A broadcast size-1 tensor keeps stride 0 through contiguous(), reshape and a plain clone.
        tensor = value.detach().to("cpu").reshape(-1).clone(memory_format=torch.contiguous_format)
        raw = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
        return ["tensor", str(value.dtype), list(value.shape), _digest(raw)]
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            return ["ndarray-object", list(array.shape), canonical(array.tolist(), stack)]
        return ["ndarray", array.dtype.str, list(array.shape), _digest(array.tobytes())]
    if isinstance(value, np.generic):
        return canonical(value.item(), stack)
    if isinstance(value, functools.partial):
        return ["partial", canonical(value.func, stack), canonical(value.args, stack),
                canonical(value.keywords, stack)]
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)):
        function = getattr(value, "__func__", value)
        return ["callable", getattr(function, "__module__", None), getattr(function, "__qualname__", repr(function))]
    marker = id(value)
    if marker in stack:
        return ["cycle", type(value).__qualname__]
    stack.add(marker)
    try:
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [canonical(item, stack) for item in value]]
        if isinstance(value, dict):
            items = [[canonical(key, stack), canonical(item, stack)] for key, item in value.items()]
            return ["dict", sorted(items, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
        if isinstance(value, (set, frozenset)):
            return ["set", sorted((canonical(item, stack) for item in value), key=json.dumps)]
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        # None-valued attributes (a new optional attribute left unset moves nothing) and integer
        # *_id counters (process-global, build-order dependent) are left out.
        if dataclasses.is_dataclass(value):
            attributes = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        else:
            attributes = dict(getattr(value, "__dict__", {}))
        for slot in () if dataclasses.is_dataclass(value) else getattr(type(value), "__slots__", ()):
            if hasattr(value, slot):
                attributes.setdefault(slot, getattr(value, slot))
        if attributes:
            return [name, {key: canonical(attributes[key], stack) for key in sorted(attributes)
                           if attributes[key] is not None and not _counter(key, attributes[key])}]
        return [name, repr(value)]
    finally:
        stack.discard(marker)


def _hash(value) -> str:
    return _digest(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode())[:24]


def case_fingerprint(case) -> dict[str, str]:
    """Per-part digests of one CoreCase (scene parts, prepared grid and the case fields)."""
    from witwin.maxwell.scene import prepare_scene

    scene = case.scene
    attributes = vars(scene)
    parts = {
        "geometry": _hash({key: attributes.get(key) for key in _GEOMETRY}),
        "structures": _hash({key: attributes.get(key) for key in _STRUCTURES}),
        "sources": _hash(attributes.get("sources")),
        "monitors": _hash(attributes.get("monitors")),
        "circuits": _hash({key: attributes.get(key) for key in _CIRCUITS}),
        "scene_other": _hash({key: value for key, value in attributes.items()
                              if key not in _GEOMETRY + _STRUCTURES + _CIRCUITS + ("sources", "monitors")}),
        "case": _hash({field.name: getattr(case, field.name) for field in dataclasses.fields(case)
                       if field.name != "scene" and getattr(case, field.name) is not None}),
    }
    prepared = prepare_scene(scene, device="cpu", lazy_meshgrid=True)
    nodes = [np.asarray(getattr(prepared, f"{axis}_nodes64"), dtype=np.float64) for axis in "xyz"]
    parts["grid"] = _hash(nodes)
    parts["grid_shape"] = "x".join(str(len(axis)) for axis in nodes)
    return dict(sorted(parts.items()))


def gate_fingerprints(names=None) -> dict[str, dict[str, str]]:
    from .scenes import GATE_SCENES, build_core_case

    return {name: case_fingerprint(build_core_case(name)) for name in (names or GATE_SCENES)}


def drift(recorded: dict, current: dict) -> dict[str, list[str]]:
    """Scene -> moved parts (``missing``/``new`` for scenes in only one of the two)."""
    moved = {}
    for name in sorted(set(recorded) | set(current)):
        if name not in current:
            moved[name] = ["missing"]
        elif name not in recorded:
            moved[name] = ["new"]
        else:
            parts = sorted(part for part in set(recorded[name]) | set(current[name])
                           if recorded[name].get(part) != current[name].get(part))
            if parts:
                moved[name] = parts
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"rewrite {FINGERPRINTS.name}")
    parser.add_argument("--output", type=Path, help="write the current fingerprints here instead")
    args = parser.parse_args()
    current = gate_fingerprints()
    if args.write or args.output:
        target = args.output or FINGERPRINTS
        target.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(current)} gate scenes to {target}")
        return
    moved = drift(json.loads(FINGERPRINTS.read_text()), current)
    print(json.dumps(moved, indent=1) if moved else f"{len(current)} gate scenes match {FINGERPRINTS.name}")
    raise SystemExit(1 if moved else 0)


if __name__ == "__main__":
    main()
