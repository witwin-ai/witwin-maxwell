"""Test-only API dialect for the BITID probes during the 0.7.0 API breaks.

The BITID files (``pin_files`` in ``scripts/dev/integration_suites.toml``) run one probe
script in fresh interpreters on frozen baseline trees (0.6 API) and on the working tree.
As 0.7.0 breaks the public API phase by phase, one script can no longer drive both trees
directly, so the probes build scenes and simulations and read results only through this
module. It detects which breaks the imported ``witwin.maxwell`` has (``BREAKS``) and
translates the probes' 0.6-style calls for that tree. On a 0.6 tree every operation is a
pure pass-through.

Each 0.7.0 phase that breaks something used here adds its marker to ``_HANDLED`` and one
branch in the affected operations, in the same branch that makes the break. The expected
hook points are marked ``HOOK <phase>`` below.

Not wrapped, because no 0.7.0 phase changes them: ``prepare()``, ``run()``,
``run_until()``, resume checkpoints, ``Simulation.batch`` and solver internals reached
through ``prepared.solver``.

Import from a probe with ``sys.path.insert(0, "<this directory>")`` then
``import api_dialect``. The editable install puts the main checkout's ``tests`` package
on ``sys.path``, so ``tests.support`` would not resolve to the probe's own tree in a
fresh interpreter. This directory holds no ``witwin`` package, so the insert cannot
change which tree ``witwin.maxwell`` comes from.

This module is test support only: ``witwin/`` must never import it. It is deleted after
the 0.7.0 re-pin of the BITID baselines (branch api070/release-notes).
"""
from __future__ import annotations

import dataclasses
import inspect
import weakref

import witwin.maxwell as _mw


def _detect_breaks() -> frozenset[str]:
    """The 0.7.0 API breaks present in the imported tree, each detected from the break itself."""
    found = set()
    # api070/p4b-cpml-config: FDTDConfig.cpml_config (dict) became FDTDConfig.cpml (CPMLConfig).
    if "cpml_config" not in {field.name for field in dataclasses.fields(_mw.FDTDConfig)}:
        found.add("p4b")
    # api070/p4c-single-entry: Simulation.fdtd keeps only frequencies, excitations and config=.
    if "run_time" not in inspect.signature(_mw.Simulation.fdtd).parameters:
        found.add("p4c")
    # api070/p5b-namespaces: moved symbols left the top level for their namespaces.
    if not hasattr(_mw, "PortData"):
        found.add("p5b")
    # api070/p6b-fdtd-typed-monitors: FDTD monitors and fields became typed data; the
    # last-step fields of a single-frequency run moved to Result.time_snapshot.
    if hasattr(_mw.Result, "time_snapshot"):
        found.add("p6b")
    # api070/p6e-result-slim: Result is slimmed (Result.select is gone, domain accessors moved to
    # namespace functions taking the Result, solver_stats merges into stats()).
    if not hasattr(_mw.Result, "select"):
        found.add("p6e")
    # HOOK p7a (api070/p7a-scene-frozen): Scene.add_* are removed.
    if not hasattr(_mw.Scene, "add_structure"):
        found.add("p7a")
    # api070/p7b-compiler-internal: the Scene/PreparedScene compile methods became functions of
    # witwin.maxwell.compiler taking the prepared scene.
    if not hasattr(_mw.Scene, "compile_ports"):
        found.add("p7b")
    return frozenset(found)


BREAKS = _detect_breaks()
_HANDLED: frozenset[str] = frozenset({"p4b", "p4c", "p5b", "p6b", "p6e", "p7a", "p7b"})
if BREAKS - _HANDLED:
    raise RuntimeError(
        f"{_mw.__file__} has API breaks {sorted(BREAKS - _HANDLED)} that tests/support/api_dialect.py "
        "does not translate yet; add the branch in the phase that made the break"
    )


# --- Public symbols -------------------------------------------------------------------
# Probes write ``mw = api_dialect.mw`` and keep 0.6 symbol names such as ``mw.PortData``.
class _TopLevel06:
    """``witwin.maxwell`` as the 0.6 top level: a name the top level no longer has is read from
    the one namespace whose ``__all__`` lists it (api070/p5b-namespaces); everything else,
    ``__file__`` included, is the module's own attribute."""

    def __init__(self):
        self._homes = {}

        def collect(module):
            for name in getattr(module, "__all__", ()):
                value = getattr(module, name)
                if type(value) is type(_mw):
                    collect(value)
                else:
                    self._homes[name] = value

        for namespace in _mw._NAMESPACES:
            collect(getattr(_mw, namespace))

    def __getattr__(self, name):
        if hasattr(_mw, name):
            return getattr(_mw, name)
        try:
            return self._homes[name]
        except KeyError:
            raise AttributeError(f"witwin.maxwell has no 0.6 name {name!r}") from None


mw = _TopLevel06() if "p5b" in BREAKS else _mw


# --- Scene construction and updates ---------------------------------------------------
def scene(**fields):
    """``mw.Scene(**fields)`` with 0.6 keywords (``subpixel_samples=``, ``device=``, lists)."""
    if "p7a" in BREAKS:
        # api070/p7a: subpixel_samples= became subpixel=; device= left the Scene for the solver
        # config (Q18), whose default is the CUDA device the probes use.
        if "subpixel_samples" in fields:
            fields["subpixel"] = fields.pop("subpixel_samples")
        device = fields.pop("device", "cuda")
        built = mw.Scene(**fields)
        _SCENE_DEVICES[built] = str(device)
        return built
    return mw.Scene(**fields)


# api070/p7a: the 0.6 scene device, kept beside the device-free 0.7.0 Scene for prepare().
_SCENE_DEVICES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def prepare(scene, **options):
    """``prepare_scene(scene)`` on the device the probe gave ``scene(device=...)``."""
    from witwin.maxwell.scene import prepare_scene

    if "p7a" in BREAKS:
        options.setdefault("device", _SCENE_DEVICES.get(scene, "cuda"))
    return prepare_scene(scene, **options)


def add(scene, *, structures=(), sources=(), monitors=(), ports=(), lumped_elements=()):
    """Append to the scene collections, in this argument order, and return the scene to use."""
    if "p7a" in BREAKS:
        # api070/p7a: add_* are removed; updated() returns the new scene.
        groups = {"structures": structures, "sources": sources, "monitors": monitors, "ports": ports,
                  "lumped_elements": lumped_elements}
        for name, items in groups.items():
            if items:
                scene = scene.updated(**{name: (*getattr(scene, name), *items)})
        return scene
    for structure in structures:
        scene.add_structure(structure)
    for source in sources:
        scene.add_source(source)
    for monitor in monitors:
        scene.add_monitor(monitor)
    for port in ports:
        scene.add_port(port)
    for element in lumped_elements:
        scene.add_lumped_element(element)
    return scene


def updated(scene, **fields):
    """A copy of ``scene`` with whole collections replaced (0.6 ``Scene.clone``)."""
    if "p7a" in BREAKS:
        # api070/p7a: Scene.clone(**overrides) became Scene.updated(**fields).
        return scene.updated(**fields)
    return scene.clone(**fields)


# --- Simulation -----------------------------------------------------------------------
def fdtd(scene, **options):
    """``mw.Simulation.fdtd`` with the 0.6 keyword set (run_time=, absorber=, cpml_config=, ...)."""
    if "p4b" in BREAKS and "cpml_config" in options:
        cpml_config = options.pop("cpml_config")
        if cpml_config is not None:
            options["cpml"] = mw.CPMLConfig(**cpml_config)
    if "p4c" in BREAKS:
        # api070/p4c-single-entry: frequency= became frequencies=; excitations= stays and
        # every other option moved into config=mw.FDTDConfig(...).
        if "frequency" in options:
            options["frequencies"] = options.pop("frequency")
        entry = {name: options.pop(name) for name in ("frequencies", "excitations") if name in options}
        return mw.Simulation.fdtd(scene, **entry, config=mw.FDTDConfig(**options))
    # api070/p7a: the scene's device moved into the config, whose CUDA default the probes use.
    return mw.Simulation.fdtd(scene, **options)


# --- Result access --------------------------------------------------------------------
def fields(result):
    """``Result.fields``: the 0.6 component-name -> tensor mapping."""
    if "p6b" in BREAKS and result.method == "fdtd":
        # api070/p6b: Result.fields holds typed phasors, Result.time_snapshot the last-step fields (Q7);
        # both wrap the stored tensors, which keep their 0.6 shapes.
        typed = result.time_snapshot if result._fields_are_time_snapshot() else result.fields
        return {name: result._fields[name] for name in typed}
    return result.fields


def stored_fields(result):
    """The stored field arrays, including multi-frequency volumes the public accessor hides."""
    return result._fields


def legacy_payload(payload):
    """A raw monitor payload with the 0.6 single-component aliases (api070/p6b removed them)."""
    if not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    payload.pop("component_coords", None)
    kind = payload.get("kind")
    components = payload.get("components")
    if kind in ("field_time", "aux_field_time"):
        if len(components) == 1:
            payload["data"] = next(iter(components.values()))
            if kind == "field_time":
                payload["field"] = payload["data"]
    elif "component" in payload and len(payload.get("fields", ())) == 1:
        payload["data"] = payload[payload["fields"][0]]
    if kind == "closed_surface":
        payload["faces"] = {label: legacy_payload(face) for label, face in payload["faces"].items()}
    return payload


def monitors(result):
    """``Result.monitors``: monitor name -> 0.6 payload dict."""
    if "p6b" in BREAKS and result.method == "fdtd":
        # api070/p6b: Result.monitors maps to typed data; 0.6 returned the stored payloads.
        return {name: legacy_payload(payload) for name, payload in result._monitors.items()}
    return result.monitors


def monitor(result, name, **selection):
    """``Result.monitor(name, ...)`` as the 0.6 payload dict."""
    if "p6b" in BREAKS and result.method == "fdtd":
        return _monitor06(result, name, **selection)
    return result.monitor(name, **selection)


def _monitor06(result, name, **selection):
    # api070/p6b: the 0.6 Result.monitor resolution on top of the raw payload.
    from witwin.maxwell import result as _result

    public = _result._find_scene_monitor(result.scene, name)
    if isinstance(public, (_mw.monitors.PowerLossMonitor, *_result.PROJECTION_MONITOR_TYPES)) or isinstance(
            result._monitors.get(name), _mw.rf.WireData):
        return result.monitor(name, **selection)
    if isinstance(public, _result.GAUSSIAN_OVERLAP_MONITOR_TYPES):
        from witwin.maxwell.postprocess.gaussian_overlap import compute_gaussian_overlap
        return compute_gaussian_overlap(result, public, **selection)
    payload = raw_monitor(result, name, **selection)
    if _result._monitor_payload_is_closed_surface(payload):
        return payload
    if _result._monitor_payload_is_diffraction(payload):
        from witwin.maxwell.postprocess.diffraction import compute_diffraction_from_payload
        return compute_diffraction_from_payload(result, name, payload)
    if _result._monitor_payload_is_dipole_emission(payload):
        return {"kind": "dipole_emission", "monitor_type": "dipole_emission",
                "power_delivered": payload.get("power_delivered"), "current_spectrum": payload.get("current_spectrum"),
                "frequencies": tuple(payload.get("frequencies", (payload.get("frequency"),))),
                "frequency": payload.get("frequency"), "polarization": payload.get("dipole_polarization"),
                "position": payload.get("dipole_position", payload.get("position")),
                "source_name": payload.get("source_name"), "purcell_factor": None}
    if not _result._monitor_payload_is_mode(payload):
        return payload
    modal = result._mode_overlap(name, payload)
    modal.update(kind="mode", fields=tuple(payload.get("fields", ())),
                 frequencies=tuple(payload.get("frequencies", (payload.get("frequency"),))),
                 frequency=payload.get("frequency"), normal_direction=payload.get("normal_direction", "+"),
                 plane=payload)
    return modal


def raw_monitor(result, name, **selection):
    """``Result.raw_monitor(name)`` (kept by p6e)."""
    if "p6b" in BREAKS and result.method == "fdtd":
        return legacy_payload(result.raw_monitor(name, **selection))
    return result.raw_monitor(name, **selection)


def stored_monitor(result, name):
    """The stored monitor payload, before the public accessor's selection."""
    if "p6b" in BREAKS and result.method == "fdtd":
        return legacy_payload(result._monitors[name])
    return result._monitors[name]


def ports(result):
    """``Result.ports``: port name -> PortData."""
    return result.ports


def port(result, name):
    """``Result.port(name)``."""
    return result.port(name)


def network(result):
    """``Result.network`` (NetworkData or None)."""
    return result.network


def s_matrix(network):
    """``NetworkData.s`` as a tensor."""
    # HOOK p6 (Q12, the branch that types NetworkData.s): SParameterData -> .values.
    return network.s


def circuits(result):
    """``Result.circuits``."""
    if "p6e" in BREAKS:
        # api070/p6e: Result.circuits moved to mw.circuits.circuits(result).
        return _mw.circuits.circuits(result)
    return result.circuits


def embedded_networks(result):
    """``Result.embedded_networks``."""
    if "p6e" in BREAKS:
        # api070/p6e: Result.embedded_networks moved to mw.rf.embedded_networks(result).
        return _mw.rf.embedded_networks(result)
    return result.embedded_networks


def solver(result):
    """The solver kept on the result (``Result.solver``, kept by api070/p6e)."""
    return result.solver


def solver_stats(result):
    """``Result.solver_stats``: the solver's own statistics."""
    if "p6e" in BREAKS:
        # api070/p6e: merged into Result.stats(), which adds the Result's frequency and payload
        # counts; the probes compare the solver's entries only.
        return dict(result._solver_stats)
    return result.solver_stats


def compile_materials(prepared_scene):
    """``PreparedScene.compile_materials()``."""
    if "p7b" in BREAKS:
        # api070/p7b: compile_materials(prepared) in witwin.maxwell.compiler.
        from witwin.maxwell.compiler import compile_materials as compile_prepared_materials

        return compile_prepared_materials(prepared_scene)
    return prepared_scene.compile_materials()
