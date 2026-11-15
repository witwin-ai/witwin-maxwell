"""Constructor-driven scene dictionaries, schema ``witwin.scene/2``.

A document is strict JSON (no NaN or infinity, string keys only) plus an optional caller-owned
table of large arrays. Every ``Scene`` field is written. Every scene object is a record
``{"type": <tag>, <constructor field>: <value>, ...}`` of one registered type (``_codecs``); the
reader rebuilds it through the same public constructor, so validation and derived state are
recomputed exactly as for a hand-written scene. An object of an unregistered type, a callable or
any other value outside the schema raises :class:`~witwin.maxwell.errors.SchemaError` naming the
document path: nothing is silently dropped.

Value encodings:

- ``None``, ``bool``, ``int``, ``str`` and finite ``float`` are JSON values; a non-finite float is
  ``{"type": "Float", "value": "inf" | "-inf" | "nan"}``; a complex number is
  ``{"type": "Complex", "real": ..., "imag": ...}``.
- A tuple is a JSON list; a list is ``{"type": "List", "items": [...]}``; a mapping (metadata
  included) is ``{"type": "Mapping", "items": [[key, value], ...]}``, which keeps its key order and
  key types through stores that sort JSON object keys.
- A tensor is ``{"type": "Tensor", "dtype", "shape", "values"}`` (complex values as ``[re, im]``
  pairs), or ``{"array": <path>}`` when an arrays table is given and it has more than 4096
  elements; the table keeps it bit-exact with its dtype. A trainable tensor adds
  ``"requires_grad": true`` and is rebuilt as a fresh leaf (optimizer state is not part of a scene).
  A NumPy array is the same with ``"type": "NDArray"`` (``"ndarray": true`` next to ``array``).
- Materials and wire conductors are stored once in ``materials`` and referenced by
  ``{"type": "MaterialRef", "key": "m<k>"}``, so sharing survives the round trip.

The key ``array`` with a string value is used for array references only. Array names are the
document paths of their values, so a scene always produces the same names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
import functools
import json
from functools import cache
from inspect import Parameter, signature
import math
import types

import numpy as np
import torch

from ..errors import SchemaError

SCHEMA = "witwin.scene/2"
ARRAY_THRESHOLD = 4096
#: A run of at least this many consecutive same-type records in a Scene collection is written as one
#: ``Records`` block of per-field columns (EMFC F1: 16,384 point monitors).
COLUMNAR_THRESHOLD = 64


@dataclass(frozen=True)
class _Codec:
    """One registered scene type: its tag, constructor fields, attribute readers and builder."""

    cls: type
    tag: str
    fields: tuple[str, ...]
    getters: Mapping[str, object]
    build: object
    ids: Mapping[str, str]

    def value(self, obj, name):
        getter = self.getters.get(name)
        return getter(obj) if getter is not None else getattr(obj, name)


def _refuse(path, what):
    raise SchemaError(f"{path or '<document>'}: {what} is not serializable in {SCHEMA}.")


def _parameters(cls):
    return {name: parameter for name, parameter in signature(cls).parameters.items() if name != "device"}


def _call(cls, kwargs):
    """Call ``cls`` with keyword values; positional-only and ``*args`` parameters go positionally."""
    parameters = _parameters(cls)
    star = any(p.kind == Parameter.VAR_POSITIONAL for p in parameters.values())
    args = []
    for name, parameter in parameters.items():
        if parameter.kind == Parameter.VAR_POSITIONAL:
            args.extend(kwargs.pop(name, ()))
        elif name in kwargs and (parameter.kind == Parameter.POSITIONAL_ONLY or (
                star and parameter.kind == Parameter.POSITIONAL_OR_KEYWORD)):
            args.append(kwargs.pop(name))
    return cls(*args, **kwargs)


def _hard_smoothness(obj):
    # A hard boolean was authored with a Python number <= 0; a tensor would make it smooth.
    return obj.smoothness if obj.smooth else obj.smoothness.item()


def _build_tfsf(cls, kwargs):
    if kwargs.get("mode", "box") == "slab":
        return cls.slab(axis=kwargs["axis"], bounds=kwargs["axis_bounds"])
    return cls(kwargs["bounds"])


def _host_authored(mesh):
    # Host-authored meshes always run the host topology checks and keep no tensor revision.
    return mesh.topology_diagnostics and mesh._topology_diagnostic_revision is None


def _mesh_array(name):
    """Vertices/faces as NumPy arrays for a host-authored mesh, so the rebuild takes the same host path."""
    def read(mesh):
        tensor = getattr(mesh, name)
        return tensor.detach().cpu().numpy() if _host_authored(mesh) else tensor
    return read


def _structure_material_id(structure):
    """``None`` when the structure inherits its material's identity, so the rebuild inherits it too."""
    inherited = getattr(structure.material, "material_id", None)
    return None if structure.material_id == inherited else structure.material_id


def _boundary_kind(spec):
    """The authored base kind: a mixed spec takes it from any face no override covers."""
    if spec.kind != "mixed":
        return spec.kind
    for axis_index, axis in enumerate("xyz"):
        for side_index, side in enumerate(("low", "high")):
            if getattr(spec, axis) is None and getattr(spec, f"{axis}_{side}") is None:
                return spec._face_kinds[axis_index][side_index]
    return "none"


def _complex_loops(slab):
    return tuple(torch.split(slab.vertices, slab.loop_sizes))


def _open_endpoints(wire):
    # A closed loop derives its (empty) endpoints; passing them back is refused.
    return wire.endpoints or None


def _perturbation_base(medium):
    """The base material a pole-unperturbed ``PerturbationMedium`` copied its fields from."""
    from ..media import Material

    if any(any(value != 0.0 for value in getattr(medium, f"{kind}_sensitivities"))
           for kind in ("debye", "drude", "lorentz")):
        raise AttributeError("pole sensitivities lower the base poles irreversibly")
    return Material(
        eps_r=medium.eps_r, mu_r=medium.mu_r, sigma_e=medium.sigma_e, name=medium.name,
        debye_poles=medium.debye_poles, drude_poles=medium.drude_poles, lorentz_poles=medium.lorentz_poles,
        mu_debye_poles=medium.mu_debye_poles, mu_drude_poles=medium.mu_drude_poles,
        mu_lorentz_poles=medium.mu_lorentz_poles, epsilon_tensor=medium.epsilon_tensor,
        mu_tensor=medium.mu_tensor, sigma_e_tensor=medium.sigma_e_tensor, kerr_chi3=medium.kerr_chi3,
        nonlinearity=medium.nonlinearity, modulation=medium.modulation,
    )


_CIRCUIT_TERMINALS = ("positive", "negative", "control_positive", "control_negative")


def _circuit_getters():
    def nodes(circuit):
        return tuple(node.name for node in circuit.nodes if node is not circuit.ground)

    def generated(circuit):
        return tuple(node.name for node in circuit.nodes if node.name.casefold() in circuit._generated_node_keys)

    def bindings(circuit):
        return tuple((binding.port_name, binding.positive.name, binding.negative.name) for binding in circuit.bindings)

    def initial_conditions(circuit):
        return tuple((name, value, constraint) for name, (value, constraint) in circuit.initial_conditions.items())

    return {"nodes": nodes, "generated_nodes": generated, "devices": lambda circuit: circuit.devices,
            "bindings": bindings, "initial_conditions": initial_conditions}


def _build_circuit(cls, kwargs):
    """Replay the builder: nodes in order, devices on the circuit's own nodes, bindings, conditions."""
    circuit = cls(kwargs["name"], parameters=kwargs.get("parameters"), config=kwargs.get("config"),
                  metadata=kwargs.get("metadata"))
    generated = set(kwargs.get("generated_nodes", ()))
    for name in kwargs.get("nodes", ()):
        circuit._intern_generated_node(name) if name in generated else circuit.node(name)

    def node(name):
        return circuit._node_keys[name.casefold()]

    for device in kwargs.get("devices", ()):
        # Decoded devices hold fresh nodes; a circuit accepts only its own node objects.
        codec = _codecs()[type(device)]
        fields = {name: codec.value(device, name) for name in codec.fields}
        terminals = {name: node(value.name) for name, value in fields.items()
                     if name in _CIRCUIT_TERMINALS and hasattr(value, "is_ground")}
        circuit.add(codec.build(codec.cls, {**fields, **terminals}) if terminals else device)
    for port_name, positive, negative in kwargs.get("bindings", ()):
        circuit.bind_port(port_name, positive=node(positive), negative=node(negative))
    for name, value, constraint in kwargs.get("initial_conditions", ()):
        circuit.set_initial_condition(node(name), value, constraint=constraint)
    return circuit


def _build_esd_waveform(cls, kwargs):
    return cls.iec_61000_4_2(kwargs["level_voltage"], discharge=kwargs["discharge"],
                             standard_revision=kwargs["standard_revision"])


def _certificate_samples(impedance):
    """The ``certificate_samples`` whose certificate reproduces the stored passivity report."""
    report = impedance.passivity
    for samples in (64, *range(2, report.sample_count + 3)):
        frequencies = impedance._certificate_frequencies(impedance.frequency_range, samples, impedance.model)
        if impedance.model.check_passivity(frequencies, tolerance=report.tolerance) == report:
            return samples
    raise AttributeError("certificate sample count not recoverable")


def _build_touchstone(cls, kwargs):
    return cls(kwargs.pop("name"), **kwargs)


@cache
def _codecs() -> dict[type, _Codec]:
    """The registry: every scene type ``witwin.scene/2`` writes, keyed by exact class."""
    from witwin.core import PhysicalMaterial, Structure
    from witwin.core import material as core_material
    from witwin.core.geometry import boolean, mesh, primitives

    from .. import (
        absorbers, breakdown_risk, breakdown_stress, circuit_devices, circuits, coordinates, copper_foil, esd, layers, lumped,
        media, monitors, network, path_integrals, ports, rational, roughness, scene, sources, thin_wire,
    )
    from ..electrostatic import api as electrostatic
    from ..geometry import planar_pec, polyslab, surface_mesh

    registry = {}

    def register(cls, *, fields=None, getters=None, build=None, skip=(), ids=None):
        if cls in registry:
            raise RuntimeError(f"{cls.__name__} registered twice")
        if fields is None:
            parameters = _parameters(cls)
            if any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
                raise RuntimeError(f"{cls.__name__} needs explicit serialized fields")
            fields = tuple(name for name in parameters if name not in skip)
        if {"type", "array"}.intersection(fields):
            raise RuntimeError(f"{cls.__name__} field names collide with the record keys 'type'/'array'")
        registry[cls] = _Codec(cls, cls.__name__, tuple(fields), dict(getters or {}), build or _call, dict(ids or {}))

    for cls in (primitives.Box, primitives.Sphere, primitives.Cylinder, primitives.Ellipsoid, primitives.Cone,
                primitives.Pyramid, primitives.Prism, primitives.ExtrudedPolygon, primitives.Torus,
                primitives.HollowBox, polyslab.PolySlab, planar_pec.PlanarPEC, surface_mesh.SurfaceMesh):
        register(cls)
    register(mesh.Mesh, getters={"vertices": _mesh_array("vertices"), "faces": _mesh_array("faces")})
    for cls in (boolean.Union, boolean.Intersection, boolean.Difference):
        register(cls, getters={"smoothness": _hard_smoothness})
    register(polyslab.ComplexPolySlab, getters={"loops": _complex_loops})
    # Identity integers come from process-wide counters; documents number them per scene (see _Writer.identity).
    register(Structure, getters={"material_id": _structure_material_id},
             ids={name: name for name in ("structure_id", "material_id", "assignment_id", "surface_id")})
    register(PhysicalMaterial, ids={"material_id": "material_id"})
    for cls in (core_material.PhaseScreen, core_material.SurfaceRoughness, core_material.MaterialLayer,
                core_material.PowerLawDispersion):
        register(cls)

    register(scene.Domain)
    register(scene.GridSpec)
    register(scene.BoundarySpec, getters={"kind": _boundary_kind})
    for cls in (scene.SubpixelSpec, scene.MeshOverrideStructure, scene.LayerRefinementSpec,
                scene.GridRefinement, scene.CornerFinderSpec, scene.MaterialRegion):
        register(cls)
    for cls in (absorbers.MurABC, absorbers.ModeABC, absorbers.InternalAbsorber):
        register(cls)

    for cls in (media.Material, copper_foil.CopperFoil, media.LossyMetalMedium, media.DebyePole, media.DrudePole,
                media.LorentzPole, media.DiagonalTensor3, media.Tensor3x3, media.ComplexTensor3x3,
                media.CustomDebyePole, media.CustomDrudePole, media.CustomLorentzPole, media.Graphene,
                media.Medium2D, media.GyromagneticFerrite, media.ModulationSpec, media.NonlinearSusceptibility,
                media.TensorNonlinearSusceptibility, media.TwoPhotonAbsorption,
                media.SurfaceImpedanceMedium, media.DielectricBreakdown, breakdown_risk.SmoothBreakdownRisk,
                roughness.GroisseRoughness, roughness.HammerstadRoughness, roughness.HurayRoughness,
                layers.WidebandDebye, layers.PolePermittivity, layers.Layer, layers.HalfSpace, layers.LayerStack):
        register(cls)
    register(media.PerturbationMedium, getters={"base": _perturbation_base})
    register(media.RationalSurfaceImpedance, getters={
        "poles": lambda impedance: impedance.model, "residues": lambda impedance: None,
        "direct": lambda impedance: 0.0, "representation": lambda impedance: impedance.representation,
        "passivity_tolerance": lambda impedance: impedance.passivity.tolerance,
        "certificate_samples": _certificate_samples,
    })

    register(sources.TFSF, fields=("mode", "bounds", "axis", "axis_bounds"), build=_build_tfsf)
    for cls in (sources.CW, sources.GaussianPulse, sources.RickerWavelet, sources.BroadbandPulse,
                sources.CustomSourceTime, sources.BasebandGaussianPulse, sources.BasebandRectangularPulse,
                sources.PointDipole, sources.PlaneWave, sources.GaussianBeam, sources.AstigmaticGaussianBeam,
                sources.ModeSource, sources.UniformCurrentSource, sources.CustomCurrentSource,
                sources.CustomFieldSource, sources.CurrentDataset, sources.FieldDataset, sources.MaxwellPlaneModes,
                breakdown_stress.ComponentRating):
        register(cls)
    for cls in (esd.ESDCurrentSource, esd.ESDVoltageSource):
        register(cls, getters={"port": lambda source: source.port_name})
    register(esd.MeasuredWaveform, getters={"provenance": lambda waveform: waveform._provenance})
    register(esd.ESDWaveform, fields=("level_voltage", "discharge", "standard_revision"), build=_build_esd_waveform)

    for cls in (monitors.PointMonitor, monitors.PlaneMonitor, monitors.FinitePlaneMonitor, monitors.FluxMonitor,
                monitors.ClosedSurfaceMonitor, monitors.ModeMonitor, monitors.AngleProjectionMonitor,
                monitors.CartesianProjectionMonitor, monitors.KSpaceProjectionMonitor,
                monitors.PermittivityMonitor, monitors.MediumMonitor, monitors.DiffractionMonitor,
                monitors.DipoleEmissionMonitor, monitors.FieldTimeMonitor, monitors.AuxFieldTimeMonitor,
                monitors.FluxTimeMonitor, monitors.SurfaceFieldTimeMonitor, monitors.GaussianOverlapMonitor,
                monitors.AstigmaticGaussianOverlapMonitor, monitors.PowerLossMonitor, monitors.WireMonitor,
                monitors.ComponentStressMonitor, monitors.IncidentPowerDensityMonitor):
        register(cls)
    register(monitors.BreakdownMonitor, getters={"region": lambda monitor: None})  # resolved to position/size

    register(ports.TerminalPort, getters={"current_surface": lambda port: port._declared_current_surface})
    for cls in (ports.ModePort, ports.LumpedPort, ports.WavePort, ports.WaveModeSpec, ports.TerminalWavePort,
                ports.CoaxialLumpedPort, ports.AxisPath, ports.TerminalRef, ports.WireNodeRef,
                ports.WirePortBinding, ports.PECFrame,
                path_integrals.AxisSegment, path_integrals.PolylinePath, path_integrals.CircleLoop,
                path_integrals.PolygonLoop, path_integrals.RectangleLoop, path_integrals.CustomVoltageIntegral,
                path_integrals.CustomCurrentIntegral, path_integrals.CompositeVoltageIntegral,
                path_integrals.CompositeCurrentIntegral,
                lumped.Resistor, lumped.Capacitor, lumped.Inductor, lumped.SeriesRLC, lumped.ParallelRLC,
                lumped.CoaxialLumpedResistor, network.NetworkBlock, network.NetworkData,
                rational.StateSpaceNetwork, rational.RationalModel, rational.RationalFitConfig,
                rational.FitReport, rational.NetworkFitReport, thin_wire.WireConductor, thin_wire.WireEnd):
        register(cls)
    register(thin_wire.ThinWire, getters={"endpoints": _open_endpoints})
    register(network.TouchstoneNetwork, fields=tuple(_parameters(network.NetworkBlock)), build=_build_touchstone)

    for cls in (circuits.CircuitNode, circuits.PortBinding, circuits.MNAConfig, circuits.PulseWaveform,
                circuits.SineWaveform, circuits.PiecewiseLinearWaveform, circuits.VoltageSource,
                circuits.CurrentSource, circuits.VoltageControlledVoltageSource,
                circuits.VoltageControlledCurrentSource, circuits.CurrentControlledVoltageSource,
                circuits.CurrentControlledCurrentSource, circuits.MutualInductor, circuits.TimedSwitch,
                circuit_devices.Diode, circuit_devices.PiecewiseLinearIV, circuit_devices.PolynomialIV,
                circuit_devices.VoltageDependentCapacitor, circuit_devices.NonlinearSolveConfig):
        register(cls)

    register(circuits.Circuit, fields=("name", "parameters", "config", "metadata", "nodes", "generated_nodes",
                                       "devices", "bindings", "initial_conditions"),
             getters=_circuit_getters(), build=_build_circuit)

    register(electrostatic.ElectrostaticTerminal)
    register(electrostatic.ChargeDensity)
    register(coordinates.BendCoordinates)
    return registry


@cache
def _codecs_by_tag() -> dict[str, _Codec]:
    tags = {}
    for codec in _codecs().values():
        if codec.tag in tags:
            raise RuntimeError(f"duplicate scene tag {codec.tag}")
        tags[codec.tag] = codec
    return tags


@cache
def _shared_types():
    from witwin.core import PhysicalMaterial

    from ..thin_wire import WireConductor

    return (PhysicalMaterial, WireConductor)


def _float(value):
    if math.isfinite(value):
        return value
    return {"type": "Float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}


def _float_from(record):
    return float(record["value"])


def _plain_values(values):
    """Nested element lists with non-finite floats in their typed form."""
    if isinstance(values, list):
        return [_plain_values(item) for item in values]
    return _float(values) if isinstance(values, float) else values


_FUNCTIONS = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, functools.partial)


def _instance_state(obj):
    """(name, value) of every stored attribute, ``__dict__`` and ``__slots__`` alike."""
    state = dict(getattr(obj, "__dict__", {}))
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in state and not name.startswith("__") and hasattr(obj, name):
                state[name] = getattr(obj, name)
    return state.items()


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


class _Writer:
    def __init__(self, arrays):
        self.arrays = arrays
        self.materials = {}
        self.material_keys = {}
        self.identities = {}
        self.tensor_sites = {}  # id(tensor) -> [tensor, [(record, path), ...]] in first-seen order

    def identity(self, kind, value):
        """Scene-local identity number: process-wide counter values numbered in document order.

        The same scene writes the same numbers whatever the process built before; equal
        identities stay equal and distinct ones distinct.
        """
        numbers = self.identities.setdefault(kind, {})
        return numbers.setdefault(int(value), len(numbers))

    def material(self, value, path):
        key = self.material_keys.get(id(value))
        if key is None:
            key = f"m{len(self.material_keys)}"
            self.material_keys[id(value)] = key
            self.materials[key] = None  # reserve the slot, so nested references keep first-seen order
            self.materials[key] = self.record(value, f"materials/{key}")
        return {"type": "MaterialRef", "key": key}

    def record(self, obj, path):
        codec = _codecs().get(type(obj))
        if codec is None:
            _refuse(path, f"{type(obj).__module__}.{type(obj).__qualname__}")
        for name, value in _instance_state(obj):
            if isinstance(value, _FUNCTIONS):
                # Constructor fields cannot carry a function (CustomSourceTime(fn) keeps it privately).
                _refuse(f"{path}/{name.lstrip('_')}", f"callable {getattr(value, '__qualname__', type(value).__name__)}")
        result = {"type": codec.tag}
        for name in codec.fields:
            try:
                value = codec.value(obj, name)
            except AttributeError as error:
                _refuse(f"{path}/{name}", f"{codec.tag}.{name} ({error})")
            if name in codec.ids and value is not None:
                value = self.identity(codec.ids[name], value)
            result[name] = self.encode(value, f"{path}/{name}")
        return result

    def tensor(self, value, path):
        sites = self.tensor_sites.setdefault(id(value), [value, []])  # holds the tensor: ids stay unique
        result = self.tensor_record(value, path)
        sites[1].append((result, path))
        return result

    def tensor_record(self, value, path):
        tensor = value.detach()
        if self.arrays is not None and tensor.numel() > ARRAY_THRESHOLD:
            self.arrays[path] = tensor
            result = {"array": path}
        else:
            host = tensor.cpu()
            values = (torch.view_as_real(host) if host.is_complex() else host).tolist()
            if host.is_floating_point() or host.is_complex():
                values = _plain_values(values)
            result = {"type": "Tensor", "dtype": _dtype_name(tensor.dtype), "shape": list(tensor.shape),
                      "values": values}
        if value.requires_grad:
            result["requires_grad"] = True
        return result

    def shared_tensors(self):
        """Move every tensor written more than once into the ``tensors`` table; its sites become references.

        Sharing is by identity: one trainable parameter used by two structures reloads as one leaf.
        """
        table = {}
        for tensor, sites in self.tensor_sites.values():
            if len(sites) < 2:
                continue
            key = f"t{len(table)}"
            first, _ = sites[0]
            table[key] = dict(first)
            for record, path in sites:
                if record is not first and self.arrays is not None and isinstance(record.get("array"), str):
                    self.arrays.pop(path, None)
                record.clear()
                record.update({"type": "TensorRef", "key": key})
        return table

    def ndarray(self, value, path):
        if value.dtype == object or value.dtype.kind not in "biufc":
            _refuse(path, f"NumPy array of dtype {value.dtype}")
        if self.arrays is not None and value.size > ARRAY_THRESHOLD:
            self.arrays[path] = torch.from_numpy(np.ascontiguousarray(value))
            return {"array": path, "ndarray": True}
        values = (value.view(value.real.dtype).reshape(*value.shape, 2) if value.dtype.kind == "c"
                  else value).tolist()
        return {"type": "NDArray", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": _plain_values(values)}

    def encode(self, value, path):
        if isinstance(value, Enum):  # before str: option enums are str subclasses
            return self.encode(value.value, path)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return str(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return _float(float(value))
        if isinstance(value, complex):
            return {"type": "Complex", "real": _float(value.real), "imag": _float(value.imag)}
        if isinstance(value, torch.Tensor):
            return self.tensor(value, path)
        if isinstance(value, np.ndarray):
            return self.ndarray(value, path)
        if isinstance(value, np.generic):
            return self.encode(value.item(), path)
        if isinstance(value, torch.dtype):
            return {"type": "DType", "value": _dtype_name(value)}
        if isinstance(value, _shared_types()):
            return self.material(value, path)
        if type(value) in _codecs():
            return self.record(value, path)
        if isinstance(value, tuple) and type(value) is tuple:
            return [self.encode(item, f"{path}/{index}") for index, item in enumerate(value)]
        if type(value) is list:
            return {"type": "List", "items": [self.encode(item, f"{path}/{index}") for index, item in enumerate(value)]}
        if isinstance(value, (dict, types.MappingProxyType)):
            # Item lists keep the key order (stores may sort JSON object keys) and any key type.
            return {"type": "Mapping", "items": [
                [self.encode(key, f"{path}/{index}/key"), self.encode(item, f"{path}/{index}/value")]
                for index, (key, item) in enumerate(value.items())]}
        if callable(value):
            _refuse(path, f"callable {getattr(value, '__qualname__', type(value).__name__)}")
        _refuse(path, f"{type(value).__module__}.{type(value).__qualname__}")


def _float_rows(values):
    """``values`` is a column of equal-length lists of plain finite floats (packable into float64)."""
    if not values or not all(type(item) is list and item for item in values):
        return False
    width = len(values[0])
    return all(len(item) == width and all(type(entry) is float for entry in item) for item in values)


def _name_pattern(values):
    """``(prefix, width, start)`` when ``values`` are ``prefix`` + consecutive zero-padded integers."""
    if not all(type(value) is str for value in values):
        return None
    first = values[0]
    digits = len(first) - len(first.rstrip("0123456789"))
    if digits == 0:
        return None
    prefix, start = first[:-digits], int(first[-digits:])
    expected = [f"{prefix}{start + index:0{digits}d}" for index in range(len(values))]
    return (prefix, digits, start) if expected == values else None


def _columnar(records, path, arrays):
    """Runs of COLUMNAR_THRESHOLD or more consecutive same-type records become ``Records`` blocks."""
    result, index = [], 0
    while index < len(records):
        record = records[index]
        tag = record.get("type") if isinstance(record, dict) else None
        stop = index + 1
        if tag in _codecs_by_tag():
            while (stop < len(records) and isinstance(records[stop], dict) and records[stop].get("type") == tag
                   and records[stop].keys() == record.keys()):
                stop += 1
        if stop - index < COLUMNAR_THRESHOLD:
            result.extend(records[index:stop])
            index = stop
            continue
        run = records[index:stop]
        columns = {}
        for name in record:
            if name == "type":
                continue
            values = [item[name] for item in run]
            pattern = _name_pattern(values)
            first = json.dumps(values[0])
            if all(json.dumps(value) == first for value in values[1:]):  # exact JSON: 1 and 1.0 differ
                columns[name] = {"constant": values[0]}
            elif pattern is not None:
                columns[name] = {"prefix": pattern[0], "digits": pattern[1], "start": pattern[2]}
            elif arrays is not None and _float_rows(values):
                key = f"{path}/{index}/columns/{name}"
                arrays[key] = torch.tensor(values, dtype=torch.float64)
                columns[name] = {"array": key}
            else:
                columns[name] = {"values": values}
        result.append({"type": "Records", "of": tag, "count": len(run), "columns": columns})
        index = stop
    return result


def _rows(block, path, arrays):
    """The records of a ``Records`` block."""
    count, columns = block["count"], block["columns"]
    expanded = {}
    for name, column in columns.items():
        if "constant" in column:
            expanded[name] = [column["constant"]] * count
        elif "prefix" in column:
            expanded[name] = [f"{column['prefix']}{column['start'] + index:0{column['digits']}d}"
                              for index in range(count)]
        elif isinstance(column.get("array"), str):
            if arrays is None or column["array"] not in arrays:
                raise SchemaError(f"{path}/columns/{name}: array {column['array']!r} is missing from the arrays table.")
            expanded[name] = arrays[column["array"]].detach().cpu().to(torch.float64).tolist()
        else:
            expanded[name] = column["values"]
        if len(expanded[name]) != count:
            raise SchemaError(f"{path}/columns/{name}: {len(expanded[name])} values for {count} records.")
    return [{"type": block["of"], **{name: values[index] for name, values in expanded.items()}}
            for index in range(count)]


def _numeric_values(values):
    if isinstance(values, list):
        return [_numeric_values(item) for item in values]
    if isinstance(values, dict):
        if values.get("type") != "Float":
            raise SchemaError(f"unexpected element {values!r} in tensor values")
        return _float_from(values)
    return values


def _rounding_only(rebuilt, recorded):
    return (type(rebuilt) is tuple and type(recorded) is tuple and len(rebuilt) == len(recorded) and rebuilt
            and all(type(a) is float and type(b) is float and abs(a - b) <= 2.0 * math.ulp(max(abs(a), abs(b)))
                    for a, b in zip(rebuilt, recorded)))


def _restore_rounding(obj, recorded):
    """Put back recorded float vectors that a constructor renormalized by a rounding step.

    Normalizing an already unit vector can move it by an ulp (a plane-wave direction); the record
    holds the bits the original object used, so the rebuilt object takes them back. Larger
    differences are left alone and show up as a failed round trip.
    """
    for name, value in recorded.items():
        if type(value) is not tuple or not value or any(type(item) is not float for item in value):
            continue
        try:
            rebuilt = getattr(obj, name)
        except Exception:  # a property that needs the scene context
            continue
        if _rounding_only(rebuilt, value) and rebuilt != value:
            object.__setattr__(obj, name, value)


class _Reader:
    def __init__(self, document, arrays):
        self.arrays = arrays
        self.material_records = document.get("materials", {})
        self.materials = {}
        self.tensor_records = document.get("tensors", {})
        self.tensors = {}

    def shared_tensor(self, key, path):
        """One object per ``tensors`` entry: every reference gets the same (leaf) tensor."""
        if key not in self.tensors:
            if key not in self.tensor_records:
                raise SchemaError(f"{path}: unknown tensor key {key!r}.")
            self.tensors[key] = self.decode(self.tensor_records[key], f"tensors/{key}")
        return self.tensors[key]

    def array(self, record, path):
        name = record["array"]
        if self.arrays is None or name not in self.arrays:
            raise SchemaError(f"{path}: array {name!r} is missing from the arrays table.")
        tensor = self.arrays[name]
        if record.get("ndarray"):
            return tensor.detach().cpu().numpy().copy()
        tensor = tensor.detach().cpu()
        if record.get("requires_grad"):
            tensor = tensor.clone().requires_grad_(True)
        return tensor

    def tensor(self, record):
        dtype = getattr(torch, record["dtype"])
        values = _numeric_values(record["values"])
        if dtype.is_complex:
            real = torch.tensor(values, dtype=torch.empty((), dtype=dtype).real.dtype)
            tensor = torch.view_as_complex(real.reshape(*record["shape"], 2).contiguous())
        else:
            tensor = torch.tensor(values, dtype=dtype).reshape(record["shape"])
        if record.get("requires_grad"):
            tensor.requires_grad_(True)
        return tensor

    def ndarray(self, record):
        dtype = np.dtype(record["dtype"])
        values = _numeric_values(record["values"])
        if dtype.kind == "c":
            real = np.asarray(values, dtype=np.zeros((), dtype=dtype).real.dtype).reshape(*record["shape"], 2)
            return np.ascontiguousarray(real).view(dtype).reshape(record["shape"])
        return np.asarray(values, dtype=dtype).reshape(record["shape"])

    def material(self, key, path):
        if key not in self.materials:
            if key not in self.material_records:
                raise SchemaError(f"{path}: unknown material key {key!r}.")
            self.materials[key] = None  # a material may not contain itself
            self.materials[key] = self.decode(self.material_records[key], f"materials/{key}")
        if self.materials[key] is None:
            raise SchemaError(f"{path}: material {key!r} refers to itself.")
        return self.materials[key]

    def record(self, record, path):
        tag = record["type"]
        codec = _codecs_by_tag().get(tag)
        if codec is None:
            raise SchemaError(f"{path}: unknown scene type {tag!r} in {SCHEMA}.")
        unknown = sorted(set(record) - {"type"} - set(codec.fields))
        if unknown:
            raise SchemaError(f"{path}: {tag} has no serialized field(s) {unknown}.")
        kwargs = {}
        for name in codec.fields:
            if name in record:
                kwargs[name] = self.decode(record[name], f"{path}/{name}")
        recorded = dict(kwargs)
        try:
            obj = codec.build(codec.cls, kwargs)
        except SchemaError:
            raise
        except Exception as error:
            raise SchemaError(f"{path}: rebuilding {tag} failed: {type(error).__name__}: {error}") from error
        _restore_rounding(obj, recorded)
        return obj

    def decode(self, value, path):
        if isinstance(value, list):
            items = []
            for index, item in enumerate(value):
                if isinstance(item, dict) and item.get("type") == "Records":
                    items.extend(self.decode(row, f"{path}/{index}/{offset}") for offset, row
                                 in enumerate(_rows(item, f"{path}/{index}", self.arrays)))
                else:
                    items.append(self.decode(item, f"{path}/{index}"))
            return tuple(items)
        if not isinstance(value, dict):
            return value
        if isinstance(value.get("array"), str):
            return self.array(value, path)
        tag = value.get("type")
        if tag is None:
            return {key: self.decode(item, f"{path}/{key}") for key, item in value.items()}
        if tag == "Tensor":
            return self.tensor(value)
        if tag == "NDArray":
            return self.ndarray(value)
        if tag == "Float":
            return _float_from(value)
        if tag == "Complex":
            return complex(self.decode(value["real"], path), self.decode(value["imag"], path))
        if tag == "DType":
            return getattr(torch, value["value"])
        if tag == "List":
            return [self.decode(item, f"{path}/{index}") for index, item in enumerate(value["items"])]
        if tag == "Mapping":
            return {self.decode(key, f"{path}/{index}/key"): self.decode(item, f"{path}/{index}/value")
                    for index, (key, item) in enumerate(value["items"])}
        if tag == "MaterialRef":
            return self.material(value["key"], path)
        if tag == "TensorRef":
            return self.shared_tensor(value["key"], path)
        return self.record(value, path)


def _scene_fields():
    from ..scene import Scene

    return tuple(field.name for field in dataclass_fields(Scene))


def scene_to_dict(scene, arrays=None):
    """Write ``scene`` as a ``witwin.scene/2`` document; large tensors go to ``arrays`` when given."""
    from .. import __version__

    writer = _Writer(arrays)
    body = {}
    for name in _scene_fields():
        body[name] = writer.encode(getattr(scene, name), name)
    tensors = writer.shared_tensors()
    for name, value in body.items():
        if isinstance(value, list):
            body[name] = _columnar(value, name, arrays)
    head = {"schema": SCHEMA, "package_version": __version__, "unit": {"length": "m", "frequency": "Hz"},
            "materials": writer.materials}
    if tensors:
        head["tensors"] = tensors
    return {**head, **body}


def scene_from_dict(d, arrays=None):
    """Rebuild a ``witwin.scene/2`` document through the public constructors (tensors on CPU).

    A ``witwin.scene/1`` document is upgraded first by :func:`witwin.maxwell.io.migrate_scene_dict`.
    """
    from ..scene import Scene

    if not isinstance(d, Mapping):
        raise SchemaError(f"a scene document is a mapping, not {type(d).__name__}.")
    schema = d.get("schema")
    if schema != SCHEMA:
        from .migrate import SCHEMA_V1, migrate_scene_dict

        if schema != SCHEMA_V1:
            raise SchemaError(f"unsupported scene schema {schema!r}; this version reads {SCHEMA!r} "
                              f"and migrates {SCHEMA_V1!r}.")
        d = migrate_scene_dict(d)
    fields = _scene_fields()
    unknown = sorted(set(d) - {"schema", "package_version", "unit", "materials", "tensors"} - set(fields))
    if unknown:
        raise SchemaError(f"unknown scene document key(s) {unknown}.")
    if d.get("unit", {"length": "m", "frequency": "Hz"}) != {"length": "m", "frequency": "Hz"}:
        raise SchemaError(f"unsupported units {d['unit']!r}; {SCHEMA} is in meters and hertz.")
    reader = _Reader(d, arrays)
    kwargs = {name: reader.decode(d[name], name) for name in fields if name in d}
    return Scene(**kwargs)
