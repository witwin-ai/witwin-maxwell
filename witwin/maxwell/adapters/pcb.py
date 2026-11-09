"""Explicit import of numerical PCB antenna scene records into standard Scene objects."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from ..media import Material
from ..ports import AxisPath, LumpedPort
from ..scene import BoundarySpec, Domain, GridSpec, Scene, SubpixelSpec
from witwin.core import Box, Structure


def _bounds(values, label):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{label} requires six finite SI bounds.")
    result = result.reshape(3, 2)
    if np.any(result[:, 1] < result[:, 0]):
        raise ValueError(f"{label} bounds must be increasing.")
    return result


def _box(bounds):
    return Box(position=torch.tensor(bounds.mean(axis=1), dtype=torch.float64),
               size=torch.tensor(bounds[:, 1] - bounds[:, 0], dtype=torch.float64))


def _contour(axes, positive, negative, aperture, normal):
    center, size = .5 * (positive + negative), np.zeros(3)
    for dim, nodes in enumerate(axes):
        halves = .5 * (nodes[:-1] + nodes[1:])
        if dim == normal:
            candidates = halves[(halves > min(positive[dim], negative[dim])) &
                                (halves < max(positive[dim], negative[dim]))]
            if not len(candidates):
                raise ValueError("Port gap contains no magnetic half-grid plane.")
            center[dim] = candidates[np.argmin(abs(candidates - center[dim]))]
        else:
            lower = halves[halves < aperture[dim, 0]]
            upper = halves[halves > aperture[dim, 1]]
            if not len(lower) or not len(upper):
                raise ValueError("Current contour cannot enclose aperture inside the declared grid.")
            center[dim] = .5 * (lower[-1] + upper[0])
            size[dim] = upper[0] - lower[-1]
    return Box(position=torch.tensor(center), size=torch.tensor(size))


def import_pcb_scene(definition, *, frequency, device="cuda", losses=True, selected_conductors=None):
    """Import physical bounds and a finite aperture without hidden excitation.

    Supported interchange schema is ``metaclaw.antenna-scene/1``. Tangent loss
    becomes static conductivity at this one observation frequency, enforced by
    Simulation. Broadband dielectric loss requires an explicit dispersive model.
    The caller drives the returned ordinary Scene using PortExcitation.
    Finite copper requires an explicit sheet support and constitutive_model;
    physical layer collapse and the associated grid/reference changes are input
    declarations, never inferred by the importer.
    """
    if not isinstance(definition, dict) or definition.get("schema") != "metaclaw.antenna-scene/1":
        raise ValueError("Unsupported numerical antenna scene schema.")
    if definition.get("units") != "m" or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("PCB import requires SI metre coordinates and a positive reference frequency.")
    if type(losses) is not bool:
        raise TypeError("losses must be boolean and is recorded in the import contract.")
    axes = [np.asarray(definition["mesh"]["axes_m"][axis], dtype=np.float64) for axis in "xyz"]
    if any(a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in axes):
        raise ValueError("PCB mesh axes must be finite, strictly increasing node arrays.")
    quadrature = definition['mesh'].get('subpixel_samples')
    if quadrature is not None:
        if (not isinstance(quadrature, dict)
                or set(quadrature)-{'samples', 'averaging', 'pec'}):
            raise ValueError('mesh.subpixel_samples requires an explicit samples/averaging/pec declaration.')
        subpixel = SubpixelSpec(**quadrature)
    else:
        subpixel = SubpixelSpec()
    materials, structures = {}, []
    for record in definition["materials"]:
        if record["id"] in materials:
            raise ValueError("Duplicate PCB material ID.")
        eps, mu = float(record["epsilon_r"]), float(record.get("mu_r", 1))
        loss = record.get("loss", {})
        if loss and loss.get("kind") != "loss_tangent":
            raise NotImplementedError("PCB import supports explicit reference-frequency tangent loss only.")
        tangent = float(loss.get("tan_delta", 0))
        if not all(math.isfinite(v) for v in (eps, mu, tangent)) or min(eps, mu) <= 0 or tangent < 0:
            raise ValueError("PCB materials require positive finite epsilon/mu and nonnegative loss.")
        materials[record["id"]] = Material(eps_r=eps, mu_r=mu,
            sigma_e=2 * math.pi * frequency * 8.8541878128e-12 * eps * tangent if losses else 0)
    names = set()
    for record in definition["dielectrics"]:
        if record["id"] in names:
            raise ValueError("Duplicate physical structure ID.")
        names.add(record["id"])
        structures.append(Structure(_box(_bounds(record["bounds_m"], record["id"])),
                                    materials[record["material_id"]], name=record["id"]))
    records = definition["conductors"]
    conductor_ids = [record["id"] for record in records]
    if len(set(conductor_ids)) != len(conductor_ids):
        raise ValueError("Duplicate conductor ID.")
    selected = list(definition["initial_conductor_ids"] if selected_conductors is None else selected_conductors)
    if len(set(selected)) != len(selected) or not set(selected) <= set(conductor_ids):
        raise ValueError("Selected conductors must be unique existing IDs.")
    active, conductor_models = [], {}
    for record in records:
        if record["role"] not in {"fixed", "design"} or record["kind"] not in {"sheet", "solid"}:
            raise ValueError("PCB conductors must declare a fixed/design role and sheet/solid kind.")
        if record["role"] != "fixed" and record["id"] not in selected:
            continue
        if record["id"] in names:
            raise ValueError("Conductor and dielectric IDs must be disjoint.")
        names.add(record["id"])
        bounds = _bounds(record["bounds_m"], record["id"])
        model = record.get("constitutive_model", {"kind": "pec"})
        if not isinstance(model, dict) or model.get("kind") not in {"pec", "copper_foil"}:
            raise ValueError("Conductor constitutive_model must explicitly select pec or copper_foil.")
        if model["kind"] == "copper_foil":
            from ..copper_foil import CopperFoil
            if not losses:
                raise ValueError("losses=False cannot disable explicitly declared copper loss; declare a PEC control input.")
            if record["kind"] != "sheet":
                raise ValueError("Copper foil import requires an explicit zero-thickness sheet support; collapse physical layers before import.")
            if set(model) != {"kind", "thickness_m", "conductivity_s_per_m", "frequency_range_hz"}:
                raise ValueError("Copper foil import requires physical thickness, conductivity and frequency range.")
            material = CopperFoil(thickness=model["thickness_m"], conductivity=model["conductivity_s_per_m"],
                                  frequency_range=model["frequency_range_hz"])
            material._frequencies(frequency)
            if np.count_nonzero(bounds[:, 1] == bounds[:, 0]) != 1:
                raise ValueError("Copper foil support requires exactly one zero-thickness axis.")
            geometry = _box(bounds)
        elif record["kind"] == "sheet":
            if set(model) != {"kind"}:
                raise ValueError("PEC constitutive_model accepts only kind; finite copper parameters require copper_foil.")
            from ..geometry.planar_pec import PlanarPEC
            zero = np.flatnonzero(bounds[:, 1] == bounds[:, 0])
            if len(zero) != 1:
                raise ValueError("A PEC sheet requires exactly one zero-thickness axis.")
            normal = int(zero[0])
            geometry = PlanarPEC.rectangle(bounds=tuple(map(tuple, np.delete(bounds, normal, axis=0))),
                                          normal_axis="xyz"[normal], coordinate=float(bounds[normal, 0]))
            material = Material.pec()
        else:
            if set(model) != {"kind"}:
                raise ValueError("PEC constitutive_model accepts only kind; finite copper parameters require copper_foil.")
            geometry = _box(bounds)
            material = Material.pec()
        structures.append(Structure(geometry, material, name=record["id"]))
        conductor_models[record["id"]] = dict(model)
        active.append(record["id"])
    ports = []
    for record in definition["ports"]:
        if record.get("type") != "lumped" or record.get("deembed_m", 0) != 0:
            raise NotImplementedError("PCB import currently requires an explicit zero-deembedded lumped port.")
        if not set(record["positive_conductor_ids"] + record["negative_conductor_ids"]) <= set(active):
            raise ValueError("Port terminal references must identify active conductors.")
        endpoints = np.asarray(record["integration_line_m"], dtype=np.float64)
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
            raise ValueError("Port integration line requires two finite SI points.")
        positive, negative = endpoints
        changed = np.flatnonzero(positive != negative)
        if len(changed) != 1:
            raise ValueError("Port integration line must be axis-aligned and nonzero.")
        normal = int(changed[0])
        aperture = _bounds(record["sheet_bounds_m"], record["id"])
        current_surface = (_box(_bounds(record["current_contour_bounds_m"], record["id"] + " current contour"))
                           if "current_contour_bounds_m" in record
                           else _contour(axes, positive, negative, aperture, normal))
        ports.append(LumpedPort(record["id"], positive=positive, negative=negative,
            voltage_path=AxisPath("xyz"[normal]), voltage_aperture=_box(aperture),
            current_surface=current_surface,
            reference_impedance=record["reference_ohm"]))
    if len({p.name for p in ports}) != len(ports):
        raise ValueError("PCB port names must be unique.")
    identity = hashlib.sha256(json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return Scene(domain=Domain(bounds=[(a[0], a[-1]) for a in axes]), grid=GridSpec.custom(*axes),
        boundary=BoundarySpec.pml(num_layers=definition["mesh"]["pml_cells"]), device=device,
        structures=structures, ports=ports, subpixel_samples=subpixel,
        metadata={"pcb_import": {"schema": "pcb-import/1", "definition_sha256": identity,
            "active_conductor_ids": active, "selected_conductor_ids": selected,
            "conductor_models": json.loads(json.dumps(conductor_models)),
            "material_quadrature": {'samples': list(subpixel.samples),
                'averaging': subpixel.averaging, 'pec': subpixel.pec},
            "losses_enabled": losses, "loss_reference_frequency_hz": frequency if losses else None,
            "source_contract": "No implicit excitation; use standard PortExcitation.",
            "current_contour_contract": {
                record["id"]: "Explicit physical native loop." if "current_contour_bounds_m" in record
                    else "Closest enclosing native half-grid loop."
                for record in definition["ports"]},
            "physical_qualification": "unqualified imported input; run convergence and reference checks"}})
