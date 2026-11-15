"""EMFC record shards for the datagen scheduler (``--format emfc``, opt-in).

One task becomes one EMFC record container (``emfc.encode_record``) appended to an
``emfc.shard`` dataset under ``<output-dir>/shards``: the builder's readouts as exact
arrays, provenance plus the ``maxwell_record`` interpretation block as strict-JSON
metadata, and the volume-DFT E phasors of every DFT frequency (``Result.fields``)
cropped to ``scene.domain.bounds`` as staggered fields ``E/f000``, ``E/f001``, ...
H is not stored; consumers derive it from E (``h_derivation``, null with reasons for a
scene where H does not follow from E) away from the TFSF H shell, which they rebuild
from ``grid`` and ``tfsf``. The cropped float64 node coordinates are exact arrays
``_grid/nodes_m/{x,y,z}``. Scenes with symmetry planes are refused, and so are samples
whose H must be stored (amendment A5: no H derivation, or a graded axis) request maxwell's
volume H DFT (``FDTDConfig.volume_dft_components``) and store ``H/fNNN`` too; on a maxwell
without that switch they are refused (``admit_sample``). Fields are encoded at
``tolerance`` (relative L2, certified per component) except for the deterministic
lossless audit subset (rule ``emfc-audit-v1``).

emfc is imported only by the functions that need it, so importing this module (and
the ``.pt`` path of the scheduler) never imports emfc. The plan and the decisions it
implements are emfc ``docs/plans/DATAGEN_EMFC.md``; the metadata is schema
``maxwell-datagen-record/1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import enum
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

SCHEMA = "maxwell-datagen-record/1"
AUDIT_RULE = "emfc-audit-v1"
# The closed boundary vocabulary of the record; any other maxwell face kind is an error.
BOUNDARY_KINDS = ("pml", "cpml", "periodic", "bloch", "pec", "pmc", "mur", "mode_abc",
                  "symmetry_pec", "symmetry_pmc")
SHARDS = "shards"
ASIDE = "shards-aside"
DATASET_FILE = "emfc_dataset.json"
# The E and H components of Result.fields, in component order. E component c is a cell
# sample along axis c and a node sample on the other two; H is the opposite.
E_NAMES = ("EX", "EY", "EZ")
H_NAMES = ("HX", "HY", "HZ")
AXES = "xyz"
# Records reach disk only when their shard seals (on roll or writer close), so a crashed
# worker loses its open shard's tasks; 1 GiB bounds that loss per rank (emfc's own default
# is 4 GiB).
DEFAULT_SHARD_TARGET_BYTES = 1 << 30


@dataclass(frozen=True)
class EmfcOptions:
    """The dataset-level settings every record and every resumed run must share."""

    dataset_seed: int
    field_tolerance: float = 1e-3
    audit_fraction: float = 0.01
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES
    dataset_uuid: str | None = None
    # Relative L2 bound of stored H (amendment A5/A6); None means the E bound.
    h_field_tolerance: float | None = None

    def __post_init__(self):
        if isinstance(self.dataset_seed, bool) or not isinstance(self.dataset_seed, int) or self.dataset_seed < 0:
            raise ValueError("dataset_seed must be a non-negative integer")
        if not (math.isfinite(self.field_tolerance) and self.field_tolerance > 0):
            raise ValueError("field_tolerance must be a positive finite relative L2 bound")
        if self.h_field_tolerance is None:
            object.__setattr__(self, "h_field_tolerance", self.field_tolerance)
        if not (math.isfinite(self.h_field_tolerance) and self.h_field_tolerance > 0):
            raise ValueError("h_field_tolerance must be a positive finite relative L2 bound")
        if not 0.0 <= self.audit_fraction <= 1.0:
            raise ValueError("audit_fraction must lie in [0, 1]")
        if self.shard_target_bytes <= 0:
            raise ValueError("shard_target_bytes must be positive")

    def settings(self) -> dict:
        """What a resumed run must match (the shard size may change between runs)."""
        return dict(schema=SCHEMA, audit_rule=AUDIT_RULE, dataset_seed=self.dataset_seed,
                    field_tolerance=self.field_tolerance, h_field_tolerance=self.h_field_tolerance,
                    audit_fraction=self.audit_fraction)


# --- Pure rules (no CUDA, no emfc) -------------------------------------------------------------
def audit_selected(dataset_seed: int, sample_id: int, fraction: float) -> bool:
    """The ``emfc-audit-v1`` rule: a pure function of the dataset seed and the sample id."""
    digest = hashlib.sha256(f"{AUDIT_RULE}:{dataset_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") < round(fraction * 2**64)


def crop_window(nodes, lo: float, hi: float) -> tuple[int, int]:
    """``(start, stop)`` (stop exclusive) of the primal nodes inside ``[lo, hi]``.

    Both bounds must be nodes (to 1e-9 of the span): ``scene.domain.bounds`` are node
    planes on every builder grid, and a periodic or Bloch axis keeps its endpoint node,
    the wrap plane, which is the image of the first.
    """
    import numpy as np

    values = np.asarray(nodes, dtype=np.float64)
    tolerance = 1e-9 * (hi - lo)
    inside = np.flatnonzero((values >= lo - tolerance) & (values <= hi + tolerance))
    if inside.size < 2:
        raise ValueError(f"fewer than two primal nodes inside [{lo}, {hi}]")
    start, stop = int(inside[0]), int(inside[-1]) + 1
    if stop - start != inside.size:
        raise ValueError("primal nodes inside the domain are not contiguous")
    if abs(values[start] - lo) > tolerance or abs(values[stop - 1] - hi) > tolerance:
        raise ValueError(f"domain bounds [{lo}, {hi}] are not primal nodes "
                         f"(nearest inside: {values[start]}, {values[stop - 1]})")
    return start, stop


def component_slices(starts, stops, kind="electric") -> tuple[tuple[slice, slice, slice], ...]:
    """Per component, the spatial slices of the node window ``[start, stop)`` per axis.

    A cell sample keeps the cells between kept nodes, ``start ... stop - 2``; a node sample
    keeps ``start ... stop - 1``. E is a cell sample along its own axis and a node sample on
    the other two; H (``kind="magnetic"``) is the opposite.
    """
    electric = {"electric": True, "magnetic": False}[kind]
    return tuple(tuple(slice(start, stop - ((axis == component) == electric))
                       for axis, (start, stop) in enumerate(zip(starts, stops)))
                 for component in range(3))


def _json(value, path="metadata"):
    """Strict-JSON form of a run record: tuples become lists, non-finite floats ``None``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return [_json(value.real), _json(value.imag)]
    if isinstance(value, dict):
        return {str(key): _json(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item, f"{path}[]") for item in value]
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:  # NumPy / torch scalars
        return _json(value.item(), path)
    if hasattr(value, "tolist"):
        return _json(value.tolist(), path)
    raise TypeError(f"{path}: {type(value).__name__} is not JSON metadata")


def boundary_kinds(face_kinds, absorber) -> dict[str, list[str]]:
    """Per axis ``[low, high]`` in the record vocabulary.

    maxwell's ``BoundaryKind`` maps one to one except ``pml``, which becomes ``cpml``
    when the run's resolved absorber (``stats()["absorber"]``) is CPML and ``pml`` for
    the other PML-family absorbers. ``none`` (and anything else) is refused. Symmetry
    faces never reach this point: such scenes are refused (``refuse_unsupported``).
    """
    kinds = {}
    for axis, faces in zip(AXES, face_kinds):
        kinds[axis] = []
        for side, face in zip(("low", "high"), faces):
            kind = str(face)
            if kind == "pml":
                kind = "cpml" if str(absorber) == "cpml" else "pml"
            if kind not in BOUNDARY_KINDS:
                raise ValueError(f"boundary kind {kind!r} on the {axis} {side} face is outside the record "
                                 f"vocabulary {BOUNDARY_KINDS}")
            kinds[axis].append(kind)
    return kinds


def h_storage_required(h_reasons, grid=None) -> str | None:
    """Amendment A5: why this sample must store H as well (``None``: E suffices).

    H is stored whenever it cannot be derived from E within the record's fidelity: no H
    derivation (``h_reasons``), or a graded axis (``grid["spacing_m"]`` has a null).
    ``grid=None`` checks only what the scene alone decides, before the run.
    """
    if h_reasons:
        return f"no H derivation ({'; '.join(h_reasons)})"
    if grid is not None and any(width is None for width in grid["spacing_m"]):
        return "graded_grid"
    return None


def refuse_h_storage(reason):
    """Refuse a sample that must store H when no volume H phasors can be had.

    Raised before the run when maxwell's ``FDTDConfig`` has no ``volume_dft_components``
    switch, and after the run when the sample demands H but ``Result.fields`` holds no
    HX/HY/HZ, rather than writing an E-only record that violates A5.
    """
    if reason is not None:
        raise ValueError(f"--format emfc must store H for this sample (amendment A5: {reason}), but this "
                         "maxwell records no volume H DFT; the sample is refused rather than stored E-only")


def scene_h_storage(scene) -> str | None:
    """A5 decided before the run, from the scene: why H will be needed (``None``: it will not).

    Exact for ``GridSpec.custom`` (graded axis inside the domain) and uniform or anisotropic
    grids; a ``GridSpec.auto`` grid may come out graded, so it asks for H conservatively. The
    record itself stores H only when the rule holds on the grid that ran.
    """
    import numpy as np

    reason = h_storage_required(h_derivation_reasons(scene))
    if reason is not None:
        return reason
    grid = scene.grid
    if grid.x_coords is not None:
        cropped = []
        for coords, (lo, hi) in zip((grid.x_coords, grid.y_coords, grid.z_coords), scene.domain.bounds):
            start, stop = crop_window(coords, float(lo), float(hi))
            cropped.append(np.asarray(coords[start:stop], dtype=np.float64))
        graded = grid_record([len(c) for c in cropped], (0, 0, 0), [len(c) for c in cropped], cropped)[0]
        return h_storage_required([], graded)
    if grid.dx is not None:
        return None
    return "graded_grid (GridSpec.auto may grade the mesh)"


def h_dft_config(config):
    """``config`` recording the volume H DFT too, or ``None`` on a maxwell without the switch."""
    from dataclasses import fields, replace

    if "volume_dft_components" not in {field.name for field in fields(config)}:
        return None
    return replace(config, full_field_dft=True,
                   volume_dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))


def admit_sample(sample):
    """Before the run: refuse what v1 cannot describe, and request the H DFT when A5 needs H."""
    refuse_unsupported(sample.scene)
    reason = scene_h_storage(sample.scene)
    if reason is None:
        return
    config = h_dft_config(sample.fdtd_kwargs["config"])
    if config is None:
        refuse_h_storage(reason)
    sample.fdtd_kwargs = dict(sample.fdtd_kwargs, config=config)


def h_derivation(reasons, mu_0):
    """``(h_derivation, reasons)``: the H-from-E recipe, or ``None`` when ``reasons`` say H does not follow."""
    if reasons:
        return None, list(reasons)
    return {"method": "h_from_e", "mu_h_per_m": float(mu_0), "sigma_m": 0, "excluded": "tfsf_h_shell"}, None


def grid_record(full_shape_nodes, starts, stops, cropped_nodes):
    """``maxwell_record.grid`` and the reference spacing ``h`` (the smallest cropped interval).

    An axis is uniform when each of its cropped cell widths equals its smallest width to 1e-9
    relative; ``spacing_m`` holds that width per axis (``None`` for a graded axis) and
    ``uniform`` is true when every axis is uniform (axes may differ). ``h`` is the scale of the
    stored node arrays (``nodes_m / h``), never the grid itself.
    """
    import numpy as np

    diffs = [np.diff(np.asarray(nodes, dtype=np.float64)) for nodes in cropped_nodes]
    widths = [float(d.min()) for d in diffs]
    per_axis = [width if bool(np.all(np.abs(d - width) <= 1e-9 * width)) else None
                for d, width in zip(diffs, widths)]
    spacing = min(widths)
    return {"full_shape_nodes": [int(n) for n in full_shape_nodes],
            "crop_node_start": [int(s) for s in starts], "crop_node_stop": [int(s) for s in stops],
            "uniform": all(width is not None for width in per_axis), "spacing_m": per_axis,
            "reference_spacing_m": spacing}, spacing


def record_metadata(*, provenance, stats, field_attrs, frequencies, boundary, grid, h_reasons, mu_0,
                    fdtd_config, audit, options, h_field_attrs=None):
    """The record metadata: provenance plus ``maxwell_record`` (schema maxwell-datagen-record/1)."""
    ratio = [float(r) for r in stats["last_step_field_ratio"]]
    if not all(math.isfinite(r) for r in ratio):
        # A non-finite ratio means the DFT sum vanished at that frequency: the sample is useless.
        raise ValueError(f"last_step_field_ratio {ratio} is not finite (zero DFT sum at some frequency "
                         f"{list(frequencies)} Hz); no record is written for this sample")
    periodic = [axis for axis in AXES if all(kind in ("periodic", "bloch") for kind in boundary[axis])]
    recipe, unavailable = h_derivation(h_reasons, mu_0)
    h_stored = h_field_attrs is not None
    tfsf = stats.get("tfsf")
    bloch_k, bloch_phase = stats.get("bloch_wavevector"), stats.get("bloch_phase")
    field_names = [f"E/f{index:03d}" for index in range(len(frequencies))]
    h_field_names = [f"H/f{index:03d}" for index in range(len(frequencies))]
    return {**_json(provenance), "maxwell_record": {
        "schema": SCHEMA,
        "quantity": "E+H" if h_stored else "E",
        "h_stored": h_stored,
        "field_names": field_names,
        **({"h_field_names": h_field_names, "h_stored_source": "maxwell_volume_dft"} if h_stored else {}),
        "field_frequencies_hz": [float(f) for f in frequencies],
        "dt_s": float(stats["dt"]),
        "steps_run": int(stats["steps_run"]),
        "field_time_offset_steps": float(field_attrs["field_time_offset_steps"]),
        # H's own sample time label, as maxwell reports it on the H FieldData.
        **({"h_field_time_offset_steps": float(h_field_attrs["field_time_offset_steps"])} if h_stored else {}),
        "phasor_convention": field_attrs["phasor_convention"],
        "normalization": field_attrs["normalization"],
        "dft_window": stats["dft_window"],
        "dft_window_requested": stats["dft_window_requested"],
        "last_step_field_ratio": ratio,
        "grid": dict(grid),
        "boundary": {axis: list(boundary[axis]) for axis in AXES},
        "periodic_axes": periodic,
        "bloch_wavevector": None if bloch_k is None else [float(k) for k in bloch_k],
        "bloch_phase": None if bloch_phase is None else [[float(complex(p).real), float(complex(p).imag)]
                                                          for p in bloch_phase],
        "tfsf": None if tfsf is None else _json({key: tfsf[key] for key in
                                                 ("mode", "axis", "node_indices", "h_corrections")}),
        "h_derivation": recipe,
        **({} if unavailable is None else {"h_derivation_unavailable": unavailable}),
        # A graded grid alone keeps the H-from-E recipe; H is stored because the recipe's error grows there.
        **({"h_stored_reason": "graded_grid"} if h_stored and recipe is not None else {}),
        "fdtd_config": fdtd_config,
        "field_fidelity": {"mode": "lossless" if audit else "tolerance",
                           "tolerance": None if audit else float(options.field_tolerance),
                           # H defaults to the E tolerance (user decision); null without H or when lossless.
                           "h_tolerance": None if audit or not h_stored else float(options.h_field_tolerance),
                           "audit": bool(audit), "audit_rule": AUDIT_RULE,
                           "audit_fraction": float(options.audit_fraction),
                           "dataset_seed": int(options.dataset_seed)},
    }}


# --- Scene and run facts (maxwell, no emfc) -----------------------------------------------------
def refuse_unsupported(scene):
    """Raise for scenes the v1 record cannot describe correctly (checked before the run)."""
    if scene.has_symmetry:
        raise ValueError("--format emfc (maxwell-datagen-record/1) refuses scenes with symmetry planes: the "
                         "maxwell PlaneWave normalisation over folded axes is an open defect; run the full domain")


def h_derivation_reasons(scene) -> list[str]:
    """Why H would not follow from the stored E by the plain discrete Faraday step (empty: it does).

    H follows from E with mu_0 and no magnetic loss unless a material is magnetic (mu_r, a
    mu tensor, magnetic dispersion, sigma_m, a gyromagnetic ferrite, PMC), a sheet model
    corrects H (``copper_foil_h_correction`` for CopperFoil; the conformal thin-sheet runtime
    replaces H rows of cut faces), or a source or port injects into the H update other than
    through the TFSF shell. Allowed injections:
    electric point dipoles, TFSF plane waves and beams, lumped/terminal ports, thin wires
    and lumped elements (all update E only); anything else is listed, conservatively.
    """
    import witwin.maxwell as mw
    from witwin.maxwell.compiler.thin_sheet import owns_thin_sheet

    reasons = []
    for index, structure in enumerate(scene.structures):
        if isinstance(structure.material, mw.media.CopperFoil):
            if "copper_foil_h_correction" not in reasons:
                reasons.append("copper_foil_h_correction")
        elif owns_thin_sheet(structure):
            reasons.append(f"structure {structure.name or index}: conformal thin sheet replaces H rows")
    materials = [("background", scene.background)]
    materials += [(f"structure {structure.name or index}", structure.material)
                  for index, structure in enumerate(scene.structures)]
    for label, material in materials:
        if material is None or not isinstance(material, mw.Material):
            continue
        if isinstance(material, mw.media.GyromagneticFerrite):
            reasons.append(f"{label}: gyromagnetic ferrite")
        if float(material.mu_r) != 1.0 or material.mu_tensor is not None:
            reasons.append(f"{label}: mu_r != 1")
        if material.is_magnetic_dispersive or material.has_custom_magnetic_poles:
            reasons.append(f"{label}: magnetic dispersion")
        if float(material.sigma_m) != 0.0:
            reasons.append(f"{label}: sigma_m != 0")
        if material.pmc:
            reasons.append(f"{label}: PMC material")
    for region in scene.material_regions:
        if tuple(region.mu_bounds) != (1.0, 1.0):
            reasons.append(f"material region {region.name}: mu_bounds {tuple(region.mu_bounds)}")
    for index, source in enumerate(scene.sources):
        tfsf = isinstance(getattr(source, "injection", None), mw.TFSF)
        if not (isinstance(source, mw.PointDipole)
                or (tfsf and isinstance(source, (mw.PlaneWave, mw.GaussianBeam, mw.sources.AstigmaticGaussianBeam)))):
            reasons.append(f"source {getattr(source, 'name', None) or index} ({type(source).__name__}"
                           f"{'' if tfsf else ', not TFSF'}) may inject into the H update")
    for port in scene.ports:
        if not isinstance(port, (mw.LumpedPort, mw.rf.TerminalPort, mw.rf.CoaxialLumpedPort)):
            reasons.append(f"port {port.name} ({type(port).__name__}) injects modal fields")
    return reasons


def _snapshot(value):
    """JSON form of ``dataclasses.asdict(config)``: enums by value, other objects by ``str``, or by
    their type name when ``str`` would show an object address (amendment A9.3)."""
    if isinstance(value, enum.Enum):
        return _snapshot(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value))
    if isinstance(value, dict):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    text = str(value)
    return type(value).__name__ if re.search(r" at 0x[0-9a-fA-F]+", text) else text


def fdtd_config(builder, config) -> dict:
    """``maxwell_record.fdtd_config`` (amendment A3): maxwell has no public config serializer, so
    the builder's own keyword arguments are authoritative and the resolved ``FDTDConfig`` is a
    snapshot for consistency checks only."""
    build = builder.build
    if getattr(build, "args", None) != (builder.name,):
        raise ValueError(f"builder {builder.name!r} is not partial(function, name, **kwargs)")
    return {"source": "builder_kwargs", "builder": builder.name, "kwargs": _json(dict(build.keywords)),
            "config_snapshot": _snapshot(asdict(config)), "authoritative": "builder_kwargs"}


def witwin_core_tree_sha256(package_dir) -> str:
    """Amendment A8: identity of the installed witwin-core package bytes.

    sha256 of the UTF-8 lines ``"<posix relative path>\0<hex file sha256>\n"``, sorted bytewise
    by relative path, over every regular file under the package directory whose name ends in
    ``.py``, ``.so`` or ``.pyd`` (compiled extensions included), skipping any path with a
    ``__pycache__`` component and ``*.pyc``. Symlinks are followed.
    """
    root = Path(package_dir)
    lines = []
    for directory, subdirectories, files in os.walk(root, followlinks=True):
        subdirectories[:] = [name for name in subdirectories if name != "__pycache__"]
        for name in files:
            path = Path(directory) / name
            if not name.endswith((".py", ".so", ".pyd")) or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            lines.append((relative.encode(), f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n"))
    return hashlib.sha256("".join(line for _, line in sorted(lines)).encode()).hexdigest()


def software_provenance() -> dict:
    """maxwell version and the witwin-core version, tree sha256 and commit (null with a reason when unknown)."""
    import importlib.metadata
    import subprocess
    import witwin.core
    import witwin.maxwell as mw

    commit, reason = None, None
    try:
        direct = json.loads(importlib.metadata.distribution("witwin").read_text("direct_url.json") or "null")
    except importlib.metadata.PackageNotFoundError:
        direct = None
    vcs = (direct or {}).get("vcs_info") or {}
    if vcs.get("commit_id"):
        commit = vcs["commit_id"]
    else:
        where = Path(witwin.core.__file__).resolve().parent
        # Only a checkout that tracks this very file counts, not some repository around site-packages.
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "__init__.py"], cwd=where,
                                 capture_output=True, text=True)
        probe = subprocess.run(["git", "rev-parse", "HEAD"], cwd=where, capture_output=True, text=True)
        if tracked.returncode == 0 and probe.returncode == 0:
            commit = probe.stdout.strip()
        else:
            reason = (f"witwin {getattr(witwin.core, '__version__', '?')} at {where} is not a git checkout and its "
                      f"install records no VCS commit (direct_url: {(direct or {}).get('url')})")
    record = {"maxwell_version": mw.__version__, "witwin_core_version": getattr(witwin.core, "__version__", None),
              "witwin_core_tree_sha256": witwin_core_tree_sha256(Path(witwin.core.__file__).parent),
              "witwin_core_commit": commit}
    if reason is not None:
        record["witwin_core_commit_reason"] = reason
    return record


# --- emfc-facing steps ------------------------------------------------------------------------
def require_emfc():
    """Import emfc and check it has the record and shard layers (emfc >= 0.2.0)."""
    try:
        import emfc
        import emfc.shard
    except ImportError as error:
        raise RuntimeError("--format emfc requires the emfc package (>= 0.2.0) to be importable") from error
    missing = [name for name in ("encode_record", "encode_staggered", "StaggeredData")
               if not hasattr(emfc, name)]
    if missing or not hasattr(emfc.shard, "ShardWriter"):
        raise RuntimeError(f"emfc {getattr(emfc, '__version__', '?')} at {emfc.__file__} has no record "
                           f"containers ({', '.join(missing) or 'emfc.shard'}); --format emfc needs >= 0.2.0")
    return emfc


def domain_window(result):
    """``(full primal nodes per axis, starts, stops)`` of ``scene.domain.bounds`` in ``Result.fields``."""
    import torch

    data = [result.fields[name] for name in E_NAMES]
    nodes = []
    for axis, name in enumerate(AXES):
        transverse = [data[c].coords[name] for c in range(3) if c != axis]
        if transverse[0].dtype != torch.float64 or not torch.equal(*transverse):
            raise ValueError(f"E node coordinates along {name} disagree between components")
        nodes.append(transverse[0].detach().cpu())
    window = [crop_window(axis_nodes.numpy(), *map(float, result.scene.domain.bounds[axis]))
              for axis, axis_nodes in enumerate(nodes)]
    return nodes, tuple(w[0] for w in window), tuple(w[1] for w in window)


def encode_fields(result, starts, stops, cropped, spacing, options_for_fields, kind="electric"):
    """``{"E/fNNN": container bytes}`` (``{"H/fNNN": ...}`` for ``kind="magnetic"``), one
    frequency cropped and encoded at a time.

    Pre-encoding bounds the extra device memory to one frequency's crop; the record then
    embeds each composite verbatim, which is byte-identical to encoding the field objects
    inside ``encode_record`` with the same options.
    """
    import torch
    import emfc

    quantity, names = ("E", E_NAMES) if kind == "electric" else ("H", H_NAMES)
    data = [result.fields[name] for name in names]
    for item in data:
        if item.values.dtype != torch.complex64:
            raise TypeError(f"stored {quantity} must be complex64 (got {item.values.dtype}); no silent narrowing")
    device = data[0].values.device
    grid = tuple((axis_nodes / spacing).to(device=device, dtype=torch.float64) for axis_nodes in cropped)
    slices = component_slices(starts, stops, kind)
    encoded = {}
    for index in range(data[0].values.shape[0]):
        components = tuple(item.values[(index, *slices[c])].contiguous() for c, item in enumerate(data))
        field = emfc.StaggeredData(components, grid, kind, reference_spacing_m=spacing)
        encoded[f"{quantity}/f{index:03d}"] = emfc.encode_staggered(field, **options_for_fields)
        del field, components
    return encoded


def scene_payload(scene, eps_r, grid):
    """The record scene (amendment A9.4): the builder's Scene as a witwin.scene/2 document.

    emfc's ``SceneData`` needs a material field beside the document; this is the compiled
    relative permittivity cropped to the window's cells (``Result.material("eps_r")`` on the
    full cells, or on the full nodes then cropped to the window's nodes), stored bit-exact.
    """
    import emfc

    full, start, stop = grid["full_shape_nodes"], grid["crop_node_start"], grid["crop_node_stop"]
    shape = tuple(int(n) for n in eps_r.shape)
    if shape == tuple(n - 1 for n in full):
        window = eps_r[tuple(slice(s, e - 1) for s, e in zip(start, stop))]
    elif shape == tuple(full):
        window = eps_r[tuple(slice(s, e) for s, e in zip(start, stop))]
    else:
        raise ValueError(f"Result.material('eps_r') has shape {shape}, neither the full cells nor the "
                         f"full nodes of {tuple(full)}")
    return emfc.SceneData(eps_r=window.detach().cpu().contiguous(), definition=scene)


def encode_sample(result, readouts, provenance, *, sample_id, options, fdtd_config, scene=None, eps_r=None):
    """One complete record container for a finished run whose solver is already released.

    ``fdtd_config`` is the ``maxwell_record.fdtd_config`` block; ``scene`` (the builder's
    Scene) and ``eps_r`` (``Result.material("eps_r")`` read before ``release_solver``) give
    the record scene. Returns ``(container, audit)``. Raises if any field certificate cannot
    be met; the caller appends nothing then.
    """
    import torch
    import emfc
    from witwin.maxwell.constants import MU_0

    audit = audit_selected(options.dataset_seed, sample_id, options.audit_fraction)
    nodes, starts, stops = domain_window(result)
    cropped = [axis_nodes[start:stop].clone() for axis_nodes, start, stop in zip(nodes, starts, stops)]
    grid, spacing = grid_record([len(n) for n in nodes], starts, stops, [n.numpy() for n in cropped])
    h_reasons = h_derivation_reasons(result.scene)
    store_h = h_storage_required(h_reasons, grid)
    if store_h is not None and not all(name in result.fields for name in H_NAMES):
        refuse_h_storage(store_h)
    ex = result.fields["EX"]
    frequencies = ex.coords["f"].detach().to("cpu", torch.float64)
    stats = result.stats()
    metadata = record_metadata(
        provenance=provenance, stats=stats, field_attrs=ex.attrs, frequencies=frequencies.tolist(),
        boundary=boundary_kinds(result.scene.boundary.face_kinds, stats["absorber"]), grid=grid,
        h_reasons=h_reasons, mu_0=MU_0, fdtd_config=fdtd_config,
        audit=audit, options=options,
        h_field_attrs=None if store_h is None else result.fields["HX"].attrs,
    )
    fields = encode_fields(result, starts, stops, cropped, spacing,
                           {} if audit else {"tolerance": options.field_tolerance})
    if store_h is not None:
        fields.update(encode_fields(result, starts, stops, cropped, spacing,
                                    {} if audit else {"tolerance": options.h_field_tolerance}, kind="magnetic"))
    arrays = dict(readouts)
    reserved = {"_field_frequencies_hz", *(f"_grid/nodes_m/{axis}" for axis in AXES)}
    if reserved & set(arrays):
        raise ValueError(f"readout names {sorted(reserved & set(arrays))} are reserved for the record")
    arrays["_field_frequencies_hz"] = frequencies
    arrays.update({f"_grid/nodes_m/{axis}": axis_nodes for axis, axis_nodes in zip(AXES, cropped)})
    payload = None if scene is None else scene_payload(scene, eps_r, grid)
    return emfc.encode_record(arrays, fields=fields, scene=payload, metadata=metadata), audit


def labels(task, family, audit):
    return {"family": family, "builder": task.builder, "seed": int(task.seed),
            "task_id": task.task_id, "audit_lossless": int(audit)}


def open_writer(output_dir, rank, options):
    import emfc.shard

    return emfc.shard.ShardWriter(Path(output_dir) / SHARDS, rank=rank, resume=True,
                                  dataset_uuid=options.dataset_uuid, target_bytes=options.shard_target_bytes)


# --- Parent: dataset identity, resume and finish ---------------------------------------------
def _write_json_atomic(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _live(path: Path) -> bool:
    """Whether a live writer holds ``path`` (the shard writer's flock claim)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        return error.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)
    finally:
        os.close(fd)
    return False


def prepare_dataset(output_dir, tasks, options: EmfcOptions):
    """Adopt or create the dataset and return ``(options with dataset_uuid, pending task indices, info)``.

    A first run writes ``emfc_dataset.json`` (uuid and settings). A resumed run must
    match its settings; ``.emfcs.tmp`` files a crash left are moved to ``shards-aside/``
    (their tasks run again; ``recover`` would assign fresh sample ids and break
    ``sample_id == task index``), and the sealed sidecars give the sample ids present.
    """
    emfc = require_emfc()
    output_dir = Path(output_dir)
    shards = output_dir / SHARDS
    dataset_file = output_dir / DATASET_FILE
    settings = options.settings()
    if dataset_file.exists():
        stored = json.loads(dataset_file.read_text())
        changed = {key: (stored.get(key), value) for key, value in settings.items() if stored.get(key) != value}
        if changed:
            raise ValueError(f"{dataset_file} was written with other settings; resume needs the same "
                             f"(stored, requested): {changed}")
        if options.dataset_uuid is not None and uuid.UUID(options.dataset_uuid).hex != stored["dataset_uuid"]:
            raise ValueError(f"{dataset_file} holds dataset {stored['dataset_uuid']}, not {options.dataset_uuid}")
        dataset_uuid = stored["dataset_uuid"]
    else:
        if shards.exists() and any(shards.iterdir()):
            raise ValueError(f"{shards} holds files but {dataset_file} is missing; not a datagen emfc dataset")
        # The 32-hex form, as emfc.shard reports dataset uuids.
        dataset_uuid = uuid.UUID(options.dataset_uuid).hex if options.dataset_uuid else uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(dataset_file, dict(settings, dataset_uuid=dataset_uuid, emfc_version=emfc.__version__))
    shards.mkdir(parents=True, exist_ok=True)

    moved = []
    for temporary in sorted(shards.glob("*.emfcs.tmp")):
        if _live(temporary):
            raise RuntimeError(f"{temporary} is held by a live writer; another run is using {output_dir}")
        aside = output_dir / ASIDE
        aside.mkdir(exist_ok=True)
        target = aside / f"{temporary.name}.{time.strftime('%Y%m%dT%H%M%S')}"
        os.replace(temporary, target)
        moved.append(target.name)

    present = set()
    if any(shards.glob("*.emfcs.json")):
        index = emfc.shard.index(shards)
        if index.dataset_uuid is not None and uuid.UUID(index.dataset_uuid).hex != dataset_uuid:
            raise ValueError(f"{shards} holds dataset {index.dataset_uuid}, not {dataset_uuid}")
        present = set(index.where())
        foreign = sorted(sid for sid in present if sid >= len(tasks))
        if foreign:
            raise ValueError(f"{shards} holds sample ids past the task list ({foreign[:5]} ...); wrong --tasks?")
        for builder in sorted({tasks[sid].builder for sid in present}):
            wrong = set(index.where(builder=builder)) ^ {sid for sid in present if tasks[sid].builder == builder}
            if wrong:
                raise ValueError(f"stored records disagree with --tasks about builder {builder!r} at sample ids "
                                 f"{sorted(wrong)[:5]}; wrong --tasks?")
        index.close()
    pending = [sample_id for sample_id in range(len(tasks)) if sample_id not in present]
    resolved = EmfcOptions(**{**asdict(options), "dataset_uuid": dataset_uuid})
    return resolved, pending, dict(present=len(present), moved_aside=moved)


def finish_dataset(output_dir, options, records, sample_ids, info):
    """Index the shards; an ``ok`` task whose record is not in the index becomes an error.

    Returns the ``emfc`` block of manifest.json.
    """
    emfc = require_emfc()
    shards = Path(output_dir) / SHARDS
    present, audited, index_info = set(), 0, {}
    if any(shards.glob("*.emfcs.json")):
        index = emfc.shard.index(shards)
        present = set(index.where())
        audited = len(index.where(audit_lossless=1))
        index_info = dict(record_count=len(index), shard_count=index.shard_count,
                          sample_id_min=index.sample_id_min, sample_id_max=index.sample_id_max,
                          total_bytes=index.total_bytes)
        index.close()
    for task_id, record in records.items():
        if record["status"] == "ok" and sample_ids[task_id] not in present:
            path = Path(output_dir) / f"{task_id}.error.txt"
            path.write_text(f"record for sample id {sample_ids[task_id]} is missing from the sealed shards\n")
            record.update(status="error", path=str(path))
    return dict(options.settings(), dataset_uuid=options.dataset_uuid, shards=str(shards),
                record_count=index_info.get("record_count", 0), audit_records=audited,
                skipped_present=info["present"], moved_aside=info["moved_aside"],
                emfc_version=emfc.__version__, **{k: v for k, v in index_info.items() if k != "record_count"})
