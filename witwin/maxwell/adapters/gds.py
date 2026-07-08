"""GDS layout import/export for extruded polygon geometry.

Usage::

    slabs = from_gds("layout.gds", layer=1, bounds=(0.0, 220e-9))
    to_gds_file(slabs, "layout_out.gds")

Requires ``gdstk`` as an optional dependency.

**Unit convention**: maxwell uses metres; GDS libraries carry an explicit
user unit in metres (``library.unit``, typically 1e-6 for micrometres).
``from_gds`` multiplies layout coordinates by ``length_scale``, defaulting
to the file's ``library.unit`` so imported vertices land in metres.
``to_gds_file`` multiplies metre coordinates by ``length_scale`` (default
1e6) and writes the library with ``unit = 1 / length_scale`` so user units
always mean what ``length_scale`` says.

Only the 2D cross-section travels through GDS: extrusion ``bounds``,
``axis``, ``sidewall_angle``, and ``reference_plane`` are supplied by the
caller on import and dropped on export.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from witwin.core.geometry import PolySlab

if TYPE_CHECKING:
    import gdstk

# Default length conversion factor for export: metres to micrometres.
_M_TO_UM = 1e6

# In-plane world-coordinate indices per extrusion axis (matches _axial_split:
# axis z -> (x, y), y -> (x, z), x -> (y, z)).
_PLANE_INDICES = {"z": (0, 1), "y": (0, 2), "x": (1, 2)}


def _ensure_gdstk():
    """Import and return gdstk, raising a clear error if missing."""
    try:
        import gdstk
        return gdstk
    except ImportError:
        raise ImportError(
            "gdstk is required for GDS import/export. "
            "Install it with: pip install gdstk"
        ) from None


def _resolve_cell(library, cell: str | None):
    """Return the requested cell, or the unique top-level cell when *cell* is None."""
    names = [candidate.name for candidate in library.cells]
    if cell is None:
        top = library.top_level()
        if len(top) != 1:
            raise ValueError(
                f"GDS library has {len(top)} top-level cells; pass cell= to pick one. "
                f"Available cells: {names}"
            )
        return top[0]
    for candidate in library.cells:
        if candidate.name == cell:
            return candidate
    raise ValueError(f"Cell '{cell}' not found in GDS library. Available cells: {names}")


def from_gds(
    gds_path,
    *,
    layer: int,
    bounds,
    datatype: int = 0,
    cell: str | None = None,
    axis="z",
    sidewall_angle=0.0,
    reference_plane: str = "middle",
    length_scale: float | None = None,
) -> list[PolySlab]:
    """Load polygons from a GDS file as extruded ``PolySlab`` geometries.

    Cell references are flattened recursively (with repetitions applied) and
    paths are converted to their polygonal representation, so the result covers
    the full selected layer of the resolved cell. GDS polygons are always
    simple boundary loops (the format has no hole concept), so each polygon
    becomes one plain ``PolySlab``; holes drawn as separate loops import as
    separate slabs.

    Args:
        gds_path: Path to the ``.gds`` file.
        layer: GDS layer number to import.
        bounds: ``(min, max)`` extrusion extent along *axis*, in metres.
        datatype: GDS datatype number to import.
        cell: Cell name to read. ``None`` selects the unique top-level cell.
        axis: Extrusion axis (``'x'``/``'y'``/``'z'`` or 0/1/2).
        sidewall_angle: ``PolySlab`` sidewall angle in radians.
        reference_plane: ``PolySlab`` reference plane
            (``'bottom'``/``'middle'``/``'top'``).
        length_scale: Metres per GDS user unit. ``None`` uses the file's
            ``library.unit`` (typically 1e-6).

    Returns:
        One ``PolySlab`` per polygon on ``(layer, datatype)``.
    """
    gdstk = _ensure_gdstk()
    path = Path(gds_path)
    if not path.is_file():
        raise ValueError(f"GDS file not found: {path}")
    library = gdstk.read_gds(str(path))
    selected = _resolve_cell(library, cell)
    scale = float(library.unit) if length_scale is None else float(length_scale)
    polygons = selected.get_polygons(
        include_paths=True, depth=None, layer=int(layer), datatype=int(datatype)
    )
    if not polygons:
        raise ValueError(
            f"No polygons on layer ({int(layer)}, {int(datatype)}) in cell '{selected.name}'."
        )
    return [
        PolySlab(
            torch.from_numpy(np.asarray(polygon.points, dtype=np.float64) * scale),
            bounds,
            axis=axis,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
        )
        for polygon in polygons
    ]


def to_gds_file(
    geometries,
    gds_path,
    *,
    cell_name: str = "TOP",
    layer: int = 0,
    datatype: int = 0,
    length_scale: float = _M_TO_UM,
):
    """Write ``PolySlab`` cross-sections to a GDS file.

    *geometries* is a single ``PolySlab`` / ``ComplexPolySlab`` or an iterable
    of them, written to ``(layer, datatype)`` with the in-plane component of
    ``position`` applied. GDS boundaries on one layer union, so multi-loop
    geometry is folded with even-odd XOR into simple keyhole-cut polygons that
    preserve ``ComplexPolySlab`` hole semantics. Extrusion ``bounds``,
    ``sidewall_angle``, and ``reference_plane`` have no GDS representation and
    are dropped: the reference-plane cross-section is written. Geometries with
    a non-identity ``rotation`` are rejected.

    Args:
        geometries: ``PolySlab`` geometry or iterable of them.
        gds_path: Output ``.gds`` file path.
        cell_name: Name of the written cell.
        layer: GDS layer number for all polygons.
        datatype: GDS datatype number for all polygons.
        length_scale: GDS user units per metre (default 1e6, i.e. micrometre
            user units; the library is written with ``unit = 1 / length_scale``).
    """
    gdstk = _ensure_gdstk()
    if isinstance(geometries, PolySlab):
        geometries = [geometries]
    # Database grid: the GDS-conventional 1e-3 user units, clamped to at most
    # 1 nm physical so coarse user units cannot silently snap coordinates away.
    unit = 1.0 / float(length_scale)
    library = gdstk.Library(unit=unit, precision=min(unit * 1e-3, 1e-9))
    cell = library.new_cell(cell_name)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for geometry in geometries:
        if not isinstance(geometry, PolySlab):
            raise ValueError(
                "Only extruded-polygon geometry (PolySlab / ComplexPolySlab) maps to "
                f"GDS polygons; got {type(geometry).__name__}."
            )
        if not torch.allclose(geometry.rotation.detach().cpu(), identity):
            raise ValueError("Cannot export rotated PolySlab geometry to GDS.")
        offset = geometry.position.detach().cpu().to(torch.float64).numpy()
        offset = offset[list(_PLANE_INDICES[geometry.axis])]
        vertices = geometry.vertices.detach().cpu().to(torch.float64).numpy() + offset
        loops = []
        start = 0
        for size in geometry.loop_sizes:
            loops.append(vertices[start:start + size] * float(length_scale))
            start += size
        if len(loops) == 1:
            cell.add(gdstk.Polygon(loops[0], layer=int(layer), datatype=int(datatype)))
        else:
            # GDS boundaries on one layer union, so reproduce the even-odd rule
            # by folding XOR over the loops: holes become keyhole-cut simple
            # polygons instead of solid boundaries.
            merged = [loops[0]]
            for loop in loops[1:]:
                merged = gdstk.boolean(
                    merged, [loop], "xor", layer=int(layer), datatype=int(datatype)
                )
            for polygon in merged:
                cell.add(polygon)
    library.write_gds(str(gds_path))
