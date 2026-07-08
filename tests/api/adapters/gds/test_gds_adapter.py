"""Tests for the GDS adapter (maxwell/adapters/gds.py).

Most tests exercise real gdstk write/read round-trips, so they carry the
``requires_gdstk`` skip marker for environments without the optional gdstk
dependency. The missing-dependency test simulates an absent gdstk via
sys.modules and always runs.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.adapters.gds import from_gds, to_gds_file
from witwin.maxwell.scene import prepare_scene

try:
    import gdstk
except ImportError:
    gdstk = None

requires_gdstk = pytest.mark.skipif(gdstk is None, reason="gdstk is not installed")

UM = 1e-6
SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
TRIANGLE = [(3.0, 0.0), (4.0, 0.0), (3.5, 1.5)]
PENTAGON = [(0.0, 2.0), (1.0, 2.0), (1.5, 3.0), (0.5, 3.5), (-0.5, 3.0)]


@pytest.fixture
def gds_file(tmp_path):
    """Write a test library: two polygons on layer (1, 0), one on (2, 0)."""
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.Polygon(SQUARE, layer=1, datatype=0))
    cell.add(gdstk.Polygon(TRIANGLE, layer=1, datatype=0))
    cell.add(gdstk.Polygon(PENTAGON, layer=2, datatype=0))
    path = tmp_path / "test.gds"
    library.write_gds(str(path))
    return path


@requires_gdstk
def test_from_gds_layer_one_returns_two_polyslabs(gds_file):
    slabs = from_gds(
        gds_file,
        layer=1,
        bounds=(0.0, 0.22 * UM),
        sidewall_angle=0.1,
        reference_plane="bottom",
    )

    assert len(slabs) == 2
    expected = {len(SQUARE): np.asarray(SQUARE), len(TRIANGLE): np.asarray(TRIANGLE)}
    for slab in slabs:
        assert type(slab) is mw.PolySlab
        assert slab.axis == "z"
        assert slab.reference_plane == "bottom"
        assert slab.sidewall_angle.item() == pytest.approx(0.1)
        assert torch.allclose(slab.bounds, torch.tensor([0.0, 0.22 * UM]))
        # Default length_scale converts the file's micron user unit to metres.
        reference = expected[slab.vertices.shape[0]] * UM
        assert np.allclose(slab.vertices.numpy(), reference, atol=1e-11)


@requires_gdstk
def test_layer_datatype_selection(gds_file):
    slabs = from_gds(gds_file, layer=2, bounds=(0.0, UM), axis="x")
    assert len(slabs) == 1
    assert slabs[0].axis == "x"
    assert slabs[0].vertices.shape[0] == len(PENTAGON)

    # Explicit length_scale override keeps raw layout user units.
    raw = from_gds(gds_file, layer=2, bounds=(0.0, 1.0), length_scale=1.0)
    assert np.allclose(raw[0].vertices.numpy(), np.asarray(PENTAGON), atol=1e-5)

    with pytest.raises(ValueError, match="No polygons"):
        from_gds(gds_file, layer=3, bounds=(0.0, UM))
    with pytest.raises(ValueError, match="No polygons"):
        from_gds(gds_file, layer=1, datatype=5, bounds=(0.0, UM))


@requires_gdstk
def test_cell_selection_and_reference_flattening(tmp_path):
    library = gdstk.Library()
    sub = library.new_cell("SUB")
    sub.add(gdstk.Polygon(SQUARE, layer=1, datatype=0))
    top = library.new_cell("CHIP")
    top.add(gdstk.Reference(sub, origin=(10.0, 0.0)))
    path = tmp_path / "cells.gds"
    library.write_gds(str(path))

    # cell=None resolves the unique top-level cell and flattens the reference.
    slabs = from_gds(path, layer=1, bounds=(0.0, UM))
    assert len(slabs) == 1
    shifted = (np.asarray(SQUARE) + [10.0, 0.0]) * UM
    assert np.allclose(slabs[0].vertices.numpy(), shifted, atol=1e-10)

    # An explicit cell name selects the sub-cell without the reference offset.
    slabs = from_gds(path, cell="SUB", layer=1, bounds=(0.0, UM))
    assert np.allclose(slabs[0].vertices.numpy(), np.asarray(SQUARE) * UM, atol=1e-11)

    with pytest.raises(ValueError, match="Available cells"):
        from_gds(path, cell="MISSING", layer=1, bounds=(0.0, UM))


@requires_gdstk
def test_ambiguous_top_cell_requires_explicit_name(tmp_path):
    library = gdstk.Library()
    library.new_cell("A").add(gdstk.Polygon(SQUARE, layer=1, datatype=0))
    library.new_cell("B").add(gdstk.Polygon(TRIANGLE, layer=1, datatype=0))
    path = tmp_path / "two_tops.gds"
    library.write_gds(str(path))

    with pytest.raises(ValueError, match="top-level"):
        from_gds(path, layer=1, bounds=(0.0, UM))

    slabs = from_gds(path, cell="A", layer=1, bounds=(0.0, UM))
    assert len(slabs) == 1


@requires_gdstk
def test_missing_file_raises():
    with pytest.raises(ValueError, match="not found"):
        from_gds("does_not_exist.gds", layer=1, bounds=(0.0, UM))


@requires_gdstk
def test_export_round_trip(tmp_path):
    slab = mw.PolySlab(
        vertices=np.asarray(SQUARE) * UM,
        bounds=(0.0, 0.22 * UM),
        position=(1.0 * UM, 2.0 * UM, 0.5 * UM),
    )
    hole = np.asarray([(0.5, 0.25), (1.5, 0.25), (1.5, 0.75), (0.5, 0.75)])
    holed = mw.ComplexPolySlab(
        loops=[np.asarray(SQUARE) * UM, hole * UM],
        bounds=(0.0, 0.22 * UM),
    )
    path = tmp_path / "out.gds"
    to_gds_file([slab, holed], path, layer=5, datatype=2)

    library = gdstk.read_gds(str(path))
    (cell,) = library.top_level()
    assert cell.name == "TOP"
    polygons = cell.get_polygons(layer=5, datatype=2)
    assert len(polygons) == 2
    # The in-plane position offset is baked into the exported vertices.
    assert np.allclose(polygons[0].points, np.asarray(SQUARE) + [1.0, 2.0], atol=1e-4)
    # The holed slab is written as a keyhole-cut simple polygon: the hole
    # interior stays empty instead of being unioned back into the boundary.
    assert not polygons[1].contain((1.0, 0.5))
    assert polygons[1].contain((0.25, 0.5))
    assert polygons[1].contain((1.0, 0.9))


@requires_gdstk
def test_export_length_scale_sets_library_unit(tmp_path):
    slab = mw.PolySlab(vertices=np.asarray(SQUARE) * UM, bounds=(0.0, 0.22 * UM))
    path = tmp_path / "nm.gds"
    to_gds_file(slab, path, length_scale=1e9)  # nanometre user units

    library = gdstk.read_gds(str(path))
    assert library.unit == pytest.approx(1e-9)
    # Default import reads the file's unit, so vertices come back in metres.
    (slab_back,) = from_gds(path, layer=0, bounds=(0.0, 0.22 * UM))
    assert np.allclose(slab_back.vertices.numpy(), np.asarray(SQUARE) * UM, atol=1e-12)


@requires_gdstk
def test_export_rejects_unsupported_geometry(tmp_path):
    path = tmp_path / "bad.gds"
    with pytest.raises(ValueError, match="PolySlab"):
        to_gds_file([mw.Box(size=(1.0, 1.0, 1.0))], path)

    rotated = mw.PolySlab(SQUARE, bounds=(0.0, UM), rotation=(0.0, 0.0, 0.3))
    with pytest.raises(ValueError, match="rotated"):
        to_gds_file(rotated, path)


def test_missing_gdstk_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "gdstk", None)
    with pytest.raises(ImportError, match="pip install gdstk"):
        from_gds(tmp_path / "x.gds", layer=1, bounds=(0.0, UM))
    with pytest.raises(ImportError, match="pip install gdstk"):
        to_gds_file([], tmp_path / "x.gds")


@requires_gdstk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU compile smoke test.")
def test_from_gds_scene_compiles_material_tensors_on_gpu(gds_file):
    slabs = from_gds(gds_file, layer=1, bounds=(-0.3 * UM, 0.3 * UM))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1 * UM, 5 * UM), (-1 * UM, 3 * UM), (-1 * UM, 1 * UM))),
        grid=mw.GridSpec.uniform(0.1 * UM),
        device="cuda",
    )
    for index, slab in enumerate(slabs):
        scene.add_structure(
            mw.Structure(name=f"slab_{index}", geometry=slab, material=mw.Material(eps_r=4.0))
        )

    prepared = prepare_scene(scene)
    eps_r, mu_r = prepared.compile_material_tensors()

    assert eps_r.device.type == "cuda"
    assert mu_r.device.type == "cuda"
    assert torch.all(torch.isfinite(eps_r))
    assert eps_r.max().item() > 3.0
