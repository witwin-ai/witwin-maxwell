"""GDS serialization of the frozen public F-A fixture for front-end gates."""

import gdstk
import numpy as np
import torch

import witwin.maxwell as mw


def process_fa(fixture, *, tolerance=0.0):
    metal = fixture.scene.structures[0].material
    return mw.ProcessStack(fixture.scene.background,
        (mw.LayerMap("lower", 1, 302*1e-6, 1e-6, metal),
         mw.LayerMap("top", 2, 305e-6, 2e-6, metal)),
        (mw.ViaMap(3, ("lower", "top"), metal),), pin_layer=(10, 0),
        breakpoint_tolerance_m=tolerance)


def write_fa_gds(fixture, path, *, jitter=False, unit=1e-6):
    library = gdstk.Library(unit=unit, precision=1e-9)
    cell = library.new_cell("FA")
    rng = np.random.default_rng(7)
    for structure in fixture.scene.structures:
        geometry = structure.geometry
        if isinstance(geometry, mw.Box):
            position, size = (x.detach().to(torch.float64).cpu().numpy()
                              for x in (geometry.position, geometry.size))
            polygons = [gdstk.rectangle((position[:2]-size[:2]/2)/unit,
                                       (position[:2]+size[:2]/2)/unit, layer=3)]
        else:
            layer = 1 if geometry.coordinate < 303e-6 else 2
            outer = geometry.vertices.detach().cpu().numpy()/unit
            polygons = [gdstk.Polygon(outer, layer=layer)]
            for hole in geometry.holes:
                polygons = gdstk.boolean(polygons, [hole.detach().cpu().numpy()/unit], "not",
                                         precision=1e-9/unit, layer=layer)
        if jitter:
            polygons = [gdstk.Polygon(p.points + rng.uniform(-2e-9/unit,2e-9/unit,p.points.shape),
                                     layer=p.layer, datatype=p.datatype) for p in polygons]
        cell.add(*polygons)
    for port in fixture.scene.ports:
        for sign, point in (("+",port.positive),("-",port.negative)):
            metal = "lower" if point[2] < 303e-6 else "top"
            cell.add(gdstk.Label(f"{port.name}:{sign}:{metal}", np.array(point[:2])/unit, layer=10))
    library.write_gds(str(path))
    return path
