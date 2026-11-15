"""Public namespace ``mw.geometry``: geometry primitives and CSG beyond the core ones at the
top level, and the Maxwell-specific planar and surface geometries."""

from witwin.core import Cone, Difference, Ellipsoid, HollowBox, Intersection, Prism, Pyramid, Torus, Union
from witwin.core import ExtrudedPolygon

from .._stability import experimental_external

from .polyslab import ComplexPolySlab, PolySlab
from .planar_pec import PlanarPEC
from .surface_mesh import SurfaceMesh

# Defined in witwin.core and published here: recorded as experimental without modifying the core class
# (no __init__ wrapper, no __stability__ attribute, so no ExperimentalWarning).
experimental_external("geometry")(ExtrudedPolygon)

__all__ = [
    "ComplexPolySlab",
    "Cone",
    "Difference",
    "Ellipsoid",
    "ExtrudedPolygon",
    "HollowBox",
    "Intersection",
    "PlanarPEC",
    "Prism",
    "Pyramid",
    "SurfaceMesh",
    "Torus",
    "Union",
]
