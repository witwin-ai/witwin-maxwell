"""Maxwell-specific geometry implementations."""

from .polyslab import ComplexPolySlab, PolySlab
from .planar_pec import PlanarPEC
from .surface_mesh import SurfaceMesh

__all__ = ["ComplexPolySlab", "PolySlab", "PlanarPEC", "SurfaceMesh"]
