"""PyTorch-native differentiable Maxwell solvers: ``Scene -> Simulation -> Result``.

The top level holds the core objects of that model. Every other public object has one
home in a domain namespace, loaded on first access::

    mw.rf, mw.rf.rational          ports, lumped elements, networks, path integrals, rational fits
    mw.media, mw.media.fit,        material models beyond the core poles; dispersion fitting;
    mw.media.library               the bundled material library
    mw.geometry, mw.grid           non-core primitives and CSG; grid refinement specs
    mw.sources, mw.monitors        non-core sources, waveforms and monitors
    mw.fdtd, mw.fdfd               solver-specific configuration, absorbers and errors
    mw.circuits, mw.antenna, mw.esd, mw.breakdown, mw.sar, mw.electrostatic,
    mw.coordinates, mw.execution, mw.capability, mw.postprocess, mw.invdes, mw.mom
    mw.io                          scene dictionary migration (``Scene.to_dict``/``from_dict`` read and write)

A namespace's ``__all__`` is its public contract. The modules that implement the objects
are internal.
"""
from ._version import __version__
from .errors import (
    CapacityError,
    ConvergenceError,
    ExperimentalWarning,
    MaxwellError,
    MaxwellWarning,
    SchemaError,
    SetupError,
    UnsupportedFeatureError,
)
from .capability import capabilities
# ``scene`` and the compiler import each other; the cycle resolves when the compiler loads first.
from . import compiler as _compiler
from witwin.core import Box, Cylinder, Mesh, Sphere, Structure
from .geometry import PolySlab
from .media import DebyePole, DiagonalTensor3, DrudePole, LorentzPole, Material
from .scene import (
    BoundaryKind,
    BoundarySpec,
    Domain,
    GridSpec,
    MaterialRegion,
    Scene,
    SceneModule,
    SubpixelSpec,
)
from .sources import CW, GaussianBeam, GaussianPulse, ModeSource, PlaneWave, PointDipole, TFSF, UniformCurrentSource
from .monitors import FinitePlaneMonitor, FluxMonitor, ModeMonitor, PlaneMonitor, PointMonitor
from .ports import AxisPath, LumpedPort, ModePort
# The layered feature group loads before ``simulation`` loads ``mom.runtime``, the order the
# consumer-metadata bridge's import probe was receipted with.
from .mom import layered as _layered  # noqa: F401

from .simulation import FDFDConfig, FDTDConfig, Simulation, SimulationMethod, TimeConfig
from .result import Result
from .results import FarFieldData, FieldData, FluxData, ModeAmplitudeData, SParameterData

from ._lazy import lazy_attributes
# Installs the GPU mesh-distance hook into witwin.core.
from .compiler.accel import mesh as _mesh_accelerator

__all__ = [
    # errors and warnings
    "MaxwellError",
    "UnsupportedFeatureError",
    "SetupError",
    "ConvergenceError",
    "SchemaError",
    "CapacityError",
    "MaxwellWarning",
    "ExperimentalWarning",
    # scene
    "Scene",
    "SceneModule",
    "Domain",
    "GridSpec",
    "BoundarySpec",
    "BoundaryKind",
    "SubpixelSpec",
    "MaterialRegion",
    "Structure",
    "Box",
    "Sphere",
    "Cylinder",
    "Mesh",
    "PolySlab",
    "Material",
    "DebyePole",
    "DrudePole",
    "LorentzPole",
    "DiagonalTensor3",
    # sources, monitors, ports
    "CW",
    "GaussianPulse",
    "PointDipole",
    "PlaneWave",
    "GaussianBeam",
    "ModeSource",
    "TFSF",
    "UniformCurrentSource",
    "PointMonitor",
    "PlaneMonitor",
    "FinitePlaneMonitor",
    "FluxMonitor",
    "ModeMonitor",
    "ModePort",
    "LumpedPort",
    "AxisPath",
    # simulation
    "Simulation",
    "SimulationMethod",
    "FDTDConfig",
    "FDFDConfig",
    "TimeConfig",
    "capabilities",
    # results
    "Result",
    "FieldData",
    "FluxData",
    "ModeAmplitudeData",
    "SParameterData",
    "FarFieldData",
]

_NAMESPACES = (
    "antenna",
    "breakdown",
    "capability",
    "circuits",
    "coordinates",
    "electrostatic",
    "esd",
    "execution",
    "fdfd",
    "fdtd",
    "geometry",
    "grid",
    "invdes",
    "io",
    "media",
    "mom",
    "monitors",
    "postprocess",
    "rf",
    "sar",
    "sources",
)

__getattr__, __dir__ = lazy_attributes(__name__, modules={name: f".{name}" for name in _NAMESPACES})
del lazy_attributes
