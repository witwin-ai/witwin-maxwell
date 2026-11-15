from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from witwin.core import Structure

from .._fixed_inputs import require_fixed_input
from ..errors import SetupError, UnsupportedFeatureError
from .._stability import experimental

_AXES = ("x", "y", "z")
_SIDES = ("low", "high")
_FACES = tuple(f"{axis}_{side}" for axis in _AXES for side in _SIDES)
_BC_KINDS = ("dirichlet", "neumann", "symmetry")


def _resolve_geometry(geometry, structure):
    """Return the geometry object described by ``geometry=`` or ``structure=``."""
    if (geometry is None) == (structure is None):
        raise ValueError(
            "ElectrostaticTerminal requires exactly one of geometry= or structure=."
        )
    if structure is not None:
        if not isinstance(structure, Structure):
            raise TypeError("ElectrostaticTerminal structure= must be a witwin.core.Structure.")
        resolved = getattr(structure, "geometry", None)
        if resolved is None:
            raise ValueError("ElectrostaticTerminal structure= has no geometry.")
        return resolved
    if not hasattr(geometry, "signed_distance"):
        raise TypeError(
            "ElectrostaticTerminal geometry= must expose signed_distance(x, y, z) "
            "(a witwin.core geometry primitive)."
        )
    return geometry


@experimental("electrostatics")
@dataclass(frozen=True)
class ElectrostaticTerminal:
    """Equipotential conductor constraint for the electrostatic solver.

    A terminal pins the potential of every cell whose centre falls inside its
    geometry. ``potential`` sets a fixed voltage, ``grounded=True`` is shorthand
    for ``potential=0``, and ``charge`` (Coulombs) declares a floating conductor
    whose potential is unknown but whose induced free charge is prescribed.
    ``potential`` and ``charge`` are mutually exclusive.

    Terminals live in a solver-specific collection (``Scene.electrostatic_terminals``)
    and never enter the RF ``Scene.ports`` set: an electrostatic terminal is an
    equipotential constraint, not an RF excitation/measurement path.
    """

    name: str
    geometry: Any = None
    structure: Any = None
    potential: float | None = None
    charge: float | None = None
    grounded: bool = False

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("ElectrostaticTerminal name must not be empty.")
        object.__setattr__(self, "name", name)
        # Terminal levels are stored as Python floats; a trainable value would detach.
        require_fixed_input(self.potential, name=f"ElectrostaticTerminal {name!r} potential")
        require_fixed_input(self.charge, name=f"ElectrostaticTerminal {name!r} charge")

        resolved_geometry = _resolve_geometry(self.geometry, self.structure)
        object.__setattr__(self, "geometry", resolved_geometry)

        grounded = bool(self.grounded)
        object.__setattr__(self, "grounded", grounded)

        if grounded:
            if self.charge is not None:
                raise ValueError(f"Terminal {name!r}: grounded conductors cannot prescribe charge.")
            if self.potential is not None and float(self.potential) != 0.0:
                raise ValueError(f"Terminal {name!r}: grounded conductors are held at 0 V.")
            object.__setattr__(self, "potential", 0.0)
            object.__setattr__(self, "charge", None)
            return

        if self.potential is not None and self.charge is not None:
            raise ValueError(
                f"Terminal {name!r}: potential= and charge= are mutually exclusive."
            )
        if self.potential is None and self.charge is None:
            raise ValueError(
                f"Terminal {name!r}: provide potential=, charge=, or grounded=True."
            )
        if self.potential is not None:
            object.__setattr__(self, "potential", float(self.potential))
        if self.charge is not None:
            object.__setattr__(self, "charge", float(self.charge))

    @property
    def is_floating(self) -> bool:
        return self.potential is None and self.charge is not None


@experimental("electrostatics")
@dataclass(frozen=True)
class ChargeDensity:
    """Volumetric free-charge source, ``density`` in C/m^3 over ``geometry``."""

    geometry: Any
    density: float | torch.Tensor

    def __post_init__(self):
        if not hasattr(self.geometry, "signed_distance"):
            raise TypeError(
                "ChargeDensity geometry must expose signed_distance(x, y, z)."
            )
        if not isinstance(self.density, torch.Tensor):
            object.__setattr__(self, "density", float(self.density))


def _normalize_bc_entry(name: str, value) -> tuple[str, float]:
    if isinstance(value, str):
        kind = value.strip().lower()
        magnitude = 0.0
    else:
        if len(value) != 2:
            raise SetupError(f"{name} boundary override must be a kind or a (kind, value) pair.")
        kind = str(value[0]).strip().lower()
        require_fixed_input(value[1], name=f"ElectrostaticBoundarySpec {name} value")
        magnitude = float(value[1])
    if kind == "open":
        raise UnsupportedFeatureError(
            "Open (infinite-domain) electrostatic boundaries are not implemented: a "
            "grounded-box (Dirichlet) or insulating (Neumann) enclosure is a finite "
            "truncation of an open problem, and there is no exact radiation condition on "
            "the scalar potential at a finite Cartesian face. Model an isolated structure "
            "by enlarging the grounded-box domain until the capacitance converges, and "
            "quantify the residual truncation error with the opt-in truncation_estimate on "
            "Simulation.capacitance(...). A boundary-element exact open boundary is a later "
            "phase.",
            feature="guard:128616df", method="electrostatic"
        )
    if kind not in _BC_KINDS:
        raise SetupError(
            f"{name} boundary kind must be one of {_BC_KINDS!r}, got {kind!r}."
        )
    if kind != "dirichlet" and magnitude != 0.0:
        raise SetupError(f"{name}: only Dirichlet boundaries carry a non-zero value.")
    return (kind, magnitude)


@experimental("electrostatics")
@dataclass(frozen=True)
class ElectrostaticBoundarySpec:
    """Per-face outer boundary conditions for the electrostatic domain.

    Each of the six faces (``x_low``, ``x_high``, ...) is ``"dirichlet"`` (fixed
    potential), ``"neumann"`` (zero normal displacement, insulating), or
    ``"symmetry"`` (mirror plane; realised as a homogeneous Neumann condition on
    the scalar potential). ``symmetry`` is treated as an even/insulating mirror.
    """

    default: str = "neumann"
    value: float = 0.0
    x_low: Any = None
    x_high: Any = None
    y_low: Any = None
    y_high: Any = None
    z_low: Any = None
    z_high: Any = None
    _faces: dict[str, tuple[str, float]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        default = _normalize_bc_entry("default", (self.default, self.value))
        faces: dict[str, tuple[str, float]] = {}
        for face in _FACES:
            override = getattr(self, face)
            faces[face] = default if override is None else _normalize_bc_entry(face, override)
        object.__setattr__(self, "default", default[0])
        object.__setattr__(self, "value", default[1])
        object.__setattr__(self, "_faces", faces)

    @classmethod
    def grounded_box(cls) -> "ElectrostaticBoundarySpec":
        """Dirichlet 0 V on all six faces (a grounded enclosure)."""
        return cls(default="dirichlet", value=0.0)

    @classmethod
    def dirichlet(cls, value: float = 0.0, **faces) -> "ElectrostaticBoundarySpec":
        require_fixed_input(value, name="ElectrostaticBoundarySpec.dirichlet value")
        return cls(default="dirichlet", value=float(value), **faces)

    @classmethod
    def neumann(cls, **faces) -> "ElectrostaticBoundarySpec":
        return cls(default="neumann", value=0.0, **faces)

    @classmethod
    def symmetry(cls, **faces) -> "ElectrostaticBoundarySpec":
        return cls(default="symmetry", value=0.0, **faces)

    def face(self, axis: str, side: str) -> tuple[str, float]:
        key = f"{str(axis).lower()}_{str(side).lower()}"
        if key not in self._faces:
            raise ValueError(f"Unknown boundary face {key!r}.")
        return self._faces[key]

    @property
    def has_dirichlet(self) -> bool:
        return any(kind == "dirichlet" for kind, _ in self._faces.values())


@experimental("electrostatics")
@dataclass(frozen=True)
class ElectrostaticSolverConfig:
    """Preconditioned conjugate-gradient configuration.

    ``dtype`` is the working precision of the solve; float64 is recommended so
    the Krylov accumulation and the reported residual are trustworthy. The
    tolerance is a relative residual ``||A x - b|| / ||b||``.
    """

    tolerance: float = 1e-10
    max_iterations: int = 20000
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        tol = float(self.tolerance)
        if tol <= 0.0:
            raise SetupError("ElectrostaticSolverConfig.tolerance must be > 0.")
        object.__setattr__(self, "tolerance", tol)
        iters = int(self.max_iterations)
        if iters <= 0:
            raise SetupError("ElectrostaticSolverConfig.max_iterations must be > 0.")
        object.__setattr__(self, "max_iterations", iters)
        if self.dtype not in (torch.float32, torch.float64):
            raise SetupError("ElectrostaticSolverConfig.dtype must be torch.float32 or torch.float64.")


def _require_boundary_and_solver(config) -> None:
    """Type-check the ``boundary``/``solver`` fields shared by the static configs."""
    if not isinstance(config.boundary, ElectrostaticBoundarySpec):
        raise TypeError("boundary must be an ElectrostaticBoundarySpec.")
    if not isinstance(config.solver, ElectrostaticSolverConfig):
        raise TypeError("solver must be an ElectrostaticSolverConfig.")


@experimental("electrostatics")
@dataclass(frozen=True)
class ElectrostaticConfig:
    """Run options of ``Simulation.electrostatic(scene, config=...)``.

    ``boundary`` sets the six outer faces of the electrostatic domain (default: a
    grounded box, Dirichlet 0 V on every face); it is independent of the
    full-wave ``Scene.boundary``, which must stay ``BoundarySpec.none()``.
    ``solver`` sets the conjugate-gradient tolerance, iteration cap and dtype.
    """

    boundary: ElectrostaticBoundarySpec = field(default_factory=ElectrostaticBoundarySpec.grounded_box)
    solver: ElectrostaticSolverConfig = field(default_factory=ElectrostaticSolverConfig)
    device: str = "cuda"

    def __post_init__(self):
        object.__setattr__(self, "device", str(torch.device(self.device)))
        _require_boundary_and_solver(self)
