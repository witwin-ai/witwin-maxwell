from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Mapping

import torch

WINDOW_FACTOR = 15.0

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_INDEX_TO_AXIS = ("x", "y", "z")


def _as_real_if_lossless(value: complex | float | int) -> complex | float:
    """Collapse a (near-)real scalar to ``float`` so the vacuum/lossless path
    keeps its original real dtype and only genuinely lossy backgrounds carry an
    imaginary part into the propagator kernels."""
    number = complex(value)
    if abs(number.imag) <= 1e-12 * (abs(number.real) + 1.0):
        return number.real
    return number


def _normalize_background(
    background_eps_r: complex | float | int,
    background_mu_r: complex | float | int,
) -> tuple[complex | float, complex | float]:
    eps_r = _as_real_if_lossless(background_eps_r)
    mu_r = _as_real_if_lossless(background_mu_r)
    if isinstance(eps_r, float) and eps_r <= 0.0:
        raise ValueError("background_eps_r must be positive for a lossless exterior.")
    if isinstance(mu_r, float) and mu_r <= 0.0:
        raise ValueError("background_mu_r must be positive for a lossless exterior.")
    return eps_r, mu_r


def _background_wavenumber_and_impedance(
    background_eps_r: complex | float,
    background_mu_r: complex | float,
    *,
    omega: float,
    c: float,
    eta_vacuum: float,
) -> tuple[complex | float, complex | float]:
    """Return ``(k, eta)`` of the homogeneous exterior background relative to the
    vacuum wavenumber ``omega/c`` and vacuum impedance ``eta_vacuum``."""
    index = _as_real_if_lossless(cmath.sqrt(complex(background_eps_r) * complex(background_mu_r)))
    impedance_ratio = _as_real_if_lossless(
        cmath.sqrt(complex(background_mu_r) / complex(background_eps_r))
    )
    k = _as_real_if_lossless((omega / c) * complex(index))
    eta = _as_real_if_lossless(eta_vacuum * complex(impedance_ratio))
    return k, eta


def _resolve_currents_background(
    surfaces,
    background_eps_r: complex | float | None,
    background_mu_r: complex | float | None,
) -> tuple[complex | float, complex | float]:
    """Resolve the exterior background for a propagator: an explicit override wins,
    otherwise fall back to the medium carried by the equivalent currents."""
    if background_eps_r is None:
        background_eps_r = surfaces[0].background_eps_r
    if background_mu_r is None:
        background_mu_r = surfaces[0].background_mu_r
    return _normalize_background(background_eps_r, background_mu_r)


def _complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128


def _resolve_tensor_device(*values) -> torch.device:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _resolve_real_dtype(*values) -> torch.dtype:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.real.dtype
    return torch.float64


def _resolve_complex_dtype(*values) -> torch.dtype:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.dtype if torch.is_complex(value) else _complex_dtype_for(value.dtype)
    return torch.complex128


def _to_real_tensor(values, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def _to_complex_tensor(values, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def _as_1d_coords(values, name: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = _to_real_tensor(values, device=device, dtype=dtype)
    if coords.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={tuple(coords.shape)}.")
    if coords.numel() < 2:
        raise ValueError(f"{name} must contain at least two points, got {coords.numel()}.")
    return coords


def _trapz_weights_1d(points: torch.Tensor) -> torch.Tensor:
    coords = points.to(dtype=points.real.dtype)
    count = int(coords.numel())
    if count <= 1:
        return torch.ones((count,), device=coords.device, dtype=coords.dtype)
    diffs = coords[1:] - coords[:-1]
    weights = torch.empty((count,), device=coords.device, dtype=coords.dtype)
    weights[0] = diffs[0] / 2.0
    weights[-1] = diffs[-1] / 2.0
    if count > 2:
        weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return weights


def gaussian_window_1d(
    points,
    bound_min: float,
    bound_max: float,
    window_frac: float,
) -> torch.Tensor:
    device = _resolve_tensor_device(points)
    dtype = _resolve_real_dtype(points)
    point_tensor = _to_real_tensor(points, device=device, dtype=dtype)
    if window_frac <= 0.0:
        return torch.ones_like(point_tensor)

    span = float(bound_max) - float(bound_min)
    if span <= 0.0:
        raise ValueError("Window bounds must satisfy bound_max > bound_min.")

    window_extent = float(window_frac) * span / 2.0
    if window_extent <= 0.0:
        return torch.ones_like(point_tensor)

    inner_min = float(bound_min) + window_extent
    inner_max = float(bound_max) - window_extent

    weights = torch.ones_like(point_tensor)
    left_mask = point_tensor < inner_min
    if torch.any(left_mask):
        u_left = (point_tensor[left_mask] - inner_min) / window_extent
        weights[left_mask] = torch.exp(-0.5 * WINDOW_FACTOR * u_left.square())

    right_mask = point_tensor > inner_max
    if torch.any(right_mask):
        u_right = (point_tensor[right_mask] - inner_max) / window_extent
        weights[right_mask] = torch.exp(-0.5 * WINDOW_FACTOR * u_right.square())

    return weights


def _normalize_axis(axis: str) -> str:
    axis_name = str(axis).lower()
    if axis_name not in _AXIS_TO_INDEX:
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return axis_name


def _normalize_window_size(window_size: tuple[float, float] | None) -> tuple[float, float]:
    if window_size is None:
        return 0.0, 0.0
    if len(window_size) != 2:
        raise ValueError("window_size must contain exactly two values.")
    return float(window_size[0]), float(window_size[1])


def _normalize_tangential_bounds(
    tangential_bounds: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]] | None:
    if tangential_bounds is None:
        return None

    normalized: dict[str, tuple[float, float]] = {}
    for axis, bounds in tangential_bounds.items():
        axis_name = _normalize_axis(axis)
        if len(bounds) != 2:
            raise ValueError(f"{axis_name}-bounds must contain exactly two values.")
        lower = float(bounds[0])
        upper = float(bounds[1])
        if not upper > lower:
            raise ValueError(f"{axis_name}-bounds must satisfy upper > lower.")
        normalized[axis_name] = (lower, upper)
    return normalized


def _normalize_normal_direction(normal_direction: str | int | float) -> float:
    if isinstance(normal_direction, str):
        if normal_direction == "+":
            return 1.0
        if normal_direction == "-":
            return -1.0
        raise ValueError("normal_direction must be '+' or '-'.")
    normal = float(normal_direction)
    if abs(normal) <= 1e-12:
        raise ValueError("normal_direction must not be zero.")
    return 1.0 if normal > 0.0 else -1.0


def _tangential_indices(axis: str) -> tuple[int, int]:
    axis_index = _AXIS_TO_INDEX[axis]
    tangential = tuple(index for index in range(3) if index != axis_index)
    return tangential[0], tangential[1]


def _select_bound_indices(
    coords: torch.Tensor,
    *,
    axis: str,
    tangential_bounds: Mapping[str, tuple[float, float]] | None,
) -> torch.Tensor | None:
    if tangential_bounds is None or axis not in tangential_bounds:
        return None

    lower, upper = tangential_bounds[axis]
    coord_tensor = coords.to(dtype=coords.real.dtype)
    tolerance = 1e-12 * max(abs(lower), abs(upper), float(torch.max(torch.abs(coord_tensor)).item()), 1.0)
    mask = (coord_tensor >= lower - tolerance) & (coord_tensor <= upper + tolerance)
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() < 2:
        raise ValueError(
            f"{axis}-bounds [{lower}, {upper}] select fewer than two monitor samples."
        )
    if indices.numel() == coord_tensor.numel():
        return None
    return indices.to(device=coords.device)


def _as_plane_component(
    values,
    name: str,
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = _to_complex_tensor(values, device=device, dtype=dtype)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}.")
    return array


def _as_vector_field(
    values,
    name: str,
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = _to_complex_tensor(values, device=device, dtype=dtype)
    if tuple(array.shape) == shape + (3,):
        return array
    if tuple(array.shape) == (3,) + shape:
        array = torch.movedim(array, 0, -1)
    if tuple(array.shape) != shape + (3,):
        raise ValueError(
            f"{name} must have shape {shape + (3,)} or {(3,) + shape}, got {tuple(array.shape)}."
        )
    return array


def _resolve_physical_constants(
    *,
    solver=None,
    c: float | None = None,
    eps0: float | None = None,
    mu0: float | None = None,
) -> tuple[float, float, float]:
    if solver is not None:
        c = getattr(solver, "c", getattr(solver, "c0", c))
        eps0 = getattr(solver, "eps0", eps0)
        mu0 = getattr(solver, "mu0", mu0)

    missing = [
        name
        for name, value in (("c", c), ("eps0", eps0), ("mu0", mu0))
        if value is None
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Missing physical constants: {missing_list}. "
            "Pass solver=... or explicit c/eps0/mu0."
        )

    return float(c), float(eps0), float(mu0)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Stratton-Chu postprocessing defaults to CUDA, but torch.cuda.is_available() is False. "
                "Pass device='cpu' explicitly to run on CPU."
            )
        return torch.device("cuda")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Stratton-Chu postprocessing requested CUDA, but torch.cuda.is_available() is False."
        )
    return resolved


def build_plane_points(axis: str, position: float, u, v) -> torch.Tensor:
    axis_name = _normalize_axis(axis)
    device = _resolve_tensor_device(u, v)
    dtype = _resolve_real_dtype(u, v)
    u_coords = _as_1d_coords(u, "u", device=device, dtype=dtype)
    v_coords = _as_1d_coords(v, "v", device=device, dtype=dtype)
    shape = (u_coords.numel(), v_coords.numel())
    points = torch.zeros(shape + (3,), device=device, dtype=dtype)
    tangential_a, tangential_b = _tangential_indices(axis_name)
    uu, vv = torch.meshgrid(u_coords, v_coords, indexing="ij")
    points[..., _AXIS_TO_INDEX[axis_name]] = float(position)
    points[..., tangential_a] = uu
    points[..., tangential_b] = vv
    return points


@dataclass(frozen=True)
class PlanarEquivalentCurrents:
    axis: str
    position: float
    frequency: float
    u: torch.Tensor
    v: torch.Tensor
    J: torch.Tensor
    M: torch.Tensor
    background_eps_r: complex | float = 1.0
    background_mu_r: complex | float = 1.0

    def __post_init__(self):
        axis_name = _normalize_axis(self.axis)
        device = _resolve_tensor_device(self.u, self.v, self.J, self.M)
        real_dtype = _resolve_real_dtype(self.u, self.v, self.J, self.M)
        complex_dtype = _resolve_complex_dtype(self.J, self.M)
        u_coords = _as_1d_coords(self.u, "u", device=device, dtype=real_dtype)
        v_coords = _as_1d_coords(self.v, "v", device=device, dtype=real_dtype)
        if float(self.frequency) <= 0.0:
            raise ValueError("frequency must be positive.")
        shape = (u_coords.numel(), v_coords.numel())
        j_field = _as_vector_field(self.J, "J", shape, device=device, dtype=complex_dtype)
        m_field = _as_vector_field(self.M, "M", shape, device=device, dtype=complex_dtype)
        background_eps_r, background_mu_r = _normalize_background(
            self.background_eps_r, self.background_mu_r
        )

        object.__setattr__(self, "axis", axis_name)
        object.__setattr__(self, "position", float(self.position))
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "u", u_coords)
        object.__setattr__(self, "v", v_coords)
        object.__setattr__(self, "J", j_field)
        object.__setattr__(self, "M", m_field)
        object.__setattr__(self, "background_eps_r", background_eps_r)
        object.__setattr__(self, "background_mu_r", background_mu_r)

    @property
    def device(self) -> torch.device:
        return self.u.device

    @property
    def coord_dtype(self) -> torch.dtype:
        return self.u.dtype

    @property
    def tangential_axes(self) -> tuple[str, str]:
        tangential = _tangential_indices(self.axis)
        return _INDEX_TO_AXIS[tangential[0]], _INDEX_TO_AXIS[tangential[1]]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.u.numel()), int(self.v.numel())

    def weights_2d(self) -> torch.Tensor:
        return _trapz_weights_1d(self.u)[:, None] * _trapz_weights_1d(self.v)[None, :]

    def plane_points(self) -> torch.Tensor:
        return build_plane_points(self.axis, self.position, self.u, self.v)

    def quadrature(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flattened ``(points, J_weighted, M_weighted)`` for the radiation
        integral: the plane's Yee-consistent trapezoid area weights are folded
        into the electric/magnetic surface currents so every source-surface type
        (planar patch or general curved surface) exposes the same interface."""
        weights = self.weights_2d()[..., None].to(dtype=self.J.real.dtype)
        points = self.plane_points().reshape(-1, 3)
        j_weighted = (self.J * weights.to(dtype=self.J.dtype)).reshape(-1, 3)
        m_weighted = (self.M * weights.to(dtype=self.M.dtype)).reshape(-1, 3)
        return points, j_weighted, m_weighted

    def cropped(
        self,
        tangential_bounds: Mapping[str, tuple[float, float]] | None,
    ) -> "PlanarEquivalentCurrents":
        normalized_bounds = _normalize_tangential_bounds(tangential_bounds)
        if normalized_bounds is None:
            return self

        axis_u, axis_v = self.tangential_axes
        u_indices = _select_bound_indices(self.u, axis=axis_u, tangential_bounds=normalized_bounds)
        v_indices = _select_bound_indices(self.v, axis=axis_v, tangential_bounds=normalized_bounds)
        if u_indices is None and v_indices is None:
            return self

        u_coords = self.u if u_indices is None else self.u.index_select(0, u_indices)
        v_coords = self.v if v_indices is None else self.v.index_select(0, v_indices)
        j_field = self.J
        m_field = self.M
        if u_indices is not None:
            j_field = j_field.index_select(0, u_indices)
            m_field = m_field.index_select(0, u_indices)
        if v_indices is not None:
            j_field = j_field.index_select(1, v_indices)
            m_field = m_field.index_select(1, v_indices)

        return PlanarEquivalentCurrents(
            axis=self.axis,
            position=self.position,
            frequency=self.frequency,
            u=u_coords,
            v=v_coords,
            J=j_field,
            M=m_field,
            background_eps_r=self.background_eps_r,
            background_mu_r=self.background_mu_r,
        )

    def windowed(self, window_size: tuple[float, float] | None) -> "PlanarEquivalentCurrents":
        frac_u, frac_v = _normalize_window_size(window_size)
        if frac_u <= 0.0 and frac_v <= 0.0:
            return self
        window_u = gaussian_window_1d(self.u, float(self.u[0].item()), float(self.u[-1].item()), frac_u)
        window_v = gaussian_window_1d(self.v, float(self.v[0].item()), float(self.v[-1].item()), frac_v)
        window_2d = (window_u[:, None] * window_v[None, :])[..., None].to(dtype=self.J.real.dtype)
        return PlanarEquivalentCurrents(
            axis=self.axis,
            position=self.position,
            frequency=self.frequency,
            u=self.u,
            v=self.v,
            J=self.J * window_2d.to(dtype=self.J.dtype),
            M=self.M * window_2d.to(dtype=self.M.dtype),
            background_eps_r=self.background_eps_r,
            background_mu_r=self.background_mu_r,
        )


@dataclass(frozen=True)
class EquivalentCurrentsSurface:
    surfaces: tuple[PlanarEquivalentCurrents, ...]

    def __init__(self, surfaces):
        resolved_surfaces = tuple(surfaces)
        if not resolved_surfaces:
            raise ValueError("EquivalentCurrentsSurface requires at least one planar surface.")
        for surface in resolved_surfaces:
            if not isinstance(surface, PlanarEquivalentCurrents):
                raise TypeError(
                    "EquivalentCurrentsSurface expects PlanarEquivalentCurrents entries, "
                    f"got {type(surface).__name__}."
                )
        reference_frequency = float(resolved_surfaces[0].frequency)
        reference_eps = complex(resolved_surfaces[0].background_eps_r)
        reference_mu = complex(resolved_surfaces[0].background_mu_r)
        for surface in resolved_surfaces[1:]:
            if not math.isclose(float(surface.frequency), reference_frequency, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("All equivalent-current surfaces must share the same frequency.")
            if not cmath.isclose(
                complex(surface.background_eps_r), reference_eps, rel_tol=1e-6, abs_tol=1e-9
            ) or not cmath.isclose(
                complex(surface.background_mu_r), reference_mu, rel_tol=1e-6, abs_tol=1e-9
            ):
                raise ValueError(
                    "All equivalent-current surfaces must share the same exterior background medium."
                )
        object.__setattr__(self, "surfaces", resolved_surfaces)

    @property
    def frequency(self) -> float:
        return float(self.surfaces[0].frequency)

    @property
    def background_eps_r(self) -> complex | float:
        return self.surfaces[0].background_eps_r

    @property
    def background_mu_r(self) -> complex | float:
        return self.surfaces[0].background_mu_r

    @property
    def device(self) -> torch.device:
        return self.surfaces[0].device

    def cropped(
        self,
        tangential_bounds: Mapping[str, tuple[float, float]] | None,
    ) -> "EquivalentCurrentsSurface":
        return EquivalentCurrentsSurface(surface.cropped(tangential_bounds) for surface in self.surfaces)

    def windowed(self, window_size: tuple[float, float] | None) -> "EquivalentCurrentsSurface":
        return EquivalentCurrentsSurface(surface.windowed(window_size) for surface in self.surfaces)


@dataclass(frozen=True)
class SurfaceEquivalentCurrents:
    """Equivalent electric/magnetic surface currents on an arbitrary closed
    surface, sampled at explicit quadrature points with per-point area weights.

    Unlike :class:`PlanarEquivalentCurrents`, the geometry is not restricted to
    an axis-aligned plane: each quadrature point carries its own position and
    the currents may point in any direction, so a genuinely curved Huygens
    surface (a sphere, an ellipsoid, or a staircased/voxelized closed surface)
    can be radiated by the same Stratton-Chu integral as a box. The surface
    equivalence far field is independent of the enclosing surface's shape, so
    the propagator kernels consume this the same way they consume planar faces.
    """

    frequency: float
    points: torch.Tensor
    weights: torch.Tensor
    J: torch.Tensor
    M: torch.Tensor
    background_eps_r: complex | float = 1.0
    background_mu_r: complex | float = 1.0

    def __post_init__(self):
        device = _resolve_tensor_device(self.points, self.weights, self.J, self.M)
        real_dtype = _resolve_real_dtype(self.points, self.weights, self.J, self.M)
        complex_dtype = _resolve_complex_dtype(self.J, self.M)
        points = _to_real_tensor(self.points, device=device, dtype=real_dtype)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}.")
        count = int(points.shape[0])
        if count < 1:
            raise ValueError("SurfaceEquivalentCurrents requires at least one quadrature point.")
        weights = _to_real_tensor(self.weights, device=device, dtype=real_dtype).reshape(-1)
        if weights.numel() != count:
            raise ValueError(f"weights must have shape ({count},), got {tuple(weights.shape)}.")
        if float(self.frequency) <= 0.0:
            raise ValueError("frequency must be positive.")
        j_field = _to_complex_tensor(self.J, device=device, dtype=complex_dtype)
        m_field = _to_complex_tensor(self.M, device=device, dtype=complex_dtype)
        for name, field in (("J", j_field), ("M", m_field)):
            if tuple(field.shape) != (count, 3):
                raise ValueError(f"{name} must have shape ({count}, 3), got {tuple(field.shape)}.")
        background_eps_r, background_mu_r = _normalize_background(
            self.background_eps_r, self.background_mu_r
        )

        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "J", j_field)
        object.__setattr__(self, "M", m_field)
        object.__setattr__(self, "background_eps_r", background_eps_r)
        object.__setattr__(self, "background_mu_r", background_mu_r)

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def coord_dtype(self) -> torch.dtype:
        return self.points.dtype

    @property
    def shape(self) -> tuple[int]:
        return (int(self.points.shape[0]),)

    def quadrature(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flattened ``(points, J_weighted, M_weighted)`` with the per-point
        surface-area weights folded into the currents, matching the interface of
        :meth:`PlanarEquivalentCurrents.quadrature`."""
        weights = self.weights[:, None]
        j_weighted = self.J * weights.to(dtype=self.J.dtype)
        m_weighted = self.M * weights.to(dtype=self.M.dtype)
        return self.points, j_weighted, m_weighted


def equivalent_surface_currents_from_surface_samples(
    *,
    frequency: float,
    points,
    normals,
    areas,
    E,
    H,
    background_eps_r: complex | float = 1.0,
    background_mu_r: complex | float = 1.0,
) -> SurfaceEquivalentCurrents:
    """Build equivalent surface currents on an arbitrary closed surface from
    fields sampled at quadrature points.

    The surface equivalence principle replaces the interior sources by
    ``J = n x H`` and ``M = -n x E`` on any closed surface enclosing them, where
    ``n`` is the outward unit normal. This constructor accepts a general point
    cloud (``points``), the outward normals (``normals``, normalized here), the
    per-point differential surface areas (``areas``), and the tangential-plus
    fields (``E``, ``H``), so a curved Huygens surface can drive the same
    Stratton-Chu / near-to-far-field transform as an axis-aligned box.
    """
    device = _resolve_tensor_device(points, normals, areas, E, H)
    real_dtype = _resolve_real_dtype(points, normals, areas)
    complex_dtype = _resolve_complex_dtype(E, H)

    point_tensor = _to_real_tensor(points, device=device, dtype=real_dtype)
    if point_tensor.ndim != 2 or point_tensor.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(point_tensor.shape)}.")
    count = int(point_tensor.shape[0])

    normal_tensor = _to_real_tensor(normals, device=device, dtype=real_dtype)
    if tuple(normal_tensor.shape) != (count, 3):
        raise ValueError(f"normals must have shape ({count}, 3), got {tuple(normal_tensor.shape)}.")
    norm = torch.linalg.norm(normal_tensor, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError("normals must be non-zero vectors.")
    unit_normal = (normal_tensor / norm).to(dtype=complex_dtype)

    area_tensor = _to_real_tensor(areas, device=device, dtype=real_dtype).reshape(-1)
    if area_tensor.numel() != count:
        raise ValueError(f"areas must have shape ({count},), got {tuple(area_tensor.shape)}.")

    e_field = _to_complex_tensor(E, device=device, dtype=complex_dtype)
    h_field = _to_complex_tensor(H, device=device, dtype=complex_dtype)
    for name, field in (("E", e_field), ("H", h_field)):
        if tuple(field.shape) != (count, 3):
            raise ValueError(f"{name} must have shape ({count}, 3), got {tuple(field.shape)}.")

    j_field = torch.cross(unit_normal, h_field, dim=-1)
    m_field = -torch.cross(unit_normal, e_field, dim=-1)
    return SurfaceEquivalentCurrents(
        frequency=frequency,
        points=point_tensor,
        weights=area_tensor,
        J=j_field,
        M=m_field,
        background_eps_r=background_eps_r,
        background_mu_r=background_mu_r,
    )


def _normalize_currents_collection(
    currents: PlanarEquivalentCurrents | EquivalentCurrentsSurface | SurfaceEquivalentCurrents,
) -> tuple[PlanarEquivalentCurrents | SurfaceEquivalentCurrents, ...]:
    if isinstance(currents, EquivalentCurrentsSurface):
        return currents.surfaces
    if isinstance(currents, (PlanarEquivalentCurrents, SurfaceEquivalentCurrents)):
        return (currents,)
    raise TypeError(
        "currents must be PlanarEquivalentCurrents, EquivalentCurrentsSurface, or "
        f"SurfaceEquivalentCurrents, got {type(currents).__name__}."
    )


def _payload_monitor_frequencies(payload: Mapping[str, object]) -> tuple[float, ...]:
    if "frequencies" in payload:
        return tuple(float(freq) for freq in payload["frequencies"] if freq is not None)
    if "frequency" in payload and payload["frequency"] is not None:
        return (float(payload["frequency"]),)
    return ()


def _coord_indices_from_bounds(coords: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    coord_tensor = coords.to(dtype=coords.real.dtype)
    tolerance = 1e-12 * max(
        abs(lower),
        abs(upper),
        float(torch.max(torch.abs(coord_tensor)).item()),
        1.0,
    )
    mask = (coord_tensor >= lower - tolerance) & (coord_tensor <= upper + tolerance)
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() < 2:
        raise NotImplementedError(
            "Closed-surface Huygens validation requires at least two exterior material samples on each tangential axis."
        )
    return indices.to(device=coords.device)


def _sample_exterior_patch(
    material: torch.Tensor,
    prepared_scene,
    face_payload: Mapping[str, object],
) -> torch.Tensor:
    axis = _normalize_axis(face_payload["axis"])
    axis_index = _AXIS_TO_INDEX[axis]
    tangential = _tangential_indices(axis)
    permuted = material.permute((axis_index, tangential[0], tangential[1]))

    normal_direction = face_payload.get("normal_direction", "+")
    sign = 1 if normal_direction == "+" else -1
    normal_coords = getattr(prepared_scene, axis)
    plane_position = float(face_payload["position"])
    face_index = int(torch.argmin(torch.abs(normal_coords - plane_position)).item())
    outside_index = face_index + sign
    if outside_index < 0 or outside_index >= normal_coords.numel():
        raise NotImplementedError(
            "Closed-surface Huygens faces must have at least one material cell outside the surface."
        )

    tangential_bounds = face_payload.get("tangential_bounds")
    if tangential_bounds is None:
        raise NotImplementedError(
            "Closed-surface Huygens faces must expose tangential_bounds metadata."
        )

    coord_a_name = _INDEX_TO_AXIS[tangential[0]]
    coord_b_name = _INDEX_TO_AXIS[tangential[1]]
    coord_a = getattr(prepared_scene, coord_a_name)
    coord_b = getattr(prepared_scene, coord_b_name)
    indices_a = _coord_indices_from_bounds(coord_a, *tangential_bounds[coord_a_name])
    indices_b = _coord_indices_from_bounds(coord_b, *tangential_bounds[coord_b_name])

    return permuted[outside_index].index_select(0, indices_a).index_select(1, indices_b)


def _validate_homogeneous_exterior(
    result, monitor: Mapping[str, object], frequency: float
) -> tuple[complex | float, complex | float] | None:
    """Validate that the medium immediately outside a closed Huygens surface is a
    single homogeneous background and return that background as relative
    ``(eps_r, mu_r)`` so the exterior wavenumber/impedance can drive the
    near-to-far-field transform. Returns ``None`` for non-closed-surface
    monitors (their background defaults to vacuum)."""
    if monitor.get("kind") != "closed_surface":
        return None

    prepared_scene = result.prepared_scene
    eps_r, mu_r = prepared_scene.compile_relative_materials(frequency=frequency)
    reference_eps = None
    reference_mu = None

    for face_payload in monitor.get("faces", {}).values():
        eps_patch = _sample_exterior_patch(eps_r, prepared_scene, face_payload)
        mu_patch = _sample_exterior_patch(mu_r, prepared_scene, face_payload)
        eps_value = eps_patch.reshape(-1)[0]
        mu_value = mu_patch.reshape(-1)[0]

        if not torch.allclose(
            eps_patch,
            torch.full_like(eps_patch, eps_value),
            rtol=1e-4,
            atol=1e-4,
        ) or not torch.allclose(
            mu_patch,
            torch.full_like(mu_patch, mu_value),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise NotImplementedError(
                "Closed-surface Huygens currently requires a homogeneous exterior medium immediately outside every face."
            )

        if reference_eps is None:
            reference_eps = eps_value
            reference_mu = mu_value
            continue
        if not torch.allclose(reference_eps.reshape(()), eps_value.reshape(()), rtol=1e-4, atol=1e-4) or not torch.allclose(
            reference_mu.reshape(()),
            mu_value.reshape(()),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise NotImplementedError(
                "Closed-surface Huygens currently requires a single homogeneous exterior medium across all faces; "
                "layered or piecewise exteriors are not supported."
            )

    if reference_eps is None:
        return None
    return complex(reference_eps.item()), complex(reference_mu.item())


def _equivalent_surface_currents_from_payload(
    monitor: Mapping[str, object],
    *,
    tangential_bounds: Mapping[str, tuple[float, float]] | None = None,
    normal_direction: str | int | float | None = None,
    window_size: tuple[float, float] | None = None,
    background: tuple[complex | float, complex | float] = (1.0, 1.0),
) -> PlanarEquivalentCurrents | EquivalentCurrentsSurface:
    if monitor.get("kind") == "closed_surface":
        surfaces = []
        for face_payload in monitor.get("faces", {}).values():
            surfaces.append(
                _equivalent_surface_currents_from_payload(
                    face_payload,
                    tangential_bounds=tangential_bounds,
                    normal_direction=normal_direction,
                    window_size=window_size,
                    background=background,
                )
            )
        return EquivalentCurrentsSurface(tuple(surfaces))

    if monitor.get("kind") != "plane":
        raise ValueError("Monitor payload must be a plane or closed-surface monitor.")

    axis = _normalize_axis(monitor["axis"])
    tangential_a, tangential_b = _tangential_indices(axis)
    coord_name_a = _INDEX_TO_AXIS[tangential_a]
    coord_name_b = _INDEX_TO_AXIS[tangential_b]
    if coord_name_a not in monitor or coord_name_b not in monitor:
        raise ValueError(
            "Plane monitor payload does not expose aligned tangential coordinates."
        )

    resolved_frequency = float(monitor.get("frequency"))
    resolved_normal_direction = monitor.get("normal_direction", "+") if normal_direction is None else normal_direction
    return equivalent_surface_currents_from_fields(
        axis=axis,
        position=float(monitor["position"]),
        frequency=resolved_frequency,
        u=monitor[coord_name_a],
        v=monitor[coord_name_b],
        fields=monitor,
        normal_direction=resolved_normal_direction,
        window_size=None,
        background_eps_r=background[0],
        background_mu_r=background[1],
    ).cropped(tangential_bounds).windowed(window_size)


def equivalent_surface_currents_from_fields(
    *,
    axis: str,
    position: float,
    frequency: float,
    u,
    v,
    fields: Mapping[str, torch.Tensor],
    normal_direction: str | int | float = "+",
    window_size: tuple[float, float] | None = None,
    background_eps_r: complex | float = 1.0,
    background_mu_r: complex | float = 1.0,
) -> PlanarEquivalentCurrents:
    axis_name = _normalize_axis(axis)
    normalized_fields = {str(name).upper(): values for name, values in fields.items()}
    device = _resolve_tensor_device(u, v, *normalized_fields.values())
    real_dtype = _resolve_real_dtype(u, v, *normalized_fields.values())
    complex_dtype = _resolve_complex_dtype(*normalized_fields.values())
    u_coords = _as_1d_coords(u, "u", device=device, dtype=real_dtype)
    v_coords = _as_1d_coords(v, "v", device=device, dtype=real_dtype)
    shape = (u_coords.numel(), v_coords.numel())
    tangential = _tangential_indices(axis_name)

    required = [
        f"E{_INDEX_TO_AXIS[index].upper()}"
        for index in tangential
    ] + [
        f"H{_INDEX_TO_AXIS[index].upper()}"
        for index in tangential
    ]
    missing = [name for name in required if name not in normalized_fields]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Missing tangential field components for {axis_name}-normal plane: {missing_list}."
        )

    e_field = torch.zeros(shape + (3,), device=device, dtype=complex_dtype)
    h_field = torch.zeros(shape + (3,), device=device, dtype=complex_dtype)
    for index, axis_label in enumerate(_INDEX_TO_AXIS):
        e_name = f"E{axis_label.upper()}"
        h_name = f"H{axis_label.upper()}"
        if e_name in normalized_fields:
            e_field[..., index] = _as_plane_component(
                normalized_fields[e_name],
                e_name,
                shape,
                device=device,
                dtype=complex_dtype,
            )
        if h_name in normalized_fields:
            h_field[..., index] = _as_plane_component(
                normalized_fields[h_name],
                h_name,
                shape,
                device=device,
                dtype=complex_dtype,
            )

    normal = torch.zeros((1, 1, 3), device=device, dtype=real_dtype)
    normal[..., _AXIS_TO_INDEX[axis_name]] = _normalize_normal_direction(normal_direction)
    normal_complex = normal.to(dtype=complex_dtype).expand_as(e_field)
    j_field = torch.cross(normal_complex, h_field, dim=-1)
    m_field = -torch.cross(normal_complex, e_field, dim=-1)

    currents = PlanarEquivalentCurrents(
        axis=axis_name,
        position=position,
        frequency=frequency,
        u=u_coords,
        v=v_coords,
        J=j_field,
        M=m_field,
        background_eps_r=background_eps_r,
        background_mu_r=background_mu_r,
    )
    return currents.windowed(window_size)


def equivalent_surface_currents_from_monitor(
    result,
    monitor_name: str,
    *,
    frequency: float | None = None,
    freq_index: int | None = None,
    tangential_bounds: Mapping[str, tuple[float, float]] | None = None,
    normal_direction: str | int | float | None = None,
    window_size: tuple[float, float] | None = None,
) -> PlanarEquivalentCurrents | EquivalentCurrentsSurface:
    monitor = result.monitor(monitor_name, frequency=frequency, freq_index=freq_index)
    monitor_frequencies = _payload_monitor_frequencies(monitor)
    if len(monitor_frequencies) > 1:
        monitor = result.monitor(monitor_name, freq_index=0)
    resolved_frequency = float(monitor.get("frequency", getattr(result, "frequency", None)))
    background = _validate_homogeneous_exterior(result, monitor, resolved_frequency)
    if background is None:
        background = (1.0, 1.0)
    return _equivalent_surface_currents_from_payload(
        monitor,
        tangential_bounds=tangential_bounds,
        normal_direction=normal_direction,
        window_size=window_size,
        background=background,
    )


def equivalent_surface_currents_from_monitors(
    result,
    monitor_names,
    *,
    frequency: float | None = None,
    freq_index: int | None = None,
    tangential_bounds: Mapping[str, tuple[float, float]] | None = None,
    monitor_tangential_bounds: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    normal_directions: Mapping[str, str | int | float] | None = None,
    window_size: tuple[float, float] | None = None,
) -> EquivalentCurrentsSurface:
    names = tuple(str(name) for name in monitor_names)
    if not names:
        raise ValueError("monitor_names must contain at least one monitor.")

    surfaces = []
    for name in names:
        override_direction = None if normal_directions is None else normal_directions.get(name)
        override_bounds = tangential_bounds if monitor_tangential_bounds is None else monitor_tangential_bounds.get(
            name,
            tangential_bounds,
        )
        resolved = equivalent_surface_currents_from_monitor(
            result,
            name,
            frequency=frequency,
            freq_index=freq_index,
            tangential_bounds=override_bounds,
            normal_direction=override_direction,
            window_size=window_size,
        )
        if isinstance(resolved, EquivalentCurrentsSurface):
            surfaces.extend(resolved.surfaces)
        else:
            surfaces.append(resolved)
    return EquivalentCurrentsSurface(tuple(surfaces))


class StrattonChuPropagator:
    def __init__(
        self,
        currents: PlanarEquivalentCurrents | EquivalentCurrentsSurface | SurfaceEquivalentCurrents,
        *,
        solver=None,
        c: float | None = None,
        eps0: float | None = None,
        mu0: float | None = None,
        background_eps_r: complex | float | None = None,
        background_mu_r: complex | float | None = None,
        device: str | torch.device | None = None,
    ):
        self.currents = currents
        self._surfaces = _normalize_currents_collection(currents)
        self.device = _resolve_device(device)
        self.c, self.eps0, self.mu0 = _resolve_physical_constants(
            solver=solver,
            c=c,
            eps0=eps0,
            mu0=mu0,
        )
        self.background_eps_r, self.background_mu_r = _resolve_currents_background(
            self._surfaces, background_eps_r, background_mu_r
        )
        self.frequency = float(self._surfaces[0].frequency)
        self.coord_dtype = self._surfaces[0].coord_dtype
        self.field_dtype = _resolve_complex_dtype(*(surface.J for surface in self._surfaces), *(surface.M for surface in self._surfaces))
        self.omega = 2.0 * math.pi * self.frequency
        self.eta0 = math.sqrt(self.mu0 / self.eps0)
        self.k, self.eta = _background_wavenumber_and_impedance(
            self.background_eps_r,
            self.background_mu_r,
            omega=self.omega,
            c=self.c,
            eta_vacuum=self.eta0,
        )
        self.k_sq = self.k**2
        # Radiation into the homogeneous exterior scales i*omega with the
        # background permeability/permittivity: mu -> mu0*mu_r, eps -> eps0*eps_r.
        self.i_omega_mu = 1j * self.omega * self.mu0 * self.background_mu_r
        self.i_omega_eps = 1j * self.omega * self.eps0 * self.background_eps_r

        src_points = []
        j_weighted = []
        m_weighted = []
        for surface in self._surfaces:
            points, surface_j, surface_m = surface.quadrature()
            src_points.append(points.reshape(-1, 3))
            j_weighted.append(surface_j.reshape(-1, 3))
            m_weighted.append(surface_m.reshape(-1, 3))
        self._src_points = torch.cat(src_points, dim=0).to(device=self.device, dtype=self.coord_dtype)
        self._J_w = torch.cat(j_weighted, dim=0).to(device=self.device, dtype=self.field_dtype)
        self._M_w = torch.cat(m_weighted, dim=0).to(device=self.device, dtype=self.field_dtype)

    def _plane_reference_surface(self, axis: str | None) -> tuple[str, PlanarEquivalentCurrents]:
        if axis is not None:
            axis_name = _normalize_axis(axis)
            for surface in self._surfaces:
                if getattr(surface, "axis", None) == axis_name:
                    return axis_name, surface
            raise ValueError(f"No source surface is aligned with axis {axis_name!r}. Pass explicit u/v coordinates.")

        axes = [getattr(surface, "axis", None) for surface in self._surfaces]
        if axes[0] is None or any(surface_axis != axes[0] for surface_axis in axes[1:]):
            raise ValueError(
                "axis must be provided when propagating a multi-plane or general surface to a plane. "
                "The current surface has no single source-plane orientation; pass explicit axis, u, and v."
            )
        return axes[0], self._surfaces[0]

    def _compute_batch(self, observation_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observation = observation_points.to(device=self.device, dtype=self.coord_dtype)
        displacement = observation[:, None, :] - self._src_points[None, :, :]
        distance = torch.linalg.norm(displacement, dim=-1)
        if torch.any(distance <= 1e-15):
            raise ValueError("Observation points must not coincide with source quadrature points.")

        direction = displacement / distance[..., None]
        direction_complex = direction.to(dtype=self.field_dtype)
        distance_complex = distance.to(dtype=self.field_dtype)

        green = torch.exp(1j * self.k * distance_complex) * (1.0 / (4.0 * math.pi)) / distance_complex
        d_green = green * ((1j * self.k * distance_complex) - 1.0) / distance_complex
        d2_green = d_green * ((1j * self.k * distance_complex) - 1.0) / distance_complex + green / distance_complex.square()
        a_coeff = d_green / distance_complex
        b_coeff = d2_green - a_coeff

        j_weighted = self._J_w[None, :, :]
        m_weighted = self._M_w[None, :, :]
        r_dot_j = torch.sum(direction_complex * j_weighted, dim=-1)
        r_dot_m = torch.sum(direction_complex * m_weighted, dim=-1)

        e_integrand = self.i_omega_mu * (
            j_weighted * green[..., None]
            + (
                j_weighted * a_coeff[..., None]
                + direction_complex * (r_dot_j * b_coeff)[..., None]
            )
            / self.k_sq
        ) - torch.cross(direction_complex, m_weighted, dim=-1) * d_green[..., None]
        h_integrand = self.i_omega_eps * (
            m_weighted * green[..., None]
            + (
                m_weighted * a_coeff[..., None]
                + direction_complex * (r_dot_m * b_coeff)[..., None]
            )
            / self.k_sq
        ) + torch.cross(direction_complex, j_weighted, dim=-1) * d_green[..., None]

        return e_integrand.sum(dim=1), h_integrand.sum(dim=1)

    def propagate_points(self, points, *, batch_size: int = 256) -> dict[str, torch.Tensor]:
        observation = _to_real_tensor(points, device=self.device, dtype=self.coord_dtype)
        if observation.ndim == 1:
            if observation.shape[0] != 3:
                raise ValueError("points must have shape (3,) or (N, 3).")
            observation = observation[None, :]
        if observation.ndim != 2 or observation.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        e_batches = []
        h_batches = []
        for start in range(0, observation.shape[0], batch_size):
            stop = min(start + batch_size, observation.shape[0])
            e_batch, h_batch = self._compute_batch(observation[start:stop])
            e_batches.append(e_batch)
            h_batches.append(h_batch)

        e_field = torch.cat(e_batches, dim=0)
        h_field = torch.cat(h_batches, dim=0)
        return {
            "points": observation,
            "E": e_field,
            "H": h_field,
            "Ex": e_field[:, 0],
            "Ey": e_field[:, 1],
            "Ez": e_field[:, 2],
            "Hx": h_field[:, 0],
            "Hy": h_field[:, 1],
            "Hz": h_field[:, 2],
        }

    def propagate_plane(
        self,
        *,
        axis: str | None = None,
        position: float,
        u=None,
        v=None,
        batch_size: int = 256,
    ) -> dict[str, torch.Tensor | float]:
        if u is not None and v is not None:
            plane_axis = _normalize_axis(axis) if axis is not None else self._plane_reference_surface(None)[0]
            u_coords = _as_1d_coords(u, "u", device=self.device, dtype=self.coord_dtype)
            v_coords = _as_1d_coords(v, "v", device=self.device, dtype=self.coord_dtype)
        else:
            plane_axis, reference_surface = self._plane_reference_surface(axis)
            u_coords = reference_surface.u if u is None else _as_1d_coords(u, "u", device=self.device, dtype=self.coord_dtype)
            v_coords = reference_surface.v if v is None else _as_1d_coords(v, "v", device=self.device, dtype=self.coord_dtype)
        grid_points = build_plane_points(plane_axis, position, u_coords, v_coords).to(device=self.device, dtype=self.coord_dtype)
        propagated = self.propagate_points(grid_points.reshape(-1, 3), batch_size=batch_size)
        shape = (int(u_coords.numel()), int(v_coords.numel()))
        tangential_axes = tuple(_INDEX_TO_AXIS[index] for index in _tangential_indices(plane_axis))

        result = {
            "axis": plane_axis,
            "position": float(position),
            "frequency": self.frequency,
            "u": u_coords,
            "v": v_coords,
            tangential_axes[0]: u_coords,
            tangential_axes[1]: v_coords,
            plane_axis: float(position),
        }

        e_field = propagated["E"].reshape(shape + (3,))
        h_field = propagated["H"].reshape(shape + (3,))
        result["E"] = e_field
        result["H"] = h_field
        result["Ex"] = e_field[..., 0]
        result["Ey"] = e_field[..., 1]
        result["Ez"] = e_field[..., 2]
        result["Hx"] = h_field[..., 0]
        result["Hy"] = h_field[..., 1]
        result["Hz"] = h_field[..., 2]
        result["E_intensity"] = torch.sum(torch.abs(e_field).square(), dim=-1)
        return result

    def propagate(
        self,
        obs_u,
        obs_v,
        obs_position: float,
        *,
        axis: str | None = None,
        batch_size: int = 256,
    ) -> dict[str, torch.Tensor | float]:
        return self.propagate_plane(
            axis=axis,
            position=obs_position,
            u=obs_u,
            v=obs_v,
            batch_size=batch_size,
        )
