"""Postprocess-time field projection onto angle, Cartesian-plane and k-space grids.

The projections radiate equivalent surface currents (``J = n x H``,
``M = -n x E`` on a closed Huygens surface) into the homogeneous exterior
background through the existing propagators:

- the far-field approximation uses :class:`NearFieldFarFieldTransformer`
  (transverse ``exp(ikr)/r`` fields, phase-referenced to the projection origin);
- the exact evaluation uses :class:`StrattonChuPropagator` (full Stratton-Chu
  integral, valid at any observation point outside the surface).

These are evaluated after the simulation from recorded surface currents. The
declarative ``AngleProjectionMonitor`` / ``CartesianProjectionMonitor`` /
``KSpaceProjectionMonitor`` record their surface during the run and are
projected through :func:`project_monitor` when the Result is queried. Every grid keeps an explicit leading frequency axis:
one equivalent-current set per frequency yields field tensors of shape
``[F, *grid, 3]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..constants import C_0, EPSILON_0, MU_0
from .nfft import NearFieldFarFieldTransformer
from .stratton_chu import (
    EquivalentCurrentsSurface,
    PlanarEquivalentCurrents,
    StrattonChuPropagator,
    SurfaceEquivalentCurrents,
    _AXIS_TO_INDEX,
    _normalize_axis,
    _normalize_normal_direction,
    _tangential_indices,
    _to_real_tensor,
    equivalent_surface_currents_from_monitor,
    equivalent_surface_currents_from_monitors,
)

_Currents = PlanarEquivalentCurrents | EquivalentCurrentsSurface | SurfaceEquivalentCurrents


@dataclass(frozen=True)
class ProjectedFieldData:
    """Fields projected onto an observation grid, with an explicit frequency axis.

    ``grid`` denotes the observation-grid shape (``theta x phi``, ``x x y`` or
    ``ux x uy``). Spherical components refer to the world frame centred at the
    projection ``origin``: ``r_hat`` points from the origin to the observation
    point and ``(theta, phi)`` are its polar/azimuthal angles about world ``+z``.

    Attributes:
        kind: ``"angle"``, ``"cartesian"`` or ``"kspace"``.
        far_field_approx: whether the transverse far-field approximation was used
            instead of the exact Stratton-Chu integral.
        frequencies: ``[F]`` frequencies in Hz.
        wavenumbers: ``[F]`` complex background wavenumbers ``k_b``.
        impedances: ``[F]`` complex background impedances ``eta_b``.
        origin: ``[3]`` projection origin.
        coords: named grid coordinates, each of shape ``grid``.
        points: ``[*grid, 3]`` observation points.
        directions: ``[*grid, 3]`` unit vectors from the origin to ``points``.
        radius: ``[*grid]`` distance from the origin to ``points``.
        theta, phi: ``[*grid]`` spherical angles of ``directions``.
        mask: ``[*grid]`` boolean validity mask (k-space propagating directions;
            all ``True`` for angle and Cartesian grids). Fields are zero where
            the mask is ``False``.
        E, H: ``[F, *grid, 3]`` complex Cartesian fields.
    """

    kind: str
    far_field_approx: bool
    frequencies: torch.Tensor
    wavenumbers: torch.Tensor
    impedances: torch.Tensor
    origin: torch.Tensor
    coords: Mapping[str, torch.Tensor]
    points: torch.Tensor
    directions: torch.Tensor
    radius: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    mask: torch.Tensor
    E: torch.Tensor
    H: torch.Tensor

    def _spherical_basis(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta, phi = self.theta, self.phi
        r_hat = torch.stack(
            (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
            dim=-1,
        )
        theta_hat = torch.stack(
            (torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi), -torch.sin(theta)),
            dim=-1,
        )
        phi_hat = torch.stack((-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)), dim=-1)
        dtype = self.E.dtype
        return r_hat.to(dtype=dtype), theta_hat.to(dtype=dtype), phi_hat.to(dtype=dtype)

    def _component(self, field: torch.Tensor, index: int) -> torch.Tensor:
        return torch.sum(field * self._spherical_basis()[index], dim=-1)

    @property
    def Ex(self) -> torch.Tensor:
        return self.E[..., 0]

    @property
    def Ey(self) -> torch.Tensor:
        return self.E[..., 1]

    @property
    def Ez(self) -> torch.Tensor:
        return self.E[..., 2]

    @property
    def Hx(self) -> torch.Tensor:
        return self.H[..., 0]

    @property
    def Hy(self) -> torch.Tensor:
        return self.H[..., 1]

    @property
    def Hz(self) -> torch.Tensor:
        return self.H[..., 2]

    @property
    def E_r(self) -> torch.Tensor:
        return self._component(self.E, 0)

    @property
    def E_theta(self) -> torch.Tensor:
        return self._component(self.E, 1)

    @property
    def E_phi(self) -> torch.Tensor:
        return self._component(self.E, 2)

    @property
    def H_r(self) -> torch.Tensor:
        return self._component(self.H, 0)

    @property
    def H_theta(self) -> torch.Tensor:
        return self._component(self.H, 1)

    @property
    def H_phi(self) -> torch.Tensor:
        return self._component(self.H, 2)

    @property
    def E_intensity(self) -> torch.Tensor:
        """``[F, *grid]`` electric intensity ``|E|^2``."""
        return torch.sum(torch.abs(self.E).square(), dim=-1)

    @property
    def power_density(self) -> torch.Tensor:
        """``[F, *grid]`` time-averaged radial Poynting flux ``0.5 Re(E x H*) . r_hat`` in W/m^2."""
        poynting = 0.5 * torch.real(torch.cross(self.E, torch.conj(self.H), dim=-1))
        return torch.sum(poynting * self.directions.to(dtype=poynting.dtype), dim=-1)

    @property
    def radiation_intensity(self) -> torch.Tensor:
        """``[F, *grid]`` radiation intensity ``r^2 S_r`` in W/sr.

        Independent of the projection radius under the far-field approximation in
        a lossless background; for exact projections it includes the near-field
        contribution at the sampled distance.
        """
        return self.power_density * self.radius.to(dtype=self.power_density.dtype).square()


def _currents_sequence(currents) -> tuple[_Currents, ...]:
    if isinstance(currents, (PlanarEquivalentCurrents, EquivalentCurrentsSurface, SurfaceEquivalentCurrents)):
        return (currents,)
    resolved = tuple(currents)
    if not resolved:
        raise ValueError("currents must contain at least one equivalent-current set.")
    return resolved


def _unit_axis(axis: str, sign: float, *, device, dtype) -> torch.Tensor:
    vector = torch.zeros(3, device=device, dtype=dtype)
    vector[_AXIS_TO_INDEX[axis]] = sign
    return vector


def _tangential_unit_axes(axis: str, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = _tangential_indices(axis)
    eye = torch.eye(3, device=device, dtype=dtype)
    return eye[first], eye[second]


def _grid_2d(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim == 1 and b.ndim == 1:
        return torch.meshgrid(a, b, indexing="ij")
    return torch.broadcast_tensors(a, b)


class FieldProjector:
    """Project equivalent surface currents onto angle, Cartesian or k-space grids.

    Args:
        currents: one equivalent-current set, or a sequence with one set per
            frequency (the leading frequency axis of every projection follows this
            order). All sets must share device and dtype.
        origin: projection origin in world coordinates (metres).
        solver, c, eps0, mu0: physical constants, as in the existing propagators;
            the package vacuum constants are used when none are given.
        background_eps_r, background_mu_r: optional override of the homogeneous
            exterior medium carried by the currents.
        device: compute device; like the other propagators this defaults to CUDA
            and needs an explicit ``device="cpu"`` for CPU postprocessing.

    All outputs are torch tensors that stay differentiable with respect to the
    currents.
    """

    def __init__(
        self,
        currents: _Currents | Sequence[_Currents],
        *,
        origin=(0.0, 0.0, 0.0),
        solver=None,
        c: float | None = None,
        eps0: float | None = None,
        mu0: float | None = None,
        background_eps_r: complex | float | None = None,
        background_mu_r: complex | float | None = None,
        device: str | torch.device | None = None,
    ):
        self.currents = _currents_sequence(currents)
        if solver is None and c is None and eps0 is None and mu0 is None:
            c, eps0, mu0 = C_0, EPSILON_0, MU_0
        options = dict(
            solver=solver,
            c=c,
            eps0=eps0,
            mu0=mu0,
            background_eps_r=background_eps_r,
            background_mu_r=background_mu_r,
            device=device,
        )
        self._far = tuple(NearFieldFarFieldTransformer(item, **options) for item in self.currents)
        self._options = options
        self._exact: dict[int, StrattonChuPropagator] = {}
        reference = self._far[0]
        self.device = reference.device
        self.coord_dtype = reference.coord_dtype
        self.field_dtype = reference.field_dtype
        for transformer in self._far[1:]:
            if transformer.coord_dtype != self.coord_dtype or transformer.field_dtype != self.field_dtype:
                raise ValueError("All equivalent-current sets must share the same dtype.")
        self.origin = _to_real_tensor(origin, device=self.device, dtype=self.coord_dtype).reshape(-1)
        if self.origin.numel() != 3:
            raise ValueError("origin must be a 3-vector.")
        self.frequencies = torch.as_tensor(
            [transformer.frequency for transformer in self._far], device=self.device, dtype=self.coord_dtype
        )
        self.wavenumbers = torch.as_tensor(
            [complex(transformer.k) for transformer in self._far], device=self.device, dtype=self.field_dtype
        )
        self.impedances = torch.as_tensor(
            [complex(transformer.eta) for transformer in self._far], device=self.device, dtype=self.field_dtype
        )

    @classmethod
    def from_monitor(
        cls,
        result,
        monitor: str | Sequence[str],
        *,
        frequencies: Sequence[float] | None = None,
        tangential_bounds: Mapping[str, tuple[float, float]] | None = None,
        window_size: tuple[float, float] | None = None,
        origin=(0.0, 0.0, 0.0),
        background_eps_r: complex | float | None = None,
        background_mu_r: complex | float | None = None,
        device: str | torch.device | None = None,
    ) -> "FieldProjector":
        """Build a projector from a closed-surface (or plane) monitor on ``Result``.

        ``monitor`` is a ``ClosedSurfaceMonitor`` / ``PlaneMonitor`` name or a
        sequence of plane-monitor names forming a closed surface. Equivalent
        currents are extracted for every frequency in ``frequencies`` (default:
        all Result frequencies), keeping the frequency axis explicit.
        """
        selected = tuple(float(value) for value in (result.frequencies if frequencies is None else frequencies))
        if not selected:
            raise ValueError("frequencies must not be empty.")
        currents = []
        for frequency in selected:
            if isinstance(monitor, str):
                currents.append(
                    equivalent_surface_currents_from_monitor(
                        result,
                        monitor,
                        frequency=frequency,
                        tangential_bounds=tangential_bounds,
                        window_size=window_size,
                    )
                )
            else:
                currents.append(
                    equivalent_surface_currents_from_monitors(
                        result,
                        monitor,
                        frequency=frequency,
                        tangential_bounds=tangential_bounds,
                        window_size=window_size,
                    )
                )
        solver = getattr(result, "solver", None)
        constants = {} if solver is not None else {"c": C_0, "eps0": EPSILON_0, "mu0": MU_0}
        return cls(
            currents,
            origin=origin,
            solver=solver,
            background_eps_r=background_eps_r,
            background_mu_r=background_mu_r,
            device=device,
            **constants,
        )

    def _exact_propagator(self, index: int) -> StrattonChuPropagator:
        if index not in self._exact:
            self._exact[index] = StrattonChuPropagator(self.currents[index], **self._options)
        return self._exact[index]

    def _real(self, values) -> torch.Tensor:
        return _to_real_tensor(values, device=self.device, dtype=self.coord_dtype)

    def _evaluate(
        self,
        points: torch.Tensor,
        *,
        far_field_approx: bool,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fields at ``points`` (``[*grid, 3]``) for every frequency."""
        shape = points.shape[:-1]
        offset = points.reshape(-1, 3) - self.origin
        radius = torch.linalg.norm(offset, dim=-1)
        if torch.any(radius <= 0.0):
            raise ValueError("Observation points must not coincide with the projection origin.")
        directions = offset / radius[:, None]
        e_fields, h_fields = [], []
        for index, transformer in enumerate(self._far):
            if far_field_approx:
                out = transformer.transform_directions(directions, radius=radius, batch_size=batch_size)
                # The transformer references phase to the world origin; shift it
                # to the projection origin: exp(i k s . o).
                phase = torch.exp(
                    1j * self.wavenumbers[index] * torch.sum(directions * self.origin, dim=-1).to(self.field_dtype)
                )[:, None]
                e_fields.append(out["E"] * phase)
                h_fields.append(out["H"] * phase)
            else:
                out = self._exact_propagator(index).propagate_points(
                    points.reshape(-1, 3), batch_size=batch_size
                )
                e_fields.append(out["E"])
                h_fields.append(out["H"])
        e_field = torch.stack(e_fields, dim=0).reshape((len(self._far),) + shape + (3,))
        h_field = torch.stack(h_fields, dim=0).reshape((len(self._far),) + shape + (3,))
        return e_field, h_field, directions.reshape(shape + (3,)), radius.reshape(shape)

    def _record(
        self,
        *,
        kind: str,
        far_field_approx: bool,
        coords: Mapping[str, torch.Tensor],
        points: torch.Tensor,
        e_field: torch.Tensor,
        h_field: torch.Tensor,
        directions: torch.Tensor,
        radius: torch.Tensor,
        theta: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> ProjectedFieldData:
        if theta is None:
            theta = torch.acos(torch.clamp(directions[..., 2], -1.0, 1.0))
            phi = torch.atan2(directions[..., 1], directions[..., 0])
        if mask is None:
            mask = torch.ones(radius.shape, device=self.device, dtype=torch.bool)
        return ProjectedFieldData(
            kind=kind,
            far_field_approx=bool(far_field_approx),
            frequencies=self.frequencies,
            wavenumbers=self.wavenumbers,
            impedances=self.impedances,
            origin=self.origin,
            coords=dict(coords),
            points=points,
            directions=directions,
            radius=radius,
            theta=theta,
            phi=phi,
            mask=mask,
            E=e_field,
            H=h_field,
        )

    def angles(
        self,
        theta,
        phi,
        *,
        radius: float = 1.0,
        far_field_approx: bool = True,
        batch_size: int = 1024,
    ) -> ProjectedFieldData:
        """Project onto a spherical ``(theta, phi)`` grid at distance ``radius``.

        One-dimensional ``theta`` and ``phi`` form a ``theta x phi`` mesh; other
        shapes are broadcast. ``far_field_approx=False`` evaluates the exact
        Stratton-Chu field at the finite radius.
        """
        theta_grid, phi_grid = _grid_2d(self._real(theta), self._real(phi))
        radius_value = float(radius)
        if radius_value <= 0.0:
            raise ValueError("radius must be positive.")
        directions = torch.stack(
            (
                torch.sin(theta_grid) * torch.cos(phi_grid),
                torch.sin(theta_grid) * torch.sin(phi_grid),
                torch.cos(theta_grid),
            ),
            dim=-1,
        )
        points = self.origin + radius_value * directions
        e_field, h_field, _, radius_grid = self._evaluate(
            points, far_field_approx=far_field_approx, batch_size=batch_size
        )
        return self._record(
            kind="angle",
            far_field_approx=far_field_approx,
            coords={"theta": theta_grid, "phi": phi_grid},
            points=points,
            e_field=e_field,
            h_field=h_field,
            directions=directions,
            radius=radius_grid,
            theta=theta_grid,
            phi=phi_grid,
        )

    def cartesian(
        self,
        x,
        y,
        *,
        axis: str = "z",
        distance: float,
        far_field_approx: bool = False,
        batch_size: int = 256,
    ) -> ProjectedFieldData:
        """Project onto a plane normal to ``axis`` at signed ``distance`` from the origin.

        ``x`` and ``y`` are local in-plane coordinates along the two remaining
        world axes in ascending order (``z``-normal: world ``x, y``;
        ``y``-normal: ``x, z``; ``x``-normal: ``y, z``), measured from the
        origin. ``far_field_approx=True`` uses the transverse ``exp(ikr)/r``
        field along each point's own direction; the default is the exact
        Stratton-Chu field, valid also close to the surface.
        """
        axis_name = _normalize_axis(axis)
        x_grid, y_grid = _grid_2d(self._real(x), self._real(y))
        normal = _unit_axis(axis_name, 1.0, device=self.device, dtype=self.coord_dtype)
        u_hat, v_hat = _tangential_unit_axes(axis_name, device=self.device, dtype=self.coord_dtype)
        points = (
            self.origin
            + float(distance) * normal
            + x_grid[..., None] * u_hat
            + y_grid[..., None] * v_hat
        )
        e_field, h_field, directions, radius_grid = self._evaluate(
            points, far_field_approx=far_field_approx, batch_size=batch_size
        )
        return self._record(
            kind="cartesian",
            far_field_approx=far_field_approx,
            coords={"x": x_grid, "y": y_grid},
            points=points,
            e_field=e_field,
            h_field=h_field,
            directions=directions,
            radius=radius_grid,
        )

    def kspace(
        self,
        ux,
        uy,
        *,
        axis: str = "z",
        normal_direction: str | int | float = "+",
        radius: float = 1.0,
        batch_size: int = 1024,
    ) -> ProjectedFieldData:
        """Project onto direction cosines ``(ux, uy)`` relative to ``axis``.

        The observation direction is ``ux u_hat + uy v_hat + s sqrt(1 - ux^2 -
        uy^2) n_hat`` with the in-plane axes ordered as in :meth:`cartesian` and
        ``s`` the sign of ``normal_direction``. Evanescent samples
        ``ux^2 + uy^2 > 1`` are masked out and their fields are zero. The
        far-field approximation is always used.
        """
        axis_name = _normalize_axis(axis)
        sign = _normalize_normal_direction(normal_direction)
        ux_grid, uy_grid = _grid_2d(self._real(ux), self._real(uy))
        radius_value = float(radius)
        if radius_value <= 0.0:
            raise ValueError("radius must be positive.")
        normal = _unit_axis(axis_name, sign, device=self.device, dtype=self.coord_dtype)
        u_hat, v_hat = _tangential_unit_axes(axis_name, device=self.device, dtype=self.coord_dtype)
        transverse_sq = ux_grid.square() + uy_grid.square()
        mask = transverse_sq <= 1.0
        normal_cosine = torch.sqrt(torch.where(mask, 1.0 - transverse_sq, torch.ones_like(transverse_sq)))
        directions = ux_grid[..., None] * u_hat + uy_grid[..., None] * v_hat + normal_cosine[..., None] * normal
        # Evanescent samples radiate along the axis as a placeholder and are zeroed below.
        directions = torch.where(mask[..., None], directions, normal.expand_as(directions))
        points = self.origin + radius_value * directions
        e_field, h_field, directions, radius_grid = self._evaluate(
            points, far_field_approx=True, batch_size=batch_size
        )
        zero = torch.zeros((), device=self.device, dtype=self.field_dtype)
        e_field = torch.where(mask[..., None], e_field, zero)
        h_field = torch.where(mask[..., None], h_field, zero)
        return self._record(
            kind="kspace",
            far_field_approx=True,
            coords={"ux": ux_grid, "uy": uy_grid},
            points=points,
            e_field=e_field,
            h_field=h_field,
            directions=directions,
            radius=radius_grid,
            mask=mask,
        )


def _payload_device(payload) -> torch.device | None:
    for face in payload.get("faces", {}).values():
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            value = face.get(name)
            if isinstance(value, torch.Tensor):
                return value.device
    return None


def project_monitor(result, monitor, *, frequency: float | None = None, freq_index: int | None = None) -> ProjectedFieldData:
    """Evaluate a declarative projection monitor on its recorded Huygens surface.

    The surface DFT fields are turned into equivalent currents exactly as
    :meth:`FieldProjector.from_monitor` does for a closed surface (including
    the homogeneous-exterior background check), then projected onto the
    monitor's angle, Cartesian or k-space grid. Without ``frequency`` /
    ``freq_index`` every monitor frequency is projected, keeping the leading
    frequency axis; the computation stays on the device of the recorded fields
    and remains differentiable with respect to them.
    """
    from ..monitors import AngleProjectionMonitor, CartesianProjectionMonitor, KSpaceProjectionMonitor
    from ..result import _resolve_frequency_index
    from .stratton_chu import _equivalent_surface_currents_from_payload, _validate_homogeneous_exterior

    available = tuple(float(value) for value in (monitor.frequencies or result.frequencies))
    index = _resolve_frequency_index(available, frequency=frequency, freq_index=freq_index)
    selected = available if index is None else (available[index],)
    currents = []
    device = None
    for value in selected:
        payload = result.raw_monitor(monitor.name, frequency=value)
        if device is None:
            device = _payload_device(payload)
        background = _validate_homogeneous_exterior(result, payload, value)
        if background is None:
            background = (1.0, 1.0)
        currents.append(
            _equivalent_surface_currents_from_payload(
                payload,
                window_size=monitor.window_size,
                background=background,
            )
        )
    solver = getattr(result, "solver", None)
    constants = {} if solver is not None else {"c": C_0, "eps0": EPSILON_0, "mu0": MU_0}
    projector = FieldProjector(currents, origin=monitor.origin, solver=solver, device=device, **constants)
    if isinstance(monitor, AngleProjectionMonitor):
        return projector.angles(
            monitor.theta, monitor.phi, radius=monitor.radius, far_field_approx=monitor.far_field_approx
        )
    if isinstance(monitor, CartesianProjectionMonitor):
        return projector.cartesian(
            monitor.x,
            monitor.y,
            axis=monitor.axis,
            distance=monitor.distance,
            far_field_approx=monitor.far_field_approx,
        )
    if isinstance(monitor, KSpaceProjectionMonitor):
        return projector.kspace(
            monitor.ux, monitor.uy, axis=monitor.axis, normal_direction=monitor.hemisphere, radius=monitor.radius
        )
    raise TypeError(f"Unsupported projection monitor {type(monitor).__name__}.")


__all__ = ["FieldProjector", "ProjectedFieldData", "project_monitor"]
