from __future__ import annotations

import math

import torch

from ..constants import resolve_complex_dtype, resolve_real_dtype
from .stratton_chu import (
    EquivalentCurrentsSurface,
    PlanarEquivalentCurrents,
    SurfaceEquivalentCurrents,
    _as_1d_coords,
    _background_wavenumber_and_impedance,
    _normalize_currents_collection,
    _resolve_currents_background,
    _resolve_device,
    _resolve_physical_constants,
    _resolve_tensor_device,
    _to_real_tensor,
)


def _normalize_directions(directions) -> torch.Tensor:
    device = _resolve_tensor_device(directions)
    dtype = resolve_real_dtype(directions)
    vectors = _to_real_tensor(directions, device=device, dtype=dtype)
    if vectors.ndim == 1:
        if vectors.shape[0] != 3:
            raise ValueError("directions must have shape (3,) or (N, 3).")
        vectors = vectors[None, :]
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3).")

    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    if torch.any(norms <= 0.0):
        raise ValueError("directions must be non-zero vectors.")
    return vectors / norms


def _broadcast_radius(radius, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radii = _to_real_tensor(radius, device=device, dtype=dtype)
    if torch.any(radii <= 0.0):
        raise ValueError("radius must be positive.")
    return torch.broadcast_to(radii, shape).clone()


class NearFieldFarFieldTransformer:
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
        self.solver = solver
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
        self.field_dtype = resolve_complex_dtype(*(surface.J for surface in self._surfaces), *(surface.M for surface in self._surfaces))
        self.omega = 2.0 * math.pi * self.frequency
        self.eta0 = math.sqrt(self.mu0 / self.eps0)
        # Near-to-far-field radiation uses the homogeneous exterior background's
        # wavenumber and intrinsic impedance (vacuum when the box sits in free
        # space, the sampled dielectric otherwise).
        self.k, self.eta = _background_wavenumber_and_impedance(
            self.background_eps_r,
            self.background_mu_r,
            omega=self.omega,
            c=self.c,
            eta_vacuum=self.eta0,
        )

        source_points = []
        weighted_j = []
        weighted_m = []
        for surface in self._surfaces:
            points, surface_j, surface_m = surface.quadrature()
            source_points.append(points.reshape(-1, 3))
            weighted_j.append(surface_j.reshape(-1, 3))
            weighted_m.append(surface_m.reshape(-1, 3))

        self._src_points = torch.cat(source_points, dim=0).to(device=self.device, dtype=self.coord_dtype)
        self._J_w = torch.cat(weighted_j, dim=0).to(device=self.device, dtype=self.field_dtype)
        self._M_w = torch.cat(weighted_m, dim=0).to(device=self.device, dtype=self.field_dtype)

    def _transform_batch(
        self,
        directions: torch.Tensor,
        radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction_real = directions.to(device=self.device, dtype=self.coord_dtype)
        radius_real = radius.to(device=self.device, dtype=self.coord_dtype)
        direction_complex = direction_real.to(dtype=self.field_dtype)
        phase = torch.exp(
            -1j * self.k * torch.matmul(direction_real, self._src_points.transpose(0, 1)).to(dtype=self.field_dtype)
        )
        n_vector = torch.matmul(phase, self._J_w)
        l_vector = torch.matmul(phase, self._M_w)

        cross_s_n = torch.cross(direction_complex, n_vector, dim=-1)
        cross_s_l = torch.cross(direction_complex, l_vector, dim=-1)
        radius_complex = radius_real.to(dtype=self.field_dtype)
        prefactor = (
            -1j
            * self.k
            * torch.exp(1j * self.k * radius_complex)
            / (4.0 * math.pi * radius_complex)
        )[:, None]

        e_field = prefactor * (
            self.eta * torch.cross(direction_complex, cross_s_n, dim=-1)
            + cross_s_l
        )
        h_field = prefactor * (
            (1.0 / self.eta) * torch.cross(direction_complex, cross_s_l, dim=-1)
            - cross_s_n
        )
        return e_field, h_field

    def transform_directions(
        self,
        directions,
        *,
        radius: float | torch.Tensor = 1.0,
        batch_size: int = 1024,
    ) -> dict[str, torch.Tensor | float]:
        direction_tensor = _normalize_directions(directions).to(device=self.device, dtype=self.coord_dtype)
        radius_tensor = _broadcast_radius(
            radius,
            direction_tensor.shape[:-1],
            device=self.device,
            dtype=self.coord_dtype,
        ).reshape(-1)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        e_batches = []
        h_batches = []
        for start in range(0, direction_tensor.shape[0], batch_size):
            stop = min(start + batch_size, direction_tensor.shape[0])
            e_batch, h_batch = self._transform_batch(
                direction_tensor[start:stop],
                radius_tensor[start:stop],
            )
            e_batches.append(e_batch)
            h_batches.append(h_batch)

        e_field = torch.cat(e_batches, dim=0)
        h_field = torch.cat(h_batches, dim=0)
        poynting = 0.5 * torch.real(torch.cross(e_field, torch.conj(h_field), dim=-1))
        power_density = torch.sum(poynting * direction_tensor, dim=-1)

        return {
            "directions": direction_tensor,
            "radius": radius_tensor,
            "frequency": self.frequency,
            "E": e_field,
            "H": h_field,
            "Ex": e_field[:, 0],
            "Ey": e_field[:, 1],
            "Ez": e_field[:, 2],
            "Hx": h_field[:, 0],
            "Hy": h_field[:, 1],
            "Hz": h_field[:, 2],
            "power_density": power_density,
        }

    def transform(
        self,
        theta,
        phi,
        *,
        radius: float | torch.Tensor = 1.0,
        batch_size: int = 1024,
    ) -> dict[str, torch.Tensor | float]:
        device = _resolve_tensor_device(theta, phi)
        dtype = resolve_real_dtype(theta, phi)
        theta_tensor = _to_real_tensor(theta, device=device, dtype=dtype)
        phi_tensor = _to_real_tensor(phi, device=device, dtype=dtype)
        theta_grid, phi_grid = torch.broadcast_tensors(theta_tensor, phi_tensor)
        direction_tensor = torch.stack(
            (
                torch.sin(theta_grid) * torch.cos(phi_grid),
                torch.sin(theta_grid) * torch.sin(phi_grid),
                torch.cos(theta_grid),
            ),
            dim=-1,
        )
        radius_tensor = _broadcast_radius(
            radius,
            theta_grid.shape,
            device=device,
            dtype=dtype,
        )
        transformed = self.transform_directions(
            direction_tensor.reshape(-1, 3),
            radius=radius_tensor.reshape(-1),
            batch_size=batch_size,
        )

        shape = theta_grid.shape
        directions = transformed["directions"].reshape(shape + (3,))
        e_field = transformed["E"].reshape(shape + (3,))
        h_field = transformed["H"].reshape(shape + (3,))
        theta_on_device = theta_grid.to(device=self.device, dtype=self.coord_dtype)
        phi_on_device = phi_grid.to(device=self.device, dtype=self.coord_dtype)

        e_theta_hat = torch.stack(
            (
                torch.cos(theta_on_device) * torch.cos(phi_on_device),
                torch.cos(theta_on_device) * torch.sin(phi_on_device),
                -torch.sin(theta_on_device),
            ),
            dim=-1,
        ).to(dtype=self.field_dtype)
        e_phi_hat = torch.stack(
            (
                -torch.sin(phi_on_device),
                torch.cos(phi_on_device),
                torch.zeros_like(phi_on_device),
            ),
            dim=-1,
        ).to(dtype=self.field_dtype)

        e_theta = torch.sum(e_field * e_theta_hat, dim=-1)
        e_phi = torch.sum(e_field * e_phi_hat, dim=-1)
        h_theta = torch.sum(h_field * e_theta_hat, dim=-1)
        h_phi = torch.sum(h_field * e_phi_hat, dim=-1)

        return {
            "theta": theta_on_device,
            "phi": phi_on_device,
            "radius": radius_tensor.to(device=self.device, dtype=self.coord_dtype),
            "directions": directions,
            "frequency": self.frequency,
            "E": e_field,
            "H": h_field,
            "Ex": e_field[..., 0],
            "Ey": e_field[..., 1],
            "Ez": e_field[..., 2],
            "Hx": h_field[..., 0],
            "Hy": h_field[..., 1],
            "Hz": h_field[..., 2],
            "E_theta": e_theta,
            "E_phi": e_phi,
            "H_theta": h_theta,
            "H_phi": h_phi,
            "power_density": transformed["power_density"].reshape(shape),
        }
