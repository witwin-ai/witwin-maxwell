"""Compiler for the gyromagnetic (Polder) ferrite layout (plan 08, slice 1b).

Lowers every :class:`~witwin.maxwell.media.GyromagneticFerrite` structure in a
scene into a structure-of-arrays (SoA) description of the active ferrite cells:
the linearized-LLG magnetization-ADE state that the FDTD runtime advances
locally at each cell (slice 1c). The gyrotropy is carried by this ADE state, not
by widening ``mu_tensor`` -- the off-diagonal ``mu_tensor`` guard stays in force.

The frozen physics contract is ``docs/reference/ferrite-physics-contract.md``.
The per-cell 2x2 state-space matrices ``P``/``Q`` and the implicit-midpoint
(Cayley) propagator ``Phi``/``Gamma`` come from the production helpers in
``media`` (``gyromagnetic_state_space``, ``gyromagnetic_update_matrices``); those
are the twins of the verification oracle in ``fdtd.ferrite_reference`` and must
agree with it bit-for-bit.

Partial-fill cells are handled by an explicit staircase (fail-closed boundary 7):
the anti-symmetric gyrotropic tensor is never scalar-averaged. A cell is an
active ferrite cell only when a ferrite structure is the topmost (highest
priority) structure covering it at or above the occupancy threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..media import (
    GyromagneticFerrite,
    gyromagnetic_local_basis,
    gyromagnetic_polder_tensor,
    gyromagnetic_state_space,
    gyromagnetic_update_matrices,
)

_STAIRCASE_THRESHOLD = 0.5


@dataclass(frozen=True)
class CompiledGyromagneticLayout:
    """SoA description of the active gyromagnetic ferrite cells.

    All per-cell arrays have leading dimension ``N`` (the active-cell count) and
    live on a single device. ``active_index`` holds the flattened node index
    (row-major over ``grid_shape``) of each active cell, so the dense mask and
    the sparse SoA are two views of the same set (the sparse/dense parity gate).

    ``local_basis[i]`` is the right-handed rotation ``R = [u|v|w]`` (lab<-local)
    whose third column ``w`` is the bias unit vector; ``m = (m_u, m_v)`` precesses
    in the transverse ``(u, v)`` plane. ``fast_axis[i]`` is ``0/1/2`` when the
    bias is ``+/-`` a Cartesian axis (the z/x/y fast path -- ``R`` is a signed
    permutation, no interpolation) and ``-1`` for a general bias.

    ``state_P``/``state_Q`` are the dt-independent 2x2 ADE matrices;
    ``phi``/``gamma`` are the dt-dependent Cayley propagator/drive, populated only
    once :meth:`with_timestep` (or ``dt=`` at compile time) fixes ``dt``.
    """

    grid_shape: tuple[int, int, int]
    active_index: torch.Tensor       # (N,) int64, flattened node index
    occupancy: torch.Tensor          # (N,) float, raw owning-structure fill fraction
    bias_unit: torch.Tensor          # (N, 3)
    local_basis: torch.Tensor        # (N, 3, 3), columns [u|v|w]
    fast_axis: torch.Tensor          # (N,) int8: 0/1/2 axis-aligned, -1 general
    omega_0: torch.Tensor            # (N,)
    omega_m: torch.Tensor            # (N,)
    gilbert_damping: torch.Tensor    # (N,)
    mu_infinity: torch.Tensor        # (N,)
    slot_owner: torch.Tensor         # (N,) int64 structure index
    state_P: torch.Tensor            # (N, 2, 2)
    state_Q: torch.Tensor            # (N, 2, 2)
    dt: float | None = None
    phi: torch.Tensor | None = None  # (N, 2, 2)
    gamma: torch.Tensor | None = None  # (N, 2, 2)

    # --- basic accessors -----------------------------------------------------

    @property
    def num_active(self) -> int:
        return int(self.active_index.shape[0])

    @property
    def enabled(self) -> bool:
        return self.num_active > 0

    @property
    def device(self) -> torch.device:
        return self.active_index.device

    @property
    def dtype(self) -> torch.dtype:
        return self.omega_0.dtype

    # --- timestep binding ----------------------------------------------------

    def with_timestep(self, dt: float) -> "CompiledGyromagneticLayout":
        """Return a copy with the Cayley ``Phi``/``Gamma`` precomputed for ``dt``.

        The propagator is frame-independent (it depends only on the scalar
        ``omega_0``/``omega_m``/``alpha``), so it is computed per unique parameter
        row and broadcast; every active cell carries its own 2x2 block so the
        runtime kernel needs no gather. Idempotent for an equal ``dt``.
        """
        dt = float(dt)
        if self.dt is not None and self.dt == dt and self.phi is not None:
            return self
        if self.num_active == 0:
            return CompiledGyromagneticLayout(
                **{**self._as_kwargs(), "dt": dt, "phi": self.state_P.clone(), "gamma": self.state_Q.clone()}
            )
        phi = torch.empty_like(self.state_P)
        gamma = torch.empty_like(self.state_Q)
        # Unique (omega_0, omega_m, alpha) rows share a propagator; compute once each.
        keys = torch.stack([self.omega_0, self.omega_m, self.gilbert_damping], dim=1)
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        for row_index in range(unique.shape[0]):
            w0, wm, alpha = (float(v) for v in unique[row_index])
            phi_row, gamma_row = gyromagnetic_update_matrices(w0, wm, alpha, dt, dtype=self.dtype)
            mask = inverse == row_index
            phi[mask] = phi_row.to(device=self.device)
            gamma[mask] = gamma_row.to(device=self.device)
        return CompiledGyromagneticLayout(**{**self._as_kwargs(), "dt": dt, "phi": phi, "gamma": gamma})

    def _as_kwargs(self) -> dict:
        return {
            "grid_shape": self.grid_shape,
            "active_index": self.active_index,
            "occupancy": self.occupancy,
            "bias_unit": self.bias_unit,
            "local_basis": self.local_basis,
            "fast_axis": self.fast_axis,
            "omega_0": self.omega_0,
            "omega_m": self.omega_m,
            "gilbert_damping": self.gilbert_damping,
            "mu_infinity": self.mu_infinity,
            "slot_owner": self.slot_owner,
            "state_P": self.state_P,
            "state_Q": self.state_Q,
        }

    # --- full-tensor mu(f) accessor -----------------------------------------

    def permeability_tensor(self, frequency, *, dtype=torch.complex128) -> torch.Tensor:
        """Full complex 3x3 Polder permeability ``mu_r(f)`` per active cell.

        Returns an ``(N, 3, 3)`` complex tensor. This is the off-diagonal accessor
        the plan flags: the x/y/z-only ``evaluate_material_components`` schema
        cannot express ``mu_xy``/``mu_yx``, so the gyromagnetic path exposes the
        full tensor here. ``frequency`` is the ordinary frequency [Hz].
        """
        if self.num_active == 0:
            return torch.empty((0, 3, 3), dtype=dtype)
        omega = 2.0 * torch.pi * torch.as_tensor(frequency, dtype=torch.float64, device=self.device)
        tensors = torch.empty((self.num_active, 3, 3), dtype=dtype, device=self.device)
        for i in range(self.num_active):
            tensors[i] = gyromagnetic_polder_tensor(
                omega,
                omega_0=float(self.omega_0[i]),
                omega_m=float(self.omega_m[i]),
                gilbert_damping=float(self.gilbert_damping[i]),
                mu_infinity=float(self.mu_infinity[i]),
                bias_unit_vector=self.bias_unit[i],
                dtype=dtype,
            ).to(self.device)
        return tensors

    # --- dense/sparse views --------------------------------------------------

    def dense_active_mask(self) -> torch.Tensor:
        """Boolean ``grid_shape`` node mask, ``True`` at every active ferrite cell."""
        mask = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device).reshape(-1)
        mask[self.active_index] = True
        return mask.reshape(self.grid_shape)

    def dense_owner(self) -> torch.Tensor:
        """Dense ``grid_shape`` int owner grid (``-1`` where no ferrite is active)."""
        owner = torch.full((int(torch.prod(torch.tensor(self.grid_shape))),), -1, dtype=torch.int64, device=self.device)
        owner[self.active_index] = self.slot_owner
        return owner.reshape(self.grid_shape)

    # --- device / serialization ---------------------------------------------

    def to(self, device) -> "CompiledGyromagneticLayout":
        device = torch.device(device)
        if self.device == device:
            return self
        moved = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in self.to_dict().items()
        }
        return CompiledGyromagneticLayout.from_dict(moved)

    def to_dict(self) -> dict:
        """Flat serializable dict of every field (tensors kept as tensors)."""
        data = self._as_kwargs()
        data["dt"] = self.dt
        data["phi"] = self.phi
        data["gamma"] = self.gamma
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CompiledGyromagneticLayout":
        return cls(
            grid_shape=tuple(int(v) for v in data["grid_shape"]),
            active_index=data["active_index"],
            occupancy=data["occupancy"],
            bias_unit=data["bias_unit"],
            local_basis=data["local_basis"],
            fast_axis=data["fast_axis"],
            omega_0=data["omega_0"],
            omega_m=data["omega_m"],
            gilbert_damping=data["gilbert_damping"],
            mu_infinity=data["mu_infinity"],
            slot_owner=data["slot_owner"],
            state_P=data["state_P"],
            state_Q=data["state_Q"],
            dt=data.get("dt"),
            phi=data.get("phi"),
            gamma=data.get("gamma"),
        )


def _empty_layout(grid_shape, device, dtype) -> CompiledGyromagneticLayout:
    zero_i = torch.empty((0,), dtype=torch.int64, device=device)
    return CompiledGyromagneticLayout(
        grid_shape=tuple(int(v) for v in grid_shape),
        active_index=zero_i,
        occupancy=torch.empty((0,), dtype=dtype, device=device),
        bias_unit=torch.empty((0, 3), dtype=dtype, device=device),
        local_basis=torch.empty((0, 3, 3), dtype=dtype, device=device),
        fast_axis=torch.empty((0,), dtype=torch.int8, device=device),
        omega_0=torch.empty((0,), dtype=dtype, device=device),
        omega_m=torch.empty((0,), dtype=dtype, device=device),
        gilbert_damping=torch.empty((0,), dtype=dtype, device=device),
        mu_infinity=torch.empty((0,), dtype=dtype, device=device),
        slot_owner=zero_i.clone(),
        state_P=torch.empty((0, 2, 2), dtype=dtype, device=device),
        state_Q=torch.empty((0, 2, 2), dtype=dtype, device=device),
    )


def compile_gyromagnetic_layout(
    scene,
    *,
    dt=None,
    device=None,
    dtype=torch.float64,
    staircase_threshold: float = _STAIRCASE_THRESHOLD,
) -> CompiledGyromagneticLayout:
    """Lower every ferrite structure into a :class:`CompiledGyromagneticLayout`.

    A cell is an active ferrite cell only where a ferrite structure is the topmost
    (highest priority, then latest index) bulk structure covering it at or above
    ``staircase_threshold`` occupancy -- an explicit staircase, never a scalar
    average of the anti-symmetric gyrotropic tensor (fail-closed boundary 7). When
    ``dt`` is given the Cayley propagator is precomputed.
    """
    from .materials import _bulk_structures, _geometry_occupancy

    target_device = torch.device(scene.device if device is None else device)
    grid_shape = (int(scene.Nx), int(scene.Ny), int(scene.Nz))

    structures = _bulk_structures(scene)
    ferrite_flags = [isinstance(getattr(s, "material", None), GyromagneticFerrite) for s in structures]
    if not any(ferrite_flags):
        return _empty_layout(grid_shape, target_device, dtype)

    # Topmost covering structure per node (staircase). Later structures in the
    # priority-sorted order overwrite earlier ones wherever they fill the cell.
    winner = torch.full(grid_shape, -1, dtype=torch.int64, device=target_device)
    fill_fraction = torch.zeros(grid_shape, dtype=dtype, device=target_device)
    for index, structure in enumerate(structures):
        occ = _geometry_occupancy(scene, structure.geometry).to(device=target_device, dtype=dtype)
        covered = occ >= staircase_threshold
        winner = torch.where(covered, torch.full_like(winner, index), winner)
        fill_fraction = torch.where(covered, occ, fill_fraction)

    ferrite_index_set = {i for i, flag in enumerate(ferrite_flags) if flag}
    active_dense = torch.zeros(grid_shape, dtype=torch.bool, device=target_device)
    for index in ferrite_index_set:
        active_dense |= winner == index
    scene.release_meshgrid()

    active_index = torch.nonzero(active_dense.reshape(-1), as_tuple=False).reshape(-1)
    num_active = int(active_index.shape[0])
    if num_active == 0:
        return _empty_layout(grid_shape, target_device, dtype)

    winner_flat = winner.reshape(-1)[active_index]
    occupancy = fill_fraction.reshape(-1)[active_index]

    # Per-active-cell scalar parameters, read (unblended) from the owning ferrite.
    omega_0 = torch.empty(num_active, dtype=dtype, device=target_device)
    omega_m = torch.empty(num_active, dtype=dtype, device=target_device)
    damping = torch.empty(num_active, dtype=dtype, device=target_device)
    mu_inf = torch.empty(num_active, dtype=dtype, device=target_device)
    bias_unit = torch.empty((num_active, 3), dtype=dtype, device=target_device)
    local_basis = torch.empty((num_active, 3, 3), dtype=dtype, device=target_device)
    fast_axis = torch.empty(num_active, dtype=torch.int8, device=target_device)
    state_P = torch.empty((num_active, 2, 2), dtype=dtype, device=target_device)
    state_Q = torch.empty((num_active, 2, 2), dtype=dtype, device=target_device)

    # Group active cells by owning structure so material params are read once.
    for index in ferrite_index_set:
        cell_mask = winner_flat == index
        count = int(cell_mask.sum())
        if count == 0:
            continue
        material: GyromagneticFerrite = structures[index].material
        b = torch.as_tensor(material.bias_unit_vector, dtype=dtype, device=target_device)
        basis = gyromagnetic_local_basis(b, dtype=dtype).to(target_device)
        p_mat, q_mat = gyromagnetic_state_space(
            material.omega_0, material.omega_m, material.gilbert_damping, dtype=dtype
        )
        p_mat = p_mat.to(target_device)
        q_mat = q_mat.to(target_device)
        omega_0[cell_mask] = float(material.omega_0)
        omega_m[cell_mask] = float(material.omega_m)
        damping[cell_mask] = float(material.gilbert_damping)
        mu_inf[cell_mask] = float(material.mu_infinity)
        bias_unit[cell_mask] = b
        local_basis[cell_mask] = basis
        fast_axis[cell_mask] = _axis_code(b)
        state_P[cell_mask] = p_mat
        state_Q[cell_mask] = q_mat

    layout = CompiledGyromagneticLayout(
        grid_shape=grid_shape,
        active_index=active_index,
        occupancy=occupancy,
        bias_unit=bias_unit,
        local_basis=local_basis,
        fast_axis=fast_axis,
        omega_0=omega_0,
        omega_m=omega_m,
        gilbert_damping=damping,
        mu_infinity=mu_inf,
        slot_owner=winner_flat.clone(),
        state_P=state_P,
        state_Q=state_Q,
    )
    if dt is not None:
        layout = layout.with_timestep(dt)
    return layout


def _axis_code(bias_unit: torch.Tensor, *, atol: float = 1.0e-9) -> int:
    """``0/1/2`` when the bias is ``+/-`` a Cartesian axis, else ``-1``."""
    axes = torch.eye(3, dtype=bias_unit.dtype, device=bias_unit.device)
    for k in range(3):
        if (
            torch.linalg.vector_norm(bias_unit - axes[k]) <= atol
            or torch.linalg.vector_norm(bias_unit + axes[k]) <= atol
        ):
            return k
    return -1
