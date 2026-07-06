import torch
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupyx.scipy.sparse.linalg import cg as cupy_cg, gmres as cupy_gmres
import time
import cupyx.scipy.sparse.linalg as cupy_linalg
from tqdm import tqdm

from ..compiler.materials import _scene_has_dispersive_material
from ..compiler.sources import compile_fdfd_sources
from ..scene import prepare_scene
from ..sources import CW, POINT_DIPOLE_IDEAL_PROFILE_SCALE, POINT_DIPOLE_REFERENCE_WIDTH, PointDipole
from .plotting import (
    plot_cross_section as plot_cross_section_impl,
    plot_isotropic_3views as plot_isotropic_3views_impl,
)
from .postprocess import (
    get_field_raw as get_field_raw_impl,
    interpolate_yee_to_center as interpolate_yee_to_center_impl,
)

def print_gpu_memory(prefix=""):
    """Print current GPU memory usage."""
    import cupy as cp
    free, total = cp.cuda.Device(0).mem_info
    used = total - free
    print(f"{prefix} GPU Memory Usage: Used {used/1024**3:.2f} GB / Total {total/1024**3:.2f} GB")


def _normalize_point_dipole_profile_gpu(profile, dist_sq, width, mask=None):
    if np.isclose(width, POINT_DIPOLE_REFERENCE_WIDTH):
        return cp.where(mask, profile, 0.0) if mask is not None else profile

    reference_width_sq = 2.0 * POINT_DIPOLE_REFERENCE_WIDTH ** 2
    if mask is not None:
        profile = cp.where(mask, profile, 0.0)
        reference_profile = cp.where(mask, cp.exp(-dist_sq / reference_width_sq), 0.0)
    else:
        reference_profile = cp.exp(-dist_sq / reference_width_sq)
    current_sum = cp.maximum(profile.sum(), np.finfo(np.float32).eps)
    return profile * (reference_profile.sum() / current_sum)


def _nearest_axis_index_gpu(coords, center):
    return int(cp.abs(coords - center).argmin())


def _support_slice_gpu(coords, center, cutoff):
    indices = cp.where(cp.abs(coords - center) <= cutoff)[0]
    if int(indices.size) == 0:
        nearest = _nearest_axis_index_gpu(coords, center)
        return nearest, nearest + 1
    return int(indices[0]), int(indices[-1]) + 1


def _add_point_dipole_component_gpu(b_component, x_coords, y_coords, z_coords,
                                    position, amplitude, width, profile_kind):
    """Accumulate one polarization component of a point dipole into the 3D view
    of the source vector, restricted to the cutoff support slice."""
    if profile_kind == 'ideal':
        ix = _nearest_axis_index_gpu(x_coords, position[0])
        iy = _nearest_axis_index_gpu(y_coords, position[1])
        iz = _nearest_axis_index_gpu(z_coords, position[2])
        b_component[ix, iy, iz] += amplitude * _ideal_point_dipole_mass_gpu(
            x_coords, y_coords, z_coords, position
        )
        return

    cutoff = 3 * max(width, 0.5 * POINT_DIPOLE_REFERENCE_WIDTH)
    bounds = []
    for coords, center in ((x_coords, position[0]), (y_coords, position[1]), (z_coords, position[2])):
        start, end = _support_slice_gpu(coords, center, cutoff)
        if end - start == 1 and float(cp.abs(coords[start] - center)) > cutoff:
            return  # no grid point within the cutoff: zero contribution
        bounds.append((start, end))
    (ix0, ix1), (iy0, iy1), (iz0, iz1) = bounds

    local_x = x_coords[ix0:ix1]
    local_y = y_coords[iy0:iy1]
    local_z = z_coords[iz0:iz1]
    dist_sq = (
        (local_x[:, None, None] - position[0]) ** 2
        + (local_y[None, :, None] - position[1]) ** 2
        + (local_z[None, None, :] - position[2]) ** 2
    )
    profile = cp.exp(-dist_sq / (2.0 * width ** 2))
    profile = _normalize_point_dipole_profile_gpu(profile, dist_sq, width)
    b_component[ix0:ix1, iy0:iy1, iz0:iz1] += amplitude * profile


def _ideal_point_dipole_mass_gpu(x_coords, y_coords, z_coords, position):
    cutoff = 3.0 * POINT_DIPOLE_REFERENCE_WIDTH
    ix_start, ix_end = _support_slice_gpu(x_coords, position[0], cutoff)
    iy_start, iy_end = _support_slice_gpu(y_coords, position[1], cutoff)
    iz_start, iz_end = _support_slice_gpu(z_coords, position[2], cutoff)
    local_x = x_coords[ix_start:ix_end]
    local_y = y_coords[iy_start:iy_end]
    local_z = z_coords[iz_start:iz_end]
    dist_sq = (
        (local_x[:, None, None] - position[0]) ** 2
        + (local_y[None, :, None] - position[1]) ** 2
        + (local_z[None, None, :] - position[2]) ** 2
    )
    reference_width_sq = 2.0 * POINT_DIPOLE_REFERENCE_WIDTH ** 2
    return POINT_DIPOLE_IDEAL_PROFILE_SCALE * cp.exp(-dist_sq / reference_width_sq).sum()


def _require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDFD requires scene.device to be CUDA, got {device}.")
    if not cp.cuda.is_available():
        raise RuntimeError("FDFD requires CUDA/CuPy, but no CUDA-capable device is available.")


def _require_uniform_grid(scene):
    if getattr(scene, "grid", None) is None:
        return
    if not bool(getattr(scene.grid, "is_uniform", False)):
        raise ValueError("FDFD currently requires a uniform GridSpec with dx == dy == dz.")


def _is_nonvacuum_magnetic_material(material) -> bool:
    if bool(getattr(material, "mu_debye_poles", ()) or getattr(material, "mu_drude_poles", ()) or getattr(material, "mu_lorentz_poles", ())):
        return True
    if getattr(material, "mu_tensor", None) is not None:
        return True
    return not np.isclose(float(getattr(material, "mu_r", 1.0)), 1.0)


def _validate_supported_fdfd_materials(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if bool(getattr(material, "is_nonlinear", False)):
            raise NotImplementedError("FDFD does not support Kerr nonlinear media in v1.")
        if _is_nonvacuum_magnetic_material(material):
            raise NotImplementedError(
                "FDFD currently supports electric anisotropy only; magnetic media and magnetic dispersion are not implemented yet."
            )



class FDFD:
    """
    Full-vector 3D FDFD solver based on the Yee staggered grid.
    Runs on CUDA via CuPy sparse linear algebra and iterative solvers.
    """
    def __init__(self, scene, frequency=2.0e9, solver_type='bicgstab', enable_plot=True, tqdm_position=None, verbose=False):
        """
        Initialize the 3D FDFD solver.

        Args:
            scene: Scene instance
            frequency: Operating frequency (Hz)
            solver_type: Solver type ('cg', 'bicgstab', 'gmres')
            enable_plot: Whether to enable plotting (set False for batch data generation)
            tqdm_position: tqdm progress bar position (for multi-GPU parallelism, set different values per GPU)
            verbose: Whether to print detailed debug information
        """
        self.scene = prepare_scene(scene)
        self.frequency = frequency
        self.enable_plot = enable_plot
        self.tqdm_position = tqdm_position
        self.verbose = verbose

        unsupported_face_kinds = sorted(
            {
                self.scene.boundary.face_kind(axis, side)
                for axis in ("x", "y", "z")
                for side in ("low", "high")
                if self.scene.boundary.face_kind(axis, side) not in {"none", "pml"}
            }
        )
        if unsupported_face_kinds:
            raise ValueError(
                "FDFD currently supports only per-face 'none' and 'pml' boundaries."
            )
        if getattr(self.scene, "has_symmetry", False):
            raise ValueError("FDFD currently does not support Scene(symmetry=...).")
        _require_cuda_scene(self.scene)
        _require_uniform_grid(self.scene)
        self.use_gpu = True

        if solver_type not in ['cg', 'bicgstab', 'gmres', 'direct']:
            raise ValueError("solver_type must be 'cg', 'bicgstab', 'gmres', or 'direct'")
        self.solver_type = solver_type

        # Physical constants
        self.c0 = 299792458.0
        self.mu0 = 4 * np.pi * 1e-7
        self.eps0 = 1 / (self.mu0 * self.c0 ** 2)

        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / self.c0

        self.E_field = None  # Stores (Ex, Ey, Ez) as torch tensors
        self.E_field_raw = None  # Raw Yee grid data generated on demand {'Ex': np.ndarray, 'Ey': np.ndarray, 'Ez': np.ndarray}
        self.A_matrix = None
        self._matrix_frequency = None  # Frequency the cached A_matrix was built for
        self.material_eps_r = None
        self.material_mu_r = None
        self.material_eps_components = None
        self.material_mu_components = None

        _validate_supported_fdfd_materials(self.scene)

        if self.verbose:
            print("Initializing full-vector FDFD solver based on Yee grid")
            print(f"Using solver: {solver_type.upper()}")
            print("Using device: GPU")

    def set_frequency(self, frequency):
        """Switch operating frequency, invalidating the cached system matrix.

        Material components are reused when all scene materials are
        non-dispersive; dispersive scenes recompile them on the next solve.
        """
        frequency = float(frequency)
        if frequency == self.frequency:
            return
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / self.c0
        self.A_matrix = None
        self._matrix_frequency = None
        if _scene_has_dispersive_material(self.scene):
            self.material_eps_components = None
            self.material_mu_components = None
            self.material_eps_r = None
            self.material_mu_r = None

    def _ensure_material_components(self):
        if self.material_eps_components is not None:
            return
        self.material_eps_components, self.material_mu_components = self.scene.compile_material_components(
            frequency=self.frequency
        )
        self.material_eps_r = (
            self.material_eps_components["x"]
            + self.material_eps_components["y"]
            + self.material_eps_components["z"]
        ) / 3.0
        self.material_mu_r = (
            self.material_mu_components["x"]
            + self.material_mu_components["y"]
            + self.material_mu_components["z"]
        ) / 3.0

    def _ensure_system_matrix(self):
        """Build the system matrix for the current frequency, reusing the
        cached one when available. Only b changes across sources, so repeated
        solves and frequency-swept non-dispersive runs skip reassembly."""
        if self.A_matrix is not None and self._matrix_frequency == self.frequency:
            return
        self._ensure_material_components()
        self.scene.release_meshgrid()
        self.A_matrix = self._build_matrix_gpu_yee_3d()
        self._matrix_frequency = self.frequency

    def _create_pml_3d(self):
        """Create PML complex stretching factors on main grid nodes"""
        Nx, Ny, Nz = self.scene.Nx, self.scene.Ny, self.scene.Nz
        s_x = torch.ones(Nx, dtype=torch.complex64, device=self.scene.device)
        s_y = torch.ones(Ny, dtype=torch.complex64, device=self.scene.device)
        s_z = torch.ones(Nz, dtype=torch.complex64, device=self.scene.device)

        if self.scene.boundary.uses_kind("pml"):
            pml_thickness = self.scene.pml_thickness
            pml_strength = self.scene.pml_strength
            # Quadratic grading profile
            profile = torch.linspace(0, 1, pml_thickness, device=self.scene.device)**2 * pml_strength
            if self.scene.boundary_face_kind("x", "low") == "pml":
                width = min(pml_thickness, Nx)
                s_x[:width] = 1 + 1j * torch.flip(profile[:width], [0])
            if self.scene.boundary_face_kind("x", "high") == "pml":
                width = min(pml_thickness, Nx)
                s_x[-width:] = 1 + 1j * profile[:width]
            if self.scene.boundary_face_kind("y", "low") == "pml":
                width = min(pml_thickness, Ny)
                s_y[:width] = 1 + 1j * torch.flip(profile[:width], [0])
            if self.scene.boundary_face_kind("y", "high") == "pml":
                width = min(pml_thickness, Ny)
                s_y[-width:] = 1 + 1j * profile[:width]
            if self.scene.boundary_face_kind("z", "low") == "pml":
                width = min(pml_thickness, Nz)
                s_z[:width] = 1 + 1j * torch.flip(profile[:width], [0])
            if self.scene.boundary_face_kind("z", "high") == "pml":
                width = min(pml_thickness, Nz)
                s_z[-width:] = 1 + 1j * profile[:width]

        return s_x, s_y, s_z

    def _build_matrix_gpu_yee_3d(self):
        """
        [CORRECTED] Build the 3D full-vector coefficient matrix for the Yee grid using CuPy on GPU.
        This version fixes sign errors in the original code.
        """
        ds = float(self.scene.dx)
        k0 = self.k0

        # ... (preceding code unchanged) ...

        Nx, Ny, Nz = self.scene.Nx, self.scene.Ny, self.scene.Nz
        Nx_ex, Ny_ex, Nz_ex = self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex
        Nx_ey, Ny_ey, Nz_ey = self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey
        Nx_ez, Ny_ez, Nz_ez = self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez
        N_ex, N_ey, N_ez = self.scene.N_ex, self.scene.N_ey, self.scene.N_ez
        N_vec = self.scene.N_vector_total

        if self.verbose:
            print("Preparing PML and permittivity for Yee grid...")
        s_x_main, s_y_main, s_z_main = self._create_pml_3d()
        s_x_main, s_y_main, s_z_main = cp.asarray(s_x_main), cp.asarray(s_y_main), cp.asarray(s_z_main)
        s_x_half = (s_x_main[:-1] + s_x_main[1:]) / 2
        s_y_half = (s_y_main[:-1] + s_y_main[1:]) / 2
        s_z_half = (s_z_main[:-1] + s_z_main[1:]) / 2
        self._ensure_material_components()
        eps_x_main = cp.asarray(self.material_eps_components["x"])
        eps_y_main = cp.asarray(self.material_eps_components["y"])
        eps_z_main = cp.asarray(self.material_eps_components["z"])
        eps_r_x = (eps_x_main[:-1, :, :] + eps_x_main[1:, :, :]) / 2
        eps_r_y = (eps_y_main[:, :-1, :] + eps_y_main[:, 1:, :]) / 2
        eps_r_z = (eps_z_main[:, :, :-1] + eps_z_main[:, :, 1:]) / 2

        if self.verbose:
            print("Building 3D Yee grid sparse matrix...")
        # nnz estimate: up to 13 nonzero elements per equation
        # (1 diagonal + 4 same-field neighbors + 4 Ey/Ex coupling + 4 Ez/Ex coupling)
        max_nnz = N_vec * 13
        rows = cp.zeros(max_nnz, dtype=cp.int32)
        cols = cp.zeros(max_nnz, dtype=cp.int32)
        data = cp.zeros(max_nnz, dtype=cp.complex64)
        nnz = 0

        ds2 = ds**2
        k02 = k0**2

        def add_entries(row_idx, col_idx, vals):
            nonlocal nnz
            # Use broadcast_arrays to ensure all arrays have consistent shapes
            row_idx, col_idx, vals = cp.broadcast_arrays(row_idx, col_idx, vals)
            count = vals.size
            if nnz + count > max_nnz:
                raise ValueError(f"max_nnz exceeded: need {nnz + count}, have {max_nnz}")
            rows[nnz:nnz+count] = row_idx.ravel()
            cols[nnz:nnz+count] = col_idx.ravel()
            data[nnz:nnz+count] = vals.ravel()
            nnz += count

        # --- 3. Fill Ex equations --- (using broadcasting instead of meshgrid to save memory)
        ix_1d = cp.arange(Nx_ex, dtype=cp.int32)
        iy_1d = cp.arange(Ny_ex, dtype=cp.int32)
        iz_1d = cp.arange(Nz_ex, dtype=cp.int32)
        # Broadcasting: (Nx_ex,1,1), (1,Ny_ex,1), (1,1,Nz_ex)
        idx_ex = (ix_1d[:, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + iz_1d[None, None, :]
        sy = s_y_main[iy_1d][None, :, None]
        sz = s_z_main[iz_1d][None, None, :]
        sx_half = s_x_half[ix_1d][:, None, None]
        diag_val = (k02 * eps_r_x - 2 / (sy**2 * ds2) - 2 / (sz**2 * ds2))
        add_entries(idx_ex, idx_ex, diag_val)

        # Neighbor indices (using slicing to avoid creating masks)
        # iy > 0: idx_ex[:, 1:, :]
        add_entries(idx_ex[:, 1:, :], (ix_1d[:, None, None] * Ny_ex + (iy_1d[None, 1:, None]-1)) * Nz_ex + iz_1d[None, None, :], 1 / (sy[:, 1:, :]**2 * ds2))
        # iy < Ny_ex - 1: idx_ex[:, :-1, :]
        add_entries(idx_ex[:, :-1, :], (ix_1d[:, None, None] * Ny_ex + (iy_1d[None, :-1, None]+1)) * Nz_ex + iz_1d[None, None, :], 1 / (sy[:, :-1, :]**2 * ds2))
        # iz > 0: idx_ex[:, :, 1:]
        add_entries(idx_ex[:, :, 1:], (ix_1d[:, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, 1:]-1), 1 / (sz[:, :, 1:]**2 * ds2))
        # iz < Nz_ex - 1: idx_ex[:, :, :-1]
        add_entries(idx_ex[:, :, :-1], (ix_1d[:, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, :-1]+1), 1 / (sz[:, :, :-1]**2 * ds2))

        # Coupling to Ey terms (d^2/dxdy)
        val = 1 / (sx_half * sy * ds2)
        # iy > 0
        add_entries(idx_ex[:, 1:, :], N_ex + (ix_1d[:, None, None] * Ny_ey + (iy_1d[None, 1:, None]-1)) * Nz_ey + iz_1d[None, None, :], -val[:, 1:, :])
        add_entries(idx_ex[:, 1:, :], N_ex + ((ix_1d[:, None, None]+1) * Ny_ey + (iy_1d[None, 1:, None]-1)) * Nz_ey + iz_1d[None, None, :], val[:, 1:, :])
        # iy < Ny_ey
        add_entries(idx_ex[:, :Ny_ey, :], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :Ny_ey, None]) * Nz_ey + iz_1d[None, None, :], val[:, :Ny_ey, :])
        add_entries(idx_ex[:, :Ny_ey, :], N_ex + ((ix_1d[:, None, None]+1) * Ny_ey + iy_1d[None, :Ny_ey, None]) * Nz_ey + iz_1d[None, None, :], -val[:, :Ny_ey, :])

        # Coupling to Ez terms (d^2/dxdz)
        val = 1 / (sx_half * sz * ds2)
        # iz > 0
        add_entries(idx_ex[:, :, 1:], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + (iz_1d[None, None, 1:]-1), -val[:, :, 1:])
        add_entries(idx_ex[:, :, 1:], N_ex + N_ey + ((ix_1d[:, None, None]+1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + (iz_1d[None, None, 1:]-1), val[:, :, 1:])
        # iz < Nz_ez
        add_entries(idx_ex[:, :, :Nz_ez], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :Nz_ez], val[:, :, :Nz_ez])
        add_entries(idx_ex[:, :, :Nz_ez], N_ex + N_ey + ((ix_1d[:, None, None]+1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :Nz_ez], -val[:, :, :Nz_ez])
        del ix_1d, iy_1d, iz_1d, idx_ex, sy, sz, sx_half, val  # Free memory

        # --- 4. Fill Ey equations --- (using broadcasting instead of meshgrid to save memory)
        ix_1d = cp.arange(Nx_ey, dtype=cp.int32)
        iy_1d = cp.arange(Ny_ey, dtype=cp.int32)
        iz_1d = cp.arange(Nz_ey, dtype=cp.int32)
        idx_ey = N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :]
        sx = s_x_main[ix_1d][:, None, None]
        sz = s_z_main[iz_1d][None, None, :]
        sy_half = s_y_half[iy_1d][None, :, None]
        diag_val = (k02 * eps_r_y - 2 / (sx**2 * ds2) - 2 / (sz**2 * ds2))
        add_entries(idx_ey, idx_ey, diag_val)

        # Neighbor indices
        add_entries(idx_ey[1:, :, :], N_ex + ((ix_1d[1:, None, None]-1) * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :], 1 / (sx[1:, :, :]**2 * ds2))
        add_entries(idx_ey[:-1, :, :], N_ex + ((ix_1d[:-1, None, None]+1) * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :], 1 / (sx[:-1, :, :]**2 * ds2))
        add_entries(idx_ey[:, :, 1:], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + (iz_1d[None, None, 1:]-1), 1 / (sz[:, :, 1:]**2 * ds2))
        add_entries(idx_ey[:, :, :-1], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + (iz_1d[None, None, :-1]+1), 1 / (sz[:, :, :-1]**2 * ds2))

        # Coupling to Ex (d^2/dydx)
        val = 1 / (sy_half * sx * ds2)
        # ix > 0
        add_entries(idx_ey[1:, :, :], ((ix_1d[1:, None, None]-1) * Ny_ex + iy_1d[None, :, None]) * Nz_ex + iz_1d[None, None, :], -val[1:, :, :])
        add_entries(idx_ey[1:, :, :], ((ix_1d[1:, None, None]-1) * Ny_ex + (iy_1d[None, :, None]+1)) * Nz_ex + iz_1d[None, None, :], val[1:, :, :])
        # ix < Nx_ex
        add_entries(idx_ey[:Nx_ex, :, :], (ix_1d[:Nx_ex, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + iz_1d[None, None, :], val[:Nx_ex, :, :])
        add_entries(idx_ey[:Nx_ex, :, :], (ix_1d[:Nx_ex, None, None] * Ny_ex + (iy_1d[None, :, None]+1)) * Nz_ex + iz_1d[None, None, :], -val[:Nx_ex, :, :])

        # Coupling to Ez (d^2/dydz)
        val = 1 / (sy_half * sz * ds2)
        # iz > 0
        add_entries(idx_ey[:, :, 1:], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + (iz_1d[None, None, 1:]-1), -val[:, :, 1:])
        add_entries(idx_ey[:, :, 1:], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, :, None]+1)) * Nz_ez + (iz_1d[None, None, 1:]-1), val[:, :, 1:])
        # iz < Nz_ez
        add_entries(idx_ey[:, :, :Nz_ez], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :Nz_ez], val[:, :, :Nz_ez])
        add_entries(idx_ey[:, :, :Nz_ez], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, :, None]+1)) * Nz_ez + iz_1d[None, None, :Nz_ez], -val[:, :, :Nz_ez])
        del ix_1d, iy_1d, iz_1d, idx_ey, sx, sz, sy_half, val

        # --- 5. Fill Ez equations --- (using broadcasting instead of meshgrid to save memory)
        ix_1d = cp.arange(Nx_ez, dtype=cp.int32)
        iy_1d = cp.arange(Ny_ez, dtype=cp.int32)
        iz_1d = cp.arange(Nz_ez, dtype=cp.int32)
        idx_ez = N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :]
        sx = s_x_main[ix_1d][:, None, None]
        sy = s_y_main[iy_1d][None, :, None]
        sz_half = s_z_half[iz_1d][None, None, :]
        diag_val = (k02 * eps_r_z - 2 / (sx**2 * ds2) - 2 / (sy**2 * ds2))
        add_entries(idx_ez, idx_ez, diag_val)

        # Neighbor indices
        add_entries(idx_ez[1:, :, :], N_ex + N_ey + ((ix_1d[1:, None, None]-1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :], 1 / (sx[1:, :, :]**2 * ds2))
        add_entries(idx_ez[:-1, :, :], N_ex + N_ey + ((ix_1d[:-1, None, None]+1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :], 1 / (sx[:-1, :, :]**2 * ds2))
        add_entries(idx_ez[:, 1:, :], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, 1:, None]-1)) * Nz_ez + iz_1d[None, None, :], 1 / (sy[:, 1:, :]**2 * ds2))
        add_entries(idx_ez[:, :-1, :], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, :-1, None]+1)) * Nz_ez + iz_1d[None, None, :], 1 / (sy[:, :-1, :]**2 * ds2))

        # Coupling to Ex (d^2/dzdx)
        val = 1 / (sz_half * sx * ds2)
        # ix > 0
        add_entries(idx_ez[1:, :, :], ((ix_1d[1:, None, None]-1) * Ny_ex + iy_1d[None, :, None]) * Nz_ex + iz_1d[None, None, :], -val[1:, :, :])
        add_entries(idx_ez[1:, :, :], ((ix_1d[1:, None, None]-1) * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, :]+1), val[1:, :, :])
        # ix < Nx_ex
        add_entries(idx_ez[:Nx_ex, :, :], (ix_1d[:Nx_ex, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + iz_1d[None, None, :], val[:Nx_ex, :, :])
        add_entries(idx_ez[:Nx_ex, :, :], (ix_1d[:Nx_ex, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, :]+1), -val[:Nx_ex, :, :])

        # Coupling to Ey (d^2/dzdy)
        val = 1 / (sz_half * sy * ds2)
        # iy > 0
        add_entries(idx_ez[:, 1:, :], N_ex + (ix_1d[:, None, None] * Ny_ey + (iy_1d[None, 1:, None]-1)) * Nz_ey + iz_1d[None, None, :], -val[:, 1:, :])
        add_entries(idx_ez[:, 1:, :], N_ex + (ix_1d[:, None, None] * Ny_ey + (iy_1d[None, 1:, None]-1)) * Nz_ey + (iz_1d[None, None, :]+1), val[:, 1:, :])
        # iy < Ny_ey
        add_entries(idx_ez[:, :Ny_ey, :], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :Ny_ey, None]) * Nz_ey + iz_1d[None, None, :], val[:, :Ny_ey, :])
        add_entries(idx_ez[:, :Ny_ey, :], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :Ny_ey, None]) * Nz_ey + (iz_1d[None, None, :]+1), -val[:, :Ny_ey, :])
        del ix_1d, iy_1d, iz_1d, idx_ez, sx, sy, sz_half, val

        # ... (following code unchanged) ...

        if self.verbose:
            print(f"Sparse matrix nonzero count: {nnz:,}")
            print(f"Sparsity: {nnz / (N_vec * N_vec) * 100:.6f}%")
        A = cpsp.coo_matrix((data[:nnz], (rows[:nnz], cols[:nnz])), shape=(N_vec, N_vec))
        return A.tocsr()

    def _build_source_vector_yee(self):
        """Build the total b vector for the Yee grid from the scene source list. Uses broadcasting instead of meshgrid to save memory."""
        b = cp.zeros(self.scene.N_vector_total, dtype=cp.complex64)
        compiled_sources = compile_fdfd_sources(self.scene, default_frequency=self.frequency)
        if not compiled_sources:
            return b

        dx, dy, dz = self.scene.grid_spacing
        const = -1j * self.omega * self.mu0

        # Prepare coordinate data on GPU (1D arrays, using broadcasting)
        x_ex = cp.asarray(self.scene.x[:-1] + dx / 2.0, dtype=cp.float32)
        y_ex = cp.asarray(self.scene.y, dtype=cp.float32)
        z_ex = cp.asarray(self.scene.z, dtype=cp.float32)

        x_ey = cp.asarray(self.scene.x, dtype=cp.float32)
        y_ey = cp.asarray(self.scene.y[:-1] + dy / 2.0, dtype=cp.float32)
        z_ey = cp.asarray(self.scene.z, dtype=cp.float32)

        x_ez = cp.asarray(self.scene.x, dtype=cp.float32)
        y_ez = cp.asarray(self.scene.y, dtype=cp.float32)
        z_ez = cp.asarray(self.scene.z[:-1] + dz / 2.0, dtype=cp.float32)

        # 3D views into b, one per field component (reshape of contiguous slices)
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        b_ex = b[:N_ex].reshape(self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex)
        b_ey = b[N_ex:N_ex + N_ey].reshape(self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey)
        b_ez = b[N_ex + N_ey:].reshape(self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez)

        for source in compiled_sources:
            pos = source['position']
            width = float(source['width'])
            profile_kind = source.get('profile', 'gaussian')
            pol = source['polarization']

            if pol[0] != 0:
                _add_point_dipole_component_gpu(b_ex, x_ex, y_ex, z_ex, pos,
                                                const * pol[0], width, profile_kind)
            if pol[1] != 0:
                _add_point_dipole_component_gpu(b_ey, x_ey, y_ey, z_ey, pos,
                                                const * pol[1], width, profile_kind)
            if pol[2] != 0:
                _add_point_dipole_component_gpu(b_ez, x_ez, y_ez, z_ez, pos,
                                                const * pol[2], width, profile_kind)

        return b

    def solve(self, max_iter=1000, tol=1e-6, restart=100):
        start_time = time.time()

        self._ensure_system_matrix()
        b = self._build_source_vector_yee()
        matrix_time = time.time() - start_time
        if self.verbose:
            print(f"GPU matrix and source build time: {matrix_time:.2f}s")
            print(f"Solving linear system with GPU {self.solver_type.upper()} solver...")
            print_gpu_memory("Before solve")
        solve_start = time.time()

        try:
            if self.solver_type == 'cg':
                x_gpu, info = cupy_cg(self.A_matrix, b, maxiter=max_iter, tol=tol)
            elif self.solver_type == 'bicgstab':
                # Use bicgstab from cupyx.scipy.sparse.linalg
                from cupyx.scipy.sparse.linalg import bicgstab as cupy_bicgstab
                x_gpu, info = cupy_bicgstab(self.A_matrix, b, maxiter=max_iter, tol=tol)
            elif self.solver_type == 'gmres':
                residuals = []
                # GMRES(m) params: restart is the number of Arnoldi iterations before each restart
                # callback is called at each restart, so total restarts = max_iter // restart
                num_restarts = max(1, max_iter // restart)

                # In multi-GPU mode (position!=None), use leave=False to avoid inter-process interference
                pbar = tqdm(total=num_restarts, desc="GMRES", unit="restart",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]',
                            position=self.tqdm_position,
                            leave=(self.tqdm_position is None))

                def callback(rk):
                    res_val = float(cp.asnumpy(rk))
                    residuals.append(res_val)
                    pbar.update(1)
                    pbar.set_postfix({'residual': f'{res_val:.2e}'})

                diagonal = self.A_matrix.diagonal()
                M_inv = cp.reciprocal(diagonal)  # 1 / diag
                M = cupy_linalg.LinearOperator(self.A_matrix.shape, matvec=lambda x: M_inv * x)

                x_gpu, info = cupy_gmres(self.A_matrix, b, M=M, maxiter=max_iter, tol=tol, restart=restart, callback=callback)
                pbar.close()

                # Save residual info
                self.solver_residuals = residuals
                self.final_residual = residuals[-1] if residuals else None

                if self.enable_plot:
                    plt.figure(figsize=(6, 4))
                    plt.semilogy(residuals, marker='o')
                    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                    plt.xlabel("Iteration")
                    plt.ylabel("Residual ||r||")
                    plt.title("GMRES Convergence")
                    plt.tight_layout()
                    plt.show()
                x_gpu = x_gpu  # Keep consistent
            elif self.solver_type == 'direct':
                # Direct solver - using cuSPARSE LU factorization
                from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
                if self.verbose:
                    print(f"Starting direct solve (matrix size: {self.A_matrix.shape[0]:,} x {self.A_matrix.shape[1]:,})...")
                    print(f"Note: LU factorization for 3D problems may take a long time, please be patient...")
                x_gpu = cupy_spsolve(self.A_matrix.tocsr(), b)
                info = 0  # Direct solver succeeded
                # Compute residual
                residual = cp.linalg.norm(self.A_matrix @ x_gpu - b) / cp.linalg.norm(b)
                self.final_residual = float(cp.asnumpy(residual))
                self.solver_residuals = [self.final_residual]
                if self.verbose:
                    print(f"Direct solve complete, relative residual: {self.final_residual:.2e}")
            solve_time = time.time() - solve_start
            if self.verbose:
                print(f"GPU solve complete, elapsed: {solve_time:.2f}s")
                print(f"Solver convergence info: {info}")
        except Exception as e:
            print(f"GPU solve failed: {e}. Note that memory may be the limiting factor for large problems.")
            return

        total_time = time.time() - start_time
        if self.verbose:
            print(f"Total solve time: {total_time:.2f}s")

        # Reshape results into 3D field components on the Yee grid
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        Nx_ex, Ny_ex, Nz_ex = self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex
        Nx_ey, Ny_ey, Nz_ey = self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey
        Nx_ez, Ny_ez, Nz_ez = self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez

        # Zero-copy CuPy -> torch via DLPack; slice/reshape stay on device
        x = torch.from_dlpack(x_gpu).to(dtype=torch.complex64)
        Ex = x[:N_ex].reshape((Nx_ex, Ny_ex, Nz_ex))
        Ey = x[N_ex:N_ex+N_ey].reshape((Nx_ey, Ny_ey, Nz_ey))
        Ez = x[N_ex+N_ey:].reshape((Nx_ez, Ny_ez, Nz_ez))
        self.E_field = (Ex, Ey, Ez)

        # E_field_raw is generated on demand from E_field to avoid duplicate storage
        # Use get_field_raw() for scenarios requiring numpy arrays
        self.E_field_raw = None  # Lazy generation

        # Save convergence info
        self.solver_info = info
        self.converged = (info == 0)

        if self.verbose:
            if info == 0:
                print("Solver converged successfully")
            else:
                print(f"Solver did not fully converge, exit code: {info}")

    def get_field_raw(self):
        return get_field_raw_impl(self)

    def _interpolate_yee_to_center(self):
        return interpolate_yee_to_center_impl(self)

    def plot_cross_section(self, axis='z', position=0.0, component='abs', field_log_scale=False, figsize=(12, 5), save_path=None, verbose=None):
        plot_cross_section_impl(
            self,
            axis=axis,
            position=position,
            component=component,
            field_log_scale=field_log_scale,
            figsize=figsize,
            save_path=save_path,
            verbose=verbose,
        )

    def plot_isotropic_3views(self, position=0.0, field_log_scale=True, figsize=(18, 5), save_path=None, vmin_db=-60, verbose=None):
        plot_isotropic_3views_impl(
            self,
            position=position,
            field_log_scale=field_log_scale,
            figsize=figsize,
            save_path=save_path,
            vmin_db=vmin_db,
            verbose=verbose,
        )

def solve_isotropic(self, source_pos, source_width, source_amplitude, max_iter=1500, tol=1e-6, restart=50, verbose=None):
    """
    Solve for an isotropic source field by running three orthogonal polarization
    simulations and performing power superposition.
    This function is optimized for the Yee staggered grid.

    Args:
        source_pos (list/tuple): Source center coordinates [x, y, z].
        source_width (float): Gaussian source width.
        source_amplitude (float): Source amplitude.
        max_iter (int): Maximum iterations for the iterative solver.
        tol (float): Solver convergence tolerance.
        restart (int): GMRES solver restart parameter.
        verbose: Whether to print detailed information.
    """
    if verbose is None:
        verbose = self.verbose

    self.scene.release_meshgrid()

    if verbose:
        print("\n==============================================")
        print("=== Starting Isotropic Source Simulation ===")
        print("==============================================")

    # Step 1/4: Pre-build system matrix A (if not already built)
    if self.A_matrix is None:
        if verbose:
            print("Step 1/4: Pre-building system matrix A...")
        t0 = time.time()
        self._ensure_system_matrix()
        if verbose:
            print(f"Matrix build complete, elapsed: {time.time() - t0:.2f}s")
    else:
        if verbose:
            print("Step 1/4: Using cached system matrix A.")

    if verbose:
        print_gpu_memory("After matrix build")

    # Initialize an accumulator on GPU for total field intensity squared
    # We interpolate to cell centers, so grid dimensions are reduced by 1
    # Using float32 to save memory; precision is sufficient
    total_intensity_gpu = cp.zeros(
        (self.scene.Nx - 1, self.scene.Ny - 1, self.scene.Nz - 1),
        dtype=cp.float32
    )

    # Define three orthogonal polarizations
    polarizations = {
        'X': (1.0, 0.0, 0.0),
        'Y': (0.0, 1.0, 0.0),
        'Z': (0.0, 0.0, 1.0)
    }

    # Steps 2 & 3: Loop and solve for each polarization direction
    for i, (pol_name, pol_vector) in enumerate(polarizations.items()):
        if verbose:
            print(f"\nStep {i+2}/4: Solving {pol_name}-polarization...")

        # a. Clear old sources and set new source for the current scene
        self.scene.sources = [
            PointDipole(
                    position=tuple(source_pos),
                polarization=pol_vector,
                width=source_width,
                source_time=CW(frequency=self.frequency, amplitude=source_amplitude),
                name=f"iso_{pol_name.lower()}",
            )
        ]

        # b. Rebuild only the b vector (Yee grid version)
        b_gpu = self._build_source_vector_yee()

        # c. Solve the linear system Ax = b
        solve_start = time.time()
        # GMRES callback is called at each restart, so progress bar total should be the number of restarts
        num_restarts = max_iter // restart
        pbar = tqdm(total=num_restarts, desc=f"GMRES-{pol_name}", unit="restart",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]',
                    position=self.tqdm_position,
                    leave=(self.tqdm_position is None))

        def iso_callback(rk):
            res_val = float(cp.asnumpy(rk))
            pbar.update(1)
            pbar.set_postfix({'residual': f'{res_val:.2e}'})

        x_gpu, info = cupy_gmres(self.A_matrix, b_gpu, maxiter=max_iter, tol=tol, restart=restart, callback=iso_callback)
        pbar.close()
        if verbose:
            print(f"Solve complete, elapsed: {time.time() - solve_start:.2f}s. Convergence info: {info}")
            if info != 0:
                print(f"  Warning: {pol_name}-polarization solver may not have fully converged.")

        # d. Reshape solution vector into field components on the Yee grid
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        Nx_ex, Ny_ex, Nz_ex = self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex
        Nx_ey, Ny_ey, Nz_ey = self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey
        Nx_ez, Ny_ez, Nz_ez = self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez

        Ex_gpu = x_gpu[:N_ex].reshape((Nx_ex, Ny_ex, Nz_ex))
        Ey_gpu = x_gpu[N_ex:N_ex+N_ey].reshape((Nx_ey, Ny_ey, Nz_ey))
        Ez_gpu = x_gpu[N_ex+N_ey:].reshape((Nx_ez, Ny_ez, Nz_ez))

        # e. [CORRECTED] Interpolate staggered field components to cell centers and accumulate power
        # Ex @ (i+0.5, j,   k) -> avg in Y and Z
        Ex_int = 0.5 * (Ex_gpu[:, :-1, :] + Ex_gpu[:, 1:, :])
        Ex_int = 0.5 * (Ex_int[:, :, :-1] + Ex_int[:, :, 1:])

        # Ey @ (i,   j+0.5, k) -> avg in X and Z
        Ey_int = 0.5 * (Ey_gpu[:-1, :, :] + Ey_gpu[1:, :, :])
        Ey_int = 0.5 * (Ey_int[:, :, :-1] + Ey_int[:, :, 1:])

        # Ez @ (i,   j,   k+0.5) -> avg in X and Y
        Ez_temp = 0.5 * (Ez_gpu[:-1, :, :] + Ez_gpu[1:, :, :])  # First, average along X
        Ez_int = 0.5 * (Ez_temp[:, :-1, :] + Ez_temp[:, 1:, :]) # Then, average the result along Y

        # Now all _int arrays have matching shapes and can be safely summed
        current_intensity_gpu = cp.abs(Ex_int)**2 + cp.abs(Ey_int)**2 + cp.abs(Ez_int)**2
        total_intensity_gpu += current_intensity_gpu

    # Step 4/4: Merge final results
    if verbose:
        print("\nStep 4/4: Merging results from all polarizations...")
    final_E_magnitude_gpu = cp.sqrt(total_intensity_gpu)

    final_E_magnitude = torch.from_dlpack(final_E_magnitude_gpu)

    self.E_field = (final_E_magnitude, final_E_magnitude, final_E_magnitude)
    if verbose:
        print("=== Isotropic source simulation complete! ===")
