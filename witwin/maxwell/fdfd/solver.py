import torch
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.cusparse as cupy_cusparse
import cupyx.scipy.sparse as cpsp
from cupyx.scipy.sparse.linalg import cg as cupy_cg, gmres as cupy_gmres
import time
import cupyx.scipy.sparse.linalg as cupy_linalg
from tqdm import tqdm

from ..compiler.materials import _scene_has_dispersive_material
from ..compiler.sources import compile_fdfd_sources
from ..scene import prepare_scene
from ..sources import CW, POINT_DIPOLE_REFERENCE_WIDTH, PointDipole
from .materials import _evaluate_components, _evaluate_edge_components
from .plotting import (
    plot_cross_section as plot_cross_section_impl,
    plot_isotropic_3views as plot_isotropic_3views_impl,
)
from .postprocess import (
    get_field_raw as get_field_raw_impl,
    interpolate_yee_to_center as interpolate_yee_to_center_impl,
)


# Iterative-refinement steps for the cuDSS direct solver. The complex64 LU of
# the indefinite curl-curl operator loses several digits; refinement recovers
# them at ~one matvec + one reuse-solve per step.
FDFD_DIRECT_IR_STEPS = 10


def _hybrid_min_device_memory(direct):
    # nvmath 1.0.0 (cuDSS 0.8.0.10) initialises cudss_data_ifc._tls.size_written only in the thread
    # that imports nvmath.sparse, so data reads from other threads (e.g. autograd backward) fail.
    # Initialise it for the current thread; remove when nvmath fixes the module-import thread-local.
    from nvmath.sparse._internal import cudss_data_ifc
    if not hasattr(cudss_data_ifc._tls, "size_written"):
        cudss_data_ifc._tls.size_written = np.empty((1,), dtype=np.uint64)
    return int(direct.plan_info.memory_estimates.hybrid_min_device_memory)


def print_gpu_memory(prefix=""):
    """Print current GPU memory usage."""
    import cupy as cp
    free, total = cp.cuda.Device(0).mem_info
    used = total - free
    print(f"{prefix} GPU Memory Usage: Used {used/1024**3:.2f} GB / Total {total/1024**3:.2f} GB")


def _sample_control_widths_gpu(coords):
    """Widths of the Yee sample control volumes, matching FDTD injection."""
    widths = cp.empty_like(coords)
    widths[0] = 0.5 * (coords[1] - coords[0])
    widths[-1] = 0.5 * (coords[-1] - coords[-2])
    widths[1:-1] = 0.5 * (coords[2:] - coords[:-2])
    return widths


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
    control_widths = tuple(_sample_control_widths_gpu(coords)
                           for coords in (x_coords, y_coords, z_coords))
    if profile_kind == 'ideal':
        ix = _nearest_axis_index_gpu(x_coords, position[0])
        iy = _nearest_axis_index_gpu(y_coords, position[1])
        iz = _nearest_axis_index_gpu(z_coords, position[2])
        volume = control_widths[0][ix] * control_widths[1][iy] * control_widths[2][iz]
        b_component[ix, iy, iz] += amplitude / volume
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
    volumes = (control_widths[0][ix0:ix1, None, None]
               * control_widths[1][None, iy0:iy1, None]
               * control_widths[2][None, None, iz0:iz1])
    # The declared amplitude is an integrated current moment (A m), as in FDTD.
    profile /= cp.sum(profile * volumes)
    b_component[ix0:ix1, iy0:iy1, iz0:iz1] += amplitude * profile


def _require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDFD requires scene.device to be CUDA, got {device}.")
    if not cp.cuda.is_available():
        raise RuntimeError("FDFD requires CUDA/CuPy, but no CUDA-capable device is available.")
    cp.cuda.Device(scene.x.device.index).use()


def _validate_supported_fdfd_materials(scene):
    for structure in getattr(scene, "structures", ()):
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if material is not None and bool(getattr(material, "is_gyromagnetic", False)):
            # The static material compile lowers a GyromagneticFerrite to its
            # diagonal background (eps_r, mu_infinity, sigma_e) only; the
            # non-reciprocal off-diagonal Polder permeability is carried by the
            # FDTD magnetization-ADE runtime, which the FDFD solver never runs.
            # Fail closed rather than silently solving a reciprocal medium.
            raise NotImplementedError(
                "FDFD does not support GyromagneticFerrite media: the frequency-domain "
                "solver would ingest only the diagonal background permeability and silently "
                "drop the non-reciprocal off-diagonal Polder gyrotropy. Use FDTD, whose "
                "magnetization-ADE runtime carries the gyromagnetic tensor."
            )
        if bool(getattr(material, "is_nonlinear", False)):
            raise NotImplementedError(
                "FDFD static parity for Kerr nonlinear media is deferred (user decision 2026-07-11): a "
                "frequency-domain solve has no single linear operator for a field-amplitude-dependent "
                "permittivity. Use FDTD, which models the nonlinearity directly."
            )
        if bool(getattr(material, "has_full_epsilon_tensor", False)):
            raise NotImplementedError(
                "FDFD static parity for full (off-diagonal Tensor3x3) anisotropic permittivity is deferred "
                "(user decision 2026-07-11): the curl-curl operator assembly here forms only per-axis "
                "diagonal material components. Use FDTD, which models the coupled tensor."
            )



class FDFD:
    """
    Full-vector 3D FDFD solver based on the Yee staggered grid.
    Runs on CUDA via CuPy sparse linear algebra and iterative solvers.
    """
    def __init__(self, scene, frequency=2.0e9, solver_type='gmres', preconditioner='jacobi',
                 precision='single', ssor_omega=0.8,
                 enable_plot=True, tqdm_position=None, verbose=False, device_memory_limit_bytes=None):
        """
        Initialize the 3D FDFD solver.

        Args:
            scene: Scene instance
            frequency: Operating frequency (Hz)
            solver_type: Solver type ('cg', 'gmres', 'direct', 'bicgstab',
                'tfqmr', 'idr', 'sqmr'). The short-recurrence engines live in
                fdfd/krylov.py; 'sqmr' relies on the complex-symmetric
                system produced by the symmetrized UPML assembly.
            preconditioner: Preconditioner for the iterative solvers
                ('none', 'jacobi', 'ssor', 'ilu'). 'ilu' is ILU(0) and is
                known to be unstable on the indefinite curl-curl operator;
                see benchmark/FDFD_PERFORMANCE.md.
            precision: Working precision of assembly, solve and fields
                ('single' or 'double'), including the direct backend.
            ssor_omega: Relaxation factor for the 'ssor' preconditioner
                (measured plateau of best convergence at 0.6-0.8).
            enable_plot: Whether to enable plotting (set False for batch data generation)
            tqdm_position: tqdm progress bar position (for multi-GPU parallelism, set different values per GPU)
            verbose: Whether to print detailed debug information
            device_memory_limit_bytes: Device budget for hybrid direct factors, in bytes
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
                if self.scene.boundary.face_kind(axis, side) not in {"none", "pml", "pec"}
            }
        )
        if unsupported_face_kinds:
            raise ValueError(
                "FDFD supports per-face 'none', 'pml' and 'pec' boundaries."
            )
        if getattr(self.scene, "has_symmetry", False):
            raise ValueError("FDFD currently does not support Scene(symmetry=...).")
        _require_cuda_scene(self.scene)
        self.use_gpu = True

        if solver_type not in ['cg', 'gmres', 'direct', 'bicgstab', 'tfqmr', 'idr', 'sqmr']:
            raise ValueError(
                "solver_type must be one of 'cg', 'gmres', 'direct', 'bicgstab', 'tfqmr', 'idr', 'sqmr'"
            )
        self.solver_type = solver_type
        if preconditioner not in ['none', 'jacobi', 'ssor', 'ilu', 'ams']:
            raise ValueError("preconditioner must be 'none', 'jacobi', 'ssor', 'ilu', or 'ams'")
        self.preconditioner = preconditioner
        if preconditioner == 'ams' and not self.scene.grid.is_uniform:
            raise ValueError("AMS requires a uniform auxiliary grid; use jacobi, ssor or direct for graded grids.")
        if precision not in ['single', 'double']:
            raise ValueError("precision must be 'single' or 'double'")
        self.precision = precision
        self.ssor_omega = float(ssor_omega)
        self.device_memory_limit_bytes = device_memory_limit_bytes

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
        self._preconditioner_op = None  # Cached preconditioner for the cached A_matrix
        self._direct_solver = None  # Cached cuDSS factorization for the cached A_matrix
        self._sym_scale = None  # Symmetrizing diagonal d = sqrt(sx*sy*sz); depends on grid, PML and frequency
        self._iteration_matrix_cache = None  # complex128 copy of A for precision='double'
        self._pec_mask = None
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
        self._preconditioner_op = None
        self._iteration_matrix_cache = None
        self._sym_scale = None  # PML stretch s = kappa + i sigma/(omega eps0) is frequency dependent
        self._release_direct_solver()
        conductive = any(float(getattr(s.material, 'sigma_e', 0.)) != 0
                         or float(getattr(s.material, 'sigma_m', 0.)) != 0
                         or getattr(s.material, 'sigma_e_tensor', None) is not None
                         for s in self.scene.structures)
        if _scene_has_dispersive_material(self.scene) or conductive:
            self.material_eps_components = None
            self.material_mu_components = None
            self.material_eps_r = None
            self.material_mu_r = None

    def _ensure_material_components(self):
        if self.material_eps_components is not None:
            return
        _validate_supported_fdfd_materials(self.scene)
        model = self.scene.compile_materials()
        self.material_eps_components, self.material_mu_components = _evaluate_components(model, self.frequency)
        self.material_eps_edges, self.material_mu_edges = _evaluate_edge_components(model, self.frequency)
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
        from ..compiler.structures import pmc_structures
        if pmc_structures(self.scene) or any(bool((component != 1).any()) for component in self.material_mu_edges.values()):
            from .magnetic import assemble_magnetic_operator
            self.A_matrix = assemble_magnetic_operator(self)
        else:
            self.A_matrix = self._build_matrix_gpu_yee_3d()
        self._matrix_frequency = self.frequency
        self._preconditioner_op = None
        self._iteration_matrix_cache = None
        self._release_direct_solver()

    def _iteration_matrix(self):
        """System matrix in the working precision of the iterative solve."""
        if self.precision == 'single':
            return self.A_matrix
        if self._iteration_matrix_cache is None:
            self._iteration_matrix_cache = self.A_matrix.astype(cp.complex128)
        return self._iteration_matrix_cache

    def _iteration_dtype(self):
        return cp.complex64 if self.precision == 'single' else cp.complex128

    def _release_direct_solver(self):
        if self._direct_solver is not None:
            self._direct_solver.free()
            self._direct_solver = None

    def _build_direct_solver(self, matrix, rhs):
        """Plan and factorize a cuDSS direct solver for the given matrix.

        LU follows the requested matrix precision and uses iterative refinement.
        Local responses require GPU-resident factors. The ordinary solver's
        existing hybrid storage option can retain factor blocks in host memory.
        """
        from nvmath.sparse.advanced import DirectSolver, ExecutionCUDA, HybridMemoryModeOptions

        limit = self.device_memory_limit_bytes
        if limit is None:
            execution = (ExecutionCUDA() if getattr(self, 'gpu_only', False) else ExecutionCUDA(
                hybrid_memory_mode_options=HybridMemoryModeOptions(hybrid_memory_mode=True)
            ))
        else:
            execution = ExecutionCUDA(hybrid_memory_mode_options=HybridMemoryModeOptions(
                hybrid_memory_mode=True, hybrid_device_memory_limit=limit))
        direct = DirectSolver(matrix, rhs, execution=execution)
        direct.plan()
        if limit is not None:
            minimum = _hybrid_min_device_memory(direct)
            if limit < minimum:
                direct.free()
                raise ValueError(
                    f"device_memory_limit_bytes={limit} is below the cuDSS hybrid minimum "
                    f"of {minimum} bytes for this system")
        direct.factorize()
        direct.solution_config.ir_num_steps = FDFD_DIRECT_IR_STEPS
        return direct

    def _solve_direct(self, b):
        """Solve Ax = b with cuDSS, factorizing once and reusing across RHS."""
        if self._direct_solver is None:
            self._direct_solver = self._build_direct_solver(self.A_matrix, b)
            self._direct_rhs_shape = b.shape
            return self._direct_solver.solve()
        # cuDSS retains the original RHS descriptor. Pad short blocks and split
        # wider ones instead of refactorizing when the caller changes nrhs.
        shape = self._direct_rhs_shape
        capacity = 1 if len(shape) == 1 else shape[1]
        columns = b[:, None] if b.ndim == 1 else b
        output = cp.empty_like(columns)
        for start in range(0, columns.shape[1], capacity):
            stop = min(start + capacity, columns.shape[1])
            padded = cp.zeros(shape, dtype=self.A_matrix.dtype, order='F')
            if len(shape) == 1:
                padded[:] = columns[:, start]
            else:
                padded[:, :stop-start] = columns[:, start:stop]
            self._direct_solver.reset_operands(b=padded)
            solved = self._direct_solver.solve()
            output[:, start:stop] = solved[:, None] if len(shape) == 1 else solved[:, :stop-start]
        return output[:, 0] if b.ndim == 1 else output

    def solve_adjoint(self, rhs, max_iter=1000, tol=1e-6, restart=100):
        """Solve the adjoint system A^T lam = rhs for a flat complex64 RHS.

        The assembled scaled system S = D A D^-1 is complex-symmetric
        (S^T = S), so A^T lam = rhs reduces to S (lam/d) = rhs/d and the
        adjoint reuses the forward operator, preconditioner, and (direct
        backend) factorization. rhs and the returned lam are in physical
        (unscaled) space.
        """
        self._ensure_system_matrix()
        scale = self._ensure_symmetrization_scale()
        rhs_scaled = rhs / scale
        if self.solver_type == 'direct':
            y = self._solve_direct(rhs_scaled)
        else:
            M = self._ensure_preconditioner()
            A_iter = self._iteration_matrix()
            rhs_iter = rhs_scaled.astype(self._iteration_dtype(), copy=False)
            if self.solver_type == 'cg':
                y, _info = cupy_cg(A_iter, rhs_iter, M=M, maxiter=max_iter, rtol=tol, atol=0.0)
            elif self.solver_type in ('bicgstab', 'tfqmr', 'idr', 'sqmr'):
                from . import krylov
                y, _info = krylov.solve(self.solver_type, A_iter, rhs_iter, M=M,
                                        tol=tol, maxiter=max_iter)
            else:
                y, _info = cupy_gmres(A_iter, rhs_iter, M=M, maxiter=max_iter, rtol=tol, atol=0.0, restart=restart)
            y = y.astype(self._iteration_dtype(), copy=False)
        return y * scale

    def _ensure_preconditioner(self):
        if self.preconditioner == 'none':
            return None
        if self._preconditioner_op is None:
            self._preconditioner_op = self._build_preconditioner(self._iteration_matrix())
        return self._preconditioner_op

    def _build_preconditioner(self, A):
        """Build the preconditioner operator for the given CSR matrix.

        All variants construct and apply on GPU. The cuSPARSE triangular
        solves upcast complex64 to complex128, so results are cast back
        after each solve.
        """
        if self.preconditioner == 'jacobi':
            inv_diagonal = cp.reciprocal(A.diagonal())

            def apply_jacobi(v):
                return inv_diagonal * v

            operator = cupy_linalg.LinearOperator(A.shape, matvec=apply_jacobi, dtype=A.dtype)
        elif self.preconditioner == 'ssor':
            # SSOR(omega): M = (D + wL) [w(2-w)]^-1 D^-1 (D + wU),
            # applied as M^-1 v via two triangular solves on A's own triangles
            omega = self.ssor_omega
            diagonal = A.diagonal()
            lower = cpsp.tril(A, k=-1).tocsr() * omega + cpsp.diags(diagonal).tocsr()
            upper = cpsp.triu(A, k=1).tocsr() * omega + cpsp.diags(diagonal).tocsr()
            scale_back = omega * (2.0 - omega)

            def apply_ssor(v):
                y = cupy_linalg.spsolve_triangular(lower, v, lower=True)
                y = y.astype(A.dtype, copy=False) * diagonal
                y = cupy_linalg.spsolve_triangular(upper, y, lower=False)
                return y.astype(A.dtype, copy=False) * scale_back

            operator = cupy_linalg.LinearOperator(A.shape, matvec=apply_ssor, dtype=A.dtype)
        elif self.preconditioner == 'ams':
            from .ams import build_ams_preconditioner

            operator = build_ams_preconditioner(self)
        else:  # 'ilu'
            factors = A.copy()
            cupy_cusparse.csrilu02(factors)
            lower = cpsp.tril(factors, k=0).tocsr()
            upper = cpsp.triu(factors, k=0).tocsr()

            def apply_ilu(v):
                y = cupy_linalg.spsolve_triangular(lower, v, lower=True, unit_diagonal=True)
                y = cupy_linalg.spsolve_triangular(upper, y.astype(A.dtype, copy=False), lower=False)
                return y.astype(A.dtype, copy=False)

            operator = cupy_linalg.LinearOperator(A.shape, matvec=apply_ilu, dtype=A.dtype)

        return operator

    def _stretched_metrics(self):
        """Primal and dual lengths including complex coordinate stretching.

        Normalize by one reference length to preserve uniform-grid scaling.
        Primal metrics belong to half nodes; dual metrics to main nodes. Each
        stretch is sampled at its own Yee position, never averaged.
        """
        reference = float(self.scene.dx_primal64[0] if self.scene.grid.is_custom else self.scene.grid.dx)
        real_dtype = cp.float64 if self.precision == 'double' else cp.float32
        main, half = [], []
        for axis, (stretch_main, stretch_half) in zip("xyz", self._pml_stretch()):
            primal = cp.asarray(getattr(self.scene, f"d{axis}_primal64") / reference, dtype=real_dtype)
            dual = cp.asarray(getattr(self.scene, f"d{axis}_dual64") / reference, dtype=real_dtype)
            main.append(stretch_main.astype(self._iteration_dtype()) * dual)
            half.append(stretch_half.astype(self._iteration_dtype()) * primal)
        return main, half

    def _ensure_pec_mask(self):
        """Exact zero tangential E constraints on voxelized PEC and domain faces."""
        if self._pec_mask is not None:
            return self._pec_mask
        # The same node-consistent staircase short set as the FDTD runtime.
        from ..fdtd.runtime.materials import _staircase_pec_shorted_edges
        shorted = _staircase_pec_shorted_edges(self.scene)
        masks = []
        for component, axis in zip(("Ex", "Ey", "Ez"), "xyz"):
            shape = tuple(getattr(self.scene, f"N{a}_{component.lower()}") for a in "xyz")
            mask = (cp.asarray(shorted[component]).copy() if shorted is not None
                    else cp.zeros(shape, dtype=cp.bool_))
            for normal, boundary_axis in enumerate("xyz"):
                if boundary_axis == axis:
                    continue
                for side, index in (("low", 0), ("high", -1)):
                    if self.scene.boundary.face_kind(boundary_axis, side) == "pec":
                        selection = [slice(None)] * 3
                        selection[normal] = index
                        mask[tuple(selection)] = True
            masks.append(mask.ravel())
        self._pec_mask = cp.concatenate(masks)
        return self._pec_mask

    def solve_physical_rhs(self, rhs, *, tol=1e-6, max_iter=1000, restart=100):
        """Checked CUDA solve for one or multiple physical-space RHS columns.

        Returns owned PyTorch storage. Explicit RHS must not excite eliminated
        PEC degrees of freedom. Direct factors are shared across calls.
        """
        if not isinstance(rhs, torch.Tensor) or not rhs.is_cuda:
            raise ValueError("FDFD RHS must be a CUDA PyTorch tensor.")
        if rhs.device != self.scene.x.device:
            raise ValueError("RHS and FDFD operator must reside on the same GPU.")
        self._ensure_system_matrix()
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.A_matrix.shape[0]:
            raise ValueError("RHS must have shape (N,) or (N, nrhs).")
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            raise ValueError("RHS needs at least one column.")
        values = cp.from_dlpack(rhs.detach().resolve_conj().contiguous()).astype(self.A_matrix.dtype)
        vector = values.ndim == 1
        values = values[:, None] if vector else values
        if not bool(cp.isfinite(values).all()):
            raise ValueError("RHS must be finite.")
        if bool(cp.any(values[self._ensure_pec_mask()] != 0)):
            raise ValueError("RHS excites constrained PEC degrees of freedom.")
        scale = self._ensure_symmetrization_scale()[:, None]
        scaled = cp.asfortranarray(values * scale)
        if self.solver_type == 'direct':
            solution = self._solve_direct(scaled).copy()
        else:
            columns = []
            for column in scaled.T:
                if not bool(cp.any(column)):
                    columns.append(cp.zeros_like(column))
                    continue
                matrix, preconditioner = self._iteration_matrix(), self._ensure_preconditioner()
                if self.solver_type == 'cg':
                    solution_column, info = cupy_cg(matrix, column, M=preconditioner,
                        rtol=tol, atol=0.0, maxiter=max_iter)
                elif self.solver_type == 'gmres':
                    solution_column, info = cupy_gmres(matrix, column, M=preconditioner,
                        rtol=tol, atol=0.0, maxiter=max_iter, restart=restart)
                else:
                    from . import krylov
                    solution_column, info = krylov.solve(self.solver_type, matrix, column,
                        M=preconditioner, tol=tol, maxiter=max_iter)
                if info != 0:
                    raise RuntimeError(f"Background {self.solver_type} failed with info={info}.")
                columns.append(solution_column)
            solution = cp.column_stack(columns)
        residual = cp.linalg.norm((self.A_matrix @ solution - scaled) / scale, axis=0)
        denominator = cp.linalg.norm(values, axis=0)
        relative = residual / cp.maximum(denominator, cp.finfo(denominator.dtype).tiny)
        if not bool(cp.isfinite(solution).all()) or bool(cp.any(relative > tol)):
            raise RuntimeError(f"FDFD physical RHS residual exceeds tolerance {tol}: {float(relative.max())}")
        self.rhs_residuals = torch.from_dlpack(relative.copy())
        fields = solution / scale
        return torch.from_dlpack((fields[:, 0] if vector else fields).copy())

    def _pml_stretch(self):
        """Complex coordinate stretch per axis as (primal-node, half-node)
        complex128 arrays at the current frequency; 1 outside the PML.

        See fdfd/pml.py for the profile and the BoundarySpec.strength
        reading shared with the FDTD CPML."""
        from .pml import stretch_profile_1d

        scene = self.scene
        stretches = []
        for axis in "xyz":
            nodes = getattr(scene, f"{axis}_nodes64")
            layers = (scene.pml_thickness_for_face(axis, "low"), scene.pml_thickness_for_face(axis, "high"))
            stretches.append(tuple(
                cp.asarray(stretch_profile_1d(
                    positions, nodes, *layers, omega=self.omega, eps0=self.eps0,
                    eta0=self.mu0 * self.c0, strength=scene.pml_strength,
                ))
                for positions in (nodes, getattr(scene, f"{axis}_half64"))
            ))
        return stretches

    def _build_matrix_gpu_yee_3d(self):
        """
        Build the 3D full-vector coefficient matrix for the Yee grid using CuPy on GPU.

        Uses the standard symmetric UPML discretization: same-field couplings
        take the mixed form 1/(s_row * s_half) so the operator is a diagonal
        similarity D A D^-1 (D = sqrt(sx*sy*sz) per unknown) away from
        complex-symmetric. The returned matrix IS the scaled, complex-symmetric
        system; solves must scale b -> d*b and unscale x -> x/d (see
        _ensure_symmetrization_scale).
        """
        ds = float(self.scene.dx_primal64[0] if self.scene.grid.is_custom else self.scene.grid.dx)
        k0 = self.k0

        Nx_ex, Ny_ex, Nz_ex = self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex
        Nx_ey, Ny_ey, Nz_ey = self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey
        Nx_ez, Ny_ez, Nz_ez = self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        N_vec = self.scene.N_vector_total

        if self.verbose:
            print("Preparing PML and permittivity for Yee grid...")
        (s_x_main, s_y_main, s_z_main), (s_x_half, s_y_half, s_z_half) = self._stretched_metrics()
        # Half-grid stretch factors padded at the walls so boundary rows reuse
        # the edge value; index m addresses position (m - 1/2), m+1 -> (m + 1/2)
        s_x_half_pad = cp.concatenate([s_x_half[:1], s_x_half, s_x_half[-1:]])
        s_y_half_pad = cp.concatenate([s_y_half[:1], s_y_half, s_y_half[-1:]])
        s_z_half_pad = cp.concatenate([s_z_half[:1], s_z_half, s_z_half[-1:]])
        self._ensure_material_components()
        eps_r_x = cp.asarray(self.material_eps_edges["x"])
        eps_r_y = cp.asarray(self.material_eps_edges["y"])
        eps_r_z = cp.asarray(self.material_eps_edges["z"])
        if self.precision == 'double':
            eps_r_x = eps_r_x.astype(cp.complex128)
            eps_r_y = eps_r_y.astype(cp.complex128)
            eps_r_z = eps_r_z.astype(cp.complex128)

        if self.verbose:
            print("Building 3D Yee grid sparse matrix...")
        # nnz estimate: up to 13 nonzero elements per equation
        # (1 diagonal + 4 same-field neighbors + 4 Ey/Ex coupling + 4 Ez/Ex coupling)
        max_nnz = N_vec * 13
        rows = cp.zeros(max_nnz, dtype=cp.int32)
        cols = cp.zeros(max_nnz, dtype=cp.int32)
        data = cp.zeros(max_nnz, dtype=self._iteration_dtype())
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
        sy_lo = s_y_half_pad[iy_1d][None, :, None]      # s_y at (j - 1/2)
        sy_hi = s_y_half_pad[iy_1d + 1][None, :, None]  # s_y at (j + 1/2)
        sz_lo = s_z_half_pad[iz_1d][None, None, :]
        sz_hi = s_z_half_pad[iz_1d + 1][None, None, :]
        diag_val = (k02 * eps_r_x
                    - (1 / sy_lo + 1 / sy_hi) / (sy * ds2)
                    - (1 / sz_lo + 1 / sz_hi) / (sz * ds2))
        add_entries(idx_ex, idx_ex, diag_val)

        # Neighbor indices (using slicing to avoid creating masks)
        # iy > 0: idx_ex[:, 1:, :]
        add_entries(idx_ex[:, 1:, :], (ix_1d[:, None, None] * Ny_ex + (iy_1d[None, 1:, None]-1)) * Nz_ex + iz_1d[None, None, :], 1 / (sy[:, 1:, :] * sy_lo[:, 1:, :] * ds2))
        # iy < Ny_ex - 1: idx_ex[:, :-1, :]
        add_entries(idx_ex[:, :-1, :], (ix_1d[:, None, None] * Ny_ex + (iy_1d[None, :-1, None]+1)) * Nz_ex + iz_1d[None, None, :], 1 / (sy[:, :-1, :] * sy_hi[:, :-1, :] * ds2))
        # iz > 0: idx_ex[:, :, 1:]
        add_entries(idx_ex[:, :, 1:], (ix_1d[:, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, 1:]-1), 1 / (sz[:, :, 1:] * sz_lo[:, :, 1:] * ds2))
        # iz < Nz_ex - 1: idx_ex[:, :, :-1]
        add_entries(idx_ex[:, :, :-1], (ix_1d[:, None, None] * Ny_ex + iy_1d[None, :, None]) * Nz_ex + (iz_1d[None, None, :-1]+1), 1 / (sz[:, :, :-1] * sz_hi[:, :, :-1] * ds2))

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
        del ix_1d, iy_1d, iz_1d, idx_ex, sy, sz, sx_half, sy_lo, sy_hi, sz_lo, sz_hi, val  # Free memory

        # --- 4. Fill Ey equations --- (using broadcasting instead of meshgrid to save memory)
        ix_1d = cp.arange(Nx_ey, dtype=cp.int32)
        iy_1d = cp.arange(Ny_ey, dtype=cp.int32)
        iz_1d = cp.arange(Nz_ey, dtype=cp.int32)
        idx_ey = N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :]
        sx = s_x_main[ix_1d][:, None, None]
        sz = s_z_main[iz_1d][None, None, :]
        sy_half = s_y_half[iy_1d][None, :, None]
        sx_lo = s_x_half_pad[ix_1d][:, None, None]      # s_x at (i - 1/2)
        sx_hi = s_x_half_pad[ix_1d + 1][:, None, None]  # s_x at (i + 1/2)
        sz_lo = s_z_half_pad[iz_1d][None, None, :]
        sz_hi = s_z_half_pad[iz_1d + 1][None, None, :]
        diag_val = (k02 * eps_r_y
                    - (1 / sx_lo + 1 / sx_hi) / (sx * ds2)
                    - (1 / sz_lo + 1 / sz_hi) / (sz * ds2))
        add_entries(idx_ey, idx_ey, diag_val)

        # Neighbor indices
        add_entries(idx_ey[1:, :, :], N_ex + ((ix_1d[1:, None, None]-1) * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :], 1 / (sx[1:, :, :] * sx_lo[1:, :, :] * ds2))
        add_entries(idx_ey[:-1, :, :], N_ex + ((ix_1d[:-1, None, None]+1) * Ny_ey + iy_1d[None, :, None]) * Nz_ey + iz_1d[None, None, :], 1 / (sx[:-1, :, :] * sx_hi[:-1, :, :] * ds2))
        add_entries(idx_ey[:, :, 1:], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + (iz_1d[None, None, 1:]-1), 1 / (sz[:, :, 1:] * sz_lo[:, :, 1:] * ds2))
        add_entries(idx_ey[:, :, :-1], N_ex + (ix_1d[:, None, None] * Ny_ey + iy_1d[None, :, None]) * Nz_ey + (iz_1d[None, None, :-1]+1), 1 / (sz[:, :, :-1] * sz_hi[:, :, :-1] * ds2))

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
        del ix_1d, iy_1d, iz_1d, idx_ey, sx, sz, sy_half, sx_lo, sx_hi, sz_lo, sz_hi, val

        # --- 5. Fill Ez equations --- (using broadcasting instead of meshgrid to save memory)
        ix_1d = cp.arange(Nx_ez, dtype=cp.int32)
        iy_1d = cp.arange(Ny_ez, dtype=cp.int32)
        iz_1d = cp.arange(Nz_ez, dtype=cp.int32)
        idx_ez = N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :]
        sx = s_x_main[ix_1d][:, None, None]
        sy = s_y_main[iy_1d][None, :, None]
        sz_half = s_z_half[iz_1d][None, None, :]
        sx_lo = s_x_half_pad[ix_1d][:, None, None]
        sx_hi = s_x_half_pad[ix_1d + 1][:, None, None]
        sy_lo = s_y_half_pad[iy_1d][None, :, None]
        sy_hi = s_y_half_pad[iy_1d + 1][None, :, None]
        diag_val = (k02 * eps_r_z
                    - (1 / sx_lo + 1 / sx_hi) / (sx * ds2)
                    - (1 / sy_lo + 1 / sy_hi) / (sy * ds2))
        add_entries(idx_ez, idx_ez, diag_val)

        # Neighbor indices
        add_entries(idx_ez[1:, :, :], N_ex + N_ey + ((ix_1d[1:, None, None]-1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :], 1 / (sx[1:, :, :] * sx_lo[1:, :, :] * ds2))
        add_entries(idx_ez[:-1, :, :], N_ex + N_ey + ((ix_1d[:-1, None, None]+1) * Ny_ez + iy_1d[None, :, None]) * Nz_ez + iz_1d[None, None, :], 1 / (sx[:-1, :, :] * sx_hi[:-1, :, :] * ds2))
        add_entries(idx_ez[:, 1:, :], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, 1:, None]-1)) * Nz_ez + iz_1d[None, None, :], 1 / (sy[:, 1:, :] * sy_lo[:, 1:, :] * ds2))
        add_entries(idx_ez[:, :-1, :], N_ex + N_ey + (ix_1d[:, None, None] * Ny_ez + (iy_1d[None, :-1, None]+1)) * Nz_ez + iz_1d[None, None, :], 1 / (sy[:, :-1, :] * sy_hi[:, :-1, :] * ds2))

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
        del ix_1d, iy_1d, iz_1d, idx_ez, sx, sy, sz_half, sx_lo, sx_hi, sy_lo, sy_hi, val

        if self.verbose:
            print(f"Sparse matrix nonzero count: {nnz:,}")
            print(f"Sparsity: {nnz / (N_vec * N_vec) * 100:.6f}%")

        # Apply the symmetrizing diagonal similarity: entry (m, n) -> d_m a_mn / d_n
        scale = self._ensure_symmetrization_scale()
        data[:nnz] *= scale[rows[:nnz]]
        data[:nnz] /= scale[cols[:nnz]]

        mask = self._ensure_pec_mask()
        keep = ~(mask[rows[:nnz]] | mask[cols[:nnz]])
        constrained = cp.flatnonzero(mask).astype(cp.int32)
        # Eliminate both rows and columns so reciprocity and adjoint reuse survive.
        A = cpsp.coo_matrix((cp.concatenate((data[:nnz][keep], cp.ones(constrained.size, dtype=data.dtype))),
                            (cp.concatenate((rows[:nnz][keep], constrained)),
                             cp.concatenate((cols[:nnz][keep], constrained)))), shape=(N_vec, N_vec))
        return A.tocsr()

    def _ensure_symmetrization_scale(self):
        """Diagonal similarity weight d = sqrt(sx*sy*sz) at each unknown's Yee
        location. With the symmetric UPML stencil, D A D^-1 is exactly
        complex-symmetric; the solver works on the scaled system throughout
        (b -> d*b at source build, x -> x/d when fields are extracted). The
        weight depends on grid, PML and frequency, not on materials."""
        if self._sym_scale is not None:
            return self._sym_scale
        (s_x, s_y, s_z), (s_x_half, s_y_half, s_z_half) = self._stretched_metrics()
        d_ex = cp.sqrt(s_x_half[:, None, None] * s_y[None, :, None] * s_z[None, None, :])
        d_ey = cp.sqrt(s_x[:, None, None] * s_y_half[None, :, None] * s_z[None, None, :])
        d_ez = cp.sqrt(s_x[:, None, None] * s_y[None, :, None] * s_z_half[None, None, :])
        self._sym_scale = cp.concatenate(
            [d_ex.ravel(), d_ey.ravel(), d_ez.ravel()]
        ).astype(self._iteration_dtype())
        return self._sym_scale

    def _build_source_vector_yee(self):
        """Build the total b vector for the Yee grid from the scene source list. Uses broadcasting instead of meshgrid to save memory."""
        b = cp.zeros(self.scene.N_vector_total, dtype=self._iteration_dtype())
        compiled_sources = compile_fdfd_sources(self.scene, default_frequency=self.frequency)
        if not compiled_sources:
            return b

        const = -1j * self.omega * self.mu0

        # Prepare coordinate data on GPU (1D arrays, using broadcasting)
        # Uniform axes use the same arithmetic as the source cutoff stencil;
        # reconstructing from float64 masters can change cutoff-plane inclusion.
        if self.scene.grid.is_custom:
            half = [getattr(self.scene, a + '_half') for a in 'xyz']
        else:
            half = [getattr(self.scene, a)[:-1] + spacing / 2
                    for a, spacing in zip('xyz', self.scene.grid_spacing)]
        x_ex = cp.asarray(half[0], dtype=cp.float32)
        y_ex = cp.asarray(self.scene.y, dtype=cp.float32)
        z_ex = cp.asarray(self.scene.z, dtype=cp.float32)

        x_ey = cp.asarray(self.scene.x, dtype=cp.float32)
        y_ey = cp.asarray(half[1], dtype=cp.float32)
        z_ey = cp.asarray(self.scene.z, dtype=cp.float32)

        x_ez = cp.asarray(self.scene.x, dtype=cp.float32)
        y_ez = cp.asarray(self.scene.y, dtype=cp.float32)
        z_ez = cp.asarray(half[2], dtype=cp.float32)

        # 3D views into b, one per field component (reshape of contiguous slices)
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        b_ex = b[:N_ex].reshape(self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex)
        b_ey = b[N_ex:N_ex + N_ey].reshape(self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey)
        b_ez = b[N_ex + N_ey:].reshape(self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez)

        for source in compiled_sources:
            if source.get('kind') == 'lumped_current':
                from .observations import gap_layout
                indices, lengths, area = gap_layout(self.scene, source['start'], source['end'])
                b[cp.from_dlpack(indices)] += const * source['current'] * cp.sign(cp.from_dlpack(lengths)) / area
                continue
            if source.get('kind') == 'uniform_current':
                for field, coords, amplitude in zip((b_ex,b_ey,b_ez),
                        ((x_ex,y_ex,z_ex),(x_ey,y_ey,z_ey),(x_ez,y_ez,z_ez)), source['polarization']):
                    mask = cp.ones(field.shape, dtype=cp.bool_)
                    for axis, (coordinate, center, size) in enumerate(zip(coords,source['center'],source['size'])):
                        selected = (cp.abs(coordinate-center) <= size*.5 if size else
                                    cp.arange(coordinate.size) == cp.abs(coordinate-center).argmin())
                        shape = [1,1,1]; shape[axis] = -1
                        mask &= selected.reshape(shape)
                    field += const * amplitude * mask
                continue
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

        # The assembled system is the symmetrized D A D^-1; scale the RHS to match
        b = b.astype(self._iteration_dtype()) * self._ensure_symmetrization_scale()
        b[self._ensure_pec_mask()] = 0
        return b

    def solve(self, max_iter=1000, tol=1e-6, restart=100):
        self.E_field = None
        self.E_field_raw = None
        self.solver_info = None
        self.converged = None
        self.final_residual = None
        self.solver_residuals = None

        start_time = time.time()

        self._ensure_system_matrix()
        b = self._build_source_vector_yee()
        matrix_time = time.time() - start_time
        if self.verbose:
            print(f"GPU matrix and source build time: {matrix_time:.2f}s")
            print(f"Solving linear system with GPU {self.solver_type.upper()} solver...")
            print_gpu_memory("Before solve")
        solve_start = time.time()

        if self.solver_type != 'direct':
            M = self._ensure_preconditioner()
            A_iter = self._iteration_matrix()
            b_iter = b.astype(self._iteration_dtype(), copy=False)
        if self.solver_type == 'cg':
            x_gpu, info = cupy_cg(A_iter, b_iter, M=M, maxiter=max_iter, rtol=tol, atol=0.0)
        elif self.solver_type in ('bicgstab', 'tfqmr', 'idr', 'sqmr'):
            from . import krylov
            x_gpu, info = krylov.solve(self.solver_type, A_iter, b_iter, M=M,
                                       tol=tol, maxiter=max_iter)
            self.final_residual = float(cp.linalg.norm(A_iter @ x_gpu - b_iter) / cp.linalg.norm(b_iter))
            self.solver_residuals = [self.final_residual]
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

            x_gpu, info = cupy_gmres(A_iter, b_iter, M=M, maxiter=max_iter, rtol=tol, atol=0.0,
                                   restart=restart, callback=callback, callback_type='pr_norm')
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
            # cuDSS sparse direct solve; factorization cached across RHS
            if self.verbose:
                reuse = "cached factorization" if self._direct_solver is not None else "factorize + solve"
                print(f"Direct solve ({reuse}, matrix size: {self.A_matrix.shape[0]:,})...")
            x_gpu = self._solve_direct(b)
            info = 0
            residual = cp.linalg.norm(self.A_matrix @ x_gpu - b) / cp.linalg.norm(b)
            self.final_residual = float(cp.asnumpy(residual))
            self.solver_residuals = [self.final_residual]
            if self.verbose:
                print(f"Direct solve complete, relative residual: {self.final_residual:.2e}")
        solve_time = time.time() - solve_start
        if self.verbose:
            print(f"GPU solve complete, elapsed: {solve_time:.2f}s")
            print(f"Solver convergence info: {info}")

        total_time = time.time() - start_time
        if self.verbose:
            print(f"Total solve time: {total_time:.2f}s")

        # Preserve working precision, then undo the symmetrization scaling.
        x_gpu = x_gpu.astype(self._iteration_dtype(), copy=False)
        x_gpu = x_gpu / self._ensure_symmetrization_scale()

        # Reshape results into 3D field components on the Yee grid
        N_ex, N_ey = self.scene.N_ex, self.scene.N_ey
        Nx_ex, Ny_ex, Nz_ex = self.scene.Nx_ex, self.scene.Ny_ex, self.scene.Nz_ex
        Nx_ey, Ny_ey, Nz_ey = self.scene.Nx_ey, self.scene.Ny_ey, self.scene.Nz_ey
        Nx_ez, Ny_ez, Nz_ez = self.scene.Nx_ez, self.scene.Ny_ez, self.scene.Nz_ez

        # Zero-copy CuPy -> torch via DLPack; slice/reshape stay on device
        x = torch.from_dlpack(x_gpu)
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

        x_gpu, info = cupy_gmres(self._iteration_matrix(),
                                 b_gpu.astype(self._iteration_dtype(), copy=False),
                                 M=self._ensure_preconditioner(),
                                 maxiter=max_iter, rtol=tol, atol=0.0, restart=restart,
                                 callback=iso_callback, callback_type='pr_norm')
        x_gpu = x_gpu.astype(cp.complex64, copy=False) / self._ensure_symmetrization_scale()
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
