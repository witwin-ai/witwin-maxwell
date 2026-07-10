"""FDFD performance benchmark.

Sweeps grid sizes for a canonical point-dipole + dielectric-cube scene and
records, for each size: matrix assembly time, solve time, matvec count,
convergence, explicit relative residual, and peak GPU memory.

This is a solver performance benchmark, not a Tidy3D validation benchmark;
it is intentionally separate from ``python -m benchmark`` (see runner.py).
Results are written to ``benchmark/FDFD_PERFORMANCE.md`` with raw JSON under
``benchmark/cache/fdfd_performance/``.

Usage:
    python -m benchmark.fdfd_performance
    python -m benchmark.fdfd_performance --sizes 32 64 --solver gmres --max-iter 2000
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_linalg
import torch

import witwin.maxwell as mw

from .paths import CACHE_DIR, ROOT

PERF_RESULTS_MD = ROOT / "FDFD_PERFORMANCE.md"
PERF_CACHE_DIR = CACHE_DIR / "fdfd_performance"

FREQUENCY = 1.0e9
RESOLUTION = 0.02  # 15 cells per wavelength at 1 GHz
PML_LAYERS = 8
PML_STRENGTH = 1.0e6
DEFAULT_SIZES = (32, 48, 64, 96, 128)


@dataclass(frozen=True)
class PerfCase:
    size: int
    unknowns: int
    nnz: int
    solver_type: str
    preconditioner: str
    precision: str
    max_iter: int
    tol: float
    restart: int
    assembly_s: float
    precond_setup_s: float
    solve_s: float
    reuse_solve_s: float
    matvecs: int
    converged: bool
    residual: float
    peak_gpu_gb: float
    status: str  # "ok", "oom", or "failed"


def build_scene(size: int, source_time=None) -> mw.Scene:
    """Canonical scene: z-polarized point dipole at the origin plus one
    dielectric cube (eps_r=4) offset in +y, sized relative to the PML-free
    interior so every sweep size keeps the same relative geometry.

    ``source_time`` defaults to a CW excitation at ``FREQUENCY`` (for FDFD).
    Pass a ``GaussianPulse`` to drive the same geometry in the time domain."""
    if source_time is None:
        source_time = mw.CW(frequency=FREQUENCY, amplitude=100.0)
    half = size * RESOLUTION / 2.0
    interior_cells = size - 2 * PML_LAYERS
    cube_side = (interior_cells / 3.0) * RESOLUTION
    cube_offset_y = (interior_cells / 4.0) * RESOLUTION
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(RESOLUTION),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS, strength=PML_STRENGTH),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="cube",
            geometry=mw.Box(
                position=(0.0, cube_offset_y, 0.0),
                size=(cube_side, cube_side, cube_side),
            ),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            width=0.05,
            source_time=source_time,
            name="src",
        )
    )
    return scene


def _failure_status(exc: Exception) -> str | None:
    """Map capacity failures to a status string; None means re-raise."""
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return "oom"
    if type(exc).__name__ == "cuDSSError":
        # ALLOC_FAILED is device-memory exhaustion; EXECUTION_FAILED at this
        # scale is cuDSS hitting a capacity limit during factorization.
        return "oom" if "ALLOC_FAILED" in str(exc) else "failed"
    return None


def _solve_counted(A, b, solver_type: str, max_iter: int, tol: float, restart: int, M=None):
    """Run the same cupy solver calls as FDFD.solve() with a matvec counter."""
    counter = {"matvecs": 0}

    def matvec(v):
        counter["matvecs"] += 1
        return A @ v

    A_op = cupy_linalg.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)
    if solver_type == "gmres":
        x, info = cupy_linalg.gmres(A_op, b, M=M, maxiter=max_iter, tol=tol, restart=restart)
    elif solver_type == "cg":
        x, info = cupy_linalg.cg(A_op, b, M=M, maxiter=max_iter, tol=tol)
    elif solver_type in ("bicgstab", "tfqmr", "idr", "sqmr"):
        from witwin.maxwell.fdfd import krylov

        x, info = krylov.solve(solver_type, A_op, b, M=M, tol=tol, maxiter=max_iter)
    else:
        raise ValueError(f"Unsupported solver_type {solver_type!r}.")
    return x, info, counter["matvecs"]


def run_case(size: int, solver_type: str, max_iter: int, tol: float, restart: int,
             preconditioner: str = "jacobi", precision: str = "single") -> PerfCase:
    scene = build_scene(size)
    solver = mw.Simulation.fdfd(
        scene,
        frequency=FREQUENCY,
        solver=mw.GMRES(max_iter=max_iter, tol=tol, restart=restart,
                        solver_type=solver_type, preconditioner=preconditioner,
                        precision=precision),
    ).prepare().solver

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    torch.cuda.synchronize()

    try:
        t0 = time.perf_counter()
        solver._ensure_system_matrix()
        cp.cuda.runtime.deviceSynchronize()
        assembly_s = time.perf_counter() - t0
        A = solver.A_matrix

        t0 = time.perf_counter()
        M = solver._ensure_preconditioner() if solver_type != "direct" else None
        cp.cuda.runtime.deviceSynchronize()
        precond_setup_s = time.perf_counter() - t0

        b = solver._build_source_vector_yee()

        reuse_solve_s = 0.0
        if solver_type == "direct":
            t0 = time.perf_counter()
            x = solver._solve_direct(b)
            cp.cuda.runtime.deviceSynchronize()
            solve_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            x = solver._solve_direct(b)
            cp.cuda.runtime.deviceSynchronize()
            reuse_solve_s = time.perf_counter() - t0
            info, matvecs = 0, 0
        else:
            A_iter = solver._iteration_matrix()
            b_iter = b.astype(solver._iteration_dtype(), copy=False)
            t0 = time.perf_counter()
            x, info, matvecs = _solve_counted(A_iter, b_iter, solver_type, max_iter, tol, restart, M=M)
            cp.cuda.runtime.deviceSynchronize()
            solve_s = time.perf_counter() - t0

        residual = float(cp.linalg.norm(A @ x.astype(cp.complex64, copy=False) - b) / cp.linalg.norm(b))
        return PerfCase(
            size=size,
            unknowns=int(A.shape[0]),
            nnz=int(A.nnz),
            solver_type=solver_type,
            preconditioner=preconditioner,
            precision=precision,
            max_iter=max_iter,
            tol=tol,
            restart=restart,
            assembly_s=assembly_s,
            precond_setup_s=precond_setup_s,
            solve_s=solve_s,
            reuse_solve_s=reuse_solve_s,
            matvecs=matvecs,
            converged=(info == 0),
            residual=residual,
            peak_gpu_gb=pool.total_bytes() / 2**30,
            status="ok",
        )
    except Exception as exc:
        status = _failure_status(exc)
        if status is None:
            raise
        return PerfCase(
            size=size,
            unknowns=0,
            nnz=0,
            solver_type=solver_type,
            preconditioner=preconditioner,
            precision=precision,
            max_iter=max_iter,
            tol=tol,
            restart=restart,
            assembly_s=0.0,
            precond_setup_s=0.0,
            solve_s=0.0,
            reuse_solve_s=0.0,
            matvecs=0,
            converged=False,
            residual=float("nan"),
            peak_gpu_gb=pool.total_bytes() / 2**30,
            status=status,
        )
    finally:
        solver._release_direct_solver()
        pool.free_all_blocks()


def render_markdown(cases: list[PerfCase], *, timestamp: str) -> str:
    device_name = torch.cuda.get_device_properties(0).name
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    lines = [
        "# FDFD Performance Benchmark",
        "",
        f"- **Updated:** {timestamp}",
        f"- **GPU:** {device_name} ({total_gb:.1f} GB)",
        f"- **CuPy:** {cp.__version__}, **PyTorch:** {torch.__version__}, **Platform:** {platform.system()}",
        f"- **Scene:** z-dipole + eps_r=4 cube, resolution {RESOLUTION} m "
        f"({299792458.0 / FREQUENCY / RESOLUTION:.0f} cells/wavelength at {FREQUENCY / 1e9:.1f} GHz), "
        f"PML {PML_LAYERS} layers",
        "- **Command:** `python -m benchmark.fdfd_performance`",
        "",
        "Peak GPU memory is the CuPy memory-pool high-water mark per case "
        "(torch-side scene tensors excluded). Matvecs count operator applications, "
        "not preconditioner applications.",
        "",
        "| Grid | Unknowns | Solver | Precond | Precision | Assembly (s) | Precond setup (s) | Solve (s) | Reuse solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |",
        "|------|----------|--------|---------|-----------|--------------|-------------------|-----------|-----------------|---------|-----------|----------|---------------|--------|",
    ]
    for case in cases:
        label = f"{case.solver_type}(mi={case.max_iter},tol={case.tol:g}"
        label += f",r={case.restart})" if case.solver_type == "gmres" else ")"
        lines.append(
            f"| {case.size}^3 | {case.unknowns:,} | {label} | {case.preconditioner} | {case.precision} | "
            f"{case.assembly_s:.2f} | {case.precond_setup_s:.2f} | "
            f"{case.solve_s:.2f} | {case.reuse_solve_s:.3f} | {case.matvecs} | {'yes' if case.converged else 'no'} | "
            f"{case.residual:.2e} | {case.peak_gpu_gb:.2f} | {case.status} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(sizes, solver_types, max_iter: int, tol: float, restart: int,
                  preconditioners=("jacobi",), precision: str = "single") -> list[PerfCase]:
    if isinstance(solver_types, str):
        solver_types = (solver_types,)
    cases = []
    for size in sizes:
        for solver_type in solver_types:
            for preconditioner in preconditioners:
                print(f"[fdfd-perf] size={size}^3 solver={solver_type} precond={preconditioner} "
                      f"precision={precision} ...", flush=True)
                case = run_case(size, solver_type, max_iter, tol, restart,
                                preconditioner=preconditioner, precision=precision)
                print(
                    f"[fdfd-perf]   unknowns={case.unknowns:,} assembly={case.assembly_s:.2f}s "
                    f"precond={case.precond_setup_s:.2f}s solve={case.solve_s:.2f}s reuse={case.reuse_solve_s:.3f}s "
                    f"matvecs={case.matvecs} converged={case.converged} "
                    f"residual={case.residual:.2e} peak={case.peak_gpu_gb:.2f}GB status={case.status}",
                    flush=True,
                )
                cases.append(case)
    return cases


def write_results(cases: list[PerfCase]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    PERF_RESULTS_MD.write_text(render_markdown(cases, timestamp=timestamp), encoding="utf-8")
    PERF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = PERF_CACHE_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    raw_path.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")
    print(f"[fdfd-perf] wrote {PERF_RESULTS_MD} and {raw_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="FDFD solver performance benchmark.")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="Cubic grid sizes to sweep (cells per axis).")
    parser.add_argument("--solver", nargs="+", default=["gmres"],
                        choices=("gmres", "cg", "direct", "bicgstab", "tfqmr", "idr", "sqmr"))
    parser.add_argument("--precond", nargs="+", default=["jacobi"],
                        choices=("none", "jacobi", "ssor", "ilu"),
                        help="Preconditioner(s) to sweep for the iterative solvers.")
    parser.add_argument("--precision", default="single", choices=("single", "double"),
                        help="Working precision of the iterative solve.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--restart", type=int, default=200)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("FDFD performance benchmark requires CUDA.")

    cases = run_benchmark(args.sizes, tuple(args.solver), args.max_iter, args.tol, args.restart,
                          preconditioners=tuple(args.precond), precision=args.precision)
    write_results(cases)


if __name__ == "__main__":
    main()
