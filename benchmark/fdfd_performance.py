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
DEFAULT_SIZES = (32, 48, 64, 96)


@dataclass(frozen=True)
class PerfCase:
    size: int
    unknowns: int
    nnz: int
    solver_type: str
    max_iter: int
    tol: float
    restart: int
    assembly_s: float
    solve_s: float
    matvecs: int
    converged: bool
    residual: float
    peak_gpu_gb: float
    status: str  # "ok" or "oom"


def build_scene(size: int) -> mw.Scene:
    """Canonical scene: z-polarized point dipole at the origin plus one
    dielectric cube (eps_r=4) offset in +y, sized relative to the PML-free
    interior so every sweep size keeps the same relative geometry."""
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
            source_time=mw.CW(frequency=FREQUENCY, amplitude=100.0),
            name="src",
        )
    )
    return scene


def _solve_counted(A, b, solver_type: str, max_iter: int, tol: float, restart: int):
    """Run the same cupy solver calls as FDFD.solve() with a matvec counter."""
    counter = {"matvecs": 0}

    def matvec(v):
        counter["matvecs"] += 1
        return A @ v

    A_op = cupy_linalg.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)
    if solver_type == "gmres":
        M_inv = cp.reciprocal(A.diagonal())
        M = cupy_linalg.LinearOperator(A.shape, matvec=lambda x: M_inv * x, dtype=A.dtype)
        x, info = cupy_linalg.gmres(A_op, b, M=M, maxiter=max_iter, tol=tol, restart=restart)
    elif solver_type == "bicgstab":
        x, info = cupy_linalg.bicgstab(A_op, b, maxiter=max_iter, tol=tol)
    elif solver_type == "cg":
        x, info = cupy_linalg.cg(A_op, b, maxiter=max_iter, tol=tol)
    elif solver_type == "direct":
        x = cupy_linalg.spsolve(A, b)
        info = 0
    else:
        raise ValueError(f"Unsupported solver_type {solver_type!r}.")
    return x, info, counter["matvecs"]


def run_case(size: int, solver_type: str, max_iter: int, tol: float, restart: int) -> PerfCase:
    scene = build_scene(size)
    solver = mw.Simulation.fdfd(
        scene,
        frequency=FREQUENCY,
        solver=mw.GMRES(max_iter=max_iter, tol=tol, restart=restart, solver_type=solver_type),
    ).prepare().solver

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    torch.cuda.synchronize()

    try:
        t0 = time.perf_counter()
        solver.material_eps_components, solver.material_mu_components = (
            solver.scene.compile_material_components(frequency=solver.frequency)
        )
        A = solver._build_matrix_gpu_yee_3d()
        cp.cuda.runtime.deviceSynchronize()
        assembly_s = time.perf_counter() - t0

        b = solver._build_source_vector_yee()

        t0 = time.perf_counter()
        x, info, matvecs = _solve_counted(A, b, solver_type, max_iter, tol, restart)
        cp.cuda.runtime.deviceSynchronize()
        solve_s = time.perf_counter() - t0

        residual = float(cp.linalg.norm(A @ x - b) / cp.linalg.norm(b))
        return PerfCase(
            size=size,
            unknowns=int(A.shape[0]),
            nnz=int(A.nnz),
            solver_type=solver_type,
            max_iter=max_iter,
            tol=tol,
            restart=restart,
            assembly_s=assembly_s,
            solve_s=solve_s,
            matvecs=matvecs,
            converged=(info == 0),
            residual=residual,
            peak_gpu_gb=pool.total_bytes() / 2**30,
            status="ok",
        )
    except cp.cuda.memory.OutOfMemoryError:
        return PerfCase(
            size=size,
            unknowns=int(scene.N_vector_total) if hasattr(scene, "N_vector_total") else 0,
            nnz=0,
            solver_type=solver_type,
            max_iter=max_iter,
            tol=tol,
            restart=restart,
            assembly_s=0.0,
            solve_s=0.0,
            matvecs=0,
            converged=False,
            residual=float("nan"),
            peak_gpu_gb=pool.total_bytes() / 2**30,
            status="oom",
        )
    finally:
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
        "| Grid | Unknowns | Solver | Assembly (s) | Solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |",
        "|------|----------|--------|--------------|-----------|---------|-----------|----------|---------------|--------|",
    ]
    for case in cases:
        label = f"{case.solver_type}(mi={case.max_iter},tol={case.tol:g}"
        label += f",r={case.restart})" if case.solver_type == "gmres" else ")"
        lines.append(
            f"| {case.size}^3 | {case.unknowns:,} | {label} | {case.assembly_s:.2f} | "
            f"{case.solve_s:.2f} | {case.matvecs} | {'yes' if case.converged else 'no'} | "
            f"{case.residual:.2e} | {case.peak_gpu_gb:.2f} | {case.status} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(sizes, solver_type: str, max_iter: int, tol: float, restart: int) -> list[PerfCase]:
    cases = []
    for size in sizes:
        print(f"[fdfd-perf] size={size}^3 solver={solver_type} ...", flush=True)
        case = run_case(size, solver_type, max_iter, tol, restart)
        print(
            f"[fdfd-perf]   unknowns={case.unknowns:,} assembly={case.assembly_s:.2f}s "
            f"solve={case.solve_s:.2f}s matvecs={case.matvecs} converged={case.converged} "
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
    parser.add_argument("--solver", default="gmres",
                        choices=("gmres", "bicgstab", "cg", "direct"))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--restart", type=int, default=200)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("FDFD performance benchmark requires CUDA.")

    cases = run_benchmark(args.sizes, args.solver, args.max_iter, args.tol, args.restart)
    write_results(cases)


if __name__ == "__main__":
    main()
