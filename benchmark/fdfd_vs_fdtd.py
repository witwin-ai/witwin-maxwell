"""FDFD (iterative) vs FDTD same-scene memory/time comparison.

Runs the *identical* canonical scene (``build_scene`` from
``fdfd_performance``) through both runtimes across a grid-size sweep and records,
for each size and each backend: wall-clock solve time and peak GPU memory. The
direct (cuDSS) FDFD path is intentionally excluded; this benchmark compares the
iterative FDFD solver against FDTD only.

The two backends use different allocators (CuPy pool for FDFD, the Torch caching
allocator for FDTD), so peak memory is measured uniformly at the driver level via
a background ``torch.cuda.mem_get_info`` poller. The reported peak is the
high-water device-usage delta above the pre-run baseline, so it captures whatever
each backend actually reserves regardless of which allocator owns it.

This is a solver performance benchmark, not a Tidy3D validation benchmark, and is
separate from ``python -m benchmark`` (see runner.py). Results are written to
``benchmark/FDFD_VS_FDTD.md`` with raw JSON under
``benchmark/cache/fdfd_vs_fdtd/``.

Usage:
    python -m benchmark.fdfd_vs_fdtd
    python -m benchmark.fdfd_vs_fdtd --sizes 32 48 64
    python -m benchmark.fdfd_vs_fdtd --fdfd-solver sqmr --precond ssor \
        --precision double --max-iter 40000
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import cupy as cp
import torch

import witwin.maxwell as mw

from .fdfd_performance import (
    FREQUENCY,
    PML_LAYERS,
    RESOLUTION,
    _failure_status,
    _solve_counted,
    build_scene,
)
from .paths import CACHE_DIR, ROOT

VS_RESULTS_MD = ROOT / "FDFD_VS_FDTD.md"
VS_CACHE_DIR = CACHE_DIR / "fdfd_vs_fdtd"

DEFAULT_SIZES = (32, 48, 64, 96, 128)


class _GpuMemSampler(threading.Thread):
    """Driver-level peak GPU-usage sampler.

    Polls ``torch.cuda.mem_get_info`` on a background thread so it captures
    device memory reserved outside the Torch/CuPy caching allocators as well.
    ``stop()`` returns the peak usage delta (GiB) above the baseline captured at
    construction, isolating the run's own footprint from the resident CUDA
    context and any other processes.
    """

    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_event = threading.Event()
        free, total = torch.cuda.mem_get_info()
        self.total_bytes = total
        self.baseline_used = total - free
        self.peak_used = self.baseline_used

    def run(self):
        while not self._stop_event.is_set():
            free, _ = torch.cuda.mem_get_info()
            used = self.total_bytes - free
            if used > self.peak_used:
                self.peak_used = used
            self._stop_event.wait(self._interval)

    def stop(self) -> float:
        self._stop_event.set()
        self.join()
        return max(self.peak_used - self.baseline_used, 0) / 2**30


def _free_all() -> None:
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@dataclass(frozen=True)
class VsCase:
    size: int
    unknowns: int
    # FDFD (iterative)
    fdfd_solver: str
    fdfd_precond: str
    fdfd_precision: str
    fdfd_time_s: float
    fdfd_peak_gb: float
    fdfd_matvecs: int
    fdfd_converged: bool
    fdfd_residual: float
    fdfd_status: str  # "ok", "oom", "failed"
    # FDTD
    fdtd_time_s: float
    fdtd_peak_gb: float
    fdtd_steps: int
    fdtd_ms_per_step: float
    fdtd_status: str


def _run_fdfd_with_peak(size, solver_type, max_iter, tol, restart, preconditioner, precision):
    """Iterative FDFD single-source solve (assembly + preconditioner + solve),
    returning timing/convergence stats plus the sampled peak GPU memory."""
    _free_all()
    sampler = _GpuMemSampler()
    sampler.start()
    solver = None
    try:
        scene = build_scene(size)
        solver = mw.Simulation.fdfd(
            scene,
            frequency=FREQUENCY,
            solver=mw.GMRES(max_iter=max_iter, tol=tol, restart=restart,
                            solver_type=solver_type, preconditioner=preconditioner,
                            precision=precision),
        ).prepare().solver

        t0 = time.perf_counter()
        solver._ensure_system_matrix()
        M = solver._ensure_preconditioner()
        b = solver._build_source_vector_yee()
        A_iter = solver._iteration_matrix()
        b_iter = b.astype(solver._iteration_dtype(), copy=False)
        x, info, matvecs = _solve_counted(A_iter, b_iter, solver_type, max_iter, tol, restart, M=M)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - t0

        A = solver.A_matrix
        residual = float(cp.linalg.norm(A @ x.astype(cp.complex64, copy=False) - b) / cp.linalg.norm(b))
        result = {
            "unknowns": int(A.shape[0]),
            "time_s": elapsed,
            "matvecs": int(matvecs),
            "converged": info == 0,
            "residual": residual,
            "status": "ok",
        }
    except Exception as exc:
        status = _failure_status(exc)
        if status is None:
            raise
        result = {"unknowns": 0, "time_s": 0.0, "matvecs": 0, "converged": False,
                  "residual": float("nan"), "status": status}
    finally:
        if solver is not None:
            solver._release_direct_solver()
        peak = sampler.stop()
        _free_all()
    result["peak_gb"] = peak
    return result


def _run_fdtd_with_peak(size, steady_cycles, transient_cycles):
    _free_all()
    sampler = _GpuMemSampler()
    sampler.start()
    try:
        # Same geometry/materials/dipole as the FDFD scene, driven by a
        # broadband pulse; the FREQUENCY response is recovered via the DFT.
        scene = build_scene(
            size,
            source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.5 * FREQUENCY),
        )
        result = mw.Simulation.fdtd(
            scene,
            frequency=FREQUENCY,
            run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles,
                                        transient_cycles=transient_cycles),
            full_field_dft=True,
        ).run()
        torch.cuda.synchronize()
        stats = result.stats()
        out = {
            "time_s": float(stats.get("elapsed_s") or 0.0),
            "steps": int(stats.get("time_steps", 0)),
            "ms_per_step": float(stats.get("ms_per_step") or 0.0),
            "status": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        out = {"time_s": 0.0, "steps": 0, "ms_per_step": float("nan"), "status": "oom"}
    finally:
        peak = sampler.stop()
        _free_all()
    out["peak_gb"] = peak
    return out


def run_case(size: int, *, fdfd_solver: str, preconditioner: str, precision: str,
             max_iter: int, tol: float, restart: int,
             steady_cycles: int, transient_cycles: int) -> VsCase:
    print(f"[vs] size={size}^3 FDFD({fdfd_solver}/{preconditioner}/{precision}) ...", flush=True)
    f = _run_fdfd_with_peak(size, fdfd_solver, max_iter, tol, restart, preconditioner, precision)
    print(f"[vs]   FDFD time={f['time_s']:.2f}s peak={f['peak_gb']:.2f}GB "
          f"matvecs={f['matvecs']} converged={f['converged']} residual={f['residual']:.2e} "
          f"status={f['status']}", flush=True)

    print(f"[vs] size={size}^3 FDTD ...", flush=True)
    t = _run_fdtd_with_peak(size, steady_cycles, transient_cycles)
    print(f"[vs]   FDTD time={t['time_s']:.2f}s peak={t['peak_gb']:.2f}GB "
          f"steps={t['steps']} ms/step={t['ms_per_step']:.3f} status={t['status']}", flush=True)

    return VsCase(
        size=size,
        unknowns=f["unknowns"],
        fdfd_solver=fdfd_solver,
        fdfd_precond=preconditioner,
        fdfd_precision=precision,
        fdfd_time_s=f["time_s"],
        fdfd_peak_gb=f["peak_gb"],
        fdfd_matvecs=f["matvecs"],
        fdfd_converged=f["converged"],
        fdfd_residual=f["residual"],
        fdfd_status=f["status"],
        fdtd_time_s=t["time_s"],
        fdtd_peak_gb=t["peak_gb"],
        fdtd_steps=t["steps"],
        fdtd_ms_per_step=t["ms_per_step"],
        fdtd_status=t["status"],
    )


def _ratio(a: float, b: float) -> str:
    if b <= 0.0 or a <= 0.0:
        return "-"
    return f"{a / b:.1f}x"


def render_markdown(cases: list[VsCase], *, timestamp: str, tol: float) -> str:
    device_name = torch.cuda.get_device_properties(0).name
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    cells_per_wl = 299792458.0 / FREQUENCY / RESOLUTION
    ref = cases[0] if cases else None
    fdfd_label = (
        f"{ref.fdfd_solver}+{ref.fdfd_precond}+{ref.fdfd_precision} (tol {tol:g})"
        if ref else "iterative"
    )
    lines = [
        "# FDFD (iterative) vs FDTD",
        "",
        f"- **Updated:** {timestamp}",
        f"- **GPU:** {device_name} ({total_gb:.1f} GB)",
        f"- **CuPy:** {cp.__version__}, **PyTorch:** {torch.__version__}, **Platform:** {platform.system()}",
        f"- **Scene:** z-dipole + eps_r=4 cube, resolution {RESOLUTION} m "
        f"({cells_per_wl:.0f} cells/wavelength at {FREQUENCY / 1e9:.1f} GHz), PML {PML_LAYERS} layers",
        f"- **FDFD config:** {fdfd_label}",
        "- **Command:** `python -m benchmark.fdfd_vs_fdtd`",
        "",
        "Identical scene through both runtimes. Peak GPU is the driver-level "
        "high-water device-usage delta above the pre-run baseline (uniform across "
        "the CuPy and Torch allocators). FDFD time is assembly + preconditioner + "
        "iterative solve for one source; FDTD time is the stepping-loop wall time "
        "for one frequency (extra frequencies are free via the running DFT). "
        "The direct (cuDSS) FDFD path is excluded by design.",
        "",
        "| Grid | Unknowns | FDFD time (s) | FDFD peak (GB) | FDFD matvecs | FDFD resid | FDTD steps | FDTD time (s) | FDTD peak (GB) | Mem FDFD/FDTD | Time FDFD/FDTD |",
        "|------|----------|---------------|----------------|--------------|------------|------------|---------------|----------------|---------------|----------------|",
    ]
    for c in cases:
        fdfd_time = "oom" if c.fdfd_status == "oom" else f"{c.fdfd_time_s:.1f}"
        fdtd_time = "oom" if c.fdtd_status == "oom" else f"{c.fdtd_time_s:.1f}"
        conv = "" if c.fdfd_converged else "*"
        lines.append(
            f"| {c.size}^3 | {c.unknowns:,} | {fdfd_time} | {c.fdfd_peak_gb:.2f} | "
            f"{c.fdfd_matvecs}{conv} | {c.fdfd_residual:.2e} | {c.fdtd_steps} | "
            f"{fdtd_time} | {c.fdtd_peak_gb:.2f} | "
            f"{_ratio(c.fdfd_peak_gb, c.fdtd_peak_gb)} | {_ratio(c.fdfd_time_s, c.fdtd_time_s)} |"
        )
    lines.append("")
    lines.append("`*` = FDFD did not reach the tolerance within the matvec budget "
                 "(time is budget-bound, not solution-bound).")
    lines.append("")
    return "\n".join(lines)


def write_results(cases: list[VsCase], *, tol: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    VS_RESULTS_MD.write_text(render_markdown(cases, timestamp=timestamp, tol=tol), encoding="utf-8")
    VS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = VS_CACHE_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    raw_path.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")
    print(f"[vs] wrote {VS_RESULTS_MD} and {raw_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="FDFD (iterative) vs FDTD memory/time comparison.")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="Cubic grid sizes to sweep (cells per axis).")
    parser.add_argument("--fdfd-solver", default="sqmr",
                        choices=("gmres", "cg", "bicgstab", "tfqmr", "idr", "sqmr"),
                        help="Iterative FDFD solver (direct/cuDSS is excluded by design).")
    parser.add_argument("--precond", default="ssor", choices=("none", "jacobi", "ssor", "ilu"))
    parser.add_argument("--precision", default="double", choices=("single", "double"))
    parser.add_argument("--max-iter", type=int, default=40000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--restart", type=int, default=200)
    parser.add_argument("--steady-cycles", type=int, default=20)
    parser.add_argument("--transient-cycles", type=int, default=15)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("FDFD-vs-FDTD benchmark requires CUDA.")

    cases = []
    for size in args.sizes:
        cases.append(run_case(
            size,
            fdfd_solver=args.fdfd_solver,
            preconditioner=args.precond,
            precision=args.precision,
            max_iter=args.max_iter,
            tol=args.tol,
            restart=args.restart,
            steady_cycles=args.steady_cycles,
            transient_cycles=args.transient_cycles,
        ))
    write_results(cases, tol=args.tol)


if __name__ == "__main__":
    main()
