"""Run independently referenced CUDA MoM sphere convergence and archive costs.

Usage: python -m benchmark.mom_convergence --levels 1 2 3 --order 6
This produces evidence, not an automatic declaration that the full G0 gate passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import scipy
import torch
import witwin.maxwell as mw

from .sie_reference import sphere_cross_sections, sphere_far_field
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig, assemble_weak, build_rhs
from witwin.maxwell.mom.linalg import DenseOperator, StoppingRule, solve_reference, correct
from witwin.maxwell.mom.fields import far_field


def far_field_error_metrics(numeric, reference, solid_angle_weights):
    """Physical angular L2 error and explicitly named sampling diagnostics."""
    numeric, reference = np.asarray(numeric), np.asarray(reference)
    weights = np.asarray(solid_angle_weights)
    if numeric.shape != reference.shape or numeric.ndim != 2 or numeric.shape[1] != 3:
        raise ValueError("Far-field comparisons require matching [direction,3] arrays")
    if weights.shape != numeric.shape[:-1] or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Far-field comparisons require finite positive solid-angle weights")
    if not np.isfinite(numeric).all() or not np.isfinite(reference).all():
        raise ValueError("Far-field comparison samples must be finite")
    error2 = np.sum(np.abs(numeric-reference)**2, axis=-1)
    reference2 = np.sum(np.abs(reference)**2, axis=-1)
    reference_norm = float(np.sqrt(np.sum(weights*reference2)))
    # A vanishing reference needs an absolute field criterion, not a fabricated
    # relative ratio obtained by dividing by an arbitrary denominator floor.
    relative = float(np.sqrt(np.sum(weights*error2))/reference_norm) if reference_norm > 0 else None
    sample_norm = float(np.sqrt(np.sum(reference2)))
    sample_relative = float(np.sqrt(np.sum(error2))/sample_norm) if sample_norm > 0 else None
    peak = float(np.sqrt(reference2.max()))
    return {"far_field_relative_error": relative,
            "far_field_error_norm": "solid_angle_weighted_relative_L2",
            "far_field_absolute_L2_error": float(np.sqrt(np.sum(weights*error2))),
            "far_field_sample_relative_error": sample_relative,
            "far_field_max_absolute_error": float(np.sqrt(error2.max())),
            "far_field_peak_normalized_error": float(np.sqrt(error2.max())/peak) if peak > 0 else None}


def run_case(level, order, epsilon_r, *, dtype=torch.complex128):
    radius, frequency = .1, 3e8
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    basis = RWGBasis(sphere_mesh(radius=radius, subdivisions=level,
                                dtype=torch.float64 if dtype == torch.complex128 else torch.float32))
    exterior = compile_medium(mw.Material(), frequency)
    interior = compile_medium(mw.Material(eps_r=epsilon_r.real,
                                         sigma_e=epsilon_r.imag*2*np.pi*frequency*mw.constants.EPSILON_0), frequency)
    integration = IntegrationConfig(order=order, singular_order=order, pair_batch=64)
    a, metadata = assemble_weak(basis, exterior, interior, integration)
    mass = basis.mass_matrix().to(dtype)
    operator = DenseOperator(a, torch.block_diag(mass, mass))
    source = mw.PlaneWave()
    b = build_rhs(basis, exterior, [source], order=order)
    stopping = StoppingRule(method="direct", rtol=1e-8 if dtype == torch.complex128 else 2e-5)
    x, log = solve_reference(operator, b, stopping)
    if not log["converged"]:
        raise RuntimeError(f"Reference algebra failed: {log}")
    handle = SimpleNamespace(basis=basis, exterior=exterior, interior=interior, sources=[source], integration=integration)
    mu, weights = np.polynomial.legendre.leggauss(12)
    phi = np.arange(24)*2*np.pi/24
    z, p = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(p), np.sqrt(1-z*z)*np.sin(p), z), -1).reshape(-1, 3)
    far, _ = far_field(handle, x, torch.as_tensor(directions, dtype=basis.mesh.dtype, device="cuda"))
    numeric = far.detach().cpu().numpy()
    reference = sphere_far_field(radius, exterior.k.real, epsilon_r, directions)
    reference_cs = sphere_cross_sections(radius, exterior.k.real, epsilon_r)
    scattering = float(np.sum((np.abs(numeric)**2).sum(-1).reshape(12, 24)*weights[:, None])*2*np.pi/24)
    errors = far_field_error_metrics(numeric, reference, np.repeat(weights, 24)*2*np.pi/24)
    forward, _ = far_field(handle, x, torch.tensor([[0., 0., 1.]], device="cuda", dtype=basis.mesh.dtype))
    extinction = float(4*np.pi/exterior.k.real*forward[0, 0].imag)
    # Fixed q correction is tested with a genuinely arbitrary perturbed initial guess.
    generator = torch.Generator(device="cuda").manual_seed(42)
    x0 = x+.1*torch.linalg.vector_norm(x)/np.sqrt(x.numel())*torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    _, correction = correct(operator, b, x0, StoppingRule(
        rtol=1e-8 if dtype == torch.complex128 else 2e-5,
        restart=min(200,len(x)), maxiter=1000,
        preconditioner="row_column_norm", fallback="direct",
    ))
    torch.cuda.synchronize()
    return {"level": level, "edges": basis.n_edges, "epsilon_r": [epsilon_r.real, epsilon_r.imag],
            **errors, "far_field_angular_rule": {"gauss_mu_order": 12, "uniform_phi_count": 24},
            "far_field_samples": {
                "schema_version": 1, "complex_encoding": "split_real_imag",
                "field_definition": "E_scattered(r*d)=exp(i*k*r)*amplitude(d)/r",
                "amplitude_units": "V", "incident_electric_amplitude_V_per_m": [1., 0.],
                "directions": directions.tolist(),
                "solid_angle_weights": (np.repeat(weights, 24)*2*np.pi/24).tolist(),
                "numeric_real": numeric.real.tolist(), "numeric_imag": numeric.imag.tolist(),
                "reference_real": reference.real.tolist(), "reference_imag": reference.imag.tolist(),
            },
            "scattering": scattering, "extinction": extinction,
            "reference_cross_sections": reference_cs,
            "scattering_relative_error": abs(scattering/reference_cs["scattering"]-1) if reference_cs["scattering"] > 1e-25 else None,
            "assembly": metadata, "solve": log, "correction": correction,
            "total_seconds": time.perf_counter()-start, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--epsilon", type=complex, default=2.25+.1j)
    parser.add_argument("--output", type=Path, default=Path("benchmark/cache/mom/convergence.json"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU backend is substituted")
    source_files = list(Path("witwin/maxwell/mom").glob("*.py"))+[Path(__file__),Path("benchmark/sie_reference.py")]
    report = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],text=True).strip()),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "torch": torch.__version__, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "torch_cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "complete_G0": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        case = run_case(level, args.order, args.epsilon)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key:case[key] for key in ("edges", "far_field_relative_error", "scattering_relative_error", "total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
