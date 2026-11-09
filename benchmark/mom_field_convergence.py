"""Archive actual CUDA MoM near/internal E/H errors against analytic partial waves.

Usage: python -m benchmark.mom_field_convergence --levels 1 2 3 --output report.json
The fixed physical query shells are shared by every mesh. This sampled sphere
comparison does not certify arbitrary points, interface traces or other shapes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
import torch
import witwin.maxwell as mw

from .sie_reference import sphere_fields, sphere_far_field, sphere_cross_sections
from .mom_convergence import far_field_error_metrics
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.operator import IntegrationConfig


def query_shells(radius, angular_order=4):
    mu, weights = np.polynomial.legendre.leggauss(angular_order)
    phi = np.arange(2*angular_order)*np.pi/angular_order
    z, p = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(p), np.sqrt(1-z*z)*np.sin(p), z), -1).reshape(-1, 3)
    weights = np.repeat(weights, 2*angular_order)*np.pi/angular_order
    return [(factor, radius*factor*directions, weights) for factor in (.3, .7, 1.15, 1.5, 3.)]


def field_error(numeric, reference, weights):
    error = np.sum(np.abs(numeric-reference)**2, -1)
    ref = np.sum(np.abs(reference)**2, -1)
    norm = np.sqrt(np.sum(weights*ref))
    return {"relative_L2": float(np.sqrt(np.sum(weights*error))/norm) if norm else None,
            "absolute_L2": float(np.sqrt(np.sum(weights*error))),
            "max_absolute": float(np.sqrt(error.max())),
            "peak_normalized_error": float(np.sqrt(error.max()/ref.max())) if ref.max() else None}


def encode(values):
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def run_case(level, order, epsilon, angular_order):
    radius, frequency = .1, 3e8
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=radius), mw.Material(
        eps_r=epsilon.real, sigma_e=epsilon.imag*2*np.pi*frequency*mw.constants.EPSILON_0))], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=level, integration=IntegrationConfig(order=order, singular_order=order),
                          query_config=FieldQueryConfig(relative_tolerance=1e-5, max_order=128))
    result = mw.Simulation.mom(scene, frequency=frequency, config=config).run()
    if not result.stats()["converged"]:
        raise RuntimeError("Actual MoM reference algebra did not converge")
    handle = result.surface_currents.handle
    records = []
    for factor, points, weights in query_shells(radius, angular_order):
        region = "interior" if factor < 1 else "exterior"
        for kind in (["total"] if factor < 1 else ["total", "scattered"]):
            try:
                numerical = result.query_fields(torch.as_tensor(points, device=scene.device, dtype=torch.float64),
                                                region_id=region, field_kind=kind)
            except RuntimeError as exc:
                raise RuntimeError(f"Field shell r/a={factor}, {region}/{kind}: {handle.last_field_query_log}") from exc
            numeric_e, numeric_h = [value.detach().cpu().numpy() for value in numerical]
            reference_e, reference_h = sphere_fields(radius, handle.exterior.k.real, epsilon, points,
                                                      eta=handle.exterior.eta.real, field_kind=kind)
            refined_e, refined_h = sphere_fields(radius, handle.exterior.k.real, epsilon, points,
                                                  eta=handle.exterior.eta.real, field_kind=kind, cutoff=24)
            records.append({"radius_over_sphere_radius": factor, "region": region, "field_kind": kind,
                            "E": field_error(numeric_e, reference_e, weights),
                            "H": field_error(numeric_h, reference_h, weights),
                            "oracle_cutoff24_delta_E": field_error(reference_e, refined_e, weights),
                            "oracle_cutoff24_delta_H": field_error(reference_h, refined_h, weights),
                            "query_log": dict(handle.last_field_query_log),
                            "points_m": points.tolist(), "solid_angle_weights": weights.tolist(),
                            "numeric_E_V_per_m": encode(numeric_e), "numeric_H_A_per_m": encode(numeric_h),
                            "reference_E_V_per_m": encode(reference_e), "reference_H_A_per_m": encode(reference_h)})
    mu, angular_weights = np.polynomial.legendre.leggauss(12)
    phi = np.arange(24)*2*np.pi/24
    z, p = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(p), np.sqrt(1-z*z)*np.sin(p), z), -1).reshape(-1, 3)
    angular_weights = np.repeat(angular_weights, 24)*2*np.pi/24
    numeric_far, _ = result.far_field(torch.as_tensor(directions, device=scene.device, dtype=torch.float64))
    numeric_far = numeric_far.detach().cpu().numpy()
    reference_far = sphere_far_field(radius, handle.exterior.k.real, epsilon, directions)
    reference_cs = sphere_cross_sections(radius, handle.exterior.k.real, epsilon)
    scattering = float(np.sum(angular_weights*np.sum(np.abs(numeric_far)**2, -1)))
    forward, _ = result.far_field(torch.tensor([[0., 0., 1.]], device=scene.device, dtype=torch.float64))
    extinction = float(4*np.pi/handle.exterior.k.real*forward[0, 0].imag)
    far_report = {**far_field_error_metrics(numeric_far, reference_far, angular_weights),
                  "angular_rule": {"gauss_mu": 12, "uniform_phi": 24},
                  "directions": directions.tolist(), "solid_angle_weights": angular_weights.tolist(),
                  "numeric_E_amplitude_V": encode(numeric_far), "reference_E_amplitude_V": encode(reference_far),
                  "scattering_m2": scattering, "extinction_m2": extinction,
                  "absorption_balance_m2": extinction-scattering, "reference_cross_sections_m2": reference_cs,
                  "scattering_relative_error": abs(scattering/reference_cs["scattering"]-1) if reference_cs["scattering"] else None}
    torch.cuda.synchronize()
    return {"level": level, "edges": handle.basis.n_edges, "epsilon_r": [epsilon.real, epsilon.imag],
            "frequency_Hz": frequency, "radius_m": radius, "assembly_order": order,
            "angular_rule": {"gauss_mu": angular_order, "uniform_phi": 2*angular_order},
            "error_norm": "solid_angle_weighted_relative_L2_on_each_fixed_radius_shell",
            "solve": result.stats(), "queries": records, "far_field": far_report,
            "total_seconds": time.perf_counter()-start, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--angular-order", type=int, default=4)
    parser.add_argument("--epsilon", type=complex, default=2.25+.1j)
    parser.add_argument("--output", type=Path, default=Path("benchmark/cache/mom/field-convergence.json"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU backend is substituted")
    paths = list(Path("witwin/maxwell/mom").glob("*.py"))+[Path(__file__), Path("benchmark/sie_reference.py"), Path("benchmark/mom_convergence.py")]
    report = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "numpy": np.__version__, "scipy": scipy.__version__, "device": torch.cuda.get_device_name(),
              "complete_G0": False, "scope": "sampled nonmagnetic sphere shells only", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        case = run_case(level, args.order, args.epsilon, args.angular_order)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"edges": case["edges"], "total_seconds": case["total_seconds"], "errors": [
            {"radius": q["radius_over_sphere_radius"], "kind": q["field_kind"],
             "E": q["E"]["relative_L2"], "H": q["H"]["relative_L2"]} for q in case["queries"]]}), flush=True)


if __name__ == "__main__":
    main()
