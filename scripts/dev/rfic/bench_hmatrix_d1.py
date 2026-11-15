"""D1 H-matrix layered backend: accuracy, ranks, memory and the far-action ratio.

Usage (correctness runs; timing only on a granted idle GPU):
    bench_hmatrix_d1.py <case> <frequencies,comma> <out.json> [--fb-mesh MAX,EDGE]
        [--reference PATH] [--action] [--cap-gib G]

Cases as bench_matrix_free_s2a.py: fa_test, fa_fine, fa_closed_fine, fb_test, fb_fine, fb_custom.
--reference: an S2a agree-*.pt (dense Y per frequency) or beyond-fb-*.json (matrix_free Y);
  records |dY|_F/|Y|_F per frequency against it.
--action: at the first frequency, the compressed far action against the exact matrix-free far action
  on the port-drive face coefficients: relative error per channel and the wall-time ratio
  (one call each, correctness-run ratio only).
Memory: H-matrix bytes per frequency (factors, leaf tiles, indices), torch peak allocated bytes of
the solve, rank statistics for same-plane and cross-plane blocks.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import torch

from bench_matrix_free_s2a import simulation
from witwin.maxwell.mom.layered import hmatrix as hm
from witwin.maxwell.mom.layered.matrix_free import LayeredMatrixFreeOperator, NearFieldLU, build_near_structure


def reference_y(path):
    if path.endswith(".pt"):
        return torch.load(path, weights_only=False)["dense"]["y"]
    record = json.loads(Path(path).read_text())
    values = torch.tensor(record["matrix_free"]["y"], dtype=torch.float64)
    values = torch.complex(values[..., 0], values[..., 1])
    return values.reshape(len(record["frequencies"]), values.shape[1], values.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("frequencies")
    parser.add_argument("out")
    parser.add_argument("--fb-fixture", default=str(Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "s2a" / "probes" / "fb_fixture.py"))
    parser.add_argument("--fb-mesh", default="10,4.9")
    parser.add_argument("--reference")
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--exact-check", action="store_true", help="verify each solve by one exact far action")
    parser.add_argument("--cap-gib", type=float, default=float(os.environ.get("D1_MEMCAP_GIB", "0")))
    args = parser.parse_args()
    if args.cap_gib:
        torch.cuda.set_per_process_memory_fraction(
            args.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    frequencies = tuple(float(f) for f in args.frequencies.split(","))
    fb_mesh = tuple(float(v) for v in args.fb_mesh.split(","))
    record = dict(case=args.case, frequencies=frequencies, fb_mesh=fb_mesh, device=torch.cuda.get_device_name(),
                  eta=hm.ETA, leaf_size=hm.LEAF_SIZE, aca_tolerance=hm.ACA_TOLERANCE)
    prepared_sim = simulation(args.case, frequencies, "hmatrix", args.fb_fixture, fb_mesh).prepare()
    prepared = prepared_sim.solver.prepared
    prepared.metadata["hmatrix_exact_check"] = args.exact_check
    record.update(unknowns=prepared.layout.basis.n_edges, faces=len(prepared.layout.mesh.faces),
                  ports=prepared.excitation.shape[1])
    print(json.dumps(record), flush=True)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = prepared_sim.run()
    torch.cuda.synchronize()
    metadata = result.network.metadata
    y = result.network.to_y().cpu()
    record["hmatrix"] = dict(
        seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated() - base,
        structure=metadata["near_field"], block_steps=metadata["block_gmres_steps"],
        admittance_bound=metadata["admittance_bound"],
        compression_bound_worst_case=metadata["compression_bound_worst_case"],
        compression_estimate_quadrature=metadata["compression_estimate_quadrature"],
        compression_exact_check=metadata["compression_exact_check"],
        factor_bytes_estimate=metadata["preconditioner_factor_bytes_estimate"],
        frequency_seconds=metadata["frequency_seconds"], action_seconds=metadata["action_seconds"],
        per_frequency=metadata["hmatrix"],
        y=[[[complex(v).real, complex(v).imag] for v in row] for m in y for row in m])
    print(json.dumps({k: v for k, v in record["hmatrix"].items() if k != "y"}, default=str), flush=True)
    if args.reference:
        reference = reference_y(args.reference)
        record["reference"] = args.reference
        record["relative_error"] = [float(torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b))
                                    for a, b in zip(y, reference)]
        print("relative |dY|/|Y|", record["relative_error"], flush=True)
    del result
    if args.action:
        torch.cuda.empty_cache()
        structure = build_near_structure(prepared)
        clusters = hm.build_cluster_blocks(prepared, structure)
        solver = NearFieldLU(structure, prepared.excitation.device)
        operator = hm.LayeredHMatrixOperator(prepared, structure, clusters, 0, solver)
        x = operator.face_coefficients(operator.drive())
        times = {}
        for name, call in (("hmatrix", operator.far_moments),
                           ("matrix_free", lambda c: LayeredMatrixFreeOperator.far_moments(operator, c))):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = call(x)
            torch.cuda.synchronize()
            times[name] = (time.perf_counter() - started, out)
        approx, exact = times["hmatrix"][1], times["matrix_free"][1]
        record["action"] = dict(
            hmatrix_seconds=times["hmatrix"][0], matrix_free_seconds=times["matrix_free"][0],
            ratio=times["hmatrix"][0] / times["matrix_free"][0], hmatrix_build_seconds=operator.hmatrix.seconds,
            vector_error=float(torch.linalg.vector_norm(approx[:, :3] - exact[:, :3])
                               / torch.linalg.vector_norm(exact[:, :3])),
            scalar_error=float(torch.linalg.vector_norm(approx[:, 3] - exact[:, 3])
                               / torch.linalg.vector_norm(exact[:, 3])))
        print(json.dumps(record["action"]), flush=True)
        solver.free()
    Path(args.out).write_text(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
