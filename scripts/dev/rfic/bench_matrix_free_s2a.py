"""S2a matrix-free layered backend: per-frequency timing, iterations and memory next to dense.

Usage (idle GPU only for reportable timing):
    bench_matrix_free_s2a.py <case> <frequencies,comma> <out.json> [--fb-fixture PATH] [--dense] [--fb-mesh MAX,EDGE]

Cases: fa_test, fa_fine, fa_closed_fine, fb_test, fb_fine, fb_custom (F-B with --fb-mesh in micrometres).
Records, per frequency: near setup, block GMRES steps, operator actions, seconds per action,
total seconds, the admittance bound and, with --dense, the dense seconds and |dY|/|Y|.
Memory: torch peak allocated bytes of the matrix-free solve plus the sparse-LU peak estimate
(the GPU direct solver allocates outside torch), and the dense admission of the same mesh on
a 48 GB device (46 GiB available).
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

import witwin.maxwell as mw
from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.network import _wall_nodes, frequency_chunk_capacity


def simulation(case, frequencies, backend, fb_fixture, fb_mesh):
    if case.startswith("fb"):
        sys.path.insert(0, str(Path(fb_fixture).parent))
        import fb_fixture as module
        if case == "fb_custom":
            module.FB_MESHES["custom"] = fb_mesh
        fixture = module.fixture_fb(mesh={"fb_test": "test", "fb_fine": "full", "fb_custom": "custom"}[case])
    else:
        fixture = fixture_fa(substrate=case != "fa_closed_fine", mesh="test" if case == "fa_test" else "full")
    layered = fixture.config.layered
    if case == "fa_closed_fine":
        layered = dataclasses.replace(layered, conductor_model="closed")
    config = dataclasses.replace(fixture.config, layered=dataclasses.replace(layered, operator_backend=backend))
    return mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("frequencies")
    parser.add_argument("out")
    parser.add_argument("--fb-fixture", default=str(Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "s2a" / "probes" / "fb_fixture.py"))
    parser.add_argument("--fb-mesh", default="10,4.9")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--cap-gib", type=float, default=float(os.environ.get("S2A_MEMCAP_GIB", "0")))
    args = parser.parse_args()
    if args.cap_gib:
        torch.cuda.set_per_process_memory_fraction(
            args.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    frequencies = tuple(float(f) for f in args.frequencies.split(","))
    fb_mesh = tuple(float(v) for v in args.fb_mesh.split(","))
    record = dict(case=args.case, frequencies=frequencies, fb_mesh=fb_mesh, device=torch.cuda.get_device_name())
    torch.cuda.synchronize()
    started = time.perf_counter()
    prepared_sim = simulation(args.case, frequencies, "matrix_free", args.fb_fixture, fb_mesh).prepare()
    torch.cuda.synchronize()
    prepared = prepared_sim.solver.prepared
    n, faces, ports = prepared.layout.basis.n_edges, len(prepared.layout.mesh.faces), prepared.excitation.shape[1]
    record.update(prepare_seconds=time.perf_counter() - started, unknowns=n, faces=faces, ports=ports,
                  loops=prepared.loop_tree.loops, dense_24N2_GiB=24 * n * n / 2**30)
    try:
        frequency_chunk_capacity(n, faces, ports, 1, 46 * 2**30, wall_nodes=_wall_nodes(prepared.table))
        record["dense_admitted_48GB"] = True
    except RuntimeError as error:
        record["dense_admitted_48GB"] = False
        record["dense_refusal"] = str(error)
    print(json.dumps(record), flush=True)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = prepared_sim.run()
    torch.cuda.synchronize()
    metadata = result.network.metadata
    actions = metadata["operator_actions"]
    record["matrix_free"] = dict(
        seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated() - base,
        near_field=metadata["near_field"], block_steps=metadata["block_gmres_steps"], actions=actions,
        admittance_bound=metadata["admittance_bound"], factor_nnz=metadata["preconditioner_factor_nnz"],
        factor_bytes_estimate=metadata["preconditioner_factor_bytes_estimate"],
        frequency_seconds=metadata["frequency_seconds"], near_setup_seconds=metadata["near_setup_seconds"],
        action_seconds=metadata["action_seconds"],
        seconds_per_action=[a / b for a, b in zip(metadata["action_seconds"], actions)])
    y = result.network.to_y().cpu()
    record["matrix_free"]["y"] = [[[complex(v).real, complex(v).imag] for v in row] for m in y for row in m]
    print(json.dumps({k: v for k, v in record["matrix_free"].items() if k != "y"}), flush=True)
    del result
    if args.dense:
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        dense = simulation(args.case, frequencies, "dense", args.fb_fixture, fb_mesh).run()
        torch.cuda.synchronize()
        y_dense = dense.network.to_y().cpu()
        record["dense"] = dict(seconds=time.perf_counter() - started,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated() - base,
                               relative_error=[float(torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b))
                                               for a, b in zip(y, y_dense)])
        print(json.dumps(record["dense"]), flush=True)
    Path(args.out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
