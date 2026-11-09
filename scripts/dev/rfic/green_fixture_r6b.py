"""Exact F-A stack comparison with the open-source layered Green function library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

HEIGHTS = (300.5e-6, 302e-6, 305e-6)
FREQUENCIES = (1e9, 3.4e9)
RHO = np.geomspace(.1e-6, 500e-6, 41)
PAIRS = tuple((a, b) for i, a in enumerate(HEIGHTS) for b in HEIGHTS[i:])

def external(directory, tools):
    directory.mkdir(parents=True, exist_ok=True)
    stack = directory / "stack.yaml"
    stack.write_text("unit: um\ndielectric_layers:\n"
        "    silicon: {zmin: 0, h: 300, epsr: 11.9, mur: 1, sigma: 10}\n"
        "    oxide: {zmin: 300, h: 6, epsr: 4.1, mur: 1, sigma: 0}\n"
        "top_halfspace: {epsr: 1, mur: 1, sigma: 0}\n"
        "bottom_halfspace: {epsr: 11.9, mur: 1, sigma: 10}\n")
    np.savetxt(directory / "rho.txt", RHO)
    values = np.empty((2, 6, 2, len(RHO)), complex)
    vertical = np.empty((2, 6, 4, len(RHO)), complex)
    receipts = []
    binary = tools / "strata-driver/strata_ref"
    for fi, frequency in enumerate(FREQUENCIES):
        for pi, (zs, zo) in enumerate(PAIRS):
            output = directory / f"external-{fi}-{pi}.txt"
            command = ["nice", "-n", "10", str(binary), str(stack.resolve()), str(frequency),
                       str(zs), str(zo), str((directory / "rho.txt").resolve()), str(output.resolve())]
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
            (directory / f"external-{fi}-{pi}.log").write_text(completed.stdout + completed.stderr)
            print(f"external {fi}/{pi}: rc={completed.returncode}", flush=True)
            receipts.append(dict(command=command, returncode=completed.returncode,
                                 seconds=time.perf_counter()-started))
            assert completed.returncode == 0
            data = np.loadtxt(output)
            values[fi, pi] = np.conj(np.stack((data[:, 1]+1j*data[:, 2], data[:, 3]+1j*data[:, 4])))
            vertical[fi, pi] = np.conj(np.stack((data[:, 5]+1j*data[:, 6],
                data[:, 7]+1j*data[:, 8], data[:, 9]+1j*data[:, 10], data[:, 3]+1j*data[:, 4])))
    np.savez_compressed(directory / "external.npz", values=values, vertical=vertical, rho=RHO,
                        pairs=PAIRS, frequencies=FREQUENCIES)
    (directory / "external_receipts.json").write_text(json.dumps(dict(runs=receipts,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()), indent=2))

def native(directory):
    import torch
    from tests.support.rfic_fixtures import fixture_fa
    from witwin.maxwell.mom.layered.green import LayeredGreenTable
    stack = fixture_fa(device="cpu").scene.background
    table = LayeredGreenTable.build(stack, HEIGHTS, FREQUENCIES, RHO[-1], device="cuda",
                                   via_spans=((HEIGHTS[0], HEIGHTS[-1]),))
    rho = torch.tensor(RHO, device="cuda", dtype=torch.float64)
    values = np.stack([table.evaluate(a, b, rho).cpu().numpy() for a, b in PAIRS], axis=1)
    vertical = np.stack([table.evaluate_vertical(rho, torch.full_like(rho,a),
                         torch.full_like(rho,b)).cpu().numpy() for a,b in PAIRS], axis=1)
    np.savez_compressed(directory / "native.npz", values=values, vertical=vertical, rho=RHO,
                        pairs=PAIRS, frequencies=FREQUENCIES)
    reference = np.load(directory / "external.npz")["values"]
    delta = np.abs(values-reference)
    report = dict(max_row_relative=(delta/np.max(np.abs(reference), axis=-1, keepdims=True)).max(-1).tolist(),
                  max_point_relative=(delta/np.abs(reference)).max(-1).tolist(),
                  interpolation_error=float(table.interpolation_error),
                  peak_bytes=torch.cuda.max_memory_allocated())
    reference_v = np.load(directory / "external.npz")["vertical"]
    delta_v = np.abs(vertical-reference_v)
    report.update(vertical_max_row_relative=(delta_v/np.max(np.abs(reference_v),
                     axis=-1,keepdims=True)).max(-1).tolist(),
                  vertical_max_point_relative=(delta_v/np.maximum(np.abs(reference_v),
                     1e-6*np.max(np.abs(reference_v),axis=-1,keepdims=True))).max(-1).tolist())
    (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def oracle(directory):
    from tests.support.rfic_fixtures import fixture_fa
    from tests.support.layered_green_oracle import spatial,vertical_spatial
    stack=fixture_fa(device="cpu").scene.background
    values=np.empty((2,3,2,len(RHO)),complex)
    vertical=np.empty((2,3,4,len(RHO)),complex)
    errors=np.empty(values.shape)
    vertical_errors=np.empty(vertical.shape)
    for fi,frequency in enumerate(FREQUENCIES):
        for pi,index in enumerate((1,2,4)):
            zs,zo=PAIRS[index]
            for ri,rho in enumerate(RHO):
                values[fi,pi,:,ri],errors[fi,pi,:,ri]=spatial(stack,frequency,rho,zs,zo)
                vertical[fi,pi,:,ri],vertical_errors[fi,pi,:,ri]=vertical_spatial(
                    stack,frequency,rho,zs,zo)
            print(f"unextracted oracle {fi}/{index} complete",flush=True)
    np.savez_compressed(directory/"oracle.npz",values=values,vertical=vertical,errors=errors,
        vertical_errors=vertical_errors,rho=RHO,frequencies=FREQUENCIES,pair_indices=[1,2,4])

def freeze(directory):
    external=np.load(directory/"external.npz")
    oracle=np.load(directory/"oracle.npz")
    values=external["values"].copy()
    vertical=external["vertical"].copy()
    values[:,oracle["pair_indices"]]=oracle["values"]
    vertical[:,oracle["pair_indices"]]=oracle["vertical"]
    target=Path(__file__).resolve().parents[3]/"tests/mom/layered/data/fixture_r6b_green_reference.npz"
    np.savez_compressed(target,values=values,vertical=vertical,rho=RHO,pairs=PAIRS,
        frequencies=FREQUENCIES,external_values=external["values"],
        external_vertical=external["vertical"],oracle_pair_indices=oracle["pair_indices"],
        oracle_error_estimates=oracle["errors"],oracle_vertical_error_estimates=oracle["vertical_errors"])
    print(f"saved {target}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("external", "native", "oracle", "freeze"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tools", type=Path,
        default=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "tools")
    args = parser.parse_args()
    if args.mode == "external":
        external(args.directory, args.tools)
    elif args.mode == "native":
        native(args.directory)
    elif args.mode == "oracle":
        oracle(args.directory)
    else:
        freeze(args.directory)
